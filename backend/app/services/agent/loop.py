"""Agent Loop 主循环：手写的 while 循环处理 tool_use/tool_result 多轮往返，
不用现成的Agent框架。调用哪个工具、调几轮、什么时候下结论完全由模型自主决定，
这里只负责工具调度执行和 max_turns 安全上限（项目书第五节5.2）。

轮次之间（turn-to-turn）依然严格顺序——模型必须先看到当前轮的工具结果才能
决定下一轮做什么，这个多轮推理链条不受影响。并行的只是"同一轮内"模型自己
打包在一起请求的多个工具调用：模型能在同一次响应里同时发起好几个tool_use，
本身就说明这几个调用互相不依赖对方的结果，之前却是排队一个个执行，现在
用 asyncio.gather 并行跑，见 run_agent_loop 里的 _execute_call。
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from app.services.agent.llm_client import CACHEABLE_CONTENT_THRESHOLD, ToolCall, ToolResult, get_llm_client
from app.services.agent.system_prompt import SYSTEM_PROMPT
from app.services.agent.tools import TOOL_SCHEMAS, execute_tool
from app.services.agent.traceability import find_unverifiable_claims, score_traceability
from app.services.polygon_client import CACHE_DIR

MAX_TURNS = 8
SUMMARY_LENGTH = 300

# 上下文压缩：get_filing_text这类工具的完整原始输出（实测一份10-K剥离HTML后
# 约39万字符、约10万token）会原样进messages历史，之后每一轮create_message都
# 把整个历史重新发一遍——查完财报原文后，剩下的每一轮（自我核查/生成简报）
# 都在重复携带这一大块。这里在往messages里追加"这一轮新的"工具结果之前，先
# 把已经被模型看过至少一轮（也就是不是这一轮刚产生的）、体积超过阈值的旧
# tool_result内容替换成占位说明——只是纯粹按体积做的结构性替换，不判断"哪段
# 内容重要该留"，跟CLAUDE.md"财报文本不做语义清洗规则"这条边界是两回事：
# 模型第一次看到工具结果时永远是完整原文，不做任何删减，压缩只发生在它已经
# 看过、且后续轮次不会再需要每次重新携带的历史部分。阈值复用llm_client.py
# 判断"值不值得加cache_control"的同一个常量——量级上的判断是同一件事：多大
# 的内容才值得做特殊处理
CONTEXT_COMPACTION_THRESHOLD_CHARS = CACHEABLE_CONTENT_THRESHOLD

# 只按体积判断压不压缩，会误伤本该精确保留的结构化数据——现在这几个工具返回
# 的都是小体积JSON（几百到几千字符），天然从来触不到体积阈值，"没被压缩"只是
# 巧合，不是有意保护。这里显式列出"即使哪天体积变大也绝不允许被压缩"的工具：
# 财务指标/板块位置/同业对比/价格反应/财报意外/主题轮动/数字核查，这些是简报
# 里数字断言的权威来源，压缩掉意味着后续轮次的自我核查、可追溯率校验都会失去
# 精确依据——跟get_filing_text这种"一段可以有损压缩、需要时能整段重新查阅的
# 叙述性原文"是完全不同的两类内容
_COMPACTION_EXEMPT_TOOLS = frozenset(
    {
        "get_financials",
        "get_sector_position",
        "get_peer_comparison",
        "get_price_reaction",
        "get_earnings_surprise",
        "get_thematic_flow",
        "verify_number",
    }
)

# 压缩审计日志：压缩本身发生在内存里的messages列表上，运行结束就随对象一起
# 消失，没有任何持久化记录"这次运行在第几轮压缩了哪个工具的结果"。这里追加
# 一份轻量的JSONL记录（跟best_of_n.py的RUNS_LOG_PATH同一个模式），只记元数据
# （工具名/原始长度/发生的轮次），不重复落库原文本身——原文本身已经在各数据源
# 服务自己的磁盘缓存里（比如get_filing_text背后的filing_text.py），重新调用
# 工具就能拿回来，这份日志的作用是"能查到这次运行确实发生过压缩、压的是什么"，
# 不是替代那份缓存
CONTEXT_COMPACTION_LOG_PATH = CACHE_DIR / "context_compaction.jsonl"


def _compaction_placeholder(original_length: int) -> str:
    return (
        f"[工具结果过大（{original_length}字符），已在上一轮完整展示并纳入分析，"
        "从这一轮起不再重复携带以节省上下文——如需重新查阅完整内容，重新调用"
        "一次该工具即可，结果有缓存，不会产生新的上游请求]"
    )


def _compact_seen_tool_results(
    messages: list[dict], tool_call_names: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """只处理已经追加进messages的历史条目——调用时机是"这一轮新的工具结果
    追加之前"，此时messages里只包含前几轮已经被发给模型看过至少一轮的内容，
    不会误伤这一轮刚产生、模型还没看过的工具结果。`tool_call_names`是
    `tool_use_id -> tool_name`的映射，用来判断这条tool_result是不是
    `_COMPACTION_EXEMPT_TOOLS`里的豁免工具——只按体积判断会误伤结构化数据。
    返回`(压缩后的messages, 这次调用里实际发生的压缩记录列表)`，记录列表给
    调用方写审计日志用，只包含这次真正被压缩的条目，不包含之前调用已经压缩
    过、这次体积已经在阈值以下的占位内容。"""
    compacted = []
    records: list[dict] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            compacted.append(message)
            continue
        new_content = []
        changed = False
        for block in content:
            block_content = block.get("content") if isinstance(block, dict) else None
            tool_use_id = block.get("tool_use_id") if isinstance(block, dict) else None
            tool_name = tool_call_names.get(tool_use_id) if tool_use_id else None
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block_content, str)
                and len(block_content) > CONTEXT_COMPACTION_THRESHOLD_CHARS
                and tool_name not in _COMPACTION_EXEMPT_TOOLS
            ):
                changed = True
                new_block = {
                    k: v for k, v in block.items() if k not in ("content", "cache_control")
                }
                new_block["content"] = _compaction_placeholder(len(block_content))
                new_content.append(new_block)
                records.append(
                    {
                        "tool_use_id": tool_use_id,
                        "tool_name": tool_name,
                        "original_length": len(block_content),
                    }
                )
            else:
                new_content.append(block)
        compacted.append({**message, "content": new_content} if changed else message)
    return compacted, records


def _append_compaction_log(ticker: str, turn: int, records: list[dict]) -> None:
    if not records:
        return
    CONTEXT_COMPACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with CONTEXT_COMPACTION_LOG_PATH.open("a") as log_file:
        for record in records:
            log_file.write(
                json.dumps({"timestamp": timestamp, "ticker": ticker, "turn": turn, **record}, ensure_ascii=False)
                + "\n"
            )

# 可追溯率gate的阈值，跟reward.py的打分权重一样是手动定的模块常量，方便调。
# 定得比较低（不是奔着"分数越高越好"去的那个~70%区间）是刻意的：
# traceability.py模块docstring里记录的3类已知限制（多期引用/模型自算的衍生
# 数字/跨公司数字混淆）本来就会让一份写得很好的报告也拿不到接近100%的分数，
# 这道gate的目标是拦住"明显不对劲"（大面积编造/归属错乱）的报告，不是逼着
# 模型为了凑分数去砍掉那些结构性验证不了、但本身可能是真实的数字
TRACEABILITY_GATE_MIN_RATIO = 0.3

# 三段式结构合规：<conclusion>本身已经是所有其它gate的触发前提（"response.text
# 里有<conclusion>"才会进入下面这一串检查），真正可能缺失的是<evidence>/<flags>——
# 这里只检查标签在不在，纯正则判断，没有语义模糊空间，是这几道gate里唯一能做到
# 接近100%确定的。只查标签存不存在，不要求非空——<flags></flags>本身就是一个
# 合法、有意义的表述（"逐条检查过，没有命中任何异常"），不是"没写"，不该被当成
# 缺失去拦截
_REQUIRED_REPORT_TAGS = ("evidence", "flags")
_REPORT_TAG_PATTERN = {tag: re.compile(rf"<{tag}>([\s\S]*?)</{tag}>") for tag in _REQUIRED_REPORT_TAGS}


def _missing_required_tags(report_text: str) -> list[str]:
    return [tag for tag in _REQUIRED_REPORT_TAGS if not _REPORT_TAG_PATTERN[tag].search(report_text)]


# 工具使用充分性：只设 get_financials 一个硬底线，不强制其它工具的调用顺序或
# 是否调用——CLAUDE.md核心原则2明确"不能写死固定的工具调用顺序"，多设几个
# 硬性要求就会侵蚀"模型自主判断该查什么"这个设计初衷。get_financials 是唯一
# 一个"简报里几乎不可能不引用、且几乎不存在'刻意跳过'的合理理由"的工具，所以
# 只对它设底线
_REQUIRED_TOOLS_BEFORE_CONCLUSION = frozenset({"get_financials"})


def _missing_required_tools(transcript: list[TranscriptEntry]) -> frozenset[str]:
    called = {entry.tool_name for entry in transcript if not entry.is_error}
    return _REQUIRED_TOOLS_BEFORE_CONCLUSION - called


# sentiment与市场反应一致性：<sentiment>是模型对自己<conclusion>的归纳，本质是
# 主观判断，代码没法判断它"对不对"——但能检测一种具体、真实存在的矛盾：
# sentiment标注positive，但get_price_reaction显示财报后股价明显下跌（或反过来），
# 这是两个独立信号在打架，值得让模型自己面对、给出解释，不是让代码替它下判断。
# 阈值定得比较宽（3%）避免把正常的小幅波动也当成矛盾——跟TRACEABILITY_GATE_
# MIN_RATIO一样是手动定的、可调的模块常量
_SENTIMENT_TAG_PATTERN = re.compile(r"<sentiment>([\s\S]*?)</sentiment>")
SENTIMENT_PRICE_REACTION_CONTRADICTION_PCT = 3.0


def _extract_sentiment(report_text: str) -> str | None:
    match = _SENTIMENT_TAG_PATTERN.search(report_text)
    if not match:
        return None
    return match.group(1).strip().lower()


def _find_price_reaction_data(raw_tool_outputs: list[str]) -> dict | None:
    """按JSON结构特征识别哪份原始工具输出是get_price_reaction的返回值（不靠
    tool_name标记，跟traceability.py的_find_financials_metrics是同一个思路）：
    PriceReactionResponse固定带has_data/price_change_pct这两个字段。"""
    for raw in raw_tool_outputs:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "has_data" in payload and "price_change_pct" in payload:
            return payload
    return None


def _sentiment_contradicts_price_reaction(sentiment: str, price_reaction: dict) -> bool:
    if not price_reaction.get("has_data"):
        return False
    pct = price_reaction.get("price_change_pct")
    if not isinstance(pct, int | float):
        return False
    if sentiment.startswith("positive") and pct <= -SENTIMENT_PRICE_REACTION_CONTRADICTION_PCT:
        return True
    return bool(sentiment.startswith("negative") and pct >= SENTIMENT_PRICE_REACTION_CONTRADICTION_PCT)


def _sentiment_still_contradicts_price_reaction(final_report: str, raw_tool_outputs: list[str]) -> bool:
    sentiment = _extract_sentiment(final_report)
    price_reaction = _find_price_reaction_data(raw_tool_outputs)
    if sentiment is None or price_reaction is None:
        return False
    return _sentiment_contradicts_price_reaction(sentiment, price_reaction)


def _latest_verification_outcomes(
    verify_number_outcomes: list[tuple[str, str, bool | None]],
) -> dict[tuple[str, str], bool | None]:
    """按 (metric, period) 分组，只保留每组最后一次核查结果——同一个指标可能
    被核查好几次（比如发现不对之后改了数字又核实一遍确认修好了），只有最新
    一次的结果才代表"这个指标现在到底有没有问题"。真实复现过的bug：如果不
    分组、只看整个调用序列的最后一个元素，会出现"核查营收发现不对、没去改，
    紧接着核查净利润正好对上"这种情况——营收那个明确核实过、确认有问题、却
    从没被处理的mismatch，会被净利润这次不相关的成功核查"顺手"洗白掉。"""
    latest: dict[tuple[str, str], bool | None] = {}
    for metric, period, matches in verify_number_outcomes:
        latest[(metric, period)] = matches
    return latest


def _unresolved_verification_mismatches(
    verify_number_outcomes: list[tuple[str, str, bool | None]],
) -> list[str]:
    return [
        f"{metric}（{period}）"
        for (metric, period), matches in _latest_verification_outcomes(verify_number_outcomes).items()
        if matches is False
    ]


def _compute_gate_resolutions(
    final_report: str | None,
    transcript: list[TranscriptEntry],
    raw_tool_outputs: list[str],
    verify_number_outcomes: list[tuple[str, str, bool | None]],
    traceable_matched: int,
    traceable_total: int,
) -> dict[str, bool]:
    """gate触发之后，模型的后续回应有没有真的解决问题，不能只看"插过nudge"这一
    个历史事实——nudge插入后，如果模型没有照做（比如只写一句"已修正"不重新
    输出标签），`_resolve_final_report`的回退机制会原样捞回没解决问题的旧草稿，
    `*_triggered=True` 就变成了一句谎言（真实复现过：结构gate触发、模型只回
    "已修正"不带标签，`structure_gate_triggered=True`但`final_report`依然缺
    `<flags>`）。这里在拿到最终 final_report/transcript 之后，用跟gate检测
    阶段完全相同的判断逻辑再查一遍，得到"这次运行结束时，这个问题是不是真的
    不存在了"这个客观结论。

    独立于`*_triggered`计算，不要求"曾经触发过"才计算：即使某道gate因为轮次
    预算耗尽、从没轮到检查（`*_triggered=False`），这里依然会如实反映"最终
    结果里这个问题存不存在"——`*_triggered=False`但`resolved=False`是合法
    组合，意味着"这个问题从没被检查过，但确实存在"，不代表数据错误，反而是
    更有价值的诊断信息。

    边界：`resolved=True`只代表"这个可机器判断的具体条件现在满足了"，不代表
    内容质量有保证——比如结构gate的resolved只看标签在不在，模型完全可以用
    一个空的`<flags></flags>`敷衍过关；可追溯率gate的resolved也可能是模型
    删掉了验证不了的断言而不是真的补充了依据。这些是客观指标类校验的天然
    局限，不是这次修复能解决的。

    Reflexion不在这个函数覆盖范围内——它评的是决策过程好不好，是另一个LLM的
    主观判断，没有客观事实可以在这里重新核对一遍，所以理论上它也有同样的
    "被回退机制悄悄绕过"的风险，但没法用同一套方法检测，只能作为已知限制
    记录，不在这个函数里处理。"""
    text = final_report or ""
    traceability_resolved = not (
        traceable_total > 0 and traceable_matched / traceable_total < TRACEABILITY_GATE_MIN_RATIO
    )
    return {
        "structure": not _missing_required_tags(text),
        "tool_coverage": not _missing_required_tools(transcript),
        "verification_mismatch": not _unresolved_verification_mismatches(verify_number_outcomes),
        "traceability": traceability_resolved,
        "sentiment_consistency": not _sentiment_still_contradicts_price_reaction(text, raw_tool_outputs),
    }


OnEvent = Callable[[dict], Awaitable[None]]
# (tool_name, 完整未截断的工具输出) —— 只给 Best-of-N 的规则打分用，现有的
# transcript/SSE事件走的仍然是 _summarize 截断过的版本，两者互不影响
OnToolResult = Callable[[str, str], Awaitable[None]]
# Reflexion：调用方（目前只有 best_of_n.py）传入一个函数，在模型想结束时用当前
# reasoning_notes/transcript 判断这次决策过程够不够好，返回 None 表示"不用改"，
# 返回字符串就是要塞回对话里的批评/整改要求。loop.py 不知道也不关心这个判断
# 是怎么做的（不直接依赖 reward.py，保持 loop.py 是不掺业务策略的通用原语），
# 只负责"要不要插一条消息、要不要再走一轮"这个机制本身。
ReflexionCheck = Callable[[list[ReasoningNote], list[TranscriptEntry]], Awaitable[str | None]]


def _summarize(content: str) -> str:
    if len(content) <= SUMMARY_LENGTH:
        return content
    return content[:SUMMARY_LENGTH] + f"...(共{len(content)}字符，已截断)"


def _has_self_verification(transcript: list[TranscriptEntry]) -> bool:
    return any(entry.tool_name == "verify_number" and not entry.is_error for entry in transcript)


# 之前6道gate是逐个检测、逐个拦截——测到第一个有问题的就continue，下一轮
# 才测下一个，最坏情况6道gate各拦一次=最多6轮，直接挤占MAX_TURNS预算（真实
# 观测到过一次max_turns_exceeded）。现在改成一次性测完6项、把所有真正有
# 问题的收集起来合成一条消息、只continue一次——最坏情况降到1轮，模型也能
# 看到问题全貌一次性处理，不用在"改完A又被B拦下"的拉锯里来回折腾。
def _build_combined_gate_nudge(issues: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {issue}" for i, issue in enumerate(issues))
    return (
        f"你的简报在下结论前有{len(issues)}个问题需要处理：\n{numbered}\n\n"
        "请一次性处理完以上全部问题，再重新完整输出一遍带 <conclusion>/<evidence>/"
        "<flags> 三个标签的简报——不能只回复一句总结性的话，必须重新输出完整内容。"
    )


# 批量nudge发出去之后，模型的回应只有两种值得区分的情况："重新输出了但还是
# 有问题"（内容质量判断，代码管不了，交给resolved字段如实记录）和"完全没
# 照做，只回一句不带任何标签的话"（比如"已处理完毕"）——后者是能明确判断的
# 格式违规，真实复现过（模型被结构gate拦下来之后只回一句收尾话，
# _resolve_final_report的回退机制会原样捞回没解决问题的旧草稿），值得再给
# 一次机会。只在"刚发过批量nudge"这个前提下触发，只拦一次。
IGNORED_FORMAT_NUDGE = (
    "你上一条回复没有包含 <conclusion> 标签，不能只回复一句总结性的话（比如"
    "\"已处理完毕\"），必须重新完整输出带 <conclusion>/<evidence>/<flags> 三个"
    "标签的简报。"
)


def _resolve_final_report(current_text: str | None, reasoning_notes: list[ReasoningNote]) -> str | None:
    """模型有时会在自我核查通过之后，最后一轮只写一句"核实通过，以上就是最终版本"
    这样的收尾话去指代前面某一轮已经写过的完整三段式简报，而不是把带标签的内容
    重新完整输出一遍——但只有这一轮的 response.text 会被当成 final_report，前面
    轮次写过的真正报告内容如果不捞回来就会被这句收尾话覆盖掉，前端会因为找不到
    <conclusion>标签而展示"未能按标准格式解析"。这里做兜底：当前轮文字没有标签时，
    往前找最近一条带标签的reasoning_note顶上去。真正的no-tag情况（比如refusal/
    max_tokens截断）本来就没有任何一轮带标签，兜底也找不到，行为不变。"""
    if current_text and "<conclusion>" in current_text:
        return current_text
    for note in reversed(reasoning_notes):
        if "<conclusion>" in note.text:
            return note.text
    return current_text


async def run_agent_loop(
    ticker: str,
    max_turns: int = MAX_TURNS,
    on_event: OnEvent | None = None,
    temperature: float | None = None,
    on_tool_result: OnToolResult | None = None,
    reflexion_check: ReflexionCheck | None = None,
    enable_compaction: bool = True,
) -> AgentRunResult:
    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    llm = get_llm_client()
    messages = [{"role": "user", "content": f"分析 {ticker} 的投资价值，形成投研简报。"}]
    transcript: list[TranscriptEntry] = []
    reasoning_notes: list[ReasoningNote] = []
    # 独立于 on_tool_result 回调（那个只服务 Best-of-N 打分）的内部收集，给
    # 事后的数字可追溯性校验用——本函数自己需要这份完整数据，不该依赖调用方
    # 有没有传这个可选回调
    raw_tool_outputs: list[str] = []
    # 按发生顺序记录每一次成功的 verify_number 调用返回的 (metric, period,
    # matches)——带上metric/period是为了能按各自的指标分别判断"最近一次
    # 核查有没有真的对上"，不能把整个列表拍平只看最后一个元素（否则核查A
    # 指标发现不对之后，紧接着核查B指标恰好对上，A的mismatch会被B"顺手"
    # 掩盖掉，见 _latest_verification_outcomes）
    verify_number_outcomes: list[tuple[str, str, bool | None]] = []
    # tool_use_id -> tool_name，供 _compact_seen_tool_results 判断某条
    # tool_result 是不是 _COMPACTION_EXEMPT_TOOLS 里的豁免工具——messages
    # 里的 tool_result block 本身只有 tool_use_id，没有工具名
    tool_call_names: dict[str, str] = {}
    response = None
    # 6道确定性gate（结构/工具覆盖/自我核查/核查一致性/可追溯率/sentiment
    # 一致性）现在批量检测、只用一个标记控制"只拦一次"——不再是每道gate各自
    # 一个nudged_for_X
    nudged_for_gate_batch = False
    # 批量nudge发出去之后，模型完全没有重新输出任何标签（不是"重新输出了但
    # 还是有问题"）时，再给一次机会——同样只拦一次
    nudged_for_ignored_format = False
    nudged_for_reflexion = False
    reflexion_triggered = False
    structure_gate_triggered = False
    tool_coverage_gate_triggered = False
    verification_mismatch_triggered = False
    traceability_gate_triggered = False
    sentiment_consistency_gate_triggered = False

    for turn in range(max_turns):
        response = await llm.create_message(
            system=SYSTEM_PROMPT, messages=messages, tools=TOOL_SCHEMAS, temperature=temperature
        )
        messages = llm.append_assistant_turn(messages, response)

        # 不管这一轮是不是同时调用了工具，模型写的文字都要留痕——之前只在最后
        # 一轮才读 response.text，中间轮次的推理文字（比如"先查数据，异常再看财报"）
        # 会被静默丢弃，既丢了第4阶段要展示的推理过程，也没法验证自我核查是不是
        # 真的发生在草稿之后
        if response.text:
            reasoning_notes.append(ReasoningNote(turn=turn, text=response.text))
            await emit({"type": "reasoning", "turn": turn, "text": response.text})

        if response.stop_reason != "tool_use":
            # 批量检测：把结构合规/工具使用底线/自我核查/核查一致性/可追溯率/
            # sentiment一致性这6项一次性测完，收集所有真正有问题的项，合成
            # 一条消息、只continue一次——不是逐个发现逐个拦。检测逻辑复用
            # `_compute_gate_resolutions`（跟最后算resolved字段用的是同一个
            # 函数，这里`False`就是"现在有问题，要写进nudge里"），自我核查
            # 单独判断（它的字段本来就是从transcript实时算的，不需要sticky
            # 标记）。
            if (
                response.stop_reason == "end_turn"
                and not nudged_for_gate_batch
                and response.text
                and "<conclusion>" in response.text
            ):
                candidate_report = _resolve_final_report(response.text, reasoning_notes) or ""
                batch_matched, batch_total = score_traceability(candidate_report, raw_tool_outputs)
                status = _compute_gate_resolutions(
                    candidate_report,
                    transcript,
                    raw_tool_outputs,
                    verify_number_outcomes,
                    batch_matched,
                    batch_total,
                )
                issues: list[str] = []
                if not _has_self_verification(transcript):
                    issues.append("还没有对任何一个关键数字调用 verify_number 做自我核查")
                if not status["structure"]:
                    structure_gate_triggered = True
                    issues.append(f"缺少标签：{'、'.join(_missing_required_tags(candidate_report))}")
                if not status["tool_coverage"]:
                    tool_coverage_gate_triggered = True
                    missing_tools = _missing_required_tools(transcript)
                    issues.append(f"还没有成功调用过：{'、'.join(sorted(missing_tools))}")
                if not status["verification_mismatch"]:
                    verification_mismatch_triggered = True
                    unresolved = _unresolved_verification_mismatches(verify_number_outcomes)
                    issues.append(
                        f"核实过的数字与真实数据不一致（verify_number 返回 matches=false），"
                        f"涉及：{'、'.join(unresolved)}——需要修正简报里这些数字，"
                        "或者在 <flags> 里如实说明无法确认"
                    )
                if not status["traceability"]:
                    traceability_gate_triggered = True
                    unverifiable = find_unverifiable_claims(candidate_report, raw_tool_outputs)
                    issues.append(
                        f"简报里有{batch_total}个数字型断言，但只有{batch_matched}个能在本次工具"
                        f"调用的原始数据里找到依据，比例明显偏低，没找到依据的包括："
                        f"{'、'.join(unverifiable[:5])}"
                    )
                if not status["sentiment_consistency"]:
                    sentiment_consistency_gate_triggered = True
                    sentiment = _extract_sentiment(candidate_report)
                    price_reaction = _find_price_reaction_data(raw_tool_outputs)
                    pct = price_reaction.get("price_change_pct") if price_reaction else None
                    issues.append(
                        f"<sentiment> 标注为 {sentiment}，但 get_price_reaction 显示财报公布后"
                        f"股价实际变动了 {pct:+.1f}% 时，方向跟 sentiment 不一致"
                        if isinstance(pct, int | float)
                        else "<sentiment> 标注跟 get_price_reaction 反映的市场反应方向不一致"
                    )

                nudged_for_gate_batch = True
                if issues:
                    messages = [*messages, {"role": "user", "content": _build_combined_gate_nudge(issues)}]
                    continue

            # 二次机会：批量nudge发出去之后，模型的回应有两种值得区分的情况——
            # "重新输出了但还是有问题"（内容质量判断，代码管不了，交给下面
            # resolved字段如实记录）和"完全没照做，只回一句不带任何标签的话"
            # （真实复现过这个bug）。后者是能明确判断的格式违规，值得再给一次
            # 机会；只在"刚发过批量nudge"这个前提下触发，只拦一次。
            if (
                response.stop_reason == "end_turn"
                and nudged_for_gate_batch
                and not nudged_for_ignored_format
                and not (response.text and "<conclusion>" in response.text)
            ):
                nudged_for_ignored_format = True
                messages = [*messages, {"role": "user", "content": IGNORED_FORMAT_NUDGE}]
                continue

            # Reflexion：批量gate/二次机会这两道关过了之后，才轮到"整体决策
            # 过程好不好"这个更宽泛的检查——先确认底线要求（结构/核查/可追溯
            # 这些确定性条件）再谈整体质量，顺序反过来意义不大。同样只检查
            # 一次（nudged_for_reflexion），不会因为
            # 模型改完之后还是不够好就无限循环下去；reflexion_check本身可能判断
            # "不用改"而返回None，这种情况也算检查过一次，不会同一轮里再查一遍。
            if (
                response.stop_reason == "end_turn"
                and not nudged_for_reflexion
                and response.text
                and "<conclusion>" in response.text
                and reflexion_check is not None
            ):
                critique = await reflexion_check(reasoning_notes, transcript)
                nudged_for_reflexion = True
                if critique is not None:
                    reflexion_triggered = True
                    messages = [*messages, {"role": "user", "content": critique}]
                    continue

            final_report = _resolve_final_report(response.text, reasoning_notes)
            traceable_matched, traceable_total = score_traceability(final_report or "", raw_tool_outputs)
            gate_resolutions = _compute_gate_resolutions(
                final_report, transcript, raw_tool_outputs, verify_number_outcomes, traceable_matched, traceable_total
            )
            return AgentRunResult(
                ticker=ticker,
                # 只有真正的 end_turn 才算成功完成；refusal/max_tokens/未识别的
                # stop_reason 都不是"顺利产出了完整简报"，不能标记成 completed
                completed=response.stop_reason == "end_turn",
                stop_reason=response.stop_reason,
                final_report=final_report,
                reasoning_notes=reasoning_notes,
                transcript=transcript,
                turns_used=turn + 1,
                structure_gate_triggered=structure_gate_triggered,
                structure_gate_resolved=gate_resolutions["structure"],
                tool_coverage_gate_triggered=tool_coverage_gate_triggered,
                tool_coverage_gate_resolved=gate_resolutions["tool_coverage"],
                self_verification_triggered=_has_self_verification(transcript),
                verification_mismatch_triggered=verification_mismatch_triggered,
                verification_mismatch_resolved=gate_resolutions["verification_mismatch"],
                traceability_gate_triggered=traceability_gate_triggered,
                traceability_gate_resolved=gate_resolutions["traceability"],
                sentiment_consistency_gate_triggered=sentiment_consistency_gate_triggered,
                sentiment_consistency_gate_resolved=gate_resolutions["sentiment_consistency"],
                reflexion_triggered=reflexion_triggered,
                traceable_numbers_matched=traceable_matched,
                traceable_numbers_total=traceable_total,
            )

        async def _execute_call(call: ToolCall, _turn: int = turn) -> tuple[ToolResult, TranscriptEntry]:
            # "开始"和"结束"两个事件都要发——像 get_filing_text 这类工具本身
            # 可能跑好几秒，前端应该先看到"正在拉取财报原文"而不是干等到结果
            # 出来才有任何反馈
            await emit(
                {
                    "type": "tool_call_started",
                    "turn": _turn,
                    "tool_name": call.name,
                    "tool_input": call.input,
                }
            )
            tool_call_names[call.id] = call.name
            output, is_error = await execute_tool(call.name, call.input)
            if not is_error:
                raw_tool_outputs.append(output)
                if call.name == "verify_number":
                    try:
                        verify_payload = json.loads(output)
                    except json.JSONDecodeError:
                        verify_payload = None
                    if isinstance(verify_payload, dict):
                        # metric/period 从响应里读（VerifyNumberResponse 会把
                        # 收到的请求参数原样回显），不从 call.input 读——两者
                        # 理论上应该一致，但响应是这次核查真正针对的
                        # (metric, period)，单一数据源更可靠
                        verify_metric = verify_payload.get("metric")
                        verify_period = verify_payload.get("period")
                        verify_matches = verify_payload.get("matches")
                        if isinstance(verify_metric, str) and isinstance(verify_period, str):
                            verify_number_outcomes.append(
                                (verify_metric, verify_period, verify_matches if isinstance(verify_matches, bool) else None)
                            )
                if on_tool_result is not None:
                    await on_tool_result(call.name, output)
            summary = _summarize(output)
            await emit(
                {
                    "type": "tool_call_finished",
                    "turn": _turn,
                    "tool_name": call.name,
                    "is_error": is_error,
                    "summary": summary,
                }
            )
            return (
                ToolResult(tool_call_id=call.id, content=output, is_error=is_error),
                TranscriptEntry(
                    turn=_turn,
                    tool_name=call.name,
                    tool_input=call.input,
                    tool_output_summary=summary,
                    is_error=is_error,
                ),
            )

        # 同一轮里模型一次性请求的多个工具调用，彼此之间天然没有依赖关系——
        # 如果某个调用真的需要先看到另一个的结果，模型自己就不会把它们打包
        # 进同一次响应里，而是会分成两轮（先调用一个、看到结果后下一轮再决定
        # 要不要调用另一个）。既然模型已经确认过它们互相独立，这里并行执行
        # 是安全的，跟之前一个个排队发网络请求相比能省下明显的等待时间，
        # 而且现在数据层的磁盘缓存都已经用 cache_lock 做了并发保护（见
        # polygon_client.py 等），不会因为并发访问同一份缓存文件出问题。
        call_pairs = await asyncio.gather(*(_execute_call(call) for call in response.tool_calls))
        results = [pair[0] for pair in call_pairs]
        transcript.extend(pair[1] for pair in call_pairs)
        # 追加"这一轮新的"工具结果之前，先压缩messages里已经被看过至少一轮的
        # 大体积旧工具结果——见CONTEXT_COMPACTION_THRESHOLD_CHARS的注释。
        # enable_compaction=False 时完全跳过（不是"压缩阈值调到无穷大"这种
        # 变相实现，是真正的旁路），给eval_report_quality.py的--no-compaction
        # 开关用，方便随时重跑压缩前后的质量对照，不用每次手改代码
        if enable_compaction:
            messages, compaction_records = _compact_seen_tool_results(messages, tool_call_names)
            # turn+1：跟AgentRunResult.turns_used统一用1-indexed，不要在日志里
            # 又冒出一套从0开始数的轮次口径
            _append_compaction_log(ticker, turn + 1, compaction_records)
        messages = llm.append_tool_results(messages, response.tool_calls, results)

    # 达到 max_turns 还没拿到 end_turn：明确记录成"未完成"，不当成崩溃
    _final_report = _resolve_final_report(response.text if response else None, reasoning_notes)
    _traceable_matched, _traceable_total = score_traceability(_final_report or "", raw_tool_outputs)
    _gate_resolutions = _compute_gate_resolutions(
        _final_report, transcript, raw_tool_outputs, verify_number_outcomes, _traceable_matched, _traceable_total
    )
    return AgentRunResult(
        ticker=ticker,
        completed=False,
        stop_reason="max_turns_exceeded",
        final_report=_final_report,
        reasoning_notes=reasoning_notes,
        transcript=transcript,
        turns_used=max_turns,
        structure_gate_triggered=structure_gate_triggered,
        structure_gate_resolved=_gate_resolutions["structure"],
        tool_coverage_gate_triggered=tool_coverage_gate_triggered,
        tool_coverage_gate_resolved=_gate_resolutions["tool_coverage"],
        self_verification_triggered=_has_self_verification(transcript),
        verification_mismatch_triggered=verification_mismatch_triggered,
        verification_mismatch_resolved=_gate_resolutions["verification_mismatch"],
        traceability_gate_triggered=traceability_gate_triggered,
        traceability_gate_resolved=_gate_resolutions["traceability"],
        sentiment_consistency_gate_triggered=sentiment_consistency_gate_triggered,
        sentiment_consistency_gate_resolved=_gate_resolutions["sentiment_consistency"],
        reflexion_triggered=reflexion_triggered,
        traceable_numbers_matched=_traceable_matched,
        traceable_numbers_total=_traceable_total,
    )
