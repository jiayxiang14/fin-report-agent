""""草稿 + 审阅 + 修改"：深度分析（Best-of-N，3个候选+裁判择优）的候选替代方案，
目前只在评测里使用（scripts/eval_report_rubric.py），还没接进产品。

借鉴 claude-quickstarts 的 roadtrip-planner：写草稿的Agent把草稿交给一个更强
模型的审阅者挑毛病，再根据审阅意见修改。成本大约是普通分析的2倍（一次完整
Agent Loop + 一次审阅 + 一次修改），而Best-of-N是3-4倍。

角色隔离（同一家供应商内部能做到的部分）：
- 模型：写草稿/修改用默认的flash，审阅用更强的deepseek-v4-pro。
- 上下文：三个角色各是一段全新的对话。审阅者只看到草稿正文和工具返回的数据，
  看不到写草稿时的推理过程和工具调用顺序——只能就"简报写的"和"数据显示的"
  对质，不会被写草稿者的思路带着走。修改时写草稿者看到审阅意见，但不知道审阅者
  是谁、用的什么模型。
- Prompt：审阅者是独立的系统Prompt，被设定为专门挑错的审稿人，不负责重写。
做不到的部分：两个模型来自同一家供应商，可能有相同的盲区——审阅者可能对某类
错误同样视而不见。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.agent import AgentRunResult
from app.services.agent.llm_client import get_llm_client
from app.services.agent.loop import run_agent_loop
from app.services.agent.system_prompt import SYSTEM_PROMPT

REVIEWER_MODEL = "deepseek-v4-pro"
NO_ISSUES_MARKER = "无需修改"
# 财报原文动辄几十万字符，原样塞给审阅者既贵又会淹没结构化数据——只给开头一段，
# 审阅重点是"简报里的数字和判断对不对得上结构化数据"
FILING_TEXT_EXCERPT_CHARS = 15_000
_REQUIRED_TAGS = ("conclusion", "evidence", "flags")
# deepseek-v4-pro会先输出很长的thinking：默认的8192被thinking用光、审阅正文一个字
# 没写就截断，真实复现过——空回复曾被当成"无需修改"，审阅等于没做、钱照付
REVIEWER_MAX_TOKENS = 32_000
# 默认的high推理强度下，审阅一份简报要输出约2万token的thinking、耗时约5分钟，偶尔
# 还会超过上面的上限被截断；low实测约5千token、1分钟，意见仍然有实质内容
REVIEWER_EFFORT = "low"

REVIEWER_SYSTEM_PROMPT = """你是一名独立的投研审稿人，只负责挑错，不负责重写简报。

你会拿到一份美股投研简报草稿，以及生成这份草稿时各个数据工具返回的原始数据。
你看不到作者的推理过程，只能拿"简报写了什么"和"数据显示什么"对质。逐条检查：

1. 事实错误：简报里的数字、涨跌方向、超预期/不及预期、板块位置，跟原始数据对不上的
2. 口径或期间混淆：把不同季度、不同口径（比如GAAP和调整后EPS）的数据当成同一回事比较的
3. 遗漏的关键信息：数据里有、对结论有实质影响、简报却没提的（比如超预期但股价大跌这种矛盾）
4. 没有依据的判断：结论或推断在数据里找不到支撑的
5. 数据缺失没有说明：某个工具报错或没返回数据，简报却没在异常提示里写明的

每条问题都要指出简报里的原话和对应的数据。按严重程度排序，最多列8条。
如果没有发现实质问题，只回复"无需修改"四个字，不要为了显得认真而挑细枝末节。"""

REVISION_INSTRUCTION = """下面是你之前写的投研简报草稿，以及一位独立审稿人的意见。请逐条核对审稿意见：
确实有道理的就改，你认为审稿人说错了的可以不改，但要以数据为准。
改完后重新完整输出一遍简报，格式要求跟原来完全一样（<conclusion>/<evidence>/<flags>/<sentiment>
四个标签都要有），只输出简报本身，不要解释你改了什么。

【原简报草稿】
{draft}

【审稿意见】
{critique}

【工具返回的原始数据】
{tool_data}"""


@dataclass
class DraftReviewResult:
    draft: AgentRunResult
    critique: str
    final_report: str | None
    # 修改后的简报缺了必需标签时退回用草稿（不让一次格式失误把整份分析弄丢），
    # 这里如实记录有没有发生退回
    fell_back_to_draft: bool
    # 审阅者没给出可用的意见（空回复/被截断）——跟"审阅后认为无需修改"是两回事，
    # 不能混为一谈
    review_failed: bool = False
    # 排查用：审阅调用的终止原因和输出token数（审阅失败时看是被截断还是别的原因）
    review_stop_reason: str | None = None
    review_output_tokens: int | None = None


def _render_tool_data(tool_outputs: list[tuple[str, str]]) -> str:
    parts = []
    for name, output in tool_outputs:
        text = output[:FILING_TEXT_EXCERPT_CHARS] + "\n…（以下省略）" if name == "get_filing_text" and len(output) > FILING_TEXT_EXCERPT_CHARS else output
        parts.append(f"### {name}\n{text}")
    return "\n\n".join(parts) if parts else "（本次没有成功返回的工具数据）"


def _has_required_tags(report: str) -> bool:
    return all(re.search(rf"<{tag}>[\s\S]*?</{tag}>", report) for tag in _REQUIRED_TAGS)


async def run_draft_review(ticker: str) -> DraftReviewResult:
    tool_outputs: list[tuple[str, str]] = []

    async def capture(tool_name: str, output: str) -> None:
        tool_outputs.append((tool_name, output))

    draft = await run_agent_loop(ticker, on_tool_result=capture)
    if not draft.final_report:
        return DraftReviewResult(draft=draft, critique="", final_report=None, fell_back_to_draft=False)

    tool_data = _render_tool_data(tool_outputs)
    reviewer = get_llm_client(model=REVIEWER_MODEL)
    review = await reviewer.create_message(
        system=REVIEWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"【简报草稿】\n{draft.final_report}\n\n【工具返回的原始数据】\n{tool_data}"}],
        tools=[],
        temperature=0.0,
        max_tokens=REVIEWER_MAX_TOKENS,
        effort=REVIEWER_EFFORT,
    )
    critique = (review.text or "").strip()
    stop_reason, output_tokens = review.stop_reason, review.usage.output_tokens
    if review.stop_reason != "end_turn" or not critique:
        return DraftReviewResult(
            draft=draft, critique=critique, final_report=draft.final_report, fell_back_to_draft=False, review_failed=True, review_stop_reason=stop_reason, review_output_tokens=output_tokens
        )
    if critique.startswith(NO_ISSUES_MARKER):
        return DraftReviewResult(draft=draft, critique=critique, final_report=draft.final_report, fell_back_to_draft=False, review_stop_reason=stop_reason, review_output_tokens=output_tokens)

    writer = get_llm_client()
    revision = await writer.create_message(
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": REVISION_INSTRUCTION.format(draft=draft.final_report, critique=critique, tool_data=tool_data),
            }
        ],
        tools=[],
    )
    revised = (revision.text or "").strip()
    if not _has_required_tags(revised):
        return DraftReviewResult(draft=draft, critique=critique, final_report=draft.final_report, fell_back_to_draft=True, review_stop_reason=stop_reason, review_output_tokens=output_tokens)
    return DraftReviewResult(draft=draft, critique=critique, final_report=revised, fell_back_to_draft=False, review_stop_reason=stop_reason, review_output_tokens=output_tokens)
