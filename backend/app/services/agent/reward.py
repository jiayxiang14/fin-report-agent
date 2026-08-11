"""Best-of-N 奖励函数：规则打分（确定性、免费）+ 两个 LLM 裁判打分——结论裁判评
`final_report`写得好不好，过程裁判评`reasoning_notes`/`transcript`这条决策轨迹
走得好不好。项目书第六节要求的权重先手动设定，跑几次后再手动调整，不做自动调
权重——所有权重/阈值都是本文件顶部的模块常量。

规则打分只依赖 AgentRunResult 里已有的字段（final_report/transcript）和调用方
另外收集的"完整未截断工具输出"（loop.py 新增的 on_tool_result 回调产出，不是
transcript.tool_output_summary——那个已经被截断到300字符，做数字核对会漏掉）。

之前奖励函数是纯outcome-only：只看final_report这段最终文本写得好不好，Agent
自己的决策路径（信息收集是否充分、有没有主动验证存疑数字、工具调用有没有绕
弯路）完全不参与打分——哪怕transcript/reasoning_notes里其实记录了这些信息。
新增的过程裁判（score_trajectory_judge）专门补这一块：喂给它的是轨迹本身，
不是最终简报，这样"结论写得漂亮但过程偷懒"和"过程扎实但文字平淡"能被区分开。
"""

from __future__ import annotations

import json
import re

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from app.models.best_of_n import RuleScoreBreakdown
from app.services.agent.llm_client import get_llm_client

TRACEABILITY_MAX = 50.0
SELF_VERIFICATION_MAX = 20.0
STRUCTURE_MAX = 20.0
LENGTH_MAX = 10.0

MIN_REPORT_LENGTH = 300
MAX_REPORT_LENGTH = 3000

# 简报文字大概率是四舍五入过的，容差比 verify_number.py 的1%(核对同一个数字来源)
# 稍宽一点
TRACEABILITY_TOLERANCE_PCT = 2.0

# 规则分依然是权重最高的单一分量（"硬性底线"，项目书原话），加入过程裁判后
# 总裁判权重从0.4涨到0.5，在结论裁判和过程裁判之间对半分——不是两者一样重要
# 所以对半分，是目前没有证据支撑"哪个更该多信"，对半分是没有偏见的起点
RULE_WEIGHT = 0.5
OUTCOME_JUDGE_WEIGHT = 0.25
TRAJECTORY_JUDGE_WEIGHT = 0.25

_TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>([\s\S]*?)</{tag}>") for tag in ("conclusion", "evidence", "flags")
}
_NUMBER_TOKEN_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")

_JUDGE_PROMPT_TEMPLATE = """你是一个投研简报质量评审员。请阅读下面这份AI生成的投研简报，
按"逻辑连贯性、有没有自相矛盾、洞察是否有价值"这几个维度打一个1到10的整数分。

只按下面的格式回复，不要有其他任何文字：
<score>分数</score>
<reason>一句话说明理由</reason>

简报内容：
{report}
"""

_TRAJECTORY_JUDGE_PROMPT_TEMPLATE = """你是一个投研Agent的决策过程评审员。下面不是最终简报，
而是Agent在生成简报之前自主进行的完整信息收集与推理轨迹——包括每一步在想什么、
调用了什么工具、工具是否成功。

请按"信息收集是否充分（有没有遗漏明显该查的数据）、有没有主动验证存疑或异常的
数字、工具调用是否高效（有没有明显绕弯路或重复劳动）"这几个维度打一个1到10的
整数分。

只按下面的格式回复，不要有其他任何文字：
<score>分数</score>
<reason>一句话说明理由</reason>

决策轨迹：
{trajectory}
"""

_SCORE_TAG_PATTERN = re.compile(r"<score>\s*(\d+)\s*</score>")
_REASON_TAG_PATTERN = re.compile(r"<reason>([\s\S]*?)</reason>")


def _extract_tag(text: str, tag: str) -> str | None:
    match = _TAG_PATTERNS[tag].search(text)
    return match.group(1).strip() if match else None


def _looks_like_year(raw_token: str, value: float) -> bool:
    return raw_token.isdigit() and len(raw_token) == 4 and 1900 <= value <= 2100


def _looks_like_trivial_integer(raw_token: str) -> bool:
    # 一位数的纯整数常见于列表序号/轮次编号，不是真正需要溯源的数据主张
    return raw_token.isdigit() and len(raw_token) == 1


def _extract_claimed_numbers(text: str) -> list[float]:
    numbers = []
    for raw_token in _NUMBER_TOKEN_PATTERN.findall(text):
        cleaned = raw_token.replace(",", "")
        if cleaned in ("", "-", "."):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if _looks_like_year(cleaned, value) or _looks_like_trivial_integer(cleaned):
            continue
        numbers.append(value)
    return numbers


def _walk_json_numbers(node: object, acc: set[float]) -> None:
    """递归收集"已知数字"。除了JSON里本来就是数值类型的叶子值，字符串叶子值
    也要用同一套正则去抽取——`get_filing_text`返回的财报原文是整段塞在`text`
    字段里的字符串，简报引用的数字如果来源是财报原文叙述而不是`get_financials`
    这类结构化字段，之前完全不会被收进"已知数字"集合，这是真实跑过AMZN后
    发现的可追溯性打分漏判的主要来源（比预想的"单位换算问题"更严重）。
    """
    if isinstance(node, bool):
        return
    if isinstance(node, int | float):
        acc.add(float(node))
    elif isinstance(node, str):
        acc.update(_extract_claimed_numbers(node))
    elif isinstance(node, dict):
        for value in node.values():
            _walk_json_numbers(value, acc)
    elif isinstance(node, list):
        for item in node:
            _walk_json_numbers(item, acc)


def _collect_known_numbers(raw_tool_outputs: list[str]) -> set[float]:
    known: set[float] = set()
    for raw in raw_tool_outputs:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        _walk_json_numbers(payload, known)
    return known


def _has_close_match(value: float, known: set[float]) -> bool:
    for candidate in known:
        if candidate == 0:
            if value == 0:
                return True
            continue
        if abs(value - candidate) / abs(candidate) * 100 <= TRACEABILITY_TOLERANCE_PCT:
            return True
    return False


def _score_traceability(report_text: str, raw_tool_outputs: list[str]) -> tuple[float, int, int]:
    evidence = _extract_tag(report_text, "evidence") or ""
    flags = _extract_tag(report_text, "flags") or ""
    claimed = _extract_claimed_numbers(f"{evidence}\n{flags}")
    if not claimed:
        # 没有可核对的数字主张，不等于"数字有问题"，给满分
        return TRACEABILITY_MAX, 0, 0
    known = _collect_known_numbers(raw_tool_outputs)
    matched = sum(1 for value in claimed if _has_close_match(value, known))
    return matched / len(claimed) * TRACEABILITY_MAX, matched, len(claimed)


def _score_self_verification(transcript: list[TranscriptEntry]) -> float:
    triggered = any(entry.tool_name == "verify_number" and not entry.is_error for entry in transcript)
    return SELF_VERIFICATION_MAX if triggered else 0.0


def _score_structure(report_text: str) -> float:
    tags_present = sum(1 for tag in ("conclusion", "evidence", "flags") if _extract_tag(report_text, tag))
    return tags_present / 3 * STRUCTURE_MAX


def _score_length(report_text: str) -> float:
    length = len(report_text)
    if length == 0:
        return 0.0
    if MIN_REPORT_LENGTH <= length <= MAX_REPORT_LENGTH:
        return LENGTH_MAX
    return LENGTH_MAX / 2  # 太短信息不够、太长可能啰嗦，都只是小幅扣分，不是0分


def _render_trajectory(reasoning_notes: list[ReasoningNote], transcript: list[TranscriptEntry]) -> str:
    """把推理文字和工具调用按轮次交替，渲染成给过程裁判读的文本——裁判评的是
    "怎么一步步走到结论的"，喂给它的应该是transcript/reasoning_notes这类原始
    结构化记录，不是最终简报正文（那是score_llm_judge评的对象）。同一轮次内
    先放推理文字再放工具调用，符合"模型先写点什么、再决定调哪个工具"这个实际
    发生顺序（list.sort是稳定排序，同key时保留原始添加顺序）。
    """
    steps: list[tuple[int, str]] = []
    for note in reasoning_notes:
        steps.append((note.turn, f"[第{note.turn}轮] 推理：{note.text}"))
    for entry in transcript:
        status = "失败" if entry.is_error else "成功"
        steps.append((entry.turn, f"[第{entry.turn}轮] 调用工具 {entry.tool_name}（{status}）：{entry.tool_output_summary}"))
    steps.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in steps)


async def _call_judge(prompt: str) -> tuple[float | None, str | None]:
    """裁判调用的共享实现：结论裁判（score_llm_judge）和过程裁判
    （score_trajectory_judge）用的是同一套"发起不带工具的LLM调用+解析
    <score>/<reason>标签"逻辑，只是喂进去的prompt内容不同，抽成一个函数
    避免两份几乎一样的try/except+正则解析代码。调用失败或没按格式回复时
    统一返回(None, None)，调用方据此软降级，不编造一个"中性分"去混合，
    那样会稀释真实信号。
    """
    llm = get_llm_client()
    try:
        response = await llm.create_message(
            system="你是一个严格但公正的评审员。",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001 - 裁判调用是软性加分项，失败要降级不能拖垮整个候选打分
        return None, None

    text = response.text or ""
    score_match = _SCORE_TAG_PATTERN.search(text)
    if not score_match:
        return None, None

    raw_score = max(1, min(10, int(score_match.group(1))))
    reason_match = _REASON_TAG_PATTERN.search(text)
    reason = reason_match.group(1).strip() if reason_match else None
    return raw_score / 10 * 100, reason


def score_rule_based(run_result: AgentRunResult, raw_tool_outputs: list[str]) -> RuleScoreBreakdown:
    report_text = run_result.final_report or ""
    traceability, matched, total = _score_traceability(report_text, raw_tool_outputs)
    self_verification = _score_self_verification(run_result.transcript)
    structure = _score_structure(report_text)
    length = _score_length(report_text)

    return RuleScoreBreakdown(
        traceability=round(traceability, 2),
        traceability_matched=matched,
        traceability_total=total,
        self_verification=self_verification,
        structure=round(structure, 2),
        length=length,
        total=round(traceability + self_verification + structure + length, 2),
    )


async def score_llm_judge(final_report: str | None) -> tuple[float | None, str | None]:
    """额外发起一次LLM调用给候选简报（最终产出的文本）打1-10分，归一化到0-100。
    评的是"结果"——跟score_trajectory_judge评"过程"是两个独立维度。
    """
    if not final_report:
        return None, None
    return await _call_judge(_JUDGE_PROMPT_TEMPLATE.format(report=final_report))


async def score_trajectory_judge(
    reasoning_notes: list[ReasoningNote], transcript: list[TranscriptEntry]
) -> tuple[float | None, str | None]:
    """额外发起一次LLM调用给Agent的决策轨迹（推理文字+工具调用序列，不是最终
    简报）打1-10分，归一化到0-100。没有任何推理文字或工具调用时（理论上不该
    发生，但防御性地处理）没有东西可评，返回(None, None)，跟final_report为空
    时score_llm_judge的处理方式一致。
    """
    trajectory = _render_trajectory(reasoning_notes, transcript)
    if not trajectory:
        return None, None
    return await _call_judge(_TRAJECTORY_JUDGE_PROMPT_TEMPLATE.format(trajectory=trajectory))


def combine_scores(
    rule_score: RuleScoreBreakdown,
    outcome_llm_score: float | None,
    trajectory_llm_score: float | None,
) -> float:
    """总分是规则分/结论裁判分/过程裁判分的加权平均。任一裁判没打成分（调用
    失败、格式解析失败、没有可评内容）时不参与计算，权重在剩余分量间按比例
    重新归一——不编造一个"中性分"去混合，那样会稀释真实信号，这是结论裁判
    从一开始就有的降级哲学，过程裁判延用同一套。
    """
    components: list[tuple[float, float]] = [(rule_score.total, RULE_WEIGHT)]
    if outcome_llm_score is not None:
        components.append((outcome_llm_score, OUTCOME_JUDGE_WEIGHT))
    if trajectory_llm_score is not None:
        components.append((trajectory_llm_score, TRAJECTORY_JUDGE_WEIGHT))

    total_weight = sum(weight for _, weight in components)
    weighted_sum = sum(score * weight for score, weight in components)
    return round(weighted_sum / total_weight, 2)
