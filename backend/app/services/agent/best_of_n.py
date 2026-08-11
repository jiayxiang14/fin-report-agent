"""Best-of-N 编排：对同一个ticker并行跑N次完整Agent Loop（不同temperature制造
候选间差异），每个候选在收尾前会先经过一次条件性的Reflexion整改（过程裁判打分
不及格才触发，见_decide_reflexion），然后用 reward.py 的奖励函数给每份候选打分，
选出总分最高的一份。

Reflexion不是重新生成整份简报，是在同一个Agent Loop对话里多续一轮——模型看到
过程裁判的具体批评后，可以选择重新调用工具（比如再核实一个数字）或者补充分析，
工具依然可用，跟纯文本层面的"重写一遍"不是一回事。跟Best-of-N本身是互补关系，
不是替代：N份候选依然是各自独立采样的，Reflexion只是让每份候选在被打分前有
机会先自己纠正一次明显的过程缺陷。

并行执行：SEC EDGAR/Polygon/Alpha Vantage的磁盘缓存现在都用 cache_lock.get_lock
按key互斥（每个函数内部自己加的锁），同一个ticker的N个候选并发请求同一份数据时
会排队+单飞（第一个真正发请求写缓存，其余的等锁后直接读到新鲜缓存），不再需要
靠"顺序执行"来规避竞态——这也是为什么这里从for循环改成了asyncio.gather。

单个候选失败不拖垮整批：真实使用时遇到过DeepSeek账户余额不足导致某个候选中途
报402的情况——如果没有隔离，已经跑完、真花了钱的其它候选会跟着这一次异常一起被
扔掉，用户什么都看不到。所以每个候选内部自己try/except，失败的返回一个带error
字段的CandidateSummary（不参与打分和选择），其余候选照常继续跑；只有N个候选
全部失败时才对外抛出BestOfNError。
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from app.models.best_of_n import BestOfNResult, CandidateSummary
from app.services.agent import reward
from app.services.agent.loop import OnEvent, ReflexionCheck, run_agent_loop
from app.services.polygon_client import CACHE_DIR

CANDIDATE_TEMPERATURES: tuple[float, ...] = (0.3, 0.6, 1.0)

# Reflexion触发门槛：过程裁判打分低于这个分数才值得让模型回头改一次，不是每个
# 候选都无条件精修——已经写得不错的过程再让模型"顺手改改"，参考文献里这类无
# 依据的自我修正经常不会变好，甚至可能改坏（Huang et al. 2023），所以只在
# 明确不及格时才触发，是一个有门槛的条件性重试，不是无脑跑两遍。
REFLEXION_SCORE_THRESHOLD = 70.0


class BestOfNError(Exception):
    pass


def _decide_reflexion(score: float | None, reason: str | None) -> str | None:
    """纯判断逻辑，不发起任何调用——分数够高或者没打成分（None）都不整改，
    分数不够时把裁判的具体批评原样组装成要塞回对话里的提示文字。跟"怎么拿到
    这个分数"（score_trajectory_judge调用+缓存）分开，方便单独测试阈值判断。
    """
    if score is None or score >= REFLEXION_SCORE_THRESHOLD:
        return None
    return (
        f"过程裁判对你目前的分析过程给出了反馈（{score:.0f}/100，低于及格线"
        f"{REFLEXION_SCORE_THRESHOLD:.0f}分）：{reason}。请据此判断是否需要"
        "补充查询或重新核实，然后再给出最终结论。"
    )


def _make_reflexion_check(cache: dict[str, float | str | bool | None]) -> ReflexionCheck:
    """给某一个候选构造它自己专属的reflexion_check闭包，同时把过程裁判这次
    算出来的(score, reason)记进传入的cache字典——run_agent_loop内部在模型
    想收尾时会调用一次这个闭包判断要不要整改；如果没触发整改，run_agent_loop
    返回后的最终transcript/reasoning_notes跟这次检查时一模一样，_run_candidate
    可以直接复用cache里的结果，不用为了拿"最终打分"再原样问一遍过程裁判——
    之前这里是重复调用两次同一个问题，白白多花一次裁判调用。
    """

    async def reflexion_check(
        reasoning_notes: list[ReasoningNote], transcript: list[TranscriptEntry]
    ) -> str | None:
        score, reason = await reward.score_trajectory_judge(reasoning_notes, transcript)
        cache["computed"] = True
        cache["score"] = score
        cache["reason"] = reason
        return _decide_reflexion(score, reason)

    return reflexion_check


# 复用 polygon_client 已经在 .gitignore 里声明过的共享缓存目录，这里只是往里面
# 多追加一个日志文件，不是缓存（只写不读）——记录"候选/得分/选择结果"，为将来
# 做策略学习积累数据（项目书6.1/6.4），MVP阶段不做训练
RUNS_LOG_PATH: Path = CACHE_DIR / "best_of_n_runs.jsonl"


def _with_candidate_index(on_event: OnEvent | None, index: int) -> OnEvent | None:
    if on_event is None:
        return None

    async def wrapped(event: dict) -> None:
        await on_event({**event, "candidate_index": index})

    return wrapped


def _append_run_log(ticker: str, candidates: list[CandidateSummary], selected_index: int) -> None:
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ticker": ticker,
        "timestamp": datetime.now(UTC).isoformat(),
        "candidates": [candidate.model_dump() for candidate in candidates],
        "selected_index": selected_index,
    }
    with RUNS_LOG_PATH.open("a") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _run_candidate(
    ticker: str, index: int, temperature: float, on_event: OnEvent | None
) -> tuple[CandidateSummary, AgentRunResult | None]:
    """跑一个候选的完整生命周期：Agent Loop（内含条件性Reflexion）→ 三项打分 →
    组装CandidateSummary。返回(summary, run_result)，失败时run_result是None——
    调用方靠这个区分"这个候选能不能参与最终选择"，不用再检查error字段。
    """

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    raw_outputs: list[str] = []
    trajectory_cache: dict[str, float | str | bool | None] = {}

    async def capture(tool_name: str, content: str) -> None:
        raw_outputs.append(content)

    try:
        run_result = await run_agent_loop(
            ticker,
            temperature=temperature,
            on_event=_with_candidate_index(on_event, index),
            on_tool_result=capture,
            reflexion_check=_make_reflexion_check(trajectory_cache),
        )
    except Exception as exc:  # noqa: BLE001 - 单个候选失败（比如上游余额不足/网络错误）
        # 不该让已经跑完、真花了钱的其它候选也被整批扔掉，跳过这一个继续下一个
        error_message = str(exc)
        await emit(
            {
                "type": "candidate_failed",
                "candidate_index": index,
                "temperature": temperature,
                "error": error_message,
            }
        )
        return (
            CandidateSummary(index=index, temperature=temperature, completed=False, final_report=None, error=error_message),
            None,
        )

    rule_score = reward.score_rule_based(run_result, raw_outputs)
    llm_score, llm_reason = await reward.score_llm_judge(run_result.final_report)

    # Reflexion没触发时，run_agent_loop内部最后一次end_turn检查用的
    # reasoning_notes/transcript就是run_result里的最终状态，trajectory_cache
    # 里已经有现成的分数，不用再问一遍过程裁判；触发过的话状态已经变了
    # （多了一轮整改），必须用新状态重新打分才准确
    if trajectory_cache.get("computed") and not run_result.reflexion_triggered:
        trajectory_score = cast("float | None", trajectory_cache["score"])
        trajectory_reason = cast("str | None", trajectory_cache["reason"])
    else:
        trajectory_score, trajectory_reason = await reward.score_trajectory_judge(
            run_result.reasoning_notes, run_result.transcript
        )

    total_score = reward.combine_scores(rule_score, llm_score, trajectory_score)

    summary = CandidateSummary(
        index=index,
        temperature=temperature,
        completed=run_result.completed,
        final_report=run_result.final_report,
        rule_score=rule_score,
        llm_score=llm_score,
        llm_reason=llm_reason,
        trajectory_score=trajectory_score,
        trajectory_reason=trajectory_reason,
        reflexion_triggered=run_result.reflexion_triggered,
        total_score=total_score,
    )

    await emit(
        {
            "type": "candidate_scored",
            "candidate_index": index,
            "temperature": temperature,
            "total_score": total_score,
            "rule_score": rule_score.model_dump(),
            "llm_score": llm_score,
            "llm_reason": llm_reason,
            "trajectory_score": trajectory_score,
            "trajectory_reason": trajectory_reason,
            "reflexion_triggered": run_result.reflexion_triggered,
        }
    )

    return summary, run_result


async def run_best_of_n(ticker: str, on_event: OnEvent | None = None) -> BestOfNResult:
    results = await asyncio.gather(
        *(
            _run_candidate(ticker, index, temperature, on_event)
            for index, temperature in enumerate(CANDIDATE_TEMPERATURES)
        )
    )
    # asyncio.gather按传入顺序返回结果（不是完成顺序），candidates/日志里的
    # 顺序依然是索引0/1/2，跟并行执行前的行为一致
    candidates = [summary for summary, _ in results]
    successful = [(summary, run_result) for summary, run_result in results if run_result is not None]

    if not successful:
        last_error = candidates[-1].error if candidates else "未知错误"
        raise BestOfNError(
            f"Best-of-N 的 {len(CANDIDATE_TEMPERATURES)} 个候选全部运行失败，最后一个错误：{last_error}"
        )

    # successful 里的候选都走过 reward.combine_scores，total_score 不会是 None——
    # 这个字段在模型上留 Optional 是给失败候选（不在 successful 里）用的
    best_summary, best_run_result = max(successful, key=lambda pair: cast(float, pair[0].total_score))
    _append_run_log(ticker, candidates, best_summary.index)

    return BestOfNResult(
        ticker=ticker,
        candidates=candidates,
        selected_index=best_summary.index,
        selected=best_run_result,
    )
