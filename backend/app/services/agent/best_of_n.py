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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from app.models.best_of_n import BestOfNResult, CandidateSummary
from app.services.agent import reward, trace_log
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

# 候选完整工具输出的短期缓存：3个候选谁会胜出，只有等全部跑完打分选择之后
# 才知道，所以每个候选先各自把完整（未截断）工具输出写进自己专属的短期文件，
# 等选出胜出者后，只把*那一个候选*的记录"提升"进trace_log的永久trace文件
# （跟单次分析路径的tool_result_full记录格式一致），其余候选的文件原地不动，
# 靠下面的机会性过期清理自然消失——不这样做的话，3个候选都写永久trace会让
# 磁盘增量变成单次分析路径的3倍，而且Best-of-N本身调用频率低、单次payload
# 可能很大（get_filing_text一次真实约39万字符），不值得为了这个不常用的
# 路径长期攒着两份没被看过的候选数据。
CANDIDATE_TRACE_DIR: Path = CACHE_DIR / "candidate_traces"
CANDIDATE_TRACE_TTL = timedelta(hours=24)


def _candidate_trace_file(trace_id: str, index: int) -> Path:
    return CANDIDATE_TRACE_DIR / f"{trace_id}_{index}.jsonl"


def _write_candidate_trace(trace_id: str, index: int, tool_calls: list[tuple[str, str]]) -> None:
    """候选成功跑完后一次性写入（不是逐个工具调用时写）——capture闭包在
    Loop运行期间只做纯内存的list.append，不碰磁盘，真正的I/O被挪到这里，
    避免任何磁盘问题拖累候选本身正在进行的Agent Loop。"""
    if not tool_calls:
        return
    try:
        CANDIDATE_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with _candidate_trace_file(trace_id, index).open("a") as f:
            for tool_name, output in tool_calls:
                f.write(json.dumps({"tool_name": tool_name, "output": output}, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - 落盘失败不能拖垮候选，跟trace_log.py同一个哲学
        pass


def _promote_candidate_trace(trace_id: str, index: int) -> None:
    """胜出候选的完整工具输出从短期缓存"提升"成永久trace记录——写进
    trace_log已有的同一个{trace_id}.jsonl文件，带上candidate_index，跟
    task_registry落的事件级（截断summary）记录共用同一份trace，按发生
    顺序自然穿插。提升成功后删掉短期文件，避免留两份重复数据；任何一步
    失败都不该往外抛——这是锦上添花的可观测性数据，不是核心链路。"""
    candidate_file = _candidate_trace_file(trace_id, index)
    if not candidate_file.exists():
        return
    try:
        for line in candidate_file.read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            trace_log.append_event(trace_id, {"type": "tool_result_full", "candidate_index": index, **record})
        candidate_file.unlink()
    except Exception:  # noqa: BLE001 - 同上，提升失败不该影响这次Best-of-N运行本身
        pass


def _evict_expired_candidate_traces() -> None:
    """机会性过期清理——跟task_registry._evict_expired()同一个模式，在每次
    新的Best-of-N运行开始时顺手清一次，不需要单独的后台调度器。

    真实存在的竞态：如果两个不同session几乎同时各自发起一次深度分析（session
    并发保护只挡同一个session内部重复提交，挡不住不同session并发），两边的
    清理扫描可能同时扫到同一个陈旧文件，一边删掉了，另一边对同一个路径调
    stat()时文件已经不存在，会抛FileNotFoundError——这个异常必须在每个文件
    自己的粒度上兜住，不能让它把整个清理动作、进而把这次完全不相关的
    run_best_of_n调用一起带崩。
    """
    if not CANDIDATE_TRACE_DIR.exists():
        return
    cutoff = datetime.now() - CANDIDATE_TRACE_TTL
    for f in CANDIDATE_TRACE_DIR.glob("*.jsonl"):
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink(missing_ok=True)
        except OSError:  # 另一个并发清理已经删了这个文件，或者其它瞬时文件系统问题
            continue


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
    ticker: str, index: int, temperature: float, on_event: OnEvent | None, trace_id: str | None = None
) -> tuple[CandidateSummary, AgentRunResult | None]:
    """跑一个候选的完整生命周期：Agent Loop（内含条件性Reflexion）→ 三项打分 →
    组装CandidateSummary。返回(summary, run_result)，失败时run_result是None——
    调用方靠这个区分"这个候选能不能参与最终选择"，不用再检查error字段。
    """

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    raw_outputs: list[str] = []
    full_tool_calls: list[tuple[str, str]] = []
    trajectory_cache: dict[str, float | str | bool | None] = {}

    async def capture(tool_name: str, content: str) -> None:
        raw_outputs.append(content)
        full_tool_calls.append((tool_name, content))

    # 用None初始化：区分下面except捕获到异常时，究竟是run_agent_loop本身没
    # 跑完（run_result还是None），还是loop已经成功、是后面的打分阶段出了
    # 问题（run_result已经被赋值）——两种情况的诊断信息不一样，日志里应该
    # 说清楚，不能都含糊地说成"这个候选失败了"
    run_result: AgentRunResult | None = None
    try:
        run_result = await run_agent_loop(
            ticker,
            temperature=temperature,
            on_event=_with_candidate_index(on_event, index),
            on_tool_result=capture,
            reflexion_check=_make_reflexion_check(trajectory_cache),
        )

        # 落到候选专属的短期缓存，不是永久trace——这时候还不知道这个候选会不会
        # 被选中，只有run_best_of_n算完选择结果后，胜出候选才会被"提升"成永久
        # 记录。放在打分之前写：即使后面打分阶段失败，候选的完整工具输出依然
        # 被捕获下来了，不会因为打分这一步的问题而丢失排查线索。
        if trace_id is not None:
            _write_candidate_trace(trace_id, index, full_tool_calls)

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
    except Exception as exc:  # noqa: BLE001 - 单个候选（Agent Loop本身，或者跑完之后的打分阶段）
        # 失败都不该拖垮整批——之前try/except只包住了run_agent_loop这一步，
        # 打分阶段（reward.score_rule_based等）如果抛异常会直接从这个函数
        # 里逃出去，顺着asyncio.gather把已经成功、真花了钱的其它候选结果
        # 也一起炸没，跟这段代码自己的设计承诺（"单个候选失败不拖垮整批"）
        # 矛盾——真实故障注入复现过这个后果。现在扩大到覆盖整个候选生命周期。
        stage = "打分阶段" if run_result is not None else "Agent Loop 本身"
        error_message = f"[{stage}失败] {exc}"
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


async def run_best_of_n(ticker: str, on_event: OnEvent | None = None, trace_id: str | None = None) -> BestOfNResult:
    """`trace_id`跟`agent.py`路由那次用的是同一个task_id——不传时（比如脚本/
    测试直接调用）整个候选trace机制原样跳过，行为不变。"""
    _evict_expired_candidate_traces()

    results = await asyncio.gather(
        *(
            _run_candidate(ticker, index, temperature, on_event, trace_id)
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

    if trace_id is not None:
        _promote_candidate_trace(trace_id, best_summary.index)

    return BestOfNResult(
        ticker=ticker,
        candidates=candidates,
        selected_index=best_summary.index,
        selected=best_run_result,
    )
