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
from collections.abc import Awaitable, Callable

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from app.services.agent.llm_client import ToolCall, ToolResult, get_llm_client
from app.services.agent.system_prompt import SYSTEM_PROMPT
from app.services.agent.tools import TOOL_SCHEMAS, execute_tool

MAX_TURNS = 8
SUMMARY_LENGTH = 300

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


# 自我核查之前完全靠 system prompt 里的措辞要求"必须核查"，代码层面没有任何强制——
# 模型不调用 verify_number 也能正常 end_turn，只有 Best-of-N 内部打分能事后看出来，
# 普通单次分析没人管。这里加一道软性拦截：模型想结束、但全程一次都没核实过数字时，
# 不直接放行，往对话里插一条提示强制再走一轮。只插一次（nudged_for_self_verification
# 挡住），避免模型如果仍然不核查就在这里死循环——max_turns 本来就是最终兜底。
SELF_VERIFICATION_NUDGE = (
    "在结束之前，你还没有对任何一个关键数字调用 verify_number 做自我核查。"
    "请先核实简报里至少一个关键数字，确认无误后再给出最终结论。"
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
) -> AgentRunResult:
    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    llm = get_llm_client()
    messages = [{"role": "user", "content": f"分析 {ticker} 的投资价值，形成投研简报。"}]
    transcript: list[TranscriptEntry] = []
    reasoning_notes: list[ReasoningNote] = []
    response = None
    nudged_for_self_verification = False
    nudged_for_reflexion = False
    reflexion_triggered = False

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
            if (
                response.stop_reason == "end_turn"
                and not nudged_for_self_verification
                and response.text
                and "<conclusion>" in response.text
                and not _has_self_verification(transcript)
            ):
                nudged_for_self_verification = True
                messages = [*messages, {"role": "user", "content": SELF_VERIFICATION_NUDGE}]
                continue

            # Reflexion：自我核查这道关过了之后，才轮到"整体决策过程好不好"这个
            # 更宽泛的检查——先确认底线要求（有没有核查数字）再谈整体质量，顺序
            # 反过来意义不大。同样只检查一次（nudged_for_reflexion），不会因为
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

            return AgentRunResult(
                ticker=ticker,
                # 只有真正的 end_turn 才算成功完成；refusal/max_tokens/未识别的
                # stop_reason 都不是"顺利产出了完整简报"，不能标记成 completed
                completed=response.stop_reason == "end_turn",
                stop_reason=response.stop_reason,
                final_report=_resolve_final_report(response.text, reasoning_notes),
                reasoning_notes=reasoning_notes,
                transcript=transcript,
                turns_used=turn + 1,
                self_verification_triggered=_has_self_verification(transcript),
                reflexion_triggered=reflexion_triggered,
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
            output, is_error = await execute_tool(call.name, call.input)
            if on_tool_result is not None and not is_error:
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
        messages = llm.append_tool_results(messages, response.tool_calls, results)

    # 达到 max_turns 还没拿到 end_turn：明确记录成"未完成"，不当成崩溃
    return AgentRunResult(
        ticker=ticker,
        completed=False,
        stop_reason="max_turns_exceeded",
        final_report=_resolve_final_report(response.text if response else None, reasoning_notes),
        reasoning_notes=reasoning_notes,
        transcript=transcript,
        turns_used=max_turns,
        self_verification_triggered=_has_self_verification(transcript),
        reflexion_triggered=reflexion_triggered,
    )
