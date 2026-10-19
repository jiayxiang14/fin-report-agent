"""run_agent_loop 从检查点恢复：同一份脚本化的 LLM 响应，"跑到一半中断再
恢复"和"一口气跑完"的最终结果必须完全一致，而且恢复后不会重复调用已经
做过的那次 LLM 请求。

存档经过真实的 JSON 序列化往返（checkpoint._state_to_dict/_state_from_dict），
不是把内存里的对象直接传回去——这样才能测出"恢复时漏还原了某个字段"。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.services.agent import checkpoint
from app.services.agent.checkpoint import ResumePoint
from app.services.agent.llm_client import LLMResponse, LLMUsage, ToolCall
from app.services.agent.loop import run_agent_loop


class FakeLLMClient:
    """按顺序返回预设响应的假LLM客户端（跟test_agent_loop.py里的同名类同一个
    思路，这里只保留需要的最小子集）。"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls = 0

    async def create_message(self, system, messages, tools, temperature=None, on_text_delta=None):
        response = self._responses[self.calls]
        self.calls += 1
        return response

    def append_assistant_turn(self, messages, response):
        return [*messages, {"role": "assistant", "content": [{"type": "text", "text": response.text or ""}]}]

    def append_tool_results(self, messages, tool_calls, results):
        return [*messages, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content} for r in results]}]


REPORT = "<conclusion>结论</conclusion><evidence>证据</evidence><flags>无</flags>"


def _responses() -> list[LLMResponse]:
    # turn0 调工具 → turn1 写出简报但还没自我核查，被批量gate打回 →
    # turn2 调 verify_number → turn3 写出最终简报
    return [
        LLMResponse(
            stop_reason="tool_use",
            text="先查财务数据",
            tool_calls=[ToolCall(id="t0", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
            usage=LLMUsage(100, 10),
        ),
        LLMResponse(stop_reason="end_turn", text=REPORT, tool_calls=[], raw=None, usage=LLMUsage(200, 20)),
        LLMResponse(
            stop_reason="tool_use",
            text="核查一下营收",
            tool_calls=[ToolCall(id="t2", name="verify_number", input={"metric": "revenue", "period": "FY2025"})],
            raw=None,
            usage=LLMUsage(300, 30),
        ),
        LLMResponse(stop_reason="end_turn", text=REPORT, tool_calls=[], raw=None, usage=LLMUsage(400, 40)),
    ]


async def _fake_execute_tool(name, tool_input):
    if name == "verify_number":
        return json.dumps({"metric": "revenue", "period": "FY2025", "matches": True}), False
    return "{}", False


def _run(responses, **kwargs):
    fake = FakeLLMClient(responses)
    tool_mock = AsyncMock(side_effect=_fake_execute_tool)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=tool_mock),
    ):
        result = asyncio.run(run_agent_loop("AAPL", **kwargs))
    return result, fake, tool_mock


def _record_checkpoints():
    """跑一遍完整流程，把每个存档点经过JSON往返后存下来。"""
    saved: dict[tuple[str, int], ResumePoint] = {}

    def on_checkpoint(phase, turn, state, pending):
        state_json = json.loads(json.dumps(checkpoint._state_to_dict(state), ensure_ascii=False))
        pending_json = json.loads(json.dumps(checkpoint._response_to_dict(pending))) if pending else None
        saved[(phase, turn)] = ResumePoint(
            phase=phase,
            turn=turn,
            state=checkpoint._state_from_dict(state_json),
            pending_response=checkpoint._response_from_dict(pending_json) if pending_json else None,
        )

    result, _, _ = _run(_responses(), on_checkpoint=on_checkpoint)
    return result, saved


def _comparable(result):
    return result.model_dump()


def test_every_turn_saves_turn_start_and_llm_done_checkpoints():
    result, saved = _record_checkpoints()

    assert result.completed is True
    assert result.turns_used == 4
    assert sorted(saved) == sorted(
        [(phase, turn) for turn in range(4) for phase in ("turn_start", "llm_done")]
    )
    # 被批量gate打回之后的那一轮开头，存档里已经带上了nudge消息和标记
    turn2_start = saved[("turn_start", 2)].state
    assert turn2_start.nudged_for_gate_batch is True
    assert turn2_start.messages[-1]["role"] == "user"


def test_resume_from_turn_start_matches_uninterrupted_run_and_skips_earlier_llm_calls():
    full_result, saved = _record_checkpoints()

    resumed_result, fake, _ = _run(_responses()[2:], resume_from=saved[("turn_start", 2)])

    assert fake.calls == 2  # 只调用了 turn2、turn3，前两轮没有重做
    assert _comparable(resumed_result) == _comparable(full_result)


def test_resume_from_llm_done_does_not_call_the_llm_again_for_that_turn():
    full_result, saved = _record_checkpoints()

    resumed_result, fake, tool_mock = _run(_responses()[3:], resume_from=saved[("llm_done", 2)])

    assert fake.calls == 1  # turn2 的 LLM 响应来自存档，只调用了 turn3
    # turn2 的工具（verify_number）重新执行了一次——工具只读，重做是安全的
    assert [c.args[0] for c in tool_mock.call_args_list] == ["verify_number"]
    assert _comparable(resumed_result) == _comparable(full_result)


def test_resume_from_llm_done_of_an_end_turn_reruns_the_gate_check():
    full_result, saved = _record_checkpoints()

    # turn1 是 end_turn：从这里恢复要重新跑批量gate，照样打回去要求自我核查
    resumed_result, fake, _ = _run(_responses()[2:], resume_from=saved[("llm_done", 1)])

    assert fake.calls == 2
    assert _comparable(resumed_result) == _comparable(full_result)


def test_resumed_run_keeps_saving_checkpoints():
    _, saved = _record_checkpoints()
    phases = []

    _run(
        _responses()[2:],
        resume_from=saved[("turn_start", 2)],
        on_checkpoint=lambda phase, turn, state, pending: phases.append((phase, turn)),
    )

    assert phases == [("turn_start", 2), ("llm_done", 2), ("turn_start", 3), ("llm_done", 3)]
