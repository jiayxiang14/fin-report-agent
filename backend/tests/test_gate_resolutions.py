"""`_compute_gate_resolutions`：真实复现过的bug——gate拦下来发了nudge，但
模型的后续回应没有真的解决问题（比如只回一句不带标签的收尾话），
`_resolve_final_report`的回退机制会原样捞回没解决问题的旧草稿，
`*_triggered=True`就变成了一句谎言。这里在拿到最终状态之后用跟gate检测
阶段相同的判断逻辑重新核对一遍，跟`*_triggered`（历史事实：有没有插过
nudge）是两个独立信号。

这个文件直接测`_compute_gate_resolutions`这个纯函数本身（快、精确），
另外补一个`max_turns_exceeded`路径下resolved字段计算是否正确的集成测试——
`test_structure_gate_nudge.py`等5个gate专属文件里已经通过真实Loop运行覆盖
了每道gate各自的triggered/resolved组合，这里不重复。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.models.agent import TranscriptEntry
from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import _compute_gate_resolutions, run_agent_loop


def _entry(tool_name: str, is_error: bool = False) -> TranscriptEntry:
    return TranscriptEntry(turn=0, tool_name=tool_name, tool_input={}, tool_output_summary="", is_error=is_error)


def test_structure_resolved_reflects_final_report_tags():
    resolutions = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion><evidence>e</evidence>",  # 缺flags
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions["structure"] is False


def test_tool_coverage_resolved_reflects_transcript_ground_truth():
    resolutions = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[_entry("verify_number")],  # 没有get_financials
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions["tool_coverage"] is False

    resolutions_ok = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[_entry("get_financials"), _entry("verify_number")],
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions_ok["tool_coverage"] is True


def test_tool_coverage_resolved_ignores_failed_tool_calls():
    """get_financials调用过，但is_error=True，不该算数——工具真的失败了，
    数据没拿到。"""
    resolutions = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[_entry("get_financials", is_error=True)],
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions["tool_coverage"] is False


def test_verification_mismatch_resolved_reflects_last_outcome_of_the_same_claim():
    resolutions_still_false = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[("revenue", "annual", False)],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions_still_false["verification_mismatch"] is False

    # 同一个(metric, period)重新核实之后对上了——真的解决了
    resolutions_now_true = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[("revenue", "annual", False), ("revenue", "annual", True)],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions_now_true["verification_mismatch"] is True


def test_verification_mismatch_on_one_metric_is_not_masked_by_a_different_metric_succeeding():
    """真实复现过的bug：核查营收发现不对（matches=False），没去改，紧接着
    核查净利润，这次对上了（matches=True）——旧逻辑只看整个调用序列的最后
    一个元素，会把营收那个明确有问题、却从没被处理的mismatch，被净利润这次
    完全不相关的成功核查"顺手"掩盖掉。营收和净利润是两个独立的(metric,period)
    组合，各自的最新结果都要单独判断——营收还是False，这道gate就不该判定成
    "已解决"。"""
    resolutions = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[("revenue", "annual", False), ("net_income", "annual", True)],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions["verification_mismatch"] is False


def test_traceability_resolved_reflects_final_ratio_against_threshold():
    resolutions_low = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=1,
        traceable_total=10,  # 10% < 30%阈值
    )
    assert resolutions_low["traceability"] is False

    resolutions_healthy = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion>",
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=8,
        traceable_total=10,  # 80% >= 30%阈值
    )
    assert resolutions_healthy["traceability"] is True


def test_sentiment_consistency_gate_resolved_reflects_final_sentiment_vs_price_reaction():
    price_reaction = json.dumps({"ticker": "AAPL", "has_data": True, "price_change_pct": -5.0})

    resolutions_still_contradicting = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion><sentiment>positive</sentiment>",
        transcript=[],
        raw_tool_outputs=[price_reaction],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions_still_contradicting["sentiment_consistency"] is False

    resolutions_fixed = _compute_gate_resolutions(
        final_report="<conclusion>c</conclusion><sentiment>neutral</sentiment>",
        transcript=[],
        raw_tool_outputs=[price_reaction],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions_fixed["sentiment_consistency"] is True


def test_all_resolved_default_true_for_empty_final_state():
    """final_report为None、transcript/raw_tool_outputs都是空的场景（比如
    refusal），不该被误判成"有问题"——没有任何检测依据时默认都是True。"""
    resolutions = _compute_gate_resolutions(
        final_report=None,
        transcript=[],
        raw_tool_outputs=[],
        verify_number_outcomes=[],
        traceable_matched=0,
        traceable_total=0,
    )
    assert resolutions == {
        "structure": False,  # None -> ""，缺evidence/flags标签，如实反映"确实没有"
        "tool_coverage": False,  # 没调用过get_financials，如实反映
        "verification_mismatch": True,  # 没有核查记录，vacuous true
        "traceability": True,  # total=0，vacuous true
        "sentiment_consistency": True,  # 没有sentiment标签，vacuous true
    }


class FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls = 0

    async def create_message(self, system, messages, tools, temperature=None):
        response = self._responses[self.calls]
        self.calls += 1
        return response

    def append_assistant_turn(self, messages, response):
        return [*messages, {"role": "assistant", "content": "fake"}]

    def append_tool_results(self, messages, tool_calls, results):
        return [*messages, {"role": "user", "content": "fake"}]


def test_resolved_fields_computed_correctly_on_max_turns_exceeded():
    """结构gate在第1轮触发、插入nudge，但预算耗尽（max_turns=2：第0轮工具
    调用，第1轮end_turn被结构gate拦下插nudge，range(2)到此为止，没有第2轮
    真正把修正版写出来）——resolved字段应该如实反映"确实还没解决"，不因为
    落到max_turns_exceeded分支就被抹成默认值。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="0", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(
            stop_reason="end_turn", text="<conclusion>结论</conclusion><flags>f</flags>", tool_calls=[], raw=None
        ),  # 缺evidence，触发结构gate
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", max_turns=2))

    assert result.stop_reason == "max_turns_exceeded"
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is False
    # 同一份状态下，从没被检查过的工具覆盖gate：get_financials确实调用过，
    # 即便triggered从未为True，resolved也该如实反映"这个条件本身是满足的"
    assert result.tool_coverage_gate_resolved is True
