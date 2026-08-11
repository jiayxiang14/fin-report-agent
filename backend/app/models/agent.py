from pydantic import BaseModel


class TranscriptEntry(BaseModel):
    turn: int
    tool_name: str
    tool_input: dict
    tool_output_summary: str  # 截断过的摘要，完整内容不适合放进前端展示
    is_error: bool


class ReasoningNote(BaseModel):
    """模型在某一轮里写的文字（不管这一轮是不是同时调用了工具）。之前这部分
    文字在中间轮次会被静默丢弃，现在完整记录下来——既是第4阶段要展示的
    "推理过程"的原始数据，也是唯一能验证"自我核查确实发生在草稿之后"的证据。"""

    turn: int
    text: str


class AgentRunResult(BaseModel):
    ticker: str
    completed: bool  # 只有 stop_reason == "end_turn" 才是 True，refusal/max_tokens/未知原因都是 False
    stop_reason: str  # "end_turn" / "max_turns_exceeded" / "refusal" / "max_tokens" / "error"
    final_report: str | None
    reasoning_notes: list[ReasoningNote]
    transcript: list[TranscriptEntry]
    turns_used: int
    # 之前只有 Best-of-N 内部打分（reward.py）能看到"这次有没有做自我核查"，普通
    # 单次分析完全没人检查。现在 loop.py 直接从 transcript 推出这个信号并对外暴露，
    # 默认 False 是给测试里手写的 AgentRunResult(...) 兜底，不强制它们都传这个字段
    self_verification_triggered: bool = False
    # Reflexion：只有传了 reflexion_check 参数的调用方（目前只有 best_of_n.py）
    # 才可能触发，普通单次分析的调用方不传这个参数，永远是 False
    reflexion_triggered: bool = False
