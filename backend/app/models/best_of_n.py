from pydantic import BaseModel

from app.models.agent import AgentRunResult


class RuleScoreBreakdown(BaseModel):
    """规则打分明细，总分100，各分量权重见 reward.py 顶部常量。"""

    traceability: float  # 数字可追溯性，满分50
    traceability_matched: int
    traceability_total: int
    self_verification: float  # 自我核查是否触发，满分20
    structure: float  # 三段式标签是否齐全，满分20
    length: float  # 简报长度是否在合理区间，满分10
    total: float


class CandidateSummary(BaseModel):
    index: int
    temperature: float
    completed: bool
    final_report: str | None
    # 候选整体运行失败时（比如上游LLM调用报402/网络错误），没有任何可打分的内容——
    # rule_score/total_score留空，error带上失败原因，不是硬凑一个0分假装"打分很差"
    rule_score: RuleScoreBreakdown | None = None
    llm_score: float | None = None  # 结论裁判：评final_report，归一化到0-100，裁判调用失败时为None
    llm_reason: str | None = None
    trajectory_score: float | None = None  # 过程裁判：评reasoning_notes/transcript这条决策轨迹，同样0-100
    trajectory_reason: str | None = None
    # Reflexion：过程裁判打分低于阈值时，run_agent_loop内部会把批评意见塞回对话
    # 让模型重新看一遍再收尾——这个候选是否真的触发过这次整改，直接来自
    # AgentRunResult.reflexion_triggered，不是重新推导
    reflexion_triggered: bool = False
    total_score: float | None = None
    error: str | None = None


class BestOfNResult(BaseModel):
    ticker: str
    candidates: list[CandidateSummary]
    selected_index: int
    selected: AgentRunResult  # 复用现有结构，前端渲染逻辑零改动
