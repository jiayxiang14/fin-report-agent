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
    # <evidence>/<flags>标签缺失时代码层面拦下来要求重新完整输出——纯正则
    # 判断，没有语义模糊空间
    structure_gate_triggered: bool = False
    # structure_gate_triggered=True 只代表"发现问题、插过nudge"这个历史事实，
    # 不代表模型真的照做了——如果模型被拦下来之后只回一句"已修正"不重新输出
    # 带标签的简报，_resolve_final_report的回退机制会原样捞回没解决问题的旧
    # 草稿，structure_gate_triggered=True 但标签依然缺失（真实复现过这个
    # bug）。这个字段是在拿到最终 final_report 之后，用跟gate检测阶段相同的
    # 判断逻辑重新核对一遍算出来的，独立于*_triggered——即使从没触发过
    # （*_triggered=False）这里也会如实反映最终状态，两者组合读：
    # False/True=从没出过问题；True/True=出过问题、确认修好了；
    # True/False=出过问题、没能确认修好；False/False=从没检查过、但确实有问题。
    # 只代表"标签在不在"这个可机器判断的条件满足了，不代表内容质量有保证
    structure_gate_resolved: bool = True
    # 下结论前没有成功调用过get_financials这个底线要求时代码层面拦下来
    tool_coverage_gate_triggered: bool = False
    # 语义同structure_gate_resolved——用transcript重新核对一遍get_financials
    # 有没有被成功调用过，不依赖*_triggered这个历史标记
    tool_coverage_gate_resolved: bool = True
    self_verification_triggered: bool = False
    # self_verification_triggered 只代表"调用过 verify_number 且没技术性报错"，
    # 不代表核实结果对得上——verify_number 会返回真正客观、代码算出来的 matches
    # 字段，之前代码完全不读它，模型看到 matches=false 也能照样 end_turn。这个
    # 字段记录的是"模型最近一次核查真的没对上、被代码层拦下来要求处理"这件事
    # 有没有发生过，是"agent自己觉得核查过了"和"核查结果真的通过了"之间缺失的
    # 那道 gate
    verification_mismatch_triggered: bool = False
    # 语义同structure_gate_resolved——重新核对每个被核实过的(metric, period)
    # 组合各自最近一次的结果是不是还是false，不依赖*_triggered这个历史标记。
    # 按(metric, period)分组是必须的：只看整个调用序列的最后一个元素会漏判——
    # 核查营收发现不对、没去改，紧接着核查净利润恰好对上，营收那个明确有问题
    # 却没被处理的mismatch不该被净利润的成功核查"顺手"掩盖掉
    verification_mismatch_resolved: bool = True
    # verify_number那道gate只管模型自己主动选去核实的那一个数字，简报里其他
    # 数字断言完全没人管——这个字段记录的是更广的一道gate：全部数字断言的
    # 可追溯率明显偏低时，代码层面拦下来要求模型处理，而不是任由模型自己
    # 觉得"这份报告写得不错"就直接定稿
    traceability_gate_triggered: bool = False
    # 语义同structure_gate_resolved——用最终的traceable_numbers_matched/total
    # 重新对一次阈值。True不代表数字都是真的可追溯，模型也可能是删掉了验证
    # 不了的断言而不是补充依据，只是让比例看起来健康了——这是客观指标类校验
    # 的天然局限
    traceability_gate_resolved: bool = True
    # 用get_price_reaction算出来的真实价格变动去交叉检查<sentiment>标签跟市场
    # 反应方向是否明显打架（跌幅超阈值还标positive，或反过来）——不判断
    # sentiment"对不对"（那是主观判断），只确保方向明显矛盾时模型不能装没看见
    sentiment_consistency_gate_triggered: bool = False
    # 语义同structure_gate_resolved——对最终的<sentiment>标签和价格反应数据
    # 重新跑一次矛盾检测，不依赖*_triggered这个历史标记
    sentiment_consistency_gate_resolved: bool = True
    # Reflexion：只有传了 reflexion_check 参数的调用方（目前只有 best_of_n.py）
    # 才可能触发，普通单次分析的调用方不传这个参数，永远是 False
    reflexion_triggered: bool = False
    # 数字可追溯性事后校验（traceability.py）：从最终简报的<evidence>/<flags>
    # 抽出的数值型断言里，有多少能在本次运行收集到的工具原始输出里找到匹配。
    # 这是一个信号，不是拦截——total 为 0 表示没有可核对的数字主张，不代表有问题
    traceable_numbers_matched: int = 0
    traceable_numbers_total: int = 0
