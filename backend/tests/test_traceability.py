"""数字可追溯性抽取/比对逻辑本身的单测（从 reward.py 抽出来的中立模块）。
更完整的"跟规则打分权重组合"的行为在 test_reward.py 里测，这里只测
`score_traceability` 这个公开函数本身的抽取/匹配语义。
"""

import json

from app.services.agent import traceability as tb
from app.services.agent.traceability import find_unverifiable_claims, score_traceability


def test_no_numeric_claims_returns_zero_total():
    report = "<conclusion>强劲</conclusion><evidence>管理层对前景表示乐观</evidence><flags></flags>"

    matched, total = score_traceability(report, [])

    assert (matched, total) == (0, 0)


def test_matched_number_found_in_raw_tool_output():
    report = "<conclusion>强劲</conclusion><evidence>营收达到950000000美元</evidence><flags></flags>"
    raw_outputs = [json.dumps({"revenue": 950000000})]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_fabricated_number_not_found_in_raw_tool_output():
    report = "<conclusion>强劲</conclusion><evidence>净利润编造成了999亿美元</evidence><flags></flags>"
    raw_outputs = [json.dumps({"revenue": 950000000})]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (0, 1)


def test_number_embedded_in_string_field_is_still_matched():
    report = "<conclusion>强劲</conclusion><evidence>存货同比增长23.5%</evidence><flags></flags>"
    raw_outputs = [json.dumps({"text": "Inventory increased 23.5% year over year."})]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def _financials_payload(revenue_val: float, net_income_val: float) -> str:
    """构造一份跟`FinancialsResponse`真实结构一致的payload（`metrics`字典，
    每项带`latest_annual`/`latest_quarterly`），归属感知匹配靠这个结构特征
    识别"这是get_financials的输出"。"""
    return json.dumps(
        {
            "ticker": "AAPL",
            "cik": "0000320193",
            "entity_name": "Apple Inc.",
            "metrics": {
                "revenue": {
                    "tag": "Revenues",
                    "label": "Revenue",
                    "unit": "USD",
                    "latest_annual": {
                        "end": "2024-09-28",
                        "val": revenue_val,
                        "fy": 2024,
                        "fp": "FY",
                        "form": "10-K",
                        "filed": "2024-11-01",
                        "yoy_change_pct": 12.5,
                    },
                    "latest_quarterly": None,
                },
                "net_income": {
                    "tag": "NetIncomeLoss",
                    "label": "Net Income",
                    "unit": "USD",
                    "latest_annual": {
                        "end": "2024-09-28",
                        "val": net_income_val,
                        "fy": 2024,
                        "fp": "FY",
                        "form": "10-K",
                        "filed": "2024-11-01",
                        "yoy_change_pct": 5.0,
                    },
                    "latest_quarterly": None,
                },
            },
            "source": "SEC EDGAR companyfacts",
            "retrieved_at": "2024-11-01T00:00:00Z",
        }
    )


def test_scoped_claim_matches_its_own_metric_field():
    report = "<conclusion>强劲</conclusion><evidence>营收达到950000000美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=950000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_scoped_claim_matches_yoy_change_pct():
    report = "<conclusion>强劲</conclusion><evidence>营收同比增长12.5%</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=950000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_scoped_claim_rejects_number_misattributed_to_wrong_metric():
    """核心回归测试：简报把营收的真实值(950000000)错标成净利润，这个数字
    客观上确实存在于数据源里（作为revenue字段的值），旧的"全量数字袋"匹配
    会误判成可追溯；归属感知匹配应该去核对net_income自己的真实值(100000000)，
    对不上就该判定成不匹配。"""
    report = "<conclusion>强劲</conclusion><evidence>净利润达到950000000美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=950000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (0, 1)


def test_yi_suffix_is_converted_before_matching_specific_field():
    """2026-08-15真实跑KO暴露的问题：中文简报的绝对金额几乎总是用"亿"表达
    （比如"营收479.41亿"），但get_financials存的是原始数值——不换算的话，
    归属感知匹配会把几乎所有绝对金额类断言都判成"无法追溯"，这不是边缘情况，
    是最常见的写法。"""
    report = "<conclusion>强劲</conclusion><evidence>营收达到479.41亿美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_wan_suffix_is_converted():
    report = "<conclusion>强劲</conclusion><evidence>营收达到4794100万美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_wanyi_suffix_is_converted():
    report = "<conclusion>强劲</conclusion><evidence>营收达到0.047941万亿美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_single_digit_with_yi_suffix_is_not_excluded_as_trivial():
    """裸的一位数会被当成列表序号排除掉，但"5亿"这种带单位的不可能是序号，
    不该被误伤——这是加换算逻辑时顺手修的一个衔接问题。"""
    report = "<conclusion>强劲</conclusion><evidence>净利润达到5亿美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=500000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_misattributed_yi_amount_still_rejected_after_unit_conversion():
    """确认单位换算不会把归属检查又变回原来的模糊匹配——营收的真实值换算成
    "亿"后错标成净利润，换算逻辑本身没问题，但净利润自己的真实值对不上，
    依然应该判定不匹配。"""
    report = "<conclusion>强劲</conclusion><evidence>净利润达到479.41亿美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (0, 1)


def test_margin_percentage_is_not_scoped_to_the_absolute_metric():
    """2026-08-15真实跑KO暴露的问题：'营业利润率34.9%'里的34.9是利润率
    （get_financials完全没有对应字段的派生比率），不是营业利润的真实值或
    同比增速，之前会被误归属到operating_income然后判定不匹配——现在应该
    完全不归属给这个指标（不代表"验证通过"，只是不该被扣在operating_income
    头上）。"""
    report = "<conclusion>强劲</conclusion><evidence>营业利润率34.9%</evidence><flags></flags>"

    scoped, unscoped = tb._scope_claims(
        (tb._extract_tag(report, "evidence") or "") + "\n" + (tb._extract_tag(report, "flags") or "")
    )

    assert scoped == []
    assert unscoped == [34.9]


def test_short_alias_net_income_is_recognized():
    """'账面净利'这类用短别名的写法，之前完全匹配不到net_income的关键词，
    会让数字退回去误匹配到句子里其他离得更近的无关关键词。"""
    report = "<conclusion>强劲</conclusion><evidence>账面净利$1亿 vs 经营现金流$2亿</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 2)  # 净利$1亿对得上；经营现金流$2亿这个候选payload里没有，对不上


def test_short_alias_still_excludes_margin_ratio_wording():
    """加回'净利'/'毛利'短别名之后，'净利润率'这种完整的利润率措辞依然不该
    被短别名''净利''意外捞回来匹配——这是负向先行断言要同时挡住'率'和
    '润率'两种情况的原因。"""
    scoped, unscoped = tb._scope_claims("净利润率28.7%")

    assert scoped == []
    assert unscoped == [28.7]


def test_scoped_claim_falls_back_to_flat_matching_without_financials_payload():
    """这次运行没有拿到get_financials的结构化输出（比如没调用这个工具），
    没有权威的分字段来源可比对，应该退回旧的全量匹配，而不是直接判定"无法
    核实"。"""
    report = "<conclusion>强劲</conclusion><evidence>营收达到950000000美元</evidence><flags></flags>"
    raw_outputs = [json.dumps({"revenue": 950000000})]  # 不是FinancialsResponse的形状

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (1, 1)


def test_four_digit_year_without_magnitude_suffix_is_excluded():
    """裸的四位数年份（1900-2100区间）不该被当成数值型断言去核对——之前只在
    "亿/万后缀存在时跳过这层过滤"这个衔接点上测过，没测过这个过滤本身在
    正常场景下确实生效。"""
    assert tb._extract_claimed_numbers("2024年营收增长") == []


def test_single_digit_without_magnitude_suffix_is_excluded_as_trivial():
    """裸的一位数（常见于列表序号/排名）不该被当成数值型断言——同样是之前只
    测过"有单位时不该被误伤"，没测过没单位时这层过滤本身确实生效。"""
    assert tb._extract_claimed_numbers("行业排名第3") == []


def test_tolerance_boundary_accepts_exactly_two_percent_relative_diff():
    """TRACEABILITY_TOLERANCE_PCT=2.0，边界值本身应该判定为匹配（<=不是<）。"""
    assert tb._has_close_match(102.0, {100.0}) is True


def test_tolerance_boundary_rejects_just_over_two_percent_relative_diff():
    """刚超出2%容差应该判定不匹配——固定这个边界，防止将来改动容差判断时
    悄悄从'<='变成'<'或反过来都不会被发现。"""
    assert tb._has_close_match(102.01, {100.0}) is False


def test_zero_valued_candidate_only_matches_zero_claim():
    """`_has_close_match`对candidate==0做了特殊处理（避免除零），这条路径
    之前完全没有测试覆盖：0该匹配0，非0数字不该被"候选里恰好有个0"误判成
    匹配。"""
    assert tb._has_close_match(0.0, {0.0}) is True
    assert tb._has_close_match(5.0, {0.0}) is False


def test_known_limitation_earlier_period_claim_not_covered_by_latest_period_data():
    """已知限制（见traceability.py模块docstring"多期引用"一条）的pin测试：
    `get_financials`只暴露最新一期年度+最新一期季度，简报如果引用了更早
    一期的数字（比如"Q1营收124.72亿"，不是latest_quarterly），即使这个数字
    在财报原文里是真实的，现在的实现也判定不出来，只能算unmatched——这是
    数据源覆盖范围的固有限制，不是这个函数的bug。如果将来真的把这个限制修
    掉了（比如让get_financials额外暴露历史期），这个测试应该跟着更新断言，
    而不是被删掉——删掉就丢失了"这里曾经是已知限制"这个信息。"""
    report = "<conclusion>强劲</conclusion><evidence>Q1营收124.72亿</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (0, 1)


def test_known_limitation_model_computed_derived_sum_not_traceable():
    """已知限制（见模块docstring"模型自己算出来的衍生数字"一条）的pin测试：
    简报自己做加法算出来的"上半年营收合计"，数据源里不会有这个字面数字，
    字面数字匹配判定不出来是正确的（不代表这个数字是错的，只代表验证不了）。
    """
    report = "<conclusion>强劲</conclusion><evidence>上半年营收合计258.5亿</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (0, 1)


def _peer_comparison_payload(ticker: str, peers: list[dict]) -> str:
    """构造一份跟`PeerComparisonResponse`真实结构一致的payload——归属感知匹配
    靠`peers`是一个list且每项都带`ticker`/`entity_name`这个结构特征识别
    "这是get_peer_comparison的输出"，跟`_financials_payload`识别`metrics`
    结构是同一个思路。"""
    return json.dumps(
        {
            "ticker": ticker,
            "sector_etf": "XLP",
            "sector_name": "Consumer Staples",
            "peers": peers,
            "note": None,
        }
    )


def test_cross_company_number_is_scoped_to_the_mentioned_peer_not_the_primary_company():
    """2026-08-19修复（见模块docstring"跨公司数字混淆"一条）：同业对比句子
    提到其他公司时（"COST营收同比+8.2%"），数字前面出现过COST这个同行ticker，
    应该去核对*COST自己*在get_peer_comparison里的营收同比，而不是被就近归属
    到`revenue`这个指标名下之后错误地去核对当前分析对象（AAPL）自己的营收——
    AAPL自己的营收同比是12.5%（见_financials_payload），跟COST的8.2%完全不
    是一回事，旧逻辑会把这个数字判成"需要核对AAPL的营收但对不上"，新逻辑应该
    正确核对COST自己的数据并判定为可追溯。"""
    report = "<conclusion>强劲</conclusion><evidence>同业对比：COST营收同比+8.2%</evidence><flags></flags>"
    raw_outputs = [
        _financials_payload(revenue_val=47941000000, net_income_val=100000000),
        _peer_comparison_payload(
            "AAPL",
            [
                {
                    "ticker": "COST",
                    "entity_name": "Costco Wholesale Corp",
                    "revenue": 254453000000,
                    "revenue_yoy_pct": 8.2,
                    "net_income": 7367000000,
                    "net_income_yoy_pct": 12.1,
                }
            ],
        ),
    ]

    scoped, unscoped = tb._scope_claims(tb._extract_tag(report, "evidence") or "", peer_tickers={"COST"})
    matched, total = score_traceability(report, raw_outputs)

    assert scoped == [("revenue", 8.2, "COST")]
    assert unscoped == []
    assert (matched, total) == (1, 1)


def test_cross_company_number_that_does_not_match_the_peer_is_correctly_flagged():
    """核心回归测试：数字前面出现过同行ticker，但这个数字其实是编造/记错的
    （COST真实营收同比是8.2%，简报写成了20%）——应该被判定成不可追溯，不能
    因为"归属到了同行"就直接放行。"""
    report = "<conclusion>强劲</conclusion><evidence>同业对比：COST营收同比+20%</evidence><flags></flags>"
    raw_outputs = [
        _peer_comparison_payload(
            "AAPL",
            [
                {
                    "ticker": "COST",
                    "entity_name": "Costco Wholesale Corp",
                    "revenue": 254453000000,
                    "revenue_yoy_pct": 8.2,
                    "net_income": 7367000000,
                    "net_income_yoy_pct": 12.1,
                }
            ],
        ),
    ]

    matched, total = score_traceability(report, raw_outputs)
    unverifiable = find_unverifiable_claims(report, raw_outputs)

    assert (matched, total) == (0, 1)
    assert unverifiable == ["revenue(COST): 20.0"]


def test_peer_metric_without_a_corresponding_snapshot_field_is_unverifiable_not_misrouted():
    """PeerFinancialSnapshot只暴露营收/净利润，同业对比提到"COST毛利率"这类
    快照里没有的字段时，应该如实判定无法追溯，不能退回去拿当前公司自己的
    毛利数据顶替——那还是在核对错误的公司。"""
    report = "<conclusion>强劲</conclusion><evidence>同业对比：COST毛利40%</evidence><flags></flags>"
    raw_outputs = [
        _financials_payload(revenue_val=47941000000, net_income_val=100000000),
        _peer_comparison_payload(
            "AAPL",
            [
                {
                    "ticker": "COST",
                    "entity_name": "Costco Wholesale Corp",
                    "revenue": 254453000000,
                    "revenue_yoy_pct": 8.2,
                    "net_income": 7367000000,
                    "net_income_yoy_pct": 12.1,
                }
            ],
        ),
    ]

    matched, total = score_traceability(report, raw_outputs)

    assert (matched, total) == (0, 1)


def test_without_peer_comparison_output_falls_back_to_old_behavior():
    """没调用过get_peer_comparison（raw_tool_outputs里没有对应结构）时，
    这个机制天然不生效，退回旧行为——数字被归属到当前公司名下，对不上就是
    对不上，不会报错或崩溃。"""
    report = "<conclusion>强劲</conclusion><evidence>同业对比：COST营收同比+8.2%</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=47941000000, net_income_val=100000000)]

    scoped, unscoped = tb._scope_claims(tb._extract_tag(report, "evidence") or "")
    matched, total = score_traceability(report, raw_outputs)

    assert scoped == [("revenue", 8.2, None)]
    assert (matched, total) == (0, 1)


def test_metric_keyword_in_a_prior_sentence_does_not_claim_an_unrelated_number():
    """回归测试：跟同行ticker归属是同一个bug，但出现在更早就存在的指标关键词
    归属逻辑里——"管理层对营收前景表示乐观。上季度股价上涨了23.5%"这种写法，
    "营收"和后一句的"23.5%"字符距离在窗口内，但股价涨幅跟营收毫无关系，不该
    被前一句的"营收"关键词强行认领（认领之后会去核对当前公司的真实营收，
    对不上就误判成"数字造假"，而它压根就不是一个营收断言）。修复后这个数字
    应该落进unscoped（没有本分句内的指标归属，退回全量匹配），不是被错误
    scope到revenue。"""
    text = "管理层对营收前景表示乐观。上季度股价上涨了23.5%"

    scoped, unscoped = tb._scope_claims(text)

    assert scoped == []
    assert unscoped == [23.5]


def test_peer_ticker_in_a_prior_sentence_does_not_claim_a_number_in_the_next_sentence():
    """回归测试：复查同行归属修复时发现的一个真实bug——同行ticker归属检查
    如果只看字符距离，会跨句子边界误伤。"COST最近扩张迅猛。营收479.41亿美元"
    这种写法里"COST"和后一句的"营收479.41亿"只隔13个字符（落在30字符窗口
    内），但中间隔着一个句号，后一句说的明明是当前分析对象自己的营收——这个
    数字应该按老办法核对当前公司自己的`get_financials`数据（能匹配上），
    不该被前一句提到的COST"顺手"认领走导致误判成不可追溯。"""
    report = (
        "<conclusion>强劲</conclusion>"
        "<evidence>同业对比：COST最近扩张迅猛。营收479.41亿美元</evidence>"
        "<flags></flags>"
    )
    raw_outputs = [
        _financials_payload(revenue_val=47941000000, net_income_val=100000000),
        _peer_comparison_payload(
            "AAPL",
            [
                {
                    "ticker": "COST",
                    "entity_name": "Costco Wholesale Corp",
                    "revenue": 254453000000,
                    "revenue_yoy_pct": 8.2,
                    "net_income": 7367000000,
                    "net_income_yoy_pct": 12.1,
                }
            ],
        ),
    ]

    scoped, unscoped = tb._scope_claims(tb._extract_tag(report, "evidence") or "", peer_tickers={"COST"})
    matched, total = score_traceability(report, raw_outputs)

    assert scoped == [("revenue", 47941000000.0, None)]
    assert (matched, total) == (1, 1)


def test_peer_ticker_in_the_same_clause_across_a_comma_still_attributes_to_the_peer():
    """逗号/顿号不算句子边界——"COST营收479亿，净利润131亿"里逗号两边说的
    还是同一家公司，两个数字都应该归属到COST，不该被逗号切断。"""
    report = "<conclusion>强劲</conclusion><evidence>同业对比：COST营收479亿，净利润131亿</evidence><flags></flags>"

    scoped, unscoped = tb._scope_claims(tb._extract_tag(report, "evidence") or "", peer_tickers={"COST"})

    assert scoped == [("revenue", 47900000000.0, "COST"), ("net_income", 13100000000.0, "COST")]


def test_find_unverifiable_claims_returns_empty_list_when_everything_matches():
    report = "<conclusion>强劲</conclusion><evidence>营收达到950000000美元</evidence><flags></flags>"
    raw_outputs = [json.dumps({"revenue": 950000000})]

    assert find_unverifiable_claims(report, raw_outputs) == []


def test_find_unverifiable_claims_labels_scoped_claim_with_its_metric_name():
    """给可追溯率gate的nudge文案用的：归属过的断言要带上指标名，模型才知道
    具体是哪个数字有问题，不用自己再去猜。"""
    report = "<conclusion>强劲</conclusion><evidence>净利润达到950000000美元</evidence><flags></flags>"
    raw_outputs = [_financials_payload(revenue_val=950000000, net_income_val=100000000)]

    assert find_unverifiable_claims(report, raw_outputs) == ["net_income: 950000000.0"]


def test_find_unverifiable_claims_leaves_unscoped_claim_as_bare_number():
    report = "<conclusion>强劲</conclusion><evidence>净利润编造成了999亿美元</evidence><flags></flags>"
    raw_outputs = [json.dumps({"revenue": 950000000})]

    assert find_unverifiable_claims(report, raw_outputs) == ["net_income: 99900000000.0"]
