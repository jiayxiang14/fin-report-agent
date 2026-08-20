"""数字可追溯性检查：从简报正文的`<evidence>`/`<flags>`区块抽取数值型断言，
跟本次运行期间收集到的全部工具原始JSON输出做比对，判断简报里写的数字是不是
真的能在数据源里找到依据。

原本这套逻辑只在 `reward.py` 的 Best-of-N 打分路径里用——但"简报数字是否可
追溯"这件事本身是纯确定性计算，不涉及打分权重/裁判这类业务策略，所以抽成
这个中立模块：`reward.py`（Best-of-N打分）和 `loop.py`（普通单次分析的事后
校验信号）都从这里调用同一套抽取/比对逻辑，不重复实现，也不需要 `loop.py`
反过来依赖 `reward.py`（`loop.py` 本来就应该是不掺业务打分策略的通用原语，
`_has_self_verification` 也是同样的定位）。

2026-08-15 补充：原来的比对是"这个数字有没有出现在全部工具输出的任意角落"，
不检查数字有没有被**正确归属**——简报把营收的真实值错标成净利润，只要这个
数字确实存在于数据源某处（哪怕是别的字段），就会被误判成"可追溯"。现在对
`get_financials`返回的结构化财务指标做了归属感知的核对：简报文字里明确点名
了某个指标（"营收"/"净利润"这类关键词）的数字，会去核对*那个指标自己*的
真实值（`latest_annual`/`latest_quarterly`的`val`和`yoy_change_pct`），而不是
退回去跟整包JSON模糊匹配。只有当这次运行没有拿到`get_financials`的结构化
输出时（没调用这个工具、或者输出形状不是`FinancialsResponse`），才退回旧的
"全量数字袋"匹配方式——这种情况下没有权威的分字段来源可比对，退回旧逻辑
好过直接判定"无法核实"。

归属感知匹配上线后真实跑出来的问题：中文投研简报的绝对金额几乎总是用"亿"/
"万"表达（比如"营收479.41亿"），但`get_financials`存的是原始数值
（47941000000）——不做单位换算，几乎所有绝对金额类断言都判不出来，而之前
"整包JSON模糊匹配"经常能在一堆无关数字里巧合碰上一个近似值，掩盖了这个本来
就存在的缺口。现在数字抽取阶段会识别紧跟在数字后面的"万亿"/"亿"/"万"后缀
并做相应换算，覆盖面上是这次修复里影响最大的一处。

同一轮复盘还修了两个更小的归属bug：①关键词后面紧跟"率"（"营业利润率"/
"净利润率"这类get_financials完全没有对应字段的派生比率）不该被当成对应
金额字段的引用，加了否定先行断言排除；②"净利"/"毛利"这类短别名之前完全
不收，导致"账面净利$131.1亿"这类写法匹配不到任何关键词，反而退回去误匹配
到句子里其他离得更近的无关关键词——现在短别名配合①的否定先行断言一起加
回来，两者互不冲突。

**已知的、这次没有修的剩余缺口**（真实跑KO验证时观察到的，判断为"数据层/
架构层的固有限制"而不是这个打分函数能修的bug，符合项目"不做复杂语义清洗"
的边界）：
- **多期引用**：`get_financials`只暴露最新一期年度+最新一期季度，简报如果
  同时引用了更早的季度（比如"Q1"+"Q2"两个季度）或半年/多年汇总数据（这些
  数据来源可能是财报原文或Alpha Vantage而不是`get_financials`），只有最新
  一期能核对上，其余会被判定"无法追溯"——不是编造，是数据源本身只覆盖一期
- **模型自己算出来的衍生数字**：比如"上半年合计=Q1+Q2"这类简报自己做加法
  算出来的数，数据源里不会有这个字面数字，任何字面数字匹配的方案都验证不了
  这类衍生计算

2026-08-19 修复：**跨公司数字混淆**——同业对比段落提到其他公司时（比如
"同期COST营收同比+8.2%"），原来的关键词窗口没有公司实体边界的概念，"营收"
这个关键词会不分青红皂白地把紧跟着的数字归属到*当前分析对象*名下，拿去核对
当前公司自己的`get_financials`数据，跟真正该核对的对象（COST自己的营收）
完全不是一回事。修法不是做真正的命名实体识别（那超出这个模块的定位），而是
复用`get_peer_comparison`已经带回来的同业ticker名单：数字前面`_SCOPE_WINDOW`
字符内如果出现过某个同业公司的ticker（比如"COST"），这个数字就改成核对
*那家同行自己*的`revenue`/`revenue_yoy_pct`/`net_income`/`net_income_yoy_pct`
（`PeerFinancialSnapshot`只暴露这四个字段，也是`_METRIC_KEYWORDS`里唯一有
同行数据可比对的两个指标——同业对比段落提到"COST毛利率"这类字段时，因为
peer快照里根本没有这个字段，会被如实判定为"无法追溯"，而不是继续拿去跟
当前公司的毛利数据混着比）。

复查这个修复时发现了它自己引入的一个真实回归：单纯按`_SCOPE_WINDOW`字符
距离判断"数字前面出现过同行ticker"，会跨句子边界误伤——"COST最近扩张迅猛。
营收479.41亿美元"这种写法里，"COST"和后一句"营收479.41亿"只隔13个字符，
落在30字符窗口内，但中间隔着一个句号，后一句明明说的是当前分析对象自己的
营收，却会被前一句提到的COST"顺手"认领走，导致这个本该核对当前公司自己
数据、原本能匹配上的数字被误判成"无法追溯"——比修之前还倒退了一步。加了
`_clause_start`：同行ticker归属检查除了看字符距离，还要求跟数字之间没有
跨过句子边界（中英文句号/感叹号/问号/换行），逗号/顿号不算边界（同一家
公司在同一分句里连续报几个指标的场景不受影响）。

**这次修复没有覆盖、依然是已知边界**：只识别精确的ticker符号提及（比如
"COST"），不识别公司的中文/英文全称或俗称（比如"好市多"或"Costco"），因为
后者需要真正的实体名称归一化，跟前面拒绝做的"命名实体识别"是同一类工程量；
另外没有调用过`get_peer_comparison`的运行（Agent没查同行对比）里，这个
机制天然不生效，退回旧行为。
"""

from __future__ import annotations

import json
import re

# 简报文字大概率是四舍五入过的，容差比 verify_number.py 的1%(核对同一个数字
# 来源)稍宽一点
TRACEABILITY_TOLERANCE_PCT = 2.0

# 数字前面这么多字符内出现过指标关键词，才算这个数字"归属"这个指标；再远的
# 关键词大概率是在说别的事情，不该被强行关联
_SCOPE_WINDOW = 30

_TAG_PATTERNS = {tag: re.compile(rf"<{tag}>([\s\S]*?)</{tag}>") for tag in ("evidence", "flags")}
_NUMBER_TOKEN_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")

# 跟 verify_number.py 的 VALID_METRICS 是同一套指标名——这里只是给每个指标名
# 配上简报文字里可能出现的中英文措辞。2026-08-15真实跑KO暴露的问题：最初
# 刻意不收"净利"/"毛利"这类短别名，以为能避开"净利率"/"毛利率"这类利润率
# 措辞的误伤——但真实碰到的碰撞其实来自完整关键词本身："营业利润"就是
# "营业利润率"的前缀，一样会被误伤，短别名不是问题的根源。真正的修法是给
# 每个关键词都加"后面紧跟'率'就不算匹配"这个否定先行断言（下面
# _METRIC_KEYWORD_PATTERN 里统一加），这样短别名可以放心加回来——"账面净利
# $131.1亿"这类用短别名的写法之前完全匹配不到任何关键词，反而会退回去匹配
# 到句子里其他离得更近的无关关键词（比如同一段里提到的"经营现金流"），这
# 才是短别名缺失真正造成的误归属
_METRIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "revenue": ("营收", "营业收入", "revenue"),
    "net_income": ("净利润", "净利", "net income"),
    "gross_profit": ("毛利润", "毛利", "gross profit"),
    "operating_income": ("营业利润", "运营利润", "operating income"),
    "eps_diluted": ("稀释每股收益", "每股收益", "eps"),
    "total_assets": ("总资产", "total assets"),
    "total_liabilities": ("总负债", "total liabilities"),
    "stockholders_equity": ("股东权益", "stockholders equity"),
    "operating_cash_flow": ("经营活动现金流", "经营现金流", "operating cash flow"),
    "capital_expenditures": ("资本支出", "capital expenditures", "capex"),
    "free_cash_flow": ("自由现金流", "free cash flow"),
}

# 关键词后面紧跟"率"（或者"润率"——"净利"这类短别名后面接的是"润率"两个字才
# 拼成"净利润率"），说明这是一个利润率/比率措辞（"营业利润率"/"净利润率"这类
# get_financials完全没有对应字段的派生比率），不该被当成对应金额字段的引用。
# 写成"润?率"而不是只判断"率"，是因为"净利润率"如果先尝试短别名"净利"匹配，
# 紧跟着的是"润率"两个字，只挡"率"一个字挡不住，会漏判
_METRIC_KEYWORD_PATTERN = re.compile(
    "|".join(
        rf"(?P<{metric}>{'|'.join(re.escape(kw) for kw in keywords)})(?!润?率)"
        for metric, keywords in _METRIC_KEYWORDS.items()
    )
)


def _extract_tag(text: str, tag: str) -> str | None:
    match = _TAG_PATTERNS[tag].search(text)
    return match.group(1).strip() if match else None


def _looks_like_year(raw_token: str, value: float) -> bool:
    return raw_token.isdigit() and len(raw_token) == 4 and 1900 <= value <= 2100


def _looks_like_trivial_integer(raw_token: str) -> bool:
    # 一位数的纯整数常见于列表序号/轮次编号，不是真正需要溯源的数据主张
    return raw_token.isdigit() and len(raw_token) == 1


# 中文投研报告的绝对金额几乎总是用"亿"/"万"表达（比如"营收479.41亿"），但
# get_financials这类结构化数据源存的是原始数值（47941000000）——不做这层
# 换算，几乎所有绝对金额类断言都会被判定成"无法追溯"，这是归属感知匹配上线
# 后真实跑出来的缺口，覆盖面比预想的大得多，不是可以放着不管的边缘情况。
# "万亿"必须排在"亿"/"万"前面，不然会被短的那个先匹配掉，永远匹配不到"万亿"
_MAGNITUDE_SUFFIXES: tuple[tuple[str, float], ...] = (("万亿", 1e12), ("亿", 1e8), ("万", 1e4))


def _magnitude_multiplier(text: str, end_pos: int) -> float:
    for suffix, multiplier in _MAGNITUDE_SUFFIXES:
        if text[end_pos : end_pos + len(suffix)] == suffix:
            return multiplier
    return 1.0


def _extract_claimed_numbers_with_positions(text: str) -> list[tuple[int, float]]:
    results = []
    for match in _NUMBER_TOKEN_PATTERN.finditer(text):
        raw_token = match.group()
        cleaned = raw_token.replace(",", "")
        if cleaned in ("", "-", "."):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        multiplier = _magnitude_multiplier(text, match.end())
        if multiplier == 1.0:
            # "像不像年份/序号"这类启发式只对没有量级单位跟着的裸数字有意义——
            # "5亿"这种带单位的数字不可能是列表序号，不该被误伤排除掉
            if _looks_like_year(cleaned, value) or _looks_like_trivial_integer(cleaned):
                continue
        results.append((match.start(), value * multiplier))
    return results


def _extract_claimed_numbers(text: str) -> list[float]:
    return [value for _, value in _extract_claimed_numbers_with_positions(text)]


_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[。！？.!?\n]")


def _clause_start(text: str, pos: int) -> int:
    """找`pos`之前最近一个句子边界（中英文句号/感叹号/问号/换行）之后的位置——
    指标关键词归属和同行ticker归属检查都要用这个而不是单纯的字符距离：
    `_SCOPE_WINDOW`（30字符）经常会跨句子边界。两个真实例子：
    ①"COST最近扩张迅猛。营收479.41亿美元"里"COST"离"营收479.41亿"只有13个
    字符，落在窗口内，但中间隔着一个句号，说明后面这句话说的是当前分析对象
    自己的营收，不该被前一句提到的同行"顺手"认领走；②"管理层对营收前景表示
    乐观。上季度股价上涨了23.5%"里"营收"离"23.5%"也在窗口内，但股价涨幅
    跟营收毫无关系，不该被前一句的"营收"关键词强行认领。
    逗号、顿号不算边界——"COST营收479亿，净利润131亿"里逗号两边说的还是
    同一家公司，不该被切断。"""
    boundary_end = 0
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text, 0, pos):
        boundary_end = match.end()
    return boundary_end


def _find_peer_ticker_positions(text: str, peer_tickers: set[str]) -> list[tuple[int, str]]:
    """在文本里找同行ticker符号的出现位置（比如"COST"）。只匹配精确的ticker
    符号，不匹配公司中英文全称/俗称——那需要真正的实体名称归一化，不是这个
    模块的定位。左右两侧用否定环视卡住字母边界，避免"COST"被"COSTLY"这种
    英文单词误伤（真实场景概率很低，但ticker本身通常很短，防一下更安全）。
    """
    if not peer_tickers:
        return []
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in sorted(peer_tickers, key=len, reverse=True)) + r")(?![A-Za-z])"
    )
    return [(m.start(), m.group(1)) for m in pattern.finditer(text)]


def _scope_claims(
    text: str, peer_tickers: set[str] | None = None
) -> tuple[list[tuple[str, float, str | None]], list[float]]:
    """把抽出来的数字分成两类：`scoped`（数字前面`_SCOPE_WINDOW`字符内出现过
    某个指标关键词，元组是(指标名, 数值, 同行ticker或None)）和`unscoped`（没有
    明确指标归属，走旧的全量匹配）。同一个数字前面可能有好几个关键词，取离它
    最近的那个。

    第三个元素（同行ticker）处理跨公司数字混淆：如果数字前面`_SCOPE_WINDOW`
    字符内*也*出现过某个同行公司的ticker（比如"同期COST营收+8.2%"里"COST"在
    "营收"之前），说明这个数字更可能是在说"那家同行的X指标"，不是当前分析
    对象自己的——记下是哪个ticker，调用方据此改去核对那家同行自己的快照数据，
    而不是当前公司的`get_financials`。
    """
    numbers = _extract_claimed_numbers_with_positions(text)
    keywords = [(m.start(), m.lastgroup) for m in _METRIC_KEYWORD_PATTERN.finditer(text) if m.lastgroup]
    peer_mentions = _find_peer_ticker_positions(text, peer_tickers or set())

    scoped: list[tuple[str, float, str | None]] = []
    unscoped: list[float] = []
    for pos, value in numbers:
        clause_start = _clause_start(text, pos)

        best_metric: str | None = None
        best_distance: int | None = None
        for kw_pos, metric in keywords:
            if kw_pos > pos or kw_pos < clause_start:
                continue
            distance = pos - kw_pos
            if distance <= _SCOPE_WINDOW and (best_distance is None or distance < best_distance):
                best_metric, best_distance = metric, distance

        if best_metric is None:
            unscoped.append(value)
            continue

        best_peer: str | None = None
        best_peer_distance: int | None = None
        for peer_pos, ticker in peer_mentions:
            if peer_pos > pos or peer_pos < clause_start:
                continue
            distance = pos - peer_pos
            if distance <= _SCOPE_WINDOW and (best_peer_distance is None or distance < best_peer_distance):
                best_peer, best_peer_distance = ticker, distance

        scoped.append((best_metric, value, best_peer))
    return scoped, unscoped


def _walk_json_numbers(node: object, acc: set[float]) -> None:
    """递归收集"已知数字"。除了JSON里本来就是数值类型的叶子值，字符串叶子值
    也要用同一套正则去抽取——`get_filing_text`返回的财报原文是整段塞在`text`
    字段里的字符串，简报引用的数字如果来源是财报原文叙述而不是`get_financials`
    这类结构化字段，不会被收进"已知数字"集合。
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


def _find_financials_metrics(raw_tool_outputs: list[str]) -> dict | None:
    """在原始工具输出里找出结构上像`FinancialsResponse`的那一份（`get_financials`
    的返回值），取它的`metrics`字段用于按指标精确核对。不靠tool_name标记——
    `loop.py`收集`raw_tool_outputs`时是一个不区分来源的扁平列表——靠JSON结构
    本身识别：`metrics`是一个dict，且至少一项同时带`latest_annual`/
    `latest_quarterly`这两个`MetricPoint`特征字段。
    """
    for raw in raw_tool_outputs:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if any(
            isinstance(entry, dict) and ("latest_annual" in entry or "latest_quarterly" in entry)
            for entry in metrics.values()
        ):
            return metrics
    return None


def _find_peer_snapshots(raw_tool_outputs: list[str]) -> dict[str, dict]:
    """在原始工具输出里找出结构上像`PeerComparisonResponse`的那一份
    （`get_peer_comparison`的返回值），按ticker建索引返回每个同行公司自己的
    快照（`revenue`/`revenue_yoy_pct`/`net_income`/`net_income_yoy_pct`）。
    结构指纹：`peers`是一个list，每一项都是带`ticker`+`entity_name`的dict——
    跟`_find_financials_metrics`识别`FinancialsResponse`是同一个思路，不靠
    tool_name标记，靠JSON形状本身识别。"""
    for raw in raw_tool_outputs:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        peers = payload.get("peers")
        if not isinstance(peers, list) or not peers:
            continue
        if not all(isinstance(p, dict) and "ticker" in p and "entity_name" in p for p in peers):
            continue
        return {p["ticker"]: p for p in peers if isinstance(p.get("ticker"), str)}
    return {}


_PEER_METRIC_FIELDS: dict[str, tuple[str, str]] = {
    "revenue": ("revenue", "revenue_yoy_pct"),
    "net_income": ("net_income", "net_income_yoy_pct"),
}


def _peer_candidate_values(snapshot: dict, metric: str) -> set[float]:
    """`PeerFinancialSnapshot`只暴露营收/净利润两个指标，其他`_METRIC_KEYWORDS`
    里的指标（比如毛利、总资产）在同行快照里没有对应字段——这种情况返回空集合，
    让调用方如实判定"无法追溯"，而不是退回去跟当前公司自己的数据混着比。"""
    fields = _PEER_METRIC_FIELDS.get(metric)
    if fields is None:
        return set()
    candidates: set[float] = set()
    for field in fields:
        value = snapshot.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            candidates.add(float(value))
    return candidates


def _financials_candidate_values(metrics: dict, metric: str) -> set[float]:
    entry = metrics.get(metric)
    if not isinstance(entry, dict):
        return set()
    candidates: set[float] = set()
    for period_key in ("latest_annual", "latest_quarterly"):
        point = entry.get(period_key)
        if not isinstance(point, dict):
            continue
        for field in ("val", "yoy_change_pct"):
            value = point.get(field)
            if isinstance(value, int | float) and not isinstance(value, bool):
                candidates.add(float(value))
    return candidates


def _has_close_match(value: float, known: set[float]) -> bool:
    for candidate in known:
        if candidate == 0:
            if value == 0:
                return True
            continue
        if abs(value - candidate) / abs(candidate) * 100 <= TRACEABILITY_TOLERANCE_PCT:
            return True
    return False


def _evaluate_claims(
    report_text: str, raw_tool_outputs: list[str]
) -> tuple[int, int, list[str]]:
    """`score_traceability`/`find_unverifiable_claims`共享的核心逻辑，避免
    两个公开函数各自重复一遍抽取/归属/匹配的过程。返回
    (matched, total, unverifiable_descriptions)：最后一项是没能找到依据的
    断言的可读描述（数字前面带上归属的指标名，方便直接塞进给模型看的nudge
    文案里，不用模型自己再去猜"是哪个数字"）。
    """
    evidence = _extract_tag(report_text, "evidence") or ""
    flags = _extract_tag(report_text, "flags") or ""
    combined = f"{evidence}\n{flags}"

    peer_snapshots = _find_peer_snapshots(raw_tool_outputs)
    scoped, unscoped = _scope_claims(combined, set(peer_snapshots.keys()))
    total = len(scoped) + len(unscoped)
    if total == 0:
        return 0, 0, []

    financials_metrics = _find_financials_metrics(raw_tool_outputs)
    known = _collect_known_numbers(raw_tool_outputs)

    matched = 0
    unverifiable: list[str] = []
    for value in unscoped:
        if _has_close_match(value, known):
            matched += 1
        else:
            unverifiable.append(str(value))
    for metric, value, peer_ticker in scoped:
        if peer_ticker is not None and peer_ticker in peer_snapshots:
            # 数字前面出现过同行ticker（比如"COST营收+8.2%"）：这是在说那家
            # 同行自己的指标，不是当前分析对象的——核对*那家同行*的快照数据，
            # 不能拿去跟当前公司的get_financials混着比，哪怕当前公司自己也
            # 恰好有个数字长得很像（那只是巧合，不是真的来源）
            ok = _has_close_match(value, _peer_candidate_values(peer_snapshots[peer_ticker], metric))
        elif financials_metrics is not None:
            # 有权威的分字段来源可查：必须对得上*这个指标自己*的真实值，
            # 不再退回去跟整包JSON模糊匹配——这是修的核心问题，简报把A指标
            # 的数字错标成B指标，即使这个数字客观上存在于数据源某处（比如
            # 正好是A指标自己的值），也不该被判定成"B指标可追溯"
            ok = _has_close_match(value, _financials_candidate_values(financials_metrics, metric))
        else:
            # 这次运行没拿到 get_financials 的结构化输出（没调用/输出形状
            # 不对），没有权威的分字段来源可比对，退回旧的全量匹配，好过
            # 直接判定"无法核实"
            ok = _has_close_match(value, known)
        if ok:
            matched += 1
        else:
            label = f"{metric}({peer_ticker})" if peer_ticker else metric
            unverifiable.append(f"{label}: {value}")
    return matched, total, unverifiable


def score_traceability(report_text: str, raw_tool_outputs: list[str]) -> tuple[int, int]:
    """返回 (matched, total)：`total`是从`<evidence>`/`<flags>`抽出的数值型
    断言个数，`matched`是其中能在`raw_tool_outputs`里找到依据的个数。
    `total == 0`表示没有可核对的数字主张，调用方应视为"无法判断"而不是"有问题"。
    """
    matched, total, _ = _evaluate_claims(report_text, raw_tool_outputs)
    return matched, total


def find_unverifiable_claims(report_text: str, raw_tool_outputs: list[str]) -> list[str]:
    """返回没能在`raw_tool_outputs`里找到依据的数字断言的可读描述列表——供
    `loop.py`的可追溯率gate在nudge文案里指出具体是哪些数字有问题，而不是
    笼统地说"有些数字有问题"，跟`verify_number`的核查结果一样是可以指名道姓
    的客观信号。"""
    _, _, unverifiable = _evaluate_claims(report_text, raw_tool_outputs)
    return unverifiable
