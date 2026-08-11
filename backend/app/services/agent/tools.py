"""Agent 可调用的8个工具：schema定义 + 执行分发器。

Schema 用 Anthropic 原生的 tool_use 格式（name/description/input_schema），
DeepSeek 的 Anthropic 兼容端点用的是同一套格式，不需要为两家供应商各写一份。

execute_tool 是唯一的失败边界：任何工具内部异常都在这里兜住，转成
(错误文本, is_error=True) 返回给 Agent，而不是让整个 Loop 崩溃——
这是项目书要求的"失败与降级处理"。
"""

from app.models.earnings_surprise import EarningsSurpriseResponse
from app.models.filing import FilingTextResponse
from app.models.financials import FinancialsResponse
from app.models.peer import PeerComparisonResponse
from app.models.price_reaction import PriceReactionResponse
from app.models.sector import SectorRotationResponse
from app.models.thematic_flow import ThematicFlowResponse
from app.models.verify import VerifyNumberResponse
from app.services.earnings_surprise import EarningsSurpriseError, get_earnings_surprise
from app.services.filing_text import FilingTextError, get_filing_text
from app.services.peer_comparison import get_peer_comparison
from app.services.price_reaction import PriceReactionError, get_price_reaction
from app.services.sec_client import SecClientError
from app.services.sec_edgar import get_financials
from app.services.sector_rotation import SectorDataError, get_sector_rotation
from app.services.thematic_flow import ThematicFlowError, get_thematic_flow
from app.services.verify_number import VALID_METRICS, VALID_PERIODS, verify_number

_ToolResult = (
    FinancialsResponse
    | SectorRotationResponse
    | PeerComparisonResponse
    | FilingTextResponse
    | PriceReactionResponse
    | ThematicFlowResponse
    | EarningsSurpriseResponse
    | VerifyNumberResponse
)

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_financials",
        "description": (
            "拉取公司最新一期年度和季度的核心财务指标（营收、净利润、毛利润、营业利润、"
            "稀释EPS、总资产、总负债、股东权益、经营活动现金流、资本支出、自由现金流）"
            "及同比变化，均为代码计算的确定性数字。自由现金流=经营活动现金流-资本支出，"
            "两个源数据都存在时才会有这个字段。分析任何公司几乎都应该第一步调用这个工具，"
            "留意净利润是正的但经营活动现金流转负这类背离——这是账面盈利和实际现金"
            "创造能力脱节的信号，比单看净利润更值得在简报里指出。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "股票代码，如 AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_sector_position",
        "description": (
            "拉取该公司所属板块相对大盘的强弱位置（RRG象限：leading/weakening/lagging/improving），"
            "帮助判断公司所在行业整体是顺风还是逆风。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "股票代码，如 AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_peer_comparison",
        "description": (
            "拉取同板块其他几家公司的营收、净利润数据，用于判断这家公司的表现是行业普遍现象"
            "还是自身特有。只覆盖预设股票池里的公司，查不到时会明确说明。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "股票代码，如 AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_filing_text",
        "description": (
            "拉取最新一期10-K（年报）或10-Q（季报）的完整原文文本，包含管理层讨论（MD&A）、"
            "风险披露等章节。这是唯一能看到财务数字背后原因、管理层措辞语气的途径。"
            "文本较长（几万到几十万字符），需要自己在阅读时判断哪部分相关。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "股票代码，如 AAPL"},
                "form": {
                    "type": "string",
                    "enum": ["10-K", "10-Q"],
                    "description": "要拉取的文件类型，默认10-K",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_reaction",
        "description": (
            "拉取该公司最近一次业绩快报（8-K，Item 2.02）发布前后的股价涨跌幅和成交量"
            "放大倍数（相对最近20个交易日的平均成交量），以及反应之后几个交易日内这段"
            "涨跌被回吐的程度（reversal_pct_of_initial_move，可能超过100%表示已反转）"
            "和回吐当天的放量情况。用于判断财报内容和市场实际反应是否一致、以及这个"
            "反应有没有持续——比如财报数字平淡但股价大涨，或者初始反应后续被大幅回吐。"
            "反应窗口锚定的是8-K业绩快报的申报时间（不是10-K/10-Q的申报日期，后者通常"
            "滞后财报发布好几天到几周）。**这几个字段只是价格/成交量的客观事实，不要"
            "把它们解读成\"洗盘\"\"诱多\"\"诱空\"\"主力出货\"这类需要判断市场参与者意图"
            "的结论——这些无法从价格数据本身验证，只能描述数据本身呈现的模式。**"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "股票代码，如 AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_earnings_surprise",
        "description": (
            "拉取该公司最新一期的实际EPS vs 分析师一致预期EPS对比（数据来自Alpha "
            "Vantage），返回verdict字段（超预期/低于预期/符合预期，代码按数值直接"
            "计算，不是猜测）。这经常是get_price_reaction里价格反应的直接原因——"
            "数字超预期通常伴随上涨、低于预期通常伴随下跌，两个工具放一起看能判断"
            "市场反应是否合理。如果Alpha Vantage没有配置或没有这只票的预期数据，"
            "has_data会是false，如实告知没有这项数据即可，不要编造预期值。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "股票代码，如 AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_thematic_flow",
        "description": (
            "拉取AI算力/基建产业链上10个细分主题（半导体、AI芯片/GPU、存储、光模块、"
            "云计算、数据中心REITs、网络设备、服务器ODM、电力、液冷散热）当前相对大盘的"
            "强弱轮动位置，每个主题都带一个chain_position字段（上游/中游/下游，硬件组件"
            "→基础设施建设运营→云端部署这个通用产业链位置分类，固定的、跟具体公司无关，"
            "不是猜测哪些公司是谁的供应商/客户）。传入ticker后，返回结果里的matched_themes"
            "是代码确定性算出来的，不是你自己判断的——来自两层匹配：(1)这家公司本身是不是"
            "某个主题篮子的成分股，(2)这家公司官方SIC行业分类是否命中半导体/存储/网络设备/"
            "服务器ODM/电力这5个SIC分类足够可靠的主题（AI芯片/GPU、光模块、云计算、数据中心"
            "REITs、液冷散热这5个主题SIC分类本身不可靠，不做这层匹配）。如果matched_themes"
            "非空，直接采用；如果是空的，说明这两层都没匹配上，你需要结合已经了解到的公司"
            "业务自己判断这10个主题里有没有相关的。只有当这家公司业务确实处于AI算力产业链"
            "某个环节时才需要调用这个工具——分析一家跟AI基建完全无关的公司（比如零售、消费品）"
            "时不需要调用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "股票代码，如 AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "verify_number",
        "description": (
            "对准备写进简报的某个具体数字发起二次核实，重新从数据源核对一遍。"
            "生成简报初稿之后，必须对至少一个关键数字调用这个工具做自我核查。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "股票代码，如 AAPL"},
                "metric": {
                    "type": "string",
                    "enum": list(VALID_METRICS),
                    "description": "要核实的指标",
                },
                "claimed_value": {"type": "number", "description": "简报里准备写的数值"},
                "period": {
                    "type": "string",
                    "enum": list(VALID_PERIODS),
                    "description": "对应的报告期间",
                },
            },
            "required": ["ticker", "metric", "claimed_value", "period"],
        },
    },
]


async def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """返回 (内容文本, is_error)。任何异常都在这里兜住，不往外抛。"""
    try:
        result: _ToolResult
        if name == "get_financials":
            result = await get_financials(tool_input["ticker"])
        elif name == "get_sector_position":
            result = await get_sector_rotation(tool_input["ticker"])
        elif name == "get_peer_comparison":
            result = await get_peer_comparison(tool_input["ticker"])
        elif name == "get_filing_text":
            result = await get_filing_text(tool_input["ticker"], tool_input.get("form", "10-K"))
        elif name == "get_price_reaction":
            result = await get_price_reaction(tool_input["ticker"])
        elif name == "get_earnings_surprise":
            result = await get_earnings_surprise(tool_input["ticker"])
        elif name == "get_thematic_flow":
            result = await get_thematic_flow(tool_input["ticker"])
        elif name == "verify_number":
            result = await verify_number(
                tool_input["ticker"],
                tool_input["metric"],
                tool_input["claimed_value"],
                tool_input.get("period", "annual"),
            )
        else:
            return f"未知工具：{name}", True
        return result.model_dump_json(), False
    except (
        SecClientError,
        SectorDataError,
        FilingTextError,
        PriceReactionError,
        ThematicFlowError,
        EarningsSurpriseError,
    ) as exc:
        return f"工具调用失败：{exc}", True
    except Exception as exc:  # noqa: BLE001 - 这是Agent Loop的失败边界，任何异常都要兜住
        return f"工具调用出现未预期的错误：{exc}", True
