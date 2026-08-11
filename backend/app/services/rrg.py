"""RRG（相对轮动图）核心数学：相对强弱(RS-Ratio)和动量(RS-Momentum)的计算，
纯函数、不涉及任何数据获取——被`sector_rotation.py`（11个SPDR大类板块）和
`thematic_flow.py`（细分主题篮子）共享，两边喂进来的都只是"一条价格序列 vs
一条基准价格序列"，计算逻辑完全一致，不应该维护两份。
"""

import pandas as pd

LONG_WINDOW = 130  # 用于 RS 的滚动均值/标准差窗口（约半年交易日）
MOMENTUM_WINDOW = 20  # RS-Ratio 变化率的回溯窗口（约1个月交易日）
MOMENTUM_NORM_WINDOW = 60  # 变化率再归一化的窗口，故意比 LONG_WINDOW 短，
                            # 避免两次滚动窗口的warm-up期叠加，吃掉太多可展示的历史
SMOOTH_WINDOW = 3  # 平滑窗口，减少日间噪音


def compute_rrg(series_closes: pd.Series, benchmark_closes: pd.Series) -> pd.DataFrame:
    """简化版 RS-Ratio / RS-Momentum：把 JdK RRG 的核心思路（相对强弱的滚动
    z-score归一化 + 变化率的滚动z-score归一化）用纯 pandas 实现，不是官方
    专利公式的精确复刻，但同样输出"以100为中枢、可分四象限"的两个序列。
    """
    aligned = pd.DataFrame({"series": series_closes, "benchmark": benchmark_closes}).dropna()

    rs = aligned["series"] / aligned["benchmark"] * 100
    rs_z = (rs - rs.rolling(LONG_WINDOW).mean()) / rs.rolling(LONG_WINDOW).std()
    rs_ratio = (100 + rs_z).rolling(SMOOTH_WINDOW).mean()

    mom_raw = rs_ratio.diff(MOMENTUM_WINDOW)
    mom_z = (
        mom_raw - mom_raw.rolling(MOMENTUM_NORM_WINDOW).mean()
    ) / mom_raw.rolling(MOMENTUM_NORM_WINDOW).std()
    rs_momentum = (100 + mom_z).rolling(SMOOTH_WINDOW).mean()

    out = pd.DataFrame({"rs_ratio": rs_ratio, "rs_momentum": rs_momentum}).dropna()
    return out


def classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "lagging"
    return "improving"
