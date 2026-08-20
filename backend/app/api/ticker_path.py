"""所有路由共用的 ticker 路径/查询参数校验。

在没有这层校验之前，`ticker`是原样拼进磁盘缓存文件名（比如
`polygon_client.py`的`f"polygon_bars_{ticker}.json"`）和上游请求URL的，
不做任何格式限制意味着调用方可以传入`../../xxx`这类值造成路径穿越。真实的
美股ticker就是1-6位字母，最多带一个"."后缀区分股票类别（比如BRK.B），
这个正则既覆盖了正常场景，又把危险字符挡在路由层，不用每个service函数
各自重复校验。
"""

from typing import Annotated

from fastapi import Path, Query

TICKER_PATTERN = r"^[A-Za-z]{1,6}(\.[A-Za-z]{1,2})?$"

TickerPath = Annotated[
    str, Path(pattern=TICKER_PATTERN, description="美股ticker，如 AAPL、BRK.B")
]

TickerQuery = Annotated[
    str | None, Query(pattern=TICKER_PATTERN, description="美股ticker，如 AAPL、BRK.B")
]
