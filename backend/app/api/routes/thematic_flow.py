from fastapi import APIRouter, HTTPException

from app.models.thematic_flow import ThematicFlowResponse
from app.services.thematic_flow import ThematicFlowError, get_thematic_flow

router = APIRouter(prefix="/api/thematic-flow", tags=["thematic_flow"])


@router.get("", response_model=ThematicFlowResponse)
async def read_thematic_flow(ticker: str | None = None) -> ThematicFlowResponse:
    """展示的始终是同样这10个AI算力/基建细分主题的当前相对轮动位置，跟具体
    分析哪家公司无关——`ticker`是可选查询参数，传了就顺带做一次确定性反查
    （这个ticker本身是不是某个主题篮子的成分股，见`matched_themes`），不传
    就只是纯展示全部10个主题。这份数据同时也是`get_thematic_flow`这个Agent
    工具背后调用的同一个函数（`tools.py`），前端同步面板和Agent自主判断是
    两个独立的消费方，互不影响。
    """
    try:
        return await get_thematic_flow(ticker)
    except ThematicFlowError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
