from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    agent,
    best_of_n,
    company_profile,
    filing,
    financials,
    financials_history,
    peer,
    sector,
    thematic_flow,
    trace,
)
from app.services.sec_client import SecClientError

app = FastAPI(title="US Equity Research Agent API")


@app.exception_handler(SecClientError)
async def handle_sec_client_error(request: Request, exc: SecClientError) -> JSONResponse:
    # 兜底：各路由对更具体的SecEdgarError（子类）已经有自己的try/except转成502，
    # 但SecClientError基类本身（比如SEC_EDGAR_USER_AGENT没配置）不会被那些
    # except SecEdgarError挡住——没有这个全局handler，配置缺失会让用户看到
    # 裸露的500堆栈，不是干净的错误提示
    return JSONResponse(status_code=502, content={"detail": str(exc)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(financials.router)
app.include_router(financials_history.router)
app.include_router(sector.router)
app.include_router(filing.router)
app.include_router(agent.router)
app.include_router(best_of_n.router)
app.include_router(company_profile.router)
app.include_router(peer.router)
app.include_router(thematic_flow.router)
app.include_router(trace.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
