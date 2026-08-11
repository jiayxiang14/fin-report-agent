from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
)

app = FastAPI(title="US Equity Research Agent API")

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
