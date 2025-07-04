import os, json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import uvicorn
from dotenv import load_dotenv

# make sure we can import compute_metrics.py
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))
from compute_metrics import (
    compute_metrics,
    _fetch_mixpanel_events,
    _load_local_events
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[0] / ".env")

class MetricsRequest(BaseModel):
    start: str  # YYYY-MM-DD
    end:   str  # YYYY-MM-DD
    fetch: bool = True
    from_date: str | None = None
    to_date:   str | None = None
    event:     str = "app_opened"

app = FastAPI(title="Growth-Model API")

# CORS so that localhost:5173 can call localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/metrics")
async def metrics(req: MetricsRequest):
    start = datetime.fromisoformat(req.start)
    end   = datetime.fromisoformat(req.end)
    if end < start:
        raise HTTPException(400, "end must be >= start")

    if req.fetch:
        secret = os.getenv("MIXPANEL_API_SECRET")
        if not secret:
            raise HTTPException(500, "MIXPANEL_API_SECRET not set")
        events = _fetch_mixpanel_events(
            secret,
            req.from_date or req.start,
            req.to_date or req.end,
            req.event,
        )
    else:
        # Example: read events from a file alongside compute_metrics.py
        events = _load_local_events([ROOT / "opens.parquet"], None, None)

    out = compute_metrics(events, start, end)
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)