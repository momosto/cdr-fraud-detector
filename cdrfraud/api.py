"""Read-only API over the latest detection run, including a block list for enforcement.

    uvicorn cdrfraud.api:app --reload

An enforcement point (an IMS, SBC or switch) polls GET /api/blocklist and bars the
numbers it returns. Only rule-confirmed numbers (risk >= 0.9) are on it by default;
model-only flags go to an analyst first, because the model has no labels to be sure with.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

DATA_DIR = Path(os.environ.get("CDR_DATA_DIR", Path(__file__).resolve().parent.parent / "dashboard" / "public" / "data"))

app = FastAPI(title="CDR Fraud Detector", version="1.0.0")


@lru_cache(maxsize=1)
def data() -> dict:
    if not (DATA_DIR / "summary.json").exists():
        from .pipeline import run, write

        write(run(), DATA_DIR)
    return {name: json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8")) for name in ("summary", "suspects", "timelines")}


@app.get("/api/metrics")
def metrics() -> dict:
    return data()["summary"]


@app.get("/api/suspects")
def suspects(min_risk: float = Query(0.0, ge=0, le=1), fraud_type: str | None = None, limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    rows = [s for s in data()["suspects"] if s["risk"] >= min_risk and (fraud_type is None or s["fraud_type"] == fraud_type)]
    return rows[:limit]


@app.get("/api/numbers/{number}")
def number(number: str) -> dict:
    match = next((s for s in data()["suspects"] if s["number"] == number), None)
    if match is None:
        raise HTTPException(404, f"{number} is not among the current suspects.")
    return {**match, "timeline": data()["timelines"]["series"].get(number), "timeline_start": data()["timelines"]["start"]}


@app.get("/api/blocklist")
def blocklist(min_risk: float = Query(0.9, ge=0, le=1)) -> dict:
    rows = [
        {"number": s["number"], "fraud_type": s["fraud_type"], "risk": s["risk"], "reason": s["reasons"][0]}
        for s in data()["suspects"] if s["risk"] >= min_risk
    ]
    return {"min_risk": min_risk, "count": len(rows), "numbers": rows}


# In Docker the built dashboard is served from the same origin, after the API routes.
DASHBOARD_DIR = Path(os.environ.get("CDR_DASHBOARD_DIR", ""))
if DASHBOARD_DIR.name and DASHBOARD_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
