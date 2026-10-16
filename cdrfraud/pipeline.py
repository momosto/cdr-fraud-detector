"""Run everything and write the results the dashboard and API read.

    python -m cdrfraud.pipeline --out dashboard/public/data [--seed 7] [--cdrs data/cdrs.csv.gz]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import detect, features
from .evaluate import evaluate
from .generator import START, Dataset, generate

TOP_SUSPECTS = 150
BLOCK_MIN_RISK = 0.9  # rule-confirmed numbers only; model-only flags go to an analyst first


@dataclass
class Result:
    data: Dataset
    features: pd.DataFrame
    scored: pd.DataFrame
    metrics: dict


def run(seed: int = 7, n_subscribers: int = 3000) -> Result:
    data = generate(seed, n_subscribers)
    feats = features.build(data.cdrs)
    scored = detect.combine(feats, detect.apply_rules(feats), detect.score_model(feats, seed))
    return Result(data, feats, scored, evaluate(scored, data.labels, data.cohorts))


def _truth(result: Result) -> dict[str, str]:
    """What each number really is, shown in the dashboard so errors are visible."""
    truth = {r.number: f"{r.fraud_type} ({r.variant})" for r in result.data.labels.itertuples()}
    for r in result.data.cohorts.itertuples():
        truth.setdefault(r.number, f"not fraud: {r.cohort}")
    return truth


def suspects(result: Result, limit: int = TOP_SUSPECTS) -> list[dict]:
    truth = _truth(result)
    rows = []
    for number, r in result.scored.head(limit).iterrows():
        rows.append({
            "number": number,
            "risk": round(float(r.risk), 4),
            "fraud_type": r.fraud_type,
            "flagged": bool(r.flagged),
            "rule_hit": bool(r.rule_hit),
            "model_flag": bool(r.model_flag),
            "model_percentile": round(float(r.model_percentile), 4),
            "reasons": r.reasons,
            "truth": truth.get(number, "not fraud: ordinary subscriber"),
            "features": {k: round(float(r[k]), 3) for k in features.FEATURES},
        })
    return rows


def write(result: Result, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    top = suspects(result)
    numbers = [s["number"] for s in top]

    hourly = features.hourly_activity(result.data.cdrs, numbers)
    hours = pd.date_range(START, periods=7 * 24, freq="h")
    timelines = {}
    for number, g in hourly.groupby("number"):
        g = g.set_index("hour").reindex(hours, fill_value=0)
        timelines[number] = {"made": g["made"].astype(int).tolist(), "received": g["received"].astype(int).tolist()}

    blocklist = [
        {"number": s["number"], "fraud_type": s["fraud_type"], "risk": s["risk"], "reason": s["reasons"][0]}
        for s in top if s["risk"] >= BLOCK_MIN_RISK
    ]
    summary = {
        "period_start": str(START.date()),
        "days": 7,
        "cdrs": int(len(result.data.cdrs)),
        "subscribers_and_callers": int(result.data.cdrs["a_number"].nunique()),
        **result.metrics,
    }

    files = {"summary.json": summary, "suspects.json": top, "timelines.json": {"start": str(START), "series": timelines}, "blocklist.json": blocklist}
    for name, payload in files.items():
        (out / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dashboard/public/data"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cdrs", type=Path, help="also save the raw CDRs here (.csv.gz)")
    args = parser.parse_args()

    result = run(args.seed)
    write(result, args.out)
    if args.cdrs:
        args.cdrs.parent.mkdir(parents=True, exist_ok=True)
        result.data.cdrs.to_csv(args.cdrs, index=False)

    for name, m in result.metrics["methods"].items():
        print(f"{name:9} flagged {m['flagged']:3}  precision {m['precision']:.2f}  recall {m['recall']:.2f}  f1 {m['f1']:.2f}")
    print("average precision:", result.metrics["average_precision"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
