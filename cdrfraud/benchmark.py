"""Run the whole pipeline on several seeds and report the spread, not one lucky run.

    python -m cdrfraud.benchmark [--seeds 1 2 3 4 5] [--out dashboard/public/data/benchmark.json]

Each seed generates a different network and different fraud, then retrains the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pipeline import run


def benchmark(seeds: list[int], n_subscribers: int = 3000) -> dict:
    rows, variants = [], []
    for seed in seeds:
        m = run(seed, n_subscribers).metrics
        for method, r in m["methods"].items():
            rows.append({"seed": seed, "method": method, **{k: r[k] for k in ("flagged", "precision", "recall", "f1")}})
            variants += [{"seed": seed, "method": method, **v} for v in r["by_variant"]]
        rows.append({"seed": seed, "method": "average_precision", "precision": m["average_precision"]["combined_risk"]})

    df = pd.DataFrame(rows)
    summary = {}
    for method, g in df[df.method != "average_precision"].groupby("method"):
        summary[method] = {k: {"mean": round(g[k].mean(), 3), "min": round(g[k].min(), 3), "max": round(g[k].max(), 3)} for k in ("precision", "recall", "f1", "flagged")}
    ap = df[df.method == "average_precision"]["precision"]

    v = pd.DataFrame(variants).groupby(["method", "fraud_type", "variant"])[["caught", "planted"]].sum()
    recall_by_variant = [
        {"method": m, "fraud_type": t, "variant": var, "recall": round(r.caught / r.planted, 3), "caught": int(r.caught), "planted": int(r.planted)}
        for (m, t, var), r in v.iterrows()
    ]
    return {
        "seeds": seeds,
        "methods": summary,
        "combined_average_precision": {"mean": round(ap.mean(), 3), "min": round(ap.min(), 3), "max": round(ap.max(), 3)},
        "recall_by_variant": recall_by_variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", type=Path, default=Path("dashboard/public/data/benchmark.json"))
    args = parser.parse_args()
    result = benchmark(args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for method, r in result["methods"].items():
        print(f"{method:9} precision {r['precision']['mean']:.2f} ({r['precision']['min']:.2f}-{r['precision']['max']:.2f})"
              f"  recall {r['recall']['mean']:.2f} ({r['recall']['min']:.2f}-{r['recall']['max']:.2f})")
    print("combined average precision", result["combined_average_precision"])
    for row in result["recall_by_variant"]:
        print(f"  {row['method']:9} {row['fraud_type']:8} {row['variant']:9} {row['caught']}/{row['planted']}")


if __name__ == "__main__":
    main()
