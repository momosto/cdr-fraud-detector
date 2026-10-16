"""Per-number behaviour features, computed from raw CDRs.

One row per calling number (A-number) that made at least MIN_CALLS calls, including
foreign numbers calling into the network, since that is where Wangiri shows up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import CC, HIGH_RISK_PREFIXES

MIN_CALLS = 5

FEATURES = [
    "out_calls_per_day",
    "in_out_ratio",
    "distinct_dest_ratio",
    "short_call_share",
    "mean_answered_s",
    "intl_share",
    "high_risk_share",
    "night_share",
    "distinct_cells",
    "imei_sim_count",
    "peak_hour_calls",
]

# Plain-English names, for reasons shown to fraud analysts.
LABELS = {
    "out_calls_per_day": "calls per active day",
    "in_out_ratio": "incoming-to-outgoing ratio",
    "distinct_dest_ratio": "share of calls to new numbers",
    "short_call_share": "share of calls of 3 s or less",
    "mean_answered_s": "average call length (s)",
    "intl_share": "share of international calls",
    "high_risk_share": "share of calls to premium-risk ranges",
    "night_share": "share of calls between 00:00 and 05:00",
    "distinct_cells": "cells used",
    "imei_sim_count": "SIMs seen on the same handset",
    "peak_hour_calls": "calls in busiest hour",
}


def annotate(cdrs: pd.DataFrame) -> pd.DataFrame:
    df = cdrs.copy()
    df["hour"] = df["start_time"].dt.floor("h")
    df["day"] = df["start_time"].dt.floor("D")
    df["intl"] = ~df["b_number"].str.startswith(CC)
    df["high_risk"] = df["b_number"].str.startswith(tuple(HIGH_RISK_PREFIXES))
    df["short"] = df["duration_s"] <= 3
    df["night"] = df["start_time"].dt.hour < 5
    df["on_net"] = df["cell_id"].notna()
    return df


def build(cdrs: pd.DataFrame) -> pd.DataFrame:
    df = annotate(cdrs)
    g = df.groupby("a_number")

    out = pd.DataFrame({
        "calls": g.size(),
        "active_days": g["day"].nunique(),
        "distinct_dest": g["b_number"].nunique(),
        "short_call_share": g["short"].mean(),
        "intl_share": g["intl"].mean(),
        "high_risk_share": g["high_risk"].mean(),
        "night_share": g["night"].mean(),
        "distinct_cells": g["cell_id"].nunique(),
        "on_net": g["on_net"].any(),
    })
    out = out[out["calls"] >= MIN_CALLS].copy()

    answered = df[df["answered"]].groupby("a_number")["duration_s"].mean()
    out["mean_answered_s"] = answered.reindex(out.index).fillna(0)
    out["out_calls_per_day"] = out["calls"] / out["active_days"]
    out["distinct_dest_ratio"] = out["distinct_dest"] / out["calls"]

    incoming = df.groupby("b_number").size()
    out["in_calls"] = incoming.reindex(out.index).fillna(0)
    out["in_out_ratio"] = out["in_calls"] / out["calls"]

    # How many different SIMs appear on each handset, then the worst handset per SIM.
    sims_per_imei = df.dropna(subset=["imei"]).groupby("imei")["a_number"].nunique()
    worst = df.dropna(subset=["imei"]).assign(n=lambda x: x["imei"].map(sims_per_imei)).groupby("a_number")["n"].max()
    out["imei_sim_count"] = worst.reindex(out.index).fillna(0)

    out["peak_hour_calls"] = df.groupby(["a_number", "hour"]).size().groupby("a_number").max().reindex(out.index)

    # Worst single day of premium-range traffic, for the IRSF rule.
    hr = df[df["high_risk"]].groupby(["a_number", "day"]).agg(n=("call_id", "size"), minutes=("duration_s", "sum"))
    daily = hr.groupby("a_number").max()
    out["high_risk_calls_max_day"] = daily["n"].reindex(out.index).fillna(0)
    out["high_risk_minutes_max_day"] = (daily["minutes"] / 60).reindex(out.index).fillna(0)

    out.index.name = "number"
    return out


def hourly_activity(cdrs: pd.DataFrame, numbers: list[str]) -> pd.DataFrame:
    """Calls made and received per hour, for drawing a timeline of each suspect."""
    df = cdrs[["start_time", "a_number", "b_number"]].copy()
    df["hour"] = df["start_time"].dt.floor("h")
    made = df[df["a_number"].isin(numbers)].groupby(["a_number", "hour"]).size().rename("made")
    received = df[df["b_number"].isin(numbers)].groupby(["b_number", "hour"]).size().rename("received")
    made.index.names = received.index.names = ["number", "hour"]
    return pd.concat([made, received], axis=1).fillna(0).astype(int).reset_index()


def model_matrix(features: pd.DataFrame) -> np.ndarray:
    """Log-scaled feature matrix: counts are heavy-tailed, ratios are already 0-1."""
    x = features[FEATURES].astype(float).copy()
    for col in ["out_calls_per_day", "mean_answered_s", "distinct_cells", "imei_sim_count", "peak_hour_calls", "in_out_ratio"]:
        x[col] = np.log1p(x[col])
    return x.to_numpy()
