"""Detection: telecom fraud rules, an Isolation Forest, and the combined risk score.

Rules encode what fraud analysts already know and give precise, explainable hits.
The model has no fraud labels at all; it learns what normal calling looks like and
scores how far each number is from it, which is how it can catch variants the rules
were never written for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .features import FEATURES, LABELS, model_matrix

# Share of numbers the model flags on its own. About 1% of numbers in the synthetic
# network are fraudulent; flagging slightly more trades precision for recall.
MODEL_FLAG_SHARE = 0.015


def apply_rules(f: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per rule hit: number, rule, reason."""
    hits = []

    # A foreign number one-ringing 50+ people in an hour. (Not "all different people":
    # repeat campaigns re-ring the same victims.)
    wangiri = f[~f.on_net & (f.peak_hour_calls >= 50) & (f.short_call_share >= 0.8)]
    for n, r in wangiri.iterrows():
        hits.append((n, "wangiri", f"Foreign number made {int(r.peak_hour_calls)} calls in one hour; "
                                   f"{r.short_call_share:.0%} lasted 3 s or less"))

    irsf = f[(f.high_risk_calls_max_day >= 20) | (f.high_risk_minutes_max_day >= 120)]
    for n, r in irsf.iterrows():
        hits.append((n, "irsf", f"{int(r.high_risk_calls_max_day)} calls and {r.high_risk_minutes_max_day:.0f} minutes "
                                f"to premium-risk ranges in a single day"))

    volume = f[f.on_net & (f.out_calls_per_day >= 100) & (f.in_out_ratio <= 0.05) & (f.distinct_cells <= 2) & (f.distinct_dest_ratio >= 0.7)]
    for n, r in volume.iterrows():
        hits.append((n, "simbox", f"{r.out_calls_per_day:.0f} outgoing calls a day from {int(r.distinct_cells)} cell(s), "
                                  f"almost none incoming ({r.in_out_ratio:.1%})"))
    shared = f[f.on_net & (f.imei_sim_count >= 3)]
    for n, r in shared.iterrows():
        hits.append((n, "simbox", f"Handset shared by {int(r.imei_sim_count)} different SIMs"))

    return pd.DataFrame(hits, columns=["number", "rule", "reason"])


def score_model(f: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    x = RobustScaler().fit_transform(model_matrix(f))
    forest = IsolationForest(n_estimators=400, random_state=seed).fit(x)
    raw = -forest.score_samples(x)  # higher = more unusual
    pct = pd.Series(raw, index=f.index).rank(pct=True)

    # Explain each score: the two features furthest from typical behaviour.
    z = pd.DataFrame(x, index=f.index, columns=FEATURES)
    drivers = z.abs().apply(lambda row: list(row.nlargest(2).index), axis=1)
    return pd.DataFrame({"model_score": raw, "model_percentile": pct, "model_drivers": drivers})


def guess_type(r: pd.Series) -> str:
    """Name the pattern a model-only flag most resembles, for analysts' triage."""
    if not r.on_net and r.short_call_share >= 0.5:
        return "wangiri"
    if r.high_risk_share >= 0.2:
        return "irsf"
    if r.on_net and r.in_out_ratio < 0.15 and r.out_calls_per_day >= 30:
        return "simbox"
    return "unusual"


def combine(f: pd.DataFrame, rule_hits: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    out = f.join(model)
    by_number = rule_hits.groupby("number")
    out["rule_types"] = by_number["rule"].agg(lambda s: sorted(set(s))).reindex(out.index)
    out["rule_reasons"] = by_number["reason"].agg(list).reindex(out.index)
    out["rule_hit"] = out["rule_types"].notna()
    out["model_flag"] = out["model_percentile"] >= 1 - MODEL_FLAG_SHARE

    # Risk: a rule hit is strong evidence (0.9 and up); otherwise the model percentile,
    # stretched so only the far tail scores high.
    model_part = out["model_percentile"] ** 20
    out["risk"] = np.where(out["rule_hit"], 0.9 + 0.1 * out["model_percentile"], 0.85 * model_part)
    out["flagged"] = out["rule_hit"] | out["model_flag"]

    def kind(r):
        return r.rule_types[0] if isinstance(r.rule_types, list) else guess_type(r)

    def reasons(r):
        items = [f"rule: {x}" for x in (r.rule_reasons if isinstance(r.rule_reasons, list) else [])]
        if r.model_flag:
            drivers = ", ".join(f"{LABELS[d]} {r[d]:.2f}".rstrip("0").rstrip(".") for d in r.model_drivers)
            top = 1 - r.model_percentile
            share = "<0.1%" if top < 0.001 else f"{top:.1%}"
            items.append(f"model: outlier, top {share} of numbers ({drivers})")
        return items

    out["fraud_type"] = out.apply(kind, axis=1)
    out["reasons"] = out.apply(reasons, axis=1)
    return out.sort_values("risk", ascending=False)
