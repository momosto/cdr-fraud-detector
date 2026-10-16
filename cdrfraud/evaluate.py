"""Score the detectors against the fraud planted by the generator."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score

METHODS = {"rules": "rule_hit", "model": "model_flag", "combined": "flagged"}


def evaluate(scored: pd.DataFrame, labels: pd.DataFrame, cohorts: pd.DataFrame) -> dict:
    truth = scored.index.isin(labels["number"])
    lab = labels.set_index("number")
    cohort = cohorts.set_index("number")["cohort"]

    methods = {}
    for name, col in METHODS.items():
        flagged = scored[col]
        tp = int((flagged & truth).sum())
        n_flagged = int(flagged.sum())
        precision = tp / n_flagged if n_flagged else 0.0
        recall = tp / len(lab)
        by_variant = (
            lab.assign(caught=scored[col].reindex(lab.index).fillna(False).astype(bool))
            .groupby(["fraud_type", "variant"])["caught"]
            .agg(["sum", "count"])
        )
        false_pos = scored.index[flagged & ~truth]
        methods[name] = {
            "flagged": n_flagged,
            "true_positives": tp,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if tp else 0.0,
            "by_variant": [
                {"fraud_type": t, "variant": v, "caught": int(r["sum"]), "planted": int(r["count"])}
                for (t, v), r in by_variant.iterrows()
            ],
            "false_positives_by_cohort": cohort.reindex(false_pos).fillna("ordinary subscriber").value_counts().to_dict(),
        }

    return {
        "numbers_scored": int(len(scored)),
        "fraud_planted": int(len(lab)),
        "methods": methods,
        # Threshold-free: how well each score ranks fraud above everything else.
        "average_precision": {
            "model_score": round(average_precision_score(truth, scored["model_score"]), 3),
            "combined_risk": round(average_precision_score(truth, scored["risk"]), 3),
        },
    }
