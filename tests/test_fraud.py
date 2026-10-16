import json

import pytest
from fastapi.testclient import TestClient

from cdrfraud import detect, features
from cdrfraud.generator import generate
from cdrfraud.pipeline import run, write


@pytest.fixture(scope="module")
def small():
    # A smaller network keeps the suite fast; every fraud pattern is still planted.
    return run(seed=11, n_subscribers=800)


def test_generator_is_deterministic():
    a, b = generate(3, 800), generate(3, 800)
    assert a.cdrs.equals(b.cdrs)
    assert a.labels.equals(b.labels)


def test_every_planted_fraud_number_appears_in_the_cdrs(small):
    callers = set(small.data.cdrs["a_number"])
    assert set(small.data.labels["number"]) <= callers
    assert small.data.labels.groupby("fraud_type").size().to_dict() == {"irsf": 11, "simbox": 42, "wangiri": 4}


def test_features_capture_simbox_handset_sharing(small):
    sims = small.data.labels.query("fraud_type == 'simbox' and variant == 'obvious'")["number"]
    assert (small.features.loc[sims, "imei_sim_count"] >= 3).all()


def test_rules_catch_every_obvious_variant(small):
    scored = small.scored
    obvious = small.data.labels.query("variant == 'obvious'")["number"]
    assert scored.loc[obvious, "rule_hit"].all()


def test_rules_never_flag_the_legitimate_lookalikes(small):
    lookalikes = small.data.cohorts["number"]
    hits = small.scored.reindex(lookalikes)["rule_hit"].astype("boolean").fillna(False)
    assert not hits.any()


def test_model_ranks_stealthy_irsf_in_the_far_tail(small):
    stealthy = small.data.labels.query("fraud_type == 'irsf' and variant == 'stealthy'")["number"]
    assert (small.scored.loc[stealthy, "model_percentile"] >= 0.95).all()


def test_every_flag_explains_itself(small):
    flagged = small.scored[small.scored["flagged"]]
    assert flagged["reasons"].map(len).min() >= 1
    assert flagged.loc[flagged["rule_hit"], "risk"].min() >= 0.9


def test_metrics_report_each_method(small):
    m = small.metrics["methods"]
    assert set(m) == {"rules", "model", "combined"}
    assert m["rules"]["precision"] >= 0.95
    assert m["combined"]["recall"] >= m["rules"]["recall"]


def test_pipeline_writes_dashboard_files(small, tmp_path):
    write(small, tmp_path)
    suspects = json.loads((tmp_path / "suspects.json").read_text())
    timelines = json.loads((tmp_path / "timelines.json").read_text())
    assert suspects[0]["risk"] >= suspects[-1]["risk"]
    assert all(len(timelines["series"][s["number"]]["made"]) == 168 for s in suspects[:10])
    assert json.loads((tmp_path / "blocklist.json").read_text())


def test_api(small, tmp_path, monkeypatch):
    from cdrfraud import api

    write(small, tmp_path)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    api.data.cache_clear()
    client = TestClient(api.app)

    block = client.get("/api/blocklist").json()
    assert block["count"] > 0 and all(n["risk"] >= 0.9 for n in block["numbers"])
    top = client.get("/api/suspects", params={"limit": 5}).json()
    detail = client.get(f"/api/numbers/{top[0]['number']}").json()
    assert len(detail["timeline"]["made"]) == 168
    assert client.get("/api/numbers/000").status_code == 404
    assert client.get("/api/metrics").json()["fraud_planted"] == 57
    api.data.cache_clear()
