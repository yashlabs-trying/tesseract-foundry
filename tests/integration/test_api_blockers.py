"""Regression tests for the pre-GPU API blockers F-01, F-02, F-03.

F-01: /v1/services/recommend and /v1/economics/evaluate returned HTTP 500.
F-02: /v1/profiling/jobs returned HTTP 500 (Decimal not JSON serializable).
F-03: fixture price fields did not match the canonical Scenario schema.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from foundry.api import app
from foundry.models import Scenario

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SCENARIO_1 = FIXTURES / "scenario-1-single-gpu.yaml"

client = TestClient(app)


def _scenario_doc() -> dict:
    doc = yaml.safe_load(SCENARIO_1.read_text())
    doc["scenario_id"] = "test-scenario"
    return doc


# --------------------------------------------------------------------------- F-01
def test_recommend_valid_returns_200_not_500():
    resp = client.post("/v1/services/recommend", json={"scenario": _scenario_doc()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "recommendation_id" in body
    assert "recommended_candidate_id" in body


def test_recommend_missing_model_returns_422():
    doc = _scenario_doc()
    doc.pop("model")
    resp = client.post("/v1/services/recommend", json={"scenario": doc})
    assert resp.status_code == 422


def test_economics_evaluate_requires_registered_hardware():
    doc = _scenario_doc()
    hw = doc["hardware"][0]
    # Register the hardware profile first.
    reg = client.post("/v1/hardware/profiles", json=hw)
    assert reg.status_code == 200, reg.text
    candidate = {
        "candidate_id": "c1",
        "model_id": "Qwen/Qwen3-0.6B",
        "hardware_id": hw["hardware_id"],
        "serving_engine": "vllm",
        "precision": "fp16",
        "gpu_count": 1,
        "replica_count": 1,
    }
    ok = client.post(
        "/v1/economics/evaluate",
        json={"candidate": candidate, "workload": doc["workload"]},
    )
    assert ok.status_code == 200, ok.text
    assert "total_cost_monthly" in ok.json()

    # Unknown hardware must be rejected, not 500.
    bad = dict(candidate, hardware_id="does-not-exist")
    resp = client.post(
        "/v1/economics/evaluate",
        json={"candidate": bad, "workload": doc["workload"]},
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- F-02
def test_profiling_jobs_returns_200_not_500():
    doc = _scenario_doc()
    payload = {
        "model": doc["model"],
        "hardware": doc["hardware"],
        "workload": doc["workload"],
    }
    resp = client.post("/v1/profiling/jobs?mode=fixture", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["candidate_count"] >= 1
    # And the job must be retrievable (proves the Decimal params persisted).
    job = client.get(f"/v1/profiling/jobs/{body['job_id']}")
    assert job.status_code == 200
    assert job.json()["params"]["mode"] == "fixture"


def test_profiling_jobs_invalid_body_returns_422():
    resp = client.post("/v1/profiling/jobs?mode=fixture", json={"model": {"model_id": "x"}})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- F-03
def test_all_fixture_scenarios_use_canonical_pricing():
    files = sorted(FIXTURES.glob("scenario-*.yaml"))
    assert files, "no scenario fixtures found"
    for path in files:
        doc = yaml.safe_load(path.read_text())
        doc.setdefault("scenario_id", path.stem)
        scenario = Scenario(**doc)
        assert scenario.output_price_per_million_input_tokens is not None, path.name
        assert scenario.output_price_per_million_output_tokens is not None, path.name


def test_negative_price_rejected():
    doc = _scenario_doc()
    doc["output_price_per_million_input_tokens"] = -1
    with pytest.raises(ValidationError):
        Scenario(**doc)


def test_incomplete_pricing_rejected():
    doc = _scenario_doc()
    doc.pop("output_price_per_million_output_tokens")
    with pytest.raises(ValidationError):
        Scenario(**doc)


def test_legacy_price_field_names_are_not_silently_accepted():
    """A scenario using the old (non-canonical) field names must not silently
    produce estimates from default prices: the canonical fields are absent, so
    the price pair is incomplete and rejected."""
    doc = _scenario_doc()
    doc["input_price_per_million"] = 0.15  # legacy name
    doc["output_price_per_million"] = 0.60
    doc.pop("output_price_per_million_input_tokens")
    # Only the output price remains -> incomplete pair -> rejected.
    with pytest.raises(ValidationError):
        Scenario(**doc)
