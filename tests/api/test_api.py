"""The HTTP surface, driven through FastAPI's TestClient.

The exit criterion for migration step 7 is that `/health` returns the model
version, so that is asserted first and hardest: a health check that only says
"the process is up" lets a load balancer route traffic to a container serving
a stale artefact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from logistics import __version__
from logistics.api.app import create_app
from logistics.settings import Settings

VALID = {
    "distance_km": 450.0,
    "weight_tons": 12.0,
    "vendor_rating": 4.0,
    "weather_condition": "Storm",
    "traffic_density": "High",
    "vehicle_type": "Diesel Truck",
}


@pytest.fixture(scope="module")
def client(repo_root):
    settings = Settings.from_env(
        project_root=repo_root, verify_artifact_checksum=False
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# --- the exit criterion -----------------------------------------------------

def test_health_returns_the_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"], "/health must identify the artefact it serves"
    assert body["model_name"] == "logistics_delay_risk"
    assert body["artifact_sha256"]
    assert body["api_version"] == __version__


def test_health_reports_unavailable_instead_of_crashing(tmp_path):
    """A missing artefact must produce a diagnosable 200/unavailable, not a 500.

    A container that exits on a bad artefact restarts forever and tells an
    operator nothing.
    """
    settings = Settings.from_env(
        project_root=tmp_path,
        model_path=tmp_path / "does-not-exist.pkl",
    )
    with TestClient(create_app(settings)) as broken:
        body = broken.get("/health").json()
        assert body["status"] == "unavailable"
        assert body["model_version"] is None
        assert body["detail"]

        # And scoring must say "not ready", not "your request was bad".
        assert broken.post("/score", json=VALID).status_code == 503


# --- /model -----------------------------------------------------------------

def test_model_endpoint_publishes_the_contract(client):
    body = client.get("/model").json()
    assert body["feature_order"] == [
        "Distance_km", "Weight_tons", "Vendor_Rating",
        "Weather_Condition", "Traffic_Density", "Vehicle_Type",
    ]
    assert set(body["thresholds"]) == {"high_risk", "medium_risk", "cost_fn_over_fp"}
    assert body["numeric_ranges"]["Vendor_Rating"]["min"] == pytest.approx(1.71)
    assert body["caveats"], "the artefact's caveats must reach a consumer"


# --- /score -----------------------------------------------------------------

def test_score_returns_a_probability_and_its_provenance(client):
    body = client.post("/score", json=VALID).json()

    assert 0.0 <= body["risk"]["probability"] <= 1.0
    assert body["risk"]["level"] in {"Low Risk", "Medium Risk", "High Risk"}
    # Provenance travels with the score; without it a stored prediction cannot
    # be traced back to the weights that produced it.
    assert body["model_version"]
    assert body["artifact_sha256"]
    assert body["scored_at"]


def test_score_computes_carbon_deterministically(client):
    first = client.post("/score", json=VALID).json()["carbon"]
    second = client.post("/score", json=VALID).json()["carbon"]
    assert first == second
    assert first["co2_tons"] == pytest.approx(first["co2_kg"] / 1000.0)
    assert first["cost"] == pytest.approx(first["co2_tons"] * first["price_per_ton"])


def test_an_unknown_category_is_rejected_by_validation(client):
    bad = VALID | {"vehicle_type": "Hovercraft"}
    assert client.post("/score", json=bad).status_code == 422


def test_an_out_of_range_value_is_rejected(client):
    assert client.post("/score", json=VALID | {"vendor_rating": 9.0}).status_code == 422
    assert client.post("/score", json=VALID | {"distance_km": -1.0}).status_code == 422


def test_an_unexpected_field_is_rejected(client):
    """extra="forbid": a typo in a field name must not be silently ignored."""
    assert client.post("/score", json=VALID | {"distnace_km": 1.0}).status_code == 422


def test_an_input_outside_the_training_range_still_scores_but_warns(client):
    """Extrapolation is reported, not refused - the caller decides."""
    body = client.post("/score", json=VALID | {"distance_km": 4000.0}).json()
    assert 0.0 <= body["risk"]["probability"] <= 1.0
    assert body["extrapolation_warnings"], "leaving the training range must be said"


# --- /score/batch -----------------------------------------------------------

def test_batch_scores_every_row(client):
    payload = {"shipments": [VALID, VALID | {"weather_condition": "Normal"}]}
    body = client.post("/score/batch", json=payload).json()

    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert all(0.0 <= r["risk"]["probability"] <= 1.0 for r in body["results"])


def test_batch_agrees_with_single_scoring(client):
    """The batch path must not be a different model.

    score_many issues one predict_proba call; score() issues one per row. If
    those disagree, a client gets different answers depending on how it asked.
    """
    single = client.post("/score", json=VALID).json()["risk"]["probability"]
    batch = client.post("/score/batch", json={"shipments": [VALID]}).json()
    assert batch["results"][0]["risk"]["probability"] == pytest.approx(single, abs=1e-12)


def test_an_empty_batch_is_rejected(client):
    assert client.post("/score/batch", json={"shipments": []}).status_code == 422


def test_an_oversized_batch_is_rejected(client):
    """Unbounded input is a memory-exhaustion vector."""
    payload = {"shipments": [VALID] * 1001}
    assert client.post("/score/batch", json=payload).status_code == 422


def test_shipment_id_is_echoed_back(client):
    body = client.post("/score", json=VALID | {"shipment_id": "SHP-00042"}).json()
    assert body["shipment_id"] == "SHP-00042"


# --- documentation ----------------------------------------------------------

def test_the_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/model", "/score", "/score/batch"}
