"""The scoring service, tested against a fake model.

None of these tests load the 8.7 MB artefact or start Streamlit. That is the
point of the refactor: the seven-step scoring procedure used to exist only
inside `if st.button(...)`, so it could not be exercised at all except through a
browser. Here it runs in milliseconds against a two-line stand-in predictor,
which means the *procedure* is under test rather than the model's accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from logistics.domain.enums import RiskLevel, TrafficDensity, VehicleType, WeatherCondition
from logistics.domain.models import ShipmentFeatures
from logistics.errors import ScoringError
from logistics.infrastructure.model_repository import ModelBundle
from logistics.infrastructure.prediction_log import JsonlPredictionLog
from logistics.services.scoring import ScoringService

FEATURE_ORDER = [
    "Distance_km",
    "Weight_tons",
    "Vendor_Rating",
    "Weather_Condition",
    "Traffic_Density",
    "Vehicle_Type",
]


class FakePredictor:
    """Returns a probability driven by distance, so results are predictable.

    Also records the frame it was handed, which is how the column-order contract
    below is verified.
    """

    def __init__(self, probability: float | None = None) -> None:
        self.fixed = probability
        self.last_frame: pd.DataFrame | None = None

    def predict_proba(self, X):  # noqa: N803 - sklearn's parameter name
        self.last_frame = X
        if self.fixed is not None:
            p = np.full(len(X), self.fixed)
        else:
            p = np.clip(np.asarray(X["Distance_km"], dtype=float) / 1000.0, 0.0, 1.0)
        return np.column_stack([1 - p, p])


def make_bundle(predictor=None, **metadata_overrides) -> ModelBundle:
    metadata = {
        "model_name": "logistics_delay_risk",
        "trained_at": "2026-08-04T10:47:32+03:00",
        "feature_order": FEATURE_ORDER,
        "thresholds": {"high_risk": 0.50, "medium_risk": 0.20, "cost_fn_over_fp": 4.0},
        "categorical_levels": {
            "Weather_Condition": ["Normal", "Rain", "Snow", "Storm"],
            "Traffic_Density": ["High", "Low", "Medium"],
            "Vehicle_Type": ["Diesel Truck", "Electric Semi", "Hybrid Van"],
        },
        "numeric_ranges": {
            "Distance_km": {"min": 51.1, "max": 1199.7},
            "Weight_tons": {"min": 1.01, "max": 24.99},
            "Vendor_Rating": {"min": 1.71, "max": 4.89},
        },
        "metrics": {"oof_roc_auc": 0.7558},
    }
    metadata.update(metadata_overrides)
    return ModelBundle(
        model=predictor or FakePredictor(),
        metadata=metadata,
        artifact_path=pytest.importorskip("pathlib").Path("fake.pkl"),
        artifact_sha256="a" * 64,
    )


def make_service(predictor=None, price: float = 50.0, log=None, **meta) -> ScoringService:
    return ScoringService(
        make_bundle(predictor, **meta), carbon_price_per_ton=price, prediction_log=log
    )


def features(**overrides) -> ShipmentFeatures:
    base = {
        "distance_km": 450.0,
        "weight_tons": 12.0,
        "vendor_rating": 4.0,
        "weather_condition": WeatherCondition.STORM,
        "traffic_density": TrafficDensity.HIGH,
        "vehicle_type": VehicleType.DIESEL_TRUCK,
    }
    return ShipmentFeatures(**(base | overrides))


# ---------------------------------------------------------------------------
# Input validation - the layer that did not exist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field, value",
    [
        ("distance_km", 0.0),
        ("distance_km", -10.0),
        ("weight_tons", 0.0),
        ("vendor_rating", 5.5),
        ("vendor_rating", 0.5),
    ],
)
def test_physically_impossible_inputs_are_rejected(field, value):
    """Previously the only guard was a Streamlit slider's range, which vanishes
    the moment the same logic is called from anywhere else."""
    with pytest.raises(ValueError):
        features(**{field: value})


def test_unknown_category_is_rejected_at_the_contract_boundary():
    with pytest.raises(ValueError):
        features(vehicle_type="CNG Truck")


def test_extra_fields_are_refused():
    """A typo in a caller's payload must not be silently ignored."""
    with pytest.raises(ValueError):
        ShipmentFeatures(
            distance_km=100.0,
            weight_tons=5.0,
            vendor_rating=4.0,
            weather_condition="Normal",
            traffic_density="Low",
            vehicle_type="Hybrid Van",
            vendor_ratng=3.0,  # typo
        )


def test_vendor_rating_below_the_old_slider_floor_is_now_scoreable():
    """app.py's slider started at 2.5 while the model was trained from 1.71, so
    the worst vendors - the whole point of an early-warning panel - could not be
    entered. The contract allows them and flags the extrapolation instead."""
    assessment = make_service().score(features(vendor_rating=1.8))
    assert assessment.risk.probability >= 0.0
    assert not assessment.is_extrapolated  # 1.8 is inside [1.71, 4.89]


# ---------------------------------------------------------------------------
# The model contract
# ---------------------------------------------------------------------------
def test_the_model_receives_columns_in_its_own_declared_order():
    """Column order is read from the artefact, never hardcoded - the classic
    source of train/serve skew."""
    shuffled = ["Vehicle_Type", "Distance_km", "Traffic_Density",
                "Weight_tons", "Weather_Condition", "Vendor_Rating"]
    predictor = FakePredictor()
    service = make_service(predictor, feature_order=shuffled)

    service.score(features())

    assert list(predictor.last_frame.columns) == shuffled


def test_thresholds_come_from_the_artefact_not_from_module_defaults():
    """Retraining under a different cost assumption must move every consumer."""
    service = make_service(
        FakePredictor(0.30),
        thresholds={"high_risk": 0.90, "medium_risk": 0.80, "cost_fn_over_fp": 0.25},
    )
    assessment = service.score(features())

    assert assessment.risk.level is RiskLevel.LOW  # 0.30 < 0.80, not Medium
    assert assessment.risk.high_threshold == 0.90
    assert assessment.risk.cost_fn_over_fp == 0.25


def test_a_model_failure_surfaces_as_a_typed_error():
    class Broken:
        def predict_proba(self, X):  # noqa: N803
            raise RuntimeError("estimator is not fitted")

    with pytest.raises(ScoringError, match="failed to score"):
        make_service(Broken()).score(features())


# ---------------------------------------------------------------------------
# Carbon must not be re-implemented
# ---------------------------------------------------------------------------
def test_carbon_matches_the_domain_function_exactly():
    from logistics.domain.carbon import compute_co2_kg

    f = features()
    assessment = make_service().score(f)
    expected = compute_co2_kg(450.0, 12.0, "Diesel Truck", "Storm", "High")

    assert assessment.carbon.co2_kg == pytest.approx(expected)
    assert assessment.carbon.co2_tons == pytest.approx(expected / 1000.0)


def test_carbon_price_is_injected_not_hardcoded():
    """The price is a jurisdiction- and year-dependent business parameter."""
    cheap = make_service(price=50.0).score(features())
    dear = make_service(price=85.0).score(features())

    assert dear.carbon.cost == pytest.approx(cheap.carbon.cost * 85.0 / 50.0)


# ---------------------------------------------------------------------------
# Extrapolation reporting
# ---------------------------------------------------------------------------
def test_out_of_range_input_scores_but_is_flagged():
    """Not an error: scoring a slightly out-of-range shipment is legitimate.
    Silently pretending it is business as usual is not."""
    assessment = make_service().score(features(distance_km=2000.0))

    assert assessment.is_extrapolated
    assert any("Distance_km" in w for w in assessment.extrapolation_warnings)
    assert 0.0 <= assessment.risk.probability <= 1.0


# ---------------------------------------------------------------------------
# Batch and frame paths share the row path's semantics
# ---------------------------------------------------------------------------
def test_score_many_agrees_with_score_row_by_row():
    service = make_service()
    batch = [features(distance_km=d) for d in (100.0, 600.0, 1100.0)]

    together = service.score_many(batch)
    apart = [make_service().score(f) for f in batch]

    assert [a.risk.probability for a in together] == [a.risk.probability for a in apart]
    assert [a.risk.level for a in together] == [a.risk.level for a in apart]


def test_score_many_issues_one_call_for_the_whole_batch():
    predictor = FakePredictor()
    service = make_service(predictor)
    service.score_many([features(distance_km=d) for d in (100.0, 500.0, 900.0)])
    assert len(predictor.last_frame) == 3


def test_score_many_on_an_empty_batch_returns_empty():
    assert make_service().score_many([]) == []


def test_score_frame_preserves_input_columns_and_adds_outputs():
    """The optimiser and the ETL need the result joinable back onto the fact
    table, so nothing may be dropped or reordered."""
    df = pd.DataFrame(
        {
            "Shipment_ID": ["SHP-00001", "SHP-00002"],
            "Distance_km": [200.0, 900.0],
            "Weight_tons": [5.0, 20.0],
            "Vendor_Rating": [4.5, 2.2],
            "Weather_Condition": ["Normal", "Snow"],
            "Traffic_Density": ["Low", "High"],
            "Vehicle_Type": ["Electric Semi", "Diesel Truck"],
        }
    )
    out = make_service().score_frame(df)

    assert list(df.columns) == list(out.columns)[: len(df.columns)]
    assert out["Shipment_ID"].tolist() == ["SHP-00001", "SHP-00002"]
    assert out["Delay_Risk_Probability"].between(0, 1).all()
    assert set(out["Risk_Level"]) <= {"Low Risk", "Medium Risk", "High Risk"}
    assert (out["CO2_Emission_kg_calculated"] > 0).all()


def test_score_frame_rejects_a_frame_missing_a_feature():
    df = pd.DataFrame({"Distance_km": [100.0], "Weight_tons": [5.0]})
    with pytest.raises(ScoringError, match="missing feature column"):
        make_service().score_frame(df)


def test_score_frame_agrees_with_the_row_path():
    df = pd.DataFrame(
        {
            "Distance_km": [450.0],
            "Weight_tons": [12.0],
            "Vendor_Rating": [4.0],
            "Weather_Condition": ["Storm"],
            "Traffic_Density": ["High"],
            "Vehicle_Type": ["Diesel Truck"],
        }
    )
    row = make_service().score(features())
    frame = make_service().score_frame(df)

    assert frame["Delay_Risk_Probability"].iloc[0] == pytest.approx(row.risk.probability)
    assert frame["Risk_Level"].iloc[0] == str(row.risk.level)
    assert frame["CO2_Emission_kg_calculated"].iloc[0] == pytest.approx(row.carbon.co2_kg)


# ---------------------------------------------------------------------------
# Provenance and the audit trail
# ---------------------------------------------------------------------------
def test_assessment_carries_the_artefact_identity():
    assessment = make_service().score(features())
    assert assessment.artifact_sha256 == "a" * 64
    assert "2026-08-04" in assessment.model_version
    assert assessment.model_version.endswith("a" * 12)


def test_every_score_is_appended_to_the_prediction_log(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    service = make_service(log=JsonlPredictionLog(log_path))

    service.score(features(distance_km=300.0))
    service.score_many([features(distance_km=800.0), features(distance_km=1100.0)])

    records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    assert len(records) == 3
    first = records[0]
    assert first["probability"] == pytest.approx(0.30)
    assert first["risk_level"] == "Medium Risk"
    # Everything needed to reconstruct the decision six months later.
    for key in ("scored_at", "model_version", "artifact_sha256",
                "high_threshold", "medium_threshold", "cost_fn_over_fp",
                "carbon_price_per_ton", "carbon_cost"):
        assert key in first


def test_a_broken_log_destination_never_breaks_scoring(tmp_path, monkeypatch):
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", explode)
    assessment = make_service(log=log).score(features())
    assert assessment.risk.probability >= 0.0
