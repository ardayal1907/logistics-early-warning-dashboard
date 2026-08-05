"""The saved model must be loadable and usable exactly as the Streamlit app uses it.

"I saved a model" and "the deployment can score with it" are different claims. These
tests close the gap: load the pickle from disk, hand it a raw DataFrame (no manual
encoding), and check the metadata contract the app depends on.
"""

import json

import pandas as pd
import pytest

SAMPLE = {
    "Distance_km": 450.0,
    "Weight_tons": 12.0,
    "Vendor_Rating": 4.0,
    "Weather_Condition": "Storm",
    "Traffic_Density": "High",
    "Vehicle_Type": "Diesel Truck",
}


def test_bundle_has_model_and_metadata(model_bundle):
    assert set(model_bundle) == {"model", "metadata"}


def test_predicts_from_a_raw_dataframe(model_bundle):
    """No manual encoding - the whole pipeline travels inside the pickle."""
    model, meta = model_bundle["model"], model_bundle["metadata"]
    X = pd.DataFrame([SAMPLE])[meta["feature_order"]]
    p = float(model.predict_proba(X)[0, 1])
    assert 0.0 <= p <= 1.0


def test_survives_an_unseen_category(model_bundle):
    """handle_unknown='ignore' must keep production alive when a new value appears."""
    model, meta = model_bundle["model"], model_bundle["metadata"]
    row = dict(SAMPLE, Weather_Condition="Fog")   # never seen in training
    p = float(model.predict_proba(pd.DataFrame([row])[meta["feature_order"]])[0, 1])
    assert 0.0 <= p <= 1.0


def test_ranks_a_bad_shipment_above_a_good_one(model_bundle):
    model, meta = model_bundle["model"], model_bundle["metadata"]
    good = dict(SAMPLE, Vendor_Rating=5.0, Weather_Condition="Normal",
                Traffic_Density="Low", Distance_km=100.0)
    bad = dict(SAMPLE, Vendor_Rating=2.5, Weather_Condition="Snow",
               Traffic_Density="High", Distance_km=1100.0)
    X = pd.DataFrame([good, bad])[meta["feature_order"]]
    p_good, p_bad = model.predict_proba(X)[:, 1]
    assert p_good < p_bad, (
        f"The model scored the favourable shipment ({p_good:.3f}) at or above the "
        f"unfavourable one ({p_bad:.3f})."
    )


def test_metadata_contract_the_app_relies_on(model_bundle):
    meta = model_bundle["metadata"]
    for key in ("feature_order", "thresholds", "metrics", "categorical_levels",
                "training_data", "versions", "caveats"):
        assert key in meta, f"metadata is missing '{key}'"
    for key in ("high_risk", "medium_risk", "cost_fn_over_fp"):
        assert key in meta["thresholds"], f"thresholds is missing '{key}'"
    assert len(meta["feature_order"]) == 6


def test_metadata_json_matches_the_pickle(model_bundle, repo_root):
    """The human-readable copy must not drift from the embedded one."""
    on_disk = json.loads(
        (repo_root / "models" / "production_risk_model_metadata.json")
        .read_text(encoding="utf-8"))
    assert on_disk == model_bundle["metadata"]


def test_metadata_thresholds_agree_with_config(model_bundle):
    import config
    th = model_bundle["metadata"]["thresholds"]
    assert th["high_risk"] == pytest.approx(config.HIGH_RISK_THRESHOLD)
    assert th["medium_risk"] == pytest.approx(config.MEDIUM_RISK_THRESHOLD)
    assert th["cost_fn_over_fp"] == pytest.approx(config.COST_FN_OVER_FP)
