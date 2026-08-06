"""The regression gate for migration step 3.

`src/app.py` used to score a shipment itself: load the joblib bundle, call
`predict_proba`, then `config.classify_risk` on the result. That code is gone —
the UI now asks `ScoringService` and renders what comes back.

A refactor of a scoring path is only safe if the numbers do not move. This
compares the two paths over every row of the star schema with `rtol=0`: not
"close enough", but the identical float. A one-ULP difference here would mean
the Streamlit demo and the Power BI report had begun to disagree.

`Fact_Shipments_with_ML.csv` does not carry the model's feature columns
(Vehicle_Type, Weather_Condition, Traffic_Density and Vendor_Rating live in the
dimensions), so the frame is rebuilt by the same join the training pipeline
uses. Its `Delay_Risk_Probability` column is deliberately NOT the reference:
those values were scored walk-forward, by a sequence of models trained on
earlier shipments only, and are expected to differ from a single fully-trained
artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from logistics.services.scoring import build_scoring_service
from logistics.settings import Settings

# Float-noise budget. The artefact's measured self-disagreement is ~5e-15; this
# leaves three orders of magnitude of headroom while staying far below any
# difference a genuine scoring regression could produce.
SELF_NOISE_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def analytical_frame(fact_with_ml, dim_vendor, dim_route, dim_date):
    """The fact table joined back to its dimensions — the model's input columns."""
    return (
        fact_with_ml
        .merge(dim_vendor, on="Vendor_ID", how="left")
        .merge(dim_route, on="Route_ID", how="left")
        .merge(dim_date[["Date_ID"]], on="Date_ID", how="left")
    )


@pytest.fixture(scope="module")
def service():
    # The checksum gate is a deployment concern and is exercised by
    # tests/test_model_artifact.py; here it would only couple this test to
    # whether a sidecar happens to be committed.
    return build_scoring_service(Settings.from_env(verify_artifact_checksum=False))


@pytest.fixture(scope="module")
def legacy_probabilities(model_bundle, analytical_frame):
    """Exactly what app.py did: raw bundle, raw predict_proba, model's own order."""
    model = model_bundle["model"]
    order = model_bundle["metadata"]["feature_order"]
    return model.predict_proba(analytical_frame[order])[:, 1]


def test_the_frame_covers_every_shipment(analytical_frame, fact_with_ml):
    """A silent row drop in the join would make every assertion below vacuous."""
    assert len(analytical_frame) == len(fact_with_ml) == 1500
    assert analytical_frame["Vehicle_Type"].notna().all()
    assert analytical_frame["Vendor_Rating"].notna().all()


def test_the_artefact_is_not_bit_reproducible_against_itself(model_bundle, analytical_frame):
    """Why the comparison below cannot use rtol=0 — and it is not the refactor.

    The production forest was fitted with `n_jobs=-1`, so `predict_proba`
    accumulates tree votes across threads. The reduction ORDER therefore varies
    between calls, and the last ULPs move with it. Measured on this artefact:
    calling the SAME object twice on the SAME frame disagrees on roughly 30 of
    1,500 rows by ~5e-15.

    This test states that baseline explicitly. If it ever starts failing —
    because the artefact was retrained with `n_jobs=1`, say — then rtol=0
    becomes achievable and the tolerance below should be tightened to 0.
    """
    model = model_bundle["model"]
    order = model_bundle["metadata"]["feature_order"]
    first = model.predict_proba(analytical_frame[order])[:, 1]
    second = model.predict_proba(analytical_frame[order])[:, 1]

    assert np.abs(first - second).max() < SELF_NOISE_TOLERANCE, (
        "The model's own run-to-run variation exceeded the tolerance this "
        "contract allows the refactor. Investigate the artefact, not the service."
    )


def test_probabilities_match_within_the_artefacts_own_noise(
    service, analytical_frame, legacy_probabilities, model_bundle
):
    """The refactor must not move a number further than the model moves itself.

    Stricter than a fixed tolerance: the budget is the artefact's own
    nondeterminism, measured in the same session on the same frame. A real
    regression — a changed column order, a dropped join, a different threshold
    — is orders of magnitude larger than 1e-14 and fails immediately.
    """
    model = model_bundle["model"]
    order = model_bundle["metadata"]["feature_order"]
    self_noise = np.abs(
        model.predict_proba(analytical_frame[order])[:, 1] - legacy_probabilities
    ).max()

    scored = service.score_frame(analytical_frame)
    observed = np.abs(
        scored["Delay_Risk_Probability"].to_numpy(dtype=float) - legacy_probabilities
    ).max()

    assert observed <= max(self_noise, SELF_NOISE_TOLERANCE), (
        f"ScoringService.score_frame diverged from the legacy predict_proba path "
        f"by {observed:.3e}, beyond the artefact's own run-to-run noise "
        f"({self_noise:.3e}). The Streamlit demo and the Power BI report would "
        f"begin to disagree."
    )


def test_risk_levels_are_identical(service, analytical_frame, legacy_probabilities):
    """The tiering rule must not have moved either — same thresholds, same edges."""
    scored = service.score_frame(analytical_frame)
    thresholds = service.thresholds
    expected = [
        config.classify_risk(float(p), thresholds["high_risk"], thresholds["medium_risk"])
        for p in legacy_probabilities
    ]
    assert list(scored["Risk_Level"]) == [str(level) for level in expected]


def test_co2_matches_the_deterministic_formula(service, analytical_frame):
    """CO2 is an engineering calculation, so the batch form must equal the scalar one."""
    scored = service.score_frame(analytical_frame)
    expected = [
        config.compute_co2_kg(
            float(row.Distance_km), float(row.Weight_tons),
            row.Vehicle_Type, row.Weather_Condition, row.Traffic_Density,
        )
        for row in analytical_frame.itertuples()
    ]
    np.testing.assert_allclose(
        scored["CO2_Emission_kg_calculated"].to_numpy(dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0,
        atol=0,
    )


def test_the_scored_frame_passes_through_its_input_columns(service, analytical_frame):
    """The result must be joinable straight back onto the fact table."""
    scored = service.score_frame(analytical_frame)
    assert set(analytical_frame.columns) <= set(scored.columns)
    pd.testing.assert_series_equal(
        scored["Shipment_ID"], analytical_frame["Shipment_ID"], check_names=False
    )
