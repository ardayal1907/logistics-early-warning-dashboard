"""Per-vehicle calibration and the v2 guard.

The properties asserted here are the ones the v2 formulation rests on. Two of
them exist because an earlier attempt got them wrong: the shared slope, and
never feeding counterfactual raw scores through the calibration. See
docs/OPTIMIZATION.md §6b for the measurements that ruled those attempts out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from logistics.domain.enums import VehicleType
from logistics.optimization.calibration import (
    MIN_ROWS_PER_VEHICLE,
    CalibrationDataError,
    calibrated_risk_panel,
    fit_vehicle_calibrations,
)
from logistics.optimization.guard import (
    DecisionInvarianceError,
    check_calibrated_vehicle_effect,
)

FLEET = [v.value for v in VehicleType]


def _observed(n_per_vehicle=400, *, delay_rate_by_vehicle=None, seed=7):
    """A scored-observations frame: score, vehicle, and what actually happened.

    The score carries real signal - the delay probability rises with it - so the
    calibration has a monotone relationship to find rather than pure noise.
    """
    rng = np.random.default_rng(seed)
    rates = delay_rate_by_vehicle or dict.fromkeys(FLEET, 0.20)

    rows = []
    for vehicle in FLEET:
        score = rng.uniform(0.02, 0.95, size=n_per_vehicle)
        # True probability tracks the score, centred on this vehicle's rate.
        p = np.clip(score * 2.0 * rates[vehicle] / score.mean(), 0.001, 0.999)
        delayed = rng.random(n_per_vehicle) < p
        rows.append(pd.DataFrame({
            "Shipment_ID": [f"SHP-{vehicle[:2]}-{i:05d}" for i in range(n_per_vehicle)],
            "Delay_Risk_Probability": score,
            "Vehicle_Type": vehicle,
            "Actual_Delay_Days": np.where(delayed, 2, 0),
        }))
    return pd.concat(rows, ignore_index=True)


# --- fitting ---------------------------------------------------------------

def test_every_vehicle_gets_a_calibration():
    calibrations = fit_vehicle_calibrations(_observed())
    assert sorted(calibrations) == sorted(FLEET)


def test_the_slope_is_shared_across_vehicles():
    """The property that makes v2 work.

    Letting each vehicle fit its own slope was tried and measured at 0.9957 worst
    spread: a steeper slope is more extreme in the tails, so the curves fan apart
    at high scores and the spread blows up. One slope, three intercepts.
    """
    calibrations = fit_vehicle_calibrations(_observed())
    slopes = {c.slope for c in calibrations.values()}
    assert len(slopes) == 1, f"slopes must be identical, got {slopes}"


def test_calibrated_means_reproduce_observed_delay_rates():
    """If the fit does not recover the rate it was fitted on, nothing else holds."""
    calibrations = fit_vehicle_calibrations(_observed())
    for calibration in calibrations.values():
        assert calibration.mean_calibrated == pytest.approx(
            calibration.observed_delay_rate, abs=0.02
        )


def test_a_thin_vehicle_raises_rather_than_fitting_noise():
    frame = _observed()
    keep = frame["Vehicle_Type"] != FLEET[1]
    thin = frame[frame["Vehicle_Type"] == FLEET[1]].head(MIN_ROWS_PER_VEHICLE - 1)
    with pytest.raises(CalibrationDataError, match="fewer than"):
        fit_vehicle_calibrations(pd.concat([frame[keep], thin], ignore_index=True))


def test_a_missing_outcome_column_is_a_typed_error():
    frame = _observed().drop(columns=["Actual_Delay_Days"])
    with pytest.raises(CalibrationDataError, match="Actual_Delay_Days"):
        fit_vehicle_calibrations(frame)


def test_calibration_is_monotone_in_the_raw_score():
    """Calibration may move the level; it may not reorder shipments by risk."""
    calibrations = fit_vehicle_calibrations(_observed())
    raw = np.linspace(0.01, 0.99, 50)
    for calibration in calibrations.values():
        calibrated = calibration.predict(raw)
        assert np.all(np.diff(calibrated) >= -1e-12)


def test_predict_stays_inside_zero_one_at_the_extremes():
    calibrations = fit_vehicle_calibrations(_observed())
    edges = np.array([0.0, 1e-12, 1.0 - 1e-12, 1.0])
    for calibration in calibrations.values():
        out = calibration.predict(edges)
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert not np.isnan(out).any()


# --- the panel -------------------------------------------------------------

def test_panel_has_one_row_per_shipment_and_one_column_per_vehicle():
    frame = _observed()
    calibrations = fit_vehicle_calibrations(frame)
    panel = calibrated_risk_panel(frame, calibrations)
    assert list(panel.columns) == FLEET
    assert len(panel) == len(frame)
    assert panel.index.name == "Shipment_ID"


def test_the_panel_never_asks_the_model_a_counterfactual():
    """Every vehicle column comes from the SAME observed score.

    Feeding each vehicle its own counterfactual raw score is the obvious reading
    of "score it as if it went by k" and it keeps the spread near 1.0 - the noise
    lives in those scores. So the columns must differ only by the level shift,
    which is exactly what an equal-intercept fleet collapsing to identical
    columns demonstrates.
    """
    frame = _observed(delay_rate_by_vehicle=dict.fromkeys(FLEET, 0.20))
    calibrations = fit_vehicle_calibrations(frame)
    panel = calibrated_risk_panel(frame, calibrations)

    shifts = {c.intercept for c in calibrations.values()}
    spread = (panel.max(axis=1) - panel.min(axis=1)).max()
    # Intercepts are close because the observed rates are; the spread has to be
    # of the same small order, not of the order of the raw model's swing.
    assert max(shifts) - min(shifts) < 0.5
    assert spread < 0.05


def test_a_real_level_difference_survives_calibration():
    """The guard must not be a rubber stamp.

    If one vehicle genuinely delays far more often, that difference is measured
    and the spread SHOULD grow past the bound - at which point v2's guard is
    supposed to refuse, exactly as v1's did for a different reason.
    """
    frame = _observed(delay_rate_by_vehicle={
        FLEET[0]: 0.10, FLEET[1]: 0.10, FLEET[2]: 0.55,
    })
    calibrations = fit_vehicle_calibrations(frame)
    panel = calibrated_risk_panel(frame, calibrations)
    assert (panel.max(axis=1) - panel.min(axis=1)).max() > 0.05


# --- the v2 guard ----------------------------------------------------------

def test_guard_passes_when_the_fleet_delays_alike():
    frame = _observed()
    panel = calibrated_risk_panel(frame, fit_vehicle_calibrations(frame))
    result = check_calibrated_vehicle_effect(panel)
    assert result.passed
    assert result.n_shipments == len(panel)
    assert set(result.per_vehicle_mean) == set(FLEET)


def test_guard_raises_on_a_measured_difference():
    frame = _observed(delay_rate_by_vehicle={
        FLEET[0]: 0.08, FLEET[1]: 0.08, FLEET[2]: 0.60,
    })
    panel = calibrated_risk_panel(frame, fit_vehicle_calibrations(frame))
    with pytest.raises(DecisionInvarianceError, match="measured difference"):
        check_calibrated_vehicle_effect(panel)


def test_guard_can_report_without_raising():
    frame = _observed(delay_rate_by_vehicle={
        FLEET[0]: 0.08, FLEET[1]: 0.08, FLEET[2]: 0.60,
    })
    panel = calibrated_risk_panel(frame, fit_vehicle_calibrations(frame))
    result = check_calibrated_vehicle_effect(panel, raise_on_failure=False)
    assert not result.passed
    assert result.worst_shipment_id is not None
