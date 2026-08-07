"""Per-vehicle probability calibration for the v2 modal-allocation formulation.

WHY THIS MODULE EXISTS
----------------------
v1 asked the delay model a counterfactual it was never fitted to answer: "score
this shipment as if it went by Electric Semi". The forest obliges - it always
obliges - and the answer moves a lot. Measured on the shipped artefact, the worst
per-shipment spread under vehicle substitution was 0.6196 against a 0.05 bound,
while the GROUP MEANS across vehicle types differed by only 0.027.

That gap is the whole story. A per-row swing of 0.62 with group means that agree
to 0.03 is not a vehicle effect; it is an ensemble reading noise off a column
that carries no causal signal. `generate.build_delays` does not read Vehicle_Type
at all, so the true effect is exactly zero by construction, and v1's guard was
right to refuse: an optimiser handed those numbers would reassign the fleet to
chase differences that do not exist.

WHAT v2 DOES INSTEAD
--------------------
Rather than trusting the raw counterfactual score, each vehicle type gets its own
calibration, fitted only on the shipments that ACTUALLY travelled on it, against
what ACTUALLY happened to them:

    p_ik = calib_k( raw_score(features_i, vehicle=k) )

`calib_k` is a Platt (sigmoid) calibration with a SHARED SLOPE and a per-vehicle
intercept:

    p_ik = sigmoid( a * logit(raw_ik) + b_k )

One `a` fitted on all 1,500 shipments; one `b_k` per vehicle type. The vehicle
can therefore shift the level and nothing else, which is exactly the hypothesis
worth testing - "does this vehicle delay more often than that one, at equal
underlying risk" - and `b_k` is the measured answer.

TWO EARLIER ATTEMPTS FAILED, AND THE WAY THEY FAILED IS THE ARGUMENT FOR THIS ONE.

  isotonic, per vehicle          worst spread 1.0000  (raw model: 0.5669)
  sigmoid, per-vehicle slope     worst spread 0.9957
  sigmoid, shared slope          see docs/OPTIMIZATION.md §6 (v2)

Isotonic is a step function: two vehicles whose raw scores differ by a hair land
on opposite sides of a jump and come out 0 versus 1. Giving each vehicle its own
slope is subtler but no better - the fitted slopes came out 3.88, 5.55 and 2.96
on 848, 287 and 365 rows, and those differences are sampling noise, not physics.
A steeper slope is more extreme in the tails, so at high raw scores the three
curves fan apart and the spread blows up again.

Both failures share one cause: giving the vehicle more degrees of freedom than
the data supports lets it absorb noise. The shared slope removes the degree of
freedom that was doing the damage while keeping the one that carries the
question. The observed delay rates - 0.2017, 0.1916, 0.1945 - say the honest
answer is "barely any difference", and a model that can only express a level
shift is able to say that.

So the differences the optimiser acts on are MEASURED, not extrapolated. The
guard does not disappear in v2; it moves onto these calibrated probabilities,
where a surviving spread would be a real, observed difference rather than forest
noise. See docs/OPTIMIZATION.md §6 (v2).

WHAT THIS IS NOT
----------------
This does not make the model causal. Vehicle assignment in the generator
correlates with distance, weight and vendor (v2 generator), so part of any
residual difference is confounding, not the vehicle. Isotonic calibration on
observational subsets cannot separate those. It bounds how far the optimiser can
run with a number; it does not license a causal reading of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from logistics.domain.enums import VehicleType
from logistics.errors import LogisticsError

logger = logging.getLogger(__name__)

# Raw scores are clipped away from 0 and 1 before the logit, which is otherwise
# infinite. 1e-6 keeps the transformed range at about +/-14, well inside float64.
_PROB_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), _PROB_EPS, 1.0 - _PROB_EPS)
    transformed: np.ndarray = np.log(clipped / (1.0 - clipped))
    return transformed


# Below this many observed shipments a per-vehicle fit is noise of a
# different kind - it would interpolate a handful of points and hand the
# optimiser a confident number built on nothing.
MIN_ROWS_PER_VEHICLE = 100

# The outcome column. A shipment is late if it lost any whole day.
OUTCOME_COLUMN = "Actual_Delay_Days"


class CalibrationDataError(LogisticsError):
    """Not enough observed data to calibrate one of the vehicle types."""


@dataclass(frozen=True)
class VehicleCalibration:
    """One vehicle's sigmoid map, plus the evidence behind it.

    `slope` is shared across all vehicles by construction; only `intercept`
    varies, and the difference between two intercepts IS the measured vehicle
    effect, in log-odds.
    """

    vehicle: str
    n_rows: int
    observed_delay_rate: float
    mean_raw_score: float
    mean_calibrated: float
    slope: float
    intercept: float

    def predict(self, raw: np.ndarray) -> np.ndarray:
        """P(delay) for this vehicle, given the model's raw score."""
        z = self.slope * _logit(raw) + self.intercept
        probability: np.ndarray = 1.0 / (1.0 + np.exp(-z))
        return probability.astype(float)


def fit_vehicle_calibrations(
    scored_observed: pd.DataFrame,
    *,
    score_column: str = "Delay_Risk_Probability",
    min_rows: int = MIN_ROWS_PER_VEHICLE,
) -> dict[str, VehicleCalibration]:
    """Fit one isotonic calibration per vehicle type on observed outcomes.

    `scored_observed` must carry, for shipments AS THEY ACTUALLY RAN, the model's
    score, the vehicle they ran on, and `Actual_Delay_Days`.

    Raises:
        CalibrationDataError: when a vehicle type has fewer than `min_rows`
            observed shipments, or the outcome column is missing.
    """
    if OUTCOME_COLUMN not in scored_observed.columns:
        raise CalibrationDataError(
            f"{OUTCOME_COLUMN} is required to calibrate against observed outcomes; "
            f"got columns {sorted(scored_observed.columns)}."
        )

    fleet = [v.value for v in VehicleType]
    y = (scored_observed[OUTCOME_COLUMN].to_numpy() > 0).astype(float)
    raw = scored_observed[score_column].to_numpy(dtype=float)
    vehicles = scored_observed["Vehicle_Type"].to_numpy()

    counts = {k: int((vehicles == k).sum()) for k in fleet}
    thin = {k: n for k, n in counts.items() if n < min_rows}
    if thin:
        raise CalibrationDataError(
            f"Vehicle types {thin} have fewer than {min_rows} observed shipments, "
            f"too few for a per-vehicle calibration. Either widen the sample or "
            f"drop the vehicle from the fleet."
        )

    # Design: [logit(raw), one dummy per vehicle except the reference]. The single
    # coefficient on logit(raw) is the shared slope; the dummies move the
    # intercept only. Reference level is the first vehicle in the fleet order.
    reference, others = fleet[0], fleet[1:]
    design = np.column_stack(
        [_logit(raw)] + [(vehicles == k).astype(float) for k in others]
    )

    # C large => effectively unpenalised. With four parameters on 1,500 rows
    # there is nothing to regularise, and shrinkage would flatten a real level
    # difference as readily as a spurious one.
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(design, y)

    slope = float(platt.coef_[0][0])
    base_intercept = float(platt.intercept_[0])
    shifts = {reference: 0.0} | {
        k: float(platt.coef_[0][1 + i]) for i, k in enumerate(others)
    }

    calibrations: dict[str, VehicleCalibration] = {}
    for vehicle in fleet:
        mask = vehicles == vehicle
        calibration = VehicleCalibration(
            vehicle=vehicle,
            n_rows=counts[vehicle],
            observed_delay_rate=float(y[mask].mean()),
            mean_raw_score=float(raw[mask].mean()),
            mean_calibrated=0.0,          # filled below, needs .predict
            slope=slope,
            intercept=base_intercept + shifts[vehicle],
        )
        fitted_mean = float(calibration.predict(raw[mask]).mean())
        calibrations[vehicle] = replace(calibration, mean_calibrated=fitted_mean)

        logger.info(
            "Calibrated %s on %d observed shipments: delay rate %.4f, "
            "mean raw %.4f -> mean calibrated %.4f (shared slope %.3f, "
            "intercept %.3f, level shift %+.3f log-odds)",
            vehicle, counts[vehicle], y[mask].mean(), raw[mask].mean(),
            fitted_mean, slope, calibration.intercept, shifts[vehicle],
        )

    return calibrations


def calibrated_risk_panel(
    observed_scored: pd.DataFrame,
    calibrations: dict[str, VehicleCalibration],
    *,
    score_column: str = "Delay_Risk_Probability",
) -> pd.DataFrame:
    """Build the shipment x vehicle panel of p_ik the optimiser consumes.

    THE COUNTERFACTUAL IS NOT ASKED OF THE FOREST. `observed_scored` carries one
    row per shipment, scored AS IT ACTUALLY RAN, and every vehicle column is
    derived from that single score by applying that vehicle's calibration:

        p_ik = sigmoid( a * logit(s_i) + b_k )

    so the only thing the vehicle changes is the measured level shift b_k.

    This is the correction that makes v2 work, and it took three attempts to
    find. Feeding each vehicle's own counterfactual raw score through the
    calibration - the obvious reading of "score it as if it went by k" - keeps
    the spread near 1.0 no matter how well behaved the calibration is, because
    the noise lives in those counterfactual scores and the shared slope of 3.77
    then amplifies it. Measured: raw counterfactual spread 0.5669, and putting it
    through a correctly fitted calibration gave 0.9876.

    Asking the forest what a Diesel Truck shipment would have scored as an
    Electric Semi is an out-of-distribution question about a column with no
    causal role in the outcome. The honest answer is "the same underlying risk,
    shifted by whatever the observed rates say", and that is what this computes.
    """
    scores = observed_scored.set_index("Shipment_ID")[score_column].astype(float)
    panel = pd.DataFrame(
        {
            vehicle: calibrations[vehicle].predict(scores.to_numpy())
            for vehicle in (v.value for v in VehicleType)
        },
        index=scores.index,
    )
    panel.index.name = "Shipment_ID"
    return panel


__all__ = [
    "MIN_ROWS_PER_VEHICLE",
    "OUTCOME_COLUMN",
    "CalibrationDataError",
    "VehicleCalibration",
    "calibrated_risk_panel",
    "fit_vehicle_calibrations",
]
