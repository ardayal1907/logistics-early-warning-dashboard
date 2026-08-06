"""The mandatory guard: does Vehicle_Type leak into the delay probability?

docs/OPTIMIZATION.md §6 makes this the first task of Phase 1, and it is not a
formality. Permutation importance puts `Vehicle_Type` at 0.007 +/- 0.016 -
statistically indistinguishable from zero - and the generator confirms it has
no causal role. But the model still CONSUMES it as a feature, and a solver
mines every spurious signal in its objective to exhaustion: it would discover
that switching vehicles "reduces risk" and prevent no delays whatsoever.

A near-zero average importance does not bound the per-row spread, so the spread
is asserted directly. If this check fails, the options are to drop the feature
and retrain, or to accept the dependency and reformulate with `y_ijk`.
Continuing quietly is not one of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from logistics.domain.enums import VehicleType
from logistics.errors import LogisticsError
from logistics.services.scoring import ScoringService

logger = logging.getLogger(__name__)


class DecisionInvarianceError(LogisticsError):
    """The modal decision moves the delay probability.

    A `LogisticsError` rather than a bare ValueError so the CLI reports it as
    a typed, expected failure (exit 1) instead of an unhandled crash (exit 2).
    It IS an expected outcome: on the current artefact it is what happens.
    """

# The bound from docs/OPTIMIZATION.md §6. A shipment's probability may move by
# less than this when only its vehicle changes; more, and the modal decision is
# entangled with the risk model.
MAX_VEHICLE_LEAK = 0.05


@dataclass(frozen=True)
class DecisionInvarianceResult:
    """What the guard measured, whether or not it passed."""

    leak: float
    threshold: float
    n_shipments: int
    worst_shipment_id: str | None
    per_vehicle_mean: dict[str, float]

    @property
    def passed(self) -> bool:
        return self.leak < self.threshold

    def describe(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        means = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.per_vehicle_mean.items()))
        return (
            f"[{verdict}] Vehicle_Type -> p spread = {self.leak:.4f} "
            f"(threshold {self.threshold}), worst shipment "
            f"{self.worst_shipment_id}, n={self.n_shipments}. Group means: {means}"
        )


def build_counterfactual_grid(
    shipments: pd.DataFrame, vehicles: list[str] | None = None
) -> pd.DataFrame:
    """One row per (shipment, vehicle type): what if this leg went by k?

    Everything except `Vehicle_Type` is held fixed, which is what makes the
    spread attributable to the vehicle alone.
    """
    fleet = vehicles or [v.value for v in VehicleType]
    frames = []
    for vehicle in fleet:
        counterfactual = shipments.copy()
        counterfactual["Vehicle_Type"] = vehicle
        frames.append(counterfactual)
    return pd.concat(frames, ignore_index=True)


def check_vehicle_invariance(
    service: ScoringService,
    shipments: pd.DataFrame,
    *,
    threshold: float = MAX_VEHICLE_LEAK,
    raise_on_failure: bool = True,
) -> DecisionInvarianceResult:
    """Score every shipment under every vehicle and measure the spread.

    The statistic is the WORST per-shipment range, not the average: an
    optimiser exploits the worst row, not the mean one.

    Raises:
        DecisionInvarianceError: when the spread exceeds `threshold` and
            `raise_on_failure` is set.
    """
    grid = build_counterfactual_grid(shipments)
    scored = service.score_frame(grid)

    by_shipment = scored.groupby("Shipment_ID")["Delay_Risk_Probability"]
    spread = by_shipment.max() - by_shipment.min()
    leak = float(spread.max())

    result = DecisionInvarianceResult(
        leak=leak,
        threshold=threshold,
        n_shipments=int(scored["Shipment_ID"].nunique()),
        worst_shipment_id=str(spread.idxmax()) if len(spread) else None,
        per_vehicle_mean={
            str(k): float(v)
            for k, v in scored.groupby("Vehicle_Type")["Delay_Risk_Probability"]
            .mean()
            .items()
        },
    )

    logger.info("%s", result.describe())
    if not result.passed and raise_on_failure:
        raise DecisionInvarianceError(
            f"Vehicle_Type leaks into the delay probability: worst per-shipment "
            f"spread {leak:.4f} >= {threshold}. Block B assumes the modal decision "
            f"does not move p. Either drop Vehicle_Type from the model and retrain, "
            f"or reformulate the optimiser with y_ijk. See docs/OPTIMIZATION.md §6."
        )
    return result


__all__ = [
    "MAX_VEHICLE_LEAK",
    "DecisionInvarianceError",
    "DecisionInvarianceResult",
    "build_counterfactual_grid",
    "check_vehicle_invariance",
]
