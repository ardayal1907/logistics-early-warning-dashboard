"""Risk tiering and the cost arithmetic behind it.

The thresholds are not percentiles and not round numbers picked for readability.
With CALIBRATED probabilities the expected-cost-minimising decision rule is
analytic:

    intervene  <=>  p * C_FN > (1 - p) * C_FP
    p*         =    C_FP / (C_FP + C_FN)  =  1 / (1 + C_FN/C_FP)

Cost assumption behind the shipped default:
    FALSE ALARM  (FP) -> a planner checks the shipment, calls the vendor,
                         reserves buffer capacity.                   ~1 unit
    MISSED DELAY (FN) -> SLA breach: contractual penalty + expedited
                         shipping + churn risk.                      ~4 units

    C_FN/C_FP = 4  ->  p* = 0.20   (Medium Risk: act under the 4:1 assumption)
                       0.50        (High Risk: act even at 1:1)

The formula is INVALID on uncalibrated scores. Calibration is a precondition,
not a nicety - a raw Random Forest output is the share of trees voting positive,
not a probability. The shipped model reduced ECE from 0.137 to 0.023 with
isotonic regression, which is what makes this arithmetic legitimate.

The same property is what lets the optimiser put `p * penalty` directly into a
linear objective function: an expected value is only linear in a probability if
the number really is a probability.
"""

from __future__ import annotations

from logistics.domain.enums import RiskLevel

# Ratio of the cost of a missed delay to the cost of a false alarm.
#
# NOT VALIDATED against real SLA penalty schedules - it is a stated assumption,
# and the honest caveat travels with the model artefact. Overridable through
# `logistics.settings.Settings.cost_fn_over_fp`.
DEFAULT_COST_FN_OVER_FP = 4.0

# Worth intervening even if a false alarm and a missed delay cost the SAME.
DEFAULT_HIGH_RISK_THRESHOLD = 0.50

# Worth intervening under the 4:1 assumption. Derived, never typed in.
DEFAULT_MEDIUM_RISK_THRESHOLD = round(1.0 / (1.0 + DEFAULT_COST_FN_OVER_FP), 2)


def threshold_for_cost_ratio(cost_fn_over_fp: float) -> float:
    """p* = 1 / (1 + C_FN/C_FP), rounded to two decimals.

    Exposed as a function so that a caller which changes the cost assumption
    gets the matching threshold instead of having to remember the algebra.
    """
    if cost_fn_over_fp <= 0:
        raise ValueError(f"cost_fn_over_fp must be positive, got {cost_fn_over_fp}")
    return round(1.0 / (1.0 + cost_fn_over_fp), 2)


def classify_risk(
    p: float,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_RISK_THRESHOLD,
) -> RiskLevel:
    """Map a calibrated probability onto an intervention tier.

    Boundaries are inclusive on the lower edge: a probability exactly equal to
    `high_threshold` is High Risk.

    The thresholds are parameters rather than module lookups so a consumer can
    pass the values recorded in the model's own metadata. If a model is
    retrained with a different cost assumption, the consumer follows the model
    rather than this module.

    Returns a `RiskLevel`, which IS a `str` - so `== "High Risk"`, CSV writing
    and DataFrame comparisons all behave exactly as before.
    """
    if p >= high_threshold:
        return RiskLevel.HIGH
    if p >= medium_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def expected_delay_cost(
    probability: float,
    penalty_per_delay: float,
    intervention_cost: float = 0.0,
) -> float:
    """Expected monetary cost of a shipment's delay exposure.

    This is the term the optimiser minimises alongside freight and carbon cost.
    It is linear in `probability`, which is only meaningful because the
    probability is calibrated - see the module docstring.

    Args:
        probability: calibrated P(delay) for this shipment.
        penalty_per_delay: the SLA penalty a delay actually costs, in currency.
        intervention_cost: cost of pre-emptive action taken regardless of
            outcome (buffer capacity, expediting).
    """
    return probability * penalty_per_delay + intervention_cost


__all__ = [
    "DEFAULT_COST_FN_OVER_FP",
    "DEFAULT_HIGH_RISK_THRESHOLD",
    "DEFAULT_MEDIUM_RISK_THRESHOLD",
    "RiskLevel",
    "classify_risk",
    "expected_delay_cost",
    "threshold_for_cost_ratio",
]
