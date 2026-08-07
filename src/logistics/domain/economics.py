"""Freight tariffs and penalty assumptions.

EVERY NUMBER IN THIS MODULE IS AN ASSUMPTION, NOT A MEASUREMENT.

That is the single most important thing about it. `logistics.domain.carbon`
holds an engineering calculation and `logistics.domain.risk` holds a derivation;
this module holds business inputs that nobody has measured. docs/OPTIMIZATION.md
calls it W1: "half the objective is not in the data ... a single '$X saved'
claim is unfalsifiable and therefore worthless."

The values are here, in one place, with their provenance stated, so that:

  * a reader can see instantly which figures are evidence and which are guesses;
  * a sensitivity sweep can vary them without editing source;
  * `LOGISTICS_*` can override them in a deployment that HAS measured them.

The only monetary parameter in this system with any grounding is
`DEFAULT_CARBON_PRICE_PER_TON` in `carbon.py`, and even that is a mid-range
convention rather than a jurisdiction's tax.
"""

from __future__ import annotations

from logistics.domain.enums import VehicleType

# ---------------------------------------------------------------------------
# Freight cost per vehicle type
# ---------------------------------------------------------------------------
# ASSUMPTION. Cost of moving one tonne one kilometre, in dollars.
#
# The ordering (electric cheapest per ton-km on energy, diesel cheapest on
# fixed cost) is the qualitative relationship a 3PL would recognise; the
# magnitudes are invented. If these are replaced with a real rate card, the
# Pareto frontier moves and every headline in docs/OPTIMIZATION.md must be
# recomputed.
COST_PER_TON_KM: dict[VehicleType, float] = {
    VehicleType.DIESEL_TRUCK: 0.085,
    VehicleType.ELECTRIC_SEMI: 0.062,
    VehicleType.HYBRID_VAN: 0.098,
}

# ASSUMPTION. Fixed cost of dispatching one leg, in dollars: driver call-out,
# loading, and the amortised capital difference between the fleets. The
# electric semi carries the highest fixed cost because its capital cost is
# highest; that is what stops "electrify everything" being free.
FIXED_COST_PER_LEG: dict[VehicleType, float] = {
    VehicleType.DIESEL_TRUCK: 42.0,
    VehicleType.ELECTRIC_SEMI: 96.0,
    VehicleType.HYBRID_VAN: 28.0,
}

# ---------------------------------------------------------------------------
# Service-level penalty
# ---------------------------------------------------------------------------
# ASSUMPTION, and the one that matters most. `Pi` converts a probability into
# currency, which is only legitimate because the probabilities are calibrated
# (see logistics.domain.risk). The ratio Pi / freight cost determines the
# optimum in Block A; it has never been measured against a real SLA schedule.
#
# Not used by Phase 1 (modal choice carries no ML), and defined here so that
# Phase 2 does not invent it inline.
DEFAULT_SLA_PENALTY_PER_DELAY = 850.0


def freight_cost(
    distance_km: float,
    weight_tons: float,
    vehicle: str | VehicleType,
    *,
    cost_per_ton_km: dict[VehicleType, float] | None = None,
    fixed_cost_per_leg: dict[VehicleType, float] | None = None,
) -> float:
    """Cost of carrying one shipment on one vehicle type, in dollars.

    `distance x tonnage x rate + fixed`. Deliberately the same shape as
    `carbon.compute_co2_kg` so the two axes of the Pareto frontier are
    computed by structurally identical functions - a difference between them
    should be a difference in the parameters, not in the arithmetic.

    Raises:
        UnknownCategoryError: if the vehicle type has no rate.
    """
    from logistics.domain.carbon import _lookup  # local: avoids a cycle

    rates = cost_per_ton_km or COST_PER_TON_KM
    fixed = fixed_cost_per_leg or FIXED_COST_PER_LEG

    rate = _lookup(rates, vehicle, "vehicle type")
    base = _lookup(fixed, vehicle, "vehicle type")
    return distance_km * weight_tons * rate + base


__all__ = [
    "COST_PER_TON_KM",
    "DEFAULT_SLA_PENALTY_PER_DELAY",
    "FIXED_COST_PER_LEG",
    "freight_cost",
]
