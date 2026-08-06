"""The carbon model — the single source of truth for emissions.

This module is the new home of the constants and the formula that previously
lived in `src/config.py`. Two things changed and nothing else:

1. It no longer knows anything about the filesystem. `config.py` mixed the CO2
   formula with `SCORED_TABLE = "Fact_Shipments_with_ML.csv"`, `sha256_file()`
   and a directory walk. That put the lowest layer in the system — the one
   everything else imports — in the business of knowing which CSV files exist.
   Move the warehouse to Delta or Postgres and the module holding the emission
   factors had to change. Those concerns now live in
   `logistics.infrastructure.fingerprint`.

2. An unknown category raises `UnknownCategoryError` instead of `KeyError`.

Nothing here imports anything outside the standard library and this package.
That is deliberate and worth preserving: a supplier portal that only needs the
carbon figure should not have to install pandas to get it.

CO2 = (distance_km x weight_tons x emission_factor + fixed_base)
      x weather_multiplier x traffic_multiplier

The result is an ENGINEERING CALCULATION, never a model prediction. The data
generator adds a N(1.0, 0.03) noise term on top for realism; this function does
not, so identical inputs always produce an identical figure.
"""

from __future__ import annotations

from collections.abc import Iterable

from logistics.domain.enums import TrafficDensity, VehicleType, WeatherCondition
from logistics.errors import UnknownCategoryError

# ---------------------------------------------------------------------------
# Emission factors
# ---------------------------------------------------------------------------
# kg CO2 per ton-kilometre. Reference ranges (approximate, informed by
# published figures):
#   Diesel truck (heavy load):  ~0.10 - 0.16
#   Electric semi-trailer:      ~0.02 - 0.045  (depends on grid carbon intensity)
#   Hybrid van:                 ~0.06 - 0.10
EMISSION_FACTOR: dict[VehicleType, float] = {
    VehicleType.DIESEL_TRUCK: 0.13,
    VehicleType.ELECTRIC_SEMI: 0.035,
    VehicleType.HYBRID_VAN: 0.08,
}

# Fixed (idle / start-up) emission component per vehicle type, in kg: the extra
# emissions from engine start-up and loading/unloading.
#
# It is tempting to say this is what gives an optimiser a reason to consolidate
# shipments. MEASURED ON THIS DATASET, THAT IS FALSE. Summed over all 1,500
# shipments (multipliers included) the fixed component is 9,137.5 kg - 0.74% of
# the 1,233.1 t total. Eliminating every single start-up would save $457/year at
# $50/ton, and realistic corridor pooling recovers only a fraction of that
# ($2/year at a one-day window, $84/year at thirty). The emission lever in this
# data is MODAL CHOICE, not consolidation: holding the observed fleet mix fixed
# and reassigning legs to vehicles by ton-km cuts 28.6% (352.5 t, $17.6k).
BASE_EMISSION: dict[VehicleType, float] = {
    VehicleType.DIESEL_TRUCK: 8.0,
    VehicleType.ELECTRIC_SEMI: 1.5,
    VehicleType.HYBRID_VAN: 4.0,
}

# Bad weather and heavy traffic -> more idling and less efficient driving.
WEATHER_CO2_MULT: dict[WeatherCondition, float] = {
    WeatherCondition.NORMAL: 1.00,
    WeatherCondition.RAIN: 1.03,
    WeatherCondition.STORM: 1.08,
    WeatherCondition.SNOW: 1.06,
}

TRAFFIC_CO2_MULT: dict[TrafficDensity, float] = {
    TrafficDensity.NORMAL_LOW: 1.00,
    TrafficDensity.MEDIUM: 1.04,
    TrafficDensity.HIGH: 1.10,
}

# Floor on the computed figure (kg), so noise can never drive it to zero or
# negative.
MIN_CO2_KG = 0.5

# ---------------------------------------------------------------------------
# Carbon pricing
# ---------------------------------------------------------------------------
# A commonly cited mid-range figure used in voluntary carbon pricing and in
# internal carbon fee models.
#
# This is the DEFAULT, not the value the system runs on. It is a jurisdiction-
# and year-dependent business parameter, so it is overridable through
# `logistics.settings.Settings.carbon_price_per_ton` (env:
# LOGISTICS_CARBON_PRICE_PER_TON). config.py said the constant was "deliberately
# isolated so it can be swapped" — but the only way to swap it was to edit the
# source and redeploy.
#
# WARNING - one copy still lives outside Python. The Power BI measure hardcodes
# it in DAX:
#
#     Carbon Tax Impact ($) = [Total CO2 Tons] * 50
#
# Power BI cannot import from Python. Until the ETL emits a Dim_Parameters table
# for the report to read, that measure must be edited by hand. Migration step 3
# closes this; tests/test_dax_constants.py pins the two together in the meantime.
DEFAULT_CARBON_PRICE_PER_TON = 50.0


def _lookup[T](table: dict[T, float], key: object, field: str) -> float:
    """Dictionary access that explains itself when it fails.

    `EMISSION_FACTOR[vehicle]` raised a bare `KeyError('CNG Truck')`, which
    Streamlit rendered as a stack trace leaking absolute file paths. The
    replacement names the field, the offending value and the accepted set.
    """
    try:
        return table[key]  # type: ignore[index]
    except KeyError:
        raise UnknownCategoryError(field, key, table.keys()) from None


def compute_co2_kg(
    distance_km: float,
    weight_tons: float,
    vehicle: str | VehicleType,
    weather: str | WeatherCondition,
    traffic: str | TrafficDensity,
) -> float:
    """Deterministic CO2 estimate for a single shipment, in kg.

    Accepts plain strings as well as enum members, so it can be called straight
    from a DataFrame row without a conversion step.

    Raises:
        UnknownCategoryError: if any category has no reference value.
    """
    factor = _lookup(EMISSION_FACTOR, vehicle, "vehicle type")
    fixed = _lookup(BASE_EMISSION, vehicle, "vehicle type")
    weather_mult = _lookup(WEATHER_CO2_MULT, weather, "weather condition")
    traffic_mult = _lookup(TRAFFIC_CO2_MULT, traffic, "traffic density")

    base = distance_km * weight_tons * factor + fixed
    return max(base * weather_mult * traffic_mult, MIN_CO2_KG)


def compute_co2_kg_many(
    distances: Iterable[float],
    weights: Iterable[float],
    vehicles: Iterable[str | VehicleType],
    weathers: Iterable[str | WeatherCondition],
    traffics: Iterable[str | TrafficDensity],
) -> list[float]:
    """Batch form of `compute_co2_kg`, for frames and counterfactual grids.

    Same formula, hoisted lookups. The scalar function above remains the single
    definition of the arithmetic; this only avoids re-resolving four dictionary
    keys and rebuilding a row object per shipment. On the optimiser's
    (shipment x vehicle) grid the row-by-row path measurably dominated
    `predict_proba`, which is the wrong thing to be slow.

    Note for the optimisation layer: emissions do not depend on the vendor, so a
    (shipment x vendor x vehicle) grid holds only |shipments| x |vehicles|
    distinct values. Compute those and join, rather than evaluating the full
    cross product.
    """
    factors = {v: (EMISSION_FACTOR[v], BASE_EMISSION[v]) for v in EMISSION_FACTOR}
    out: list[float] = []
    for distance, weight, vehicle, weather, traffic in zip(
        distances, weights, vehicles, weathers, traffics, strict=True
    ):
        try:
            factor, fixed = factors[vehicle]  # type: ignore[index]
        except KeyError:
            raise UnknownCategoryError("vehicle type", vehicle, factors.keys()) from None
        weather_mult = _lookup(WEATHER_CO2_MULT, weather, "weather condition")
        traffic_mult = _lookup(TRAFFIC_CO2_MULT, traffic, "traffic density")
        base = distance * weight * factor + fixed
        out.append(max(base * weather_mult * traffic_mult, MIN_CO2_KG))
    return out


def carbon_cost(co2_kg: float, price_per_ton: float = DEFAULT_CARBON_PRICE_PER_TON) -> float:
    """Monetary cost of an emission figure. Pure unit conversion and pricing.

    Split out of the Streamlit page, where it was one inline multiplication, so
    that the optimiser's objective function and the dashboard cannot disagree
    about what a ton of CO2 costs.
    """
    return co2_kg / 1000.0 * price_per_ton


__all__ = [
    "BASE_EMISSION",
    "DEFAULT_CARBON_PRICE_PER_TON",
    "EMISSION_FACTOR",
    "MIN_CO2_KG",
    "TRAFFIC_CO2_MULT",
    "WEATHER_CO2_MULT",
    "carbon_cost",
    "compute_co2_kg",
    "compute_co2_kg_many",
]
