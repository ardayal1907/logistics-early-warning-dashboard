"""Pandera schemas for the shipment tables.

Column-level rules (types, ranges, allowed categories) are ordinary Pandera
checks. The one rule that is not is `validate_distance_consistency`: it is a
CROSS-COLUMN invariant — the distance between two cities does not depend on
which shipment you happen to be looking at — and it is the rule that catches
the defect described below.

    THIS RULE FAILS ON TODAY'S DATA, AND IT IS RIGHT TO FAIL.

`generate.build_base_columns` draws Distance_km independently of the
(Origin, Destination) pair, so Istanbul -> Denizli carries distances from
67.9 km to 1,169.2 km across six shipments. Any model that uses distance as a
proxy for route difficulty is learning noise, and any optimiser that plans on
these numbers is planning against a world that does not exist. The fix is a
20x20 distance matrix with +/-5% noise in the generator; until that lands, the
test that asserts this rule is expected to be red.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from logistics.domain.enums import RiskLevel, TrafficDensity, VehicleType, WeatherCondition

# Derived, never retyped - see the module docstring in logistics.contracts.
VEHICLE_TYPES = [v.value for v in VehicleType]
WEATHER_CONDITIONS = [w.value for w in WeatherCondition]
TRAFFIC_DENSITIES = [t.value for t in TrafficDensity]
RISK_LEVELS = [r.value for r in RiskLevel]

# The bound below which within-pair distance spread is treated as noise rather
# than as a broken generator. A real road network varies by routing choice and
# by which depot served the leg; 15% is generous for that, and the observed
# violation is an order of magnitude larger.
MAX_DISTANCE_CV = 0.15

# Content-addressed keys, RT- plus the first 10 hex characters of sha1 over the
# natural key. The transitional alternative that also accepted positional
# RT-00001 keys is GONE: the pipeline has been re-run, data/processed/ carries
# durable keys throughout, and keeping both forms accepted would have meant the
# contract could no longer tell a durable key from the defect it replaced.
ROUTE_ID_PATTERN = r"^RT-[0-9a-f]{10}$"

# Kept as a distinct name because callers reference it; identical by definition
# now that the transition is complete.
DURABLE_ROUTE_ID_PATTERN = ROUTE_ID_PATTERN


def _positive(name: str, maximum: float | None = None) -> Column:
    checks = [Check.gt(0, error=f"{name} must be positive")]
    if maximum is not None:
        checks.append(Check.le(maximum, error=f"{name} exceeds {maximum}"))
    return Column(float, checks, nullable=False)


RAW_SHIPMENTS_SCHEMA = DataFrameSchema(
    {
        "Shipment_ID": Column(str, Check.str_matches(r"^SHP-\d{5}$"), unique=True),
        "Shipment_Date": Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "Vendor_ID": Column(str, Check.str_matches(r"^VEND-\d{3}$")),
        "Vendor_Rating": Column(float, [Check.ge(1.0), Check.le(5.0)]),
        "Origin": Column(str),
        "Destination": Column(str),
        "Distance_km": _positive("Distance_km", maximum=5000.0),
        "Weight_tons": _positive("Weight_tons", maximum=100.0),
        "Weather_Condition": Column(str, Check.isin(WEATHER_CONDITIONS)),
        "Traffic_Density": Column(str, Check.isin(TRAFFIC_DENSITIES)),
        "Vehicle_Type": Column(str, Check.isin(VEHICLE_TYPES)),
        "Actual_Delay_Days": Column(int, [Check.ge(0), Check.le(30)]),
        "CO2_Emission_kg": _positive("CO2_Emission_kg"),
    },
    strict=False,
    coerce=True,
    name="raw_shipments",
)

DIM_VENDOR_SCHEMA = DataFrameSchema(
    {
        "Vendor_ID": Column(str, Check.str_matches(r"^VEND-\d{3}$"), unique=True),
        "Vendor_Rating": Column(float, [Check.ge(1.0), Check.le(5.0)]),
    },
    strict=False,
    coerce=True,
    name="dim_vendor",
)

DIM_ROUTE_SCHEMA = DataFrameSchema(
    {
        "Route_ID": Column(str, Check.str_matches(ROUTE_ID_PATTERN), unique=True),
        "Origin": Column(str),
        "Destination": Column(str),
        "Vehicle_Type": Column(str, Check.isin(VEHICLE_TYPES)),
        "Weather_Condition": Column(str, Check.isin(WEATHER_CONDITIONS)),
        "Traffic_Density": Column(str, Check.isin(TRAFFIC_DENSITIES)),
    },
    strict=False,
    coerce=True,
    name="dim_route",
)

FACT_SHIPMENTS_SCHEMA = DataFrameSchema(
    {
        "Shipment_ID": Column(str, Check.str_matches(r"^SHP-\d{5}$"), unique=True),
        "Date_ID": Column(int, Check.in_range(19000101, 99991231)),
        "Vendor_ID": Column(str, Check.str_matches(r"^VEND-\d{3}$")),
        "Route_ID": Column(str, Check.str_matches(ROUTE_ID_PATTERN)),
        "Weight_tons": _positive("Weight_tons", maximum=100.0),
        "Distance_km": _positive("Distance_km", maximum=5000.0),
        "Actual_Delay_Days": Column(int, [Check.ge(0), Check.le(30)]),
        "CO2_Emission_kg": _positive("CO2_Emission_kg"),
    },
    strict=False,
    coerce=True,
    name="fact_shipments",
)

# The scored table adds two columns and must keep every guarantee above. A
# probability outside [0, 1] means the calibration layer is broken, and a
# Risk_Level outside the vocabulary means Power BI measures silently return
# blank rather than erroring.
SCORED_SHIPMENTS_SCHEMA = FACT_SHIPMENTS_SCHEMA.add_columns(
    {
        "Delay_Risk_Probability": Column(float, [Check.ge(0.0), Check.le(1.0)]),
        "Risk_Level": Column(str, Check.isin(RISK_LEVELS)),
    }
)
SCORED_SHIPMENTS_SCHEMA.name = "scored_shipments"


# ---------------------------------------------------------------------------
# The cross-column rule
# ---------------------------------------------------------------------------
def distance_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """Within-pair coefficient of variation of Distance_km, worst pair first.

    Columns: Origin, Destination, n, min_km, max_km, mean_km, cv.

    Pairs seen only once are excluded: a single observation has no spread, so
    including them would dilute the statistic with rows that cannot fail.
    """
    grouped = df.groupby(["Origin", "Destination"])["Distance_km"]
    report = grouped.agg(
        n="size", min_km="min", max_km="max", mean_km="mean", std_km="std"
    ).reset_index()
    report = report[report["n"] > 1].copy()
    report["cv"] = report["std_km"] / report["mean_km"]
    return report.sort_values("cv", ascending=False).reset_index(drop=True)


def validate_distance_consistency(
    df: pd.DataFrame, max_cv: float = MAX_DISTANCE_CV
) -> pd.DataFrame:
    """Raise unless every city pair's distance spread is within `max_cv`.

    Returns the offending rows on success-by-emptiness so a caller can log
    them; raises `ValueError` when the bound is exceeded.
    """
    report = distance_consistency_report(df)
    violations = report[report["cv"] > max_cv]
    if not violations.empty:
        worst = violations.iloc[0]
        raise ValueError(
            f"Distance_km is not a function of (Origin, Destination): "
            f"{len(violations)} city pair(s) exceed cv={max_cv}. Worst is "
            f"{worst['Origin']} -> {worst['Destination']}: {int(worst['n'])} "
            f"shipments spanning {worst['min_km']:.1f}-{worst['max_km']:.1f} km "
            f"(cv={worst['cv']:.2f}). The generator draws distance independently "
            f"of the city pair; fix is a 20x20 distance matrix with +/-5% noise."
        )
    return violations


__all__ = [
    "DIM_ROUTE_SCHEMA",
    "DIM_VENDOR_SCHEMA",
    "DURABLE_ROUTE_ID_PATTERN",
    "FACT_SHIPMENTS_SCHEMA",
    "MAX_DISTANCE_CV",
    "RAW_SHIPMENTS_SCHEMA",
    "ROUTE_ID_PATTERN",
    "SCORED_SHIPMENTS_SCHEMA",
    "distance_consistency_report",
    "validate_distance_consistency",
]
