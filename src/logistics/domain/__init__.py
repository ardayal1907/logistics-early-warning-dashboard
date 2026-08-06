"""Pure business rules. No I/O, no framework, no dependency outside stdlib +
pydantic.

Everything an auditor would want to challenge about this system lives here: the
emission factors, the carbon price, the cost ratio behind the risk tiers and the
threshold arithmetic they imply. Keeping them free of infrastructure is what
makes them reviewable by someone who does not read Python plumbing.
"""

from __future__ import annotations

from logistics.domain.carbon import (
    BASE_EMISSION,
    DEFAULT_CARBON_PRICE_PER_TON,
    EMISSION_FACTOR,
    MIN_CO2_KG,
    TRAFFIC_CO2_MULT,
    WEATHER_CO2_MULT,
    carbon_cost,
    compute_co2_kg,
)
from logistics.domain.enums import (
    RiskLevel,
    TrafficDensity,
    VehicleType,
    WeatherCondition,
)
from logistics.domain.models import (
    CarbonFootprint,
    RiskAssessment,
    ShipmentAssessment,
    ShipmentFeatures,
)
from logistics.domain.risk import (
    DEFAULT_COST_FN_OVER_FP,
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_MEDIUM_RISK_THRESHOLD,
    classify_risk,
    expected_delay_cost,
    threshold_for_cost_ratio,
)

__all__ = [
    "BASE_EMISSION",
    "DEFAULT_CARBON_PRICE_PER_TON",
    "DEFAULT_COST_FN_OVER_FP",
    "DEFAULT_HIGH_RISK_THRESHOLD",
    "DEFAULT_MEDIUM_RISK_THRESHOLD",
    "EMISSION_FACTOR",
    "MIN_CO2_KG",
    "TRAFFIC_CO2_MULT",
    "WEATHER_CO2_MULT",
    "CarbonFootprint",
    "RiskAssessment",
    "RiskLevel",
    "ShipmentAssessment",
    "ShipmentFeatures",
    "TrafficDensity",
    "VehicleType",
    "WeatherCondition",
    "carbon_cost",
    "classify_risk",
    "compute_co2_kg",
    "expected_delay_cost",
    "threshold_for_cost_ratio",
]
