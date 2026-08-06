"""The controlled vocabularies of the domain.

These were previously implicit: the same three vehicle names appeared as dict
keys in `config.EMISSION_FACTOR`, as a hardcoded list in the Streamlit
`selectbox`, as `VEHICLE_TYPES` in the data generator and as one-hot columns
inside the model. Four copies, no compiler check between them.

`StrEnum` was chosen over `Enum` on purpose: every member IS a `str`, so all of
the following keep working with no conversion layer anywhere -

    EMISSION_FACTOR[VehicleType.DIESEL_TRUCK]   # enum key
    EMISSION_FACTOR["Diesel Truck"]             # plain string from a CSV
    df["Vehicle_Type"] == VehicleType.DIESEL_TRUCK
    VehicleType("Diesel Truck")                 # parse/validate

The member VALUES are the exact strings that appear in the data files and in
the model's `categorical_levels` metadata. They are a data contract, not a
naming preference: changing one invalidates the trained artefact.
"""

from __future__ import annotations

from enum import StrEnum


class VehicleType(StrEnum):
    """Fleet composition. Drives both the emission factor and the modal choice
    available to the optimiser."""

    DIESEL_TRUCK = "Diesel Truck"
    ELECTRIC_SEMI = "Electric Semi"
    HYBRID_VAN = "Hybrid Van"


class WeatherCondition(StrEnum):
    """Observed conditions for a shipment.

    Storm and Snow cannot be reliably separated by the current model - the
    training data holds 128 Storm against 193 Snow shipments. That is a data
    limitation, documented in docs/METHODOLOGY.md, not a modelling defect.
    """

    NORMAL = "Normal"
    RAIN = "Rain"
    STORM = "Storm"
    SNOW = "Snow"


class TrafficDensity(StrEnum):
    NORMAL_LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RiskLevel(StrEnum):
    """Intervention tiers.

    The boundaries are NOT percentiles. They fall out of a cost assumption:
    `p* = 1 / (1 + C_FN/C_FP)`. See logistics.domain.risk for the derivation.
    """

    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


__all__ = ["RiskLevel", "TrafficDensity", "VehicleType", "WeatherCondition"]
