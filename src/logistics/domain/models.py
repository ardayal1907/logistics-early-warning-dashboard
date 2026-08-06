"""The data contracts that cross every layer boundary.

Until now the only thing standing between the model and a nonsensical input was
the range of a Streamlit slider. That is not a validation layer - it is a UI
affordance that disappears the moment the same scoring logic is called from a
REST endpoint, a batch job or the optimiser.

Worse, the sliders and the model disagreed. `app.py` offered Vendor_Rating in
[2.5, 5.0]; the model's own metadata records the training range as
[1.71, 4.89]. So the interface simultaneously

  * blocked the worst vendors (1.71-2.50) - precisely the high-risk shipments
    the early-warning panel exists to catch, and
  * invited extrapolation into 4.89-5.00, where no training data exists.

The split below is deliberate:

  * `ShipmentFeatures` rejects the PHYSICALLY IMPOSSIBLE (negative distance,
    a probability above 1). These bounds are static and safe to hardcode.
  * `check_extrapolation()` reports values outside the range the model was
    actually trained on. Those are not errors - scoring a slightly out-of-range
    shipment is legitimate - but the answer carries lower confidence and the
    caller is told so instead of finding out never.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from logistics.domain.enums import RiskLevel, TrafficDensity, VehicleType, WeatherCondition

# Physical / contractual bounds. Wide on purpose - this layer rejects the
# impossible, not the unusual. Narrowing to the training range happens in
# check_extrapolation(), driven by the model's metadata rather than by a
# constant that would go stale on the next retrain.
Distance = Annotated[float, Field(gt=0, le=5000, description="Leg distance in km")]
Weight = Annotated[float, Field(gt=0, le=60, description="Payload in metric tons")]
Rating = Annotated[float, Field(ge=1.0, le=5.0, description="Vendor scorecard, 1-5")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class ShipmentFeatures(BaseModel):
    """Exactly the six inputs the model consumes, validated.

    Field names match the training columns one-for-one. `to_model_row()` is the
    only place that maps them into the model's expected column order, and it
    reads that order from the artefact's metadata rather than hardcoding it -
    the same discipline app.py already applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    distance_km: Distance
    weight_tons: Weight
    vendor_rating: Rating
    weather_condition: WeatherCondition
    traffic_density: TrafficDensity
    vehicle_type: VehicleType

    # Optional identifiers. Not model inputs - they exist so a prediction can be
    # traced back to a shipment in the audit log and joined to the star schema.
    shipment_id: str | None = None
    vendor_id: str | None = None
    route_id: str | None = None

    def to_model_row(self, feature_order: list[str]) -> dict[str, Any]:
        """Project onto the training column names, in the model's own order."""
        row = {
            "Distance_km": self.distance_km,
            "Weight_tons": self.weight_tons,
            "Vendor_Rating": self.vendor_rating,
            "Weather_Condition": str(self.weather_condition),
            "Traffic_Density": str(self.traffic_density),
            "Vehicle_Type": str(self.vehicle_type),
        }
        missing = [c for c in feature_order if c not in row]
        if missing:
            raise ValueError(
                f"The model expects feature(s) this contract cannot supply: {missing}. "
                f"ShipmentFeatures and the trained artefact have diverged."
            )
        return {name: row[name] for name in feature_order}

    def check_extrapolation(self, numeric_ranges: dict[str, dict[str, float]]) -> list[str]:
        """Which numeric inputs fall outside the model's training range.

        Returns human-readable descriptions; an empty list means every value is
        interpolation. `numeric_ranges` comes from the artefact metadata, so the
        check follows the model automatically after a retrain.
        """
        observed = {
            "Distance_km": self.distance_km,
            "Weight_tons": self.weight_tons,
            "Vendor_Rating": self.vendor_rating,
        }
        warnings: list[str] = []
        for column, value in observed.items():
            bounds = numeric_ranges.get(column)
            if not bounds:
                continue
            low, high = bounds.get("min"), bounds.get("max")
            if low is not None and value < low:
                warnings.append(f"{column}={value:g} is below the training minimum {low:g}")
            elif high is not None and value > high:
                warnings.append(f"{column}={value:g} is above the training maximum {high:g}")
        return warnings


class CarbonFootprint(BaseModel):
    """A deterministic emission figure and its price. NOT a model output."""

    model_config = ConfigDict(frozen=True)

    co2_kg: float = Field(ge=0)
    price_per_ton: float = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def co2_tons(self) -> float:
        return self.co2_kg / 1000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        return self.co2_tons * self.price_per_ton


class RiskAssessment(BaseModel):
    """A calibrated probability, the tier it falls in, and the rule that decided.

    The thresholds travel WITH the assessment rather than being looked up later.
    A stored prediction whose thresholds are not recorded cannot be audited: six
    months on, nobody can tell whether a shipment was flagged because it was
    risky or because the cost assumption had changed that week.
    """

    model_config = ConfigDict(frozen=True)

    probability: Probability
    level: RiskLevel
    high_threshold: Probability
    medium_threshold: Probability
    cost_fn_over_fp: float = Field(gt=0)


class ShipmentAssessment(BaseModel):
    """The complete answer for one shipment: risk, carbon and provenance.

    `model_version` and `artifact_sha256` are what make the record auditable.
    Without them a stored score is a number with no way back to the code and
    weights that produced it.
    """

    model_config = ConfigDict(frozen=True)

    features: ShipmentFeatures
    risk: RiskAssessment
    carbon: CarbonFootprint

    model_name: str
    model_version: str
    artifact_sha256: str | None = None
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Non-fatal: the score is still returned, the caller is simply told the
    # inputs left the region the model has evidence for.
    extrapolation_warnings: tuple[str, ...] = ()

    @property
    def is_extrapolated(self) -> bool:
        return bool(self.extrapolation_warnings)

    def to_audit_record(self) -> dict[str, Any]:
        """A flat, append-only record for the prediction log.

        Deliberately flat: log aggregators index scalar fields, and a nested
        structure makes the eventual drift query (`percentile of Distance_km
        over the last 30 days`) far harder than it needs to be.
        """
        return {
            "scored_at": self.scored_at.isoformat(timespec="seconds"),
            "shipment_id": self.features.shipment_id,
            "vendor_id": self.features.vendor_id,
            "route_id": self.features.route_id,
            "distance_km": self.features.distance_km,
            "weight_tons": self.features.weight_tons,
            "vendor_rating": self.features.vendor_rating,
            "weather_condition": str(self.features.weather_condition),
            "traffic_density": str(self.features.traffic_density),
            "vehicle_type": str(self.features.vehicle_type),
            "probability": self.risk.probability,
            "risk_level": str(self.risk.level),
            "high_threshold": self.risk.high_threshold,
            "medium_threshold": self.risk.medium_threshold,
            "cost_fn_over_fp": self.risk.cost_fn_over_fp,
            "co2_kg": self.carbon.co2_kg,
            "carbon_price_per_ton": self.carbon.price_per_ton,
            "carbon_cost": self.carbon.cost,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_sha256": self.artifact_sha256,
            "extrapolated": self.is_extrapolated,
        }


__all__ = [
    "CarbonFootprint",
    "RiskAssessment",
    "ShipmentAssessment",
    "ShipmentFeatures",
]
