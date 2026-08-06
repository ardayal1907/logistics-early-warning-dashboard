"""Request and response bodies.

These are deliberately NOT `logistics.domain.models` re-exported. The domain
models are an internal contract that the refactor is free to change; an HTTP
body is a published one. Keeping them separate means renaming a domain field
is not automatically a breaking API change.

The field names here are snake_case, unlike the training columns
(`Distance_km`), because that is what a JSON client expects. The mapping
happens in one place, `ShipmentFeatures.to_model_row`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from logistics.domain.enums import TrafficDensity, VehicleType, WeatherCondition


class ScoreRequest(BaseModel):
    """One shipment to score."""

    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(gt=0, le=5000, examples=[450.0])
    weight_tons: float = Field(gt=0, le=100, examples=[12.0])
    vendor_rating: float = Field(ge=1.0, le=5.0, examples=[4.0])
    weather_condition: WeatherCondition = Field(examples=["Storm"])
    traffic_density: TrafficDensity = Field(examples=["High"])
    vehicle_type: VehicleType = Field(examples=["Diesel Truck"])

    shipment_id: str | None = None
    vendor_id: str | None = None
    route_id: str | None = None


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bounded on purpose. An unbounded list is a memory-exhaustion vector, and
    # 1,000 rows is already two orders of magnitude above the interactive use
    # this endpoint exists for.
    shipments: list[ScoreRequest] = Field(min_length=1, max_length=1000)


class RiskOut(BaseModel):
    probability: float
    level: str
    high_threshold: float
    medium_threshold: float
    cost_fn_over_fp: float


class CarbonOut(BaseModel):
    co2_kg: float
    co2_tons: float
    price_per_ton: float
    cost: float


class ScoreResponse(BaseModel):
    """The full answer, including what produced it.

    `model_version` and `artifact_sha256` travel with every score. A stored
    prediction whose artefact cannot be identified is not auditable.
    """

    risk: RiskOut
    carbon: CarbonOut
    model_name: str
    model_version: str
    artifact_sha256: str | None = None
    scored_at: str
    extrapolation_warnings: list[str] = []
    shipment_id: str | None = None


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


class HealthResponse(BaseModel):
    """What /health returns.

    `model_version` is the required field: the roadmap's exit criterion for
    this step is that a health check identifies the artefact being served, not
    merely that the process is alive. A load balancer that only knows the
    process is up will happily route traffic to a container serving last
    month's model.
    """

    status: str
    model_version: str | None = None
    model_name: str | None = None
    artifact_sha256: str | None = None
    api_version: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Everything a consumer needs to avoid hardcoding the model's contract."""

    model_name: str
    model_version: str
    artifact_sha256: str | None
    trained_at: str | None
    feature_order: list[str]
    categorical_levels: dict[str, list[str]]
    numeric_ranges: dict[str, dict[str, float]]
    thresholds: dict[str, float]
    metrics: dict[str, Any]
    caveats: list[str]


__all__ = [
    "BatchScoreRequest",
    "BatchScoreResponse",
    "CarbonOut",
    "HealthResponse",
    "ModelInfoResponse",
    "RiskOut",
    "ScoreRequest",
    "ScoreResponse",
]
