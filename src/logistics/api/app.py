"""The HTTP application.

Endpoints:
    GET  /health        liveness AND which artefact is loaded
    GET  /model         the model's own contract, so clients hardcode nothing
    POST /score         one shipment
    POST /score/batch   up to 1,000, in a single predict_proba call

Nothing here reimplements domain logic. Every handler translates a request
into `ShipmentFeatures`, asks `ScoringService`, and translates the answer
back.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from logistics import __version__
from logistics.api.deps import AppState, build_state
from logistics.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    CarbonOut,
    HealthResponse,
    ModelInfoResponse,
    RiskOut,
    ScoreRequest,
    ScoreResponse,
)
from logistics.domain.models import ShipmentAssessment, ShipmentFeatures
from logistics.errors import LogisticsError
from logistics.services.scoring import ScoringService
from logistics.settings import Settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the artefact before the first request, not during it."""
    app.state.app_state = build_state(getattr(app.state, "settings", None))
    yield


def get_state(request: Request) -> AppState:
    return request.app.state.app_state  # type: ignore[no-any-return]


def get_service(state: AppState = Depends(get_state)) -> ScoringService:
    """503 rather than 500 when the artefact never loaded.

    The distinction matters to an orchestrator: 503 means "not ready, keep me
    out of the pool", 500 means "this request was bad".
    """
    if state.service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.startup_error or "The model is not loaded.",
        )
    return state.service


def _to_features(payload: ScoreRequest) -> ShipmentFeatures:
    return ShipmentFeatures(
        distance_km=payload.distance_km,
        weight_tons=payload.weight_tons,
        vendor_rating=payload.vendor_rating,
        weather_condition=payload.weather_condition,
        traffic_density=payload.traffic_density,
        vehicle_type=payload.vehicle_type,
        shipment_id=payload.shipment_id,
        vendor_id=payload.vendor_id,
        route_id=payload.route_id,
    )


def _to_response(assessment: ShipmentAssessment) -> ScoreResponse:
    return ScoreResponse(
        risk=RiskOut(
            probability=float(assessment.risk.probability),
            level=str(assessment.risk.level),
            high_threshold=float(assessment.risk.high_threshold),
            medium_threshold=float(assessment.risk.medium_threshold),
            cost_fn_over_fp=float(assessment.risk.cost_fn_over_fp),
        ),
        carbon=CarbonOut(
            co2_kg=assessment.carbon.co2_kg,
            co2_tons=assessment.carbon.co2_tons,
            price_per_ton=assessment.carbon.price_per_ton,
            cost=assessment.carbon.cost,
        ),
        model_name=assessment.model_name,
        model_version=assessment.model_version,
        artifact_sha256=assessment.artifact_sha256,
        scored_at=assessment.scored_at.isoformat(timespec="seconds"),
        extrapolation_warnings=list(assessment.extrapolation_warnings),
        shipment_id=assessment.features.shipment_id,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Takes Settings so a test can point it anywhere."""
    app = FastAPI(
        title="Logistics Delay-Risk API",
        version=__version__,
        summary="Calibrated delay-risk probabilities and deterministic carbon cost.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(LogisticsError)
    async def _domain_error(_: Request, exc: LogisticsError) -> JSONResponse:
        """Domain failures are 422, not 500.

        An unknown vehicle type or an unreadable artefact is a fact about the
        request or the deployment, not a bug in the handler, and a client that
        sees 500 will retry a request that can never succeed.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error": type(exc).__name__},
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(state: AppState = Depends(get_state)) -> HealthResponse:
        """Liveness plus the identity of the artefact being served.

        Returning only {"status": "ok"} would let a load balancer route
        traffic to a container serving a stale model. The version is the
        point of this endpoint.
        """
        if state.service is None:
            return HealthResponse(
                status="unavailable",
                api_version=__version__,
                detail=state.startup_error or "The model is not loaded.",
            )
        bundle = state.service.bundle
        return HealthResponse(
            status="ok",
            model_version=bundle.version,
            model_name=bundle.name,
            artifact_sha256=bundle.artifact_sha256,
            api_version=__version__,
        )

    @app.get("/model", response_model=ModelInfoResponse, tags=["ops"])
    def model_info(
        service: ScoringService = Depends(get_service),
    ) -> ModelInfoResponse:
        meta: dict[str, Any] = service.bundle.metadata
        return ModelInfoResponse(
            model_name=service.bundle.name,
            model_version=service.bundle.version,
            artifact_sha256=service.bundle.artifact_sha256,
            trained_at=meta.get("trained_at"),
            feature_order=service.feature_order,
            categorical_levels=service.categorical_levels,
            numeric_ranges=service.numeric_ranges,
            thresholds=service.thresholds,
            metrics=meta.get("metrics", {}),
            caveats=list(meta.get("caveats", [])),
        )

    @app.post("/score", response_model=ScoreResponse, tags=["scoring"])
    def score(
        payload: ScoreRequest,
        service: ScoringService = Depends(get_service),
    ) -> ScoreResponse:
        return _to_response(service.score(_to_features(payload)))

    @app.post("/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
    def score_batch(
        payload: BatchScoreRequest,
        service: ScoringService = Depends(get_service),
    ) -> BatchScoreResponse:
        """One predict_proba call over the whole batch, not a loop over /score."""
        assessments = service.score_many(
            _to_features(item) for item in payload.shipments
        )
        results = [_to_response(a) for a in assessments]
        return BatchScoreResponse(results=results, count=len(results))

    return app


# Module-level application for `uvicorn logistics.api.app:app`.
app = create_app()

__all__ = ["app", "create_app", "get_service", "get_state"]
