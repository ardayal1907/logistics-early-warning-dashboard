"""The scoring service — the keystone of the layered architecture.

Every individual piece of the scoring workflow already lived outside the
Streamlit page: `classify_risk` and `compute_co2_kg` came from `config.py`, the
column order came from the artefact's metadata, and the encoder travelled inside
the pickle. That discipline is why this refactor is small.

What did NOT exist anywhere was the thing that puts those pieces in order:

    validate input -> assemble the feature row in the model's column order
                   -> predict_proba -> classify against the model's thresholds
                   -> compute CO2 -> price the carbon -> record the decision

Those seven steps existed only inside `if st.button(...)`. A REST endpoint, a
nightly batch job, an ERP integration or the optimiser could not call them; each
would have had to re-create the sequence, and a re-created sequence is correct
on day one and subtly wrong by day thirty. That is the same class of drift
`config.py` was built to prevent, one level up: not duplicated constants, but a
duplicated *procedure*.

So this module owns the procedure and nothing else. It imports no UI framework,
holds no global state, and takes its collaborators as constructor arguments.

Three entry points, one implementation behind all of them:

    score(features)        -> ShipmentAssessment       single shipment, UI/API
    score_many(features)   -> list[ShipmentAssessment] batch
    score_frame(df)        -> DataFrame                vectorised, for the ETL
                                                       and the optimiser

`score_frame` is not a convenience wrapper. The optimisation layer needs a
calibrated probability for every (shipment x vehicle) combination it might
choose - tens of thousands of rows - and calling a Pydantic-validated
single-row path that many times would dominate the solve time. It shares the
carbon and threshold logic with the row-wise path; only the transport differs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from logistics.domain.carbon import carbon_cost, compute_co2_kg, compute_co2_kg_many
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
)
from logistics.errors import ScoringError
from logistics.infrastructure.model_repository import ModelBundle, load_bundle
from logistics.infrastructure.prediction_log import (
    JsonlPredictionLog,
    NullPredictionLog,
    PredictionLog,
)
from logistics.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = logging.getLogger(__name__)


class ScoringService:
    """Turns shipment features into a risk-and-carbon assessment.

    Stateless apart from its collaborators, so a single instance is safe to
    share across requests and cheap to construct in a test with a fake model.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        carbon_price_per_ton: float,
        prediction_log: PredictionLog | None = None,
    ) -> None:
        self._bundle = bundle
        self._carbon_price = carbon_price_per_ton
        self._log = prediction_log or NullPredictionLog()

        # Thresholds come from the ARTEFACT, not from this module's defaults.
        # If a model is retrained under a different cost assumption, every
        # consumer follows the model automatically. The defaults are only a
        # fallback for an artefact that predates the convention.
        thresholds = bundle.thresholds
        self._high = float(thresholds.get("high_risk", DEFAULT_HIGH_RISK_THRESHOLD))
        self._medium = float(thresholds.get("medium_risk", DEFAULT_MEDIUM_RISK_THRESHOLD))
        self._cost_ratio = float(
            thresholds.get("cost_fn_over_fp", DEFAULT_COST_FN_OVER_FP)
        )
        self._feature_order = bundle.feature_order

    # -- introspection, for UIs that must not hardcode anything --------------
    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    @property
    def feature_order(self) -> list[str]:
        return list(self._feature_order)

    @property
    def categorical_levels(self) -> dict[str, list[str]]:
        """The categories the model was actually trained on."""
        return self._bundle.categorical_levels

    @property
    def numeric_ranges(self) -> dict[str, dict[str, float]]:
        """Training ranges per numeric feature.

        A UI should derive its slider bounds from this rather than from typed-in
        numbers. app.py offered Vendor_Rating in [2.5, 5.0] while the model was
        trained on [1.71, 4.89] - so the worst vendors, the ones the early
        warning panel exists to catch, could not be entered at all.
        """
        return self._bundle.numeric_ranges

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "high_risk": self._high,
            "medium_risk": self._medium,
            "cost_fn_over_fp": self._cost_ratio,
        }

    @property
    def carbon_price_per_ton(self) -> float:
        return self._carbon_price

    # -- scoring -------------------------------------------------------------
    def score(self, features: ShipmentFeatures) -> ShipmentAssessment:
        """Score one shipment and record the decision."""
        assessment = self._assess(features, self._predict_one(features))
        self._log.record(assessment)
        return assessment

    def score_many(self, features: Iterable[ShipmentFeatures]) -> list[ShipmentAssessment]:
        """Score a batch in a single `predict_proba` call.

        Not a loop over `score()`: one call over N rows is what makes batch
        scoring viable, and it keeps every row on the identical code path.
        """
        items: Sequence[ShipmentFeatures] = list(features)
        if not items:
            return []

        probabilities = self._predict_batch(items)
        assessments = [
            self._assess(item, probability)
            for item, probability in zip(items, probabilities, strict=True)
        ]
        for assessment in assessments:
            self._log.record(assessment)
        return assessments

    def score_frame(
        self,
        df: pd.DataFrame,
        *,
        probability_column: str = "Delay_Risk_Probability",
        level_column: str = "Risk_Level",
        co2_column: str = "CO2_Emission_kg_calculated",
        carbon_cost_column: str = "Carbon_Cost",
    ) -> pd.DataFrame:
        """Vectorised scoring for the ETL and the optimisation layer.

        The frame must already carry the model's feature columns; anything else
        it holds (Shipment_ID, Route_ID, ...) is passed through untouched, so
        the result can be joined straight back onto the fact table.

        Deliberately skips Pydantic validation - it is the wrong tool for
        100k rows, and the caller here is the pipeline, not an external client.
        Callers that need validation should use `score_many`.
        """
        missing = [c for c in self._feature_order if c not in df.columns]
        if missing:
            raise ScoringError(
                f"The frame is missing feature column(s) the model requires: "
                f"{missing}. Present: {sorted(df.columns)}"
            )

        result = df.copy()
        probabilities = self._bundle.model.predict_proba(df[self._feature_order])[:, 1]

        result[probability_column] = probabilities
        result[level_column] = [
            str(classify_risk(float(p), self._high, self._medium)) for p in probabilities
        ]
        # Batch form, not a per-row loop. On the optimiser's counterfactual grid
        # the row-by-row version took longer than predict_proba itself.
        result[co2_column] = compute_co2_kg_many(
            df["Distance_km"].to_numpy(dtype=float),
            df["Weight_tons"].to_numpy(dtype=float),
            df["Vehicle_Type"].to_numpy(),
            df["Weather_Condition"].to_numpy(),
            df["Traffic_Density"].to_numpy(),
        )
        result[carbon_cost_column] = [
            carbon_cost(v, self._carbon_price) for v in result[co2_column]
        ]
        return result

    # -- internals -----------------------------------------------------------
    def _predict_one(self, features: ShipmentFeatures) -> float:
        return self._predict_batch([features])[0]

    def _predict_batch(self, items: Sequence[ShipmentFeatures]) -> list[float]:
        import pandas as pd

        rows = [item.to_model_row(self._feature_order) for item in items]
        frame = pd.DataFrame(rows, columns=self._feature_order)
        try:
            proba = self._bundle.model.predict_proba(frame)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed domain error
            raise ScoringError(
                f"The model failed to score {len(items)} row(s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return [float(p) for p in proba[:, 1]]

    def _assess(self, features: ShipmentFeatures, probability: float) -> ShipmentAssessment:
        # CO2 is NOT a model output. It is a deterministic engineering figure
        # from the same module the data generator draws its constants from, so
        # the two can never disagree.
        co2_kg = compute_co2_kg(
            features.distance_km,
            features.weight_tons,
            features.vehicle_type,
            features.weather_condition,
            features.traffic_density,
        )

        return ShipmentAssessment(
            features=features,
            risk=RiskAssessment(
                probability=probability,
                level=classify_risk(probability, self._high, self._medium),
                high_threshold=self._high,
                medium_threshold=self._medium,
                cost_fn_over_fp=self._cost_ratio,
            ),
            carbon=CarbonFootprint(co2_kg=co2_kg, price_per_ton=self._carbon_price),
            model_name=self._bundle.name,
            model_version=self._bundle.version,
            artifact_sha256=self._bundle.artifact_sha256,
            extrapolation_warnings=tuple(
                features.check_extrapolation(self.numeric_ranges)
            ),
        )


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------
def build_scoring_service(settings: Settings | None = None, **kwargs: Any) -> ScoringService:
    """Wire the service from configuration. The only place that decides *what*.

    Every other module receives its collaborators; this function is where the
    concrete choices are made, so a test can bypass it entirely and construct
    `ScoringService` with a fake predictor.
    """
    cfg = settings or Settings.from_env(**kwargs)

    bundle = load_bundle(
        cfg.model_path,
        verify_checksum=cfg.verify_artifact_checksum,
        verify_versions=cfg.verify_library_versions,
    )

    if cfg.prediction_log_path is None:
        logger.warning(
            "No prediction log configured (LOGISTICS_PREDICTION_LOG_PATH is unset). "
            "Scores will not be auditable and no data will accumulate for drift "
            "monitoring. Acceptable for a local demo; not for production."
        )
        log: PredictionLog = NullPredictionLog()
    else:
        log = JsonlPredictionLog(cfg.prediction_log_path)

    return ScoringService(
        bundle,
        carbon_price_per_ton=cfg.carbon_price_per_ton,
        prediction_log=log,
    )


__all__ = ["ScoringService", "build_scoring_service"]
