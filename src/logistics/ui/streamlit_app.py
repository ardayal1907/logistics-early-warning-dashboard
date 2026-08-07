"""Streamlit demo: Shipment Delay Risk & Carbon Cost.

Run with:   streamlit run src/app.py

This module re-implements NO machine learning and NO domain logic. It builds a
`ScoringService` and asks it for an answer; every number on the page comes back
from that call or from the artefact's own metadata.

What changed when this moved out of `src/app.py`:

* `find_model_path()` and `find_processed_dir()` are gone. Both walked
  `(parent.parent, parent, Path.cwd())` and took the first hit, so a `models/`
  directory left in whatever directory the process happened to start in won
  over the repository's own artefact — the loaded model depended on the working
  directory. The path is now configuration (`Settings`), not a search.

* The sliders derive their bounds from `service.numeric_ranges` instead of
  typed-in numbers. The Vendor_Rating slider ran `[2.5, 5.0]` while the model
  was trained on `[1.71, 4.89]`, so the worst vendors — precisely the ones an
  early-warning panel exists to catch — could not be entered at all.

* `@st.cache_resource` now wraps `build_scoring_service(...)`, which knows
  nothing about Streamlit. The cache is a presentation concern; what it caches
  is not.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from logistics.domain.enums import TrafficDensity, VehicleType, WeatherCondition
from logistics.domain.models import ShipmentAssessment, ShipmentFeatures
from logistics.errors import LogisticsError
from logistics.infrastructure.fingerprint import (
    compare_data_fingerprint,
    compute_data_fingerprint,
)
from logistics.services.scoring import ScoringService, build_scoring_service
from logistics.settings import Settings

# The model cannot reliably separate these two because of small sample sizes
# (see README / docs/METHODOLOGY.md).
LOW_SAMPLE_WEATHER = {"Storm", "Snow"}

# Fallbacks used only when the artefact's metadata carries no range for a
# numeric feature. They are the widest values the UI ever offered, so an older
# artefact degrades to the previous behaviour rather than to an empty slider.
_RANGE_FALLBACK: dict[str, tuple[float, float]] = {
    "Vendor_Rating": (1.0, 5.0),
    "Distance_km": (50.0, 1200.0),
    "Weight_tons": (1.0, 25.0),
}


@st.cache_resource
def get_service() -> ScoringService:
    """Build the scoring service once per Streamlit session.

    `Settings.from_env()` is read here because this module is a composition
    root. The service itself receives resolved collaborators and never looks at
    configuration.
    """
    return build_scoring_service(Settings.from_env())


def slider_bounds(
    service: ScoringService, feature: str
) -> tuple[float, float]:
    """Training range for a numeric feature, widened to a friendly step.

    Returns the model's own `[min, max]`. Offering values the model never saw
    is not a kindness: it produces a confident number from an input the
    artefact has no evidence for.
    """
    recorded = service.numeric_ranges.get(feature)
    if not recorded:
        return _RANGE_FALLBACK.get(feature, (0.0, 1.0))
    return float(recorded["min"]), float(recorded["max"])


def data_sync_mismatches(service: ScoringService, processed_dir: Any) -> list[str]:
    """Has the data been regenerated since this model was trained?

    Returns a list of mismatch descriptions — empty when in sync, and empty
    when the check cannot be performed. Silence on success is deliberate: a
    banner that appears every single run is one users learn to ignore.
    """
    if processed_dir is None or not processed_dir.is_dir():
        return []
    meta = service.bundle.metadata
    recorded = {
        "scored_table": meta.get("training_data_sha256"),
        "source_tables": meta.get("source_tables_sha256"),
    }
    return compare_data_fingerprint(recorded, compute_data_fingerprint(processed_dir))


def _render_header(service: ScoringService) -> dict[str, Any]:
    meta = service.bundle.metadata
    metrics = meta["metrics"]

    st.title("🚚 Logistics Delay-Risk Prediction & Carbon Cost")
    st.caption(
        f"Calibrated Random Forest · out-of-fold ROC-AUC {metrics['oof_roc_auc']:.3f} · "
        f"chronological holdout {metrics['chronological_holdout_roc_auc']:.3f} · "
        f"model trained {meta['trained_at'][:10]} · artefact {service.bundle.version}"
    )
    return meta


def _render_sidebar(service: ScoringService) -> ShipmentFeatures | None:
    """Collect the six model inputs. Returns None if they do not validate."""
    st.sidebar.header("Shipment Details")

    levels = service.categorical_levels
    rating_min, rating_max = slider_bounds(service, "Vendor_Rating")
    distance_min, distance_max = slider_bounds(service, "Distance_km")
    weight_min, weight_max = slider_bounds(service, "Weight_tons")

    vendor_rating = st.sidebar.slider(
        "Vendor Rating", rating_min, rating_max,
        round((rating_min + rating_max) / 2, 1), 0.01,
        help=f"The model was trained on ratings in [{rating_min}, {rating_max}].",
    )
    traffic = st.sidebar.selectbox(
        "Traffic Density", levels.get("Traffic_Density", ["Low", "Medium", "High"])
    )
    weather = st.sidebar.selectbox(
        "Weather Condition",
        levels.get("Weather_Condition", ["Normal", "Rain", "Snow", "Storm"]),
    )
    vehicle = st.sidebar.selectbox(
        "Vehicle Type",
        levels.get("Vehicle_Type", ["Diesel Truck", "Electric Semi", "Hybrid Van"]),
    )
    distance_km = st.sidebar.slider(
        "Distance (km)", distance_min, distance_max,
        round((distance_min + distance_max) / 2, 1), 1.0,
    )
    weight_tons = st.sidebar.slider(
        "Weight (tons)", weight_min, weight_max,
        round((weight_min + weight_max) / 2, 1), 0.1,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "This is a **demo**: the model was trained on synthetic data and does not "
        "represent real-world performance. See *About this demo* on the main screen "
        "for details."
    )

    # The selectboxes are populated from the artefact's own categorical levels,
    # so an option that is not a known enum member means the model was trained
    # on a category this build does not know about. Say so rather than coerce.
    try:
        return ShipmentFeatures(
            distance_km=distance_km,
            weight_tons=weight_tons,
            vendor_rating=vendor_rating,
            weather_condition=WeatherCondition(weather),
            traffic_density=TrafficDensity(traffic),
            vehicle_type=VehicleType(vehicle),
        )
    except ValueError as exc:
        st.sidebar.error(f"Those inputs are not valid: {exc}")
        return None


def _render_assessment(assessment: ShipmentAssessment, weather: str) -> None:
    risk = assessment.risk
    carbon = assessment.carbon
    probability = float(risk.probability)
    level = str(risk.level)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delay Probability", f"{probability * 100:.1f}%",
              help="The model's calibrated prediction (machine learning output).")
    c2.metric("Risk Level", level,
              help=f"Thresholds: High ≥ {risk.high_threshold}, "
                   f"Medium ≥ {risk.medium_threshold}")
    c3.metric("CO₂ (tons) · calculated", f"{carbon.co2_tons:.3f}",
              help="NOT a model prediction - derived deterministically from "
                   "distance, tonnage and vehicle type.")
    c4.metric("Carbon Tax ($) · calculated", f"${carbon.cost:,.2f}",
              help=f"CO₂ tonnage × ${carbon.price_per_ton:.0f}/ton assumption.")

    if level == "High Risk":
        st.error(
            f"**High Risk — {probability * 100:.1f}%**\n\n"
            "Worth intervening even if a false alarm cost the *same* as a missed "
            "delay. Contact the vendor, reserve buffer capacity, or consider a "
            "route/vehicle change."
        )
    elif level == "Medium Risk":
        st.warning(
            f"**Medium Risk — {probability * 100:.1f}%**\n\n"
            f"Worth intervening under the assumption that a missed delay costs "
            f"{risk.cost_fn_over_fp:.0f}× more than a false alarm. Keep it under watch."
        )
    else:
        st.success(
            f"**Low Risk — {probability * 100:.1f}%**\n\n"
            f"Not worth intervening even at the {risk.cost_fn_over_fp:.0f}:1 cost "
            "assumption. Can proceed on the standard flow."
        )

    # The service reports when an input left the region the model has evidence
    # for. Previously impossible to trigger: the sliders could not leave it.
    if assessment.is_extrapolated:
        st.warning(
            "⚠️ **Outside the training range.** "
            + " ".join(assessment.extrapolation_warnings)
        )

    # A user may flip between Storm and Snow and see risk move the "wrong" way.
    # Rather than hide that, explain it right here.
    if weather in LOW_SAMPLE_WEATHER:
        st.info(
            f"ℹ️ **Limited sample for {weather}.** The training data contains 128 Storm "
            "and 193 Snow shipments. The *ordering* between these two conditions is "
            "not reliable — switching from Snow to Storm may show risk going slightly "
            "**down**. This is a data limitation, not a model bug: the model learns "
            "the rates observed in the data, and these cells hold too few "
            "observations. That both conditions are riskier *than Normal or Rain* is "
            "reliable."
        )


def _render_about(service: ScoringService, meta: dict[str, Any]) -> None:
    thresholds = service.thresholds
    metrics = meta["metrics"]
    training = meta["training_data"]
    price = service.carbon_price_per_ton

    with st.expander("ℹ️ About this demo — limits and assumptions", expanded=False):
        st.markdown(
            f"""
**The data is synthetic.** {training['n_rows']} shipments from a
rule-based simulation ({training['date_min']} – {training['date_max']}). What is on
display is the **methodology**, not real-world performance.

**The probabilities are calibrated — but weak in the tail.** A raw Random Forest
output is the fraction of trees voting positive, not a probability.
`CalibratedClassifierCV` (isotonic) cut the calibration error (ECE) from 0.137 to
{metrics['oof_ece']:.3f}, so "30%" really does mean *about 30%*. Above
p > 0.8 there are few observations and reliability drops — treat very high
probabilities with more caution than mid-range ones.

**The thresholds come from a cost assumption, not percentiles.** A missed delay (SLA
breach, penalty, expedited shipping) is assumed
**{thresholds['cost_fn_over_fp']:.0f}× more expensive** than a false alarm (a
planner's phone call), which makes the optimal intervention threshold analytic:
`p* = 1 / (1 + {thresholds['cost_fn_over_fp']:.0f}) = {thresholds['medium_risk']}`.
`High Risk` ({thresholds['high_risk']}) is the 1:1 threshold — worth acting on even
if both errors cost the same. **The ratio has not been validated against real SLA
penalty schedules.**

**Storm and Snow cannot be separated reliably.** Training data holds 128 Storm
shipments against 193 Snow. Across a 216-combination sweep the true generating process
makes Storm riskier than Snow 100% of the time; the model captures that ordering only
61% of the time. A **data limitation, not a model bug** — the model learns the rates
present in the data, and these cells hold too few observations.

**CO₂ and carbon tax are NOT model outputs.** Distance × tonnage × vehicle emission
factor + fixed base, times weather/traffic multipliers, priced at
${price:.0f}/ton — the same deterministic formula the data generator
uses, imported from the same module. No prediction, only unit conversion and pricing.

**This demo scores slightly "sharper" than the Power BI report.** The report's CSV is
scored *walk-forward* (each shipment by a model trained only on earlier shipments);
this model is trained on all the data, which is correct for deployment. Hence
`High Risk` is 10.3% in the report versus 17.7% here. Both are right — the report
*reports the past*, the demo *scores a new shipment*.

**Model performance (all out-of-sample):**

| Metric | Value |
|---|---|
| ROC-AUC (out-of-fold) | {metrics['oof_roc_auc']:.3f} |
| ROC-AUC (chronological holdout) | {metrics['chronological_holdout_roc_auc']:.3f} |
| ROC-AUC (chronological CV) | {metrics['chronological_cv_roc_auc_mean']:.3f} ± {metrics['chronological_cv_roc_auc_std']:.3f} |
| Brier score | {metrics['oof_brier']:.4f} |
| Calibration error (ECE) | {metrics['oof_ece']:.4f} |

The chronological figure is lower because the test period is a *future* season, and
the ±{metrics['chronological_cv_roc_auc_std']:.2f} fold spread has the same
cause: bad weather gives the model something to discriminate on in winter and nothing
in summer. Not a regression — the only honest measurement.

*The notebook and `docs/METHODOLOGY.md` carry the same limitations in more depth,
including the ones that do not affect this single-shipment view (warm-up scoring, one
seasonal cycle, no concept drift).*
"""
        )


def main() -> None:
    st.set_page_config(page_title="Logistics Delay-Risk Prediction", page_icon="🚚",
                       layout="wide")

    try:
        service = get_service()
    except (LogisticsError, FileNotFoundError) as exc:
        st.error(
            f"{exc}\n\nRun `python src/ml_delay_risk_pipeline.py` first, or point "
            "`LOGISTICS_MODEL_PATH` at an existing artefact."
        )
        st.stop()
        return

    meta = _render_header(service)

    # Stale-model guard. Silent when everything lines up.
    mismatches = data_sync_mismatches(service, Settings.from_env().processed_dir)
    if mismatches:
        st.warning(
            "⚠️ **This model may have been trained on different data than the "
            "files currently on disk.** The predictions below may not reflect the "
            "current dataset.\n\n"
            "Changed since the model was trained: "
            + ", ".join(mismatches)
            + f".\n\nThe model was trained on {meta['trained_at'][:10]}. "
            "Re-run `python src/ml_delay_risk_pipeline.py` to retrain against the "
            "current data."
        )

    features = _render_sidebar(service)

    if features is not None and st.button("🚀 Calculate Risk & Cost", type="primary"):
        try:
            assessment = service.score(features)
        except LogisticsError as exc:
            st.error(f"Scoring failed: {exc}")
        else:
            _render_assessment(assessment, str(features.weather_condition))
            st.divider()
            st.markdown("**Input summary**")
            st.dataframe(
                {
                    "Value": features.to_model_row(service.feature_order),
                },
                use_container_width=False,
            )
    else:
        st.info(
            "👈 Enter the shipment details in the sidebar, then press "
            "**🚀 Calculate Risk & Cost**."
        )

    _render_about(service, meta)


if __name__ == "__main__":  # pragma: no cover
    main()
