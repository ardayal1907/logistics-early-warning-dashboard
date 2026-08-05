"""
Streamlit demo: Shipment Delay Risk & Carbon Cost
=========================================================
Run with:   streamlit run src/app.py

This app re-implements NO machine learning logic. It loads the complete pipeline
stored in `models/production_risk_model.pkl` (ColumnTransformer + OneHotEncoder +
CalibratedClassifierCV) exactly as it was saved and hands it a raw DataFrame. Doing
the encoding by hand here is the classic source of train/serve skew: the column order
or the unknown-category handling drifts slightly and the model silently starts
predicting the wrong thing.

Thresholds, feature order and metrics are NOT hardcoded either — they are read from
the metadata bundled in the same file. That way, if the model is retrained with
different thresholds, this app stays consistent automatically.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# CO2 calculation - imported, never re-declared
# ---------------------------------------------------------------------------
# IMPORTANT: CO2 is NOT a model prediction. It is an engineering figure computed
# deterministically from distance, tonnage and vehicle type by `config.compute_co2_kg`
# — the same module the data generator draws its constants from, so the two can never
# drift apart. The only difference is that the generator adds ±3% random noise on top
# for realism, while this app does not: the same input must always produce the same
# output.
from config import (  # noqa: E402
    CARBON_PRICE_PER_TON, classify_risk, compare_data_fingerprint,
    compute_co2_kg, compute_data_fingerprint,
)

# The model cannot reliably separate these two because of small sample sizes
# (see README / docs/METHODOLOGY.md).
LOW_SAMPLE_WEATHER = {"Storm", "Snow"}


def find_model_path() -> Path:
    """Locate the model in both flat and src/ layouts (on Streamlit Cloud the
    working directory is the repository root)."""
    here = Path(__file__).resolve()
    for base in (here.parent.parent, here.parent, Path.cwd()):
        candidate = base / "models" / "production_risk_model.pkl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "models/production_risk_model.pkl not found. "
        "Run `python src/ml_delay_risk_pipeline.py` first."
    )


@st.cache_resource
def load_bundle():
    bundle = joblib.load(find_model_path())
    return bundle["model"], bundle["metadata"]


def find_processed_dir() -> Path | None:
    """Locate data/processed/ the same way find_model_path() locates the model."""
    here = Path(__file__).resolve()
    for base in (here.parent.parent, here.parent, Path.cwd()):
        candidate = base / "data" / "processed"
        if candidate.is_dir():
            return candidate
    return None


def data_sync_mismatches(meta: dict) -> list:
    """Has the data been regenerated since this model was trained?

    Returns a list of mismatch descriptions (empty when in sync, or when the check
    cannot be performed). Silence on success is deliberate: a banner that appears
    every single run is one users learn to ignore.
    """
    processed_dir = find_processed_dir()
    if processed_dir is None:
        return []
    recorded = {
        "scored_table": meta.get("training_data_sha256"),
        "source_tables": meta.get("source_tables_sha256"),
    }
    return compare_data_fingerprint(recorded, compute_data_fingerprint(processed_dir))


# ---------------------------------------------------------------------------
# Everything below lives inside main(). Streamlit executes the script with
# __name__ == "__main__", so the app still starts normally with
# `streamlit run src/app.py` — but importing this module (as the test suite does)
# runs nothing, loads no 8.7 MB model, and opens no page.
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Logistics Delay-Risk Prediction", page_icon="🚚",
                       layout="wide")

    try:
        model, meta = load_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    thresholds = meta["thresholds"]
    levels = meta["categorical_levels"]

    st.title("🚚 Logistics Delay-Risk Prediction & Carbon Cost")
    st.caption(
        f"Calibrated Random Forest · out-of-fold ROC-AUC {meta['metrics']['oof_roc_auc']:.3f} · "
        f"chronological holdout {meta['metrics']['chronological_holdout_roc_auc']:.3f} · "
        f"model trained {meta['trained_at'][:10]}"
    )

    # Stale-model guard. Silent when everything lines up.
    mismatches = data_sync_mismatches(meta)
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

    # ---------------------------------------------------------------------------
    # Sidebar - inputs
    # ---------------------------------------------------------------------------
    st.sidebar.header("Shipment Details")

    vendor_rating = st.sidebar.slider("Vendor Rating", 2.5, 5.0, 4.0, 0.1)
    traffic = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High"], index=1)
    weather = st.sidebar.selectbox("Weather Condition",
                                   ["Normal", "Rain", "Snow", "Storm"], index=0)
    vehicle = st.sidebar.selectbox("Vehicle Type",
                                   ["Diesel Truck", "Electric Semi", "Hybrid Van"], index=0)
    distance_km = st.sidebar.slider("Distance (km)", 50, 1200, 450, 10)
    weight_tons = st.sidebar.slider("Weight (tons)", 1.0, 24.0, 12.0, 0.5)

    # Are the UI options still consistent with the model's metadata? If the model is
    # retrained and its categories change, warn here instead of silently mispredicting.
    for col, chosen in [("Weather_Condition", weather), ("Traffic_Density", traffic),
                        ("Vehicle_Type", vehicle)]:
        if chosen not in levels.get(col, [chosen]):
            st.sidebar.warning(
                f"'{chosen}' does not appear in the model's training data ({col}). "
                "The prediction will be made with this category ignored."
            )

    st.sidebar.divider()
    st.sidebar.caption(
        "This is a **demo**: the model was trained on synthetic data and does not "
        "represent real-world performance. See *About this demo* on the main screen "
        "for details."
    )

    # ---------------------------------------------------------------------------
    # Main screen
    # ---------------------------------------------------------------------------
    if st.button("🚀 Calculate Risk & Cost", type="primary"):
        # Column order comes from the metadata - it is not hardcoded.
        features = pd.DataFrame([{
            "Vendor_Rating": vendor_rating,
            "Traffic_Density": traffic,
            "Weather_Condition": weather,
            "Vehicle_Type": vehicle,
            "Distance_km": float(distance_km),
            "Weight_tons": float(weight_tons),
        }])[meta["feature_order"]]

        probability = float(model.predict_proba(features)[0, 1])
        # Thresholds come from the model's metadata, so the app follows the model.
        risk_level = classify_risk(probability,
                                   thresholds["high_risk"], thresholds["medium_risk"])

        co2_kg = compute_co2_kg(float(distance_km), float(weight_tons),
                                vehicle, weather, traffic)
        co2_tons = co2_kg / 1000.0
        carbon_tax = co2_tons * CARBON_PRICE_PER_TON

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delay Probability", f"{probability * 100:.1f}%",
                  help="The model's calibrated prediction (machine learning output).")
        c2.metric("Risk Level", risk_level,
                  help=f"Thresholds: High ≥ {thresholds['high_risk']}, "
                       f"Medium ≥ {thresholds['medium_risk']}")
        c3.metric("CO₂ (tons) · calculated", f"{co2_tons:.3f}",
                  help="NOT a model prediction - derived deterministically from "
                       "distance, tonnage and vehicle type.")
        c4.metric("Carbon Tax ($) · calculated", f"${carbon_tax:,.2f}",
                  help=f"CO₂ tonnage × ${CARBON_PRICE_PER_TON:.0f}/ton assumption.")

        if risk_level == "High Risk":
            st.error(
                f"**High Risk — {probability * 100:.1f}%**\n\n"
                "Worth intervening even if a false alarm cost the *same* as a missed "
                "delay. Contact the vendor, reserve buffer capacity, or consider a "
                "route/vehicle change."
            )
        elif risk_level == "Medium Risk":
            st.warning(
                f"**Medium Risk — {probability * 100:.1f}%**\n\n"
                f"Worth intervening under the assumption that a missed delay costs "
                f"{thresholds['cost_fn_over_fp']:.0f}× more than a false alarm. Keep it "
                "under watch."
            )
        else:
            st.success(
                f"**Low Risk — {probability * 100:.1f}%**\n\n"
                f"Not worth intervening even at the "
                f"{thresholds['cost_fn_over_fp']:.0f}:1 cost assumption. Can proceed on "
                "the standard flow."
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

        st.divider()
        st.markdown("**Input summary**")
        st.dataframe(
            features.T.rename(columns={0: "Value"}),
            use_container_width=False,
        )

    else:
        st.info(
            "👈 Enter the shipment details in the sidebar, then press "
            "**🚀 Calculate Risk & Cost**."
        )

    # ---------------------------------------------------------------------------
    # Transparency
    # ---------------------------------------------------------------------------
    with st.expander("ℹ️ About this demo — limits and assumptions", expanded=False):
        st.markdown(
            f"""
**The data is synthetic.** {meta['training_data']['n_rows']} shipments from a
rule-based simulation ({meta['training_data']['date_min']} –
{meta['training_data']['date_max']}). What is on display is the **methodology**, not
real-world performance.

**The probabilities are calibrated — but weak in the tail.** A raw Random Forest
output is the fraction of trees voting positive, not a probability.
`CalibratedClassifierCV` (isotonic) cut the calibration error (ECE) from 0.137 to
{meta['metrics']['oof_ece']:.3f}, so "30%" really does mean *about 30%*. Above
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
${CARBON_PRICE_PER_TON:.0f}/ton — the same deterministic formula the data generator
uses, imported from the same module. No prediction, only unit conversion and pricing.

**This demo scores slightly "sharper" than the Power BI report.** The report's CSV is
scored *walk-forward* (each shipment by a model trained only on earlier shipments);
this model is trained on all the data, which is correct for deployment. Hence
`High Risk` is 10.3% in the report versus 17.7% here. Both are right — the report
*reports the past*, the demo *scores a new shipment*.

**Model performance (all out-of-sample):**

| Metric | Value |
|---|---|
| ROC-AUC (out-of-fold) | {meta['metrics']['oof_roc_auc']:.3f} |
| ROC-AUC (chronological holdout) | {meta['metrics']['chronological_holdout_roc_auc']:.3f} |
| ROC-AUC (chronological CV) | {meta['metrics']['chronological_cv_roc_auc_mean']:.3f} ± {meta['metrics']['chronological_cv_roc_auc_std']:.3f} |
| Brier score | {meta['metrics']['oof_brier']:.4f} |
| Calibration error (ECE) | {meta['metrics']['oof_ece']:.4f} |

The chronological figure is lower because the test period is a *future* season, and
the ±{meta['metrics']['chronological_cv_roc_auc_std']:.2f} fold spread has the same
cause: bad weather gives the model something to discriminate on in winter and nothing
in summer. Not a regression — the only honest measurement.

*The notebook and `docs/METHODOLOGY.md` carry the same limitations in more depth,
including the ones that do not affect this single-shipment view (warm-up scoring, one
seasonal cycle, no concept drift).*
"""
        )


if __name__ == "__main__":
    main()
