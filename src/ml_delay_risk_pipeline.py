"""
ML Pipeline: Shipment Delay-Risk Prediction (Calibrated Random Forest)
========================================================================
Designed to run in a Microsoft Fabric Notebook (or any Python/pandas
environment). If you use a Fabric Lakehouse, the CSV read/write lines can be
swapped for `spark.read.csv(...)` / Delta table read/write (see the "Fabric
notes" at the end of the script).

Steps (each one is a function; main() runs them in order):
  1) load_star_schema / build_analytical_dataset
  2) add_target
  3) build_model
  4) evaluate_chronologically / compare_split_strategies
  5) score_walk_forward
  6) assign_risk_levels
  7) build_fact_with_ml / validate_output_schema
  8) compute_importances
  9) train_production_model / save_production_model

Methodological decisions and the numbers behind them
----------------------------------------------------
* GROUPED SPLIT (StratifiedGroupKFold, Vendor_ID): having the same vendor in
  both the training and test sets can let the model memorise "this vendor is
  always like that". Measured: random split 0.774, grouped split 0.773 -> there
  is NO memorisation in this dataset. The reason is that in the generator the
  vendor effect is a smooth function of the rating; the model learns the rating,
  not the identity, and that transfers to unseen vendors. The grouped split is
  kept anyway: on real data a vendor scorecard is noisy and stale, and this
  guarantee would not come for free.
* CALIBRATION (isotonic): raw Random Forest probabilities are the fraction of
  trees voting positive, not probabilities. Measured: uncalibrated ECE 0.137
  (where it said 0.65, the realised rate was 0.36), 0.021 after isotonic.
  Isotonic gave a better ECE than sigmoid on 5 out of 5 seeds
  (difference -0.0065 +/- 0.0043).
* COST-BASED THRESHOLDS: the thresholds are derived from a cost matrix rather
  than percentiles. Measured: the old percentile threshold (0.65) caught only
  19% of delays and, under a 4:1 cost assumption, was 51% more expensive than
  the optimal threshold.

Importing this module has no side effects: nothing runs until main() is called.
"""

import json
import platform
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_val_predict, StratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit,
)
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
N_SPLITS = 5

# Paths resolve relative to the repository root, so it does not matter which
# directory you invoke the script from.
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Risk tiers and the cost assumption behind them live in one place - see src/config.py.
from config import (  # noqa: E402
    COST_FN_OVER_FP, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, classify_risk,
    compute_data_fingerprint,
)

NUMERIC_FEATURES = ["Distance_km", "Weight_tons", "Vendor_Rating"]
CATEGORICAL_FEATURES = ["Weather_Condition", "Traffic_Density", "Vehicle_Type"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPECTED_COLUMNS = [
    "Shipment_ID", "Date_ID", "Vendor_ID", "Route_ID", "Weight_tons", "Distance_km",
    "Actual_Delay_Days", "CO2_Emission_kg", "Delay_Risk_Probability", "Risk_Level",
]


# ---------------------------------------------------------------------------
# 1) Read the star schema tables and join them temporarily for training
# ---------------------------------------------------------------------------
def load_star_schema(processed_dir: Path = PROCESSED_DIR) -> dict:
    return {
        "fact": pd.read_csv(processed_dir / "Fact_Shipments.csv"),
        "dim_vendor": pd.read_csv(processed_dir / "Dim_Vendor.csv"),
        "dim_route": pd.read_csv(processed_dir / "Dim_Route.csv"),
        "dim_date": pd.read_csv(processed_dir / "Dim_Date.csv"),
    }


def build_analytical_dataset(tables: dict) -> pd.DataFrame:
    """Join the star schema into one flat frame and sort it chronologically.

    Everything downstream depends on the timeline, so the sort happens here once.
    """
    fact = tables["fact"]
    df = (
        fact
        .merge(tables["dim_vendor"], on="Vendor_ID", how="left")
        .merge(tables["dim_route"], on="Route_ID", how="left")
        .merge(tables["dim_date"][["Date_ID", "Full_Date", "Month", "Season"]],
               on="Date_ID", how="left")
    )
    assert len(df) == len(fact), "The row count changed after the join!"

    df["Full_Date"] = pd.to_datetime(df["Full_Date"])
    return df.sort_values("Full_Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) Target variable: is_delayed
# ---------------------------------------------------------------------------
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_delayed"] = (df["Actual_Delay_Days"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3) Model definition
# ---------------------------------------------------------------------------
def build_base_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """One-Hot for the nominal columns, passthrough for the numeric ones.

    `handle_unknown="ignore"` means an unseen category in production sets all its
    dummies to 0 instead of crashing the model.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def build_model(random_state: int = RANDOM_STATE,
                n_splits: int = N_SPLITS) -> CalibratedClassifierCV:
    """The base pipeline wrapped in isotonic calibration.

    Calibration fits the RF on `n_splits` inner folds and corrects the
    probabilities, so `predict_proba` becomes a probability rather than a vote share.
    """
    return CalibratedClassifierCV(build_base_pipeline(random_state),
                                  method="isotonic", cv=n_splits)


# ---------------------------------------------------------------------------
# 4) Honest performance measurement - CHRONOLOGICAL split
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple:
    """Index arrays for a past/future split. The frame must already be sorted."""
    split_at = int(len(df) * train_fraction)
    train_idx = np.arange(split_at)
    test_idx = np.arange(split_at, len(df))
    dates = df["Full_Date"]
    assert dates.iloc[train_idx].max() <= dates.iloc[test_idx].min(), \
        "Chronological leakage: the training set contains a date after the test period!"
    return train_idx, test_idx


def evaluate_chronologically(model, X, y, df, train_idx, test_idx) -> dict:
    """Train on the past, test on the future, and report the distribution shift.

    This is the only realistic imitation of the deployment scenario. Because the
    data is seasonal it carries a cost, and that cost is REAL: the last 20% falls
    in one particular season while the training set covers others. The resulting
    distribution shift is not a measurement error — it is exactly what will happen
    in production.

    Returns the fitted model plus the metrics, so the caller can reuse the model
    that has never seen the test set (needed for honest permutation importance).
    """
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "roc_auc": roc_auc_score(y_test, y_proba_test),
        "report": classification_report(
            y_test, y_pred, target_names=["On time (0)", "Delayed (1)"]),
    }


def weather_distribution_shift(df, train_idx, test_idx) -> pd.DataFrame:
    shift = pd.DataFrame({
        "Train": df.iloc[train_idx]["Weather_Condition"].value_counts(normalize=True),
        "Test": df.iloc[test_idx]["Weather_Condition"].value_counts(normalize=True),
    }).fillna(0.0)
    shift["Diff"] = shift["Test"] - shift["Train"]
    return shift


def compare_split_strategies(model, X, y, groups) -> dict:
    """Random vs vendor-grouped cross-validation on the same data and model.

    Each restriction costs a few points, and that cost is the measurement being
    corrected rather than performance being lost.
    """
    strategies = {
        "Random (StratifiedKFold)": (
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE), None),
        "Vendor-grouped (GroupKFold)": (
            StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            groups),
    }
    out = {}
    for label, (splitter, grp) in strategies.items():
        p = cross_val_predict(model, X, y, cv=splitter, groups=grp,
                              method="predict_proba", n_jobs=-1)[:, 1]
        out[label] = roc_auc_score(y, p)
    return out


def chronological_cv(model, X, y, dates, n_splits: int = N_SPLITS) -> list:
    """Expanding-window CV, so we do not depend on the luck of a single period.

    Returns a list of (fold_number, auc, first_date, last_date, delay_rate).
    """
    results = []
    for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        m = deepcopy(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        auc = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
        results.append((i, auc, dates.iloc[te].min(), dates.iloc[te].max(),
                        y.iloc[te].mean()))
    return results


# ---------------------------------------------------------------------------
# 5) Production scoring: CALIBRATED + OUT-OF-SAMPLE probabilities
# ---------------------------------------------------------------------------
def score_walk_forward(model, X, y, groups, n_splits: int = N_SPLITS) -> tuple:
    """Score every shipment with a model that never saw it, or its vendor.

    Training on all the data and scoring that same data produces memorised
    probabilities (measured: ROC-AUC 0.998, with a realised delay rate of 0.000 in
    the "Low Risk" tier - impossibly clean). Writing such a column into Power BI
    would turn the Early Warning Panel into a copy of the past rather than a
    prediction.

    Walk-forward means each shipment is scored by a model trained only on
    shipments that occurred BEFORE it — exactly what happens in production.

    Unavoidable constraint: there is no "prior data" for the first block. Those
    rows fall back to vendor-grouped out-of-fold scores, and the mask is returned
    so the caller can report how many rows that affects.

    Returns (probabilities, warm_up_mask).
    """
    proba = np.full(len(X), np.nan)

    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = deepcopy(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        proba[te] = m.predict_proba(X.iloc[te])[:, 1]

    warm_up_mask = np.isnan(proba)
    if warm_up_mask.any():
        fallback = cross_val_predict(
            model, X, y,
            cv=StratifiedGroupKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
            groups=groups, method="predict_proba", n_jobs=-1,
        )[:, 1]
        proba[warm_up_mask] = fallback[warm_up_mask]

    return proba, warm_up_mask


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Weighted gap between the predicted probability and the realised rate."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        upper = proba < edges[i + 1] if i < n_bins - 1 else proba <= 1.0
        mask = (proba >= edges[i]) & upper
        if mask.sum():
            total += mask.sum() / len(proba) * abs(proba[mask].mean() - y_true[mask].mean())
    return total


def scoring_quality(y, proba) -> dict:
    return {
        "roc_auc": roc_auc_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "ece": expected_calibration_error(y.values, proba),
        "baseline_brier": brier_score_loss(y, np.full(len(y), y.mean())),
    }


# ---------------------------------------------------------------------------
# 6) Risk_Level: thresholds derived from a COST MATRIX
# ---------------------------------------------------------------------------
# Cost assumption:
#   FALSE ALARM (FP)  -> a planner checks the shipment, calls the vendor and
#                        reserves buffer capacity. ~1 unit.
#   MISSED DELAY (FN) -> an SLA breach: contractual penalty + expedited/re-routing
#                        cost + churn risk.  -> 4 units.
#
# With calibrated probabilities, the expected-cost-minimising threshold is analytic:
#     intervene  <=>  p * C_FN > (1 - p) * C_FP
#     p* = C_FP / (C_FP + C_FN) = 1 / (1 + ratio)
# This formula is invalid if the probabilities are NOT CALIBRATED.
#
# COST_FN_OVER_FP, the thresholds and `classify_risk` are imported from
# src/config.py so that the Streamlit demo and this pipeline cannot drift apart.
def assign_risk_levels(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["Delay_Risk_Probability"] = proba.round(2)
    df["Risk_Level"] = df["Delay_Risk_Probability"].apply(classify_risk)
    return df


def calibration_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does the mean prediction equal the realised rate, tier by tier?"""
    check = (
        df.groupby("Risk_Level")
        .agg(Shipments=("Shipment_ID", "count"),
             Mean_Predicted=("Delay_Risk_Probability", "mean"),
             Realised_Rate=("is_delayed", "mean"),
             Mean_Delay_Days=("Actual_Delay_Days", "mean"))
        .reindex(["High Risk", "Medium Risk", "Low Risk"])
    )
    check["Deviation"] = check["Mean_Predicted"] - check["Realised_Rate"]
    return check


def alert_cost(proba: np.ndarray, y_true: np.ndarray, threshold: float,
               cost_ratio: float = COST_FN_OVER_FP) -> dict:
    """Expected cost of alerting at `threshold`, in false-alarm units."""
    alert = proba >= threshold
    fp = int((alert & (y_true == 0)).sum())
    fn = int((~alert & (y_true == 1)).sum())
    tp = int((alert & (y_true == 1)).sum())
    return {"cost": fp + cost_ratio * fn, "tp": tp, "fp": fp, "fn": fn,
            "alert_rate": float(alert.mean())}


# ---------------------------------------------------------------------------
# 7) Add only 2 new columns without disturbing the Fact_Shipments.csv structure
# ---------------------------------------------------------------------------
def build_fact_with_ml(fact: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    return fact.merge(
        scored[["Shipment_ID", "Delay_Risk_Probability", "Risk_Level"]],
        on="Shipment_ID", how="left",
    )


def validate_output_schema(fact_with_ml: pd.DataFrame, fact: pd.DataFrame,
                           tables: dict) -> None:
    """The .pbix relationships depend on the column names and their order.

    Do not let that break silently: Power BI shows blank measures rather than an
    error when a relationship stops resolving.
    """
    assert len(fact_with_ml) == len(fact), "The row count changed - possible join error!"
    assert fact_with_ml["Shipment_ID"].is_unique, "Shipment_ID (PK) uniqueness was broken!"
    assert fact_with_ml["Delay_Risk_Probability"].notna().all(), \
        "A missing risk score was found!"
    assert list(fact_with_ml.columns) == EXPECTED_COLUMNS, (
        f"The column structure changed! Power BI relationships would break.\n"
        f"  Expected: {EXPECTED_COLUMNS}\n  Got     : {list(fact_with_ml.columns)}"
    )
    assert fact_with_ml["Vendor_ID"].isin(tables["dim_vendor"]["Vendor_ID"]).all(), \
        "Invalid Vendor_ID foreign key!"
    assert fact_with_ml["Route_ID"].isin(tables["dim_route"]["Route_ID"]).all(), \
        "Invalid Route_ID foreign key!"
    assert fact_with_ml["Date_ID"].isin(tables["dim_date"]["Date_ID"]).all(), \
        "Invalid Date_ID foreign key!"


# ---------------------------------------------------------------------------
# 8) Feature Importance
# ---------------------------------------------------------------------------
def group_name(feature_name: str) -> str:
    """Collapse a one-hot dummy back onto its parent column."""
    for cat in CATEGORICAL_FEATURES:
        if feature_name.startswith(cat + "_"):
            return cat
    return feature_name


def impurity_importance(eval_model) -> pd.Series:
    """Averaged across the CalibratedClassifierCV sub-models, grouped by parent column.

    Impurity importance systematically INFLATES continuous, high-cardinality
    columns: a tree finds many candidate split points in such a column. Reported
    for contrast only - see permutation_importance_table.
    """
    folds = eval_model.calibrated_classifiers_
    importances = np.mean(
        [cc.estimator.named_steps["classifier"].feature_importances_ for cc in folds],
        axis=0)
    ohe = folds[0].estimator.named_steps["preprocessor"].named_transformers_["cat"]
    names = NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))

    return (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .assign(Group=lambda d: d["Feature"].apply(group_name))
        .groupby("Group")["Importance"].sum()
        .sort_values(ascending=False)
        .round(4)
    )


def permutation_importance_table(eval_model, X_test, y_test) -> pd.DataFrame:
    """The authoritative importance table.

    Measured on this dataset: Weight_tons does NOT APPEAR AT ALL in the delay
    process of generate_logistics_data.py - it is pure noise - yet impurity
    importance ranks it ABOVE Weather_Condition, a genuinely strong driver.
    Permutation importance shuffles the column and looks at how much the model
    actually degrades, which recovers the correct ordering.

    `eval_model` must be the model fitted on the training split only; permuting a
    model that was refitted on all the data would be a silent leak (measured: 0.24
    instead of 0.14).
    """
    perm = permutation_importance(
        eval_model, X_test, y_test, n_repeats=20,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "Feature": FEATURE_COLS,
            "AUC_drop": perm.importances_mean,
            "std": perm.importances_std,
        })
        .sort_values("AUC_drop", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 9) Production model + metadata
# ---------------------------------------------------------------------------
def train_production_model(X, y) -> CalibratedClassifierCV:
    """Fit on ALL the data - correct for a model about to be deployed.

    The metrics reported elsewhere are honest out-of-sample measurements; they do
    NOT come from this final fit.
    """
    model = build_model()
    model.fit(X, y)
    return model


def build_metadata(df: pd.DataFrame, y, metrics: dict,
                   fingerprint: dict | None = None) -> dict:
    """Everything a consumer needs so it never has to hardcode anything.

    `fingerprint` records which data this model was trained on, so a consumer can
    detect that the data has since been regenerated without a retrain. Computed by
    config.compute_data_fingerprint() AFTER the scored table has been written.
    """
    return {
        "model_name": "logistics_delay_risk",
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),

        # The consumer reads the column order from here - it must NOT hardcode it.
        "feature_order": list(FEATURE_COLS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        # The option lists in the UI can also be driven from here.
        "categorical_levels": {
            c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max())}
            for c in NUMERIC_FEATURES
        },

        # Risk thresholds - derived from the cost matrix, not from percentiles.
        "thresholds": {
            "high_risk": HIGH_RISK_THRESHOLD,
            "medium_risk": MEDIUM_RISK_THRESHOLD,
            "cost_fn_over_fp": COST_FN_OVER_FP,
            "derivation": "p* = 1 / (1 + C_FN/C_FP); high_risk = the 1:1 ratio threshold",
        },

        # Metrics: all OUT-OF-SAMPLE. These are not the saved model's performance
        # on its own training data.
        "metrics": metrics,

        "training_data": {
            "n_rows": int(len(df)),
            "positive_rate": round(float(y.mean()), 4),
            "date_min": df["Full_Date"].min().strftime("%Y-%m-%d"),
            "date_max": df["Full_Date"].max().strftime("%Y-%m-%d"),
            "is_synthetic": True,
        },

        # Which data this model was trained on. The Streamlit app recomputes these
        # at startup and warns if they no longer match - see
        # config.compute_data_fingerprint / compare_data_fingerprint.
        "training_data_sha256": (fingerprint or {}).get("scored_table"),
        "source_tables_sha256": (fingerprint or {}).get("source_tables"),

        # A version mismatch breaks pickle loading; let the consumer check.
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },

        "caveats": [
            "Trained on synthetic data; does not represent real-world performance.",
            "Probabilities were calibrated with isotonic regression (ECE ~0.03), but "
            "reliability is low above p>0.8 because of sparse observations.",
            "Thresholds are derived from a 4:1 cost assumption; that ratio has not "
            "been validated against real SLA penalty schedules.",
            "CO2_Emission_kg is NOT an output of this model; it is computed by a "
            "deterministic formula (see generate_logistics_data.py).",
        ],
    }


def save_production_model(model, metadata: dict,
                          model_dir: Path = MODEL_DIR) -> tuple:
    """Persist the WHOLE pipeline plus its metadata in one joblib file.

    Saving only the classifier would force the consumer to rebuild the encoding by
    hand - the classic source of train/serve skew: the category order, the
    unknown-category behaviour or the column order drifts slightly and the model
    silently predicts the wrong thing.

    compress=3: CalibratedClassifierCV holds 5 sub-models x 300 trees = 1,500
    trees, which is 43 MB uncompressed. That is close to GitHub's 50 MB warning
    threshold and would bloat the repository history on every retrain. The
    compression is LOSSLESS (measured: predictions identical bit for bit) and
    brings the file down to 8.7 MB, with a 0.4 s load time.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "production_risk_model.pkl"
    meta_path = model_dir / "production_risk_model_metadata.json"

    joblib.dump({"model": model, "metadata": metadata}, model_path, compress=3)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return model_path, meta_path


def verify_saved_model(model_path: Path, in_memory_model, feature_order: list) -> float:
    """Reload from disk and confirm it predicts identically.

    Closes the gap between "I saved it" and "it is usable".
    """
    loaded = joblib.load(model_path)
    smoke = pd.DataFrame([{
        "Distance_km": 450.0, "Weight_tons": 12.0, "Vendor_Rating": 4.0,
        "Weather_Condition": "Storm", "Traffic_Density": "High",
        "Vehicle_Type": "Diesel Truck",
    }])[feature_order]
    p = float(loaded["model"].predict_proba(smoke)[0, 1])
    assert 0.0 <= p <= 1.0, "The reloaded model produced an invalid probability!"
    original = float(in_memory_model.predict_proba(smoke)[0, 1])
    assert abs(p - original) < 1e-9, \
        "The reloaded model disagrees with the in-memory one!"
    return p


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    tables = load_star_schema()
    print(f"Fact_Shipments : {tables['fact'].shape}")
    print(f"Dim_Vendor     : {tables['dim_vendor'].shape}")
    print(f"Dim_Route      : {tables['dim_route'].shape}")
    print(f"Dim_Date       : {tables['dim_date'].shape}")

    df = add_target(build_analytical_dataset(tables))
    print(f"Joined analytical dataset: {df.shape}")
    print(f"Date range: {df.Full_Date.min():%Y-%m-%d} .. {df.Full_Date.max():%Y-%m-%d}\n")
    print("Target variable distribution:")
    print(df["is_delayed"].value_counts(normalize=True).round(3), "\n")

    X, y, groups = df[FEATURE_COLS], df["is_delayed"], df["Vendor_ID"]
    dates = df["Full_Date"]
    model = build_model()

    # --- 4) Chronological evaluation ---------------------------------------
    train_idx, test_idx = chronological_split(df)
    ev = evaluate_chronologically(model, X, y, df, train_idx, test_idx)
    # Keep a copy that never saw the test set, for permutation importance.
    eval_model = deepcopy(ev["model"])

    print(f"Train: {len(train_idx)} rows | {dates.iloc[train_idx].min():%Y-%m-%d} .. "
          f"{dates.iloc[train_idx].max():%Y-%m-%d}")
    print(f"Test : {len(test_idx)} rows | {dates.iloc[test_idx].min():%Y-%m-%d} .. "
          f"{dates.iloc[test_idx].max():%Y-%m-%d}  (entirely AFTER the training period)\n")
    print("--- Test set performance report (chronological) ---")
    print(ev["report"])
    chrono_auc = ev["roc_auc"]
    print(f"ROC-AUC (chronological holdout): {chrono_auc:.3f}\n")

    print("--- Train vs Test distribution shift ---")
    print((weather_distribution_shift(df, train_idx, test_idx) * 100).round(1).to_string())
    print(f"\nDelay rate  train {y.iloc[train_idx].mean() * 100:.1f}% -> "
          f"test {y.iloc[test_idx].mean() * 100:.1f}%")
    print("Seasons in the test period    :",
          df.iloc[test_idx]["Season"].value_counts().to_dict())
    print("Seasons in the training period:",
          df.iloc[train_idx]["Season"].value_counts().to_dict())

    print("\n--- Split strategy comparison (same data, same model) ---")
    for label, auc in compare_split_strategies(model, X, y, groups).items():
        print(f"  {label:<30}: ROC-AUC {auc:.3f}")

    ts_results = chronological_cv(model, X, y, dates)
    for i, auc, d_min, d_max, rate in ts_results:
        print(f"  Chronological fold {i}: AUC {auc:.3f}  (test "
              f"{d_min:%Y-%m} .. {d_max:%Y-%m}, delay rate {rate * 100:.0f}%)")
    ts_aucs = [r[1] for r in ts_results]
    print(f"  {'Chronological CV mean':<30}: ROC-AUC {np.mean(ts_aucs):.3f} "
          f"± {np.std(ts_aucs):.3f}\n")

    # --- 5) Walk-forward scoring -------------------------------------------
    proba, warm_up_mask = score_walk_forward(model, X, y, groups)
    if warm_up_mask.any():
        print(f"Walk-forward scoring: {(~warm_up_mask).sum()} rows "
              f"({(~warm_up_mask).mean() * 100:.0f}%) scored using PAST data only.")
        print(f"The first {warm_up_mask.sum()} rows (warm-up period, "
              f"{dates[warm_up_mask].min():%Y-%m-%d} .. "
              f"{dates[warm_up_mask].max():%Y-%m-%d}) have no prior data;\n"
              f"  they were filled with vendor-grouped out-of-fold scores.\n")

    q = scoring_quality(y, proba)
    print("--- Out-of-fold scoring quality ---")
    print(f"ROC-AUC : {q['roc_auc']:.3f}   (ranking ability)")
    print(f"Brier   : {q['brier']:.4f}  (lower = better; a model that always predicts "
          f"the base rate scores: {q['baseline_brier']:.4f})")
    print(f"ECE     : {q['ece']:.4f}  (it was ~0.137 before calibration)\n")

    # --- 6) Risk tiers ------------------------------------------------------
    df = assign_risk_levels(df, proba)
    print(f"Risk thresholds: High >= {HIGH_RISK_THRESHOLD} | "
          f"Medium >= {MEDIUM_RISK_THRESHOLD}  (C_FN:C_FP = {COST_FN_OVER_FP:.0f}:1)")
    print("Risk_Level distribution:")
    print(df["Risk_Level"].value_counts().to_string(), "\n")
    print(f"High Risk Rate % (the Power BI KPI equivalent): "
          f"{(df['Risk_Level'] == 'High Risk').mean() * 100:.1f}%")
    print(f"Flagged for action (High + Medium)            : "
          f"{(df['Delay_Risk_Probability'] >= MEDIUM_RISK_THRESHOLD).mean() * 100:.1f}%\n")

    print("--- Calibration check: predicted vs realised ---")
    print(calibration_check(df).round(3).to_string())
    print("If the deviation is close to zero, the score really is a probability.\n")

    print("--- Threshold comparison (C_FN:C_FP = 4:1) ---")
    p_arr, y_arr = df["Delay_Risk_Probability"].values, y.values
    for label, th in [("Old percentile threshold (0.65)", 0.65),
                      (f"Cost-optimal ({MEDIUM_RISK_THRESHOLD})", MEDIUM_RISK_THRESHOLD)]:
        c = alert_cost(p_arr, y_arr, th)
        print(f"  {label:<32} alerts {c['alert_rate'] * 100:4.1f}% | "
              f"caught {c['tp']}/{c['tp'] + c['fn']} "
              f"({c['tp'] / (c['tp'] + c['fn']) * 100:.0f}%) | "
              f"false alarms {c['fp']:3d} | cost {c['cost']:.0f}")
    print()

    # --- 7) Persist the scored fact table ----------------------------------
    fact_with_ml = build_fact_with_ml(tables["fact"], df)
    validate_output_schema(fact_with_ml, tables["fact"], tables)
    output_path = PROCESSED_DIR / "Fact_Shipments_with_ML.csv"
    fact_with_ml.to_csv(output_path, index=False)
    print(f"Saved: {output_path}  (shape: {fact_with_ml.shape})")
    print("Schema validation passed: column names/order and all foreign keys preserved.")
    print("\n--- Fact_Shipments_with_ML.csv preview ---")
    print(fact_with_ml.head())

    # --- 8) Feature importance ---------------------------------------------
    print("\n--- Feature Importance (grouped, impurity/Gini) ---")
    print(impurity_importance(eval_model).to_string())

    perm_df = permutation_importance_table(eval_model, ev["X_test"], ev["y_test"])
    print("\n--- Permutation Importance (ROC-AUC drop on the test set) ---")
    print(perm_df.round(4).to_string(index=False))
    print("\nNOTE: if the two tables disagree, permutation importance is authoritative.")
    print("      Weight_tons and Vehicle_Type play no part in the delay process; they")
    print("      are control variables kept in the model deliberately (see README).")

    # --- 9) Production model ------------------------------------------------
    production_model = train_production_model(X, y)
    metrics = {
        "oof_roc_auc": round(float(q["roc_auc"]), 4),
        "oof_brier": round(float(q["brier"]), 4),
        "oof_ece": round(float(q["ece"]), 4),
        "chronological_holdout_roc_auc": round(float(chrono_auc), 4),
        "chronological_cv_roc_auc_mean": round(float(np.mean(ts_aucs)), 4),
        "chronological_cv_roc_auc_std": round(float(np.std(ts_aucs)), 4),
    }
    # Fingerprint the data AFTER the scored table has been written, so the hash
    # describes exactly the files on disk that this model belongs to.
    fingerprint = compute_data_fingerprint(PROCESSED_DIR)
    metadata = build_metadata(df, y, metrics, fingerprint)
    model_path, meta_path = save_production_model(production_model, metadata)

    print("\n--- Production model saved ---")
    print(f"{model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{meta_path}   (human-readable copy)")
    print(f"Data fingerprint recorded: scored table "
          f"{(fingerprint['scored_table'] or '?')[:12]}… | source tables "
          f"{(fingerprint['source_tables'] or '?')[:12]}…")

    p = verify_saved_model(model_path, production_model, metadata["feature_order"])
    print(f"Verification: model reloaded, sample prediction = {p:.3f} "
          f"(identical to the in-memory model) ✓")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Fabric notes
# ---------------------------------------------------------------------------
# - This script runs as-is with pandas inside a Microsoft Fabric Notebook cell.
# - If you are connected to a Lakehouse and want Delta tables instead of CSVs,
#   swap the reads in load_star_schema() for:
#     spark.read.format("delta").load("Tables/Fact_Shipments").toPandas()
#   and the write in main() for:
#     spark.createDataFrame(fact_with_ml).write.format("delta") \
#         .mode("overwrite").save("Tables/Fact_Shipments_with_ML")
# - On large datasets (> a few million rows) a distributed Random Forest via
#   SynapseML is worth considering; note, however, that distributed equivalents
#   of CalibratedClassifierCV and StratifiedGroupKFold must be built by hand.
