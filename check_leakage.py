"""Introspect the production risk model and the scored fact table.

Answers two questions:
  1. Which columns was models/production_risk_model.pkl trained on?
  2. Where does Vehicle_Type actually live relative to
     data/processed/Fact_Shipments_with_ML.csv?

Run:  python check_leakage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "production_risk_model.pkl"
META_PATH = ROOT / "models" / "production_risk_model_metadata.json"
PROCESSED = ROOT / "data" / "processed"
FACT_ML_PATH = PROCESSED / "Fact_Shipments_with_ML.csv"


def model_feature_names(model: object) -> list[str] | None:
    """Best-effort extraction of the column names an sklearn estimator saw."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)

    # Pipeline: the first step is the one that touched raw columns.
    steps = getattr(model, "steps", None)
    if steps:
        for _, step in steps:
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None


def main() -> int:
    print("=" * 70)
    print("1. MODEL")
    print("=" * 70)

    # The .pkl is a zlib-compressed joblib dump (magic 0x78), not a plain
    # pickle, and it wraps the estimator in a dict alongside its metadata.
    payload = joblib.load(MODEL_PATH)
    embedded_meta: dict | None = None
    if isinstance(payload, dict):
        print(f"payload    : dict with keys {sorted(payload)}")
        model = payload.get("model", payload)
        embedded_meta = payload.get("metadata")
    else:
        model = payload

    print(f"path       : {MODEL_PATH.relative_to(ROOT)}")
    print(f"type       : {type(model).__name__}")

    steps = getattr(model, "steps", None)
    if steps:
        print(f"pipeline   : {' -> '.join(name for name, _ in steps)}")

    names = model_feature_names(model)
    if names is None:
        print("feature_names_in_ : (not exposed by this estimator)")
    else:
        print(f"trained on {len(names)} column(s) [from feature_names_in_]:")
        for n in names:
            print(f"  - {n}")

    meta = embedded_meta
    if meta is None and META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta is not None:
        order = meta.get("feature_order", [])
        print(f"\nmetadata feature_order ({len(order)}):")
        for n in order:
            print(f"  - {n}")
        if names is not None:
            if list(names) == list(order):
                print("  => matches feature_names_in_ exactly")
            else:
                print(f"  => MISMATCH vs feature_names_in_: {set(order) ^ set(names)}")
        print(f"categorical: {meta.get('categorical_features')}")
        print(f"numeric    : {meta.get('numeric_features')}")

    print()
    print("=" * 70)
    print("2. SCORED FACT TABLE")
    print("=" * 70)

    fact = pd.read_csv(FACT_ML_PATH)
    print(f"path  : {FACT_ML_PATH.relative_to(ROOT)}")
    print(f"shape : {fact.shape[0]} rows x {fact.shape[1]} cols")
    print("columns:")
    for c in fact.columns:
        print(f"  - {c}  ({fact[c].dtype})")

    print()
    print("=" * 70)
    print("3. WHERE IS Vehicle_Type?")
    print("=" * 70)

    hits = [c for c in fact.columns if "vehicle" in c.lower() or "type" in c.lower()]
    print(f"vehicle/type-like columns in the fact table: {hits or 'NONE'}")

    print("\nscanning the other processed tables:")
    for csv_path in sorted(PROCESSED.glob("*.csv")):
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        found = [c for c in cols if "vehicle" in c.lower()]
        marker = f"  <-- {found}" if found else ""
        print(f"  {csv_path.name:<32} {len(cols):>2} cols{marker}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("Vehicle_Type is NOT in Fact_Shipments_with_ML.csv.")
    print("It lives in Dim_Route.csv and must be joined on Route_ID.")
    print("The model's probability column in the fact table is "
          "'Delay_Risk_Probability'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
