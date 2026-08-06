"""
Shared constants and helpers — the single source of truth.
========================================================================
Before this module existed, the emission factors were declared twice (once in
`generate_logistics_data.py`, once in `app.py`) and the carbon price three times
(`app.py`, the Power BI DAX measure, and the README). Duplicated constants do not
stay in sync: someone edits one, the others drift silently, and the Streamlit demo
starts disagreeing with the dashboard for reasons nobody can see.

Everything the CO2 calculation needs now lives here, and both the generator and
the Streamlit app import from this module.

⚠️ ONE COPY REMAINS OUTSIDE PYTHON. The Power BI `Carbon Tax Impact ($)` measure
   hardcodes the same $50/ton figure in DAX:

       Carbon Tax Impact ($) = [Total CO2 Tons] * 50

   Power BI cannot import from Python, so if CARBON_PRICE_PER_TON changes here,
   that DAX measure MUST be edited by hand in the .pbix file. This is a known,
   deliberate limitation - documented rather than hidden.
"""

import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# CO2 emission model
# ---------------------------------------------------------------------------
# Realistic CO2 emission factor per vehicle type (kg CO2 / ton-km).
# Reference ranges (approximate, informed by published figures):
#   Diesel truck (heavy load):  ~0.10 - 0.16 kgCO2/ton-km
#   Electric semi-trailer:      ~0.02 - 0.045 kgCO2/ton-km (depends on grid carbon intensity)
#   Hybrid van:                 ~0.06 - 0.10 kgCO2/ton-km
EMISSION_FACTOR = {
    "Diesel Truck": 0.13,
    "Electric Semi": 0.035,
    "Hybrid Van": 0.08,
}

# Fixed (idle/start-up) emission component per vehicle type (kg) - the extra
# emissions from engine start-up and loading/unloading.
BASE_EMISSION = {
    "Diesel Truck": 8.0,
    "Electric Semi": 1.5,
    "Hybrid Van": 4.0,
}

# Bad weather and heavy traffic -> more idling / inefficient driving -> extra emissions.
WEATHER_CO2_MULT = {"Normal": 1.00, "Rain": 1.03, "Storm": 1.08, "Snow": 1.06}
TRAFFIC_CO2_MULT = {"Low": 1.00, "Medium": 1.04, "High": 1.10}

# Floor applied to the computed emission figure (kg), so noise can never drive it
# to zero or negative.
MIN_CO2_KG = 0.5

# ---------------------------------------------------------------------------
# Carbon pricing
# ---------------------------------------------------------------------------
# A commonly cited mid-range figure used in voluntary carbon pricing and internal
# carbon fee models. Deliberately isolated so it can be swapped for a
# jurisdiction-specific carbon tax or an internal shadow price.
#
# ⚠️ Mirrored in the Power BI DAX measure - see the module docstring.
CARBON_PRICE_PER_TON = 50.0

# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
# Cost assumption:
#   FALSE ALARM (FP)  -> a planner checks the shipment, calls the vendor and
#                        reserves buffer capacity. ~1 unit.
#   MISSED DELAY (FN) -> an SLA breach: contractual penalty + expedited/re-routing
#                        cost + churn risk.  -> 4 units.
#
# With calibrated probabilities the expected-cost-minimising threshold is analytic:
#     intervene  <=>  p * C_FN > (1 - p) * C_FP
#     p* = C_FP / (C_FP + C_FN) = 1 / (1 + ratio)
# The formula is invalid on uncalibrated probabilities - calibration is a
# precondition, not a nicety.
COST_FN_OVER_FP = 4.0

#   High Risk   : worth intervening even if a false alarm and a missed delay cost
#                 the SAME (the 1:1 ratio threshold)
#   Medium Risk : worth intervening under the 4:1 assumption
#   Low Risk    : not worth intervening even at 4:1
HIGH_RISK_THRESHOLD = 0.50
MEDIUM_RISK_THRESHOLD = round(1.0 / (1.0 + COST_FN_OVER_FP), 2)   # 4:1 -> 0.20


# ---------------------------------------------------------------------------
# Data / model synchronisation
# ---------------------------------------------------------------------------
# A model artefact and the data it was trained on can drift apart without anyone
# noticing: regenerate the data, forget to retrain, and the Streamlit demo happily
# serves stale predictions against fresh data. Nothing errors, nothing looks wrong.
#
# The pipeline records a fingerprint of the data at training time; the app
# recomputes it at startup and warns on a mismatch. Both sides call the functions
# below, so the two can never disagree about how the fingerprint is computed.

# The scored fact table the Power BI report reads and the app compares against.
SCORED_TABLE = "Fact_Shipments_with_ML.csv"

# The tables the model actually learns from. Hashing these is what catches the
# "data regenerated but model not retrained" case: SCORED_TABLE alone would not,
# because it is written by the same run that trains the model.
SOURCE_TABLES = ["Fact_Shipments.csv", "Dim_Vendor.csv", "Dim_Route.csv",
                 "Dim_Date.csv"]


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large files stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_data_fingerprint(processed_dir) -> dict:
    """Fingerprint the dataset a model was (or would be) trained on.

    Returns {"scored_table": <sha256 or None>, "source_tables": <sha256 or None>}.
    A missing file yields None for that entry rather than raising, so a partial
    checkout degrades to "cannot verify" instead of crashing the app.

    `source_tables` is a hash over the concatenated per-file hashes, in the fixed
    order of SOURCE_TABLES, so it is stable and order-independent of the filesystem.
    """
    processed_dir = Path(processed_dir)

    scored_path = processed_dir / SCORED_TABLE
    scored = sha256_file(scored_path) if scored_path.exists() else None

    parts = []
    for name in SOURCE_TABLES:
        p = processed_dir / name
        if not p.exists():
            parts = None
            break
        parts.append(f"{name}:{sha256_file(p)}")
    source = hashlib.sha256("\n".join(parts).encode()).hexdigest() if parts else None

    return {"scored_table": scored, "source_tables": source}


def compare_data_fingerprint(recorded: dict, current: dict) -> list:
    """Return a list of human-readable mismatch descriptions (empty = in sync).

    An entry that is None on either side is skipped: "cannot verify" is not the
    same as "mismatch", and warning about the former would train users to ignore
    the warning.
    """
    labels = {
        "scored_table": f"the scored fact table ({SCORED_TABLE})",
        "source_tables": "the star-schema tables the model was trained on",
    }
    mismatches = []
    for key, label in labels.items():
        rec, cur = (recorded or {}).get(key), (current or {}).get(key)
        if rec and cur and rec != cur:
            mismatches.append(label)
    return mismatches


def classify_risk(p: float, high_threshold: float = HIGH_RISK_THRESHOLD,
                  medium_threshold: float = MEDIUM_RISK_THRESHOLD) -> str:
    """Map a calibrated probability onto a risk tier.

    The boundaries are inclusive on the lower edge: a probability exactly equal to
    `high_threshold` is High Risk, and one exactly equal to `medium_threshold` is
    Medium Risk.

    The thresholds are parameters rather than hardcoded lookups so that the
    Streamlit app can pass the values stored in the model's metadata. If a model is
    retrained with different thresholds, the consuming app follows the model rather
    than this module.
    """
    if p >= high_threshold:
        return "High Risk"
    if p >= medium_threshold:
        return "Medium Risk"
    return "Low Risk"


def compute_co2_kg(distance_km: float, weight_tons: float, vehicle: str,
                   weather: str, traffic: str) -> float:
    """Deterministic CO2 estimate for a single shipment, in kg.

    CO2 = (distance x tonnage x emission factor + fixed base) x weather x traffic

    This is NOT a model prediction; it is an engineering calculation. The data
    generator adds a small random noise term on top of this for realism, but the
    Streamlit demo uses this function directly so that identical inputs always
    yield identical output.
    """
    base = distance_km * weight_tons * EMISSION_FACTOR[vehicle] + BASE_EMISSION[vehicle]
    return max(base * WEATHER_CO2_MULT[weather] * TRAFFIC_CO2_MULT[traffic], MIN_CO2_KG)
