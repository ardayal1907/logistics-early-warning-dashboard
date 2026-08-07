"""Compatibility shim — the constants now live in the `logistics` package.

This module used to BE the single source of truth. It no longer holds any
definition of its own: every name below is re-exported from

    logistics.domain.carbon          the CO2 model
    logistics.domain.risk            the risk tiers and the cost assumption
    logistics.infrastructure.fingerprint   data/model synchronisation

Why a re-export rather than a rewrite of the callers: `from X import name`
binds the *same object*, so `generate_logistics_data.EMISSION_FACTOR is
config.EMISSION_FACTOR` remains true. That identity is asserted by
tests/test_single_source_of_truth.py and is the reason the existing suite
survives this move untouched.

`src/app.py`, `src/generate_logistics_data.py` and
`src/ml_delay_risk_pipeline.py` still import from here. They move into the
package in migration steps 3 and 4; this file stays afterwards, because
Streamlit Cloud's entrypoint is a file path.

⚠️ ONE COPY REMAINS OUTSIDE PYTHON. The Power BI `Carbon Tax Impact ($)`
   measure hardcodes the same $50/ton figure in DAX:

       Carbon Tax Impact ($) = [Total CO2 Tons] * 50

   Power BI cannot import from Python, so if the carbon price changes in
   logistics.domain.carbon, that DAX measure MUST be edited by hand in the
   .pbix file. This is a known, deliberate limitation — documented rather than
   hidden, and cross-checked by tests/test_dax_constants.py.
"""

from logistics.domain.carbon import (
    BASE_EMISSION,
    EMISSION_FACTOR,
    MIN_CO2_KG,
    TRAFFIC_CO2_MULT,
    WEATHER_CO2_MULT,
    compute_co2_kg,
)
from logistics.domain.carbon import DEFAULT_CARBON_PRICE_PER_TON as CARBON_PRICE_PER_TON
from logistics.domain.risk import DEFAULT_COST_FN_OVER_FP as COST_FN_OVER_FP
from logistics.domain.risk import DEFAULT_HIGH_RISK_THRESHOLD as HIGH_RISK_THRESHOLD
from logistics.domain.risk import DEFAULT_MEDIUM_RISK_THRESHOLD as MEDIUM_RISK_THRESHOLD
from logistics.domain.risk import (
    classify_risk,
)
from logistics.infrastructure.fingerprint import (
    SCORED_TABLE,
    SOURCE_TABLES,
    compare_data_fingerprint,
    compute_data_fingerprint,
    sha256_file,
)

__all__ = [
    "BASE_EMISSION",
    "CARBON_PRICE_PER_TON",
    "COST_FN_OVER_FP",
    "EMISSION_FACTOR",
    "HIGH_RISK_THRESHOLD",
    "MEDIUM_RISK_THRESHOLD",
    "MIN_CO2_KG",
    "SCORED_TABLE",
    "SOURCE_TABLES",
    "TRAFFIC_CO2_MULT",
    "WEATHER_CO2_MULT",
    "classify_risk",
    "compare_data_fingerprint",
    "compute_co2_kg",
    "compute_data_fingerprint",
    "sha256_file",
]
