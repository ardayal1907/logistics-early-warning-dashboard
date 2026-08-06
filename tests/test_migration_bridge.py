"""The guard that makes this migration safe to do in steps.

`src/config.py` and `src/logistics/domain/` both define the emission factors,
the carbon price and the risk thresholds right now. That duplication is exactly
what `config.py` was created to abolish, so it is tolerable only because it is
temporary AND machine-checked.

This file is the machine check. Every shared value must agree, in both
directions, and every shared function must return identical results across a
sweep of the input space. Change one side without the other and the build goes
red immediately.

The duplication ends at migration step 2, when `src/config.py` becomes a
re-export shim:

    from logistics.domain.carbon import (
        BASE_EMISSION, EMISSION_FACTOR, MIN_CO2_KG,
        TRAFFIC_CO2_MULT, WEATHER_CO2_MULT, compute_co2_kg,
    )
    from logistics.domain.risk import classify_risk
    from logistics.infrastructure.fingerprint import (...)
    CARBON_PRICE_PER_TON = DEFAULT_CARBON_PRICE_PER_TON
    ...

At that point the identity assertions in test_single_source_of_truth.py
(`gen.EMISSION_FACTOR is config.EMISSION_FACTOR`) still hold, because a
re-export binds the same object - which is why that suite can stay untouched.
DELETE THIS FILE when the shim lands; keeping it would only assert that a module
agrees with itself.
"""

from __future__ import annotations

import itertools

import pytest

import config  # the legacy flat module
from logistics.domain import carbon as new_carbon
from logistics.domain import risk as new_risk
from logistics.infrastructure import fingerprint as new_fingerprint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "legacy_name, new_table",
    [
        ("EMISSION_FACTOR", new_carbon.EMISSION_FACTOR),
        ("BASE_EMISSION", new_carbon.BASE_EMISSION),
        ("WEATHER_CO2_MULT", new_carbon.WEATHER_CO2_MULT),
        ("TRAFFIC_CO2_MULT", new_carbon.TRAFFIC_CO2_MULT),
    ],
)
def test_reference_tables_agree(legacy_name, new_table):
    legacy = getattr(config, legacy_name)

    # Compared by string key, because the new tables are keyed by StrEnum
    # members. StrEnum hashes as its value, so this is a like-for-like check.
    assert {str(k): v for k, v in new_table.items()} == dict(legacy), (
        f"{legacy_name} has diverged between src/config.py and "
        f"logistics.domain.carbon. Update BOTH or complete migration step 2."
    )


def test_scalar_constants_agree():
    assert config.MIN_CO2_KG == new_carbon.MIN_CO2_KG
    assert config.CARBON_PRICE_PER_TON == new_carbon.DEFAULT_CARBON_PRICE_PER_TON
    assert config.COST_FN_OVER_FP == new_risk.DEFAULT_COST_FN_OVER_FP
    assert config.HIGH_RISK_THRESHOLD == new_risk.DEFAULT_HIGH_RISK_THRESHOLD
    assert config.MEDIUM_RISK_THRESHOLD == new_risk.DEFAULT_MEDIUM_RISK_THRESHOLD


def test_fingerprint_table_names_agree():
    """The Power BI report and the model both depend on these exact filenames."""
    assert config.SCORED_TABLE == new_fingerprint.SCORED_TABLE
    assert list(config.SOURCE_TABLES) == list(new_fingerprint.SOURCE_TABLES)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
def test_compute_co2_agrees_across_the_whole_category_grid():
    """216 combinations - every category triple at three distances and weights.

    A value check would miss a sign error in one multiplier; sweeping the grid
    does not.
    """
    checked = 0
    for vehicle, weather, traffic, distance, weight in itertools.product(
        config.EMISSION_FACTOR,
        config.WEATHER_CO2_MULT,
        config.TRAFFIC_CO2_MULT,
        (50.0, 625.0, 1200.0),
        (1.0, 12.0, 25.0),
    ):
        legacy = config.compute_co2_kg(distance, weight, vehicle, weather, traffic)
        current = new_carbon.compute_co2_kg(distance, weight, vehicle, weather, traffic)
        assert current == pytest.approx(legacy, rel=1e-12), (
            f"CO2 diverged for {vehicle}/{weather}/{traffic} "
            f"at {distance} km, {weight} t: {legacy} vs {current}"
        )
        checked += 1

    assert checked == 3 * 4 * 3 * 3 * 3


@pytest.mark.parametrize(
    "p",
    [0.0, 0.0001, 0.19, 0.199999, 0.20, 0.2001, 0.35, 0.499999, 0.50, 0.5001, 0.99, 1.0],
)
def test_classify_risk_agrees_including_at_the_boundaries(p):
    assert new_risk.classify_risk(p) == config.classify_risk(p)


def test_classify_risk_agrees_with_custom_thresholds():
    """The Streamlit app passes thresholds read from the model metadata, so the
    parameterised path matters as much as the default one."""
    for p in (0.10, 0.30, 0.85, 0.95):
        assert new_risk.classify_risk(p, 0.9, 0.8) == config.classify_risk(p, 0.9, 0.8)


def test_fingerprint_functions_agree(tmp_path):
    (tmp_path / config.SCORED_TABLE).write_bytes(b"scored\n")
    for i, name in enumerate(config.SOURCE_TABLES):
        (tmp_path / name).write_bytes(f"table-{i}\n".encode())

    assert new_fingerprint.compute_data_fingerprint(
        tmp_path
    ) == config.compute_data_fingerprint(tmp_path)

    recorded = {"scored_table": "a", "source_tables": "b"}
    current = {"scored_table": "CHANGED", "source_tables": "b"}
    assert new_fingerprint.compare_data_fingerprint(
        recorded, current
    ) == config.compare_data_fingerprint(recorded, current)


def test_sha256_agrees_byte_for_byte(tmp_path):
    payload = tmp_path / "sample.csv"
    payload.write_bytes(b"Shipment_ID,Risk_Level\nSHP-00001,Low Risk\n")
    assert new_fingerprint.sha256_file(payload) == config.sha256_file(payload)
