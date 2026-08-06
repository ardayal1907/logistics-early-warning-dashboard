"""Risk tiering must be exact at the boundaries.

Off-by-one-epsilon bugs at a threshold are easy to introduce (`>` vs `>=`) and
invisible in aggregate statistics: a handful of shipments quietly land in the wrong
tier. These tests pin the boundary semantics down.
"""

import pytest

import config

HIGH = config.HIGH_RISK_THRESHOLD      # 0.50
MED = config.MEDIUM_RISK_THRESHOLD     # 0.20


@pytest.mark.parametrize("p, expected", [
    (0.00, "Low Risk"),
    (0.19, "Low Risk"),
    (0.199999, "Low Risk"),
    (MED, "Medium Risk"),          # exactly on the boundary -> the higher tier
    (0.20, "Medium Risk"),
    (0.35, "Medium Risk"),
    (0.499999, "Medium Risk"),
    (HIGH, "High Risk"),           # exactly on the boundary -> the higher tier
    (0.50, "High Risk"),
    (0.99, "High Risk"),
    (1.00, "High Risk"),
])
def test_boundaries_are_inclusive_on_the_lower_edge(p, expected):
    assert config.classify_risk(p) == expected


def test_thresholds_derive_from_the_cost_ratio():
    """MEDIUM_RISK_THRESHOLD must be p* = 1 / (1 + C_FN/C_FP), not a magic number."""
    expected = round(1.0 / (1.0 + config.COST_FN_OVER_FP), 2)
    assert expected == MED, (
        f"MEDIUM_RISK_THRESHOLD is {MED} but the {config.COST_FN_OVER_FP}:1 cost "
        f"ratio implies {expected}. The threshold must stay derived from the cost "
        "assumption rather than being hardcoded."
    )


def test_high_threshold_corresponds_to_a_one_to_one_ratio():
    assert pytest.approx(1.0 / (1.0 + 1.0)) == HIGH, \
        "HIGH_RISK_THRESHOLD should be the 1:1 cost-ratio threshold (0.50)."


def test_tiers_are_ordered():
    assert 0.0 < MED < HIGH < 1.0


def test_custom_thresholds_override_the_defaults():
    """The Streamlit app passes thresholds read from the model metadata."""
    assert config.classify_risk(0.30, 0.9, 0.8) == "Low Risk"
    assert config.classify_risk(0.85, 0.9, 0.8) == "Medium Risk"
    assert config.classify_risk(0.95, 0.9, 0.8) == "High Risk"


def test_scored_data_matches_the_thresholds(fact_with_ml):
    """Every Risk_Level in the shipped CSV must agree with classify_risk."""
    recomputed = fact_with_ml.Delay_Risk_Probability.apply(config.classify_risk)
    mismatches = (recomputed != fact_with_ml.Risk_Level).sum()
    assert mismatches == 0, (
        f"{mismatches} rows in Fact_Shipments_with_ML.csv carry a Risk_Level that "
        "does not follow from their Delay_Risk_Probability."
    )
