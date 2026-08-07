"""The data contracts, applied to the data that is actually committed.

Most of this file is unremarkable: the shipped tables must satisfy the schemas
that describe them. The interesting part is at the bottom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from logistics.contracts import shipments as contracts
from logistics.domain.enums import (
    RiskLevel,
    TrafficDensity,
    VehicleType,
    WeatherCondition,
)

# --- the categorical domains are derived, not retyped -----------------------

@pytest.mark.parametrize(
    "declared, enum",
    [
        (contracts.VEHICLE_TYPES, VehicleType),
        (contracts.WEATHER_CONDITIONS, WeatherCondition),
        (contracts.TRAFFIC_DENSITIES, TrafficDensity),
        (contracts.RISK_LEVELS, RiskLevel),
    ],
    ids=["vehicle", "weather", "traffic", "risk"],
)
def test_the_contract_vocabulary_comes_from_the_enum(declared, enum):
    """A second hardcoded copy here would be a fifth place for these to drift."""
    assert declared == [member.value for member in enum]


# --- the committed tables satisfy their schemas -----------------------------

def test_raw_data_satisfies_its_schema(raw_data):
    contracts.RAW_SHIPMENTS_SCHEMA.validate(raw_data, lazy=True)


def test_dim_vendor_satisfies_its_schema(dim_vendor):
    contracts.DIM_VENDOR_SCHEMA.validate(dim_vendor, lazy=True)


def test_dim_route_satisfies_its_schema(dim_route):
    contracts.DIM_ROUTE_SCHEMA.validate(dim_route, lazy=True)


def test_the_scored_table_satisfies_its_schema(fact_with_ml):
    contracts.SCORED_SHIPMENTS_SCHEMA.validate(fact_with_ml, lazy=True)


def test_a_probability_above_one_is_rejected(fact_with_ml):
    """Proof the schema can fail — a green suite otherwise proves nothing."""
    broken = fact_with_ml.copy()
    broken.loc[0, "Delay_Risk_Probability"] = 1.4
    with pytest.raises(Exception, match="Delay_Risk_Probability"):
        contracts.SCORED_SHIPMENTS_SCHEMA.validate(broken, lazy=True)


def test_an_unknown_risk_level_is_rejected(fact_with_ml):
    broken = fact_with_ml.copy()
    broken.loc[0, "Risk_Level"] = "Catastrophic Risk"
    with pytest.raises(Exception, match="Risk_Level"):
        contracts.SCORED_SHIPMENTS_SCHEMA.validate(broken, lazy=True)


# --- the cross-column rule --------------------------------------------------

def test_the_distance_rule_accepts_a_consistent_dataset():
    """The rule must be capable of passing, or its failure below means nothing."""
    consistent = pd.DataFrame({
        "Origin": ["Istanbul"] * 4 + ["Ankara"] * 4,
        "Destination": ["Denizli"] * 4 + ["Izmir"] * 4,
        "Distance_km": [600.0, 610.0, 590.0, 605.0, 580.0, 585.0, 575.0, 590.0],
    })
    assert contracts.validate_distance_consistency(consistent).empty


def test_distance_is_a_function_of_the_city_pair(raw_data):
    """Distance between two cities does not depend on which shipment you look at.

    Was xfail(strict=True) for the whole refactor series: the v1 generator drew
    Distance_km from uniform(50, 1200) with no reference to the city pair, and
    328 of 347 pairs breached the bound. The v2 generator derives the base
    distance from CITY_COORDS through ROUTE_DISTANCE_KM and applies only +/-5%
    per-shipment noise, so the rule now holds and the marker is gone.
    """
    contracts.validate_distance_consistency(raw_data)


def test_the_scale_of_the_distance_defect_is_what_we_think_it_is(raw_data):
    """Measure the defect rather than only asserting that it exists.

    This test PASSES. It pins today's numbers, so if the generator is changed
    and the spread merely shrinks instead of disappearing, the change is
    visible here rather than silently absorbed by the xfail above.
    """
    report = contracts.distance_consistency_report(raw_data)
    violations = report[report["cv"] > contracts.MAX_DISTANCE_CV]

    assert len(report) == 347, "number of city pairs with more than one shipment"
    assert len(violations) == 0, "city pairs whose distance spread is inconsistent"

    # The residual spread is the +/-5% dispatch noise and nothing else. A uniform
    # +/-5% band has cv ~ 0.029 in expectation, and small groups scatter around
    # that, so 0.06 is a ceiling on noise rather than a restatement of the 0.15
    # contract bound - it would catch a regression long before the contract does.
    assert report["cv"].max() < 0.06, "residual spread is wider than dispatch noise"
    assert report["cv"].median() == pytest.approx(0.0265, abs=0.005)

    # The pair the v1 defect was reported on: 67.9 km to 1,169.2 km across six
    # shipments of the same lane. The matrix puts the lane at 449.1 km.
    istanbul_denizli = report[
        (report["Origin"] == "Istanbul") & (report["Destination"] == "Denizli")
    ].iloc[0]
    assert int(istanbul_denizli["n"]) == 6
    assert istanbul_denizli["min_km"] == pytest.approx(427.3)
    assert istanbul_denizli["max_km"] == pytest.approx(470.4)
