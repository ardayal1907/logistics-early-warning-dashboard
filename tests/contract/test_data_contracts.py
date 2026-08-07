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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, not a flaky test. generate.build_base_columns draws "
        "Distance_km independently of the (Origin, Destination) pair, so the "
        "same city pair carries wildly different distances. This is expected to "
        "fail until the generator uses a 20x20 distance matrix with +/-5% noise. "
        "strict=True: when that lands, this turns RED so the marker cannot be "
        "left behind."
    ),
)
def test_distance_is_a_function_of_the_city_pair(raw_data):
    """Distance between two cities does not depend on which shipment you look at.

    Deliberately left failing. Forcing it green - by widening the bound or
    deleting the rule - would hide the defect it exists to name.
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
    assert len(violations) == 328, "city pairs whose distance spread is inconsistent"

    istanbul_denizli = report[
        (report["Origin"] == "Istanbul") & (report["Destination"] == "Denizli")
    ].iloc[0]
    assert int(istanbul_denizli["n"]) == 6
    assert istanbul_denizli["min_km"] == pytest.approx(67.9)
    assert istanbul_denizli["max_km"] == pytest.approx(1169.2)
