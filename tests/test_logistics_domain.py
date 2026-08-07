"""The domain layer's rules, tested without touching a file or a model.

The existing suite covers `config.compute_co2_kg` and `config.classify_risk`
well. What it could not cover is the behaviour that did not exist yet: an
unknown category producing something better than a bare `KeyError`, and the
mathematical invariants that hold across the whole input space rather than at
eleven hand-picked points.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from logistics.domain import carbon, risk
from logistics.domain.enums import RiskLevel, TrafficDensity, VehicleType, WeatherCondition
from logistics.errors import UnknownCategoryError

_LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


# ---------------------------------------------------------------------------
# Carbon: unknown categories
# ---------------------------------------------------------------------------
# The model already survives an unseen category - OneHotEncoder(handle_unknown=
# "ignore"), covered by test_model_artifact.py::test_survives_an_unseen_category.
# The carbon path did not: it raised a bare KeyError and Streamlit rendered a
# stack trace. Adding a CNG truck to the fleet is a routine business event.
@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"vehicle": "CNG Truck"}, "vehicle type"),
        ({"weather": "Fog"}, "weather condition"),
        ({"traffic": "Gridlock"}, "traffic density"),
    ],
)
def test_unknown_category_raises_a_typed_actionable_error(kwargs, field):
    call = {
        "distance_km": 450.0,
        "weight_tons": 12.0,
        "vehicle": VehicleType.DIESEL_TRUCK,
        "weather": WeatherCondition.NORMAL,
        "traffic": TrafficDensity.MEDIUM,
    } | kwargs

    with pytest.raises(UnknownCategoryError) as excinfo:
        carbon.compute_co2_kg(**call)

    message = str(excinfo.value)
    assert field in message, "the message must name the field that failed"
    assert next(iter(kwargs.values())) in message, "it must quote the offending value"
    assert excinfo.value.known, "and list what would have been accepted"


def test_enum_and_string_keys_are_interchangeable():
    """StrEnum members hash and compare as their value, so a CSV read (plain
    strings) and typed code reach the identical reference table entry."""
    from_enum = carbon.compute_co2_kg(
        400.0, 10.0, VehicleType.HYBRID_VAN, WeatherCondition.RAIN, TrafficDensity.HIGH
    )
    from_string = carbon.compute_co2_kg(400.0, 10.0, "Hybrid Van", "Rain", "High")
    assert from_enum == from_string


# ---------------------------------------------------------------------------
# Carbon: invariants
# ---------------------------------------------------------------------------
@given(
    distance=st.floats(1, 3000, allow_nan=False),
    weight=st.floats(0.1, 40, allow_nan=False),
    extra=st.floats(0.1, 500, allow_nan=False),
)
def test_co2_is_monotone_in_distance(distance, weight, extra):
    base = carbon.compute_co2_kg(distance, weight, "Diesel Truck", "Normal", "Low")
    further = carbon.compute_co2_kg(distance + extra, weight, "Diesel Truck", "Normal", "Low")
    assert further >= base


@given(
    distance=st.floats(1, 3000, allow_nan=False),
    weight=st.floats(0.1, 40, allow_nan=False),
)
def test_vehicle_ordering_holds_everywhere(distance, weight):
    """Diesel >= Hybrid >= Electric for every load, not just on average.

    test_co2_formula.py checks this on the generated dataset's means. The
    ordering is a property of the constants, so it should hold pointwise.
    """
    args = ("Normal", "Low")
    diesel = carbon.compute_co2_kg(distance, weight, "Diesel Truck", *args)
    hybrid = carbon.compute_co2_kg(distance, weight, "Hybrid Van", *args)
    electric = carbon.compute_co2_kg(distance, weight, "Electric Semi", *args)
    assert diesel >= hybrid >= electric


@given(
    distance=st.floats(0.001, 5000, allow_nan=False),
    weight=st.floats(0.001, 60, allow_nan=False),
)
def test_floor_is_never_breached(distance, weight):
    value = carbon.compute_co2_kg(distance, weight, "Electric Semi", "Normal", "Low")
    assert value >= carbon.MIN_CO2_KG


def test_bad_conditions_never_reduce_emissions():
    """Every weather and traffic multiplier is >= 1.0 by construction."""
    assert min(carbon.WEATHER_CO2_MULT.values()) >= 1.0
    assert min(carbon.TRAFFIC_CO2_MULT.values()) >= 1.0


# ---------------------------------------------------------------------------
# The batch path is a second implementation of the same formula, so it has to be
# pinned to the scalar one - otherwise it is exactly the duplication config.py
# was created to abolish, just faster.
# ---------------------------------------------------------------------------
def test_batch_co2_matches_the_scalar_definition_on_the_full_grid():
    import itertools

    combos = list(
        itertools.product(
            carbon.EMISSION_FACTOR,
            carbon.WEATHER_CO2_MULT,
            carbon.TRAFFIC_CO2_MULT,
            (0.1, 51.1, 625.0, 1199.7, 3000.0),
            (0.01, 1.01, 12.0, 24.99, 40.0),
        )
    )
    vehicles, weathers, traffics, distances, weights = (
        [c[0] for c in combos],
        [c[1] for c in combos],
        [c[2] for c in combos],
        [c[3] for c in combos],
        [c[4] for c in combos],
    )

    batch = carbon.compute_co2_kg_many(distances, weights, vehicles, weathers, traffics)
    scalar = [
        carbon.compute_co2_kg(d, w, v, wea, t)
        for d, w, v, wea, t in zip(
            distances, weights, vehicles, weathers, traffics, strict=True
        )
    ]

    assert len(batch) == 3 * 4 * 3 * 5 * 5
    assert batch == scalar, "compute_co2_kg_many has drifted from compute_co2_kg"


def test_batch_co2_rejects_unknown_categories_too():
    with pytest.raises(UnknownCategoryError):
        carbon.compute_co2_kg_many([100.0], [5.0], ["CNG Truck"], ["Normal"], ["Low"])
    with pytest.raises(UnknownCategoryError):
        carbon.compute_co2_kg_many([100.0], [5.0], ["Hybrid Van"], ["Fog"], ["Low"])


def test_batch_co2_rejects_ragged_input():
    """`strict=True` on the zip: a length mismatch is a caller bug that would
    otherwise silently truncate the result and misalign it with the frame."""
    with pytest.raises(ValueError):
        carbon.compute_co2_kg_many([100.0, 200.0], [5.0], ["Hybrid Van"], ["Normal"], ["Low"])


def test_carbon_cost_is_linear_in_tonnage():
    assert carbon.carbon_cost(1000.0, 50.0) == pytest.approx(50.0)
    assert carbon.carbon_cost(2000.0, 50.0) == pytest.approx(100.0)
    assert carbon.carbon_cost(1000.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
@given(p1=st.floats(0, 1, allow_nan=False), p2=st.floats(0, 1, allow_nan=False))
def test_risk_tier_is_monotone_in_probability(p1, p2):
    """A higher probability may never map to a lower tier.

    test_risk_thresholds.py pins eleven specific points; monotonicity is the
    property those points are sampling, so it is worth stating directly.
    """
    low, high = sorted((p1, p2))
    assert _LEVEL_ORDER[risk.classify_risk(low)] <= _LEVEL_ORDER[risk.classify_risk(high)]


@given(ratio=st.floats(0.01, 100, allow_nan=False))
def test_threshold_derivation_matches_the_cost_algebra(ratio):
    """p* = 1 / (1 + C_FN/C_FP). Falls in (0, 1) and decreases as FN gets dearer.

    The tolerance is half a unit in the last rounded place PLUS a float-noise
    margin, not exactly half. `threshold_for_cost_ratio` rounds to two decimals,
    so the worst case is a tie exactly on the midpoint: ratio=0.6 gives
    1/1.6 = 0.625, which rounds half-to-even down to 0.62, and the residual
    evaluates to 0.005000000000000004 — just over a plain `abs=0.005`. Hypothesis
    found that example. The margin encodes "no worse than the rounding" without
    making the assertion depend on which side of a tie a float lands.
    """
    p_star = risk.threshold_for_cost_ratio(ratio)
    assert 0.0 <= p_star <= 1.0
    assert p_star == pytest.approx(1.0 / (1.0 + ratio), abs=0.005 + 1e-9)


def test_threshold_is_monotone_decreasing_in_the_cost_ratio():
    """The more a missed delay costs, the lower the bar for intervening."""
    thresholds = [risk.threshold_for_cost_ratio(r) for r in (1, 2, 4, 8, 16)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_shipped_defaults_are_derived_not_typed():
    assert risk.threshold_for_cost_ratio(
        risk.DEFAULT_COST_FN_OVER_FP
    ) == risk.DEFAULT_MEDIUM_RISK_THRESHOLD
    assert risk.threshold_for_cost_ratio(1.0) == risk.DEFAULT_HIGH_RISK_THRESHOLD


def test_zero_or_negative_cost_ratio_is_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            risk.threshold_for_cost_ratio(bad)


def test_expected_delay_cost_is_linear_in_probability():
    assert risk.expected_delay_cost(0.0, 1000.0) == pytest.approx(0.0)
    assert risk.expected_delay_cost(0.5, 1000.0) == pytest.approx(500.0)
    assert risk.expected_delay_cost(
        0.5, 1000.0, intervention_cost=40.0
    ) == pytest.approx(540.0)


def test_risk_level_is_str_compatible():
    """Consumers compare against plain strings and write the value to CSV."""
    level = risk.classify_risk(0.9)
    assert level == "High Risk"
    assert str(level) == "High Risk"
    assert f"{level}" == "High Risk"
