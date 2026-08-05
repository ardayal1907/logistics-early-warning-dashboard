"""Unit tests for the extracted pipeline functions.

These were impossible before the refactor: every script was top-level procedural
code, so importing one ran the whole pipeline. Now each stage is a function that can
be called on a small hand-built frame, and the failure messages point at a specific
transformation rather than at "the pipeline".

Nothing here touches the filesystem or the real dataset unless a fixture provides it.
"""

import numpy as np
import pandas as pd
import pytest

import etl_star_schema as etl
import generate_logistics_data as gen
import ml_delay_risk_pipeline as ml


@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    """Six shipments, three vendors, two routes, spanning four days."""
    return pd.DataFrame({
        "Shipment_ID": [f"SHP-{i:05d}" for i in range(1, 7)],
        "Shipment_Date": ["2026-01-01", "2026-01-01", "2026-01-02",
                          "2026-01-03", "2026-01-04", "2026-01-04"],
        "Vendor_ID": ["VEND-001", "VEND-001", "VEND-002",
                      "VEND-002", "VEND-003", "VEND-003"],
        # VEND-001 gets 4.0 and 4.2 -> the dimension must collapse to the mean 4.1
        "Vendor_Rating": [4.0, 4.2, 3.0, 3.0, 2.0, 2.0],
        "Origin": ["Ankara"] * 3 + ["Izmir"] * 3,
        "Destination": ["Izmir"] * 3 + ["Ankara"] * 3,
        "Distance_km": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0],
        "Weight_tons": [5.0, 5.0, 10.0, 10.0, 15.0, 15.0],
        "Weather_Condition": ["Normal"] * 3 + ["Snow"] * 3,
        "Traffic_Density": ["Low"] * 3 + ["High"] * 3,
        "Vehicle_Type": ["Diesel Truck"] * 3 + ["Hybrid Van"] * 3,
        "Actual_Delay_Days": [0, 0, 1, 0, 3, 2],
        "CO2_Emission_kg": [100.0, 101.0, 200.0, 202.0, 300.0, 303.0],
    })


# --- ETL -------------------------------------------------------------------

def test_build_dim_vendor_collapses_to_one_row_per_vendor(tiny_raw):
    dim = etl.build_dim_vendor(tiny_raw)
    assert len(dim) == 3
    assert dim.Vendor_ID.is_unique
    # 4.0 and 4.2 average to 4.1
    assert dim.loc[dim.Vendor_ID == "VEND-001", "Vendor_Rating"].iloc[0] == pytest.approx(4.1)


def test_build_dim_route_assigns_one_key_per_distinct_combination(tiny_raw):
    dim = etl.build_dim_route(tiny_raw)
    assert len(dim) == 2                       # two distinct route combinations
    assert dim.Route_ID.is_unique
    assert list(dim.Route_ID) == ["RT-00001", "RT-00002"]


def test_build_dim_date_is_contiguous_and_covers_empty_days(tiny_raw):
    """2026-01-01..04 is four days even though no shipment falls on some of them."""
    dim = etl.build_dim_date(tiny_raw)
    assert len(dim) == 4
    assert list(dim.Date_ID) == [20260101, 20260102, 20260103, 20260104]
    assert dim.Season.unique().tolist() == ["Winter"]
    assert dim.Date_ID.dtype.kind == "i"       # integer key for Power BI


def test_build_dim_date_fills_gaps_between_shipments():
    """A month-long gap must still produce a day-by-day calendar."""
    sparse = pd.DataFrame({"Shipment_Date": ["2026-01-01", "2026-02-01"]})
    dim = etl.build_dim_date(sparse)
    assert len(dim) == 32                      # 31 January days + 1 February


def test_build_fact_shipments_resolves_keys_and_projects_columns(tiny_raw):
    dim_route = etl.build_dim_route(tiny_raw)
    fact = etl.build_fact_shipments(tiny_raw, dim_route)
    assert list(fact.columns) == etl.FACT_COLUMNS
    assert len(fact) == len(tiny_raw)           # a lookup must not fan out rows
    assert fact.Route_ID.notna().all()
    assert fact.Date_ID.iloc[0] == 20260101


def test_validate_star_schema_accepts_a_consistent_set(tiny_raw):
    dim_vendor = etl.build_dim_vendor(tiny_raw)
    dim_route = etl.build_dim_route(tiny_raw)
    dim_date = etl.build_dim_date(tiny_raw)
    fact = etl.build_fact_shipments(tiny_raw, dim_route)
    etl.validate_star_schema(fact, dim_vendor, dim_route, dim_date)   # must not raise


def test_validate_star_schema_rejects_an_orphan_foreign_key(tiny_raw):
    dim_vendor = etl.build_dim_vendor(tiny_raw)
    dim_route = etl.build_dim_route(tiny_raw)
    dim_date = etl.build_dim_date(tiny_raw)
    fact = etl.build_fact_shipments(tiny_raw, dim_route)
    fact.loc[0, "Vendor_ID"] = "VEND-999"       # a vendor that does not exist
    with pytest.raises(AssertionError, match="Vendor_ID"):
        etl.validate_star_schema(fact, dim_vendor, dim_route, dim_date)


def test_load_raw_data_rejects_a_missing_column(tmp_path, tiny_raw):
    path = tmp_path / "broken.csv"
    tiny_raw.drop(columns=["CO2_Emission_kg"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="CO2_Emission_kg"):
        etl.load_raw_data(path)


# --- Generator -------------------------------------------------------------

def test_generate_dataset_is_deterministic():
    """The same seed must reproduce the dataset exactly - the whole pipeline
    depends on this."""
    a, intercept_a = gen.generate_dataset(seed=7, n_rows=200)
    b, intercept_b = gen.generate_dataset(seed=7, n_rows=200)
    pd.testing.assert_frame_equal(a, b)
    assert intercept_a == intercept_b


def test_a_different_seed_produces_different_data():
    a, _ = gen.generate_dataset(seed=7, n_rows=200)
    b, _ = gen.generate_dataset(seed=8, n_rows=200)
    assert not a["Actual_Delay_Days"].equals(b["Actual_Delay_Days"])


def test_solve_intercept_hits_the_target_rate():
    linear = np.linspace(-2, 2, 1000)
    for target in (0.10, 0.22, 0.50):
        c = gen.solve_intercept(linear, target)
        achieved = (1 / (1 + np.exp(-(linear + c)))).mean()
        assert achieved == pytest.approx(target, abs=1e-6)


def test_truncated_poisson_stays_inside_the_allowed_range():
    rng = np.random.default_rng(0)
    lam = np.full(5000, 1.5)
    days = gen.truncated_poisson_days(lam, gen.MAX_DELAY_DAYS, rng)
    assert days.min() >= 1
    assert days.max() <= gen.MAX_DELAY_DAYS


def test_truncated_poisson_is_monotonically_decreasing():
    """The bug this replaced: np.clip piles the tail at the maximum, producing more
    5-day than 4-day delays."""
    rng = np.random.default_rng(0)
    days = gen.truncated_poisson_days(np.full(20000, 0.8), gen.MAX_DELAY_DAYS, rng)
    counts = pd.Series(days).value_counts().sort_index()
    assert list(counts) == sorted(counts, reverse=True), \
        f"Day distribution is not monotonically decreasing: {counts.to_dict()}"


def test_target_delay_rate_matches_the_documented_design():
    """Pinned to the literal value, not to the constant.

    Comparing the generated rate against gen.TARGET_DELAY_RATE alone would be
    tautological: change the constant and the test moves with it. The ~22% figure is
    a design decision documented in the README, the generator docstring and
    docs/METHODOLOGY.md, so it is pinned here explicitly. Changing it should require
    changing this test - and therefore noticing the documentation.
    """
    assert gen.TARGET_DELAY_RATE == 0.22


def test_generated_delay_rate_is_near_the_target():
    df, _ = gen.generate_dataset(seed=42, n_rows=1500)
    rate = (df.Actual_Delay_Days > 0).mean()
    assert rate == pytest.approx(0.22, abs=0.03)


def test_shipped_dataset_has_the_expected_delay_rate(raw_data):
    """Guards the committed CSV, not just a freshly generated frame.

    If someone edits the generator and forgets to regenerate the data (or
    regenerates with different settings), the file Power BI reads drifts away from
    the documented design. This catches that.
    """
    rate = (raw_data.Actual_Delay_Days > 0).mean()
    assert rate == pytest.approx(0.215, abs=0.01), (
        f"The committed dataset has a {rate:.1%} delay rate; the documented design "
        "is ~22%. Either the generator changed without the data being regenerated, "
        "or the data was regenerated with different settings."
    )


def test_bad_weather_raises_the_delay_rate():
    df, _ = gen.generate_dataset(seed=42, n_rows=1500)
    by_weather = df.assign(late=df.Actual_Delay_Days > 0) \
                   .groupby("Weather_Condition").late.mean()
    assert by_weather["Storm"] > by_weather["Normal"]
    assert by_weather["Snow"] > by_weather["Normal"]


def test_july_has_no_snow_and_january_does():
    """Seasonality: the month drives the weather distribution."""
    df, _ = gen.generate_dataset(seed=42, n_rows=1500)
    month = pd.to_datetime(df.Shipment_Date).dt.month
    assert (df.Weather_Condition[month == 7] == "Snow").sum() == 0
    assert (df.Weather_Condition[month == 1] == "Snow").sum() > 0


# --- ML pipeline -----------------------------------------------------------

def test_chronological_split_puts_all_test_dates_after_training():
    df = pd.DataFrame({"Full_Date": pd.date_range("2026-01-01", periods=100, freq="D")})
    train_idx, test_idx = ml.chronological_split(df, train_fraction=0.8)
    assert len(train_idx) == 80 and len(test_idx) == 20
    assert df.Full_Date.iloc[train_idx].max() <= df.Full_Date.iloc[test_idx].min()


def test_chronological_split_rejects_unsorted_input():
    df = pd.DataFrame({"Full_Date": pd.to_datetime(
        ["2026-03-01", "2026-01-01", "2026-02-01", "2026-04-01"])})
    with pytest.raises(AssertionError, match="Chronological leakage"):
        ml.chronological_split(df, train_fraction=0.5)


def test_expected_calibration_error_is_zero_when_predictions_match_outcomes():
    y = np.array([0] * 50 + [1] * 50)
    exact = np.array([0.0] * 50 + [1.0] * 50)
    assert ml.expected_calibration_error(y, exact) == pytest.approx(0.0, abs=1e-12)


def test_expected_calibration_error_measures_the_gap_exactly():
    """Each bucket's mean prediction sits 0.05 away from its realised rate, and the
    two buckets are equally sized, so the weighted average is exactly 0.05."""
    y = np.array([0] * 50 + [1] * 50)
    slightly_off = np.array([0.05] * 50 + [0.95] * 50)
    assert ml.expected_calibration_error(y, slightly_off) == pytest.approx(0.05, abs=1e-9)


def test_expected_calibration_error_grows_with_overconfidence():
    y = np.array([0] * 50 + [1] * 50)
    mild = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])
    severe = np.concatenate([np.full(50, 0.45), np.full(50, 0.55)])
    assert ml.expected_calibration_error(y, severe) > ml.expected_calibration_error(y, mild)


def test_alert_cost_counts_the_confusion_matrix_correctly():
    proba = np.array([0.1, 0.3, 0.7, 0.9])
    y = np.array([0, 1, 0, 1])
    c = ml.alert_cost(proba, y, threshold=0.5, cost_ratio=4.0)
    assert c["tp"] == 1        # 0.9 -> late
    assert c["fp"] == 1        # 0.7 -> on time
    assert c["fn"] == 1        # 0.3 -> late but not alerted
    assert c["cost"] == pytest.approx(1 + 4 * 1)
    assert c["alert_rate"] == pytest.approx(0.5)


def test_alert_cost_penalises_missed_delays_more_than_false_alarms():
    proba = np.array([0.1, 0.9])
    one_miss = ml.alert_cost(proba, np.array([1, 1]), threshold=0.5)["cost"]
    one_false_alarm = ml.alert_cost(proba, np.array([0, 0]), threshold=0.5)["cost"]
    assert one_miss > one_false_alarm


def test_group_name_collapses_one_hot_dummies():
    assert ml.group_name("Weather_Condition_Storm") == "Weather_Condition"
    assert ml.group_name("Vehicle_Type_Diesel Truck") == "Vehicle_Type"
    assert ml.group_name("Distance_km") == "Distance_km"     # numeric passes through


def test_add_target_derives_is_delayed_from_the_outcome():
    df = pd.DataFrame({"Actual_Delay_Days": [0, 1, 5, 0]})
    assert list(ml.add_target(df)["is_delayed"]) == [0, 1, 1, 0]


def test_add_target_does_not_mutate_the_input():
    df = pd.DataFrame({"Actual_Delay_Days": [0, 1]})
    ml.add_target(df)
    assert "is_delayed" not in df.columns


def test_assign_risk_levels_rounds_and_tiers():
    df = pd.DataFrame({"Shipment_ID": ["a", "b", "c"]})
    out = ml.assign_risk_levels(df, np.array([0.049, 0.334, 0.777]))
    assert list(out.Delay_Risk_Probability) == [0.05, 0.33, 0.78]
    assert list(out.Risk_Level) == ["Low Risk", "Medium Risk", "High Risk"]


def test_build_model_returns_a_calibrated_pipeline():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    m = ml.build_model()
    assert isinstance(m, CalibratedClassifierCV)
    assert isinstance(m.estimator, Pipeline)
    assert list(m.estimator.named_steps) == ["preprocessor", "classifier"]


def test_validate_output_schema_rejects_a_renamed_column(fact_with_ml, dim_vendor,
                                                         dim_route, dim_date):
    tables = {"dim_vendor": dim_vendor, "dim_route": dim_route, "dim_date": dim_date}
    broken = fact_with_ml.rename(columns={"Risk_Level": "RiskLevel"})
    with pytest.raises(AssertionError, match="column structure changed"):
        ml.validate_output_schema(broken, fact_with_ml, tables)
