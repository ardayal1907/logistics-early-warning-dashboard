"""Schema and referential-integrity guarantees the Power BI report depends on.

If a column is renamed, reordered or dropped, or a foreign key stops resolving, Power
BI does not raise an error - it silently shows blank measures. That failure mode is
worse than a crash, so it is pinned down here.
"""

EXPECTED_COLUMNS = [
    "Shipment_ID", "Date_ID", "Vendor_ID", "Route_ID", "Weight_tons", "Distance_km",
    "Actual_Delay_Days", "CO2_Emission_kg", "Delay_Risk_Probability", "Risk_Level",
]


def test_column_names_and_order(fact_with_ml):
    assert list(fact_with_ml.columns) == EXPECTED_COLUMNS, (
        "The column structure of Fact_Shipments_with_ML.csv changed. Power BI "
        "relationships and measures are bound to these names and this order.\n"
        f"  Expected: {EXPECTED_COLUMNS}\n  Got     : {list(fact_with_ml.columns)}"
    )


def test_has_ten_columns(fact_with_ml):
    assert len(fact_with_ml.columns) == 10


def test_primary_key_is_unique_and_complete(fact_with_ml):
    assert fact_with_ml.Shipment_ID.is_unique, "Shipment_ID (PK) is not unique!"
    assert fact_with_ml.Shipment_ID.notna().all()


def test_no_missing_risk_scores(fact_with_ml):
    assert fact_with_ml.Delay_Risk_Probability.notna().all()
    assert fact_with_ml.Risk_Level.notna().all()


def test_probabilities_are_in_range(fact_with_ml):
    p = fact_with_ml.Delay_Risk_Probability
    assert p.between(0.0, 1.0).all(), "Delay_Risk_Probability outside [0, 1]."


def test_risk_level_vocabulary(fact_with_ml):
    assert set(fact_with_ml.Risk_Level) <= {"High Risk", "Medium Risk", "Low Risk"}


# --- Foreign keys ---------------------------------------------------------

def test_vendor_fk_resolves(fact_with_ml, dim_vendor):
    orphans = ~fact_with_ml.Vendor_ID.isin(dim_vendor.Vendor_ID)
    assert not orphans.any(), (
        f"{orphans.sum()} rows carry a Vendor_ID absent from Dim_Vendor: "
        f"{sorted(fact_with_ml.Vendor_ID[orphans].unique())[:5]}"
    )


def test_route_fk_resolves(fact_with_ml, dim_route):
    orphans = ~fact_with_ml.Route_ID.isin(dim_route.Route_ID)
    assert not orphans.any(), (
        f"{orphans.sum()} rows carry a Route_ID absent from Dim_Route: "
        f"{sorted(fact_with_ml.Route_ID[orphans].unique())[:5]}"
    )


def test_date_fk_resolves(fact_with_ml, dim_date):
    orphans = ~fact_with_ml.Date_ID.isin(dim_date.Date_ID)
    assert not orphans.any(), (
        f"{orphans.sum()} rows carry a Date_ID absent from Dim_Date: "
        f"{sorted(fact_with_ml.Date_ID[orphans].unique())[:5]}"
    )


def test_dimension_keys_are_unique(dim_vendor, dim_route, dim_date):
    assert dim_vendor.Vendor_ID.is_unique
    assert dim_route.Route_ID.is_unique
    assert dim_date.Date_ID.is_unique


def test_date_dimension_is_contiguous(dim_date):
    """A date table with gaps silently breaks DAX time intelligence."""
    import pandas as pd
    d = pd.to_datetime(dim_date.Full_Date)
    expected_days = (d.max() - d.min()).days + 1
    assert len(dim_date) == expected_days, (
        f"Dim_Date has {len(dim_date)} rows but spans {expected_days} days - "
        "the calendar has gaps."
    )


def test_date_id_type_matches_across_tables(fact_with_ml, dim_date):
    """Power BI cannot relate an integer column to a text one."""
    assert fact_with_ml.Date_ID.dtype == dim_date.Date_ID.dtype
