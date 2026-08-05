"""
ETL: smart_logistics_data.csv (raw/flat data) -> Star Schema
============================================================================
Tables produced:
  1) Dim_Vendor.csv       - Vendor_ID (PK), Vendor_Rating
  2) Dim_Route.csv        - Route_ID (PK), Origin, Destination, Vehicle_Type,
                             Weather_Condition, Traffic_Density
  3) Dim_Date.csv         - Date_ID (PK, YYYYMMDD), Full_Date, Year, Quarter,
                             Month, Month_Name, Month_Year, Season,
                             Day_Of_Week, Is_Weekend
  4) Fact_Shipments.csv   - Shipment_ID (PK), Date_ID (FK), Vendor_ID (FK),
                             Route_ID (FK), Weight_tons, Distance_km,
                             Actual_Delay_Days, CO2_Emission_kg

Important design note:
  In the raw data, Vendor_Rating varies slightly from shipment to shipment for
  the same Vendor_ID, because per-shipment noise was added for realism. In a
  star schema a dimension row must represent a SINGLE fact per row, so
  build_dim_vendor collapses each Vendor_ID to one row using the mean rating.
  If the per-shipment original rating is ever needed, it can be added to
  Fact_Shipments as a separate column (this script does not do so by default,
  because the specification does not ask for Vendor_Rating in the fact table).

Importing this module has no side effects: every step is a function, and nothing
runs until main() is called.
"""

from pathlib import Path

import pandas as pd

# Paths resolve relative to the repository root, so it does not matter which
# directory you invoke the script from.
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = ROOT / "data" / "raw" / "smart_logistics_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

REQUIRED_COLUMNS = [
    "Shipment_ID", "Shipment_Date", "Vendor_ID", "Vendor_Rating", "Origin",
    "Destination", "Distance_km", "Weight_tons", "Weather_Condition",
    "Traffic_Density", "Vehicle_Type", "Actual_Delay_Days", "CO2_Emission_kg",
]

# The columns whose distinct combination defines a route.
ROUTE_COLUMNS = ["Origin", "Destination", "Vehicle_Type", "Weather_Condition",
                 "Traffic_Density"]

FACT_COLUMNS = [
    "Shipment_ID", "Date_ID", "Vendor_ID", "Route_ID", "Weight_tons",
    "Distance_km", "Actual_Delay_Days", "CO2_Emission_kg",
]

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


# ---------------------------------------------------------------------------
# 0) Read the raw data
# ---------------------------------------------------------------------------
def load_raw_data(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the flat source file and fail loudly if a required column is absent."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in the raw CSV: {missing}")
    return df


# ---------------------------------------------------------------------------
# 1) Dim_Vendor - Vendor_ID (PK), Vendor_Rating
# ---------------------------------------------------------------------------
def build_dim_vendor(df: pd.DataFrame) -> pd.DataFrame:
    """One row per vendor.

    A single Vendor_ID can carry several (slightly different) Vendor_Rating
    values, so we de-duplicate by taking the mean rating per vendor.
    """
    return (
        df.groupby("Vendor_ID", as_index=False)["Vendor_Rating"]
        .mean()
        .round(2)
        .sort_values("Vendor_ID")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2) Dim_Route - Route_ID (PK) + the route attribute combination
# ---------------------------------------------------------------------------
def build_dim_route(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct route combination, with a surrogate key."""
    dim_route = (
        df[ROUTE_COLUMNS]
        .drop_duplicates()
        .sort_values(ROUTE_COLUMNS)
        .reset_index(drop=True)
    )
    dim_route.insert(
        0, "Route_ID",
        [f"RT-{str(i).zfill(5)}" for i in range(1, len(dim_route) + 1)],
    )
    return dim_route


# ---------------------------------------------------------------------------
# 3) Dim_Date - Date_ID (PK) + standard calendar attributes
# ---------------------------------------------------------------------------
def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """A Kimball-style date dimension. Two design decisions:

    1) Date_ID = YYYYMMDD integer (a "smart key"). This is the classic Kimball
       convention: it sorts naturally, joins cheaply, and gives Power BI a
       numeric relationship column.
    2) The table also covers days on which NO shipment occurred (a contiguous
       calendar). This is mandatory for time-intelligence measures: on a date
       table with gaps, moving averages, YTD and year-over-year comparisons
       silently return wrong results.
    """
    shipment_dates = pd.to_datetime(df["Shipment_Date"])
    calendar = pd.date_range(shipment_dates.min(), shipment_dates.max(), freq="D")

    return pd.DataFrame({
        "Date_ID": calendar.strftime("%Y%m%d").astype(int),
        "Full_Date": calendar.strftime("%Y-%m-%d"),
        "Year": calendar.year,
        "Quarter": "Q" + calendar.quarter.astype(str),
        "Month": calendar.month,
        "Month_Name": calendar.strftime("%B"),
        "Month_Year": calendar.strftime("%Y-%m"),
        "Season": [SEASON_BY_MONTH[m] for m in calendar.month],
        "Day_Of_Week": calendar.strftime("%A"),
        "Is_Weekend": calendar.dayofweek >= 5,
    })


# ---------------------------------------------------------------------------
# 4) Fact_Shipments - the grain is one row per shipment
# ---------------------------------------------------------------------------
def build_fact_shipments(df: pd.DataFrame, dim_route: pd.DataFrame) -> pd.DataFrame:
    """Resolve the foreign keys and project the fact columns.

    Route_ID is resolved by matching the raw data against Dim_Route on
    ROUTE_COLUMNS (a lookup/merge, which produces no new rows).
    """
    fact = df.merge(dim_route, on=ROUTE_COLUMNS, how="left")

    # Foreign key into the date dimension: Shipment_Date -> Date_ID (YYYYMMDD)
    fact["Date_ID"] = (
        pd.to_datetime(fact["Shipment_Date"]).dt.strftime("%Y%m%d").astype(int)
    )

    return fact[FACT_COLUMNS].sort_values("Shipment_ID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) Integrity checks
# ---------------------------------------------------------------------------
def validate_star_schema(fact: pd.DataFrame, dim_vendor: pd.DataFrame,
                         dim_route: pd.DataFrame, dim_date: pd.DataFrame) -> None:
    """Every primary key unique, every foreign key resolvable, calendar contiguous.

    Raises AssertionError on the first violation. These are deliberately asserts
    rather than warnings: a broken key silently produces blank Power BI measures,
    which is far more dangerous than a crash.
    """
    assert dim_vendor["Vendor_ID"].is_unique, "Dim_Vendor.Vendor_ID must be unique!"
    assert dim_route["Route_ID"].is_unique, "Dim_Route.Route_ID must be unique!"
    assert dim_date["Date_ID"].is_unique, "Dim_Date.Date_ID must be unique!"

    calendar = pd.to_datetime(dim_date["Full_Date"])
    expected_days = (calendar.max() - calendar.min()).days + 1
    assert len(dim_date) == expected_days, \
        "The Dim_Date calendar has gaps! Time-intelligence measures would break."

    assert fact["Shipment_ID"].is_unique, "Fact_Shipments.Shipment_ID must be unique!"
    assert fact["Route_ID"].notna().all(), "Unmatched Route_ID found!"
    assert fact["Date_ID"].notna().all(), "Unmatched Date_ID found!"
    assert fact["Vendor_ID"].isin(dim_vendor["Vendor_ID"]).all(), \
        "Invalid Vendor_ID foreign key!"
    assert fact["Route_ID"].isin(dim_route["Route_ID"]).all(), \
        "Invalid Route_ID foreign key!"
    assert fact["Date_ID"].isin(dim_date["Date_ID"]).all(), \
        "Invalid Date_ID foreign key!"


# ---------------------------------------------------------------------------
# 6) Persistence
# ---------------------------------------------------------------------------
def write_star_schema(fact: pd.DataFrame, dim_vendor: pd.DataFrame,
                      dim_route: pd.DataFrame, dim_date: pd.DataFrame,
                      out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_vendor.to_csv(out_dir / "Dim_Vendor.csv", index=False)
    dim_route.to_csv(out_dir / "Dim_Route.csv", index=False)
    dim_date.to_csv(out_dir / "Dim_Date.csv", index=False)
    fact.to_csv(out_dir / "Fact_Shipments.csv", index=False)


# ---------------------------------------------------------------------------
# 7) Console report
# ---------------------------------------------------------------------------
def print_summary(raw: pd.DataFrame, fact: pd.DataFrame, dim_vendor: pd.DataFrame,
                  dim_route: pd.DataFrame, dim_date: pd.DataFrame) -> None:
    print(f"Raw data loaded: {len(raw)} rows, {len(raw.columns)} columns\n")
    print(f"Dim_Vendor.csv      -> {len(dim_vendor)} rows (unique vendors)")
    print(f"Dim_Route.csv       -> {len(dim_route)} rows (unique route combinations)")
    print(f"Dim_Date.csv        -> {len(dim_date)} rows (contiguous calendar: "
          f"{dim_date.Full_Date.min()} .. {dim_date.Full_Date.max()})")
    print(f"Fact_Shipments.csv  -> {len(fact)} rows "
          f"(same as the original shipment count: {len(raw)})")

    print("\n--- Dim_Vendor preview ---")
    print(dim_vendor.head())

    print("\n--- Dim_Route preview ---")
    print(dim_route.head())

    print("\n--- Dim_Date preview ---")
    print(dim_date.head())
    print(f"\nDays with no shipments: {len(dim_date) - fact.Date_ID.nunique()} "
          f"(present in the calendar, absent from the fact table - this is NORMAL "
          f"and required)")

    print("\n--- Fact_Shipments preview ---")
    print(fact.head())

    print("\nIntegrity checks passed: all primary keys unique, all foreign keys "
          "present in the dimension tables.")


def main() -> None:
    raw = load_raw_data()
    dim_vendor = build_dim_vendor(raw)
    dim_route = build_dim_route(raw)
    dim_date = build_dim_date(raw)
    fact = build_fact_shipments(raw, dim_route)

    validate_star_schema(fact, dim_vendor, dim_route, dim_date)
    write_star_schema(fact, dim_vendor, dim_route, dim_date)
    print_summary(raw, fact, dim_vendor, dim_route, dim_date)


if __name__ == "__main__":
    main()
