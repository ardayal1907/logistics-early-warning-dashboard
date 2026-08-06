"""ETL: the raw flat CSV becomes a star schema.

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

Importing this module has no side effects: every step is a function, and
nothing runs until main() is called.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from logistics.errors import DataIntegrityError
from logistics.infrastructure.fingerprint import write_csv_deterministically
from logistics.settings import Settings

logger = logging.getLogger(__name__)

RAW_FILENAME = "smart_logistics_data.csv"

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
def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the flat source file and fail loudly if a required column is absent.

    Stays a `ValueError`, deliberately. This check was never an `assert`, so it
    was never stripped by `python -O` and there is nothing here to repair; the
    conversion in this step targets the guarantees that WERE assertions.
    """
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
def route_key(*values: object) -> str:
    """A Route_ID derived from the route's own attributes, not its row number.

    `RT-` + the first 10 hex characters of sha1 over the natural key. The
    separator is the ASCII unit separator, which cannot occur in a city or
    vehicle name, so ("AB", "C") and ("A", "BC") cannot collide by
    concatenation.

    Why this replaced `RT-{row_position:05d}`: the old key was assigned from
    position in a sorted, de-duplicated frame, so inserting ONE new route
    combination renumbered every existing route. Measured on the shipped data:
    1,346 of 1,346 Route_IDs changed. Incremental loading was structurally
    impossible, and the failure was silent - old facts kept pointing at the
    same key, which now meant a different route.

    10 hex characters is 40 bits. For 1,346 rows the collision probability is
    about 8e-7, and a collision is not silent either way: validate_star_schema
    asserts Route_ID uniqueness and this function's output feeds straight into
    it.
    """
    joined = "\x1f".join(str(v) for v in values)
    return f"RT-{hashlib.sha1(joined.encode('utf-8')).hexdigest()[:10]}"


def build_dim_route(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct route combination, with a content-addressed key.

    The sort is kept for a stable, diff-friendly row ORDER, but the key no
    longer depends on it: reordering or inserting rows leaves every existing
    Route_ID untouched.
    """
    dim_route = (
        df[ROUTE_COLUMNS]
        .drop_duplicates()
        .sort_values(ROUTE_COLUMNS)
        .reset_index(drop=True)
    )
    dim_route.insert(
        0, "Route_ID",
        [route_key(*row) for row in dim_route[ROUTE_COLUMNS].itertuples(index=False)],
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
def _require(condition: bool, message: str) -> None:
    """Raise instead of `assert`.

    `python -O` strips assert statements. This function's body does not
    disappear, so `python -O -m logistics.pipelines.etl` still enforces every
    guarantee below - which is the whole point of having them.
    """
    if not condition:
        raise DataIntegrityError(message)


def validate_star_schema(fact: pd.DataFrame, dim_vendor: pd.DataFrame,
                         dim_route: pd.DataFrame, dim_date: pd.DataFrame) -> None:
    """Every primary key unique, every foreign key resolvable, calendar contiguous.

    Raises DataIntegrityError on the first violation. These are deliberately
    hard failures rather than warnings: a broken key silently produces blank
    Power BI measures, which is far more dangerous than a crash.
    """
    _require(bool(dim_vendor["Vendor_ID"].is_unique),
             "Dim_Vendor.Vendor_ID must be unique!")
    _require(bool(dim_route["Route_ID"].is_unique),
             "Dim_Route.Route_ID must be unique!")
    _require(bool(dim_date["Date_ID"].is_unique),
             "Dim_Date.Date_ID must be unique!")

    calendar = pd.to_datetime(dim_date["Full_Date"])
    expected_days = (calendar.max() - calendar.min()).days + 1
    _require(
        len(dim_date) == expected_days,
        "The Dim_Date calendar has gaps! Time-intelligence measures would break.",
    )

    _require(bool(fact["Shipment_ID"].is_unique),
             "Fact_Shipments.Shipment_ID must be unique!")
    _require(bool(fact["Route_ID"].notna().all()), "Unmatched Route_ID found!")
    _require(bool(fact["Date_ID"].notna().all()), "Unmatched Date_ID found!")
    _require(bool(fact["Vendor_ID"].isin(dim_vendor["Vendor_ID"]).all()),
             "Invalid Vendor_ID foreign key!")
    _require(bool(fact["Route_ID"].isin(dim_route["Route_ID"]).all()),
             "Invalid Route_ID foreign key!")
    _require(bool(fact["Date_ID"].isin(dim_date["Date_ID"]).all()),
             "Invalid Date_ID foreign key!")


# ---------------------------------------------------------------------------
# 6) Persistence
# ---------------------------------------------------------------------------
def write_star_schema(fact: pd.DataFrame, dim_vendor: pd.DataFrame,
                      dim_route: pd.DataFrame, dim_date: pd.DataFrame,
                      out_dir: str | Path) -> None:
    """Write the four tables with byte-deterministic line endings.

    `to_csv` defaults to `os.linesep`, so the same frame produced different
    bytes on Windows and Linux and every recorded SHA-256 became
    platform-dependent.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv_deterministically(dim_vendor, directory / "Dim_Vendor.csv")
    write_csv_deterministically(dim_route, directory / "Dim_Route.csv")
    write_csv_deterministically(dim_date, directory / "Dim_Date.csv")
    write_csv_deterministically(fact, directory / "Fact_Shipments.csv")


# ---------------------------------------------------------------------------
# 7) Composition
# ---------------------------------------------------------------------------
def run_etl(settings: Settings) -> dict[str, pd.DataFrame]:
    """Build and validate the star schema. Does not write anything."""
    raw_path = settings.raw_dir / RAW_FILENAME
    logger.info("Reading raw data from %s", raw_path)
    raw = load_raw_data(raw_path)

    dim_vendor = build_dim_vendor(raw)
    dim_route = build_dim_route(raw)
    dim_date = build_dim_date(raw)
    fact = build_fact_shipments(raw, dim_route)

    validate_star_schema(fact, dim_vendor, dim_route, dim_date)
    logger.info(
        "Star schema built: %d facts, %d vendors, %d routes, %d calendar days",
        len(fact), len(dim_vendor), len(dim_route), len(dim_date),
    )
    return {"raw": raw, "fact": fact, "dim_vendor": dim_vendor,
            "dim_route": dim_route, "dim_date": dim_date}


def main(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Entry point. Reads its directories from Settings, never from __file__."""
    settings = settings or Settings.from_env()
    tables = run_etl(settings)
    write_star_schema(
        tables["fact"], tables["dim_vendor"], tables["dim_route"],
        tables["dim_date"], settings.processed_dir,
    )
    logger.info("Wrote four tables to %s", settings.processed_dir)
    return tables


if __name__ == "__main__":  # pragma: no cover
    from logistics.pipelines.report import render_etl_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    result = main()
    print(render_etl_report(**result))
