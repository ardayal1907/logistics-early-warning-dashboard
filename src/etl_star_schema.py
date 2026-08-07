"""Compatibility shim — the ETL now lives in `logistics.pipelines.etl`.

`python src/etl_star_schema.py` keeps working, because the path is what an
existing scheduler or runbook calls. New callers should use the console script
instead:

    logistics-etl                       # reads LOGISTICS_* from the environment

The one behaviour that changed: paths come from `Settings`, not from
`ROOT = Path(__file__).resolve().parent.parent`. That line pointed at
site-packages once the project was installed, which is why the package could
not be shipped as a wheel.
"""

from logistics.pipelines.etl import (
    FACT_COLUMNS,
    RAW_FILENAME,
    REQUIRED_COLUMNS,
    ROUTE_COLUMNS,
    SEASON_BY_MONTH,
    build_dim_date,
    build_dim_route,
    build_dim_vendor,
    build_fact_shipments,
    load_raw_data,
    main,
    route_key,
    run_etl,
    validate_star_schema,
    write_star_schema,
)

__all__ = [
    "FACT_COLUMNS",
    "RAW_FILENAME",
    "REQUIRED_COLUMNS",
    "ROUTE_COLUMNS",
    "SEASON_BY_MONTH",
    "build_dim_date",
    "build_dim_route",
    "build_dim_vendor",
    "build_fact_shipments",
    "load_raw_data",
    "main",
    "route_key",
    "run_etl",
    "validate_star_schema",
    "write_star_schema",
]


if __name__ == "__main__":
    import logging

    from logistics.pipelines.report import render_etl_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    print(render_etl_report(**main()))
