"""Compatibility shim — the generator now lives in `logistics.pipelines.generate`.

`python src/generate_logistics_data.py` keeps working. New callers should use
the console script, which takes its directories from the environment:

    logistics-generate

The emission constants are re-exported through `config` rather than imported
from the domain package directly, so that
`generate_logistics_data.EMISSION_FACTOR is config.EMISSION_FACTOR` stays true.
tests/test_single_source_of_truth.py asserts exactly that.
"""

from config import (
    BASE_EMISSION,
    EMISSION_FACTOR,
    MIN_CO2_KG,
    TRAFFIC_CO2_MULT,
    WEATHER_CO2_MULT,
)
from logistics.pipelines.generate import (
    CITIES,
    DATA_END_DATE,
    DATA_START_DATE,
    MAX_DELAY_DAYS,
    N_ROWS,
    RAW_FILENAME,
    RNG_SEED,
    TARGET_DELAY_RATE,
    TRAFFIC_LEVELS,
    VEHICLE_TYPES,
    VENDOR_IDS,
    WEATHER_CONDITIONS,
    assemble_dataframe,
    build_base_columns,
    build_co2,
    build_delays,
    generate_dataset,
    main,
    solve_intercept,
    truncated_poisson_days,
)

__all__ = [
    "BASE_EMISSION",
    "CITIES",
    "DATA_END_DATE",
    "DATA_START_DATE",
    "EMISSION_FACTOR",
    "MAX_DELAY_DAYS",
    "MIN_CO2_KG",
    "N_ROWS",
    "RAW_FILENAME",
    "RNG_SEED",
    "TARGET_DELAY_RATE",
    "TRAFFIC_CO2_MULT",
    "TRAFFIC_LEVELS",
    "VEHICLE_TYPES",
    "VENDOR_IDS",
    "WEATHER_CO2_MULT",
    "WEATHER_CONDITIONS",
    "assemble_dataframe",
    "build_base_columns",
    "build_co2",
    "build_delays",
    "generate_dataset",
    "main",
    "solve_intercept",
    "truncated_poisson_days",
]


if __name__ == "__main__":
    import logging

    from logistics.pipelines.report import render_generation_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    print(render_generation_report(main(), TARGET_DELAY_RATE))
