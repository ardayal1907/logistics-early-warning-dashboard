"""Shared pytest fixtures and data-artefact guards.

`src/` reaches the import path through `pythonpath = ["src"]` in
pyproject.toml, not through a `sys.path.insert` here — one mechanism, declared
in one place, and it applies to collection as well as to run time.

`app.py` and the three pipeline scripts ARE imported by the suite
(test_single_source_of_truth.py). That is safe because every one of them is
import-safe: their work happens in `main()`, and nothing runs at module level.
The identity assertions in that file depend on it.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(ROOT)} is missing - run the pipeline first "
            "(python src/generate_logistics_data.py && python src/etl_star_schema.py "
            "&& python src/ml_delay_risk_pipeline.py)"
        )
    return path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    return pd.read_csv(_require(RAW / "smart_logistics_data.csv"))


@pytest.fixture(scope="session")
def fact_with_ml() -> pd.DataFrame:
    return pd.read_csv(_require(PROCESSED / "Fact_Shipments_with_ML.csv"))


@pytest.fixture(scope="session")
def dim_vendor() -> pd.DataFrame:
    return pd.read_csv(_require(PROCESSED / "Dim_Vendor.csv"))


@pytest.fixture(scope="session")
def dim_route() -> pd.DataFrame:
    return pd.read_csv(_require(PROCESSED / "Dim_Route.csv"))


@pytest.fixture(scope="session")
def dim_date() -> pd.DataFrame:
    return pd.read_csv(_require(PROCESSED / "Dim_Date.csv"))


@pytest.fixture(scope="session")
def model_bundle():
    import joblib
    return joblib.load(_require(MODELS / "production_risk_model.pkl"))
