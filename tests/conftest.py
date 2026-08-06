"""Shared pytest fixtures and path setup.

`src/` is put on sys.path so tests can `import config` the same way the scripts do.
Note that `app.py` and the three pipeline scripts are deliberately NOT imported
anywhere in the test suite: importing them would launch Streamlit or re-run the whole
pipeline as a side effect. Where a test needs to assert something about those files,
it does so statically (see test_single_source_of_truth.py).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

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
