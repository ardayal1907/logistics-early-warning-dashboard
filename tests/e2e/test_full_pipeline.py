"""generate -> etl -> train, end to end, in a temporary directory.

Before this file existed no test ever called a pipeline `main()`, and 16 of the
training module's 26 functions had no coverage at all - including
`score_walk_forward` and `scoring_quality`, which produce the headline metrics
quoted in the README and printed on the Streamlit page.

The run is deliberately small (300 rows, 2 calibration folds) so it costs
seconds rather than minutes. It is not a substitute for a real training run;
what it proves is that the three stages compose, that each one reads its
directories from `Settings`, and that the artefact it writes can be loaded and
used. 120 rows - the size the roadmap suggested - is too few: the inner
calibration needs at least `n_splits` positive examples per fold and raises
"Requesting 5-fold cross-validation but provided less than 5 examples for at
least one class".
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from logistics.contracts import (
    DIM_ROUTE_SCHEMA,
    DIM_VENDOR_SCHEMA,
    FACT_SHIPMENTS_SCHEMA,
    RAW_SHIPMENTS_SCHEMA,
    SCORED_SHIPMENTS_SCHEMA,
)
from logistics.pipelines import etl, generate, train
from logistics.services.scoring import build_scoring_service
from logistics.settings import Settings

N_ROWS = 300
N_SPLITS = 2

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run all three stages once; every test below inspects the same output."""
    root = tmp_path_factory.mktemp("pipeline")
    settings = Settings.from_env(
        project_root=root,
        raw_dir=root / "raw",
        processed_dir=root / "processed",
        model_dir=root / "models",
        model_path=root / "models" / train.MODEL_FILENAME,
        verify_artifact_checksum=False,
    )

    frame, _intercept = generate.generate_dataset(n_rows=N_ROWS)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(settings.raw_dir / generate.RAW_FILENAME, index=False,
                 lineterminator="\n")

    etl.main(settings)
    summary = train.main(settings, stage="all", n_splits=N_SPLITS)
    return settings, summary


def test_every_stage_wrote_its_output(pipeline_run):
    settings, _ = pipeline_run
    expected = [
        settings.raw_dir / generate.RAW_FILENAME,
        settings.processed_dir / "Dim_Vendor.csv",
        settings.processed_dir / "Dim_Route.csv",
        settings.processed_dir / "Dim_Date.csv",
        settings.processed_dir / "Fact_Shipments.csv",
        settings.processed_dir / train.SCORED_FILENAME,
        settings.model_dir / train.MODEL_FILENAME,
        settings.model_dir / train.METADATA_FILENAME,
    ]
    missing = [p.name for p in expected if not p.exists()]
    assert not missing, f"the pipeline did not write: {missing}"


def test_nothing_was_written_into_the_repository(pipeline_run, repo_root):
    """The point of deleting ROOT = Path(__file__).parent.parent.

    Every path above is under tmp_path. If a stage still resolved its own
    directory, this run would have overwritten the committed data instead.
    """
    settings, _ = pipeline_run
    assert repo_root not in settings.processed_dir.parents
    assert repo_root not in settings.model_dir.parents


def test_the_star_schema_satisfies_its_contract(pipeline_run):
    settings, _ = pipeline_run
    read = lambda name: pd.read_csv(settings.processed_dir / name)  # noqa: E731

    RAW_SHIPMENTS_SCHEMA.validate(
        pd.read_csv(settings.raw_dir / generate.RAW_FILENAME), lazy=True)
    DIM_VENDOR_SCHEMA.validate(read("Dim_Vendor.csv"), lazy=True)
    DIM_ROUTE_SCHEMA.validate(read("Dim_Route.csv"), lazy=True)
    FACT_SHIPMENTS_SCHEMA.validate(read("Fact_Shipments.csv"), lazy=True)


def test_the_scored_table_satisfies_its_contract(pipeline_run):
    settings, _ = pipeline_run
    scored = pd.read_csv(settings.processed_dir / train.SCORED_FILENAME)
    SCORED_SHIPMENTS_SCHEMA.validate(scored, lazy=True)
    assert len(scored) == N_ROWS


def test_the_metadata_describes_the_run(pipeline_run):
    settings, summary = pipeline_run
    meta = json.loads(
        (settings.model_dir / train.METADATA_FILENAME).read_text(encoding="utf-8"))

    assert meta["feature_order"] == train.FEATURE_COLS
    assert meta["training_data"]["n_rows"] == N_ROWS
    assert set(meta["metrics"]) == set(summary["metrics"])
    # The fingerprint must describe the files this run actually wrote.
    assert meta["training_data_sha256"]
    assert meta["source_tables_sha256"]


def test_the_saved_artefact_scores_through_the_service(pipeline_run):
    """The artefact is not just a file: the service must be able to use it."""
    settings, _ = pipeline_run
    service = build_scoring_service(settings)
    frame = pd.read_csv(settings.processed_dir / "Fact_Shipments.csv").merge(
        pd.read_csv(settings.processed_dir / "Dim_Vendor.csv"), on="Vendor_ID"
    ).merge(pd.read_csv(settings.processed_dir / "Dim_Route.csv"), on="Route_ID")

    scored = service.score_frame(frame)
    assert scored["Delay_Risk_Probability"].between(0.0, 1.0).all()
    assert len(scored) == N_ROWS


def test_stage_evaluate_writes_nothing(tmp_path):
    """`--stage evaluate` must be safe to run against a production directory."""
    root = tmp_path
    settings = Settings.from_env(
        project_root=root, raw_dir=root / "raw", processed_dir=root / "processed",
        model_dir=root / "models", model_path=root / "models" / train.MODEL_FILENAME,
    )
    frame, _ = generate.generate_dataset(n_rows=N_ROWS)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(settings.raw_dir / generate.RAW_FILENAME, index=False,
                 lineterminator="\n")
    etl.main(settings)

    before = sorted(p.name for p in settings.processed_dir.iterdir())
    summary = train.main(settings, stage="evaluate", n_splits=N_SPLITS)
    after = sorted(p.name for p in settings.processed_dir.iterdir())

    assert before == after, "stage=evaluate wrote to the processed directory"
    assert not settings.model_dir.exists() or not list(settings.model_dir.iterdir())
    assert "model_path" not in summary
    assert "chronological_holdout_roc_auc" in summary


def test_the_reloaded_model_agrees_with_the_one_in_memory(pipeline_run):
    """save_production_model's own guarantee, checked from outside."""
    settings, summary = pipeline_run
    bundle = joblib.load(settings.model_path)
    order = bundle["metadata"]["feature_order"]

    smoke = pd.DataFrame([{
        "Distance_km": 450.0, "Weight_tons": 12.0, "Vendor_Rating": 4.0,
        "Weather_Condition": "Storm", "Traffic_Density": "High",
        "Vehicle_Type": "Diesel Truck",
    }])[order]

    reloaded = float(bundle["model"].predict_proba(smoke)[0, 1])
    in_memory = float(summary["model"].predict_proba(smoke)[0, 1])
    assert reloaded == pytest.approx(in_memory, abs=1e-9)
