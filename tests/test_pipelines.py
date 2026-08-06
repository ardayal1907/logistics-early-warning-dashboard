"""The pipeline stages, driven through their real entry points.

Before migration step 4 no test ever called a `main()`. These do, in a
`tmp_path`, which is what makes the two claims of that step checkable rather
than asserted in a commit message:

  1. the stages take their directories from `Settings`, so they run somewhere
     other than the repository they were written in;
  2. the integrity guarantees are exceptions, so `python -O` cannot switch
     them off.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

from logistics.errors import DataIntegrityError
from logistics.pipelines import etl, generate, report
from logistics.settings import Settings

SMALL = 150


@pytest.fixture(scope="module")
def raw_frame():
    """A small synthetic dataset — the generator's own output, not a fixture file."""
    frame, _intercept = generate.generate_dataset(n_rows=SMALL)
    return frame


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        model_dir=tmp_path / "models",
        model_path=tmp_path / "models" / "production_risk_model.pkl",
    )


# --- generate ---------------------------------------------------------------

def test_generate_writes_where_settings_says(settings):
    """The stage must not resolve its own path from __file__."""
    frame = generate.main(settings)
    written = settings.raw_dir / generate.RAW_FILENAME

    assert written.exists()
    assert len(frame) == generate.N_ROWS
    assert len(pd.read_csv(written)) == generate.N_ROWS


def test_generated_csv_has_unix_line_endings(settings):
    """The recorded SHA-256 must not depend on the operating system."""
    generate.main(settings)
    raw_bytes = (settings.raw_dir / generate.RAW_FILENAME).read_bytes()
    assert b"\r\n" not in raw_bytes


def test_the_generator_is_reproducible(raw_frame):
    again, _ = generate.generate_dataset(n_rows=SMALL)
    pd.testing.assert_frame_equal(raw_frame, again)


# --- etl --------------------------------------------------------------------

def test_etl_round_trip(settings, raw_frame):
    """generate -> etl, entirely inside tmp_path."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(settings.raw_dir / etl.RAW_FILENAME, index=False,
                     lineterminator="\n")

    tables = etl.main(settings)

    for name in ("Dim_Vendor.csv", "Dim_Route.csv", "Dim_Date.csv",
                 "Fact_Shipments.csv"):
        assert (settings.processed_dir / name).exists(), f"{name} was not written"

    assert len(tables["fact"]) == SMALL
    assert list(tables["fact"].columns) == etl.FACT_COLUMNS
    assert tables["fact"]["Route_ID"].isin(tables["dim_route"]["Route_ID"]).all()


def test_etl_output_has_unix_line_endings(settings, raw_frame):
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(settings.raw_dir / etl.RAW_FILENAME, index=False,
                     lineterminator="\n")
    etl.main(settings)
    for name in ("Dim_Vendor.csv", "Fact_Shipments.csv"):
        assert b"\r\n" not in (settings.processed_dir / name).read_bytes()


def test_etl_reports_a_missing_raw_file_as_a_typed_failure(settings):
    with pytest.raises(FileNotFoundError):
        etl.main(settings)


def test_load_raw_data_still_raises_value_error(tmp_path, raw_frame):
    """This check was never an assert, so it was never stripped. Unchanged."""
    path = tmp_path / "broken.csv"
    raw_frame.drop(columns=["CO2_Emission_kg"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="CO2_Emission_kg"):
        etl.load_raw_data(path)


# --- the point of the assert conversion ------------------------------------

def test_integrity_failures_are_typed(raw_frame):
    dim_vendor = etl.build_dim_vendor(raw_frame)
    dim_route = etl.build_dim_route(raw_frame)
    dim_date = etl.build_dim_date(raw_frame)
    fact = etl.build_fact_shipments(raw_frame, dim_route)
    fact.loc[0, "Route_ID"] = "RT-99999"

    with pytest.raises(DataIntegrityError, match="Route_ID"):
        etl.validate_star_schema(fact, dim_vendor, dim_route, dim_date)


def _run_optimised(program: str, repo_root) -> subprocess.CompletedProcess[str]:
    """Run a snippet under `python -O` with src/ importable.

    A subprocess does not inherit pytest's `pythonpath` setting, so it is passed
    explicitly rather than relying on the package being pip-installed - this
    test must work in a bare checkout too.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-O", "-c", program],
        capture_output=True, text=True, env=env,
    )


def test_optimised_mode_still_enforces_integrity(repo_root):
    """`python -O` strips assert. It must not strip these checks.

    This is the whole reason the 21 bare asserts became DataIntegrityError: with
    `assert`, validate_star_schema became an empty function under -O and the
    corruption it exists to catch passed through silently.
    """
    # First prove -O really is in effect, so a green result below means
    # something. A bare `assert False` must NOT raise.
    stripped = _run_optimised("assert False, 'asserts are live'\n", repo_root)
    assert stripped.returncode == 0, (
        "The -O smoke check did not run with assertions disabled; the rest of "
        f"this test would prove nothing.\n{stripped.stderr}"
    )

    enforced = _run_optimised(
        "import pandas as pd\n"
        "from logistics.pipelines import etl\n"
        "from logistics.errors import DataIntegrityError\n"
        "fact = pd.DataFrame({'Shipment_ID': ['A', 'A']})\n"
        "try:\n"
        "    etl._require(bool(fact['Shipment_ID'].is_unique), 'PK not unique')\n"
        "except DataIntegrityError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
        repo_root,
    )
    assert enforced.returncode == 0, (
        f"Integrity checks were skipped under python -O.\n{enforced.stderr}"
    )


# --- report -----------------------------------------------------------------

def test_etl_report_renders_without_touching_the_filesystem(raw_frame):
    dim_vendor = etl.build_dim_vendor(raw_frame)
    dim_route = etl.build_dim_route(raw_frame)
    dim_date = etl.build_dim_date(raw_frame)
    fact = etl.build_fact_shipments(raw_frame, dim_route)

    text = report.render_etl_report(raw_frame, fact, dim_vendor, dim_route, dim_date)

    assert "STAR SCHEMA" in text
    assert f"{len(fact)} rows" in text
    assert "Integrity checks passed" in text


def test_generation_report_states_target_and_realised_rate(raw_frame):
    text = report.render_generation_report(raw_frame, generate.TARGET_DELAY_RATE)
    assert "Rows generated" in text
    assert f"target {generate.TARGET_DELAY_RATE:.4f}" in text


def test_training_report_degrades_on_a_partial_summary():
    """`--stage evaluate` produces no model path and no importances."""
    text = report.render_training_report({"metrics": {"oof_roc_auc": 0.75}})
    assert "0.7500" in text
    assert "Model written to" not in text


def test_training_report_includes_what_a_full_run_produces():
    text = report.render_training_report({
        "metrics": {"oof_roc_auc": 0.75, "oof_brier": 0.14},
        "model_path": "/tmp/model.pkl",
        "risk_level_counts": pd.Series({"Low Risk": 10, "High Risk": 2}),
    })
    assert "Model written to: /tmp/model.pkl" in text
    assert "High Risk" in text
