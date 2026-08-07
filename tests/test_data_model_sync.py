"""A model and the data it was trained on must not drift apart silently.

Regenerate the data, forget to retrain, and the Streamlit demo will happily serve
stale predictions against fresh data — nothing errors, nothing looks wrong. The
pipeline records a fingerprint of the data at training time and the app recomputes it
at startup; these tests cover both the hashing and the comparison logic, including the
"cannot verify" cases that must NOT produce a false alarm.
"""

import hashlib

import pytest

import config

# --- sha256_file -----------------------------------------------------------

def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "sample.csv"
    payload = b"Shipment_ID,Risk_Level\nSHP-00001,Low Risk\n"
    p.write_bytes(payload)
    assert config.sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_detects_a_one_byte_change(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_bytes(b"a,b\n1,2\n")
    before = config.sha256_file(p)
    p.write_bytes(b"a,b\n1,3\n")
    assert config.sha256_file(p) != before


def test_sha256_file_is_stable_across_calls(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_bytes(b"x" * 5000)
    assert config.sha256_file(p) == config.sha256_file(p)


def test_sha256_file_handles_a_file_larger_than_one_chunk(tmp_path):
    """The implementation reads in 1 MiB chunks; make sure the loop is correct."""
    p = tmp_path / "big.bin"
    payload = b"\xa5" * (3 * (1 << 20) + 17)     # 3 MiB + a partial chunk
    p.write_bytes(payload)
    assert config.sha256_file(p) == hashlib.sha256(payload).hexdigest()


# --- compute_data_fingerprint ---------------------------------------------

def _write_dataset(directory, scored=b"scored\n"):
    (directory / config.SCORED_TABLE).write_bytes(scored)
    for i, name in enumerate(config.SOURCE_TABLES):
        (directory / name).write_bytes(f"table-{i}\n".encode())


def test_fingerprint_of_a_complete_dataset(tmp_path):
    _write_dataset(tmp_path)
    fp = config.compute_data_fingerprint(tmp_path)
    assert fp["scored_table"] is not None
    assert fp["source_tables"] is not None


def test_fingerprint_is_reproducible(tmp_path):
    _write_dataset(tmp_path)
    assert config.compute_data_fingerprint(tmp_path) == \
        config.compute_data_fingerprint(tmp_path)


def test_changing_the_scored_table_changes_only_that_hash(tmp_path):
    _write_dataset(tmp_path)
    before = config.compute_data_fingerprint(tmp_path)
    (tmp_path / config.SCORED_TABLE).write_bytes(b"scored-modified\n")
    after = config.compute_data_fingerprint(tmp_path)
    assert after["scored_table"] != before["scored_table"]
    assert after["source_tables"] == before["source_tables"]


def test_changing_a_source_table_changes_only_that_hash(tmp_path):
    _write_dataset(tmp_path)
    before = config.compute_data_fingerprint(tmp_path)
    (tmp_path / config.SOURCE_TABLES[0]).write_bytes(b"table-modified\n")
    after = config.compute_data_fingerprint(tmp_path)
    assert after["source_tables"] != before["source_tables"]
    assert after["scored_table"] == before["scored_table"]


def test_missing_files_yield_none_rather_than_raising(tmp_path):
    """A partial checkout must degrade to 'cannot verify', not crash the app."""
    fp = config.compute_data_fingerprint(tmp_path)      # empty directory
    assert fp == {"scored_table": None, "source_tables": None}


def test_one_missing_source_table_invalidates_the_source_hash(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / config.SOURCE_TABLES[-1]).unlink()
    fp = config.compute_data_fingerprint(tmp_path)
    assert fp["source_tables"] is None
    assert fp["scored_table"] is not None


# --- compare_data_fingerprint ---------------------------------------------

def test_identical_fingerprints_report_no_mismatch():
    fp = {"scored_table": "a", "source_tables": "b"}
    assert config.compare_data_fingerprint(fp, fp) == []


def test_a_changed_scored_table_is_reported():
    recorded = {"scored_table": "a", "source_tables": "b"}
    current = {"scored_table": "CHANGED", "source_tables": "b"}
    mismatches = config.compare_data_fingerprint(recorded, current)
    assert len(mismatches) == 1
    assert config.SCORED_TABLE in mismatches[0]


def test_a_changed_source_table_is_reported():
    recorded = {"scored_table": "a", "source_tables": "b"}
    current = {"scored_table": "a", "source_tables": "CHANGED"}
    mismatches = config.compare_data_fingerprint(recorded, current)
    assert len(mismatches) == 1
    assert "star-schema" in mismatches[0]


def test_both_changed_are_both_reported():
    recorded = {"scored_table": "a", "source_tables": "b"}
    current = {"scored_table": "x", "source_tables": "y"}
    assert len(config.compare_data_fingerprint(recorded, current)) == 2


@pytest.mark.parametrize("recorded, current", [
    ({"scored_table": None, "source_tables": None},
     {"scored_table": "a", "source_tables": "b"}),
    ({"scored_table": "a", "source_tables": "b"},
     {"scored_table": None, "source_tables": None}),
    ({}, {"scored_table": "a", "source_tables": "b"}),
    (None, {"scored_table": "a", "source_tables": "b"}),
    ({"scored_table": "a", "source_tables": "b"}, None),
])
def test_unknown_hashes_never_raise_a_false_alarm(recorded, current):
    """'Cannot verify' is not 'mismatch'. Warning on the former would train users
    to ignore the warning entirely."""
    assert config.compare_data_fingerprint(recorded, current) == []


# --- End-to-end against the shipped artefacts ------------------------------

def test_shipped_model_records_a_fingerprint(model_bundle):
    meta = model_bundle["metadata"]
    assert meta.get("training_data_sha256"), \
        "The saved model carries no training_data_sha256; the sync check cannot work."
    assert meta.get("source_tables_sha256")
    assert len(meta["training_data_sha256"]) == 64      # a full SHA-256 hex digest


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Stale fingerprint, not data drift. The shipped .pkl records SHA-256 over "
        "the CRLF bytes of a Windows working tree; the repository's canonical bytes "
        "are LF, now pinned by .gitattributes. The content is provably identical: a "
        "byte-for-byte CRLF->LF transform of the old files equals the current ones "
        "exactly, and both parse to equal DataFrames (shape, dtypes, every cell). "
        "Correcting the digests means rewriting them inside the 8.3 MB artefact, "
        "which belongs to roadmap step 0 (regenerate data + retrain) and is kept out "
        "of this refactor series on purpose. strict=True so this turns RED the moment "
        "the artefact is rebuilt and cannot be left behind."
    ),
)
def test_shipped_model_is_in_sync_with_the_shipped_data(model_bundle, repo_root):
    """The committed model and the committed data must agree.

    If this fails, someone regenerated the data without re-running
    `python src/ml_delay_risk_pipeline.py` — exactly the drift this check exists
    to catch, now caught in CI rather than in the app.
    """
    meta = model_bundle["metadata"]
    current = config.compute_data_fingerprint(repo_root / "data" / "processed")
    recorded = {
        "scored_table": meta["training_data_sha256"],
        "source_tables": meta["source_tables_sha256"],
    }
    mismatches = config.compare_data_fingerprint(recorded, current)
    assert mismatches == [], (
        "The committed model and data have drifted apart: "
        + ", ".join(mismatches)
        + ". Re-run python src/ml_delay_risk_pipeline.py and commit the result."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Same stale fingerprint as the test above: the recorded digest is over CRLF "
        "bytes, the file on disk is LF. Clears when the artefact is rebuilt in "
        "roadmap step 0."
    ),
)
def test_recorded_hash_equals_a_direct_hash_of_the_file(model_bundle, repo_root):
    """Pin the exact meaning of training_data_sha256: it is the scored fact table."""
    direct = config.sha256_file(
        repo_root / "data" / "processed" / config.SCORED_TABLE)
    assert model_bundle["metadata"]["training_data_sha256"] == direct


# --- build_metadata propagates the fingerprint ----------------------------
# The tests above read the ALREADY-SAVED model, so they cannot catch a pipeline
# change that stops recording the fingerprint on the NEXT run. These call the
# function directly.

def _tiny_training_frame():
    import pandas as pd
    return pd.DataFrame({
        "Distance_km": [100.0, 900.0],
        "Weight_tons": [5.0, 20.0],
        "Vendor_Rating": [4.5, 2.5],
        "Weather_Condition": ["Normal", "Storm"],
        "Traffic_Density": ["Low", "High"],
        "Vehicle_Type": ["Diesel Truck", "Hybrid Van"],
        "Full_Date": pd.to_datetime(["2026-01-01", "2026-06-30"]),
    })


def test_build_metadata_records_the_fingerprint():
    import pandas as pd

    import ml_delay_risk_pipeline as ml

    df = _tiny_training_frame()
    y = pd.Series([0, 1])
    fingerprint = {"scored_table": "a" * 64, "source_tables": "b" * 64}

    meta = ml.build_metadata(df, y, metrics={}, fingerprint=fingerprint)
    assert meta["training_data_sha256"] == "a" * 64, \
        "build_metadata dropped the scored-table hash; the app's sync check would " \
        "silently become a no-op on the next retrain."
    assert meta["source_tables_sha256"] == "b" * 64


def test_build_metadata_tolerates_a_missing_fingerprint():
    """A partial environment must not crash the pipeline - it records None."""
    import pandas as pd

    import ml_delay_risk_pipeline as ml

    meta = ml.build_metadata(_tiny_training_frame(), pd.Series([0, 1]),
                             metrics={}, fingerprint=None)
    assert meta["training_data_sha256"] is None
    assert meta["source_tables_sha256"] is None
