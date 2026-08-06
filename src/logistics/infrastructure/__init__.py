"""Adapters to the outside world: files, artefacts, logs.

Everything in here can fail for reasons that have nothing to do with logistics —
a missing file, a truncated download, a read-only volume. That is exactly why it
is separated from `domain`, which cannot fail for any reason except a value the
business considers invalid.
"""

from __future__ import annotations

from logistics.infrastructure.fingerprint import (
    SCORED_TABLE,
    SOURCE_TABLES,
    compare_data_fingerprint,
    compute_data_fingerprint,
    hash_artifact,
    sha256_file,
    write_checksum_sidecar,
    write_csv_deterministically,
)
from logistics.infrastructure.model_repository import (
    ModelBundle,
    load_bundle,
    verify_artifact_integrity,
    verify_library_versions,
)
from logistics.infrastructure.prediction_log import (
    JsonlPredictionLog,
    NullPredictionLog,
    PredictionLog,
)

__all__ = [
    "SCORED_TABLE",
    "SOURCE_TABLES",
    "JsonlPredictionLog",
    "ModelBundle",
    "NullPredictionLog",
    "PredictionLog",
    "compare_data_fingerprint",
    "compute_data_fingerprint",
    "hash_artifact",
    "load_bundle",
    "sha256_file",
    "verify_artifact_integrity",
    "verify_library_versions",
    "write_checksum_sidecar",
    "write_csv_deterministically",
]
