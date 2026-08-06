"""Loading the production model artefact — the one file that executes code.

The project's supply-chain thinking was built out carefully on the DATA side:
`compute_data_fingerprint` hashes the star-schema tables, the training run
records the digest in the artefact's metadata, the app recomputes it at startup
and warns on a mismatch, and CI fails the build if the committed model and the
committed data have drifted apart.

None of that applied to the artefact itself. `joblib.load(find_model_path())`
took whatever bytes it found, and `joblib.load` is a pickle read: it runs
whatever `__reduce__` the file asks it to run. So the single file in the
repository capable of arbitrary code execution was the one file with no
integrity gate in front of it. In a deployment where the artefact arrives from a
CI runner, an object store or a shared drive, anyone able to replace it gets
code execution inside the application process.

Three gates, in order, before anything is deserialised or trusted:

  1. CHECKSUM  - SHA-256 of the file compared with a sidecar (or an explicitly
                 supplied digest). This is the only check that can still refuse
                 the file; after `joblib.load` returns, any code in it has run.
  2. SCHEMA    - the bundle really holds {"model", "metadata"} and the metadata
                 carries the keys consumers depend on.
  3. VERSIONS  - the running scikit-learn / numpy / pandas match the versions
                 recorded at training time. requirements.txt describes the
                 failure this prevents in its own header: a mismatched
                 scikit-learn "SILENTLY produces different predictions". Silent
                 is the operative word - the calibration (ECE 0.023) and every
                 threshold derived from it become invalid with no symptom.
                 metadata["versions"] was already being written, with the
                 comment "let the consumer check". No consumer checked.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from logistics.errors import (
    ArtifactIntegrityError,
    EnvironmentMismatchError,
    ModelArtifactError,
)
from logistics.infrastructure.fingerprint import hash_artifact, read_checksum_sidecar

logger = logging.getLogger(__name__)

# Metadata keys the scoring service will not run without.
REQUIRED_METADATA_KEYS = ("feature_order", "thresholds", "categorical_levels", "metrics")

# Libraries whose version affects how a pickled sklearn estimator behaves.
_VERSION_SENSITIVE = ("scikit_learn", "numpy", "pandas")


@runtime_checkable
class Predictor(Protocol):
    """The narrow slice of scikit-learn the service actually uses.

    Declaring it as a Protocol keeps the service testable without the 8.7 MB
    artefact, and keeps the door open for a non-pickle serving format (ONNX,
    skops) that offers the same two-column `predict_proba`.
    """

    def predict_proba(self, X: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A verified artefact plus everything needed to cite it in an audit record."""

    model: Predictor
    metadata: dict[str, Any]
    artifact_path: Path
    artifact_sha256: str

    @property
    def name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def version(self) -> str:
        """A stable identifier for this exact artefact.

        The pipeline records `trained_at` but no version field, and writes to a
        fixed filename, so two different models are indistinguishable by name.
        Until the artefact is content-addressed on disk, the training timestamp
        plus the first 12 hex characters of the digest is a unique, verifiable
        identity that can be printed in a UI and stored in a log.
        """
        explicit = self.metadata.get("model_version")
        if explicit:
            return str(explicit)
        trained_at = str(self.metadata.get("trained_at", "unknown"))[:19]
        return f"{trained_at}+{self.artifact_sha256[:12]}"

    @property
    def feature_order(self) -> list[str]:
        return list(self.metadata["feature_order"])

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self.metadata["thresholds"])

    @property
    def categorical_levels(self) -> dict[str, list[str]]:
        return dict(self.metadata.get("categorical_levels", {}))

    @property
    def numeric_ranges(self) -> dict[str, dict[str, float]]:
        return dict(self.metadata.get("numeric_ranges", {}))


# ---------------------------------------------------------------------------
# Gate 1: integrity
# ---------------------------------------------------------------------------
def verify_artifact_integrity(path: Path, expected_sha256: str | None = None) -> str:
    """Hash the file and compare. Returns the digest; raises on mismatch.

    When no expectation is supplied and no sidecar exists, the digest is still
    computed and returned - it becomes the artefact's identity - but nothing is
    refused. That is a deliberate degradation: a first run has nothing to
    compare against, and failing closed there would only teach people to pass
    `verify_artifact_checksum=False`.
    """
    actual = hash_artifact(path)
    expected = expected_sha256 or read_checksum_sidecar(
        path.with_suffix(path.suffix + ".sha256")
    )

    if expected is None:
        logger.warning(
            "No checksum recorded for %s. The artefact was loaded unverified; "
            "run logistics.infrastructure.fingerprint.write_checksum_sidecar() "
            "after training so future loads can be verified.",
            path,
        )
        return actual

    if actual != expected:
        raise ArtifactIntegrityError(
            f"Checksum mismatch for {path}.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            f"The artefact was NOT loaded. Deserialising a pickle executes code "
            f"contained in it, so an unexpected digest is refused rather than "
            f"warned about. Re-download the artefact or re-run training."
        )
    return actual


# ---------------------------------------------------------------------------
# Gate 3: environment
# ---------------------------------------------------------------------------
def _running_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def verify_library_versions(metadata: dict[str, Any], *, strict: bool = True) -> list[str]:
    """Compare running versions with the ones recorded at training time.

    A major.minor difference on a version-sensitive library raises; a patch
    difference is logged. Returns the list of differences found.
    """
    recorded = metadata.get("versions") or {}
    if not recorded:
        logger.warning("Artefact metadata records no library versions; skipping check.")
        return []

    running = _running_versions()
    hard: list[str] = []
    soft: list[str] = []

    for library, trained_with in recorded.items():
        current = running.get(library)
        if current is None or current == trained_with:
            continue
        message = f"{library}: trained with {trained_with}, running {current}"
        if library in _VERSION_SENSITIVE and _minor(current) != _minor(trained_with):
            hard.append(message)
        else:
            soft.append(message)

    for message in soft:
        logger.warning("Version drift (patch level): %s", message)

    if hard and strict:
        raise EnvironmentMismatchError(
            "The running environment does not match the one that trained this model:\n"
            + "\n".join(f"  - {m}" for m in hard)
            + "\nA pickled scikit-learn estimator loaded under a different minor "
            "version can silently produce different probabilities, which would "
            "invalidate the calibration and every threshold derived from it. "
            "Install the pinned versions (see requirements.txt) or retrain."
        )
    return hard + soft


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_bundle(
    model_path: str | Path,
    *,
    expected_sha256: str | None = None,
    verify_checksum: bool = True,
    verify_versions: bool = True,
) -> ModelBundle:
    """Load, verify and wrap the production artefact.

    Unlike the previous `find_model_path()`, this does not search. The path is
    configuration. A loader that falls back to `Path.cwd()` means the model a
    process serves depends on the directory it was started from, which cannot be
    audited.
    """
    import joblib

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise ModelArtifactError(
            f"Model artefact not found: {path}\n"
            f"Run `python src/ml_delay_risk_pipeline.py` to train one, or set "
            f"LOGISTICS_MODEL_PATH to an existing artefact."
        )

    digest = (
        verify_artifact_integrity(path, expected_sha256)
        if verify_checksum
        else hash_artifact(path)
    )

    try:
        raw = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed domain error
        raise ModelArtifactError(
            f"Failed to deserialise {path}: {type(exc).__name__}: {exc}. "
            f"The file may be truncated, or it may have been written by an "
            f"incompatible scikit-learn version."
        ) from exc

    if not isinstance(raw, dict) or not {"model", "metadata"} <= set(raw):
        raise ModelArtifactError(
            f"{path} does not contain the expected bundle. "
            f"Expected a dict with keys 'model' and 'metadata', got "
            f"{type(raw).__name__}"
            + (f" with keys {sorted(raw)}" if isinstance(raw, dict) else "")
        )

    metadata = raw["metadata"]
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ModelArtifactError(
            f"{path} metadata is missing required key(s): {missing}. "
            f"Consumers read the feature order, thresholds and category levels "
            f"from the artefact precisely so they never hardcode them."
        )

    if verify_versions:
        verify_library_versions(metadata, strict=True)

    logger.info(
        "Loaded model %s (sha256 %s…, trained %s)",
        metadata.get("model_name", "?"),
        digest[:12],
        str(metadata.get("trained_at", "?"))[:10],
    )

    return ModelBundle(
        model=raw["model"],
        metadata=metadata,
        artifact_path=path,
        artifact_sha256=digest,
    )


__all__ = [
    "REQUIRED_METADATA_KEYS",
    "ModelBundle",
    "Predictor",
    "load_bundle",
    "verify_artifact_integrity",
    "verify_library_versions",
]
