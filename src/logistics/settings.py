"""Runtime configuration — the one place paths and business parameters enter.

The project previously had no configuration seam at all: `argparse`, `click`,
`os.environ` and `getenv` appear zero times across `src/` and `tests/`. Paths
were frozen at import time in four files, and the carbon price — a value whose
own comment says it exists "so it can be swapped for a jurisdiction-specific
carbon tax" — could only be swapped by editing source and redeploying.

There is a second, less obvious problem this fixes. Every module resolved the
repository root as:

    ROOT = Path(__file__).resolve().parent.parent

Once the project is installed as a wheel, `__file__` is inside site-packages,
so ROOT points at site-packages and the data is never found. That single line
actively prevented the packaging this refactor is built on.

Resolution order, highest priority first:
    1. explicit argument to `Settings(...)`
    2. environment variable, prefix LOGISTICS_
    3. project root discovered by walking up for a marker file
    4. the documented default

`app.py`'s `find_model_path()` walked `(parent.parent, parent, Path.cwd())` and
took the first hit. The `Path.cwd()` branch meant a `models/` directory left in
whatever directory the process happened to start in would win over the
repository's own artefact — the loaded model depended on the working directory.
It is not reproduced here: the model path is configuration, not a search.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from logistics.domain.carbon import DEFAULT_CARBON_PRICE_PER_TON
from logistics.domain.risk import DEFAULT_COST_FN_OVER_FP

ENV_PREFIX: Final = "LOGISTICS_"

# Files that only ever exist at the repository root. Used to locate the project
# when running from a source checkout; irrelevant once paths are configured
# explicitly, which is how a deployed service should run.
_ROOT_MARKERS: Final = ("pyproject.toml", "requirements.txt")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards looking for a repository marker.

    Falls back to the current working directory rather than raising: a
    deployment that configures its paths explicitly must not be blocked by the
    absence of a source checkout.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and any((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    return Path.cwd()


def _env(name: str) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}")


def _env_path(name: str) -> Path | None:
    raw = _env(name)
    return Path(raw).expanduser().resolve() if raw else None


def _env_float(name: str) -> float | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str) -> bool | None:
    raw = _env(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Everything the application needs to know about its environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    model_dir: Path
    model_path: Path

    # Where prediction records are appended. None disables the audit log, which
    # is acceptable for a local demo and not acceptable in production - the
    # scoring service says so once, at startup, rather than silently.
    prediction_log_path: Path | None = None

    # --- business parameters ------------------------------------------------
    carbon_price_per_ton: float = Field(default=DEFAULT_CARBON_PRICE_PER_TON, ge=0)
    cost_fn_over_fp: float = Field(default=DEFAULT_COST_FN_OVER_FP, gt=0)

    # --- integrity gates ----------------------------------------------------
    # Verify the artefact's SHA-256 before deserialising it. joblib.load is a
    # pickle read and therefore executes code from the file; this is the only
    # moment at which a tampered artefact can still be refused.
    verify_artifact_checksum: bool = True

    # Refuse to load when the running scikit-learn / numpy / pandas major.minor
    # differs from the versions recorded in the artefact's metadata.
    verify_library_versions: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from the environment, with explicit overrides winning."""
        root = (
            overrides.pop("project_root", None)
            or _env_path("PROJECT_ROOT")
            or find_project_root()
        )
        root = Path(root)

        data_dir = _env_path("DATA_DIR") or root / "data"
        model_dir = _env_path("MODEL_DIR") or root / "models"

        defaults: dict[str, Any] = {
            "project_root": root,
            "raw_dir": _env_path("RAW_DIR") or data_dir / "raw",
            "processed_dir": _env_path("PROCESSED_DIR") or data_dir / "processed",
            "model_dir": model_dir,
            "model_path": (
                _env_path("MODEL_PATH") or model_dir / "production_risk_model.pkl"
            ),
            "prediction_log_path": _env_path("PREDICTION_LOG_PATH"),
        }

        for key, value in (
            ("carbon_price_per_ton", _env_float("CARBON_PRICE_PER_TON")),
            ("cost_fn_over_fp", _env_float("COST_FN_OVER_FP")),
            ("verify_artifact_checksum", _env_bool("VERIFY_ARTIFACT_CHECKSUM")),
            ("verify_library_versions", _env_bool("VERIFY_LIBRARY_VERSIONS")),
        ):
            if value is not None:
                defaults[key] = value

        defaults.update(overrides)
        return cls(**defaults)

    @property
    def checksum_path(self) -> Path:
        """Sidecar holding the artefact's expected SHA-256."""
        return self.model_path.with_suffix(self.model_path.suffix + ".sha256")


__all__ = ["ENV_PREFIX", "Settings", "find_project_root"]
