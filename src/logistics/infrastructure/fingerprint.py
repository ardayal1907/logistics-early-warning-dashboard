"""Content fingerprinting for data files and model artefacts.

Lifted out of `src/config.py` unchanged in behaviour. It sat there next to the
CO2 formula, which meant the module holding the emission factors also knew the
names of the CSV files on disk — so migrating the warehouse to Delta or Postgres
would have forced a change to the carbon model. Storage is infrastructure; it
belongs here.

Two capabilities are new.

`sha256_file` is unchanged, but `write_csv_deterministically` exists because the
byte hash it produces was quietly platform-dependent. `DataFrame.to_csv()` uses
`os.linesep`, so the committed CSVs were written with CRLF on Windows (verified:
data/processed/Fact_Shipments_with_ML.csv is CRLF). CI runs on ubuntu-latest,
and Git for Windows defaults to `core.autocrlf=true`, so those same files arrive
at the runner as LF. Identical content, different digest — meaning the project's
flagship integrity test can go red on CI while green locally, for no real
reason. A test that cries wolf gets disabled, and then the drift it was
protecting against arrives unannounced. Pinning the line terminator removes the
ambiguity at the source; `.gitattributes` should pin it a second time.

`hash_artifact` extends the same discipline to the model file, which previously
had none. The pipeline hashed the DATA it trained on but never the .pkl itself —
and the .pkl is the only file in the repository whose deserialisation executes
arbitrary code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_CHUNK: Final = 1 << 20  # 1 MiB

# The scored fact table the Power BI report reads and the app compares against.
SCORED_TABLE: Final = "Fact_Shipments_with_ML.csv"

# The tables the model actually learns from. Hashing these is what catches the
# "data regenerated but model not retrained" case; SCORED_TABLE alone would not,
# because it is written by the same run that trains the model.
SOURCE_TABLES: Final = (
    "Fact_Shipments.csv",
    "Dim_Vendor.csv",
    "Dim_Route.csv",
    "Dim_Date.csv",
)


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large files stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_deterministically(df: pd.DataFrame, path: str | Path, **kwargs: object) -> None:
    """Write a CSV whose bytes do not depend on the operating system.

    `lineterminator="\\n"` and an explicit UTF-8 encoding make the digest
    reproducible across Windows, Linux and macOS. Use this everywhere a file's
    hash is recorded or compared.
    """
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8", **kwargs)


def compute_data_fingerprint(processed_dir: str | Path) -> dict[str, str | None]:
    """Fingerprint the dataset a model was (or would be) trained on.

    Returns `{"scored_table": <sha256 | None>, "source_tables": <sha256 | None>}`.
    A missing file yields None for that entry rather than raising, so a partial
    checkout degrades to "cannot verify" instead of crashing the caller.

    `source_tables` hashes the concatenated per-file digests in the fixed order
    of SOURCE_TABLES, so the result does not depend on filesystem ordering.
    """
    directory = Path(processed_dir)

    scored_path = directory / SCORED_TABLE
    scored = sha256_file(scored_path) if scored_path.exists() else None

    parts: list[str] = []
    complete = True
    for name in SOURCE_TABLES:
        candidate = directory / name
        if not candidate.exists():
            complete = False
            break
        parts.append(f"{name}:{sha256_file(candidate)}")

    source = (
        hashlib.sha256("\n".join(parts).encode()).hexdigest()
        if complete and parts
        else None
    )
    return {"scored_table": scored, "source_tables": source}


def compare_data_fingerprint(
    recorded: dict[str, str | None] | None,
    current: dict[str, str | None] | None,
) -> list[str]:
    """Human-readable mismatch descriptions; an empty list means in sync.

    An entry that is None on either side is skipped. "Cannot verify" is not the
    same as "mismatch", and warning about the former teaches users to ignore the
    warning — at which point the real one goes unread too.
    """
    labels = {
        "scored_table": f"the scored fact table ({SCORED_TABLE})",
        "source_tables": "the star-schema tables the model was trained on",
    }
    mismatches: list[str] = []
    for key, label in labels.items():
        rec = (recorded or {}).get(key)
        cur = (current or {}).get(key)
        if rec and cur and rec != cur:
            mismatches.append(label)
    return mismatches


# ---------------------------------------------------------------------------
# Model artefact integrity
# ---------------------------------------------------------------------------
def hash_artifact(path: str | Path) -> str:
    """SHA-256 of a model artefact. Same algorithm, named for its purpose."""
    return sha256_file(path)


def write_checksum_sidecar(path: str | Path) -> Path:
    """Write `<artefact>.sha256` next to the artefact and return its path.

    Called by the training pipeline immediately after `joblib.dump`. The sidecar
    is what the loader checks before it deserialises anything, so the file that
    can execute code is the one file whose bytes are verified first.
    """
    artefact = Path(path)
    digest = hash_artifact(artefact)
    sidecar = artefact.with_suffix(artefact.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {artefact.name}\n", encoding="utf-8")
    return sidecar


def read_checksum_sidecar(path: str | Path) -> str | None:
    """Read the expected digest from a sidecar, or None if there is not one."""
    sidecar = Path(path)
    if not sidecar.exists():
        return None
    first = sidecar.read_text(encoding="utf-8").strip().split()
    return first[0] if first else None


__all__ = [
    "SCORED_TABLE",
    "SOURCE_TABLES",
    "compare_data_fingerprint",
    "compute_data_fingerprint",
    "hash_artifact",
    "read_checksum_sidecar",
    "sha256_file",
    "write_checksum_sidecar",
    "write_csv_deterministically",
]
