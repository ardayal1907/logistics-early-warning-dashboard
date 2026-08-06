"""The project's exception hierarchy.

Every failure this system can produce on purpose descends from `LogisticsError`,
so a caller can write one `except LogisticsError` and be sure it is not also
swallowing a genuine bug.

Two of these classes replace failure modes that were previously untyped:

* `UnknownCategoryError` replaces the bare `KeyError` that `compute_co2_kg`
  raised when it met a vehicle or weather value it had no factor for. The ML
  path already degraded gracefully there (`OneHotEncoder(handle_unknown=
  "ignore")`, covered by tests/test_model_artifact.py::test_survives_an_unseen_
  category); the carbon path crashed the page instead. Adding a CNG truck to the
  fleet is routine business, not an exceptional event.

* `DataIntegrityError` replaces the bare `assert` statements that carried every
  integrity guarantee in the ETL and the scoring pipeline. `python -O` strips
  `assert`, which turns `validate_star_schema` and `validate_output_schema` into
  empty functions and reintroduces exactly the silent corruption their
  docstrings say they exist to prevent ("a broken key silently produces blank
  Power BI measures, which is far more dangerous than a crash").
"""

from __future__ import annotations


class LogisticsError(Exception):
    """Base class for every deliberate failure in this system."""


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
class UnknownCategoryError(LogisticsError):
    """A categorical value has no entry in the reference tables.

    Carries the offending value and the set of known ones, so the message is
    actionable without opening the source.
    """

    def __init__(self, field: str, value: object, known: object) -> None:
        self.field = field
        self.value = value
        self.known = sorted(str(k) for k in known)  # type: ignore[call-overload]
        super().__init__(
            f"Unknown {field}: {value!r}. Known values: {', '.join(self.known)}. "
            f"Add an entry to the corresponding reference table in "
            f"logistics.domain.carbon before this value can be scored."
        )


class DataIntegrityError(LogisticsError):
    """A dataset violated a structural guarantee (key, schema or foreign key)."""


# ---------------------------------------------------------------------------
# Model artefacts
# ---------------------------------------------------------------------------
class ModelArtifactError(LogisticsError):
    """The model artefact could not be loaded or does not honour its contract."""


class ArtifactIntegrityError(ModelArtifactError):
    """The artefact's bytes do not match the checksum recorded for it.

    Raised BEFORE deserialisation. `joblib.load` is a pickle read and therefore
    executes whatever `__reduce__` the file asks for; verifying the digest first
    is the only point at which that can still be refused.
    """


class EnvironmentMismatchError(ModelArtifactError):
    """The running library versions differ from the ones that trained the model.

    requirements.txt states the risk in its own header: a mismatched
    scikit-learn either fails outright or "SILENTLY produces different
    predictions". The second outcome invalidates the calibration (ECE 0.023) and
    therefore the entire cost-matrix threshold derivation, without any symptom.
    """


class ScoringError(LogisticsError):
    """Scoring could not be completed for the supplied input."""
