"""The audit trail for scoring decisions.

Nothing recorded a prediction. `probability` and `carbon_tax` were computed,
rendered into `st.metric` and discarded. Yet the panel's output triggers real
operational action — call the vendor, reserve buffer capacity, consider a route
change — and produces a currency figure that is reported upwards.

That leaves three questions unanswerable:

  * Dispute:   "What did the system say about this shipment on 3 March, with
                which inputs and which model?"
  * Drift:     nothing accumulates the live input distribution, so there is no
                data on which to compute PSI or a KS statistic later. The
                existing fingerprint check answers "did the file change?",
                which is not the same question as "did the world change?".
  * Assurance: model governance regimes want traceability from an output back
                to the exact artefact and thresholds that produced it.

The format is JSON Lines: append-only, one flat object per line, crash-safe at
line granularity, and readable by every log platform and by `pd.read_json(...,
lines=True)`. The `PredictionLog` protocol keeps the service independent of the
destination, so swapping the file for Kafka, Application Insights or a Delta
table changes one line in the composition root.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from logistics.domain.models import ShipmentAssessment

logger = logging.getLogger(__name__)


class PredictionLog(Protocol):
    """Somewhere a scoring decision can be recorded."""

    def record(self, assessment: ShipmentAssessment) -> None: ...


class NullPredictionLog:
    """Discards records. The explicit, greppable version of "no audit trail".

    Used by default in local development so the demo needs no writable path,
    and never silently: `build_scoring_service` logs a warning when scoring runs
    without a destination.
    """

    def record(self, assessment: ShipmentAssessment) -> None:  # noqa: D102
        return None


class JsonlPredictionLog:
    """Appends one JSON object per scoring decision.

    Append mode plus a process-level lock is enough for the single-process
    Streamlit and FastAPI cases. On POSIX, writes below PIPE_BUF are atomic in
    append mode, so concurrent workers interleave whole lines rather than
    corrupting them; on Windows that guarantee does not hold, which is why the
    lock is unconditional. Multi-host deployments should point this at a log
    collector instead.
    """

    def __init__(self, path: str | Path, *, ensure_parent: bool = True) -> None:
        self.path = Path(path)
        if ensure_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, assessment: ShipmentAssessment) -> None:
        line = json.dumps(
            assessment.to_audit_record(), ensure_ascii=False, separators=(",", ":")
        )
        try:
            with self._lock, open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
                # The audit trail is worth its cost: a prediction that reached a
                # user but is missing from the log is worse than a slow write.
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            # Never let logging take down scoring. A failed write is loud in the
            # application log, but the user still gets their answer.
            logger.error("Could not append to the prediction log %s: %s", self.path, exc)


__all__ = ["JsonlPredictionLog", "NullPredictionLog", "PredictionLog"]
