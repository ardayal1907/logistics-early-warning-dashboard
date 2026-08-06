"""Composition root for the HTTP layer.

The service is built ONCE, at application startup, and handed to request
handlers by dependency injection. Two reasons it is not built per request:
the artefact is 8.7 MB and takes ~0.4 s to deserialise, and a process that
loads the model lazily reports itself healthy before it can actually answer -
so the first real request is the one that discovers the pickle is unreadable.

`/health` reflects that: it is only green once the bundle is in memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from logistics.services.scoring import ScoringService, build_scoring_service
from logistics.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """What the process holds for its whole lifetime."""

    settings: Settings
    service: ScoringService | None = None
    startup_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.service is not None


def build_state(settings: Settings | None = None) -> AppState:
    """Load the artefact eagerly; record the failure rather than raising.

    A container that exits on a bad artefact restarts forever and tells an
    operator nothing. One that starts, reports 503 and names the reason can be
    inspected.
    """
    cfg = settings or Settings.from_env()
    state = AppState(settings=cfg)
    try:
        state.service = build_scoring_service(cfg)
        logger.info("Model loaded: %s", state.service.bundle.version)
    except Exception as exc:  # reported below, not swallowed
        state.startup_error = f"{type(exc).__name__}: {exc}"
        logger.error("Model could not be loaded: %s", state.startup_error)
    return state


__all__ = ["AppState", "build_state"]
