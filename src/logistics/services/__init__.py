"""Orchestration: the workflows that combine domain rules with infrastructure.

A service owns a *procedure*. `config.py` proved that duplicated constants drift
apart; a duplicated procedure drifts the same way and is harder to spot, because
each copy looks locally reasonable.
"""

from __future__ import annotations

from logistics.services.scoring import ScoringService, build_scoring_service

__all__ = ["ScoringService", "build_scoring_service"]
