"""Logistics early-warning and carbon management.

Layering, strictly one-directional — an inner layer never imports an outer one:

    domain/          pure business rules. Standard library only: the emission
                     model, the risk tiering and the cost arithmetic. Importable
                     by anything, including a service that has no pandas.
    infrastructure/  the outside world: artefact loading and verification,
                     content fingerprinting, the prediction log.
    services/        orchestration. Puts domain rules and infrastructure
                     together into a workflow (scoring today, optimisation next).
    optimization/    the operations-research layer, built on services/.
    ui/, api/        delivery. Streamlit and FastAPI. No business logic.

`settings.py` sits beside them: it is read by the composition roots, never by
domain code.

The flat modules still in `src/` (config.py, app.py, etl_star_schema.py,
generate_logistics_data.py, ml_delay_risk_pipeline.py) are the pre-refactor
codebase. They keep working unchanged and are migrated one at a time; see
docs/MIGRATION.md for the order and the exit criterion of each step.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
