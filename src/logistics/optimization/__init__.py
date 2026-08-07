"""The decision layer.

The rest of the system predicts (a calibrated delay probability) and
calculates (CO2). It does not decide anything. This package does.

Phase 1 implements BLOCK B ONLY — modal allocation. See docs/OPTIMIZATION.md.

The two blocks are separable on this data, and that is a finding rather than a
convenience: the delay logit in the generator contains neither `Vehicle_Type`
nor `Weight_tons`, and `compute_co2_kg` takes no vendor. So:

    BLOCK B  modal choice      4,500 vars   NO ML INVOLVED
    BLOCK A  vendor assignment 37,500 vars  ALL of the ML lives here

Packaging a deterministic engineering calculation together with a
machine-learning result and calling the pair "AI optimisation" would be a
misrepresentation. Block B is pure arithmetic and linear algebra.

WHAT PHASE 1 MAY NOT CLAIM. The headline abatement figure measures the
generator's randomness, not fleet inefficiency: the synthetic data draws
`Vehicle_Type` independently of weight and distance, so the status quo is a
random assignment and cannot be optimal for anything. Say "vehicle assignment
in this dataset is random; removing that randomness yields X%" — never "we
saved X%". docs/OPTIMIZATION.md W3.
"""

from logistics.optimization.guard import (
    DecisionInvarianceError,
    DecisionInvarianceResult,
    check_vehicle_invariance,
)
from logistics.optimization.modal import (
    FrontierPoint,
    ModalPlan,
    build_cost_panel,
    build_emission_panel,
    optimise_modal_allocation,
    sweep_frontier,
)

__all__ = [
    "DecisionInvarianceError",
    "DecisionInvarianceResult",
    "FrontierPoint",
    "ModalPlan",
    "build_cost_panel",
    "build_emission_panel",
    "check_vehicle_invariance",
    "optimise_modal_allocation",
    "sweep_frontier",
]
