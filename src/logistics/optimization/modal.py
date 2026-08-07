"""Block B — modal allocation. Which vehicle type carries which shipment.

    minimise   sum_ik f_ik z_ik  -  delta * s / range_CO2
    subject to V1  sum_k z_ik = 1            for every shipment    (1,500 rows)
               V2  sum_i z_ik <= Fleet_k     for every vehicle     (3 rows)
               V4  sum_ik e_ik z_ik + s = epsilon                  (1 row)
               z_ik in [0, 1],  s >= 0

`z` is CONTINUOUS, not binary. Written with capacities in trip counts, V1 and
V2 form a transportation polytope, which is totally unimodular, so the LP
relaxation is integral and binaries buy nothing. The single side constraint V4
breaks TU, after which a basic optimal solution can have at most one fractional
shipment. A TONNAGE capacity would break TU properly and turn this into a
Generalized Assignment Problem (NP-hard); Phase 1 deliberately omits it.

Carbon is NOT priced in the objective. Pricing it there AND bounding it in V4
double-counts, and it makes the dual uninterpretable: the shadow price stops
being "the marginal cost of the next tonne" and becomes "the premium on top of
$50". The objective stays in pure currency and carbon stays in physical kg on
the constraint, which is the only way the dual of V4 can be compared against
`Settings.carbon_price_per_ton`.

The `- delta * s / range_CO2` term is AUGMECON2's slack bonus: it removes
weakly-dominated points from the frontier without materially perturbing the
objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import pulp

from logistics.domain.carbon import compute_co2_kg_many
from logistics.domain.economics import DEFAULT_SLA_PENALTY_PER_DELAY, freight_cost
from logistics.domain.enums import VehicleType
from logistics.errors import LogisticsError

logger = logging.getLogger(__name__)

# AUGMECON2 slack coefficient. Small enough not to distort the cost objective,
# large enough to break ties towards lower emissions.
AUGMECON_DELTA = 1e-3

FLEET = [v.value for v in VehicleType]


class InfeasiblePlanError(LogisticsError):
    """No allocation satisfies the fleet caps and the emission ceiling."""


@dataclass
class ModalPlan:
    """One solved allocation."""

    status: str
    freight_cost: float
    co2_kg: float
    epsilon_kg: float | None
    shadow_price_per_ton: float | None
    assignment: pd.DataFrame
    vehicle_counts: dict[str, float] = field(default_factory=dict)
    fractional_shipments: int = 0
    # v2 only: expected number of late shipments under this plan, summed from
    # the calibrated p_ik. None when the plan was solved without a risk panel.
    expected_delays: float | None = None

    @property
    def co2_tons(self) -> float:
        return self.co2_kg / 1000.0


@dataclass(frozen=True)
class FrontierPoint:
    epsilon_kg: float
    freight_cost: float
    co2_kg: float
    shadow_price_per_ton: float | None


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def build_emission_panel(shipments: pd.DataFrame,
                         fleet: list[str] | None = None) -> pd.DataFrame:
    """e_ik: CO2 in kg if shipment i were carried by vehicle k.

    Computed from `compute_co2_kg_many`, NOT read from the `CO2_Emission_kg`
    column. That column carries N(1, 0.03) generator noise and only exists for
    the vehicle actually used; a counterfactual needs the formula. Verified in
    docs/OPTIMIZATION.md: stored total 1,232.6 t against 1,233.1 t recomputed.
    """
    vehicles = fleet or FLEET
    distance = shipments["Distance_km"].to_numpy(dtype=float)
    weight = shipments["Weight_tons"].to_numpy(dtype=float)
    weather = shipments["Weather_Condition"].to_numpy()
    traffic = shipments["Traffic_Density"].to_numpy()

    panel = pd.DataFrame(index=shipments.index)
    for vehicle in vehicles:
        panel[vehicle] = compute_co2_kg_many(
            distance, weight, [vehicle] * len(shipments), weather, traffic
        )
    return panel


def build_cost_panel(shipments: pd.DataFrame,
                     fleet: list[str] | None = None) -> pd.DataFrame:
    """f_ik: freight cost in dollars if shipment i were carried by vehicle k.

    EVERY VALUE HERE RESTS ON AN ASSUMPTION - see logistics.domain.economics.
    """
    vehicles = fleet or FLEET
    panel = pd.DataFrame(index=shipments.index)
    for vehicle in vehicles:
        panel[vehicle] = [
            freight_cost(float(d), float(w), vehicle)
            for d, w in zip(shipments["Distance_km"], shipments["Weight_tons"],
                            strict=True)
        ]
    return panel


def status_quo(shipments: pd.DataFrame,
               emissions: pd.DataFrame | None = None,
               costs: pd.DataFrame | None = None) -> ModalPlan:
    """What the shipped assignment costs and emits, on the same arithmetic.

    Comparing an optimised plan against the `CO2_Emission_kg` column instead of
    against this would attribute the generator's noise to the optimiser.
    """
    e = emissions if emissions is not None else build_emission_panel(shipments)
    f = costs if costs is not None else build_cost_panel(shipments)

    chosen = shipments["Vehicle_Type"].to_numpy()
    co2 = float(sum(e.iloc[i][v] for i, v in enumerate(chosen)))
    cost = float(sum(f.iloc[i][v] for i, v in enumerate(chosen)))

    return ModalPlan(
        status="StatusQuo",
        freight_cost=cost,
        co2_kg=co2,
        epsilon_kg=None,
        shadow_price_per_ton=None,
        assignment=pd.DataFrame({
            "Shipment_ID": shipments["Shipment_ID"].to_numpy(),
            "Vehicle_Type": chosen,
            "share": 1.0,
        }),
        vehicle_counts=shipments["Vehicle_Type"].value_counts().to_dict(),
    )


def observed_fleet_caps(shipments: pd.DataFrame) -> dict[str, int]:
    """Fleet_k held at the status quo mix: no vehicle purchases.

    Relaxing these caps yields "electrify everything", which is a capex
    decision, not an optimisation result.
    """
    counts = shipments["Vehicle_Type"].value_counts().to_dict()
    return {vehicle: int(counts.get(vehicle, 0)) for vehicle in FLEET}


# ---------------------------------------------------------------------------
# The programme
# ---------------------------------------------------------------------------
def _solver(msg: bool = False) -> pulp.LpSolver:
    """HiGHS when it is installed, CBC otherwise.

    docs/OPTIMIZATION.md pins highspy, but CBC ships with PuLP and returns
    correct LP duals, so a missing HiGHS degrades performance rather than
    correctness. The choice is logged so a run's numbers can be traced to it.
    """
    for name in ("HiGHS", "PULP_CBC_CMD"):
        if name in pulp.listSolvers(onlyAvailable=True):
            logger.debug("Using solver %s", name)
            return pulp.getSolver(name, msg=msg)
    raise LogisticsError("No LP solver is available; install pulp's CBC or highspy.")


def optimise_modal_allocation(
    shipments: pd.DataFrame,
    *,
    epsilon_kg: float | None = None,
    fleet_caps: dict[str, int] | None = None,
    emissions: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
    risk_panel: pd.DataFrame | None = None,
    sla_penalty: float = DEFAULT_SLA_PENALTY_PER_DELAY,
    co2_range: float | None = None,
    msg: bool = False,
) -> ModalPlan:
    """Solve Block B once, at one emission ceiling.

    `epsilon_kg=None` means no ceiling: pure cost minimisation subject to the
    fleet caps, which is the left-hand end of the frontier.

    `risk_panel` turns this into the v2 formulation. Without it the objective is
    freight cost alone and the plan implicitly assumes the modal choice does not
    move delay risk - the assumption v1's guard had to police. With it, the
    decision variable carries an expected SLA cost per (shipment, vehicle):

        min  sum_ik [ freight_ik + sla_penalty * p_ik ] * y_ik

    where p_ik comes from `calibration.calibrated_risk_panel`. The optimiser can
    then trade a cheaper vehicle against a likelier delay explicitly, on measured
    probabilities, instead of being forbidden from noticing the trade exists.
    """
    e = emissions if emissions is not None else build_emission_panel(shipments)
    f = costs if costs is not None else build_cost_panel(shipments)
    caps = fleet_caps or observed_fleet_caps(shipments)

    n = len(shipments)
    ids = shipments["Shipment_ID"].to_numpy()

    r = None
    if risk_panel is not None:
        # Align to the shipment order the panels use; a silent reindex mismatch
        # here would attach one shipment's risk to another's decision.
        r = risk_panel.reindex(ids)
        if r.isna().to_numpy().any():
            missing = int(r.isna().any(axis=1).sum())
            raise InfeasiblePlanError(
                f"risk_panel is missing {missing} of {n} shipments. Every shipment "
                f"needs a calibrated probability for every vehicle type."
            )

    problem = pulp.LpProblem("modal_allocation", pulp.LpMinimize)
    z = {
        (i, k): pulp.LpVariable(f"z_{i}_{k}", lowBound=0, upBound=1)
        for i in range(n)
        for k in FLEET
    }

    def unit_cost(i: int, k: str) -> float:
        freight = float(f.iloc[i][k])
        if r is None:
            return freight
        return freight + sla_penalty * float(r.iloc[i][k])

    cost_term = pulp.lpSum(unit_cost(i, k) * z[(i, k)] for i in range(n) for k in FLEET)

    if epsilon_kg is None:
        problem += cost_term
    else:
        span = co2_range or float(e.max(axis=1).sum() - e.min(axis=1).sum()) or 1.0
        slack = pulp.LpVariable("s", lowBound=0)
        problem += cost_term - AUGMECON_DELTA * slack / span
        problem += (
            pulp.lpSum(float(e.iloc[i][k]) * z[(i, k)] for i in range(n) for k in FLEET)
            + slack == epsilon_kg,
            "V4_emission_ceiling",
        )

    for i in range(n):                                              # V1
        problem += pulp.lpSum(z[(i, k)] for k in FLEET) == 1, f"V1_assign_{i}"

    for k in FLEET:                                                 # V2
        problem += (
            pulp.lpSum(z[(i, k)] for i in range(n)) <= caps.get(k, n),
            f"V2_fleet_{k.replace(' ', '_')}",
        )

    problem.solve(_solver(msg=msg))
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise InfeasiblePlanError(
            f"Modal allocation is {status} at epsilon={epsilon_kg}. The emission "
            f"ceiling is below what the fleet caps can deliver."
        )

    rows, total_co2, total_cost, fractional = [], 0.0, 0.0, 0
    expected_delays = 0.0
    for i in range(n):
        shares = {k: float(z[(i, k)].value() or 0.0) for k in FLEET}
        if max(shares.values()) < 1 - 1e-6:
            fractional += 1
        for k, share in shares.items():
            if share > 1e-9:
                rows.append({"Shipment_ID": ids[i], "Vehicle_Type": k, "share": share})
                total_co2 += float(e.iloc[i][k]) * share
                # freight_cost stays FREIGHT. The SLA term steers the decision but
                # reporting it inside freight would make v1 and v2 plans
                # incomparable on the axis the frontier is drawn in.
                total_cost += float(f.iloc[i][k]) * share
                if r is not None:
                    expected_delays += float(r.iloc[i][k]) * share

    assignment = pd.DataFrame(rows)
    shadow = None
    if epsilon_kg is not None:
        constraint = problem.constraints.get("V4_emission_ceiling")
        # The dual is per kg; the dashboard talks in dollars per tonne. Negated
        # because tightening a <=-style ceiling increases cost.
        if constraint is not None and constraint.pi is not None:
            shadow = -float(constraint.pi) * 1000.0

    return ModalPlan(
        status=status,
        freight_cost=total_cost,
        co2_kg=total_co2,
        epsilon_kg=epsilon_kg,
        shadow_price_per_ton=shadow,
        assignment=assignment,
        vehicle_counts=assignment.groupby("Vehicle_Type")["share"].sum().to_dict(),
        fractional_shipments=fractional,
        expected_delays=expected_delays if r is not None else None,
    )


def sweep_frontier(
    shipments: pd.DataFrame,
    *,
    points: int = 12,
    fleet_caps: dict[str, int] | None = None,
    msg: bool = False,
) -> list[FrontierPoint]:
    """Trace the cost/emission Pareto frontier.

    A single "$X saved" headline is unfalsifiable while half the objective is
    assumed (docs/OPTIMIZATION.md W1). The frontier is the honest deliverable:
    it shows the exchange rate, and the reader supplies their own carbon price.
    """
    e = build_emission_panel(shipments)
    f = build_cost_panel(shipments)
    caps = fleet_caps or observed_fleet_caps(shipments)

    # The two ends of the frontier: cheapest-possible and greenest-possible.
    # They bound the epsilon sweep, so neither is guessed.
    cheapest = optimise_modal_allocation(
        shipments, fleet_caps=caps, emissions=e, costs=f, msg=msg)
    greenest = _minimum_emission_plan(shipments, e, f, caps, msg)

    hi, lo = cheapest.co2_kg, greenest.co2_kg
    span = max(hi - lo, 1e-9)

    frontier: list[FrontierPoint] = []
    for step in range(points):
        epsilon = hi - span * step / max(points - 1, 1)
        try:
            plan = optimise_modal_allocation(
                shipments, epsilon_kg=epsilon, fleet_caps=caps,
                emissions=e, costs=f, co2_range=span, msg=msg,
            )
        except InfeasiblePlanError:
            logger.info("epsilon=%.1f kg is infeasible; frontier ends here", epsilon)
            break
        frontier.append(FrontierPoint(
            epsilon_kg=epsilon,
            freight_cost=plan.freight_cost,
            co2_kg=plan.co2_kg,
            shadow_price_per_ton=plan.shadow_price_per_ton,
        ))
    return frontier


def _minimum_emission_plan(shipments: pd.DataFrame, e: pd.DataFrame,
                           f: pd.DataFrame, caps: dict[str, int],
                           msg: bool) -> ModalPlan:
    """The greenest allocation the fleet caps allow: minimise CO2, ignore cost."""
    n = len(shipments)
    problem = pulp.LpProblem("min_emission", pulp.LpMinimize)
    z = {(i, k): pulp.LpVariable(f"z_{i}_{k}", lowBound=0, upBound=1)
         for i in range(n) for k in FLEET}

    problem += pulp.lpSum(float(e.iloc[i][k]) * z[(i, k)]
                          for i in range(n) for k in FLEET)
    for i in range(n):
        problem += pulp.lpSum(z[(i, k)] for k in FLEET) == 1
    for k in FLEET:
        problem += pulp.lpSum(z[(i, k)] for i in range(n)) <= caps.get(k, n)

    problem.solve(_solver(msg=msg))
    if pulp.LpStatus[problem.status] != "Optimal":
        raise InfeasiblePlanError("The minimum-emission plan is infeasible.")

    co2 = sum(float(e.iloc[i][k]) * float(z[(i, k)].value() or 0.0)
              for i in range(n) for k in FLEET)
    cost = sum(float(f.iloc[i][k]) * float(z[(i, k)].value() or 0.0)
               for i in range(n) for k in FLEET)
    return ModalPlan(status="Optimal", freight_cost=cost, co2_kg=co2,
                     epsilon_kg=None, shadow_price_per_ton=None,
                     assignment=pd.DataFrame())


__all__ = [
    "AUGMECON_DELTA",
    "FLEET",
    "FrontierPoint",
    "InfeasiblePlanError",
    "ModalPlan",
    "build_cost_panel",
    "build_emission_panel",
    "observed_fleet_caps",
    "optimise_modal_allocation",
    "status_quo",
    "sweep_frontier",
]
