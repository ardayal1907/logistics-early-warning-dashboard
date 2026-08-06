"""Block B — modal allocation.

The LP is small and fast, so most of these run on the full 1,500 shipments.
The two tests that matter most are at the bottom: the ones that stop the
optimiser being believed for the wrong reasons.
"""

from __future__ import annotations

import pandas as pd
import pytest

from logistics.domain.economics import COST_PER_TON_KM, FIXED_COST_PER_LEG, freight_cost
from logistics.errors import UnknownCategoryError
from logistics.optimization import modal
from logistics.optimization.guard import (
    MAX_VEHICLE_LEAK,
    DecisionInvarianceError,
    build_counterfactual_grid,
    check_vehicle_invariance,
)
from logistics.services.scoring import build_scoring_service
from logistics.settings import Settings


@pytest.fixture(scope="module")
def shipments(fact_with_ml, dim_route, dim_vendor):
    """Fact rows joined to route AND vendor attributes.

    Vendor_Rating comes from Dim_Vendor and is one of the model's six features,
    so the guard cannot score without it. The LP itself does not use it.
    """
    return (
        fact_with_ml
        .merge(dim_route, on="Route_ID", how="left")
        .merge(dim_vendor, on="Vendor_ID", how="left")
    )


@pytest.fixture(scope="module")
def panels(shipments):
    return modal.build_emission_panel(shipments), modal.build_cost_panel(shipments)


# --- economics: assumptions, but arithmetic that must still be right --------

def test_freight_cost_is_distance_times_tonnage_plus_fixed():
    expected = 450.0 * 12.0 * COST_PER_TON_KM["Diesel Truck"] \
        + FIXED_COST_PER_LEG["Diesel Truck"]
    assert freight_cost(450.0, 12.0, "Diesel Truck") == pytest.approx(expected)


def test_freight_cost_rejects_an_unknown_vehicle():
    with pytest.raises(UnknownCategoryError):
        freight_cost(1.0, 1.0, "Hovercraft")


# --- panels -----------------------------------------------------------------

def test_the_emission_panel_covers_every_shipment_and_vehicle(shipments, panels):
    emissions, _ = panels
    assert emissions.shape == (len(shipments), 3)
    assert list(emissions.columns) == modal.FLEET
    assert (emissions > 0).all().all()


def test_the_emission_panel_is_computed_not_read_from_the_column(shipments, panels):
    """The CO2_Emission_kg column carries generator noise and only covers the
    vehicle actually used. A counterfactual needs the formula."""
    emissions, _ = panels
    chosen = shipments["Vehicle_Type"].to_numpy()
    recomputed = pd.Series(
        [emissions.iloc[i][v] for i, v in enumerate(chosen)], index=shipments.index
    )
    stored = shipments["CO2_Emission_kg"]

    # Close, because it is the same formula, but NOT equal - that difference is
    # exactly the N(1, 0.03) noise, and attributing it to the optimiser would
    # be a fabricated saving.
    assert recomputed.sum() == pytest.approx(stored.sum(), rel=0.01)
    assert not recomputed.equals(stored)


# --- the programme ----------------------------------------------------------

def test_every_shipment_is_assigned_exactly_once(shipments, panels):
    """V1. Shares may split, but they must total one."""
    emissions, costs = panels
    plan = modal.optimise_modal_allocation(
        shipments, emissions=emissions, costs=costs)
    totals = plan.assignment.groupby("Shipment_ID")["share"].sum()
    assert len(totals) == len(shipments)
    assert totals.between(1 - 1e-6, 1 + 1e-6).all()


def test_the_fleet_caps_are_respected(shipments, panels):
    """V2. Without this the answer is 'electrify everything', which is capex."""
    emissions, costs = panels
    caps = modal.observed_fleet_caps(shipments)
    plan = modal.optimise_modal_allocation(
        shipments, emissions=emissions, costs=costs, fleet_caps=caps)

    for vehicle, used in plan.vehicle_counts.items():
        assert used <= caps[vehicle] + 1e-6, f"{vehicle} exceeded its cap"


def test_the_optimum_is_no_worse_than_the_status_quo(shipments, panels):
    emissions, costs = panels
    baseline = modal.status_quo(shipments, emissions, costs)
    plan = modal.optimise_modal_allocation(
        shipments, emissions=emissions, costs=costs)
    assert plan.freight_cost <= baseline.freight_cost + 1e-6


def test_the_relaxation_is_essentially_integral(shipments, panels):
    """V1+V2 form a transportation polytope, so binaries buy nothing.

    Without the emission ceiling the polytope is totally unimodular and every
    vertex is integral - so zero fractional shipments, exactly.

    The ceiling (V4) breaks TU. The textbook bound is "at most one fractional
    shipment in a BASIC optimal solution", and the solver is under no
    obligation to return a basic one: an interior-point method lands on a
    point in the middle of an optimal face. Measured here: 2 of 1,500. The
    assertion is therefore "negligible", not "exactly one" - claiming the
    tighter bound would be asserting something the solver does not promise.
    """
    emissions, costs = panels
    plan = modal.optimise_modal_allocation(
        shipments, emissions=emissions, costs=costs)
    assert plan.fractional_shipments == 0, "the TU relaxation must be integral"

    baseline = modal.status_quo(shipments, emissions, costs)
    tightened = modal.optimise_modal_allocation(
        shipments, epsilon_kg=baseline.co2_kg * 0.80,
        emissions=emissions, costs=costs)
    assert tightened.fractional_shipments <= 5, (
        f"{tightened.fractional_shipments} of {len(shipments)} shipments split "
        f"across vehicles. One side constraint should perturb only a handful; "
        f"many more suggests the formulation, not the solver."
    )


def test_a_binding_ceiling_produces_a_positive_shadow_price(shipments, panels):
    """The dual of V4 is the marginal abatement cost, in dollars per tonne."""
    emissions, costs = panels
    loose = modal.optimise_modal_allocation(
        shipments, emissions=emissions, costs=costs)

    binding = modal.optimise_modal_allocation(
        shipments, epsilon_kg=loose.co2_kg * 0.95,
        emissions=emissions, costs=costs)

    assert binding.co2_kg <= loose.co2_kg * 0.95 + 1e-3
    assert binding.shadow_price_per_ton is not None
    assert binding.shadow_price_per_ton > 0
    # Tightening cannot make freight cheaper.
    assert binding.freight_cost >= loose.freight_cost - 1e-6


def test_a_slack_ceiling_has_a_zero_shadow_price(shipments, panels):
    """A non-binding constraint is worth nothing at the margin."""
    emissions, costs = panels
    loose = modal.optimise_modal_allocation(
        shipments, emissions=emissions, costs=costs)
    plan = modal.optimise_modal_allocation(
        shipments, epsilon_kg=loose.co2_kg * 1.10,
        emissions=emissions, costs=costs)
    assert plan.shadow_price_per_ton == pytest.approx(0.0, abs=1e-6)


def test_an_impossible_ceiling_is_reported_not_silently_relaxed(shipments, panels):
    emissions, costs = panels
    with pytest.raises(modal.InfeasiblePlanError):
        modal.optimise_modal_allocation(
            shipments, epsilon_kg=1.0, emissions=emissions, costs=costs)


def test_the_frontier_is_monotone(shipments):
    """Less carbon must never cost less. A crossing would mean a modelling bug."""
    frontier = modal.sweep_frontier(shipments, points=5)
    assert len(frontier) >= 3

    ordered = sorted(frontier, key=lambda p: p.co2_kg)
    costs = [p.freight_cost for p in ordered]
    assert costs == sorted(costs, reverse=True), (
        "freight cost must fall as the emission ceiling is relaxed"
    )


# --- the guard: what stops this being believed for the wrong reasons --------

def test_the_counterfactual_grid_varies_only_the_vehicle(shipments):
    grid = build_counterfactual_grid(shipments.head(10))
    assert len(grid) == 30
    assert set(grid["Vehicle_Type"]) == set(modal.FLEET)
    # Everything else is held fixed, which is what makes the spread attributable.
    for column in ("Distance_km", "Weight_tons", "Weather_Condition"):
        assert grid.groupby("Shipment_ID")[column].nunique().max() == 1


@pytest.mark.slow
def test_vehicle_type_leaks_into_the_probability(repo_root, shipments):
    """MEASURED FAILURE, recorded rather than hidden.

    docs/OPTIMIZATION.md §6 requires the worst per-shipment spread to be under
    0.05 before Block B's separability may be assumed. On this artefact it is
    ~0.62 - twelve times the bound.

    The weaker check (comparing GROUP MEANS across vehicle types) gives 0.027
    and passes, which is precisely the trap the doc warns about: "a near-zero
    permutation importance does not bound the per-row spread".

    This test asserts the failure so it cannot be forgotten. When the feature
    is dropped and the model retrained, it turns red and must be rewritten as
    the passing assertion.
    """
    settings = Settings.from_env(
        project_root=repo_root, verify_artifact_checksum=False)
    service = build_scoring_service(settings)

    result = check_vehicle_invariance(service, shipments, raise_on_failure=False)

    assert not result.passed
    assert result.leak > 0.5, (
        f"The leak was {result.leak:.4f}. If it has fallen below "
        f"{MAX_VEHICLE_LEAK}, the model was retrained without Vehicle_Type - "
        f"rewrite this test as `assert result.passed` and drop --skip-guard "
        f"from the documented workflow."
    )

    # The group-mean view, which is the one that misleads.
    means = result.per_vehicle_mean
    assert max(means.values()) - min(means.values()) < MAX_VEHICLE_LEAK


@pytest.mark.slow
def test_the_guard_raises_by_default(repo_root, shipments):
    """Continuing quietly is not an option - docs/OPTIMIZATION.md §6."""
    settings = Settings.from_env(
        project_root=repo_root, verify_artifact_checksum=False)
    service = build_scoring_service(settings)

    with pytest.raises(DecisionInvarianceError, match="leaks into the delay probability"):
        check_vehicle_invariance(service, shipments.head(200))
