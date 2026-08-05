"""The carbon and risk constants must exist in exactly one place: src/config.py.

Before the scripts were refactored into functions, these checks had to be static
(AST-based), because importing `app.py` launched Streamlit and importing a pipeline
script re-ran the whole pipeline. Now every module is import-safe, so the checks are
made by IDENTITY (`is`) — a far stronger guarantee than equal values: if a consumer
re-declared a constant, it would hold a different object even when the number matched.

One AST guard remains, deliberately and not as a workaround: `is` comparison on floats
is CPython-implementation-dependent, so a re-declared scalar could accidentally pass an
identity check. Asserting that no module-level assignment statement exists is the
precise way to express "this name must be imported, never assigned".
"""

import ast

import app
import config
import generate_logistics_data as gen
import ml_delay_risk_pipeline as ml
import pytest

SHARED_NAMES = {
    "EMISSION_FACTOR", "BASE_EMISSION", "WEATHER_CO2_MULT", "TRAFFIC_CO2_MULT",
    "MIN_CO2_KG", "CARBON_PRICE_PER_TON", "COST_FN_OVER_FP",
    "HIGH_RISK_THRESHOLD", "MEDIUM_RISK_THRESHOLD",
}
SHARED_FUNCS = {"compute_co2_kg", "classify_risk"}
CONSUMERS = ["app.py", "generate_logistics_data.py", "ml_delay_risk_pipeline.py"]


# --- Identity: the consumers hold the very same objects --------------------

def test_app_shares_the_carbon_function_object():
    assert app.compute_co2_kg is config.compute_co2_kg


def test_app_shares_the_risk_function_object():
    assert app.classify_risk is config.classify_risk


def test_pipeline_shares_the_risk_function_object():
    assert ml.classify_risk is config.classify_risk


def test_generator_shares_the_emission_dictionaries():
    """Dictionaries make identity checks airtight - a copy would fail immediately."""
    assert gen.EMISSION_FACTOR is config.EMISSION_FACTOR
    assert gen.BASE_EMISSION is config.BASE_EMISSION
    assert gen.WEATHER_CO2_MULT is config.WEATHER_CO2_MULT
    assert gen.TRAFFIC_CO2_MULT is config.TRAFFIC_CO2_MULT


def test_pipeline_thresholds_are_the_config_values():
    assert ml.HIGH_RISK_THRESHOLD == config.HIGH_RISK_THRESHOLD
    assert ml.MEDIUM_RISK_THRESHOLD == config.MEDIUM_RISK_THRESHOLD
    assert ml.COST_FN_OVER_FP == config.COST_FN_OVER_FP


def test_the_generator_and_the_app_agree_on_a_concrete_case():
    """Same inputs, same deterministic figure, through two different modules."""
    args = (450.0, 12.0, "Diesel Truck", "Storm", "High")
    assert app.compute_co2_kg(*args) == config.compute_co2_kg(*args)
    manual = (450.0 * 12.0 * gen.EMISSION_FACTOR["Diesel Truck"]
              + gen.BASE_EMISSION["Diesel Truck"]) \
        * gen.WEATHER_CO2_MULT["Storm"] * gen.TRAFFIC_CO2_MULT["High"]
    assert config.compute_co2_kg(*args) == pytest.approx(manual)


# --- Import safety: the refactor's core promise ----------------------------

@pytest.mark.parametrize("module", [app, gen, ml],
                         ids=["app", "generate_logistics_data", "ml_delay_risk_pipeline"])
def test_modules_expose_a_main_entry_point(module):
    """Importing must do nothing; work happens only when main() is called.

    That these modules were importable at all during this test session is the
    proof - a side-effecting module would have run the pipeline on import.
    """
    assert callable(getattr(module, "main", None)), \
        f"{module.__name__} has no main(); its body would run on import."


@pytest.mark.parametrize("filename", CONSUMERS)
def test_scripts_guard_execution_behind_main(repo_root, filename):
    source = (repo_root / "src" / filename).read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in source, \
        f"src/{filename} is missing the `if __name__ == \"__main__\"` guard."


# --- The one remaining static guard, with its reason ----------------------

@pytest.mark.parametrize("filename", CONSUMERS)
def test_consumers_never_assign_a_shared_constant(repo_root, filename):
    """`is` on floats is unreliable, so scalars are checked syntactically."""
    tree = ast.parse((repo_root / "src" / filename).read_text(encoding="utf-8"))
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    clash = assigned & SHARED_NAMES
    assert not clash, (
        f"src/{filename} re-declares {sorted(clash)} instead of importing them from "
        "config.py. Duplicated constants drift apart silently - that is exactly what "
        "config.py was created to prevent."
    )


@pytest.mark.parametrize("filename", CONSUMERS)
def test_consumers_never_redefine_a_shared_function(repo_root, filename):
    tree = ast.parse((repo_root / "src" / filename).read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert not (defined & SHARED_FUNCS), (
        f"src/{filename} defines its own {sorted(defined & SHARED_FUNCS)}; it should "
        "import the shared implementation from config.py."
    )


def test_config_defines_every_shared_name():
    missing = SHARED_NAMES - set(vars(config))
    assert not missing, f"src/config.py does not define: {sorted(missing)}"
    missing_funcs = SHARED_FUNCS - set(vars(config))
    assert not missing_funcs, f"src/config.py does not define: {sorted(missing_funcs)}"


def test_carbon_price_is_documented_as_manually_mirrored_in_dax(repo_root):
    """The DAX copy cannot be imported, so the duplication must at least be flagged."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "Carbon Tax Impact" in readme and "config.py" in readme, (
        "README.md must document that CARBON_PRICE_PER_TON is mirrored by hand in "
        "the Power BI Carbon Tax Impact DAX measure."
    )
    assert "DAX" in (repo_root / "src" / "config.py").read_text(encoding="utf-8"), \
        "src/config.py should warn that the Power BI measure must be edited by hand."
