"""Python and DAX must agree on the numbers that reach the executive dashboard.

The carbon price lives in two places that cannot import from each other:

    src/config.py:64          CARBON_PRICE_PER_TON = 50.0
    powerbi/measures.dax:79   [Total CO2 Tons] * 50

Both the module docstring and the README document the duplication honestly. But
documenting is not solving. The only test guarding it was
`test_carbon_price_is_documented_as_manually_mirrored_in_dax`, which asserts
that the strings "Carbon Tax Impact" and "config.py" appear in the README and
that the word "DAX" appears in config.py. Under that test, changing
CARBON_PRICE_PER_TON to 85.0 leaves the whole suite green, the badge green and
the README still claiming the two are mirrored - while Streamlit reports $85/ton
and the board's dashboard reports $50/ton, each of them confident.

The measure is plain text in the repository, so the real assertion costs five
lines. This file makes it. The existing documentation test stays: it checks that
the duplication is disclosed, which is a different guarantee from checking that
the numbers match.

The duplication itself disappears at migration step 3, when the ETL emits
`Dim_Parameters.csv` and the measure becomes

    Carbon Tax Impact ($) = [Total CO2 Tons] * MAX ( Dim_Parameters[Carbon_Price_Per_Ton] )

reading the value from DATA rather than from a literal. At that point Power BI's
"cannot import from Python" constraint stops mattering, because it no longer has
to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from logistics.domain.carbon import DEFAULT_CARBON_PRICE_PER_TON
from logistics.domain.enums import RiskLevel

MEASURES = Path(__file__).resolve().parent.parent / "powerbi" / "measures.dax"

# `[Total CO2 Tons] * 50`, tolerant of whitespace and a decimal point.
CARBON_TAX_PATTERN = re.compile(
    r"\[\s*Total CO2 Tons\s*\]\s*\*\s*([0-9]+(?:\.[0-9]+)?)"
)


@pytest.fixture(scope="module")
def dax() -> str:
    if not MEASURES.exists():
        pytest.skip(f"{MEASURES.name} is missing; nothing to cross-check.")
    return MEASURES.read_text(encoding="utf-8")


def test_dax_carbon_price_equals_the_python_constant(dax):
    matches = CARBON_TAX_PATTERN.findall(dax)
    assert matches, (
        "Could not find the `[Total CO2 Tons] * <price>` multiplication in "
        f"{MEASURES.name}. If the measure was rewritten, update this pattern - "
        "do not delete the test: it is the only thing tying the report's carbon "
        "cost to src/config.py."
    )
    for literal in matches:
        assert float(literal) == pytest.approx(DEFAULT_CARBON_PRICE_PER_TON), (
            f"powerbi/measures.dax prices carbon at {literal}/ton while Python "
            f"uses {DEFAULT_CARBON_PRICE_PER_TON}/ton. The Streamlit app and the "
            f"Power BI report are reporting different money. Edit the DAX measure "
            f"by hand (Power BI cannot import from Python) or complete migration "
            f"step 3 and move the price into Dim_Parameters."
        )


def test_dax_risk_vocabulary_matches_the_enum(dax):
    """`Risk_Level = "High Risk"` in DAX is a string comparison against data the
    Python side writes. A renamed tier silently returns zero rows - Power BI
    shows a blank KPI rather than an error, which is the failure mode the star
    schema tests already guard against elsewhere."""
    quoted = set(re.findall(r'Risk_Level\s*\]?\s*=\s*"([^"]+)"', dax))
    if not quoted:
        pytest.skip("No literal Risk_Level comparison in the measures file.")

    vocabulary = {level.value for level in RiskLevel}
    unknown = quoted - vocabulary
    assert not unknown, (
        f"measures.dax filters on Risk_Level value(s) {sorted(unknown)} that the "
        f"Python side never produces. Accepted values: {sorted(vocabulary)}. "
        f"A non-matching filter yields a blank measure, not an error."
    )
