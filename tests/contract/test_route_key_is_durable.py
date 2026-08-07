"""Route_ID must survive an insert.

The defect this pins down: `Route_ID` was assigned from row position in a
sorted, de-duplicated frame. Adding one new route combination therefore
renumbered every existing route. Measured on the shipped data before the fix,
1,346 of 1,346 keys changed — and nothing failed, because the fact table still
pointed at keys that were all still valid and now meant different routes.

That is the shape of bug a test suite exists for: no exception, no warning, a
silently different answer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from logistics.contracts.shipments import DURABLE_ROUTE_ID_PATTERN
from logistics.pipelines import etl


@pytest.fixture(scope="module")
def raw(raw_data):
    return raw_data


def test_new_keys_match_the_durable_format(raw):
    dim = etl.build_dim_route(raw)
    assert dim["Route_ID"].str.fullmatch(DURABLE_ROUTE_ID_PATTERN[1:-1]).all()


def test_keys_are_unique_across_the_whole_dataset(raw):
    """40 bits over 1,346 rows: a collision is ~8e-7, and it must not be silent."""
    dim = etl.build_dim_route(raw)
    assert dim["Route_ID"].is_unique
    assert len(dim) == dim[etl.ROUTE_COLUMNS].drop_duplicates().shape[0]


def test_inserting_a_new_route_leaves_every_existing_key_untouched(raw):
    """The regression that motivated the change. Previously: 1346 of 1346 moved."""
    before = etl.build_dim_route(raw)

    extra = raw.iloc[[0]].copy()
    extra["Origin"] = "Zzzville"          # a combination that cannot already exist
    after = etl.build_dim_route(pd.concat([raw, extra], ignore_index=True))

    assert len(after) == len(before) + 1

    merged = before.merge(after, on=etl.ROUTE_COLUMNS, suffixes=("_before", "_after"))
    assert len(merged) == len(before), "a pre-existing route combination vanished"

    moved = merged[merged["Route_ID_before"] != merged["Route_ID_after"]]
    assert moved.empty, (
        f"{len(moved)} of {len(before)} Route_IDs changed when one route was "
        f"added. Incremental loading is impossible while this is true, and the "
        f"failure is silent: old facts keep pointing at a key that now means a "
        f"different route."
    )


def test_the_key_does_not_depend_on_row_order(raw):
    shuffled = raw.sample(frac=1.0, random_state=7).reset_index(drop=True)
    original = etl.build_dim_route(raw).set_index(etl.ROUTE_COLUMNS)["Route_ID"]
    reordered = etl.build_dim_route(shuffled).set_index(etl.ROUTE_COLUMNS)["Route_ID"]
    pd.testing.assert_series_equal(original.sort_index(), reordered.sort_index())


def test_the_key_changes_when_an_attribute_changes(raw):
    """A durable key is not a constant: a different route must get a different id."""
    dim = etl.build_dim_route(raw)
    first = dim.iloc[0]
    altered = etl.route_key(
        first["Origin"], first["Destination"], "Electric Semi",
        first["Weather_Condition"], first["Traffic_Density"],
    )
    same = etl.route_key(*[first[c] for c in etl.ROUTE_COLUMNS])
    assert same == first["Route_ID"]
    if first["Vehicle_Type"] != "Electric Semi":
        assert altered != first["Route_ID"]


def test_field_boundaries_cannot_be_forged():
    """("AB", "C") and ("A", "BC") must not collide through concatenation."""
    assert etl.route_key("AB", "C") != etl.route_key("A", "BC")


def test_the_committed_data_now_carries_durable_keys(dim_route):
    """The transition is complete: data/processed/ holds content-addressed keys.

    This replaces the test that used to assert the OPPOSITE - that the committed
    data still carried positional RT-00001 keys - which was correct for as long
    as the pipeline had not been re-run. It has been, so every key on disk is
    now sha1-derived and ROUTE_ID_PATTERN no longer accepts the positional form.
    """
    assert dim_route["Route_ID"].str.fullmatch(DURABLE_ROUTE_ID_PATTERN).all(), (
        "data/processed/Dim_Route.csv still contains positional keys; the ETL "
        "was not re-run against this data."
    )
    assert not dim_route["Route_ID"].str.fullmatch(r"RT-\d{5}").any()
    assert dim_route["Route_ID"].is_unique
