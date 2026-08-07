"""Data contracts — what a table must look like for the rest of the system to work.

These are Pandera schemas rather than a page in the README, so a violation is a
build failure instead of a surprise three joins downstream.

The categorical domains are DERIVED from `logistics.domain.enums`, never
retyped. Those enum values are the exact strings in the data files and in the
model's `categorical_levels` metadata; a second copy here would be a fifth
place for them to drift.
"""

from logistics.contracts.shipments import (
    DIM_ROUTE_SCHEMA,
    DIM_VENDOR_SCHEMA,
    FACT_SHIPMENTS_SCHEMA,
    RAW_SHIPMENTS_SCHEMA,
    SCORED_SHIPMENTS_SCHEMA,
    distance_consistency_report,
    validate_distance_consistency,
)

__all__ = [
    "DIM_ROUTE_SCHEMA",
    "DIM_VENDOR_SCHEMA",
    "FACT_SHIPMENTS_SCHEMA",
    "RAW_SHIPMENTS_SCHEMA",
    "SCORED_SHIPMENTS_SCHEMA",
    "distance_consistency_report",
    "validate_distance_consistency",
]
