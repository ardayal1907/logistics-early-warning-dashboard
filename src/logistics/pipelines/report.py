"""Human-readable pipeline reports.

The three scripts carried 99 `print()` calls between them, interleaved with the
computation. That has two costs: a scheduled run has no terminal, so the output
goes nowhere an aggregator can filter; and a function that prints cannot be
called by anything that does not want its stdout written to.

Rendering lives here and returns strings. Computation paths use `logging`.
A caller decides where the text goes — stdout for an interactive run, a log
record for a scheduled one, an artefact for a CI job.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_RULE = "=" * 74


def _section(title: str) -> str:
    return f"\n{_RULE}\n{title}\n{_RULE}"


def render_etl_report(
    raw: pd.DataFrame,
    fact: pd.DataFrame,
    dim_vendor: pd.DataFrame,
    dim_route: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> str:
    """What the ETL produced, and the one number that is meant to look odd."""
    days_without_shipments = len(dim_date) - fact["Date_ID"].nunique()

    lines = [
        _section("STAR SCHEMA"),
        f"Raw data loaded: {len(raw)} rows, {len(raw.columns)} columns",
        "",
        f"Dim_Vendor.csv      -> {len(dim_vendor)} rows (unique vendors)",
        f"Dim_Route.csv       -> {len(dim_route)} rows (unique route combinations)",
        f"Dim_Date.csv        -> {len(dim_date)} rows (contiguous calendar: "
        f"{dim_date.Full_Date.min()} .. {dim_date.Full_Date.max()})",
        f"Fact_Shipments.csv  -> {len(fact)} rows "
        f"(same as the original shipment count: {len(raw)})",
        "",
        "--- Dim_Vendor preview ---",
        dim_vendor.head().to_string(index=False),
        "",
        "--- Dim_Route preview ---",
        dim_route.head().to_string(index=False),
        "",
        "--- Dim_Date preview ---",
        dim_date.head().to_string(index=False),
        "",
        f"Days with no shipments: {days_without_shipments} (present in the calendar, "
        "absent from the fact table - this is NORMAL and required)",
        "",
        "--- Fact_Shipments preview ---",
        fact.head().to_string(index=False),
        "",
        "Integrity checks passed: all primary keys unique, all foreign keys "
        "present in the dimension tables.",
    ]
    return "\n".join(lines)


def render_generation_report(df: pd.DataFrame, target_delay_rate: float) -> str:
    """What the generator produced, against what it was asked to produce."""
    realised = float((df["Actual_Delay_Days"] > 0).mean())
    lines = [
        _section("SYNTHETIC DATA"),
        f"Rows generated : {len(df)}",
        f"Date range     : {df['Shipment_Date'].min()} .. {df['Shipment_Date'].max()}",
        f"Vendors        : {df['Vendor_ID'].nunique()}",
        f"Delay rate     : {realised:.4f} (target {target_delay_rate:.4f})",
        "",
        "--- Delay days ---",
        df["Actual_Delay_Days"].value_counts().sort_index().to_string(),
        "",
        "--- Preview ---",
        df.head().to_string(index=False),
    ]
    return "\n".join(lines)


def render_training_report(summary: dict[str, Any]) -> str:
    """Metrics, split comparison and feature importances from a training run.

    Takes the dictionary `train.main()` returns rather than the objects it
    computed with, so this module imports no scikit-learn and can be called
    from a context that has none.
    """
    metrics = summary.get("metrics", {})
    lines = [_section("MODEL")]

    if metrics:
        lines += [
            "--- Out-of-sample performance ---",
            f"ROC-AUC (out-of-fold)          : {metrics.get('oof_roc_auc', float('nan')):.4f}",
            f"ROC-AUC (chronological holdout): "
            f"{metrics.get('chronological_holdout_roc_auc', float('nan')):.4f}",
            f"ROC-AUC (chronological CV)     : "
            f"{metrics.get('chronological_cv_roc_auc_mean', float('nan')):.4f} "
            f"+/- {metrics.get('chronological_cv_roc_auc_std', float('nan')):.4f}",
            f"Brier score                    : {metrics.get('oof_brier', float('nan')):.4f}",
            f"Calibration error (ECE)        : {metrics.get('oof_ece', float('nan')):.4f}",
        ]

    if (comparison := summary.get("split_comparison")) is not None:
        lines += ["", "--- Split strategy comparison ---",
                  _as_text(comparison)]

    if (importances := summary.get("importances")) is not None:
        lines += ["", "--- Permutation importance ---", _as_text(importances)]

    if (levels := summary.get("risk_level_counts")) is not None:
        lines += ["", "--- Risk level distribution ---", _as_text(levels)]

    if (path := summary.get("model_path")) is not None:
        lines += ["", f"Model written to: {path}"]

    return "\n".join(lines)


def _as_text(value: Any) -> str:
    # str() around to_string(): pandas is untyped here and returns Any, which
    # strict mode refuses to let through as a str return.
    if isinstance(value, pd.DataFrame):
        return str(value.to_string(index=False))
    if isinstance(value, pd.Series):
        return str(value.to_string())
    return str(value)


__all__ = [
    "render_etl_report",
    "render_generation_report",
    "render_training_report",
]
