"""Console entry points.

`argparse`, `click`, `os.environ` and `getenv` appeared zero times across the
whole project. Every pipeline stage was invoked as `python src/<file>.py`, with
its paths frozen at import time, so an orchestrator could only ever call it by
file path and no stage could be pointed at a different environment.

These commands are installed by `pip install -e .` and are what an orchestrator
(Airflow, Fabric, Argo, or a Makefile) should call:

    logistics-generate      synthetic source data
    logistics-etl           raw CSV -> star schema
    logistics-train         train, evaluate and persist the model
    logistics-score         batch-score a CSV through the new service

Every one of them now runs on `logistics.pipelines`, which takes its
directories from `Settings`. The bridge imports of the legacy flat modules are
gone (V5 in docs/MIGRATION.md closed in migration step 4), so these commands
work from an installed wheel with no repository checkout:

    pip install .
    LOGISTICS_RAW_DIR=/data/raw LOGISTICS_PROCESSED_DIR=/data/out logistics-etl
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from logistics import __version__
from logistics.errors import LogisticsError
from logistics.settings import Settings

logger = logging.getLogger("logistics")

# Imported by name rather than from logistics.pipelines.train, so that
# `logistics-etl --help` does not pay for scikit-learn's import time.
TRAIN_STAGES = ("evaluate", "score", "train", "all")


def _configure_logging(level: str) -> None:
    """Replaces the 99 bare `print()` calls as the observability channel.

    A scheduled run has no terminal. Without a level, a timestamp and a logger
    name, log aggregation cannot filter, alert or correlate - and the data/model
    drift warning, currently visible only in a browser, reaches nothing.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
        force=True,
    )


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Repository root. Overrides LOGISTICS_PROJECT_ROOT.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _run(stage: str, fn) -> int:  # type: ignore[no-untyped-def]
    """Shared execution wrapper: log the boundaries, translate failures to exits."""
    logger.info("%s: starting", stage)
    try:
        fn()
    except LogisticsError as exc:
        logger.error("%s: failed - %s", stage, exc)
        return 1
    except Exception:
        logger.exception("%s: unexpected failure", stage)
        return 2
    logger.info("%s: done", stage)
    return 0


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def _settings_from(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        **({"project_root": args.project_root} if args.project_root else {})
    )


def generate_data(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Generate the synthetic source dataset.")
    parser.add_argument("--report", action="store_true",
                        help="Print the full summary to stdout as well as logging.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    from logistics.pipelines import generate
    from logistics.pipelines.report import render_generation_report

    settings = _settings_from(args)

    def _generate() -> None:
        frame = generate.main(settings)
        if args.report:
            print(render_generation_report(frame, generate.TARGET_DELAY_RATE))

    return _run("generate", _generate)


def run_etl(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Build the star schema from the raw CSV.")
    parser.add_argument("--report", action="store_true",
                        help="Print the full summary to stdout as well as logging.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    from logistics.pipelines import etl
    from logistics.pipelines.report import render_etl_report

    settings = _settings_from(args)

    def _etl() -> None:
        tables = etl.main(settings)
        if args.report:
            print(render_etl_report(**tables))

    return _run("etl", _etl)


def train_model(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Train, evaluate and persist the delay-risk model.")
    parser.add_argument("--stage", default="all", choices=list(TRAIN_STAGES),
                        help="Run one stage instead of the whole pipeline.")
    parser.add_argument("--report", action="store_true",
                        help="Print the metrics report to stdout as well as logging.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    from logistics.infrastructure.fingerprint import write_checksum_sidecar
    from logistics.pipelines import train
    from logistics.pipelines.report import render_training_report

    settings = _settings_from(args)

    def _train_then_seal() -> None:
        summary = train.main(settings, stage=args.stage)
        if args.report:
            print(render_training_report(summary))
        # Close the gap the pipeline left open: it hashes the DATA it trained on
        # but never the artefact, which is the only file whose deserialisation
        # executes code. The sidecar is what load_bundle() checks before joblib
        # touches the bytes.
        if args.stage in ("train", "all") and settings.model_path.exists():
            sidecar = write_checksum_sidecar(settings.model_path)
            logger.info("Wrote artefact checksum: %s", sidecar.name)

    return _run("train", _train_then_seal)


# ---------------------------------------------------------------------------
# Batch scoring - fully on the new service
# ---------------------------------------------------------------------------
def score_batch(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Score a CSV of shipments and write the results.")
    parser.add_argument("input_csv", type=Path,
                        help="CSV carrying the model's feature columns.")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Destination CSV.")
    parser.add_argument("--no-verify-checksum", action="store_true",
                        help="Skip the artefact integrity gate. Not for production.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    import pandas as pd

    from logistics.infrastructure.fingerprint import write_csv_deterministically
    from logistics.services.scoring import build_scoring_service

    overrides: dict[str, object] = {}
    if args.project_root:
        overrides["project_root"] = args.project_root
    if args.no_verify_checksum:
        overrides["verify_artifact_checksum"] = False

    def _score() -> None:
        service = build_scoring_service(Settings.from_env(**overrides))
        frame = pd.read_csv(args.input_csv)
        logger.info("Scoring %d row(s) with model %s",
                    len(frame), service.bundle.version)
        scored = service.score_frame(frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic line endings: the byte hash of an output must not depend
        # on the operating system that produced it.
        write_csv_deterministically(scored, args.output)
        logger.info("Wrote %s (%d rows, %d columns)",
                    args.output, len(scored), len(scored.columns))

    return _run("score", _score)


# ---------------------------------------------------------------------------
# Optimisation - Phase 1, Block B only
# ---------------------------------------------------------------------------
def optimise(argv: Sequence[str] | None = None) -> int:
    """Modal allocation: which vehicle type carries which shipment.

    Block A (vendor assignment) is NOT implemented; this command decides
    vehicles only, and no machine learning is involved in that decision.
    """
    parser = _base_parser("Optimise modal allocation (Phase 1, Block B).")
    parser.add_argument("--epsilon-kg", type=float, default=None,
                        help="CO2 ceiling in kg. Omit for pure cost minimisation.")
    parser.add_argument("--frontier", type=int, default=0, metavar="N",
                        help="Trace N points of the cost/CO2 Pareto frontier.")
    parser.add_argument("--skip-guard", action="store_true",
                        help="Skip the Vehicle_Type decision-invariance check. "
                             "Only when no model artefact is available.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write the assignment plan to this CSV.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    import pandas as pd

    from logistics.infrastructure.fingerprint import write_csv_deterministically
    from logistics.optimization import modal
    from logistics.optimization.guard import check_vehicle_invariance

    settings = _settings_from(args)

    def _optimise() -> None:
        fact = pd.read_csv(settings.processed_dir / "Fact_Shipments.csv")
        route = pd.read_csv(settings.processed_dir / "Dim_Route.csv")
        shipments = fact.merge(route, on="Route_ID", how="left")

        # docs/OPTIMIZATION.md §6 (v2): build the calibrated risk panel, guard on
        # IT, and hand it to the solver. A solver mines every spurious signal in
        # its objective to exhaustion, so what it is allowed to see matters more
        # than what it is forbidden from concluding.
        risk_panel = None
        if not args.skip_guard:
            from logistics.optimization.calibration import (
                calibrated_risk_panel,
                fit_vehicle_calibrations,
            )
            from logistics.optimization.guard import check_calibrated_vehicle_effect
            from logistics.services.scoring import build_scoring_service

            vendor = pd.read_csv(settings.processed_dir / "Dim_Vendor.csv")
            service = build_scoring_service(settings)
            featured = shipments.merge(vendor, on="Vendor_ID", how="left")

            observed = service.score_frame(featured)
            observed["Actual_Delay_Days"] = featured["Actual_Delay_Days"].to_numpy()
            observed["Vehicle_Type"] = featured["Vehicle_Type"].to_numpy()

            calibrations = fit_vehicle_calibrations(observed)
            risk_panel = calibrated_risk_panel(observed, calibrations)

            result = check_calibrated_vehicle_effect(risk_panel)
            logger.info("Decision-invariance guard (v2, calibrated): %s",
                        result.describe())

            raw = check_vehicle_invariance(service, featured, raise_on_failure=False)
            logger.info("For comparison, the raw v1 statistic: %s", raw.describe())

        baseline = modal.status_quo(shipments)
        logger.info("Status quo: %.1f t CO2, $%.0f freight",
                    baseline.co2_tons, baseline.freight_cost)

        if args.frontier:
            for point in modal.sweep_frontier(shipments, points=args.frontier):
                shadow = (f"{point.shadow_price_per_ton:8.2f}"
                          if point.shadow_price_per_ton is not None else "       -")
                logger.info("epsilon %9.1f kg | CO2 %8.1f t | cost $%10.0f | "
                            "shadow $%s/t",
                            point.epsilon_kg, point.co2_kg / 1000.0,
                            point.freight_cost, shadow)
            return

        plan = modal.optimise_modal_allocation(
            shipments, epsilon_kg=args.epsilon_kg, risk_panel=risk_panel)
        logger.info("Optimised: %.1f t CO2 (%.1f%% vs status quo), $%.0f freight",
                    plan.co2_tons,
                    100.0 * (plan.co2_kg - baseline.co2_kg) / baseline.co2_kg,
                    plan.freight_cost)
        if plan.expected_delays is not None:
            logger.info("Expected late shipments under this plan: %.1f of %d "
                        "(the SLA term the v2 objective is trading against)",
                        plan.expected_delays, len(shipments))
        if plan.fractional_shipments:
            logger.info("%d shipment(s) split across vehicles (expected: at most "
                        "one, the emission ceiling breaks total unimodularity)",
                        plan.fractional_shipments)
        logger.warning(
            "This is SYNTHETIC data. Vehicle assignment is now dispatched from "
            "distance, weight and vendor fleet mix rather than drawn at random "
            "(v2 generator), so the headroom is no longer pure noise - but it is "
            "still the headroom in a generator's dispatch rule, not in a real "
            "fleet. Quote it as a demonstration of the method, never as a saving "
            "an operator can bank."
        )

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_csv_deterministically(plan.assignment, args.output)
            logger.info("Wrote %s (%d rows)", args.output, len(plan.assignment))

    return _run("optimise", _optimise)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(score_batch())
