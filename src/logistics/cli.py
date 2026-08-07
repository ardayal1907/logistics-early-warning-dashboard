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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(score_batch())
