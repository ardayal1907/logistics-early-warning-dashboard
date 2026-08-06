"""Compatibility shim — training now lives in `logistics.pipelines.train`.

`python src/ml_delay_risk_pipeline.py` keeps working. New callers should use
the console script, which takes its directories from the environment and can
run one stage at a time:

    logistics-train                     # evaluate + score + train
    logistics-train --stage evaluate    # measure only, write nothing

The risk constants are re-exported through `config` rather than imported from
the domain package directly, so that `ml_delay_risk_pipeline.classify_risk is
config.classify_risk` stays true. tests/test_single_source_of_truth.py asserts
exactly that.
"""

from config import (
    COST_FN_OVER_FP,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    classify_risk,
    compute_data_fingerprint,
)
from logistics.pipelines.train import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    FEATURE_COLS,
    METADATA_FILENAME,
    MODEL_FILENAME,
    N_SPLITS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SCORED_FILENAME,
    STAGES,
    add_target,
    alert_cost,
    assign_risk_levels,
    build_analytical_dataset,
    build_base_pipeline,
    build_fact_with_ml,
    build_metadata,
    build_model,
    calibration_check,
    chronological_cv,
    chronological_split,
    compare_split_strategies,
    evaluate_chronologically,
    expected_calibration_error,
    group_name,
    impurity_importance,
    load_star_schema,
    main,
    permutation_importance_table,
    run_evaluation,
    run_scoring,
    run_training,
    save_production_model,
    score_walk_forward,
    scoring_quality,
    train_production_model,
    validate_output_schema,
    verify_saved_model,
    weather_distribution_shift,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "COST_FN_OVER_FP",
    "EXPECTED_COLUMNS",
    "FEATURE_COLS",
    "HIGH_RISK_THRESHOLD",
    "MEDIUM_RISK_THRESHOLD",
    "METADATA_FILENAME",
    "MODEL_FILENAME",
    "NUMERIC_FEATURES",
    "N_SPLITS",
    "RANDOM_STATE",
    "SCORED_FILENAME",
    "STAGES",
    "add_target",
    "alert_cost",
    "assign_risk_levels",
    "build_analytical_dataset",
    "build_base_pipeline",
    "build_fact_with_ml",
    "build_metadata",
    "build_model",
    "calibration_check",
    "chronological_cv",
    "chronological_split",
    "classify_risk",
    "compare_split_strategies",
    "compute_data_fingerprint",
    "evaluate_chronologically",
    "expected_calibration_error",
    "group_name",
    "impurity_importance",
    "load_star_schema",
    "main",
    "permutation_importance_table",
    "run_evaluation",
    "run_scoring",
    "run_training",
    "save_production_model",
    "score_walk_forward",
    "scoring_quality",
    "train_production_model",
    "validate_output_schema",
    "verify_saved_model",
    "weather_distribution_shift",
]


if __name__ == "__main__":
    import logging

    from logistics.pipelines.report import render_training_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    print(render_training_report(main()))
