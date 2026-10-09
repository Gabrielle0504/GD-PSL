"""Command-line entry point for the GD-PSL fill-then-judge pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional, Sequence

from gd_psl.config import ALLOWED_EA_FILL_SPLITS, DEFAULT_CONFIG, ExperimentConfig
from gd_psl.runner import run_experiments


def _build_parser(defaults: ExperimentConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a PlatEMO search followed by GD-PSL front completion."
    )
    parser.add_argument("--algorithms", nargs="+", default=list(defaults.algorithms))
    parser.add_argument("--problems", nargs="+", default=list(defaults.problems))
    parser.add_argument("--n-objectives", type=int, default=defaults.n_objectives)
    parser.add_argument("--runs", type=int, default=defaults.runs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--population-size", type=int, default=defaults.population_size)
    parser.add_argument(
        "--population-size-per-dimension",
        type=int,
        default=defaults.population_size_per_dimension,
    )
    parser.add_argument("--max-fe", type=int, default=defaults.max_fe)
    parser.add_argument(
        "--max-fe-per-dimension", type=int, default=defaults.max_fe_per_dimension
    )
    parser.add_argument(
        "--ea-fill-split", choices=ALLOWED_EA_FILL_SPLITS, default=defaults.ea_fill_split
    )
    parser.add_argument(
        "--ea-only", action=argparse.BooleanOptionalAction, default=defaults.ea_only
    )
    parser.add_argument(
        "--candidate-pool-size", type=int, default=defaults.candidate_pool_size
    )
    parser.add_argument(
        "--gap-threshold-factor", type=float, default=defaults.gap_threshold_factor
    )
    parser.add_argument(
        "--hole-support-neighbors", type=int, default=defaults.hole_support_neighbors
    )
    parser.add_argument(
        "--hole-minimum-support", type=float, default=defaults.hole_minimum_support
    )
    parser.add_argument(
        "--hole-extreme-gap-quantile",
        type=float,
        default=defaults.hole_extreme_gap_quantile,
    )
    parser.add_argument(
        "--hole-pilot-fraction", type=float, default=defaults.hole_pilot_fraction
    )
    parser.add_argument(
        "--hole-success-neighbors", type=int, default=defaults.hole_success_neighbors
    )
    parser.add_argument(
        "--hole-local-perturbation-scale",
        type=float,
        default=defaults.hole_local_perturbation_scale,
    )
    parser.add_argument(
        "--hole-boundary-probe-fraction",
        type=float,
        default=defaults.hole_boundary_probe_fraction,
    )
    parser.add_argument(
        "--hole-boundary-probe-support-floor",
        type=float,
        default=defaults.hole_boundary_probe_support_floor,
    )
    parser.add_argument(
        "--hole-boundary-probe-branches",
        type=int,
        default=defaults.hole_boundary_probe_branches,
    )
    parser.add_argument(
        "--exhaust-model-fe-budget",
        action=argparse.BooleanOptionalAction,
        default=defaults.exhaust_model_fe_budget,
    )
    parser.add_argument(
        "--direction-match-factor", type=float, default=defaults.direction_match_factor
    )
    parser.add_argument(
        "--local-front-neighbors", type=int, default=defaults.local_front_neighbors
    )
    parser.add_argument(
        "--local-front-error-quantile",
        type=float,
        default=defaults.local_front_error_quantile,
    )
    parser.add_argument(
        "--local-front-min-neighbors",
        type=int,
        default=defaults.local_front_min_neighbors,
    )
    parser.add_argument("--local-front-ridge", type=float, default=defaults.local_front_ridge)
    parser.add_argument(
        "--local-front-calibration-samples",
        type=int,
        default=defaults.local_front_calibration_samples,
    )
    parser.add_argument(
        "--local-refinement-enabled",
        action=argparse.BooleanOptionalAction,
        default=defaults.local_refinement_enabled,
    )
    parser.add_argument(
        "--local-refinement-candidate-neighbors",
        type=int,
        default=defaults.local_refinement_candidate_neighbors,
    )
    parser.add_argument(
        "--local-refinement-branch-neighbors",
        type=int,
        default=defaults.local_refinement_branch_neighbors,
    )
    parser.add_argument(
        "--local-refinement-ridge", type=float, default=defaults.local_refinement_ridge
    )
    parser.add_argument("--hidden-width", type=int, default=defaults.hidden_width)
    parser.add_argument("--pretrain-epochs", type=int, default=defaults.pretrain_epochs)
    parser.add_argument(
        "--minimum-training-epochs", type=int, default=defaults.minimum_training_epochs
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=defaults.early_stopping_patience
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=defaults.early_stopping_min_delta
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--fine-tune-epochs", type=int, default=defaults.fine_tune_epochs)
    parser.add_argument(
        "--fine-tune-batch-size", type=int, default=defaults.fine_tune_batch_size
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=defaults.fine_tune_learning_rate,
    )
    parser.add_argument(
        "--model-inference-batch-size",
        type=int,
        default=defaults.model_inference_batch_size,
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--optimizer",
        choices=("auto", "adamw", "schedulefree_adamw"),
        default=defaults.optimizer,
    )
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--hypervolume-reference-value",
        type=float,
        default=defaults.hypervolume_reference_value,
    )
    parser.add_argument(
        "--hypervolume-monte-carlo-samples",
        type=int,
        default=defaults.hypervolume_monte_carlo_samples,
    )
    parser.add_argument("--hypervolume-seed", type=int, default=defaults.hypervolume_seed)
    parser.add_argument(
        "--igd-infinity-reference-samples",
        type=int,
        default=defaults.igd_infinity_reference_samples,
    )
    parser.add_argument(
        "--feasibility-tolerance", type=float, default=defaults.feasibility_tolerance
    )
    parser.add_argument(
        "--consistency-sample-size", type=int, default=defaults.consistency_sample_size
    )
    parser.add_argument(
        "--consistency-relative-tolerance",
        type=float,
        default=defaults.consistency_relative_tolerance,
    )
    parser.add_argument(
        "--consistency-absolute-tolerance",
        type=float,
        default=defaults.consistency_absolute_tolerance,
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=defaults.plot
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=defaults.skip_completed,
    )
    parser.add_argument("--platemo-root", default=defaults.platemo_root)
    parser.add_argument(
        "--algorithm-parameters",
        nargs="*",
        type=float,
        default=list(defaults.algorithm_parameters),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    defaults = DEFAULT_CONFIG.validate()
    values = vars(_build_parser(defaults).parse_args(argv))
    values["algorithms"] = tuple(values["algorithms"])
    values["problems"] = tuple(problem.lower() for problem in values["problems"])
    values["algorithm_parameters"] = tuple(values["algorithm_parameters"])
    return replace(defaults, **values).validate()


if __name__ == "__main__":
    config = parse_args()
    run_experiments(config, skip_completed=config.skip_completed)
