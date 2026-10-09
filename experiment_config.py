"""Single editable configuration source for fill-then-judge experiments.

Edit ``DEFAULT_CONFIG`` below, then run ``python experiment_config.py`` to
validate and print the effective configuration.  Command-line options in
``run_fill_then_judge.py`` may still override these defaults for one run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional


# Numerical/implementation settings used across the pipeline.
NUMERICAL_EPSILON = 1e-12
NONDOMINATED_CHUNK_SIZE = 512
COMPLETION_METHOD = "GD-PSL"
COMPARISON_POPULATION_SIZE = 91
COMPARISON_TOTAL_FE_BUDGET = 100_000
IGD_INFINITY_REFERENCE_SAMPLES = 50_000
ALLOWED_EA_FILL_SPLITS = ("50:50", "60:40", "70:30", "80:20", "90:10")


@dataclass(frozen=True)
class ExperimentConfig:
    # Experiment task.
    algorithms: tuple[str, ...] = ("NSGAII",)
    problems: tuple[str, ...] = ("dtlz7",)
    n_objectives: Optional[int] = 3
    runs: int = 20
    seed: int = 1

    # Common true-evaluation budget B for fair comparisons. An EA-only
    # baseline receives all B evaluations. This completion method splits B
    # between its EA stage and model-candidate evaluations.
    population_size: Optional[int] = COMPARISON_POPULATION_SIZE
    population_size_per_dimension: int = 100
    max_fe: Optional[int] = COMPARISON_TOTAL_FE_BUDGET
    max_fe_per_dimension: int = 10000
    algorithm_parameters: tuple[float, ...] = ()
    platemo_root: str = os.environ.get("PLATEMO_ROOT", "")

    # Preference-space hole detection.
    candidate_pool_size: int = 50000
    ea_fill_split: str = "90:10"
    ea_only: bool = False
    gap_threshold_factor: float = 2.5
    hole_support_neighbors: int = 32
    hole_minimum_support: float = 0.20
    hole_extreme_gap_quantile: float = 0.95
    hole_pilot_fraction: float = 0.2
    hole_success_neighbors: int = 16
    hole_local_perturbation_scale: float = 1.0
    hole_boundary_probe_fraction: float = 0.10
    hole_boundary_probe_support_floor: float = 0.05
    hole_boundary_probe_branches: int = 3
    # Spend the complete model FE allocation for equal-total-FE comparisons.
    # Queries after the geometric hole threshold is met are labelled as
    # coverage refinement rather than additional holes.
    exhaust_model_fe_budget: bool = True

    # Runtime-only candidate validation. These checks use the empirical EA
    # archive and evaluated model candidates; no reference PF is consulted.
    direction_match_factor: float = 2.5
    local_front_neighbors: int = 20
    local_front_error_quantile: float = 0.99
    local_front_min_neighbors: int = 8
    local_front_ridge: float = 1e-8
    local_front_calibration_samples: int = 4096

    # Reference-free output correction. The MLP selects one nearby decision
    # branch; a local archive fit then places the candidate at the requested
    # preference without spending an extra true evaluation.
    local_refinement_enabled: bool = True
    local_refinement_candidate_neighbors: int = 64
    local_refinement_branch_neighbors: int = 32
    local_refinement_ridge: float = 1e-6

    # Pareto-set model and optimizer.
    hidden_width: int = 1024
    pretrain_epochs: int = 200  # Maximum epochs; early stopping may finish sooner.
    minimum_training_epochs: int = 20
    validation_fraction: float = 0.2
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    batch_size: int = 4096
    fine_tune_epochs: int = 15
    fine_tune_batch_size: int = 1024
    fine_tune_learning_rate: float = 0.00025
    model_inference_batch_size: int = 4096
    learning_rate: float = 0.0025
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # adamw, schedulefree_adamw, or legacy auto resolution
    warmup_steps: int = 10
    device: str = "cuda"

    # Hypervolume reporting after fixed external ideal/nadir normalization.
    hypervolume_reference_value: float = 1.1
    hypervolume_monte_carlo_samples: int = 131072
    hypervolume_seed: int = 2026
    igd_infinity_reference_samples: int = IGD_INFINITY_REFERENCE_SAMPLES

    # Feasibility and MATLAB/Python consistency checks.
    feasibility_tolerance: float = 1e-12
    consistency_sample_size: int = 32
    consistency_relative_tolerance: float = 1e-4
    consistency_absolute_tolerance: float = 1e-7

    # Artifacts.
    plot: bool = False
    output_dir: str = "results"
    completion_method: str = COMPLETION_METHOD
    skip_completed: bool = False

    def validate(self) -> "ExperimentConfig":
        if not self.algorithms or any(not value.strip() for value in self.algorithms):
            raise ValueError("algorithms must contain at least one non-empty PlatEMO algorithm name")
        if not self.problems or any(not value.strip() for value in self.problems):
            raise ValueError("problems must contain at least one non-empty problem name")
        if self.n_objectives is not None and self.n_objectives < 2:
            raise ValueError("n_objectives must be at least 2")
        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if self.population_size is not None and self.population_size <= 0:
            raise ValueError("population_size must be positive or None")
        if self.population_size_per_dimension <= 0:
            raise ValueError("population_size_per_dimension must be positive")
        if self.max_fe is not None and self.max_fe <= 0:
            raise ValueError("max_fe must be positive or None")
        if self.max_fe_per_dimension <= 0:
            raise ValueError("max_fe_per_dimension must be positive")
        if self.candidate_pool_size < 2:
            raise ValueError("candidate_pool_size must be at least 2")
        if self.ea_fill_split not in ALLOWED_EA_FILL_SPLITS:
            raise ValueError(
                f"ea_fill_split must be one of {', '.join(ALLOWED_EA_FILL_SPLITS)}"
            )
        if self.gap_threshold_factor < 0:
            raise ValueError("gap_threshold_factor cannot be negative")
        if self.hole_support_neighbors < 2:
            raise ValueError("hole_support_neighbors must be at least 2")
        if not 0.0 <= self.hole_minimum_support <= 1.0:
            raise ValueError("hole_minimum_support must be in [0, 1]")
        if not 0.5 < self.hole_extreme_gap_quantile < 1.0:
            raise ValueError("hole_extreme_gap_quantile must be between 0.5 and 1")
        if not 0.0 < self.hole_pilot_fraction <= 1.0:
            raise ValueError("hole_pilot_fraction must be in (0, 1]")
        if self.hole_success_neighbors < 1:
            raise ValueError("hole_success_neighbors must be positive")
        if self.hole_local_perturbation_scale <= 0:
            raise ValueError("hole_local_perturbation_scale must be positive")
        if not 0.0 <= self.hole_boundary_probe_fraction < 1.0:
            raise ValueError("hole_boundary_probe_fraction must be in [0, 1)")
        if not 0.0 <= self.hole_boundary_probe_support_floor < self.hole_minimum_support:
            raise ValueError(
                "hole_boundary_probe_support_floor must be below hole_minimum_support"
            )
        if self.hole_boundary_probe_branches < 1:
            raise ValueError("hole_boundary_probe_branches must be positive")
        if self.direction_match_factor <= 0:
            raise ValueError("direction_match_factor must be positive")
        if self.local_front_neighbors <= 1:
            raise ValueError("local_front_neighbors must be greater than 1")
        if not 0.5 < self.local_front_error_quantile < 1.0:
            raise ValueError("local_front_error_quantile must be between 0.5 and 1")
        if self.local_front_min_neighbors < 2:
            raise ValueError("local_front_min_neighbors must be at least 2")
        if self.local_front_min_neighbors > self.local_front_neighbors:
            raise ValueError("local_front_min_neighbors cannot exceed local_front_neighbors")
        if self.local_front_ridge <= 0:
            raise ValueError("local_front_ridge must be positive")
        if self.local_front_calibration_samples < 2:
            raise ValueError("local_front_calibration_samples must be at least 2")
        if self.local_refinement_candidate_neighbors < 2:
            raise ValueError("local_refinement_candidate_neighbors must be at least 2")
        if self.local_refinement_branch_neighbors < 2:
            raise ValueError("local_refinement_branch_neighbors must be at least 2")
        if self.local_refinement_branch_neighbors > self.local_refinement_candidate_neighbors:
            raise ValueError(
                "local_refinement_branch_neighbors cannot exceed "
                "local_refinement_candidate_neighbors"
            )
        if self.local_refinement_ridge <= 0:
            raise ValueError("local_refinement_ridge must be positive")
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if self.pretrain_epochs <= 0:
            raise ValueError("pretrain_epochs must be positive")
        if not 1 <= self.minimum_training_epochs <= self.pretrain_epochs:
            raise ValueError("minimum_training_epochs must be in [1, pretrain_epochs]")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.fine_tune_epochs < 0:
            raise ValueError("fine_tune_epochs cannot be negative")
        if self.fine_tune_batch_size <= 0:
            raise ValueError("fine_tune_batch_size must be positive")
        if self.fine_tune_learning_rate <= 0:
            raise ValueError("fine_tune_learning_rate must be positive")
        if self.model_inference_batch_size <= 0:
            raise ValueError("model_inference_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.optimizer not in {"auto", "adamw", "schedulefree_adamw"}:
            raise ValueError("optimizer must be auto, adamw, or schedulefree_adamw")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if not self.device.strip():
            raise ValueError("device cannot be empty")
        if self.hypervolume_reference_value <= 1.0:
            raise ValueError("hypervolume_reference_value must be greater than 1")
        if self.hypervolume_monte_carlo_samples <= 0:
            raise ValueError("hypervolume_monte_carlo_samples must be positive")
        if self.igd_infinity_reference_samples < 2:
            raise ValueError("igd_infinity_reference_samples must be at least 2")
        if self.feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance cannot be negative")
        if self.consistency_sample_size <= 0:
            raise ValueError("consistency_sample_size must be positive")
        if self.consistency_relative_tolerance < 0 or self.consistency_absolute_tolerance < 0:
            raise ValueError("consistency tolerances cannot be negative")
        if not self.output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        if self.completion_method != COMPLETION_METHOD:
            raise ValueError(f"completion_method must remain {COMPLETION_METHOD!r}")
        return self

    def resolve_evaluation_budgets(self, n_dimensions: int) -> tuple[int, int, int]:
        """Return common total, completion-EA, and completion-model FE budgets."""
        if n_dimensions <= 0:
            raise ValueError("n_dimensions must be positive")
        total = self.max_fe
        if total is None:
            total = self.max_fe_per_dimension * n_dimensions

        if self.ea_only:
            return total, total, 0
        ea_percent = int(self.ea_fill_split.split(":", 1)[0])
        ea = total * ea_percent // 100
        model = total - ea
        if ea <= 0:
            raise ValueError("FE split leaves no evaluations for the EA stage")
        return total, ea, model

    def resolve_population_size(self, n_dimensions: int) -> int:
        """Return the explicit comparison population or the legacy fallback."""
        if n_dimensions <= 0:
            raise ValueError("n_dimensions must be positive")
        if self.population_size is not None:
            return self.population_size
        return self.population_size_per_dimension * n_dimensions

    def to_dict(self) -> dict:
        return asdict(self)


# Edit this object to define the default experiment.
DEFAULT_CONFIG = ExperimentConfig(
    algorithms=("NSGAII",),
    problems=("dtlz7",),
    n_objectives=3,
    runs=20,
    seed=1,
    population_size=COMPARISON_POPULATION_SIZE,
    population_size_per_dimension=100,
    max_fe=COMPARISON_TOTAL_FE_BUDGET,
    max_fe_per_dimension=10000,
    algorithm_parameters=(),
    platemo_root=os.environ.get("PLATEMO_ROOT", ""),
    candidate_pool_size=50000,
    ea_fill_split="90:10",
    ea_only=False,
    gap_threshold_factor=2.5,
    hole_support_neighbors=32,
    hole_minimum_support=0.20,
    hole_extreme_gap_quantile=0.95,
    hole_pilot_fraction=0.2,
    hole_success_neighbors=16,
    hole_local_perturbation_scale=1.0,
    hole_boundary_probe_fraction=0.10,
    hole_boundary_probe_support_floor=0.05,
    hole_boundary_probe_branches=3,
    exhaust_model_fe_budget=True,
    direction_match_factor=2.5,
    local_front_neighbors=20,
    local_front_error_quantile=0.99,
    local_front_min_neighbors=8,
    local_front_ridge=1e-8,
    local_front_calibration_samples=4096,
    local_refinement_enabled=True,
    local_refinement_candidate_neighbors=64,
    local_refinement_branch_neighbors=32,
    local_refinement_ridge=1e-6,
    hidden_width=1024,
    pretrain_epochs=200,
    minimum_training_epochs=20,
    validation_fraction=0.2,
    early_stopping_patience=20,
    early_stopping_min_delta=1e-6,
    batch_size=4096,
    fine_tune_epochs=15,
    fine_tune_batch_size=1024,
    fine_tune_learning_rate=0.00025,
    model_inference_batch_size=4096,
    learning_rate=0.0025,
    weight_decay=0.01,
    optimizer="adamw",
    warmup_steps=10,
    device="cuda",
    hypervolume_reference_value=1.1,
    hypervolume_monte_carlo_samples=131072,
    hypervolume_seed=2026,
    igd_infinity_reference_samples=IGD_INFINITY_REFERENCE_SAMPLES,
    feasibility_tolerance=1e-12,
    consistency_sample_size=32,
    consistency_relative_tolerance=1e-4,
    consistency_absolute_tolerance=1e-7,
    plot=False,
    output_dir="results",
)


if __name__ == "__main__":
    config = DEFAULT_CONFIG.validate()
    print("Configuration is valid.")
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=True))
