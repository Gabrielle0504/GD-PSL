"""Pareto-set model training and true objective evaluation."""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiment_config import ExperimentConfig
from pareto_set_model import ParetoSetModel
from .local_refinement import refine_model_decisions

try:
    import schedulefree
except ImportError:
    schedulefree = None


class PreferenceDataset(Dataset):
    def __init__(self, preferences: torch.Tensor, decisions: torch.Tensor):
        self.preferences = preferences
        self.decisions = decisions

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.preferences[index], self.decisions[index]


def synchronize(device: torch.device | str) -> None:
    value = torch.device(device)
    if value.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(value)


def train_model(
    decisions: np.ndarray,
    preferences: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_objectives: int,
    config: ExperimentConfig,
    training_seed: Optional[int] = None,
) -> tuple[ParetoSetModel, object, dict]:
    """Fit preference-to-decision regression with deterministic validation."""
    synchronize(config.device)
    started = time.perf_counter()
    device = torch.device(config.device)
    seed = config.seed if training_seed is None else training_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ranges = upper_bounds - lower_bounds
    if np.any(ranges <= 0):
        raise ValueError("Every decision variable must have a positive range")
    normalized_x = np.clip((decisions - lower_bounds) / ranges, 0.0, 1.0)
    x = torch.as_tensor(normalized_x, dtype=torch.float32, device=device)
    preference = torch.as_tensor(preferences, dtype=torch.float32, device=device)
    if not len(x):
        raise ValueError("At least one archive solution is required")

    permutation = np.random.default_rng(seed).permutation(len(x))
    if len(permutation) == 1:
        train_indices = validation_indices = permutation
        split_strategy = "shared_single_sample"
    else:
        validation_count = min(
            len(permutation) - 1,
            max(1, round(len(permutation) * config.validation_fraction)),
        )
        validation_indices = permutation[:validation_count]
        train_indices = permutation[validation_count:]
        split_strategy = "deterministic_holdout"

    train_index = torch.as_tensor(train_indices, device=device)
    validation_index = torch.as_tensor(validation_indices, device=device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PreferenceDataset(preference[train_index], x[train_index]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = x[validation_index]
    validation_preferences = preference[validation_index]

    model = ParetoSetModel(
        decisions.shape[1], n_objectives, hidden_width=config.hidden_width
    ).to(device)
    use_schedulefree = config.optimizer == "schedulefree_adamw" or (
        config.optimizer == "auto" and schedulefree is not None
    )
    resolved_optimizer = "schedulefree_adamw" if use_schedulefree else "adamw"
    if use_schedulefree:
        if schedulefree is None:
            raise RuntimeError("schedulefree_adamw requires the schedulefree package")
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    history = {"epoch": [], "train_loss": [], "validation_loss": []}
    best_loss, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    early_stopped = False
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()
        squared_error, element_count = 0.0, 0
        for batch_preferences, batch_x in loader:
            prediction = model(batch_preferences)
            loss = torch.mean((prediction - batch_x) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach().cpu()) * batch_x.numel()
            element_count += batch_x.numel()

        if hasattr(optimizer, "eval"):
            optimizer.eval()
        model.eval()
        with torch.no_grad():
            validation_loss = torch.mean(
                (model(validation_preferences) - validation_x) ** 2
            )
        train_loss = squared_error / element_count
        validation_value = float(validation_loss.detach().cpu())
        if not np.isfinite(train_loss) or not np.isfinite(validation_value):
            raise RuntimeError(f"Non-finite model loss at epoch {epoch}")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_value)

        if best_loss - validation_value > config.early_stopping_min_delta:
            best_loss, best_epoch, stale_epochs = validation_value, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if epoch >= config.minimum_training_epochs and stale_epochs >= config.early_stopping_patience:
            early_stopped = True
            break

    synchronize(device)
    model.load_state_dict(best_state)

    fine_tune_completed = 0
    if config.fine_tune_epochs:
        fine_generator = torch.Generator().manual_seed(seed + 1)
        fine_loader = DataLoader(
            PreferenceDataset(preference[train_index], x[train_index]),
            batch_size=config.fine_tune_batch_size,
            shuffle=True,
            generator=fine_generator,
        )
        fine_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.fine_tune_learning_rate,
            weight_decay=config.weight_decay,
        )
        for offset in range(1, config.fine_tune_epochs + 1):
            model.train()
            squared_error, element_count = 0.0, 0
            for batch_preferences, batch_x in fine_loader:
                prediction = model(batch_preferences)
                loss = torch.mean((prediction - batch_x) ** 2)
                fine_optimizer.zero_grad()
                loss.backward()
                fine_optimizer.step()
                squared_error += float(loss.detach().cpu()) * batch_x.numel()
                element_count += batch_x.numel()

            model.eval()
            with torch.no_grad():
                validation_value = float(
                    torch.mean((model(validation_preferences) - validation_x) ** 2)
                    .detach()
                    .cpu()
                )
            train_loss = squared_error / element_count
            if not np.isfinite(train_loss) or not np.isfinite(validation_value):
                raise RuntimeError(f"Non-finite model loss during fine-tune epoch {offset}")
            epoch = len(history["epoch"]) + 1
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_value)
            fine_tune_completed += 1
            if best_loss - validation_value > config.early_stopping_min_delta:
                best_loss, best_epoch = validation_value, epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    synchronize(device)
    model.load_state_dict(best_state)
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    model.eval()
    best_train_loss = history["train_loss"][best_epoch - 1]
    history.update(
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "train_loss_at_best_epoch": best_train_loss,
            "generalization_gap_at_best_epoch": best_loss - best_train_loss,
            "epochs_trained": len(history["epoch"]),
            "main_training_epochs": len(history["epoch"]) - fine_tune_completed,
            "fine_tune_epochs_completed": fine_tune_completed,
            "fine_tune_batch_size": config.fine_tune_batch_size,
            "fine_tune_learning_rate": config.fine_tune_learning_rate,
            "early_stopped": early_stopped,
            "converged": early_stopped,
            "stop_reason": "validation_plateau" if early_stopped else "maximum_epochs_reached",
            "convergence_rule": "validation patience with a fixed minimum improvement",
            "interpretation": "A validation plateau does not prove a global optimum or PF completion",
            "training_time_seconds": time.perf_counter() - started,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "split_strategy": split_strategy,
            "split_seed": seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer_requested": config.optimizer,
            "optimizer_resolved": resolved_optimizer,
            "fine_tune_optimizer_resolved": "adamw" if fine_tune_completed else None,
        }
    )
    return model, optimizer, history


def evaluate_model(
    model: ParetoSetModel,
    preferences: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    problem,
    device: str,
    batch_size: int = 4096,
    archive_decisions: Optional[np.ndarray] = None,
    archive_preferences: Optional[np.ndarray] = None,
    refinement_enabled: bool = True,
    refinement_candidate_neighbors: int = 64,
    refinement_branch_neighbors: int = 32,
    refinement_ridge: float = 1e-6,
    refinement_branch_ranks: Optional[np.ndarray] = None,
    refinement_maximum_branches: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate decisions and evaluate them with the true Python problem."""
    if not len(preferences):
        return (
            np.empty((0, len(lower_bounds))),
            np.empty((0, problem.n_obj)),
            {
                "enabled": bool(refinement_enabled),
                "strategy": "not_applied_no_queries",
                "seconds": 0.0,
                "reference_pf_used": False,
                "additional_true_evaluations": 0,
            },
        )
    lower = torch.as_tensor(lower_bounds, dtype=torch.float64, device=device)
    upper = torch.as_tensor(upper_bounds, dtype=torch.float64, device=device)
    decision_batches = []
    with torch.no_grad():
        for start in range(0, len(preferences), batch_size):
            target = torch.as_tensor(
                preferences[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            decisions = torch.clamp(model(target) * (upper - lower) + lower, lower, upper)
            decision_batches.append(decisions.cpu().numpy())
    raw_decisions = np.vstack(decision_batches)
    if refinement_enabled:
        if archive_decisions is None or archive_preferences is None:
            raise ValueError("Local refinement requires the evaluated EA archive")
        decisions, refinement = refine_model_decisions(
            raw_decisions,
            preferences,
            archive_decisions,
            archive_preferences,
            lower_bounds,
            upper_bounds,
            refinement_candidate_neighbors,
            refinement_branch_neighbors,
            refinement_ridge,
            refinement_branch_ranks,
            refinement_maximum_branches,
        )
    else:
        decisions = raw_decisions
        refinement = {
            "enabled": False,
            "strategy": "disabled",
            "seconds": 0.0,
            "reference_pf_used": False,
            "additional_true_evaluations": 0,
        }

    objective_batches = []
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            batch = torch.as_tensor(
                decisions[start : start + batch_size],
                dtype=torch.float64,
                device=device,
            )
            objective_batches.append(problem.evaluate(batch).detach().cpu().numpy())
    return decisions, np.vstack(objective_batches), refinement
