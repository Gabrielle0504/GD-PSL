from __future__ import annotations

import time
import unittest
from dataclasses import replace

import numpy as np

from gd_psl.config import ALLOWED_EA_FILL_SPLITS, DEFAULT_CONFIG
from gd_psl.archive import classify_model_candidates
from gd_psl.candidate_validation import (
    _inside_empirical_hull,
    validate_empirical_candidates,
)
from gd_psl.holes import (
    adapt_gap_preferences,
    boundary_probe_capacity,
    preference_spacing,
    select_boundary_probe_preferences,
    select_gap_preferences,
)
from gd_psl.local_refinement import (
    interpolate_archive_decisions,
    refine_model_decisions,
)
from gd_psl.runner import _merge_completion_archive


class BudgetSplitTests(unittest.TestCase):
    def test_only_registered_splits_resolve_to_expected_budgets(self) -> None:
        expected = {
            "50:50": (100_000, 50_000, 50_000),
            "60:40": (100_000, 60_000, 40_000),
            "70:30": (100_000, 70_000, 30_000),
            "80:20": (100_000, 80_000, 20_000),
            "90:10": (100_000, 90_000, 10_000),
        }
        self.assertEqual(set(ALLOWED_EA_FILL_SPLITS), set(expected))
        for split, budgets in expected.items():
            config = replace(DEFAULT_CONFIG, ea_fill_split=split).validate()
            self.assertEqual(config.resolve_evaluation_budgets(12), budgets)

    def test_unregistered_split_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(DEFAULT_CONFIG, ea_fill_split="75:25").validate()


class ArchiveAdmissionPolicyTests(unittest.TestCase):
    def test_validation_diagnostics_do_not_remove_nondominated_evaluations(self) -> None:
        base_x = np.array([[0.0], [1.0]])
        base_f = np.array([[0.0, 1.0], [1.0, 0.0]])
        model_x = np.array([[0.5], [0.8]])
        model_f = np.array([[0.5, 0.5], [1.2, 1.2]])
        validation = {
            "eligible": np.array([False, True]),
            "statuses": np.array(
                ["model_off_empirical_front", "model_candidate_pending"],
                dtype=object,
            ),
        }

        _, completed_f, sources, archive_statuses, diagnostics, retained = (
            _merge_completion_archive(
                base_x, base_f, model_x, model_f, validation
            )
        )

        self.assertTrue(np.any(np.all(completed_f == [0.5, 0.5], axis=1)))
        self.assertEqual(int(np.count_nonzero(sources == 1)), 1)
        self.assertEqual(archive_statuses.tolist(), ["model_nondominated", "model_dominated"])
        self.assertEqual(
            diagnostics.tolist(),
            ["model_off_empirical_front", "model_dominated"],
        )
        self.assertEqual(retained.tolist(), [True, False])


class LocalRefinementTests(unittest.TestCase):
    def test_interpolation_baseline_does_not_require_model_output(self) -> None:
        first = np.linspace(0.0, 1.0, 41)
        preferences = np.column_stack((first, 1.0 - first))
        decisions = np.column_stack((first, first**2))
        targets = np.array([[0.35, 0.65], [0.70, 0.30]])
        interpolated, diagnostics = interpolate_archive_decisions(
            targets,
            decisions,
            preferences,
            np.zeros(2),
            np.ones(2),
            neighbors=12,
        )
        self.assertTrue(np.allclose(interpolated[:, 0], targets[:, 0], atol=1e-2))
        self.assertEqual(diagnostics["additional_true_evaluations"], 0)
        self.assertFalse(diagnostics["reference_pf_used"])

    def test_refinement_uses_model_output_to_select_a_decision_branch(self) -> None:
        first = np.linspace(0.0, 1.0, 41)
        preferences = np.column_stack((first, 1.0 - first))
        archive_preferences = np.vstack((preferences, preferences))
        lower_branch = np.column_stack((first, np.full_like(first, 0.1)))
        upper_branch = np.column_stack((first, np.full_like(first, 0.9)))
        archive_decisions = np.vstack((lower_branch, upper_branch))
        targets = np.array([[0.37, 0.63], [0.62, 0.38]])
        raw = np.array([[0.20, 0.08], [0.80, 0.92]])

        refined, diagnostics = refine_model_decisions(
            raw,
            targets,
            archive_decisions,
            archive_preferences,
            np.zeros(2),
            np.ones(2),
            candidate_neighbors=32,
            branch_neighbors=12,
        )

        self.assertAlmostEqual(refined[0, 0], 0.37, places=2)
        self.assertLess(refined[0, 1], 0.2)
        self.assertAlmostEqual(refined[1, 0], 0.62, places=2)
        self.assertGreater(refined[1, 1], 0.8)
        self.assertEqual(diagnostics["additional_true_evaluations"], 0)
        self.assertFalse(diagnostics["reference_pf_used"])

    def test_refinement_respects_problem_bounds(self) -> None:
        preferences = np.column_stack(
            (np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20))
        )
        archive = np.column_stack(
            (np.linspace(-2.0, 2.0, 20), np.linspace(3.0, 7.0, 20))
        )
        refined, _ = refine_model_decisions(
            np.array([[2.0, 3.0]]),
            np.array([[1.2, -0.2]]),
            archive,
            preferences,
            np.array([-2.0, 3.0]),
            np.array([2.0, 7.0]),
            candidate_neighbors=12,
            branch_neighbors=8,
        )
        self.assertTrue(np.all(refined >= np.array([-2.0, 3.0])))
        self.assertTrue(np.all(refined <= np.array([2.0, 7.0])))


class CandidateValidationTests(unittest.TestCase):
    def test_direction_miss_does_not_change_archive_dominance(self) -> None:
        base_x = np.array([[0.0], [1.0]])
        base_f = np.array([[0.0, 1.0], [1.0, 0.0]])
        candidate_x = np.array([[0.5]])
        candidate_f = np.array([[0.4, 0.4]])
        archive_status = classify_model_candidates(
            base_x, base_f, candidate_x, candidate_f
        )
        self.assertEqual(archive_status[0], "model_nondominated")

    def setUp(self) -> None:
        angles = np.linspace(0.0, np.pi / 2.0, 101)
        self.base_f = np.column_stack((np.cos(angles), np.sin(angles)))
        self.ideal = self.base_f.min(axis=0)
        self.nadir = self.base_f.max(axis=0)
        normalized = (self.base_f - self.ideal) / (self.nadir - self.ideal)
        self.base_preferences = normalized / normalized.sum(axis=1, keepdims=True)
        self.spacing = preference_spacing(self.base_preferences)

    def _validate(self, targets: np.ndarray, objectives: np.ndarray) -> dict:
        return validate_empirical_candidates(
            self.base_f,
            targets,
            np.zeros((len(objectives), 1)),
            objectives,
            self.ideal,
            self.nadir,
            self.spacing,
            direction_match_factor=2.5,
            local_neighbors=20,
            error_quantile=0.99,
            minimum_neighbors=8,
            ridge=1e-8,
            calibration_samples=4096,
        )

    def test_on_front_candidate_in_requested_direction_is_eligible(self) -> None:
        value = np.sqrt(0.5)
        result = self._validate(np.array([[0.5, 0.5]]), np.array([[value, value]]))
        self.assertTrue(result["eligible"][0])

    def test_candidate_that_misses_requested_direction_is_rejected(self) -> None:
        value = np.sqrt(0.5)
        result = self._validate(np.array([[0.8, 0.2]]), np.array([[value, value]]))
        self.assertEqual(result["statuses"][0], "model_direction_mismatch")

    def test_candidate_that_reduces_a_hole_passes_direction_gate(self) -> None:
        angles = np.concatenate(
            (np.linspace(0.0, 0.62, 80), np.linspace(0.95, np.pi / 2.0, 80))
        )
        base_f = np.column_stack((np.cos(angles), np.sin(angles)))
        ideal, nadir = base_f.min(axis=0), base_f.max(axis=0)
        normalized = (base_f - ideal) / (nadir - ideal)
        base_preferences = normalized / normalized.sum(axis=1, keepdims=True)
        target = np.array([[0.5, 0.5]])
        candidate_angle = 0.70
        candidate = np.array(
            [[np.cos(candidate_angle), np.sin(candidate_angle)]]
        )
        result = validate_empirical_candidates(
            base_f,
            target,
            np.zeros((1, 1)),
            candidate,
            ideal,
            nadir,
            preference_spacing(base_preferences),
            direction_match_factor=2.5,
            local_neighbors=20,
            error_quantile=0.99,
            minimum_neighbors=8,
            ridge=1e-8,
            calibration_samples=4096,
        )
        self.assertGreater(result["coverage_improvement"][0], 0.0)
        self.assertNotEqual(result["statuses"][0], "model_direction_mismatch")

    def test_three_objective_hull_distinguishes_interpolation_from_extrapolation(self) -> None:
        base = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ]
        )
        queries = np.array(
            [
                [0.25, 0.25, 0.5],
                [1.05, 0.0, -0.05],
            ]
        )
        inside = _inside_empirical_hull(base, queries)
        self.assertTrue(inside[0])
        self.assertFalse(inside[1])

    def test_candidate_behind_empirical_front_is_rejected(self) -> None:
        value = 1.5 * np.sqrt(0.5)
        result = self._validate(np.array([[0.5, 0.5]]), np.array([[value, value]]))
        self.assertEqual(result["statuses"][0], "model_off_empirical_front")

    def test_genuinely_better_radius_is_not_rejected(self) -> None:
        value = 0.8 * np.sqrt(0.5)
        result = self._validate(np.array([[0.5, 0.5]]), np.array([[value, value]]))
        self.assertTrue(result["eligible"][0])

    def test_validation_never_uses_a_reference_front(self) -> None:
        value = np.sqrt(0.5)
        result = self._validate(np.array([[0.5, 0.5]]), np.array([[value, value]]))
        self.assertFalse(result["reference_pf_used"])

    def test_discontinuous_local_branches_are_not_bridged(self) -> None:
        left = np.linspace(0.1, 0.49, 2500)
        right = np.linspace(0.51, 0.9, 2500)
        left_f = np.column_stack((left, 1.0 - left))
        right_f = 1.8 * np.column_stack((right, 1.0 - right))
        base_f = np.vstack((left_f, right_f))
        target = np.array([[0.5, 0.5]])
        candidate = np.array([[0.75, 0.75]])
        result = validate_empirical_candidates(
            base_f,
            target,
            np.zeros((1, 1)),
            candidate,
            np.zeros(2),
            np.ones(2),
            preference_spacing(
                np.vstack(
                    (
                        np.column_stack((left, 1.0 - left)),
                        np.column_stack((right, 1.0 - right)),
                    )
                )
            ),
            direction_match_factor=2.5,
            local_neighbors=20,
            error_quantile=0.99,
            minimum_neighbors=8,
            ridge=1e-8,
            calibration_samples=4096,
        )
        self.assertEqual(result["statuses"][0], "model_unsupported_topology")

    def test_one_sided_extrapolation_beyond_observed_branch_is_rejected(self) -> None:
        first = np.linspace(0.2, 0.8, 301)
        base_f = np.column_stack((first, 1.0 - first))
        base_preferences = base_f / base_f.sum(axis=1, keepdims=True)
        result = validate_empirical_candidates(
            base_f,
            np.array([[0.9, 0.1]]),
            np.zeros((1, 1)),
            np.array([[0.9, 0.1]]),
            np.zeros(2),
            np.ones(2),
            preference_spacing(base_preferences),
            direction_match_factor=2.5,
            local_neighbors=20,
            error_quantile=0.99,
            minimum_neighbors=8,
            ridge=1e-8,
            calibration_samples=4096,
        )
        self.assertEqual(result["statuses"][0], "model_unsupported_topology")

    def test_calibration_work_is_bounded_for_many_candidates(self) -> None:
        rng = np.random.default_rng(19)
        base_angles = np.linspace(0.0, np.pi / 2.0, 5001)
        base_f = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        normalized_base = (base_f - self.ideal) / (self.nadir - self.ideal)
        base_preferences = normalized_base / normalized_base.sum(axis=1, keepdims=True)
        targets = rng.dirichlet(np.ones(2), size=10_000)
        angles = np.arctan2(targets[:, 1], targets[:, 0])
        objectives = np.column_stack((np.cos(angles), np.sin(angles)))
        started = time.perf_counter()
        result = validate_empirical_candidates(
            base_f,
            targets,
            np.zeros((len(objectives), 1)),
            objectives,
            self.ideal,
            self.nadir,
            preference_spacing(base_preferences),
            direction_match_factor=2.5,
            local_neighbors=20,
            error_quantile=0.99,
            minimum_neighbors=8,
            ridge=1e-8,
            calibration_samples=4096,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(result["calibration_samples"], 4096)
        self.assertEqual(len(result["statuses"]), 10_000)
        self.assertLess(elapsed, 10.0)


class HoleSelectionCostTests(unittest.TestCase):
    def test_detection_pool_expands_only_when_supported_capacity_is_short(self) -> None:
        first = np.linspace(0.2, 0.8, 101)
        observed = np.column_stack((first, 1.0 - first))
        selected, diagnostics, _ = select_gap_preferences(
            observed,
            max_samples=600,
            candidate_pool_size=1000,
            threshold_factor=2.5,
            rng=np.random.default_rng(29),
            return_details=True,
            exhaust_budget=True,
            minimum_support=0.80,
        )
        self.assertEqual(len(selected), 600)
        self.assertEqual(diagnostics["initial_candidate_count"], 1000)
        self.assertGreater(diagnostics["candidate_pool_expansion_count"], 0)
        self.assertGreater(diagnostics["candidate_pool_expansion_rounds"], 0)

    def test_boundary_probe_request_is_capped_to_available_candidates(self) -> None:
        candidates = np.column_stack(
            (np.linspace(0.0, 1.0, 10), np.linspace(1.0, 0.0, 10))
        )
        details = {
            "candidate_preferences": candidates,
            "selected_indices": np.array([0, 1, 2]),
            "initial_distances": np.full(10, 1.0),
            "local_spacing": np.full(10, 0.25),
            "support_scores": np.array(
                [0.8, 0.8, 0.8, 0.4, 0.3, 0.19, 0.15, 0.10, 0.05, 0.0]
            ),
            "minimum_support": 0.2,
            "gap_threshold": 0.5,
            "threshold_factor": 2.5,
        }
        capacity = boundary_probe_capacity(details)
        selected, indices = select_boundary_probe_preferences(
            details,
            count=20,
            support_floor=0.1,
            rng=np.random.default_rng(31),
        )
        self.assertEqual(capacity, 5)
        self.assertEqual(len(selected), capacity)
        self.assertEqual(len(indices), capacity)
        self.assertEqual(details["boundary_probe_available_count"], capacity)

    def test_unused_boundary_quota_can_be_returned_to_completion(self) -> None:
        total_queries = 20_000
        requested = 2_000
        available = 317
        actual = min(requested, available)
        completion = total_queries - actual
        self.assertEqual(completion + actual, total_queries)
        self.assertEqual(completion, 19_683)

    def test_complete_budget_path_selects_from_larger_detection_pool(self) -> None:
        rng = np.random.default_rng(7)
        observed = rng.dirichlet(np.ones(3), size=1000)
        started = time.perf_counter()
        selected, diagnostics, details = select_gap_preferences(
            observed,
            max_samples=10_000,
            candidate_pool_size=50_000,
            threshold_factor=2.5,
            rng=rng,
            return_details=True,
            exhaust_budget=True,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(selected.shape, (10_000, 3))
        self.assertEqual(len(details["selected_indices"]), 10_000)
        self.assertEqual(diagnostics["candidate_count"], 50_000)
        self.assertEqual(
            diagnostics["selection_strategy"],
            "support_aware_gap_sampling",
        )
        self.assertIn("support_scores", details)
        self.assertIn("selection_scores", details)
        selected_support = details["support_scores"][details["selected_indices"]]
        self.assertTrue(np.all(selected_support >= 0.20))
        self.assertEqual(
            diagnostics["supported_candidate_count"],
            int(np.count_nonzero(details["support_scores"] >= 0.20)),
        )
        self.assertGreaterEqual(diagnostics["excluded_extreme_unsupported"], 0)
        self.assertLess(elapsed, 5.0)

    def test_followup_budget_moves_toward_successful_pilot_region(self) -> None:
        candidates = np.column_stack(
            (
                np.linspace(0.0, 1.0, 200),
                np.linspace(1.0, 0.0, 200),
            )
        )
        details = {
            "candidate_preferences": candidates,
            "selected_indices": np.array([20, 180, 60, 140]),
            "initial_distances": np.ones(len(candidates)),
            "selection_scores": np.ones(len(candidates)),
            "gap_threshold": 0.0,
        }
        selected, updated = adapt_gap_preferences(
            details,
            pilot_count=2,
            pilot_success=np.array([True, False]),
            total_samples=80,
            success_neighbors=1,
            rng=np.random.default_rng(11),
        )
        followup = selected[2:, 0]
        self.assertEqual(selected.shape, (80, 2))
        self.assertEqual(updated["pilot_success_count"], 1)
        success_distance = np.abs(followup - candidates[20, 0])
        failure_distance = np.abs(followup - candidates[180, 0])
        self.assertTrue(np.all(success_distance <= failure_distance))

    def test_local_followup_exhausts_budget_without_reusing_failed_regions(self) -> None:
        candidates = np.column_stack(
            (
                np.linspace(0.0, 1.0, 120),
                np.linspace(1.0, 0.0, 120),
            )
        )
        pilot_indices = np.arange(10, 110, 5)
        success = np.zeros(len(pilot_indices), dtype=bool)
        success[len(success) // 2] = True
        details = {
            "candidate_preferences": candidates,
            "observed_preferences": candidates[::2],
            "selected_indices": pilot_indices,
            "initial_distances": np.full(len(candidates), 0.01),
            "final_distances": np.full(len(candidates), 0.01),
            "support_scores": np.ones(len(candidates)),
            "local_spacing": np.full(len(candidates), 0.01),
            "selection_scores": np.ones(len(candidates)),
            "eligible_candidate_mask": np.ones(len(candidates), dtype=bool),
            "gap_threshold": 0.0,
            "threshold_factor": 0.0,
            "minimum_support": 0.2,
            "support_neighbors": 8,
        }
        selected, updated = adapt_gap_preferences(
            details,
            pilot_count=len(pilot_indices),
            pilot_success=success,
            total_samples=100,
            success_neighbors=4,
            rng=np.random.default_rng(23),
        )
        self.assertEqual(selected.shape, (100, 2))
        self.assertEqual(updated["followup_count"], 80)
        self.assertGreater(updated["generated_local_followup_count"], 0)
        self.assertEqual(updated["adaptive_fallback"], "none")


if __name__ == "__main__":
    unittest.main()
