import unittest

import numpy as np

from evaluation_metrics import (
    archive_igd_infinity_report,
    load_benchmark_reference_front,
    pang_projected_igd_infinity,
    pang_projection_references,
)
from run_pang_archive_baseline import default_pang_population_size, parse_args


class PangProtocolTests(unittest.TestCase):
    def test_re_reference_front_is_loaded_from_project_data(self) -> None:
        reference, source = load_benchmark_reference_front("re37", 3)

        self.assertEqual(reference.shape, (1500, 3))
        self.assertEqual(source, "benchmark_file:data/RE/ParetoFront/RE37.dat")

    def test_re_igd_uses_fixed_normalized_reference_front(self) -> None:
        reference = np.array([[10.0, 100.0], [20.0, 50.0], [30.0, 10.0]])
        report = archive_igd_infinity_report(
            reference,
            reference,
            "re_test",
            3,
            np.array([10.0, 10.0]),
            np.array([30.0, 100.0]),
            reference,
        )

        self.assertEqual(report["completed_igd_infinity"], 0.0)
        self.assertEqual(
            report["distance_space"],
            "fixed_ideal_nadir_normalized_objective_space",
        )

    def test_project_formal_run_count_and_author_population_defaults(self) -> None:
        arguments = parse_args(["--platemo-root", "PlatEMO"])

        self.assertEqual(arguments.runs, 20)
        self.assertIsNone(arguments.population_size)
        self.assertEqual(default_pang_population_size(3), 91)
        self.assertEqual(default_pang_population_size(5), 210)

    def test_dtlz2_author_reference_matches_supplied_sample_count(self) -> None:
        references = pang_projection_references("dtlz2", 3, 50_000)

        self.assertEqual(set(references), {(0, 1), (0, 2), (1, 2)})
        self.assertTrue(
            all(values.shape == (49_770, 2) for values in references.values())
        )
        self.assertTrue(np.allclose(references[(0, 1)], references[(1, 2)]))

    def test_dtlz7_author_references_distinguish_last_objective(self) -> None:
        references = pang_projection_references("dtlz7", 5, 50_000)

        self.assertEqual(len(references), 10)
        self.assertTrue(
            all(values.shape == (49_729, 2) for values in references.values())
        )
        self.assertTrue(np.all((references[(0, 1)] >= 0) & (references[(0, 1)] <= 1)))
        self.assertTrue(np.all((references[(0, 4)] >= 0) & (references[(0, 4)] <= 1)))
        self.assertFalse(np.allclose(references[(0, 1)], references[(0, 4)]))

    def test_pang_igd_normalizes_with_true_front_extrema(self) -> None:
        true_front = np.array(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 4.0],
                [0.0, 1.0, 6.0],
                [1.0, 1.0, 5.0],
            ]
        )
        archive = true_front.copy()

        value, pair_scores, reference_count = pang_projected_igd_infinity(
            archive, "dtlz7", true_front, 100
        )

        self.assertTrue(np.isfinite(value))
        self.assertEqual(len(pair_scores), 3)
        self.assertEqual(reference_count, 100)


if __name__ == "__main__":
    unittest.main()
