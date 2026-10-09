import unittest

import numpy as np

from gd_psl.metrics import hypervolume


def brute_force_3d(points: np.ndarray, reference: np.ndarray) -> float:
    axes = [np.unique(np.r_[points[:, axis], reference[axis]]) for axis in range(3)]
    volume = 0.0
    for i in range(len(axes[0]) - 1):
        for j in range(len(axes[1]) - 1):
            for k in range(len(axes[2]) - 1):
                lower = np.array([axes[0][i], axes[1][j], axes[2][k]])
                if np.any(np.all(points <= lower, axis=1)):
                    volume += np.prod(
                        [
                            axes[0][i + 1] - axes[0][i],
                            axes[1][j + 1] - axes[1][j],
                            axes[2][k + 1] - axes[2][k],
                        ]
                    )
    return float(volume)


class HypervolumeTest(unittest.TestCase):
    def test_exact_3d_matches_cell_decomposition(self) -> None:
        reference = np.array([1.1, 1.1, 1.1])
        for seed in range(10):
            points = np.random.default_rng(seed).uniform(0.0, 1.0, size=(12, 3))
            expected = brute_force_3d(points, reference)
            actual, method = hypervolume(points, reference)
            self.assertEqual(method, "exact_recursive")
            self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
