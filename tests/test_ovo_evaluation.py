import unittest

import numpy as np
from scipy.special import expit

from src.run_ovo_evaluation import (
    _pair_targets,
    mean_pairwise_cosine,
    mean_pairwise_jaccard,
    weighted_agreement,
    weighted_mse,
    weighted_r2,
)
from src.benchmark_ovo_runtime import make_local_problem


class PairwiseTargetTests(unittest.TestCase):
    def test_probability_and_logratio_are_consistent(self):
        proba = np.array([[0.6, 0.3, 0.1], [0.0, 0.8, 0.2]])
        q, ell = _pair_targets(proba, 0, 1)
        np.testing.assert_allclose(expit(ell), q)

    def test_reversing_pair_reverses_target(self):
        proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        q_cd, ell_cd = _pair_targets(proba, 0, 1)
        q_dc, ell_dc = _pair_targets(proba, 1, 0)
        np.testing.assert_allclose(q_dc, 1.0 - q_cd)
        np.testing.assert_allclose(ell_dc, -ell_cd)


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0.1, 0.6, 0.9])
        weights = np.array([1.0, 2.0, 1.0])
        self.assertAlmostEqual(weighted_mse(y, y, weights), 0.0)
        self.assertAlmostEqual(weighted_r2(y, y, weights), 1.0)
        self.assertAlmostEqual(weighted_agreement(y, y, weights), 1.0)

    def test_stability_metrics(self):
        vectors = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        sets = [frozenset({0, 1}), frozenset({0, 1})]
        self.assertAlmostEqual(mean_pairwise_cosine(vectors), 1.0)
        self.assertAlmostEqual(mean_pairwise_jaccard(sets), 1.0)


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_generated_problem_shapes_and_probabilities(self):
        Z, weights, proba, feature_std = make_local_problem(
            n_classes=4,
            n_perturbations=30,
            n_features=6,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(Z.shape, (30, 6))
        self.assertEqual(weights.shape, (30,))
        self.assertEqual(proba.shape, (30, 4))
        self.assertEqual(feature_std.shape, (6,))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
