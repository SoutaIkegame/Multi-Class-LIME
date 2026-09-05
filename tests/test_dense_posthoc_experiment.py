import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from src.run_dense_posthoc_experiment import (
    _fit_soft_logistic_calibration,
    _regularized_fisher_direction,
    choose_smallest_acceptable_budget,
    choose_energy_budget,
    choose_best_validation_budget,
    contribution_energy,
    direction_cosine,
    effective_feature_count,
    fit_dense_methods,
    predict_with_displayed_features,
    top_k_features,
)


class DensePosthocTests(unittest.TestCase):
    def test_top_k_and_energy_use_dense_direction(self):
        direction = np.array([1.0, -3.0, 2.0, 0.5])
        selected = top_k_features(direction, 2)
        self.assertEqual(selected, frozenset({1, 2}))
        self.assertAlmostEqual(contribution_energy(direction, selected), 13.0 / 14.25)
        self.assertAlmostEqual(direction_cosine(direction, direction), 1.0)
        self.assertGreater(effective_feature_count(direction), 1.0)

    def test_hiding_features_preserves_prediction_at_local_origin(self):
        fit = {
            "method": "ovo_probability",
            "coef": np.array([0.2, -0.3]),
            "intercept": 0.6,
            "direction": np.array([0.2, -0.3]),
        }
        H = np.zeros((1, 2))
        dense = predict_with_displayed_features(fit, H, frozenset({0, 1}))
        sparse = predict_with_displayed_features(fit, H, frozenset({1}))
        np.testing.assert_allclose(dense, sparse)

    def test_soft_logistic_calibration_recovers_monotone_probability(self):
        score = np.linspace(-2.0, 2.0, 101)
        q = 1.0 / (1.0 + np.exp(-(0.3 + 1.7 * score)))
        a, b = _fit_soft_logistic_calibration(score, q, np.ones_like(q))
        self.assertAlmostEqual(a, 0.3, places=3)
        self.assertAlmostEqual(b, 1.7, places=3)

    def test_pairwise_fisher_points_toward_positive_membership(self):
        H = np.array([[-2.0, 0.0], [-1.0, 0.1], [1.0, 0.0], [2.0, -0.1]])
        q = np.array([0.05, 0.1, 0.9, 0.95])
        direction = _regularized_fisher_direction(
            H, np.ones(4), q, 1.0 - q, shrinkage=1e-3
        )
        self.assertGreater(direction[0], 0.0)

    def test_unregularized_pairwise_soft_fisher_is_collinear_with_weighted_ols(self):
        rng = np.random.default_rng(12)
        H = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
        q = 1.0 / (1.0 + np.exp(-(H @ rng.normal(size=5))))
        weights = rng.uniform(0.1, 1.0, size=H.shape[0])
        ols = LinearRegression().fit(H, q, sample_weight=weights).coef_
        fisher = _regularized_fisher_direction(
            H, weights, q, 1.0 - q, shrinkage=0.0
        )
        self.assertAlmostEqual(direction_cosine(ols, fisher), 1.0, places=10)

    def test_all_dense_methods_return_finite_predictions(self):
        rng = np.random.default_rng(0)
        H = rng.normal(size=(80, 4))
        logits = np.column_stack([H[:, 0], -H[:, 0] + 0.3 * H[:, 1], 0.2 * H[:, 2]])
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        fits = fit_dense_methods(H, np.ones(80), proba, 0, 1)
        self.assertEqual(set(fits), {
            "ovr_probability", "ovo_probability",
            "multiclass_soft_fisher", "pairwise_soft_fisher",
        })
        for fit in fits.values():
            pred = predict_with_displayed_features(fit, H, top_k_features(fit["direction"], 2))
            self.assertTrue(np.all(np.isfinite(pred)))
            self.assertTrue(np.all((pred >= 0.0) & (pred <= 1.0)))

    def test_adaptive_budget_uses_smallest_acceptable_k(self):
        H = np.column_stack([np.linspace(-1.0, 1.0, 100), np.zeros(100), np.zeros(100)])
        q = np.clip(0.5 + 0.2 * H[:, 0], 0.0, 1.0)
        fit = {
            "method": "ovo_probability",
            "coef": np.array([0.2, 0.1, 0.05]),
            "intercept": 0.5,
            "direction": np.array([0.2, 0.1, 0.05]),
        }
        chosen = choose_smallest_acceptable_budget(
            fit, H, q, np.ones(100), [1, 2, None],
            r2_tolerance=0.0, agreement_tolerance=0.0,
        )
        self.assertEqual(chosen, 1)

    def test_energy_budget_uses_smallest_k_passing_threshold(self):
        direction = np.array([4.0, 2.0, 1.0, 0.0])
        self.assertEqual(choose_energy_budget(direction, 0.75), 1)
        self.assertEqual(choose_energy_budget(direction, 0.95), 2)
        self.assertEqual(choose_energy_budget(direction, 1.0), 3)

    def test_best_validation_budget_can_keep_dense_when_top_one_is_harmful(self):
        H = np.column_stack([
            np.linspace(-1.0, 1.0, 100),
            np.linspace(-1.0, 1.0, 100) ** 2,
        ])
        q = np.clip(0.5 + 0.2 * H[:, 0], 0.0, 1.0)
        fit = {
            "method": "ovo_probability",
            "coef": np.array([0.2, 0.4]),
            "intercept": 0.5,
            "direction": np.array([0.2, 0.4]),
        }
        self.assertIsNone(
            choose_best_validation_budget(fit, H, q, np.ones(H.shape[0]))
        )


if __name__ == "__main__":
    unittest.main()
