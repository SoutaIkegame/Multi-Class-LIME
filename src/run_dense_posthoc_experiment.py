"""Evaluate dense fitting followed by post-hoc feature display.

This experiment implements the practical proposal:

1. fit every surrogate once with all features;
2. rank features from the dense pairwise direction;
3. display only the top-k terms without refitting the surrogate;
4. choose the smallest acceptable k on a validation neighborhood; and
5. report fidelity on a separate held-out neighborhood.

All models operate on the standardized local coordinates
``h(z) = (z - x) / feature_std``.  Consequently, absolute coefficients are
already comparable as one-standard-deviation local contributions, and
removing a feature preserves the surrogate prediction exactly at ``x``.

Compared methods
----------------
``ovr_probability``
    Dense Ridge fits for p_c and p_d; the same displayed feature set is
    retained in both fits before forming q_cd.
``ovo_probability``
    One dense Ridge fit directly to q_cd = p_c / (p_c + p_d).
``multiclass_soft_fisher``
    The existing all-class soft Fisher fit with one pooled within-class
    scatter matrix; extract the c-vs-d direction and calibrate its scalar
    score to q_cd.
``pairwise_soft_fisher``
    A pair-specific soft Fisher direction using q_cd and 1-q_cd as soft
    memberships, followed by the same scalar probability calibration.

Usage
-----
    python3 src/run_dense_posthoc_experiment.py --quick
    python3 src/run_dense_posthoc_experiment.py
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from run_ovo_evaluation import (  # noqa: E402
    _pair_targets,
    mean_pairwise_cosine,
    mean_pairwise_jaccard,
    weighted_agreement,
    weighted_mse,
    weighted_r2,
)
from surrogates import fit_fisher_soft  # noqa: E402


METHODS = (
    "ovr_probability",
    "ovo_probability",
    "multiclass_soft_fisher",
    "pairwise_soft_fisher",
)


def _standardize_local(Z: np.ndarray, x: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    """Return local coordinates whose origin is the explained instance."""
    return (Z - x[None, :]) / feature_std[None, :]


def _fit_ridge(H: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    model = Ridge(alpha=alpha)
    model.fit(H, y, sample_weight=weights)
    return model.coef_.copy(), float(model.intercept_)


def _fit_soft_logistic_calibration(
    score: np.ndarray,
    q: np.ndarray,
    weights: np.ndarray,
    l2: float = 1e-6,
) -> tuple[float, float]:
    """Fit sigmoid(a + b * score) to fractional labels with b >= 0."""
    w = weights / weights.sum()
    q_bar = float(np.clip(np.sum(w * q), 1e-6, 1.0 - 1e-6))
    initial = np.array([np.log(q_bar / (1.0 - q_bar)), 1.0])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = theta
        eta = a + b * score
        loss = float(np.sum(w * (np.logaddexp(0.0, eta) - q * eta)) + 0.5 * l2 * b * b)
        residual = expit(eta) - q
        grad = np.array([
            np.sum(w * residual),
            np.sum(w * residual * score) + l2 * b,
        ])
        return loss, grad

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None), (0.0, None)],
    )
    if not result.success:
        raise RuntimeError(f"Soft logistic calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def _regularized_fisher_direction(
    H: np.ndarray,
    weights: np.ndarray,
    membership_c: np.ndarray,
    membership_d: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Compute a two-group Fisher direction from fractional memberships."""
    wc = weights * membership_c
    wd = weights * membership_d
    if wc.sum() <= 1e-12 or wd.sum() <= 1e-12:
        return np.zeros(H.shape[1], dtype=float)

    mu_c = np.sum(wc[:, None] * H, axis=0) / wc.sum()
    mu_d = np.sum(wd[:, None] * H, axis=0) / wd.sum()
    dev_c = H - mu_c[None, :]
    dev_d = H - mu_d[None, :]
    scatter = (wc[:, None] * dev_c).T @ dev_c + (wd[:, None] * dev_d).T @ dev_d
    scale = np.trace(scatter) / H.shape[1] if np.trace(scatter) > 0 else 1.0
    scatter = scatter + shrinkage * scale * np.eye(H.shape[1])
    return np.linalg.solve(scatter, mu_c - mu_d)


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else np.zeros_like(direction)


def fit_dense_methods(
    H: np.ndarray,
    weights: np.ndarray,
    proba: np.ndarray,
    c: int,
    d: int,
    alpha: float = 1.0,
    shrinkage: float = 1e-3,
) -> dict[str, dict]:
    """Fit all methods densely; no feature selection or refitting occurs."""
    q, _ = _pair_targets(proba, c, d)

    coef_c, intercept_c = _fit_ridge(H, proba[:, c], weights, alpha)
    coef_d, intercept_d = _fit_ridge(H, proba[:, d], weights, alpha)
    fits: dict[str, dict] = {
        "ovr_probability": {
            "method": "ovr_probability",
            "coef_c": coef_c,
            "intercept_c": intercept_c,
            "coef_d": coef_d,
            "intercept_d": intercept_d,
            "direction": coef_c - coef_d,
        }
    }

    coef_q, intercept_q = _fit_ridge(H, q, weights, alpha)
    fits["ovo_probability"] = {
        "method": "ovo_probability",
        "coef": coef_q,
        "intercept": intercept_q,
        "direction": coef_q,
    }

    classes = np.arange(proba.shape[1])
    multi = fit_fisher_soft(H, weights, proba, classes, shrinkage=shrinkage)
    multi_direction = multi["pairwise_direction"](c, d)
    if multi_direction is None:
        multi_direction = np.zeros(H.shape[1], dtype=float)
    multi_direction = _normalize_direction(multi_direction)
    a_multi, b_multi = _fit_soft_logistic_calibration(H @ multi_direction, q, weights)
    fits["multiclass_soft_fisher"] = {
        "method": "multiclass_soft_fisher",
        "coef": b_multi * multi_direction,
        "intercept": a_multi,
        "direction": b_multi * multi_direction,
    }

    pair_direction = _regularized_fisher_direction(
        H, weights, q, 1.0 - q, shrinkage=shrinkage
    )
    pair_direction = _normalize_direction(pair_direction)
    a_pair, b_pair = _fit_soft_logistic_calibration(H @ pair_direction, q, weights)
    fits["pairwise_soft_fisher"] = {
        "method": "pairwise_soft_fisher",
        "coef": b_pair * pair_direction,
        "intercept": a_pair,
        "direction": b_pair * pair_direction,
    }
    return fits


def top_k_features(direction: np.ndarray, k: int | None) -> frozenset[int]:
    """Select post-hoc displayed features from a dense standardized direction."""
    n_features = direction.shape[0]
    if k is None or k >= n_features:
        return frozenset(range(n_features))
    return frozenset(int(i) for i in np.argsort(-np.abs(direction))[:k])


def predict_with_displayed_features(fit: dict, H: np.ndarray, selected: frozenset[int]) -> np.ndarray:
    """Predict after hiding dense coefficients outside ``selected``.

    H=0 at the explained instance, so retaining the original intercept keeps
    the dense and displayed predictions identical at that point.
    """
    idx = np.array(sorted(selected), dtype=int)
    method = fit["method"]
    if method == "ovr_probability":
        pc = np.clip(fit["intercept_c"] + H[:, idx] @ fit["coef_c"][idx], 1e-6, 1.0)
        pd_ = np.clip(fit["intercept_d"] + H[:, idx] @ fit["coef_d"][idx], 1e-6, 1.0)
        return pc / (pc + pd_)

    score = fit["intercept"] + H[:, idx] @ fit["coef"][idx]
    if method == "ovo_probability":
        return np.clip(score, 0.0, 1.0)
    if method in {"multiclass_soft_fisher", "pairwise_soft_fisher"}:
        return expit(score)
    raise ValueError(f"Unknown method: {method}")


def contribution_energy(direction: np.ndarray, selected: frozenset[int]) -> float:
    total = float(direction @ direction)
    if total <= 1e-15:
        return 0.0
    idx = np.array(sorted(selected), dtype=int)
    return float(direction[idx] @ direction[idx] / total)


def direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return float("nan")
    return float((left @ right) / (left_norm * right_norm))


def effective_feature_count(direction: np.ndarray) -> float:
    """Participation ratio of squared standardized contributions."""
    squared = direction ** 2
    total = squared.sum()
    if total <= 1e-15:
        return 0.0
    shares = squared / total
    return float(1.0 / np.sum(shares ** 2))


def choose_smallest_acceptable_budget(
    fit: dict,
    H_validation: np.ndarray,
    q_validation: np.ndarray,
    weights: np.ndarray,
    candidate_budgets: list[int | None],
    r2_tolerance: float,
    agreement_tolerance: float,
) -> int | None:
    """Choose k on validation data, never on the final test neighborhood."""
    all_features = top_k_features(fit["direction"], None)
    dense_pred = predict_with_displayed_features(fit, H_validation, all_features)
    dense_mse = weighted_mse(q_validation, dense_pred, weights)
    dense_agreement = weighted_agreement(q_validation, dense_pred, weights)
    q_mean = np.average(q_validation, weights=weights)
    q_variance = float(np.average((q_validation - q_mean) ** 2, weights=weights))
    mse_limit = dense_mse + r2_tolerance * q_variance
    agreement_limit = dense_agreement - agreement_tolerance

    finite = sorted({int(k) for k in candidate_budgets if k is not None and k < fit["direction"].shape[0]})
    for k in finite:
        selected = top_k_features(fit["direction"], k)
        pred = predict_with_displayed_features(fit, H_validation, selected)
        if (
            weighted_mse(q_validation, pred, weights) <= mse_limit
            and weighted_agreement(q_validation, pred, weights) >= agreement_limit
        ):
            return k
    return None


def choose_energy_budget(direction: np.ndarray, threshold: float) -> int | None:
    """Choose the fewest coefficients that retain a fixed squared energy."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    squared = np.sort(direction ** 2)[::-1]
    total = squared.sum()
    if total <= 1e-15:
        return None
    k = int(np.searchsorted(np.cumsum(squared) / total, threshold) + 1)
    return None if k >= direction.shape[0] else k


def choose_best_validation_budget(
    fit: dict,
    H_validation: np.ndarray,
    q_validation: np.ndarray,
    weights: np.ndarray,
) -> int | None:
    """Choose the post-hoc truncation with minimum validation MSE."""
    n_features = fit["direction"].shape[0]
    candidates: list[int | None] = [*range(1, n_features), None]
    scored = []
    for k in candidates:
        selected = top_k_features(fit["direction"], k)
        pred = predict_with_displayed_features(fit, H_validation, selected)
        # Prefer fewer displayed features for an exact numerical tie.
        scored.append((weighted_mse(q_validation, pred, weights), len(selected), k))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def _unit_direction(direction: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else None


def _evaluate_fit(
    fit: dict,
    H_test: np.ndarray,
    q_test: np.ndarray,
    weights: np.ndarray,
    k: int | None,
) -> dict:
    selected = top_k_features(fit["direction"], k)
    pred = predict_with_displayed_features(fit, H_test, selected)
    dense_selected = top_k_features(fit["direction"], None)
    dense_pred = predict_with_displayed_features(fit, H_test, dense_selected)
    return {
        "q_mse": weighted_mse(q_test, pred, weights),
        "q_r2": weighted_r2(q_test, pred, weights),
        "decision_agreement": weighted_agreement(q_test, pred, weights),
        "compression_mse": weighted_mse(dense_pred, pred, weights),
        "contribution_energy": contribution_energy(fit["direction"], selected),
        "n_features_displayed": len(selected),
    }


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    adaptive_rows: list[dict] = []
    scenario_id = 0
    started = time.time()

    if args.dataset == "digits":
        digits = load_digits()
        problems = [
            (dataset_repeat, digits.data.shape[1], len(np.unique(digits.target)),
             digits.data, digits.target, int(rng.integers(0, 1_000_000)), dataset_repeat)
            for dataset_repeat in range(args.dataset_repeats)
        ]
    else:
        problems = []
        for dataset_repeat in range(args.dataset_repeats):
            for n_features in args.n_features:
                for n_classes in args.n_classes:
                    data_seed = int(rng.integers(0, 1_000_000))
                    n_informative = max(3, n_classes)
                    X, y = make_classification(
                        n_samples=args.n_observations,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_redundant=max(0, n_features - n_informative),
                        n_repeated=0,
                        n_classes=n_classes,
                        n_clusters_per_class=1,
                        class_sep=1.2,
                        random_state=data_seed,
                    )
                    problems.append((
                        dataset_repeat, n_features, n_classes, X, y,
                        data_seed, 0,
                    ))

    for dataset_repeat, n_features, n_classes, X, y, data_seed, split_seed in problems:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=split_seed, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=args.n_trees,
            random_state=data_seed,
            n_jobs=-1,
        ).fit(X_train, y_train)
        feature_std = X_train.std(axis=0)
        feature_std[feature_std == 0] = 1.0
        instances = pick_contested_instances(clf, X_test, args.n_instances)
        print(
            f"[{time.time() - started:6.1f}s] dataset={args.dataset}, "
            f"repeat={dataset_repeat + 1}/{args.dataset_repeats}, "
            f"features={n_features}, classes={n_classes}",
            flush=True,
        )

        for x in instances:
                    scenario_id += 1
                    order = np.argsort(clf.predict_proba(x[None, :])[0])[::-1]
                    c, d = int(order[0]), int(order[1])

                    Z_train, w_train = sample_perturbations(
                        x, feature_std, args.n_train_perturbations, rng
                    )
                    H_train = _standardize_local(Z_train, x, feature_std)
                    p_train = clf.predict_proba(Z_train)
                    fits = fit_dense_methods(
                        H_train, w_train, p_train, c, d,
                        alpha=args.alpha, shrinkage=args.shrinkage,
                    )

                    Z_validation, w_validation = sample_perturbations(
                        x, feature_std, args.n_validation_perturbations, rng
                    )
                    H_validation = _standardize_local(Z_validation, x, feature_std)
                    q_validation, _ = _pair_targets(clf.predict_proba(Z_validation), c, d)

                    Z_test_local, w_test = sample_perturbations(
                        x, feature_std, args.n_test_perturbations, rng
                    )
                    H_test = _standardize_local(Z_test_local, x, feature_std)
                    q_test, _ = _pair_targets(clf.predict_proba(Z_test_local), c, d)

                    ovo_reference = fits["ovo_probability"]
                    ovo_direction = ovo_reference["direction"]
                    ovo_top5 = top_k_features(ovo_direction, min(5, n_features))
                    ovo_dense_pred = predict_with_displayed_features(
                        ovo_reference, H_test, top_k_features(ovo_direction, None)
                    )

                    repeated_directions = {method: [] for method in METHODS}
                    repeated_sets = {method: [] for method in METHODS}
                    for _ in range(args.stability_repeats):
                        Z_repeat, w_repeat = sample_perturbations(
                            x, feature_std, args.n_train_perturbations, rng
                        )
                        H_repeat = _standardize_local(Z_repeat, x, feature_std)
                        repeat_fits = fit_dense_methods(
                            H_repeat, w_repeat, clf.predict_proba(Z_repeat), c, d,
                            alpha=args.alpha, shrinkage=args.shrinkage,
                        )
                        for method, repeat_fit in repeat_fits.items():
                            unit = _unit_direction(repeat_fit["direction"])
                            if unit is not None:
                                repeated_directions[method].append(unit)
                            repeated_sets[method].append(
                                top_k_features(repeat_fit["direction"], min(5, n_features))
                            )

                    common = {
                        "dataset": args.dataset,
                        "scenario_id": scenario_id,
                        "dataset_repeat": dataset_repeat,
                        "n_features": n_features,
                        "n_classes": n_classes,
                        "class_c": c,
                        "class_d": d,
                    }
                    for method, fit in fits.items():
                        method_top5 = top_k_features(fit["direction"], min(5, n_features))
                        method_dense_pred = predict_with_displayed_features(
                            fit, H_test, top_k_features(fit["direction"], None)
                        )
                        structure = {
                            "cosine_vs_ovo": direction_cosine(fit["direction"], ovo_direction),
                            "top5_jaccard_vs_ovo": (
                                len(method_top5 & ovo_top5) / len(method_top5 | ovo_top5)
                            ),
                            "dense_prediction_mse_vs_ovo": weighted_mse(
                                ovo_dense_pred, method_dense_pred, w_test
                            ),
                            "top5_energy": contribution_energy(fit["direction"], method_top5),
                            "effective_feature_count": effective_feature_count(fit["direction"]),
                            "n_nonzero_dense": int(np.sum(np.abs(fit["direction"]) > 1e-12)),
                        }
                        stability = {
                            "dense_direction_cosine": mean_pairwise_cosine(repeated_directions[method]),
                            "top5_jaccard": mean_pairwise_jaccard(repeated_sets[method]),
                        }
                        effective_budgets = [
                            k for k in args.budgets
                            if k is None or k < n_features
                        ]
                        if None not in effective_budgets:
                            effective_budgets.append(None)
                        for k in effective_budgets:
                            metrics = _evaluate_fit(fit, H_test, q_test, w_test, k)
                            rows.append({
                                **common,
                                "method": method,
                                "budget": "all" if k is None or k >= n_features else str(k),
                                **metrics,
                                **stability,
                                **structure,
                            })

                        validation_chosen = choose_smallest_acceptable_budget(
                            fit, H_validation, q_validation, w_validation,
                            [*range(1, n_features), None],
                            args.r2_tolerance, args.agreement_tolerance,
                        )
                        dense_metrics = _evaluate_fit(fit, H_test, q_test, w_test, None)
                        energy_chosen = choose_energy_budget(
                            fit["direction"], args.energy_threshold
                        )
                        best_validation_chosen = choose_best_validation_budget(
                            fit, H_validation, q_validation, w_validation
                        )
                        for selection_policy, chosen in (
                            ("retain_dense", validation_chosen),
                            ("best_validation", best_validation_chosen),
                            (f"energy_{args.energy_threshold:.2f}", energy_chosen),
                        ):
                            adaptive_metrics = _evaluate_fit(fit, H_test, q_test, w_test, chosen)
                            adaptive_rows.append({
                                **common,
                                "method": method,
                                "selection_policy": selection_policy,
                                "chosen_budget": "all" if chosen is None else str(chosen),
                                "needed_all_features": chosen is None,
                                "display_fraction": (
                                    n_features if chosen is None else chosen
                                ) / n_features,
                                **adaptive_metrics,
                                "q_r2_drop_vs_dense": dense_metrics["q_r2"] - adaptive_metrics["q_r2"],
                                "agreement_drop_vs_dense": (
                                    dense_metrics["decision_agreement"]
                                    - adaptive_metrics["decision_agreement"]
                                ),
                                **stability,
                                **structure,
                            })

    return pd.DataFrame(rows), pd.DataFrame(adaptive_rows)


def summarize_by_independent_dataset(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "q_mse", "q_r2", "decision_agreement", "compression_mse",
        "contribution_energy", "n_features_displayed", "dense_direction_cosine", "top5_jaccard",
        "cosine_vs_ovo", "top5_jaccard_vs_ovo", "dense_prediction_mse_vs_ovo",
        "top5_energy", "effective_feature_count", "needed_all_features",
        "display_fraction", "q_r2_drop_vs_dense", "agreement_drop_vs_dense",
        "n_nonzero_dense",
    ]
    available = [metric for metric in metrics if metric in rows.columns]
    cluster_keys = ["dataset_repeat", "n_features", "n_classes", *group_columns]
    cluster = rows.groupby(cluster_keys, sort=False)[available].mean().reset_index()
    grouped = cluster.groupby(group_columns, sort=False)[available]
    summary = grouped.agg(["mean", "std", "count"])
    for metric in available:
        summary[(metric, "ci95")] = 1.96 * summary[(metric, "std")] / np.sqrt(summary[(metric, "count")])
    return summary.sort_index(axis=1, level=0)


def paired_method_differences(rows: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Paired deltas with uncertainty over independently fitted black boxes."""
    metrics = [
        "q_mse", "q_r2", "decision_agreement", "n_features_displayed",
        "q_r2_drop_vs_dense", "agreement_drop_vs_dense",
    ]
    dataset_keys = ["dataset_repeat", "n_features", "n_classes", "selection_policy"]
    cluster = rows.groupby([*dataset_keys, "method"], sort=False)[metrics].mean().reset_index()
    output = []
    for selection_policy in rows["selection_policy"].unique():
        policy_cluster = cluster[cluster["selection_policy"] == selection_policy]
        base = policy_cluster[policy_cluster["method"] == baseline]
        for method in METHODS:
            if method == baseline:
                continue
            other = policy_cluster[policy_cluster["method"] == method]
            merged = base.merge(other, on=dataset_keys, suffixes=("_baseline", "_method"))
            for metric in metrics:
                delta = merged[f"{metric}_method"] - merged[f"{metric}_baseline"]
                output.append({
                    "selection_policy": selection_policy,
                    "baseline": baseline,
                    "method": method,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "ci95": float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta))),
                    "n_independent_black_boxes": int(len(delta)),
                })
    return pd.DataFrame(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test grid")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--shrinkage", type=float, default=1e-3)
    parser.add_argument("--r2-tolerance", type=float, default=0.02)
    parser.add_argument("--agreement-tolerance", type=float, default=0.01)
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--digits", action="store_true", help="Use sklearn Digits instead of synthetic grids")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "results")
    args = parser.parse_args()

    args.dataset = "digits" if args.digits else "synthetic"
    if args.digits:
        args.n_features = [64]
        args.n_classes = [10]
        args.dataset_repeats = 2
        args.n_instances = 15
        args.stability_repeats = 8
        args.n_train_perturbations = 500
        args.n_validation_perturbations = 800
        args.n_test_perturbations = 1500
        args.n_observations = 1797
        args.n_trees = 250
        args.budgets = [1, 3, 5, 8, 10, 15, 20, 30, None]
    elif args.quick:
        args.n_features = [8]
        args.n_classes = [3]
        args.dataset_repeats = 1
        args.n_instances = 3
        args.stability_repeats = 3
        args.n_train_perturbations = 150
        args.n_validation_perturbations = 250
        args.n_test_perturbations = 300
        args.n_observations = 1000
        args.n_trees = 80
        args.budgets = [1, 3, 5, None]
    else:
        args.n_features = [8, 14, 20]
        args.n_classes = [3, 4, 5]
        args.dataset_repeats = 2
        args.n_instances = 10
        args.stability_repeats = 8
        args.n_train_perturbations = 300
        args.n_validation_perturbations = 500
        args.n_test_perturbations = 1000
        args.n_observations = 2000
        args.n_trees = 200
        args.budgets = [1, 2, 3, 5, 8, 10, None]
    return args


def main() -> None:
    args = parse_args()
    rows, adaptive = run_experiment(args)
    fixed_summary = summarize_by_independent_dataset(rows, ["budget", "method"])
    adaptive_summary = summarize_by_independent_dataset(adaptive, ["selection_policy", "method"])
    paired_vs_ovr = paired_method_differences(adaptive, "ovr_probability")
    paired_vs_ovo = paired_method_differences(adaptive, "ovo_probability")
    paired = pd.concat([paired_vs_ovr, paired_vs_ovo], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "dense_posthoc_rows.csv", index=False)
    adaptive.to_csv(args.output_dir / "dense_posthoc_adaptive_rows.csv", index=False)
    fixed_summary.to_csv(args.output_dir / "dense_posthoc_summary.csv")
    adaptive_summary.to_csv(args.output_dir / "dense_posthoc_adaptive_summary.csv")
    paired.to_csv(args.output_dir / "dense_posthoc_paired_differences.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Fixed display budgets (means over explanation points) ===")
    print(rows.groupby(["budget", "method"], sort=False)[[
        "q_mse", "q_r2", "decision_agreement", "contribution_energy",
        "n_features_displayed", "dense_direction_cosine", "top5_jaccard",
    ]].mean())
    print("\n=== Validation-selected display budgets, held-out test results ===")
    print(adaptive.groupby(["selection_policy", "method"], sort=False)[[
        "q_mse", "q_r2", "decision_agreement", "q_r2_drop_vs_dense",
        "agreement_drop_vs_dense", "contribution_energy", "n_features_displayed",
    ]].mean())
    print("\n=== Chosen-budget distribution ===")
    print(pd.crosstab(
        [adaptive["selection_policy"], adaptive["method"]],
        adaptive["chosen_budget"], normalize="index"
    ))
    print("\n=== Paired adaptive deltas over independent black boxes ===")
    print(paired.to_string(index=False))
    print("\n=== Diagnostic: why sparse display sometimes fails ===")
    print(adaptive.groupby(["selection_policy", "n_features", "method"], sort=False)[[
        "needed_all_features", "top5_energy", "effective_feature_count",
        "cosine_vs_ovo", "top5_jaccard_vs_ovo", "dense_prediction_mse_vs_ovo",
    ]].mean())
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
