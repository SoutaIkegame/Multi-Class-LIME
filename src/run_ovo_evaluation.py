"""Held-out evaluation of one-vs-rest and one-vs-one LIME targets.

The experiment answers a narrower question than the older
``run_contrastive_experiment.py``: when the explanation question is
"why class c rather than class d?", which surrogate target best preserves
the black box's pairwise conditional probability on unseen local
perturbations?

Compared methods
----------------
``ovr_probability``
    Fit p_c and p_d separately, then form p_c / (p_c + p_d).
``ovr_logit``
    Fit logit(p_c) and logit(p_d) separately, then subtract the two fitted
    logits.  This is a CLIMAX-style target-vs-rest baseline, not an exact
    implementation of the complete CLIMAX algorithm.
``ovo_probability``
    Directly fit q_cd = p_c / (p_c + p_d).
``ovo_logratio``
    Directly fit log(p_c / p_d), then map back with the sigmoid function.

Every method sees the same training perturbations and is evaluated on a
separate perturbation sample.  Stability is measured across independently
resampled training neighborhoods for the same instance and class pair.

Usage
-----
    python3 src/run_ovo_evaluation.py
    python3 src/run_ovo_evaluation.py --quick

Outputs are written under ``results/`` (which is intentionally gitignored).
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402


METHODS = (
    "ovr_probability",
    "ovr_logit",
    "ovo_probability",
    "ovo_logratio",
)
EPS = 1e-6


def _pair_targets(proba: np.ndarray, c: int, d: int, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Return mutually consistent pairwise probability and log-ratio."""
    pc = proba[:, c] + eps
    pd_ = proba[:, d] + eps
    q = pc / (pc + pd_)
    ell = np.log(pc / pd_)
    return q, ell


def _fit_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X, y, sample_weight=weights)
    return model


def _refit_selected(
    Z: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_std: np.ndarray,
    alpha: float,
    k: int | None,
) -> tuple[np.ndarray, float, frozenset[int]]:
    """Rank with a full Ridge fit, then refit on the selected features.

    Ranking uses ``abs(beta_j * feature_std_j)`` so feature scale does not
    determine which variables are selected.  ``k=None`` keeps all features.
    """
    n_features = Z.shape[1]
    full = _fit_ridge(Z, y, weights, alpha)
    if k is None or k >= n_features:
        selected = np.arange(n_features)
        model = full
    else:
        importance = np.abs(full.coef_ * feature_std)
        selected = np.argsort(-importance)[:k]
        model = _fit_ridge(Z[:, selected], y, weights, alpha)

    coef = np.zeros(n_features, dtype=float)
    coef[selected] = model.coef_
    return coef, float(model.intercept_), frozenset(int(i) for i in selected)


def fit_method(
    method: str,
    Z: np.ndarray,
    weights: np.ndarray,
    proba: np.ndarray,
    c: int,
    d: int,
    feature_std: np.ndarray,
    alpha: float,
    k: int | None,
) -> dict:
    """Fit one method and return sufficient state for prediction/evaluation."""
    q, ell = _pair_targets(proba, c, d)

    if method == "ovr_probability":
        coef_c, intercept_c, selected_c = _refit_selected(
            Z, proba[:, c], weights, feature_std, alpha, k
        )
        coef_d, intercept_d, selected_d = _refit_selected(
            Z, proba[:, d], weights, feature_std, alpha, k
        )
        # The displayed OVR contrast is the difference of the two classwise
        # explanations.  Direction stability is evaluated after scale
        # normalization, so probability-vs-log-odds units do not dominate it.
        direction = coef_c - coef_d
        return {
            "method": method,
            "coef_c": coef_c,
            "intercept_c": intercept_c,
            "coef_d": coef_d,
            "intercept_d": intercept_d,
            "direction": direction,
            "selected": selected_c | selected_d,
        }

    if method == "ovr_logit":
        pc = np.clip(proba[:, c], EPS, 1.0 - EPS)
        pd_ = np.clip(proba[:, d], EPS, 1.0 - EPS)
        logit_c = np.log(pc / (1.0 - pc))
        logit_d = np.log(pd_ / (1.0 - pd_))
        coef_c, intercept_c, selected_c = _refit_selected(
            Z, logit_c, weights, feature_std, alpha, k
        )
        coef_d, intercept_d, selected_d = _refit_selected(
            Z, logit_d, weights, feature_std, alpha, k
        )
        return {
            "method": method,
            "coef_c": coef_c,
            "intercept_c": intercept_c,
            "coef_d": coef_d,
            "intercept_d": intercept_d,
            "direction": coef_c - coef_d,
            "selected": selected_c | selected_d,
        }

    target = q if method == "ovo_probability" else ell
    if method not in {"ovo_probability", "ovo_logratio"}:
        raise ValueError(f"Unknown method: {method}")
    coef, intercept, selected = _refit_selected(
        Z, target, weights, feature_std, alpha, k
    )
    return {
        "method": method,
        "coef": coef,
        "intercept": intercept,
        "direction": coef,
        "selected": selected,
    }


def predict_pairwise_probability(fit: dict, Z: np.ndarray) -> np.ndarray:
    """Map every surrogate to the common evaluation target q_cd."""
    method = fit["method"]
    if method == "ovr_probability":
        pc = np.clip(Z @ fit["coef_c"] + fit["intercept_c"], EPS, 1.0)
        pd_ = np.clip(Z @ fit["coef_d"] + fit["intercept_d"], EPS, 1.0)
        return pc / (pc + pd_)
    if method == "ovr_logit":
        logit_c = Z @ fit["coef_c"] + fit["intercept_c"]
        logit_d = Z @ fit["coef_d"] + fit["intercept_d"]
        return expit(logit_c - logit_d)
    score = Z @ fit["coef"] + fit["intercept"]
    if method == "ovo_probability":
        return np.clip(score, 0.0, 1.0)
    if method == "ovo_logratio":
        return expit(score)
    raise ValueError(f"Unknown method: {method}")


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((y_true - y_pred) ** 2, weights=weights))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(y_true, weights=weights)
    denominator = np.average((y_true - mean) ** 2, weights=weights)
    if denominator <= 1e-15:
        return float("nan")
    return float(1.0 - weighted_mse(y_true, y_pred, weights) / denominator)


def weighted_agreement(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((y_true >= 0.5) == (y_pred >= 0.5), weights=weights))


def _unit_contribution_direction(coef: np.ndarray, feature_std: np.ndarray) -> np.ndarray | None:
    contribution = coef * feature_std
    norm = np.linalg.norm(contribution)
    if norm <= 1e-12:
        return None
    return contribution / norm


def mean_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return float("nan")
    values = [float(vectors[i] @ vectors[j]) for i, j in combinations(range(len(vectors)), 2)]
    return float(np.mean(values))


def mean_pairwise_jaccard(sets: list[frozenset[int]]) -> float:
    if len(sets) < 2:
        return float("nan")
    values = []
    for i, j in combinations(range(len(sets)), 2):
        union = sets[i] | sets[j]
        values.append(1.0 if not union else len(sets[i] & sets[j]) / len(union))
    return float(np.mean(values))


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["q_mse", "q_r2", "decision_agreement", "stability_cosine", "stability_jaccard", "n_features_used"]
    # Ten explanation points share the same fitted black box.  Treating all
    # points as independent would make confidence intervals too narrow, so
    # uncertainty is computed over the 18 independently generated datasets /
    # black boxes (2 repeats x 3 dimensions x 3 class counts).
    cluster_keys = ["dataset_repeat", "n_features", "n_classes", "budget", "method"]
    cluster_means = rows.groupby(cluster_keys, sort=False)[metrics].mean().reset_index()
    grouped = cluster_means.groupby(["budget", "method"], sort=False)[metrics]
    summary = grouped.agg(["mean", "std", "count"])
    for metric in metrics:
        summary[(metric, "ci95")] = 1.96 * summary[(metric, "std")] / np.sqrt(summary[(metric, "count")])
    summary = summary.sort_index(axis=1, level=0)

    paired_rows = []
    for budget in rows["budget"].unique():
        baseline = cluster_means[
            (cluster_means["method"] == "ovr_probability") & (cluster_means["budget"] == budget)
        ]
        for method in METHODS[1:]:
            other = cluster_means[
                (cluster_means["method"] == method) & (cluster_means["budget"] == budget)
            ]
            dataset_keys = ["dataset_repeat", "n_features", "n_classes", "budget"]
            merged = baseline.merge(other, on=dataset_keys, suffixes=("_baseline", "_method"))
            for metric in ["q_mse", "q_r2", "decision_agreement", "stability_cosine", "stability_jaccard"]:
                delta = merged[f"{metric}_method"] - merged[f"{metric}_baseline"]
                ci95 = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else float("nan")
                paired_rows.append({
                    "budget": budget,
                    "method": method,
                    "metric": metric,
                    "mean_delta_vs_ovr_probability": float(delta.mean()),
                    "ci95": float(ci95),
                    "n": int(len(delta)),
                })
    paired = pd.DataFrame(paired_rows)
    return summary, paired


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    scenario_index = 0
    t0 = time.time()

    for dataset_repeat in range(args.dataset_repeats):
        for n_features in args.n_features:
            for n_classes in args.n_classes:
                data_seed = int(rng.integers(0, 1_000_000))
                n_informative = max(3, n_classes)
                n_redundant = max(0, n_features - n_informative)
                X, y = make_classification(
                    n_samples=args.n_observations,
                    n_features=n_features,
                    n_informative=n_informative,
                    n_redundant=n_redundant,
                    n_repeated=0,
                    n_classes=n_classes,
                    n_clusters_per_class=1,
                    class_sep=1.2,
                    random_state=data_seed,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=0, stratify=y
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
                    f"[{time.time() - t0:6.1f}s] repeat={dataset_repeat + 1}/{args.dataset_repeats}, "
                    f"features={n_features}, classes={n_classes}",
                    flush=True,
                )

                for x in instances:
                    scenario_index += 1
                    x_proba = clf.predict_proba(x[None, :])[0]
                    order = np.argsort(x_proba)[::-1]
                    c, d = int(order[0]), int(order[1])

                    Z_train, w_train = sample_perturbations(
                        x, feature_std, args.n_train_perturbations, rng
                    )
                    p_train = clf.predict_proba(Z_train)
                    Z_test_local, w_test = sample_perturbations(
                        x, feature_std, args.n_test_perturbations, rng
                    )
                    p_test = clf.predict_proba(Z_test_local)
                    q_test, _ = _pair_targets(p_test, c, d)

                    for k in args.budgets:
                        budget = "all" if k is None else str(k)
                        fitted = {
                            method: fit_method(
                                method, Z_train, w_train, p_train, c, d,
                                feature_std, args.alpha, k,
                            )
                            for method in METHODS
                        }

                        repeated_directions: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
                        repeated_sets: dict[str, list[frozenset[int]]] = {m: [] for m in METHODS}
                        for _ in range(args.stability_repeats):
                            Z_repeat, w_repeat = sample_perturbations(
                                x, feature_std, args.n_train_perturbations, rng
                            )
                            p_repeat = clf.predict_proba(Z_repeat)
                            for method in METHODS:
                                repeat_fit = fit_method(
                                    method, Z_repeat, w_repeat, p_repeat, c, d,
                                    feature_std, args.alpha, k,
                                )
                                direction = _unit_contribution_direction(
                                    repeat_fit["direction"], feature_std
                                )
                                if direction is not None:
                                    repeated_directions[method].append(direction)
                                repeated_sets[method].append(repeat_fit["selected"])

                        for method, fit in fitted.items():
                            q_pred = predict_pairwise_probability(fit, Z_test_local)
                            rows.append({
                                "scenario_id": scenario_index,
                                "dataset_repeat": dataset_repeat,
                                "n_features": n_features,
                                "n_classes": n_classes,
                                "class_c": c,
                                "class_d": d,
                                "budget": budget,
                                "method": method,
                                "q_mse": weighted_mse(q_test, q_pred, w_test),
                                "q_r2": weighted_r2(q_test, q_pred, w_test),
                                "decision_agreement": weighted_agreement(q_test, q_pred, w_test),
                                "stability_cosine": mean_pairwise_cosine(repeated_directions[method]),
                                "stability_jaccard": mean_pairwise_jaccard(repeated_sets[method]),
                                "n_features_used": len(fit["selected"]),
                            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test grid")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "results")
    args = parser.parse_args()

    if args.quick:
        args.n_features = [8]
        args.n_classes = [3]
        args.dataset_repeats = 1
        args.n_instances = 3
        args.stability_repeats = 4
        args.n_train_perturbations = 150
        args.n_test_perturbations = 300
        args.n_observations = 1000
        args.n_trees = 80
        args.budgets = [None, 3]
    else:
        args.n_features = [8, 14, 20]
        args.n_classes = [3, 4, 5]
        args.dataset_repeats = 2
        args.n_instances = 10
        args.stability_repeats = 10
        args.n_train_perturbations = 300
        args.n_test_perturbations = 1000
        args.n_observations = 2000
        args.n_trees = 200
        args.budgets = [None, 3, 5]
    return args


def main() -> None:
    args = parse_args()
    rows = run_experiment(args)
    summary, paired = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "ovo_evaluation_rows.csv", index=False)
    summary.to_csv(args.output_dir / "ovo_evaluation_summary.csv")
    paired.to_csv(args.output_dir / "ovo_evaluation_paired_differences.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Mean held-out metrics ===")
    print(rows.groupby(["budget", "method"], sort=False)[
        ["q_mse", "q_r2", "decision_agreement", "stability_cosine", "stability_jaccard", "n_features_used"]
    ].mean())
    print("\n=== Paired deltas versus ovr_probability ===")
    print(paired.to_string(index=False))
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
