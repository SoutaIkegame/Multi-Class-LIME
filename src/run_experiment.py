"""Main experiment: one-vs-rest LIME vs Fisher LIME, compared on
(1) consistency -- sum-to-one deviation & transitivity violation rate, and
(2) stability -- variance of the competing-pair direction vector under
    repeated perturbation resampling,
across a grid of (n_features, n_classes).

Usage: python3 src/run_experiment.py
Output: results/experiment_results.csv (+ printed summary)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import fit_onevsrest, fit_fisher  # noqa: E402
from metrics import sum_to_one_deviation, transitivity_violation_rate, total_variance  # noqa: E402

N_FEATURES_GRID = [5, 10, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_STABILITY_REPEATS = 15
N_PERTURB_SAMPLES = 300
SEED = 0


def pick_contested_instances(clf, X_pool, n_instances):
    proba = clf.predict_proba(X_pool)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    order = np.argsort(margin)  # smallest margin first: most contested points
    idx = order[:n_instances]
    return X_pool[idx]


def run_one_cell(n_features: int, n_classes: int, rng: np.random.Generator) -> list[dict]:
    n_informative = min(n_features, max(n_classes, 3))
    X, y = make_classification(
        n_samples=2000,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)

    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    classes = np.arange(n_classes)

    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        # --- one shared-sampling run for the consistency metrics ---
        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)

        ovr = fit_onevsrest(Z, weights, proba, x)
        fisher = fit_fisher(Z, weights, hard_labels, classes)

        local_preds = {c: ovr[c]["local_pred"] for c in range(n_classes)}
        sum_dev = sum_to_one_deviation(local_preds)

        ovr_diff_fn = lambda c1, c2: local_preds[c1] - local_preds[c2]  # noqa: E731

        def fisher_diff_fn(c1, c2):
            v = fisher["pairwise_direction"](c1, c2)
            if v is None:
                return None
            return float(v @ x)

        ovr_violation = transitivity_violation_rate(ovr_diff_fn, classes)
        fisher_violation = transitivity_violation_rate(fisher_diff_fn, classes)

        # --- competing pair: predicted class vs runner-up, by black box ---
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c_star, c_runner = int(order[0]), int(order[1])

        # --- repeated resampling for the stability metric ---
        ovr_diffs, fisher_diffs = [], []
        for k in range(N_STABILITY_REPEATS):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            probak = clf.predict_proba(Zk)
            hard_k = probak.argmax(axis=1)

            ovr_k = fit_onevsrest(Zk, wk, probak, x)
            fisher_k = fit_fisher(Zk, wk, hard_k, classes)

            ovr_diffs.append(ovr_k[c_star]["coef"] - ovr_k[c_runner]["coef"])
            v_k = fisher_k["pairwise_direction"](c_star, c_runner)
            if v_k is not None:
                fisher_diffs.append(v_k)

        ovr_var = total_variance(ovr_diffs)
        fisher_var = total_variance(fisher_diffs) if len(fisher_diffs) >= 2 else float("nan")

        rows.append(dict(
            n_features=n_features, n_classes=n_classes,
            sum_to_one_deviation=sum_dev,
            ovr_transitivity_violation_rate=ovr_violation,
            fisher_transitivity_violation_rate=fisher_violation,
            ovr_pairdiff_variance=ovr_var,
            fisher_pairdiff_variance=fisher_var,
        ))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{time.time()-t0:6.1f}s] running n_features={n_features}, n_classes={n_classes} ...")
            rows = run_one_cell(n_features, n_classes, rng)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "experiment_results.csv", index=False)

    summary = df.groupby(["n_features", "n_classes"]).mean(numeric_only=True)
    summary.to_csv(out_dir / "experiment_summary.csv")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== per-(n_features, n_classes) mean over instances/repeats ===")
    print(summary)

    print("\n=== overall mean across all grid cells ===")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
