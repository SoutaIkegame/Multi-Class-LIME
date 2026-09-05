"""Do proposal B (contest-weighted kernel) and proposal C (soft-label
logistic loss) stack? They are orthogonal changes -- B reweights which
perturbations matter, C changes how the pair target is fit -- and both
independently improved overall/moderate-region fidelity and stability by
similar margins. This tests all four combinations on the RF black box.

  standard    fit_contrastive with the plain LIME kernel (baseline)
  kernel      fit_contrastive with the contest-weighted kernel (B alone)
  logistic    fit_ovo_logistic with the plain LIME kernel (C alone)
  combined    fit_ovo_logistic with the contest-weighted kernel (B+C)

Usage: python3 src/run_combined_bc_experiment.py
Output: results/combined_bc_{results,stats}.csv
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
from surrogates import fit_contrastive, fit_ovo_logistic  # noqa: E402
from metrics import total_variance_normalized  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from run_pair_kernel_experiment import contest_weights  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
N_STABILITY_REPEATS = 8
EXTREME_THRESHOLD = 0.15
SEED = 0
N_DATASET_SEEDS = 20

VARIANTS = ["standard", "kernel", "logistic", "combined"]


def _fit(name, Z, w, proba, c1, c2, x):
    if name == "standard":
        return fit_contrastive(Z, w, proba, c1, c2, x)
    if name == "kernel":
        return fit_contrastive(Z, contest_weights(w, proba, c1, c2), proba, c1, c2, x)
    if name == "logistic":
        return fit_ovo_logistic(Z, w, proba, c1, c2, x)
    if name == "combined":
        return fit_ovo_logistic(Z, contest_weights(w, proba, c1, c2), proba, c1, c2, x)
    raise ValueError(name)


def _sign_acc(coef, b, Z, proba, c1, c2, w, mask=None):
    pred = np.sign(Z @ coef + b)
    true = np.sign(proba[:, c1] - proba[:, c2])
    if mask is None:
        mask = np.ones(len(w), dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(pred[mask] == true[mask], weights=w[mask]))


def run_one_cell(n_features, n_classes, rng):
    n_informative = max(3, n_classes)
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])
        Z, w = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        extreme = np.minimum(proba[:, c1], proba[:, c2]) < EXTREME_THRESHOLD

        fits = {v: _fit(v, Z, w, proba, c1, c2, x) for v in VARIANTS}
        row = dict(n_features=n_features, n_classes=n_classes)
        for v, f in fits.items():
            row[f"{v}_fidelity"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w)
            row[f"{v}_extreme"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w, extreme)
            row[f"{v}_moderate"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w, ~extreme)

        hist = {v: [fits[v]["coef"]] for v in VARIANTS}
        for _ in range(N_STABILITY_REPEATS - 1):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            pk = clf.predict_proba(Zk)
            for v in VARIANTS:
                hist[v].append(_fit(v, Zk, wk, pk, c1, c2, x)["coef"])
        for v in VARIANTS:
            row[f"{v}_direction_variance"] = total_variance_normalized(hist[v])
        rows.append(row)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{time.time()-t0:6.1f}s] n_features={n_features}, n_classes={n_classes}", flush=True)
            for seed in range(N_DATASET_SEEDS):
                rows = run_one_cell(n_features, n_classes, rng)
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "combined_bc_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = [c for c in df.columns if c not in ("n_features", "n_classes", "seed")]
    print("\n=== overall ===")
    print(df[cols].mean().round(4))

    metrics = ["fidelity", "extreme", "moderate", "direction_variance"]
    pairs = []
    for m in metrics:
        pairs += [
            (m, f"combined_{m}", f"standard_{m}"),
            (m, f"combined_{m}", f"kernel_{m}"),
            (m, f"combined_{m}", f"logistic_{m}"),
            (m, f"kernel_{m}", f"standard_{m}"),
            (m, f"logistic_{m}", f"standard_{m}"),
        ]
    stats = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats.to_csv(out / "combined_bc_stats.csv", index=False)
    print("\n=== paired tests (20 seeds, Holm within metric+pair); mean_diff = a - b ===")
    print(stats[["n_features", "n_classes", "metric", "method_a", "method_b", "mean_a", "mean_b",
                 "mean_diff", "p_value", "effect_size", "p_value_holm_reject"]].to_string(index=False))


if __name__ == "__main__":
    main()
