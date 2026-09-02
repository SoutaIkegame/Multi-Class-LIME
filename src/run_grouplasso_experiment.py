"""Four-way comparison: one-vs-rest, Fisher(hard), Contrastive, and the new
Group-Lasso joint log-ratio method (reference-class log-odds, fit jointly
with row-wise/group sparsity, probability recovered via multinomial-logit
inverse).

Metrics:
  (1) fidelity -- two versions:
      (a) Hellinger distance between a genuine recovered probability vector
          and the black box's actual predict_proba (only well-defined for
          methods that produce valid probabilities: one-vs-rest via
          clip+renormalize, Fisher via its LDA density model, Group-Lasso
          via its native multinomial-logit inverse -- Contrastive doesn't
          produce a full probability vector by itself so is excluded here).
      (b) pairwise sign-agreement on the black-box-chosen (c*, c') pair,
          matching every earlier experiment in this codebase, so
          Group-Lasso and Contrastive are directly comparable to the rest.
  (2) stability -- total_variance_normalized of the (c*, c') pairwise
      direction vector under repeated resampling.
  (3) feature overlap -- for one-vs-rest/Fisher, the usual Lasso-based
      Jaccard overlap across per-class explanations. For Group-Lasso, the
      feature set is IDENTICAL across all class-pair columns by
      construction (one shared row-support) -- this is verified directly
      rather than estimated, and reported alongside the others for scale.

Usage: python3 src/run_grouplasso_experiment.py
Output: results/grouplasso_results.csv (+ printed summary)
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
from surrogates import fit_onevsrest, fit_onevsrest_lasso, fit_fisher, top_k_indices  # noqa: E402
from grouplasso import fit_grouplasso, fit_grouplasso_sparse, recover_proba  # noqa: E402
from fidelity import (  # noqa: E402
    onevsrest_predict_proba, fisher_predict_proba,
    hard_label_priors, weighted_hellinger_loss,
)
from metrics import total_variance_normalized, mean_pairwise_feature_overlap  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
K_FRACS = [0.3, 0.6]
N_INSTANCES = 8
N_STABILITY_REPEATS = 15
N_PERTURB_SAMPLES = 300
GROUPLASSO_ALPHA = 0.001
SEED = 0


def pairwise_sign_accuracy(pred_scores: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                            weights: np.ndarray) -> float:
    true_sign = np.sign(proba[:, c1] - proba[:, c2])
    pred_sign = np.sign(pred_scores)
    return float(np.average(pred_sign == true_sign, weights=weights))


def run_one_cell(n_features: int, n_classes: int, rng: np.random.Generator) -> list[dict]:
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
    classes = np.arange(n_classes)
    k_values = sorted({max(1, round(frac * n_features)) for frac in K_FRACS})

    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c_star, c_runner = int(order[0]), int(order[1])
        r = c_star  # reference class is always the black box's predicted class

        # --- (1a) probability fidelity + (1b) pairwise-sign fidelity ---
        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)

        ovr = fit_onevsrest(Z, weights, proba, x)
        ovr_proba = onevsrest_predict_proba(ovr, Z, n_classes)
        ovr_hellinger = weighted_hellinger_loss(ovr_proba, proba, weights)
        ovr_scores = Z @ (ovr[c_star]["coef"] - ovr[c_runner]["coef"]) + \
            (ovr[c_star]["intercept"] - ovr[c_runner]["intercept"])
        ovr_sign_fid = pairwise_sign_accuracy(ovr_scores, proba, c_star, c_runner, weights)

        fisher = fit_fisher(Z, weights, hard_labels, classes)
        hard_priors = hard_label_priors(hard_labels, weights, classes)
        fisher_proba = fisher_predict_proba(fisher, Z, hard_priors, n_classes)
        fisher_hellinger = weighted_hellinger_loss(fisher_proba, proba, weights)
        v = fisher["pairwise_direction"](c_star, c_runner)
        if v is not None and c_star in fisher["mu"] and c_runner in fisher["mu"]:
            midpoint = 0.5 * (fisher["mu"][c_star] + fisher["mu"][c_runner])
            fisher_sign_fid = pairwise_sign_accuracy((Z - midpoint[None, :]) @ v, proba, c_star, c_runner, weights)
        else:
            fisher_sign_fid = float("nan")

        gl = fit_grouplasso(Z, weights, proba, r, classes, x, alpha=GROUPLASSO_ALPHA)
        gl_proba = recover_proba(gl, Z, n_classes)
        gl_hellinger = weighted_hellinger_loss(gl_proba, proba, weights)
        col = gl["other_classes"].index(c_runner)
        gl_scores = Z @ gl["B"][:, col] + gl["b"][col]
        gl_sign_fid = pairwise_sign_accuracy(gl_scores, proba, c_star, c_runner, weights)

        # --- (2) stability: repeated resampling for the same (c*, c') pair ---
        ovr_diffs, fisher_diffs, gl_diffs = [], [], []
        for _ in range(N_STABILITY_REPEATS):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            probak = clf.predict_proba(Zk)
            hard_k = probak.argmax(axis=1)

            ovr_k = fit_onevsrest(Zk, wk, probak, x)
            ovr_diffs.append(ovr_k[c_star]["coef"] - ovr_k[c_runner]["coef"])

            fisher_k = fit_fisher(Zk, wk, hard_k, classes)
            v_k = fisher_k["pairwise_direction"](c_star, c_runner)
            if v_k is not None:
                fisher_diffs.append(v_k)

            gl_k = fit_grouplasso(Zk, wk, probak, r, classes, x, alpha=GROUPLASSO_ALPHA)
            col_k = gl_k["other_classes"].index(c_runner)
            gl_diffs.append(gl_k["B"][:, col_k])

        ovr_var = total_variance_normalized(ovr_diffs)
        fisher_var = total_variance_normalized(fisher_diffs) if len(fisher_diffs) >= 2 else float("nan")
        gl_var = total_variance_normalized(gl_diffs)

        # --- (3) feature overlap ---
        fisher_ovr_vecs = {c: v for c in classes if (v := fisher["onevsrest_direction"](c)) is not None}

        for K in k_values:
            ovr_lasso = fit_onevsrest_lasso(Z, weights, proba, x, K)
            ovr_topk = {c: ovr_lasso[c]["selected"] for c in range(n_classes)}
            fisher_topk = {c: top_k_indices(v, K) for c, v in fisher_ovr_vecs.items()}
            ovr_overlap = mean_pairwise_feature_overlap(ovr_topk)
            fisher_overlap = mean_pairwise_feature_overlap(fisher_topk)

            gl_sparse = fit_grouplasso_sparse(Z, weights, proba, r, classes, x, K)
            # verify the "always identical support" claim directly: every
            # column of B_masked should share the SAME nonzero row set.
            nonzero_per_col = [
                frozenset(np.nonzero(gl_sparse["B"][:, j])[0].tolist())
                for j in range(gl_sparse["B"].shape[1])
            ]
            gl_topk = {j: s for j, s in enumerate(nonzero_per_col)}
            gl_overlap = mean_pairwise_feature_overlap(gl_topk)

            rows.append(dict(
                n_features=n_features, n_classes=n_classes, K=K,
                ovr_hellinger=ovr_hellinger, fisher_hellinger=fisher_hellinger, gl_hellinger=gl_hellinger,
                ovr_sign_fidelity=ovr_sign_fid, fisher_sign_fidelity=fisher_sign_fid, gl_sign_fidelity=gl_sign_fid,
                ovr_stability_normalized=ovr_var, fisher_stability_normalized=fisher_var,
                gl_stability_normalized=gl_var,
                ovr_feature_overlap=ovr_overlap, fisher_feature_overlap=fisher_overlap,
                gl_feature_overlap=gl_overlap,
            ))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{time.time()-t0:6.1f}s] running n_features={n_features}, n_classes={n_classes} ...")
            all_rows.extend(run_one_cell(n_features, n_classes, rng))

    df = pd.DataFrame(all_rows)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "grouplasso_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    summary = df.groupby(["n_features", "n_classes", "K"]).mean(numeric_only=True)
    print("\n=== per-(n_features, n_classes, K) mean ===")
    print(summary)
    print("\n=== overall mean ===")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
