"""The two surrogate-fitting methods being compared.

Both take the identical perturbed neighborhood (Z, weights) produced by
perturbation.sample_perturbations, and the identical black-box outputs on
that neighborhood. Only the fitting step differs -- this isolates exactly
the thing the research question is about.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def fit_onevsrest(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                   alpha: float = 1.0) -> dict:
    """Classic LIME one-vs-rest: one independent weighted ridge regression
    per class, each regressed directly on that class's black-box probability.

    Returns a dict: class_index -> {"coef": w, "local_pred": predicted P(c) at x}
    """
    n_classes = proba.shape[1]
    out = {}
    for c in range(n_classes):
        model = Ridge(alpha=alpha)
        model.fit(Z, proba[:, c], sample_weight=weights)
        local_pred = float(model.predict(x[None, :])[0])
        out[c] = {"coef": model.coef_.copy(), "local_pred": local_pred}
    return out


def fit_fisher(Z: np.ndarray, weights: np.ndarray, hard_labels: np.ndarray,
               classes: np.ndarray, shrinkage: float = 1e-3) -> dict:
    """Multiclass Fisher LDA surrogate with a single pooled S_W across all
    classes present in the (hard-labeled) perturbed neighborhood.

    Returns a dict with the pooled S_W^{-1}, per-class weighted centroids,
    and a helper to compute pairwise directions v(X, Y) = S_W^{-1}(mu_X-mu_Y).
    """
    n_features = Z.shape[1]
    present = np.unique(hard_labels)

    mu = {}
    S_W = np.zeros((n_features, n_features))
    for c in present:
        mask = hard_labels == c
        w_c = weights[mask]
        Z_c = Z[mask]
        w_sum = w_c.sum()
        if w_sum <= 0:
            continue
        mu_c = (w_c[:, None] * Z_c).sum(axis=0) / w_sum
        mu[c] = mu_c
        dev = Z_c - mu_c[None, :]
        S_W += (w_c[:, None] * dev).T @ dev

    # shrinkage regularization: S_W + eps * trace(S_W)/d * I
    eps = shrinkage * (np.trace(S_W) / n_features if np.trace(S_W) > 0 else 1.0)
    S_W_reg = S_W + eps * np.eye(n_features)
    S_W_inv = np.linalg.inv(S_W_reg)

    def pairwise_direction(c1, c2):
        if c1 not in mu or c2 not in mu:
            return None
        return S_W_inv @ (mu[c1] - mu[c2])

    return {
        "mu": mu,
        "S_W": S_W_reg,
        "S_W_inv": S_W_inv,
        "present_classes": present,
        "pairwise_direction": pairwise_direction,
    }
