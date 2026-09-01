"""The two surrogate-fitting methods being compared.

Both take the identical perturbed neighborhood (Z, weights) produced by
perturbation.sample_perturbations, and the identical black-box outputs on
that neighborhood. Only the fitting step differs -- this isolates exactly
the thing the research question is about.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, Ridge


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
        out[c] = {"coef": model.coef_.copy(), "intercept": float(model.intercept_), "local_pred": local_pred}
    return out


def _sparsest_lasso_with_at_least_k(Z, weights, y, k, lo=1e-5, hi=10.0, iters=15):
    """Binary-search alpha for the sparsest weighted Lasso fit that still has
    >= k nonzero coefficients (nnz is ~monotone non-increasing in alpha)."""
    def fit_at(alpha):
        m = Lasso(alpha=alpha, max_iter=5000)
        m.fit(Z, y, sample_weight=weights)
        nnz = int(np.sum(np.abs(m.coef_) > 1e-10))
        return nnz, m

    n_lo, m_lo = fit_at(lo)
    n_hi, m_hi = fit_at(hi)
    if n_hi >= k:
        return m_hi
    if n_lo < k:
        return m_lo  # even the loosest alpha can't reach k features

    best = m_lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        n_mid, m_mid = fit_at(mid)
        if n_mid >= k:
            lo, best = mid, m_mid
        else:
            hi = mid
    return best


def fit_onevsrest_lasso(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                         K: int) -> dict:
    """A closer reimplementation of LIME's actual 'lasso_path'-style feature
    selection: for each class, find the sparsest weighted Lasso fit with at
    least K nonzero features, keep its top-K by magnitude as the SELECTED
    feature set for that class, then read off intercept/coef/local_pred.

    Unlike fit_onevsrest's Ridge-then-truncate ('highest_weights' mode),
    here the K features actually differ in *which ones enter the model* per
    class -- the mechanism LIMEtree (Sec. 3, p.5) names as the source of
    "diverse, inconsistent... explanations": models that "split on
    different feature subsets".
    """
    n_classes = proba.shape[1]
    out = {}
    for c in range(n_classes):
        model = _sparsest_lasso_with_at_least_k(Z, weights, proba[:, c], K)
        coef = model.coef_.copy()
        idx = np.argsort(-np.abs(coef))[:K]
        mask = np.zeros_like(coef, dtype=bool)
        mask[idx] = True
        coef_masked = np.where(mask, coef, 0.0)
        intercept = float(model.intercept_)
        local_pred = float(intercept + coef_masked @ x)
        out[c] = {
            "coef": coef_masked,
            "intercept": intercept,
            "local_pred": local_pred,
            "selected": frozenset(idx.tolist()),
        }
    return out


def fit_fisher(Z: np.ndarray, weights: np.ndarray, hard_labels: np.ndarray,
               classes: np.ndarray, shrinkage: float = 1e-3) -> dict:
    """Multiclass Fisher LDA surrogate with a single pooled S_W across all
    classes present in the (hard-labeled) perturbed neighborhood.

    Returns a dict with the pooled S_W^{-1}, per-class weighted centroids,
    a helper to compute pairwise directions v(X, Y) = S_W^{-1}(mu_X-mu_Y),
    and a one-vs-rest-style per-class direction v(c, not-c) = S_W^{-1}(mu_c -
    mu_not_c) -- the Fisher analogue of LIME's "one surrogate per class",
    used to compare feature usage across classes on equal footing.
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

    def onevsrest_direction(c):
        if c not in mu:
            return None
        mask = hard_labels != c
        w_rest = weights[mask]
        if w_rest.sum() <= 0:
            return None
        mu_rest = (w_rest[:, None] * Z[mask]).sum(axis=0) / w_rest.sum()
        return S_W_inv @ (mu[c] - mu_rest)

    return {
        "mu": mu,
        "S_W": S_W_reg,
        "S_W_inv": S_W_inv,
        "present_classes": present,
        "pairwise_direction": pairwise_direction,
        "onevsrest_direction": onevsrest_direction,
    }


def fit_fisher_soft(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray,
                     classes: np.ndarray, shrinkage: float = 1e-3) -> dict:
    """Soft-label variant of fit_fisher: instead of assigning each z_i to a
    single hard argmax class, every class c gets a weight pi_i * f_c(z_i)
    from every sample. This keeps LIME's original spirit of using the
    continuous probabilities rather than discarding them (see the original
    "soft-label extension" idea), and -- the point of this variant -- it
    means a class is never silently missing from the local neighborhood
    just because it was nobody's single most-likely class: as long as
    f_c(z_i) > 0 somewhere in the neighborhood, mu_c and its contribution to
    S_W are still well-defined. This was empirically found to be the actual
    cause of Fisher's hard-label version losing its feature-overlap
    advantage at higher n_classes (classes vanishing from the local
    argmax-labeled neighborhood), not the feature-redundancy budget.
    """
    n_features = Z.shape[1]
    mu = {}
    S_W = np.zeros((n_features, n_features))
    for idx, c in enumerate(classes):
        soft_w = weights * proba[:, idx]
        w_sum = soft_w.sum()
        if w_sum <= 1e-12:
            continue
        mu_c = (soft_w[:, None] * Z).sum(axis=0) / w_sum
        mu[c] = mu_c
        dev = Z - mu_c[None, :]
        S_W += (soft_w[:, None] * dev).T @ dev

    eps = shrinkage * (np.trace(S_W) / n_features if np.trace(S_W) > 0 else 1.0)
    S_W_reg = S_W + eps * np.eye(n_features)
    S_W_inv = np.linalg.inv(S_W_reg)

    def pairwise_direction(c1, c2):
        if c1 not in mu or c2 not in mu:
            return None
        return S_W_inv @ (mu[c1] - mu[c2])

    def onevsrest_direction(c):
        if c not in mu:
            return None
        idx = int(np.where(classes == c)[0][0])
        soft_w_rest = weights * (1.0 - proba[:, idx])
        w_sum = soft_w_rest.sum()
        if w_sum <= 1e-12:
            return None
        mu_rest = (soft_w_rest[:, None] * Z).sum(axis=0) / w_sum
        return S_W_inv @ (mu[c] - mu_rest)

    return {
        "mu": mu,
        "S_W": S_W_reg,
        "S_W_inv": S_W_inv,
        "present_classes": np.array(sorted(mu.keys())),
        "pairwise_direction": pairwise_direction,
        "onevsrest_direction": onevsrest_direction,
    }


def top_k_indices(vec: np.ndarray, k: int) -> frozenset:
    k = min(k, vec.shape[0])
    order = np.argsort(-np.abs(vec))[:k]
    return frozenset(order.tolist())
