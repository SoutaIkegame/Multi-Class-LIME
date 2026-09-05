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


def fit_contrastive(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                     x: np.ndarray, eps: float = 1e-6, alpha: float = 1.0) -> dict:
    """"Contrastive LIME": instead of fitting two one-vs-rest surrogates and
    subtracting them, directly regress log((p_c1+eps)/(p_c2+eps)) on z. For
    a softmax black box this target equals the raw score difference
    s_c1 - s_c2, so sign(prediction) == sign(p_c1(z) - p_c2(z)) exactly,
    and third classes cannot distort the c1-vs-c2 boundary the way they can
    distort the magnitude of a raw probability difference p_c1 - p_c2.

    Unlike fit_onevsrest_lasso's per-class independent fits or fit_fisher's
    shared-S_W pooling, this is fit independently PER PAIR: nothing forces
    g(A,B), g(B,C), g(A,C) to share structure, so LIMEtree's "different
    feature subsets" failure mode can still occur across pairs even though
    each individual pair's fidelity should be excellent (it directly
    optimizes the quantity being displayed).
    """
    y = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    model = Ridge(alpha=alpha)
    model.fit(Z, y, sample_weight=weights)
    local_pred = float(model.predict(x[None, :])[0])
    return {"coef": model.coef_.copy(), "intercept": float(model.intercept_), "local_pred": local_pred}


def fit_contrastive_lasso(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                           x: np.ndarray, K: int, eps: float = 1e-6) -> dict:
    """Lasso-selected (top-K feature) version of fit_contrastive, for the
    feature-overlap experiment -- mirrors fit_onevsrest_lasso's methodology
    but on the log-odds-of-the-pair target."""
    y = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    model = _sparsest_lasso_with_at_least_k(Z, weights, y, K)
    coef = model.coef_.copy()
    idx = np.argsort(-np.abs(coef))[:K]
    mask = np.zeros_like(coef, dtype=bool)
    mask[idx] = True
    coef_masked = np.where(mask, coef, 0.0)
    intercept = float(model.intercept_)
    local_pred = float(intercept + coef_masked @ x)
    return {
        "coef": coef_masked,
        "intercept": intercept,
        "local_pred": local_pred,
        "selected": frozenset(idx.tolist()),
    }


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


# ---------------------------------------------------------------------------
# Two-stage "shared support" surrogates (proposal, 2026-09-05)
#
# Diagnosis (src/diagnose_fisher_direction.py): as a COEFFICIENT estimator
# Fisher's S_W^{-1}(mu_c - mu_d) is dominated by direct log-odds regression,
# and the residual gap is inherent (within-class scatter is the wrong metric
# for a discriminative target). But pooling S_W across all classes -- the
# thing that gives Fisher its cross-class shared structure -- costs nothing.
# So the role left for Fisher is STRUCTURE, not coefficients: choose ONE
# feature subset shared by every class/pair (LIMEtree's "common structure"
# desideratum), then let Contrastive regression supply the coefficients.
#
# The honest control is the same two-stage scheme with the shared subset
# chosen from plain OVR ridge magnitudes instead of Fisher directions. If
# that matches Fisher, the Fisher step adds nothing and the thesis should
# say so.
# ---------------------------------------------------------------------------

def _aggregate_support(vectors: list[np.ndarray], K: int) -> frozenset:
    """Shared top-K support from several per-class direction vectors: each
    vector is normalized to unit norm (so no class dominates by scale), the
    absolute values are summed feature-wise, and the K largest are kept."""
    score = np.zeros_like(vectors[0], dtype=float)
    for v in vectors:
        n = np.linalg.norm(v)
        if n > 1e-12:
            score += np.abs(v) / n
    return top_k_indices(score, K)


def shared_support_fisher_soft(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray,
                               classes: np.ndarray, K: int) -> frozenset:
    """Proposal, stage 1: one feature subset for ALL classes, chosen from the
    soft-label pooled-S_W Fisher one-vs-rest directions
    v_c = S_W^{-1}(mu_c - mu_{not c}). Soft labels because hard-label sample
    starvation was the single largest fixable error in the diagnosis."""
    fit = fit_fisher_soft(Z, weights, proba, classes)
    vecs = [v for c in classes if (v := fit["onevsrest_direction"](c)) is not None]
    if not vecs:
        return frozenset(range(min(K, Z.shape[1])))
    return _aggregate_support(vecs, K)


def shared_support_ridge(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                         K: int) -> frozenset:
    """Control for stage 1: same aggregation rule, but the per-class vectors
    are the dense OVR ridge coefficients (fit_onevsrest) instead of Fisher
    directions. Isolates whether the Fisher geometry itself picks a better
    shared subset than ordinary regression magnitudes do."""
    fit = fit_onevsrest(Z, weights, proba, x)
    vecs = [fit[c]["coef"] for c in sorted(fit)]
    return _aggregate_support(vecs, K)


def fit_contrastive_on_support(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray,
                               c1: int, c2: int, x: np.ndarray, support: frozenset,
                               eps: float = 1e-6, alpha: float = 1.0) -> dict:
    """Stage 2: Contrastive log-odds ridge restricted to `support`. Returns a
    dense coef vector (zeros off-support) so it is drop-in comparable with
    fit_contrastive / fit_contrastive_lasso outputs."""
    idx = np.array(sorted(support), dtype=int)
    y = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    model = Ridge(alpha=alpha)
    model.fit(Z[:, idx], y, sample_weight=weights)
    coef = np.zeros(Z.shape[1])
    coef[idx] = model.coef_
    intercept = float(model.intercept_)
    return {
        "coef": coef,
        "intercept": intercept,
        "local_pred": float(intercept + coef @ x),
        "selected": frozenset(idx.tolist()),
    }
