"""Reference-class joint log-ratio LIME with row-wise (group) sparsity.

Core idea (from the user's proposal, items 3+4+5 of the design discussion):
instead of K independent one-vs-rest surrogates, or K-1 independently-fit
pairwise Contrastive surrogates, fit ONE multi-output regression that
predicts all K-1 log-odds-ratios log(p_r/p_k) (k != reference class r)
*simultaneously*, with a penalty that couples the K-1 output columns through
a shared row-sparsity pattern (MultiTaskLasso's L2,1 penalty:
alpha * sum_j ||B[j,:]||_2 -- a feature's row is either all-zero or active
across every class-pair column together, unlike per-column Lasso where each
column picks its own subset independently).

This directly targets LIMEtree's actual named failure mode (Sec. 3, p.5:
models that "split on different feature subsets") by construction, rather
than relying on emergent structure the way Contrastive's independent
per-pair fits do.

The reference class r is always the black box's predicted class at x (so
r == c_star in the other experiments' terminology): column k of B is then
literally the pairwise direction v(c_star, k) used everywhere else in this
codebase, with intercept b[k].

A valid, bounded, sum-to-one probability vector is recoverable from the
fitted log-ratios via the standard multinomial-logit inverse (see
recover_proba) -- this is the one thing that is NOT just a re-derivation of
something already proven false in this codebase's history: LIMEtree itself
does not guarantee this (Fig. 2's caption admits its own tree-based
surrogate's per-node probabilities need not sum to 1), and a plain
independent one-vs-rest regression's raw output is not naturally bounded
either (LIMEtree p.5's other complaint) -- this method fixes both by
construction.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import MultiTaskLasso, Ridge


def _build_log_ratio_targets(proba: np.ndarray, r: int, classes: np.ndarray, eps: float = 1e-6):
    """Returns Y (N, n_classes-1) and the ordered list of non-reference
    classes (column j of Y/B corresponds to other_classes[j])."""
    other_classes = [int(c) for c in classes if c != r]
    Y = np.column_stack([
        np.log((proba[:, r] + eps) / (proba[:, k] + eps)) for k in other_classes
    ])
    return Y, other_classes


def fit_grouplasso(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, r: int,
                    classes: np.ndarray, x: np.ndarray, alpha: float = 0.01,
                    eps: float = 1e-6) -> dict:
    """Dense-ish (small default alpha) group-lasso joint fit, for
    fidelity/stability experiments where we want the natural, un-truncated
    surrogate rather than a specific K-sparse display version."""
    Y, other_classes = _build_log_ratio_targets(proba, r, classes, eps)
    model = MultiTaskLasso(alpha=alpha, max_iter=5000)
    model.fit(Z, Y, sample_weight=weights)
    # MultiTaskLasso.coef_ has shape (n_targets, n_features); transpose to (n_features, n_targets)
    B = model.coef_.T.copy()
    b = model.intercept_.copy()
    local_pred = x @ B + b
    return {
        "r": r, "other_classes": other_classes, "B": B, "b": b,
        "local_pred": dict(zip(other_classes, local_pred.tolist())),
    }


def _active_row_count(B: np.ndarray, tol: float = 1e-10) -> int:
    return int(np.sum(np.linalg.norm(B, axis=1) > tol))


def fit_grouplasso_sparse(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, r: int,
                           classes: np.ndarray, x: np.ndarray, K: int,
                           eps: float = 1e-6, lo: float = 1e-5, hi: float = 10.0,
                           iters: int = 15) -> dict:
    """Binary-search alpha for the sparsest group-lasso fit with >= K active
    feature ROWS (shared across all class-pair columns by construction --
    there is only ever ONE feature subset to report here, unlike
    one-vs-rest/Contrastive which need a separate top-K per class/pair)."""
    Y, other_classes = _build_log_ratio_targets(proba, r, classes, eps)

    def fit_at(a):
        m = MultiTaskLasso(alpha=a, max_iter=5000)
        m.fit(Z, Y, sample_weight=weights)
        B = m.coef_.T
        return _active_row_count(B), m

    n_lo, m_lo = fit_at(lo)
    n_hi, m_hi = fit_at(hi)
    best = m_lo if n_lo >= K else m_hi
    if n_lo >= K:
        cur_lo, cur_hi = lo, hi
        for _ in range(iters):
            mid = float(np.sqrt(cur_lo * cur_hi))
            n_mid, m_mid = fit_at(mid)
            if n_mid >= K:
                cur_lo, best = mid, m_mid
            else:
                cur_hi = mid

    B = best.coef_.T.copy()
    b = best.intercept_.copy()
    row_norms = np.linalg.norm(B, axis=1)
    active = frozenset(np.argsort(-row_norms)[:K].tolist())
    mask = np.zeros(B.shape[0], dtype=bool)
    mask[list(active)] = True
    B_masked = B * mask[:, None]
    local_pred = x @ B_masked + b
    return {
        "r": r, "other_classes": other_classes, "B": B_masked, "b": b,
        "selected": active,
        "local_pred": dict(zip(other_classes, local_pred.tolist())),
    }


def recover_proba(fit_result: dict, Z: np.ndarray, n_classes: int) -> np.ndarray:
    """Multinomial-logit inverse: turns the fitted log(p_r/p_k) surface
    into a genuine probability vector, bounded in (0,1) and summing to 1
    by construction, for every row of Z."""
    r = fit_result["r"]
    other_classes = fit_result["other_classes"]
    B, b = fit_result["B"], fit_result["b"]

    ell = Z @ B + b[None, :]  # (N, n_classes-1): log(p_r/p_k) for each k
    ell = np.clip(ell, -30, 30)  # avoid overflow in exp
    # ell_k = log(p_r/p_k)  =>  p_k = p_r * exp(-ell_k)
    # p_r + sum_k p_r*exp(-ell_k) = 1  =>  p_r = 1 / (1 + sum_k exp(-ell_k))
    denom = 1.0 + np.exp(-ell).sum(axis=1)
    p_r = 1.0 / denom

    proba = np.zeros((Z.shape[0], n_classes))
    proba[:, r] = p_r
    for j, k in enumerate(other_classes):
        proba[:, k] = np.exp(-ell[:, j]) * p_r
    return proba
