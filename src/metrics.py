"""Consistency and stability metrics for one-vs-rest LIME vs Fisher LIME.

Consistency (Q1-2 in the discussion):
  - sum-to-one deviation: |P(A)+P(B)+P(C)+... - 1| for one-vs-rest's three
    independently-fit local probability surrogates.
  - transitivity violation rate: how often a cyclic/contradictory ranking
    (diff(X,Y)>0 and diff(Y,Z)>0 but diff(X,Z)<=0) appears across all
    ordered triples of classes.
    * one-vs-rest score: diff(X,Y) = local_pred_X - local_pred_Y
    * Fisher score: diff(X,Y) = v(X,Y) . x   (the *uncentered* score --
      this is exactly the quantity the additivity proof covers: a centered/
      thresholded classification rule would break the exact algebraic
      guarantee, see write-up)

Stability (Q1-3 in the discussion):
  - repeat the perturbation-and-fit procedure K times for the same instance,
    for a fixed competitor pair (predicted class c*, runner-up class c').
  - onevsrest diff^(k) = coef_{c*}^(k) - coef_{c'}^(k)   (vector, per repeat)
  - fisher diff^(k)    = v(c*, c')^(k)                    (vector, per repeat)
  - report trace(Cov(diff across k)) for each method: the total variance of
    the pairwise-direction vector under resampled perturbation noise.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


def sum_to_one_deviation(local_preds: dict) -> float:
    return abs(sum(local_preds.values()) - 1.0)


def transitivity_violation_rate(diff_fn, classes: np.ndarray) -> float:
    """diff_fn(c1, c2) -> signed score or None if unavailable."""
    n_checked = 0
    n_violated = 0
    for x_c, y_c, z_c in permutations(classes, 3):
        dxy = diff_fn(x_c, y_c)
        dyz = diff_fn(y_c, z_c)
        dxz = diff_fn(x_c, z_c)
        if dxy is None or dyz is None or dxz is None:
            continue
        n_checked += 1
        if dxy > 0 and dyz > 0 and not (dxz > 0):
            n_violated += 1
    if n_checked == 0:
        return float("nan")
    return n_violated / n_checked


def total_variance(vectors: list[np.ndarray]) -> float:
    """trace(Cov(vectors)) -- sum of per-coordinate variance across repeats."""
    if len(vectors) < 2:
        return float("nan")
    M = np.vstack(vectors)
    return float(np.var(M, axis=0, ddof=1).sum())
