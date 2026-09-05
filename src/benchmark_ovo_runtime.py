"""Runtime scaling benchmark for OVR and OVO local surrogates.

This benchmark isolates the surrogate-fitting stage after black-box
probabilities have already been obtained.  Black-box inference is shared by
all methods, so measuring it here would confound the cost of the explainer
with the cost of a particular black-box implementation.

Three scaling axes are measured:

1. number of classes n (and therefore n(n-1)/2 pairs),
2. number of local perturbations M,
3. number of interpretable features D.

Compared workloads:

``ovr_all``
    Fit one sparse class-probability surrogate for every class (n fits).
``ovr_selected_pair``
    Fit the two class-probability surrogates needed to contrast a pre-selected
    target and foil (two fits).
``ovo_all_pairs``
    Fit one sparse q_cd surrogate for every pair (n(n-1)/2 fits).
``ovo_selected_pair``
    Fit only the pre-selected target/foil pair (one fit).

Each sparse surrogate uses the same procedure as ``run_ovo_evaluation``:
rank features with a full weighted Ridge model and refit on the selected K
features.  Timings are measured with BLAS thread pools limited to one thread.

Usage:
    python3 src/benchmark_ovo_runtime.py
    python3 src/benchmark_ovo_runtime.py --quick
"""
from __future__ import annotations

import argparse
import gc
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from run_ovo_evaluation import _refit_selected, fit_method  # noqa: E402


METHODS = (
    "ovr_all",
    "ovr_selected_pair",
    "ovo_all_pairs",
    "ovo_selected_pair",
)


def make_local_problem(
    n_classes: int,
    n_perturbations: int,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic-sized local design and softmax BB outputs."""
    Z = rng.normal(size=(n_perturbations, n_features))
    kernel_width = 0.75 * np.sqrt(n_features)
    distance_sq = np.sum(Z**2, axis=1)
    weights = np.sqrt(np.exp(-distance_sq / kernel_width**2))

    black_box_coef = rng.normal(size=(n_features, n_classes)) / np.sqrt(n_features)
    logits = Z @ black_box_coef
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    feature_std = np.ones(n_features)
    return Z, weights, proba, feature_std


def _fit_ovr_all(Z, weights, proba, feature_std, alpha, k) -> None:
    for c in range(proba.shape[1]):
        _refit_selected(Z, proba[:, c], weights, feature_std, alpha, k)


def _fit_ovr_pair(Z, weights, proba, feature_std, alpha, k) -> None:
    for c in (0, 1):
        _refit_selected(Z, proba[:, c], weights, feature_std, alpha, k)


def _fit_ovo_all(Z, weights, proba, feature_std, alpha, k) -> None:
    for c, d in combinations(range(proba.shape[1]), 2):
        fit_method("ovo_probability", Z, weights, proba, c, d, feature_std, alpha, k)


def _fit_ovo_one(Z, weights, proba, feature_std, alpha, k) -> None:
    fit_method("ovo_probability", Z, weights, proba, 0, 1, feature_std, alpha, k)


def benchmark_case(
    axis: str,
    scale_value: int,
    n_classes: int,
    n_perturbations: int,
    n_features: int,
    k: int,
    repeats: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict]:
    Z, weights, proba, feature_std = make_local_problem(
        n_classes, n_perturbations, n_features, rng
    )
    effective_k = min(k, n_features)
    jobs = {
        "ovr_all": lambda: _fit_ovr_all(
            Z, weights, proba, feature_std, alpha, effective_k
        ),
        "ovr_selected_pair": lambda: _fit_ovr_pair(
            Z, weights, proba, feature_std, alpha, effective_k
        ),
        "ovo_all_pairs": lambda: _fit_ovo_all(
            Z, weights, proba, feature_std, alpha, effective_k
        ),
        "ovo_selected_pair": lambda: _fit_ovo_one(
            Z, weights, proba, feature_std, alpha, effective_k
        ),
    }

    # Warm imports, allocations and solver dispatch without paying for a full
    # all-pairs pass before every case.
    _fit_ovo_one(Z, weights, proba, feature_std, alpha, effective_k)
    _refit_selected(Z, proba[:, 0], weights, feature_std, alpha, effective_k)

    rows = []
    for repeat in range(repeats):
        # Rotate order so no method always benefits from being first or last.
        order = METHODS[repeat % len(METHODS):] + METHODS[:repeat % len(METHODS)]
        for method in order:
            # A single selected-pair fit is only a few milliseconds.  Batch it
            # to reduce timer/scheduler noise, then report per-explanation time.
            inner_loops = 20 if method in {"ovr_selected_pair", "ovo_selected_pair"} else 1
            gc.collect()
            gc.disable()
            start = perf_counter()
            for _ in range(inner_loops):
                jobs[method]()
            elapsed_ms = (perf_counter() - start) * 1000.0 / inner_loops
            gc.enable()
            rows.append({
                "axis": axis,
                "scale_value": scale_value,
                "n_classes": n_classes,
                "n_pairs": n_classes * (n_classes - 1) // 2,
                "n_perturbations": n_perturbations,
                "n_features": n_features,
                "K": effective_k,
                "method": method,
                "repeat": repeat,
                "inner_loops": inner_loops,
                "time_ms": elapsed_ms,
            })
    return rows


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "axis", "scale_value", "n_classes", "n_pairs",
        "n_perturbations", "n_features", "K", "method",
    ]
    summary = raw.groupby(keys, sort=False)["time_ms"].agg(
        median_ms="median",
        q1_ms=lambda x: x.quantile(0.25),
        q3_ms=lambda x: x.quantile(0.75),
        mean_ms="mean",
        std_ms="std",
        repeats="count",
    ).reset_index()

    id_cols = [
        "axis", "scale_value", "n_classes", "n_pairs",
        "n_perturbations", "n_features", "K",
    ]
    table = summary.pivot(index=id_cols, columns="method", values="median_ms").reset_index()
    table.columns.name = None
    table["ovo_all_vs_ovr_ratio"] = table["ovo_all_pairs"] / table["ovr_all"]
    table["ovr_pair_vs_ovo_pair_ratio"] = (
        table["ovr_selected_pair"] / table["ovo_selected_pair"]
    )
    table["ovr_vs_selected_ovo_ratio"] = table["ovr_all"] / table["ovo_selected_pair"]
    return summary, table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.quick:
        cases = [
            ("n_classes", n, n, 300, 20, 5) for n in (5, 10)
        ]
        repeats = min(args.repeats, 2)
    else:
        cases = []
        cases.extend(("n_classes", n, n, 1000, 50, 10) for n in (3, 5, 10, 20, 50, 100))
        cases.extend(("n_perturbations", m, 20, m, 50, 10) for m in (100, 300, 1000, 3000, 10000))
        cases.extend(("n_features", d, 10, 1000, d, 10) for d in (10, 20, 50, 100, 200))
        repeats = args.repeats

    raw_rows = []
    with threadpool_limits(limits=1):
        for i, (axis, value, n, m, d, k) in enumerate(cases, start=1):
            print(
                f"[{i:02d}/{len(cases):02d}] {axis}={value}: "
                f"n={n}, M={m}, D={d}, K={min(k, d)}, pairs={n * (n - 1) // 2}",
                flush=True,
            )
            raw_rows.extend(benchmark_case(
                axis, value, n, m, d, k, repeats, args.alpha, rng
            ))

    raw = pd.DataFrame(raw_rows)
    summary, table = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "runtime_benchmark_raw.csv", index=False)
    summary.to_csv(args.output_dir / "runtime_benchmark_summary.csv", index=False)
    table.to_csv(args.output_dir / "runtime_benchmark_table.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== Median runtime (ms) ===")
    print(table.to_string(index=False))
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
