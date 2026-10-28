"""
Aggregate marl-eval-style JSONs from BenchMARL runs and produce cross-algo plots.

BenchMARL writes one JSON per run (``create_json: True`` by default) with shape:
    {env: {task: {algo: {"seed_N": {"absolute_metrics": {...},
                                    "step_0": {"step_count": int, "<metric>": [list]},
                                    "step_1": {...}, ...}}}}}

This script merges every JSON under ``--root`` and emits four standard figures
using only numpy + matplotlib (no marl-eval / rliable dependency):

    sample_efficiency_curves.png     mean +/- std over seeds, one line per algo
    aggregate_scores.png             bar chart with bootstrap CI per algo
    performance_profile.png          fraction of (task, seed) runs above threshold tau
    probability_of_improvement.png   Pr(A_score > B_score) across (task, seed) pairs

Usage:
    python scripts/plot.py
    python scripts/plot.py --root outputs/checkpoints --out outputs/plots
    python scripts/plot.py --metric return --normalize
    python scripts/plot.py --compare mappo,ippo --compare mappo,qmix
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _merge(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _merge(d[k], v)
        else:
            d[k] = v
    return d


def load_runs(root: Path) -> dict:
    """Recursively load every BenchMARL JSON under ``root`` and merge them."""
    merged: dict = {}
    files = [p for p in root.rglob("*.json")
             if "wandb" not in p.name and not p.name.startswith(".")]
    for p in files:
        try:
            with open(p) as f:
                _merge(merged, json.load(f))
        except Exception as e:
            print(f"[plot.py] WARN: skipping {p}: {e}", flush=True)
    print(f"[plot.py] merged {len(files)} json file(s)")
    return merged


def iter_seeds(env_dict: dict, env: str):
    """Yield (task, algo, seed_key, run_dict) for each run in a given env."""
    for task, algos in env_dict[env].items():
        for algo, seeds in algos.items():
            for seed_key, run in seeds.items():
                yield task, algo, seed_key, run


def extract_curve(run: dict, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps, mean-per-step) over the eval episodes inside one run."""
    steps, vals = [], []
    for k, v in run.items():
        if not k.startswith("step_") or not isinstance(v, dict):
            continue
        if metric not in v:
            continue
        episode_vals = v[metric]
        if not episode_vals:
            continue
        steps.append(int(v["step_count"]))
        vals.append(float(np.mean(episode_vals)))
    if not steps:
        return None
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(vals)[order]


def final_score(run: dict, metric: str) -> float | None:
    """Use absolute_metrics if present (marl-eval convention), else last step."""
    abs_m = run.get("absolute_metrics", {}).get(metric)
    if abs_m:
        return float(abs_m[0])
    curve = extract_curve(run, metric)
    if curve is None:
        return None
    return float(curve[1][-1])


def collect_curves(env_dict: dict, env: str, metric: str
                   ) -> dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """{algo: {task: [(steps, vals_per_seed), ...]}}"""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for task, algo, _, run in iter_seeds(env_dict, env):
        c = extract_curve(run, metric)
        if c is not None:
            out[algo][task].append(c)
    return out


def collect_finals(env_dict: dict, env: str, metric: str
                   ) -> dict[str, dict[str, list[float]]]:
    """{algo: {task: [seed_final_score, ...]}}"""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for task, algo, _, run in iter_seeds(env_dict, env):
        s = final_score(run, metric)
        if s is not None:
            out[algo][task].append(s)
    return out


def normalize_finals(finals: dict[str, dict[str, list[float]]]
                     ) -> dict[str, dict[str, list[float]]]:
    """Min-max normalize per task across all algos+seeds, in place-style copy."""
    tasks = {t for algo in finals.values() for t in algo}
    out = {a: {t: list(vs) for t, vs in tasks_d.items()}
           for a, tasks_d in finals.items()}
    for t in tasks:
        all_vals = [v for algo in out.values() for v in algo.get(t, [])]
        if not all_vals:
            continue
        lo, hi = min(all_vals), max(all_vals)
        rng = hi - lo if hi > lo else 1.0
        for algo in out.values():
            if t in algo:
                algo[t] = [(v - lo) / rng for v in algo[t]]
    return out


def _interp_to_grid(curves: list[tuple[np.ndarray, np.ndarray]], n_grid: int = 100
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a list of (steps, vals) to a common linear grid; return (grid, M[seeds, grid])."""
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    if hi <= lo:
        # fall back to the shortest curve's range
        lo = min(c[0][0] for c in curves)
        hi = min(c[0][-1] for c in curves)
    grid = np.linspace(lo, hi, n_grid)
    M = np.stack([np.interp(grid, s, v) for s, v in curves], axis=0)
    return grid, M


def plot_sample_efficiency(curves: dict, out_path: Path, metric: str):
    """One subplot per task + one aggregated subplot. Mean +/- std across seeds."""
    tasks = sorted({t for algo in curves.values() for t in algo})
    if not tasks:
        print("[plot.py] no curves to plot — skipping sample_efficiency")
        return

    n = len(tasks) + 1
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab10")
    algo_color = {a: cmap(i % 10) for i, a in enumerate(sorted(curves))}

    # per-task panels
    for ax, task in zip(axes_flat[:-1], tasks):
        for algo in sorted(curves):
            seeds = curves[algo].get(task, [])
            if not seeds:
                continue
            grid, M = _interp_to_grid(seeds)
            mu, sigma = M.mean(0), M.std(0)
            ax.plot(grid, mu, label=algo, color=algo_color[algo])
            ax.fill_between(grid, mu - sigma, mu + sigma, alpha=0.2, color=algo_color[algo])
        ax.set_title(task, fontsize=10)
        ax.set_xlabel("env steps")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

    # aggregated (mean over tasks of per-algo per-seed-mean curves)
    ax = axes_flat[len(tasks)]
    for algo in sorted(curves):
        all_task_means = []
        common_grid = None
        for task in tasks:
            seeds = curves[algo].get(task, [])
            if not seeds:
                continue
            g, M = _interp_to_grid(seeds)
            if common_grid is None:
                common_grid = g
                all_task_means.append(M.mean(0))
            else:
                all_task_means.append(np.interp(common_grid, g, M.mean(0)))
        if all_task_means:
            stacked = np.stack(all_task_means, axis=0)
            ax.plot(common_grid, stacked.mean(0), label=algo, color=algo_color[algo])
            ax.fill_between(common_grid,
                            stacked.mean(0) - stacked.std(0),
                            stacked.mean(0) + stacked.std(0),
                            alpha=0.2, color=algo_color[algo])
    ax.set_title("aggregated (mean over tasks)", fontsize=10)
    ax.set_xlabel("env steps")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Sample efficiency — {metric}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot.py] wrote {out_path}")


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  reducer=np.mean) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = np.array([reducer(rng.choice(values, size=len(values), replace=True))
                      for _ in range(n_boot)])
    return float(reducer(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def plot_aggregate_scores(finals: dict, out_path: Path, metric: str):
    """Bar chart per algo of mean / IQM / median over (task x seed) with bootstrap CI."""
    algos = sorted(finals)
    if not algos:
        return
    pooled = {a: np.array([v for t_vals in finals[a].values() for v in t_vals])
              for a in algos}

    def iqm(x):
        x = np.sort(x)
        if len(x) < 4:
            return float(np.mean(x)) if len(x) else float("nan")
        lo, hi = int(0.25 * len(x)), int(0.75 * len(x))
        return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))

    reducers = [("Mean", np.mean), ("IQM", iqm), ("Median", np.median)]
    fig, axes = plt.subplots(1, len(reducers), figsize=(5 * len(reducers), 4), squeeze=False)
    for ax, (label, fn) in zip(axes[0], reducers):
        means, los, his = [], [], []
        for a in algos:
            m, lo, hi = _bootstrap_ci(pooled[a], reducer=fn)
            means.append(m); los.append(m - lo); his.append(hi - m)
        ax.bar(algos, means, yerr=[los, his], capsize=4,
               color=[plt.get_cmap("tab10")(i % 10) for i in range(len(algos))])
        ax.set_title(label)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle(f"Aggregate scores — {metric}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot.py] wrote {out_path}")


def plot_performance_profile(finals: dict, out_path: Path, metric: str):
    """For each algo, plot the fraction of (task, seed) runs scoring >= tau as tau sweeps."""
    algos = sorted(finals)
    if not algos:
        return
    pooled = {a: np.array([v for t_vals in finals[a].values() for v in t_vals])
              for a in algos}
    all_vals = np.concatenate(list(pooled.values()))
    if all_vals.size == 0:
        return
    taus = np.linspace(all_vals.min(), all_vals.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, a in enumerate(algos):
        v = pooled[a]
        if v.size == 0:
            continue
        frac = np.array([(v >= t).mean() for t in taus])
        ax.plot(taus, frac, label=a, color=cmap(i % 10), linewidth=2)
    ax.set_xlabel(f"{metric} threshold tau")
    ax.set_ylabel("fraction of runs >= tau")
    ax.set_title(f"Performance profile — {metric}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot.py] wrote {out_path}")


def plot_prob_improvement(finals: dict, pairs: list[tuple[str, str]],
                          out_path: Path, metric: str):
    """For each (A, B), estimate P(A > B) by uniform sampling from each algo's pooled
    final scores. Bar with bootstrap CI."""
    if not pairs:
        return
    rng = np.random.default_rng(0)
    pooled = {a: np.array([v for t_vals in d.values() for v in t_vals])
              for a, d in finals.items()}
    labels, means, los, his = [], [], [], []
    for A, B in pairs:
        if A not in pooled or B not in pooled or pooled[A].size == 0 or pooled[B].size == 0:
            print(f"[plot.py] WARN: skipping pair ({A},{B}) — missing data")
            continue
        a, b = pooled[A], pooled[B]
        # P(A > B) on full pool
        point = float((a[:, None] > b[None, :]).mean())
        boots = []
        for _ in range(1000):
            ai = rng.choice(a, size=len(a), replace=True)
            bi = rng.choice(b, size=len(b), replace=True)
            boots.append((ai[:, None] > bi[None, :]).mean())
        boots = np.array(boots)
        labels.append(f"P({A} > {B})")
        means.append(point)
        los.append(point - np.quantile(boots, 0.025))
        his.append(np.quantile(boots, 0.975) - point)

    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 4))
    ax.bar(labels, means, yerr=[los, his], capsize=4, color="steelblue")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    ax.set_title(f"Probability of improvement — {metric}")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot.py] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("outputs/checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--metric", default="return")
    ap.add_argument("--env", default=None,
                    help="Environment key inside the merged dict (default: first found)")
    ap.add_argument("--normalize", action="store_true",
                    help="Min-max normalize finals per task before aggregate/profile/prob plots")
    ap.add_argument("--compare", action="append", default=[],
                    help="Pair for probability-of-improvement, e.g. --compare mappo,ippo (repeatable). "
                         "If omitted, every unordered algo pair is compared.")
    args = ap.parse_args()

    if not args.root.exists():
        sys.exit(f"Root not found: {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)

    merged = load_runs(args.root)
    if not merged:
        sys.exit("No JSONs found.")
    env = args.env or next(iter(merged))
    print(f"[plot.py] env={env!r}, tasks={sorted(merged[env])}")

    curves = collect_curves(merged, env, args.metric)
    finals = collect_finals(merged, env, args.metric)
    if not curves:
        sys.exit(f"Metric {args.metric!r} not found in any run.")

    if args.normalize:
        finals = normalize_finals(finals)

    plot_sample_efficiency(curves, args.out / "sample_efficiency_curves.png", args.metric)
    plot_aggregate_scores(finals, args.out / "aggregate_scores.png", args.metric)
    plot_performance_profile(finals, args.out / "performance_profile.png", args.metric)
    if args.compare:
        pairs = [tuple(p.split(",", 1)) for p in args.compare if "," in p]
    else:
        algos = sorted(finals)
        pairs = [(a, b) for i, a in enumerate(algos) for b in algos[i + 1:]]
        if pairs:
            print(f"[plot.py] auto-comparing all {len(pairs)} algo pair(s); "
                  "use --compare A,B to restrict")
    plot_prob_improvement(finals, pairs, args.out / "probability_of_improvement.png", args.metric)


if __name__ == "__main__":
    main()
