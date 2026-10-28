"""
Evaluate trained doom-marl-cl checkpoints.

Episodes run through the same environment stack the policy was trained on:
BenchMARL's evaluation env is rebuilt from the run's ``config.pkl``, so the
scenario variant, action set, frame stacking, recurrent (LSTM/GRU) state and
episode timeout all match training.

Usage:
    uv run python eval.py --checkpoint outputs/checkpoints/<run>/checkpoints/checkpoint_N.pt
    uv run python eval.py --experiment <substring of run folder name>
    uv run python eval.py --compare <checkpoint_a.pt> <checkpoint_b.pt>

Outputs land in eval_outputs/<run>_<timestamp>/:
    pov_ep{N}.mp4      full-resolution agent POVs with a live minimap
    topdown_ep{N}.mp4  top-down trajectory animation
(--compare writes compare_pov_ep{N}.mp4 / compare_topdown_ep{N}.mp4 instead.)
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS  = PROJECT_ROOT / "outputs" / "checkpoints"
EVAL_OUTPUTS = PROJECT_ROOT / "eval_outputs"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a doom-marl-cl checkpoint.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, metavar="PATH")
    src.add_argument("--experiment", type=str, metavar="PATTERN",
                     help="Substring of a run folder under outputs/checkpoints/; "
                          "the most recent match's latest checkpoint is used.")
    src.add_argument("--compare", type=Path, nargs=2, metavar=("CKPT_A", "CKPT_B"),
                     help="Comparative eval of two checkpoints on the same seeds.")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", type=Path, default=EVAL_OUTPUTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-topdown", action="store_true")
    ap.add_argument("--no-pov", action="store_true")
    ap.add_argument("--pov-fps", type=int, default=15)
    ap.add_argument("--topdown-fps", type=int, default=15)
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def _experiment_dirs(pattern: Optional[str] = None) -> List[Path]:
    dirs = [d for d in CHECKPOINTS.iterdir() if d.is_dir()]
    if pattern:
        dirs = [d for d in dirs if pattern.lower() in d.name.lower()]
    if not dirs:
        sys.exit(f"[eval] No dirs matching {pattern!r} in {CHECKPOINTS}")
    dirs.sort(key=lambda d: d.stat().st_mtime)
    return dirs


def _latest_checkpoint(exp_dir: Path) -> Path:
    ckpt_dir = exp_dir / "checkpoints"
    pts = sorted(ckpt_dir.glob("checkpoint_*.pt"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    if not pts:
        sys.exit(f"[eval] No .pt files in {ckpt_dir}")
    return pts[-1]


def _existing_checkpoint(path: Path) -> Path:
    pt = path.expanduser().resolve()
    if not pt.exists():
        sys.exit(f"[eval] Not found: {pt}")
    return pt


def resolve_checkpoint(args) -> Path:
    if args.checkpoint:
        return _existing_checkpoint(args.checkpoint)
    exp_dir = _experiment_dirs(args.experiment)[-1]
    print(f"[eval] Matched: {exp_dir.name}")
    return _latest_checkpoint(exp_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Load experiment (trained policy + the training-time eval env)
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment(checkpoint: Path, *, seed: int, device: str, out_dir: Path,
                    record_pov: bool, pov_fps: int):
    from doom_marl.experiment import VizdoomExperiment

    exp_folder = checkpoint.parent.parent.resolve()
    config_file = exp_folder / "config.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not config_file.exists():
        sys.exit(f"[eval] config.pkl not found at {config_file}")

    with open(config_file, "rb") as f:
        task              = pickle.load(f)
        task_config       = pickle.load(f)
        algorithm_config  = pickle.load(f)
        model_config      = pickle.load(f)
        _seed             = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        _callbacks        = pickle.load(f)

    # POV video comes from the same in-env recorder training uses for eval
    # videos; it is attached to BenchMARL's test_env and records every episode.
    task_config["seed"] = seed
    task_config["enable_video"] = record_pov
    task_config["eval_video_every_n_evals"] = 1
    task_config["video_fps"] = pov_fps
    task.config = task_config
    # First get_env_fun call builds the eval env (see VizdoomTask.get_env_fun).
    task._eval_env_built = False

    experiment_config.restore_file         = str(checkpoint)
    experiment_config.restore_map_location = device
    experiment_config.loggers              = ["csv"]
    experiment_config.save_folder          = str(out_dir)
    experiment_config.train_device         = device
    experiment_config.sampling_device      = "cpu"
    experiment_config.buffer_device        = "cpu"
    experiment_config.evaluation_episodes  = 1
    # BenchMARL always builds a training collector; keep it to a single
    # serial env so eval doesn't spawn a full set of ViZDoom workers.
    experiment_config.parallel_collection  = False
    experiment_config.on_policy_n_envs_per_worker  = 1
    experiment_config.off_policy_n_envs_per_worker = 1

    experiment = VizdoomExperiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
        callbacks=[],
    )
    print(f"[eval] Loaded: {experiment.name}")
    return experiment, task_config


# ─────────────────────────────────────────────────────────────────────────────
# Run episodes and extract trajectories
# ─────────────────────────────────────────────────────────────────────────────

def run_episodes(experiment, *, episodes: int, seed: int) -> List:
    """Roll out the deterministic policy on the experiment's eval env."""
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    env = experiment.test_env
    rollouts = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for ep in range(episodes):
            try:
                env.set_seed(seed + ep)
            except NotImplementedError:
                pass
            td = env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
            rollouts.append(td.cpu())
            print(f"    episode {ep + 1}/{episodes}: "
                  f"reward={episode_reward(td):.1f}  steps={td.batch_size[0]}")
    return rollouts


def episode_reward(td) -> float:
    """Total reward over all agents and steps (matches training's episode_reward)."""
    return float(td.get(("next", "agent", "reward")).sum())


def _agent_key(td, key: str) -> Optional[np.ndarray]:
    try:
        return td.get(("agent", key)).detach().cpu().numpy()
    except KeyError:
        return None


def extract_trajectory(td):
    """Return (positions, objects_per_step) in the format the renderers expect.

    positions: {agent_i: {"x", "y", "angle"}} arrays over time (pre-action state).
    objects_per_step: list over time of [(x, y), ...] active pickups, or None.
    """
    x, y, angle = (_agent_key(td, k) for k in ("POSITION_X", "POSITION_Y", "ANGLE"))
    if x is None or y is None:
        return {}, None
    if angle is None:
        angle = np.zeros_like(x)
    n_agents = x.shape[1]
    positions = {
        f"agent_{i}": {"x": x[:, i, 0], "y": y[:, i, 0], "angle": angle[:, i, 0]}
        for i in range(n_agents)
    }

    ox, oy, present = (_agent_key(td, k) for k in ("OBJECT_X", "OBJECT_Y", "OBJECT_PRESENT"))
    objects_per_step = None
    if ox is not None and oy is not None and present is not None:
        # Objects are global, so agent 0's view is enough.
        objects_per_step = [
            [(float(ox[t, 0, j]), float(oy[t, 0, j]))
             for j in np.flatnonzero(present[t, 0] > 0.5)]
            for t in range(ox.shape[0])
        ]
    return positions, objects_per_step


def collect_pov_videos(experiment, episodes: int) -> List[Path]:
    """POV mp4s the in-env recorder wrote for this experiment, in episode order.

    Call after ``experiment.close()``: the recorder finalizes the last episode
    on env close. Only the last ``episodes`` files are returned, in case env
    setup itself recorded a short warm-up episode.
    """
    video_dir = Path(experiment.folder_name) / "videos"
    vids = sorted(
        video_dir.glob("worker_*_episode_*.mp4"),
        key=lambda p: int(re.search(r"episode_(\d+)", p.stem).group(1)),
    )
    return vids[-episodes:]


def _read_video(path: Path) -> List[np.ndarray]:
    import imageio.v2 as imageio

    with imageio.get_reader(str(path)) as reader:
        return [np.asarray(frame) for frame in reader]


def evaluate_checkpoint(checkpoint: Path, args, out_dir: Path, device: str):
    """Load a checkpoint, run the episodes, and return everything rendering needs."""
    from doom_marl.callbacks import _parse_wad_map

    experiment, task_config = load_experiment(
        checkpoint, seed=args.seed, device=device, out_dir=out_dir,
        record_pov=not args.no_pov, pov_fps=args.pov_fps,
    )
    try:
        rollouts = run_episodes(experiment, episodes=args.episodes, seed=args.seed)
    finally:
        # Always shut down ViZDoom workers; a crash here used to leave the
        # collector's env processes alive and hang the script.
        experiment.close()

    pov_videos = collect_pov_videos(experiment, args.episodes) if not args.no_pov else []
    map_data = (
        _parse_wad_map(task_config.get("scenario"), task_config.get("config_file"))
        if not args.no_topdown else {}
    )
    return {
        "name": experiment.name,
        "experiment_dir": Path(experiment.folder_name),
        "rewards": [episode_reward(td) for td in rollouts],
        "trajectories": [extract_trajectory(td) for td in rollouts],
        "pov_videos": pov_videos,
        "map_data": map_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Render top-down trajectory video
# ─────────────────────────────────────────────────────────────────────────────

def render_topdown_video(
    positions: Dict[str, Dict],
    map_data: Dict,
    out_path: Path,
    ep_idx: int,
    fps: int = 15,
    objects_per_step: Optional[List] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import imageio

    AGENT_COLORS = ["#FFD700", "#FF4FB8", "#00C8FF", "#FF6B35"]
    BG    = (0.05, 0.07, 0.09)
    PANEL = (0.10, 0.13, 0.18)

    agents = list(positions.keys())
    T = len(positions[agents[0]]["x"])

    # Compute map bounds from wall data
    blocking = map_data.get("blocking_lines", [])
    nonblocking = map_data.get("nonblocking_lines", [])
    all_lines = blocking + nonblocking
    if all_lines:
        all_x = [l[0] for l in all_lines] + [l[2] for l in all_lines]
        all_y = [l[1] for l in all_lines] + [l[3] for l in all_lines]
        xlo, xhi = min(all_x) - 50, max(all_x) + 50
        ylo, yhi = min(all_y) - 50, max(all_y) + 50
    else:
        xlo, xhi = -700, 1200
        ylo, yhi = -300, 1400

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)

    frames = []
    for t in range(T):
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.tick_params(colors=(0.35, 0.4, 0.45), labelsize=6)
        for spine in ax.spines.values():
            spine.set_color((0.25, 0.3, 0.35))
        ax.set_title(
            f"Top-Down  ·  Episode {ep_idx}  ·  step {t+1}/{T}",
            color="white", fontsize=10, pad=6,
        )

        # ── Draw map walls ──────────────────────────────────────────
        if blocking:
            wall_segs = [[(l[0], l[1]), (l[2], l[3])] for l in blocking]
            lc = mc.LineCollection(wall_segs, colors=(0.55, 0.6, 0.65, 0.8),
                                   linewidths=0.6, zorder=1)
            ax.add_collection(lc)

        if nonblocking:
            other_segs = [[(l[0], l[1]), (l[2], l[3])] for l in nonblocking]
            lc2 = mc.LineCollection(other_segs, colors=(0.3, 0.35, 0.4, 0.3),
                                    linewidths=0.3, zorder=0)
            ax.add_collection(lc2)

        # ── Draw agents ─────────────────────────────────────────────
        for aidx, aname in enumerate(agents):
            pos = positions[aname]
            color = AGENT_COLORS[aidx % len(AGENT_COLORS)]

            # Trail
            trail_start = max(0, t - 5)
            if t > 0:
                ax.plot(pos["x"][trail_start:t+1], pos["y"][trail_start:t+1],
                        "-", color=color, alpha=0.45, lw=2.0, zorder=2)

            cx, cy = pos["x"][t], pos["y"][t]

            # Heading arrow
            ang_rad = math.radians(pos["angle"][t])
            dx = math.cos(ang_rad) * 60
            dy = math.sin(ang_rad) * 60
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                mutation_scale=14),
                zorder=4,
            )

            # Agent marker
            ax.plot(cx, cy, "o", color=color, markersize=10, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.text(cx + 15, cy + 15, f"A{aidx}", fontsize=8, fontweight="bold",
                    color=color, zorder=6)

        # ── Draw armor objects ───────────────────────────────────────
        if objects_per_step and t < len(objects_per_step):
            for obj in objects_per_step[t]:
                ox, oy = obj[0], obj[1]
                ax.plot(ox, oy, "s", color="#00FFCC", markersize=7, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.5)

        # ── Rasterize ────────────────────────────────────────────────
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(img.reshape(h, w, 4)[..., :3].copy())

    plt.close(fig)

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                            quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  [topdown] Saved: {out_path}  ({T} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparative POV video
# ─────────────────────────────────────────────────────────────────────────────

def render_compare_pov_video(
    frames_a: List[np.ndarray],
    frames_b: List[np.ndarray],
    label_a: str,
    label_b: str,
    out_path: Path,
    fps: int = 15,
):
    """Render two POV recordings (lists of HxWxC frames) stacked vertically with labels."""
    import imageio
    import cv2

    T = min(len(frames_a), len(frames_b))
    width = max(frames_a[0].shape[1], frames_b[0].shape[1])

    def _row(frame: np.ndarray, label: str) -> np.ndarray:
        pad = width - frame.shape[1]
        if pad:
            frame = np.pad(frame, ((0, 0), (0, pad), (0, 0)))
        bar = np.zeros((32, width, 3), dtype=np.uint8)
        cv2.putText(bar, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        return np.concatenate([bar, frame[..., :3]], axis=0)

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                            quality=8) as writer:
        for t in range(T):
            writer.append_data(np.concatenate(
                [_row(frames_a[t], label_a), _row(frames_b[t], label_b)], axis=0,
            ))
    print(f"  [compare-pov] Saved: {out_path}  ({T} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparative top-down video
# ─────────────────────────────────────────────────────────────────────────────

def render_compare_topdown_video(
    positions_a: Dict[str, Dict],
    positions_b: Dict[str, Dict],
    map_data: Dict,
    label_a: str,
    label_b: str,
    out_path: Path,
    ep_idx: int,
    fps: int = 15,
    objects_a: Optional[List] = None,
    objects_b: Optional[List] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import imageio

    AGENT_COLORS = ["#FFD700", "#FF4FB8", "#00C8FF", "#FF6B35"]
    BG    = (0.05, 0.07, 0.09)
    PANEL = (0.10, 0.13, 0.18)

    agents = list(positions_a.keys())
    T_a = len(positions_a[agents[0]]["x"])
    T_b = len(positions_b[agents[0]]["x"])
    T = min(T_a, T_b)

    blocking = map_data.get("blocking_lines", [])
    nonblocking = map_data.get("nonblocking_lines", [])
    all_lines = blocking + nonblocking
    if all_lines:
        all_x = [l[0] for l in all_lines] + [l[2] for l in all_lines]
        all_y = [l[1] for l in all_lines] + [l[3] for l in all_lines]
        xlo, xhi = min(all_x) - 50, max(all_x) + 50
        ylo, yhi = min(all_y) - 50, max(all_y) + 50
    else:
        xlo, xhi = -700, 1200
        ylo, yhi = -300, 1400

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)

    def _draw_panel(ax, positions, label, objects_per_step, t):
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.tick_params(colors=(0.35, 0.4, 0.45), labelsize=5)
        for spine in ax.spines.values():
            spine.set_color((0.25, 0.3, 0.35))
        ax.set_title(
            f"{label}  ·  step {t+1}/{T}",
            color="white", fontsize=9, pad=6,
        )

        if blocking:
            wall_segs = [[(l[0], l[1]), (l[2], l[3])] for l in blocking]
            lc = mc.LineCollection(wall_segs, colors=(0.55, 0.6, 0.65, 0.8),
                                   linewidths=0.6, zorder=1)
            ax.add_collection(lc)
        if nonblocking:
            other_segs = [[(l[0], l[1]), (l[2], l[3])] for l in nonblocking]
            lc2 = mc.LineCollection(other_segs, colors=(0.3, 0.35, 0.4, 0.3),
                                    linewidths=0.3, zorder=0)
            ax.add_collection(lc2)

        for aidx, aname in enumerate(agents):
            pos = positions[aname]
            color = AGENT_COLORS[aidx % len(AGENT_COLORS)]
            trail_start = max(0, t - 5)
            if t > 0:
                ax.plot(pos["x"][trail_start:t+1], pos["y"][trail_start:t+1],
                        "-", color=color, alpha=0.45, lw=2.0, zorder=2)
            cx, cy = pos["x"][t], pos["y"][t]
            ang_rad = math.radians(pos["angle"][t])
            dx = math.cos(ang_rad) * 60
            dy = math.sin(ang_rad) * 60
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                mutation_scale=14),
                zorder=4,
            )
            ax.plot(cx, cy, "o", color=color, markersize=10, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.text(cx + 15, cy + 15, f"A{aidx}", fontsize=8, fontweight="bold",
                    color=color, zorder=6)

        if objects_per_step and t < len(objects_per_step):
            for obj in objects_per_step[t]:
                ox, oy = obj[0], obj[1]
                ax.plot(ox, oy, "s", color="#00FFCC", markersize=6, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.5)

    frames = []
    for t in range(T):
        _draw_panel(ax_l, positions_a, label_a, objects_a, t)
        _draw_panel(ax_r, positions_b, label_b, objects_b, t)
        fig.suptitle(f"Episode {ep_idx}", color="white", fontsize=11, y=0.98)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(img.reshape(h, w, 4)[..., :3].copy())

    plt.close(fig)

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                            quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  [compare-topdown] Saved: {out_path}  ({T} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# Comparative eval
# ─────────────────────────────────────────────────────────────────────────────

def compare_main(args):
    ckpt_a, ckpt_b = (_existing_checkpoint(p) for p in args.compare)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / f"compare_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  doom-marl-cl  ·  comparative eval")
    print(f"{'='*60}")
    print(f"  A          : {ckpt_a}")
    print(f"  B          : {ckpt_b}")
    print(f"  episodes   : {args.episodes}")
    print(f"  seed       : {args.seed}")
    print(f"  device     : {device}")
    print(f"  output     : {out_dir}")
    print(f"{'='*60}\n")

    # Evaluated one after the other so only one set of ViZDoom processes
    # is alive at a time. Both use the same per-episode seeds.
    results = []
    for tag, ckpt in (("a", ckpt_a), ("b", ckpt_b)):
        print(f"[compare] Evaluating {tag.upper()}: {ckpt.parent.parent.name}")
        results.append(evaluate_checkpoint(ckpt, args, out_dir / tag, device))
    res_a, res_b = results
    label_a, label_b = f"A: {res_a['name']}", f"B: {res_b['name']}"

    for ep in range(args.episodes):
        ep_num = ep + 1
        if ep < len(res_a["pov_videos"]) and ep < len(res_b["pov_videos"]):
            render_compare_pov_video(
                _read_video(res_a["pov_videos"][ep]),
                _read_video(res_b["pov_videos"][ep]),
                label_a, label_b,
                out_dir / f"compare_pov_ep{ep_num}.mp4", fps=args.pov_fps,
            )
        if not args.no_topdown:
            (pos_a, objs_a), (pos_b, objs_b) = (
                res_a["trajectories"][ep], res_b["trajectories"][ep],
            )
            if pos_a and pos_b:
                render_compare_topdown_video(
                    pos_a, pos_b, res_a["map_data"], label_a, label_b,
                    out_dir / f"compare_topdown_ep{ep_num}.mp4", ep_num,
                    fps=args.topdown_fps, objects_a=objs_a, objects_b=objs_b,
                )

    for res in results:
        shutil.rmtree(res["experiment_dir"].parent, ignore_errors=True)

    rewards_a, rewards_b = res_a["rewards"], res_b["rewards"]
    print(f"\n{'='*60}")
    print(f"  Comparative eval complete")
    print(f"{'='*60}")
    print(f"  Episodes         : {args.episodes}")
    print(f"  {'':18s} {'A':>12s}  {'B':>12s}")
    print(f"  {'Reward mean':18s} {np.mean(rewards_a):12.2f}  {np.mean(rewards_b):12.2f}")
    print(f"  {'Reward max':18s} {np.max(rewards_a):12.2f}  {np.max(rewards_b):12.2f}")
    print(f"  {'Reward min':18s} {np.min(rewards_a):12.2f}  {np.min(rewards_b):12.2f}")
    print(f"  {'Reward std':18s} {np.std(rewards_a):12.2f}  {np.std(rewards_b):12.2f}")
    print(f"  Output dir       : {out_dir}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    if args is None:
        args = parse_args()

    checkpoint = resolve_checkpoint(args)
    exp_name   = checkpoint.parent.parent.name
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = args.out / f"{exp_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  doom-marl-cl  ·  eval")
    print(f"{'='*60}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  episodes   : {args.episodes}")
    print(f"  device     : {device}")
    print(f"  output     : {out_dir}")
    print(f"{'='*60}\n")

    res = evaluate_checkpoint(checkpoint, args, out_dir / "_run", device)

    for ep in range(args.episodes):
        ep_num = ep + 1
        if ep < len(res["pov_videos"]):
            pov_path = out_dir / f"pov_ep{ep_num}.mp4"
            shutil.move(str(res["pov_videos"][ep]), pov_path)
            print(f"  [pov] Saved: {pov_path}")
        positions, objects_per_step = res["trajectories"][ep]
        if positions and not args.no_topdown:
            render_topdown_video(
                positions, res["map_data"], out_dir / f"topdown_ep{ep_num}.mp4",
                ep_num, fps=args.topdown_fps, objects_per_step=objects_per_step,
            )

    shutil.rmtree(out_dir / "_run", ignore_errors=True)

    rewards = res["rewards"]
    print(f"\n{'='*60}")
    print(f"  Eval complete")
    print(f"  Episodes     : {len(rewards)}")
    if rewards:
        print(f"  Reward mean  : {np.mean(rewards):.2f}")
        print(f"  Reward max   : {np.max(rewards):.2f}")
        print(f"  Reward min   : {np.min(rewards):.2f}")
    print(f"  Output dir   : {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _args = parse_args()
    if _args.compare:
        compare_main(_args)
    else:
        main(_args)
