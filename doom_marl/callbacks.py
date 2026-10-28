from __future__ import annotations

import math
import re
import struct
import time
from pathlib import Path

import torch
from benchmarl.experiment.callback import Callback

from doom_marl.log import get_console, log_event, log_info


def _fmt_seconds(seconds: float) -> str:
    """Render a duration as a compact ``HH:MM:SS`` / ``MM:SS`` string."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _wad_path_from_config(config_file: str | Path | None) -> Path | None:
    if not config_file:
        return None
    cfg_path = Path(config_file)
    if not cfg_path.is_file():
        return None
    for line in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip().replace("_", "").lower() != "doomscenariopath":
            continue
        wad = Path(value.strip())
        if not wad.is_absolute():
            wad = cfg_path.parent / wad
        return wad
    return None


def _resolve_wad_path(scenario: str | None, config_file: str | Path | None) -> Path | None:
    wad_path = _wad_path_from_config(config_file)
    if wad_path and wad_path.is_file():
        return wad_path
    if not scenario:
        return None
    try:
        import vizdoom as vzd
    except Exception:
        return None
    wad_path = Path(vzd.scenarios_path) / f"{scenario}.wad"
    return wad_path if wad_path.is_file() else None


def _parse_wad_map(
    scenario: str | None,
    config_file: str | Path | None = None,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Parse UDMF TEXTMAP wall lines, matching eval.py's top-down renderer."""
    wad_path = _resolve_wad_path(scenario, config_file)
    if wad_path is None:
        return {"blocking_lines": [], "nonblocking_lines": []}

    with wad_path.open("rb") as f:
        data = f.read()
    if len(data) < 12:
        return {"blocking_lines": [], "nonblocking_lines": []}

    num_lumps = struct.unpack_from("<I", data, 4)[0]
    dir_offset = struct.unpack_from("<I", data, 8)[0]

    textmap = None
    for i in range(num_lumps):
        off = dir_offset + i * 16
        if off + 16 > len(data):
            break
        lump_off, lump_size = struct.unpack_from("<II", data, off)
        name = data[off + 8:off + 16].rstrip(b"\x00").decode("ascii", errors="replace")
        if name == "TEXTMAP":
            textmap = data[lump_off:lump_off + lump_size].decode("utf-8", errors="replace")
            break

    if textmap is None:
        return {"blocking_lines": [], "nonblocking_lines": []}

    blocks: dict[str, list[str]] = {}
    for match in re.finditer(r"(\w+)\s*(?://[^{]*)?\{([^}]+)\}", textmap):
        blocks.setdefault(match.group(1), []).append(match.group(2))

    vertices: list[tuple[float, float]] = []
    for body in blocks.get("vertex", []):
        xm = re.search(r"x\s*=\s*([-\d.]+)", body)
        ym = re.search(r"y\s*=\s*([-\d.]+)", body)
        if xm and ym:
            vertices.append((float(xm.group(1)), float(ym.group(1))))

    blocking: list[tuple[float, float, float, float]] = []
    nonblocking: list[tuple[float, float, float, float]] = []
    for body in blocks.get("linedef", []):
        v1m = re.search(r"v1\s*=\s*(\d+)", body)
        v2m = re.search(r"v2\s*=\s*(\d+)", body)
        if not v1m or not v2m:
            continue
        v1, v2 = int(v1m.group(1)), int(v2m.group(1))
        if v1 >= len(vertices) or v2 >= len(vertices):
            continue
        x1, y1 = vertices[v1]
        x2, y2 = vertices[v2]
        line = (x1, y1, x2, y2)
        if re.search(r"blocking\s*=\s*true", body):
            blocking.append(line)
        else:
            nonblocking.append(line)

    return {"blocking_lines": blocking, "nonblocking_lines": nonblocking}


class _BMOuterPbarStub:
    """Stand-in for ``benchmarl.experiment.experiment.tqdm`` that disables the
    outer progress bar and feeds iteration boundaries to an
    :class:`IterProgressDisplay` instead.

    BenchMARL's :meth:`Experiment._collection_loop` builds a single tqdm bar
    spanning the entire run and repeatedly calls ``set_description`` /
    ``update`` on it. Hijacking that constructor lets us replace the bar with
    structured per-iteration output (collect → train pbar → metrics tables)
    without forking the loop body.
    """

    callback_ref: "IterProgressDisplay | None" = None

    def __init__(self, initial=0, total=None, *args, **kwargs):
        self.n = int(initial or 0)
        self.total = int(total) if total is not None else None
        cb = _BMOuterPbarStub.callback_ref
        if cb is not None:
            cb._set_total_iters(self.total, self.n)

    def set_description(self, *_a, **_k):
        return None

    def update(self, n: int = 1):
        self.n += int(n)
        cb = _BMOuterPbarStub.callback_ref
        if cb is not None:
            cb._on_iter_completed()

    def close(self):
        return None

    def __iter__(self):
        return iter(())


def _install_outer_pbar_patch():
    """Replace BenchMARL's outer ``tqdm`` symbol with our stub. Idempotent."""
    import benchmarl.experiment.experiment as _bm

    if getattr(_bm.tqdm, "__name__", "") != "_BMOuterPbarStub":
        _bm.tqdm = _BMOuterPbarStub  # type: ignore[assignment]


class IterProgressDisplay(Callback):
    """Pretty per-iteration training progress.

    Replaces BenchMARL's monolithic outer tqdm bar with a structured layout:

    1. ``iter N/T ─ collecting experience  (ETA …)``
    2. once collection finishes, a tqdm bar over training minibatch updates
       (``n_optimizer_steps × num_minibatches × n_groups``)
    3. a Rich table of training loss metrics + an evaluation metrics table
       (when an evaluation ran this iter), then a one-line iter footer

    The outer bar is suppressed via :class:`_BMOuterPbarStub`, which forwards
    iteration boundaries to this callback's :meth:`_on_iter_completed`.
    """

    # (full metric key in the training tensordict, human-readable label)
    _TRAIN_LABELS = {
        "loss_objective":     "policy loss",
        "loss_critic":        "critic loss",
        "loss_entropy":       "entropy loss",
        "entropy":            "entropy",
        "clip_fraction":      "PPO clip fraction",
        "explained_variance": "explained variance",
        "kl_approx":          "approx KL divergence",
        "ESS":                "effective sample size",
        "loss":               "total loss",
    }

    def __init__(self):
        super().__init__()
        self._total_iters: int | None = None
        self._iter_idx: int = 1
        self._iter_start_t: float | None = None
        self._collect_start_t: float | None = None
        self._iter_times: list[float] = []
        self._train_pbar = None
        self._train_total: int = 0
        self._group_metrics: dict[str, dict[str, float]] = {}
        self._eval_metrics: dict[str, float] = {}
        self._collect_frames: int = 0
        self._collect_eps: int = 0
        self._mean_return: float = float("nan")
        self._eval_ran_this_iter: bool = False

        # Wire ourselves into the patched outer tqdm and install the patch.
        _BMOuterPbarStub.callback_ref = self
        _install_outer_pbar_patch()

    # ------------------------------------------------------------------
    # Hooks driven by _BMOuterPbarStub (outer "iteration counter" bar)
    # ------------------------------------------------------------------
    def _set_total_iters(self, total: int | None, initial: int):
        self._total_iters = total
        self._iter_idx = initial + 1
        self._iter_start_t = time.time()
        self._announce_collecting()

    def _on_iter_completed(self):
        if self._train_pbar is not None:
            try:
                self._train_pbar.close()
            except Exception:
                pass
            self._train_pbar = None

        iter_time = time.time() - (self._iter_start_t or time.time())
        self._iter_times.append(iter_time)
        self._print_iter_summary(iter_time)

        self._iter_idx += 1
        self._iter_start_t = time.time()
        self._group_metrics.clear()
        self._eval_metrics.clear()
        self._collect_frames = 0
        self._collect_eps = 0
        self._eval_ran_this_iter = False

        if self._total_iters is None or self._iter_idx <= self._total_iters:
            self._announce_collecting()

    def _eta_str(self) -> str:
        if not self._iter_times or self._total_iters is None:
            return ""
        window = self._iter_times[-10:]
        avg = sum(window) / len(window)
        remaining_iters = max(self._total_iters - (self._iter_idx - 1), 0)
        return _fmt_seconds(remaining_iters * avg)

    def _announce_collecting(self):
        from rich.text import Text

        self._collect_start_t = time.time()
        msg = Text()
        total_str = str(self._total_iters) if self._total_iters else "?"
        msg.append(f"iter {self._iter_idx}/{total_str}", style="bold cyan")
        msg.append("  collecting experience", style="cyan")
        eta = self._eta_str()
        if eta:
            msg.append(f"   eta {eta}", style="dim")
        get_console().print(msg)

    # ------------------------------------------------------------------
    # BenchMARL Callback hooks
    # ------------------------------------------------------------------
    def on_batch_collected(self, batch):
        elapsed = max(time.time() - (self._collect_start_t or time.time()), 1e-6)
        try:
            frames = int(batch.numel())
        except Exception:
            frames = 0
        self._collect_frames = frames

        try:
            done = None
            for k in (("next", "done"), ("next", "terminated"), ("next", "truncated")):
                if k in batch.keys(include_nested=True):
                    d = batch.get(k)
                    done = d if done is None else (done | d)
            if done is not None:
                self._collect_eps = int(done.any(dim=-1).sum().item())
        except Exception:
            pass

        try:
            self._mean_return = float(self.experiment.mean_return)
        except Exception:
            self._mean_return = float("nan")

        ret_s = (
            f"{self._mean_return:.3f}"
            if self._mean_return == self._mean_return
            else "nan"
        )
        get_console().print(
            f"  [dim]collected {frames:,} frames in "
            f"{_fmt_seconds(elapsed)} ({frames / elapsed:,.0f} fps), "
            f"episodes={self._collect_eps}, mean return={ret_s}[/dim]"
        )

        # Build the training pbar over total minibatch updates this iter.
        from tqdm import tqdm

        e = self.experiment
        try:
            on_policy = e.on_policy
            n_opt = int(e.config.n_optimizer_steps(on_policy))
            bs = int(e.config.train_batch_size(on_policy))
            mb = int(e.config.train_minibatch_size(on_policy))
            n_mb = max(1, -(-bs // mb))
            n_groups = max(1, len(e.train_group_map))
            self._train_total = n_opt * n_mb * n_groups
        except Exception:
            self._train_total = 0

        if self._train_total > 0:
            self._train_pbar = tqdm(
                total=self._train_total,
                desc=f"  iter {self._iter_idx} training",
                leave=False,
                dynamic_ncols=True,
            )

    def on_train_step(self, batch, group):
        if self._train_pbar is not None:
            try:
                self._train_pbar.update(1)
            except Exception:
                pass
        return None

    def on_train_end(self, training_td, group):
        td = training_td.detach()
        keys = set(td.keys())
        captured: dict[str, float] = {}
        for full, _label in self._TRAIN_LABELS.items():
            if full in keys:
                try:
                    captured[full] = float(td.get(full).mean().item())
                except Exception:
                    pass
        self._group_metrics[group] = captured

    _SOLO_MIN = 0.5
    _COOP_MIN = 1.5

    def on_evaluation_end(self, rollouts):
        self._eval_ran_this_iter = True
        try:
            import torch

            returns: list[float] = []
            lens: list[int] = []
            total_solo = 0
            total_coop = 0
            for td in rollouts:
                lens.append(int(td.batch_size[0]))
                reward = None
                for key in (
                    ("next", "agents", "reward"),
                    ("next", "agent", "reward"),
                    ("next", "reward"),
                ):
                    if key in td.keys(include_nested=True):
                        reward = td.get(key)
                        break
                if reward is None:
                    continue
                # Sum over time, then mean over agent dim if present.
                summed = reward.sum(0)
                returns.append(float(summed.mean().item()))

                # Pickup classification from reward magnitudes.
                r = reward.detach().to(torch.float32)
                if r.size(-1) == 1:
                    r = r.squeeze(-1)
                if r.dim() >= 3:
                    r = r.amax(dim=-1)
                r_flat = r.reshape(-1)
                coop_mask = r_flat >= self._COOP_MIN
                solo_mask = (r_flat >= self._SOLO_MIN) & (~coop_mask)
                total_solo += int(solo_mask.sum().item())
                total_coop += int(coop_mask.sum().item())

            if returns:
                t = torch.tensor(returns)
                self._eval_metrics["episode return (mean)"] = float(t.mean().item())
                self._eval_metrics["episode return (min)"]  = float(t.min().item())
                self._eval_metrics["episode return (max)"]  = float(t.max().item())
                self._eval_metrics["evaluation episodes"]   = float(len(returns))
            if lens:
                self._eval_metrics["episode length (mean)"] = sum(lens) / len(lens)

            total_pickups = total_solo + total_coop
            if total_pickups > 0:
                self._eval_metrics["solo pickups"] = float(total_solo)
                self._eval_metrics["coop pickups"] = float(total_coop)
                self._eval_metrics["total pickups"] = float(total_pickups)
                self._eval_metrics["coop rate"] = total_coop / total_pickups
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-iter summary rendering
    # ------------------------------------------------------------------
    def _print_iter_summary(self, iter_time: float):
        from rich.table import Table

        console = get_console()

        if self._group_metrics:
            tbl = Table(
                title=f"iter {self._iter_idx} training metrics",
                title_style="bold magenta",
                header_style="bold",
                show_lines=False,
                expand=False,
            )
            tbl.add_column("metric", style="cyan", no_wrap=True)
            groups = list(self._group_metrics.keys())
            for g in groups:
                tbl.add_column(g, justify="right")
            present_keys = [
                k for k in self._TRAIN_LABELS
                if any(k in self._group_metrics[g] for g in groups)
            ]
            for k in present_keys:
                row = [self._TRAIN_LABELS[k]]
                for g in groups:
                    v = self._group_metrics[g].get(k)
                    row.append("—" if v is None else f"{v:.4g}")
                tbl.add_row(*row)
            console.print(tbl)

        if self._eval_metrics:
            tbl = Table(
                title=f"iter {self._iter_idx} evaluation metrics",
                title_style="bold green",
                header_style="bold",
                expand=False,
            )
            tbl.add_column("metric", style="cyan", no_wrap=True)
            tbl.add_column("value", justify="right")
            for k, v in self._eval_metrics.items():
                tbl.add_row(k, f"{v:.4g}")
            console.print(tbl)

        e = self.experiment
        ret_s = (
            f"{self._mean_return:.3f}"
            if self._mean_return == self._mean_return
            else "nan"
        )
        eta = self._eta_str()
        footer = (
            f"  [dim]iter {self._iter_idx} done in {_fmt_seconds(iter_time)} "
            f"· total frames={getattr(e, 'total_frames', 0):,} "
            f"· mean return={ret_s}"
        )
        if eta:
            footer += f" · eta {eta}"
        footer += "[/dim]"
        console.print(footer)


# Backward-compat alias: older callers / configs may still import the old
# name. ``VerboseProgressBar`` now points at the new structured display.
VerboseProgressBar = IterProgressDisplay


class TrajectoryWandbLogger(Callback):
    """Plot agent x/y trajectories from eval rollouts and log to W&B.

    Fires on ``on_evaluation_end`` every ``every_n_evals`` evaluation calls
    (so the natural cadence is ``eval.every_n_iters * every_n_evals``
    training iterations). Reads position tensors from the first eval
    rollout's TensorDict and uploads both a matplotlib image and a point
    table. Logging through W&B media/table history avoids depending on any
    special info tab. Episode terminations within the rollout are marked
    with stars.
    """

    def __init__(
        self,
        every_n_evals: int = 1,
        env_idx: int = 0,
        scenario: str | None = None,
        config_file: str | Path | None = None,
        x_key: str = "POSITION_X",
        y_key: str = "POSITION_Y",
        object_x_key: str = "OBJECT_X",
        object_y_key: str = "OBJECT_Y",
        object_present_key: str = "OBJECT_PRESENT",
        log_key: str = "trajectory/eval",
        agent_bounds: tuple[float, float, float, float] | None = (-320.0, 352.0, 128.0, 1024.0),
        spawn_bounds: tuple[float, float, float, float] | None = (-608.0, 1056.0, -224.0, 1312.0),
    ):
        super().__init__()
        self.every = max(1, int(every_n_evals))
        self._eval_count: int = 0
        self.env_idx = int(env_idx)
        self.scenario = scenario
        self.config_file = config_file
        self.x_key = x_key
        self.y_key = y_key
        self.object_x_key = object_x_key
        self.object_y_key = object_y_key
        self.object_present_key = object_present_key
        self.log_key = log_key
        self.agent_bounds = agent_bounds
        self.spawn_bounds = spawn_bounds
        self._map_data: dict[str, list[tuple[float, float, float, float]]] | None = None
        self._map_announced = False
        self._announced = False
        self._missing_announced = False

    @staticmethod
    def _draw_rect(ax, bounds, **kwargs):
        x_min, x_max, y_min, y_max = bounds
        xs = [x_min, x_max, x_max, x_min, x_min]
        ys = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(xs, ys, **kwargs)

    def _get_tensor(self, batch, candidates):
        for key in candidates:
            try:
                value = batch.get(key, default=None)
            except Exception:
                value = None
            if value is not None:
                return value, key
        return None, None

    def _get_positions(self, batch):
        x, x_source = self._get_tensor(
            batch,
            [
                ("agent", self.x_key),
                ("next", "agent", self.x_key),
                ("agent", "info", self.x_key),
                ("next", "agent", "info", self.x_key),
            ],
        )
        y, y_source = self._get_tensor(
            batch,
            [
                ("agent", self.y_key),
                ("next", "agent", self.y_key),
                ("agent", "info", self.y_key),
                ("next", "agent", "info", self.y_key),
            ],
        )
        if x is None or y is None:
            return None, None, None
        return x, y, (x_source, y_source)

    def _get_objects(self, batch):
        x, _x_source = self._get_tensor(
            batch,
            [
                ("agent", self.object_x_key),
                ("next", "agent", self.object_x_key),
                ("agent", "info", self.object_x_key),
                ("next", "agent", "info", self.object_x_key),
            ],
        )
        y, _y_source = self._get_tensor(
            batch,
            [
                ("agent", self.object_y_key),
                ("next", "agent", self.object_y_key),
                ("agent", "info", self.object_y_key),
                ("next", "agent", "info", self.object_y_key),
            ],
        )
        present, _present_source = self._get_tensor(
            batch,
            [
                ("agent", self.object_present_key),
                ("next", "agent", self.object_present_key),
                ("agent", "info", self.object_present_key),
                ("next", "agent", "info", self.object_present_key),
            ],
        )
        if x is None or y is None:
            return None, None, None
        return x, y, present

    def _to_env_time_agent(self, tensor):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    def _to_env_time_agent_object(self, tensor):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr

    def _get_map_data(self):
        if self._map_data is None:
            self._map_data = _parse_wad_map(self.scenario, self.config_file)
            if not self._map_announced:
                n_block = len(self._map_data.get("blocking_lines", []))
                n_other = len(self._map_data.get("nonblocking_lines", []))
                if n_block or n_other:
                    log_event(
                        "trajectory", "loaded",
                        blocking_lines=n_block, other_lines=n_other,
                    )
                else:
                    log_event(
                        "trajectory", "skipped",
                        reason="no WAD map geometry found",
                    )
                self._map_announced = True
        return self._map_data

    @staticmethod
    def _unique_object_points(obj_x, obj_y, obj_present, env_idx: int):
        if obj_x is None or obj_y is None or obj_x.ndim != 4 or obj_y.ndim != 4:
            return []
        if not 0 <= env_idx < obj_x.shape[0]:
            return []
        if obj_present is None:
            mask = (obj_x != 0.0) | (obj_y != 0.0)
        else:
            mask = obj_present > 0.5
        points = {}
        e = env_idx
        for t in range(obj_x.shape[1]):
            for m in range(obj_x.shape[3]):
                if not bool(mask[e, t, :, m].any()):
                    continue
                x_val = float(obj_x[e, t, 0, m])
                y_val = float(obj_y[e, t, 0, m])
                points[(round(x_val, 1), round(y_val, 1))] = (x_val, y_val)
        return list(points.values())

    @staticmethod
    def _draw_lines(ax, lines, *, color, linewidth, alpha, zorder):
        if not lines:
            return
        import matplotlib.collections as mc

        segments = [[(l[0], l[1]), (l[2], l[3])] for l in lines]
        ax.add_collection(
            mc.LineCollection(
                segments, colors=color, linewidths=linewidth, alpha=alpha, zorder=zorder
            )
        )

    def on_evaluation_end(self, rollouts):
        self._eval_count += 1
        if self._eval_count % self.every != 0:
            return
        if not rollouts:
            return
        # Eval rollouts: list of per-episode TensorDicts (one per eval episode).
        # We plot the first episode — with `evaluation_episodes=1` it's the
        # only one, and grid-plotting >1 would clutter the W&B image.
        batch = rollouts[0]
        it = getattr(self.experiment, "n_iters_performed", 0)

        if not self._announced:
            try:
                leaves = list(batch.keys(include_nested=True, leaves_only=True))
            except TypeError:
                leaves = list(batch.keys())
            log_info(
                f"all leaf keys in rollout batch ({len(leaves)}): "
                + ", ".join(str(k) for k in leaves),
                tag="trajectory",
            )
            self._announced = True

        x_tensor, y_tensor, _sources = self._get_positions(batch)
        if x_tensor is None or y_tensor is None:
            if not self._missing_announced:
                log_event(
                    "trajectory", "warning", level="warn",
                    reason="no position tensors found",
                    expected=f"('agent', '{self.x_key}')/('{self.y_key}') or info keys",
                )
                self._missing_announced = True
            return

        x = self._to_env_time_agent(x_tensor)
        y = self._to_env_time_agent(y_tensor)
        if x.ndim != 3:
            return
        E, T, A = x.shape
        if not 0 <= self.env_idx < E:
            return

        done = None
        try:
            d = batch.get(("next", "done")).detach().cpu().numpy()
            done = d.reshape(E, T, -1)[..., 0]
        except Exception:
            pass

        obj_x_tensor, obj_y_tensor, obj_present_tensor = self._get_objects(batch)
        obj_x = self._to_env_time_agent_object(obj_x_tensor) if obj_x_tensor is not None else None
        obj_y = self._to_env_time_agent_object(obj_y_tensor) if obj_y_tensor is not None else None
        obj_present = (
            self._to_env_time_agent_object(obj_present_tensor)
            if obj_present_tensor is not None
            else None
        )
        object_points = self._unique_object_points(obj_x, obj_y, obj_present, self.env_idx)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        e = self.env_idx
        map_data = self._get_map_data()
        blocking = map_data.get("blocking_lines", [])
        nonblocking = map_data.get("nonblocking_lines", [])

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
        ax.set_facecolor("#151b23")
        self._draw_lines(
            ax, nonblocking, color="#5f6b7a", linewidth=0.4, alpha=0.35, zorder=0
        )
        self._draw_lines(
            ax, blocking, color="#c9d1d9", linewidth=0.8, alpha=0.75, zorder=1
        )
        if self.spawn_bounds is not None:
            self._draw_rect(
                ax, self.spawn_bounds, lw=1.0, ls="--", color="#00ffcc", alpha=0.35,
                label="pickup spawn region",
            )
        if self.agent_bounds is not None:
            self._draw_rect(
                ax, self.agent_bounds, lw=1.2, color="#f2cc60", alpha=0.45,
                label="agent respawn zone",
            )
        if object_points:
            ox, oy = zip(*object_points)
            ax.scatter(
                ox, oy, marker="s", s=26, c="#00ffcc", alpha=0.85,
                edgecolor="white", linewidth=0.35, zorder=3,
                label=f"active reward items ({len(object_points)})",
            )

        # Vial pickup events: classified by reward value (ground truth).
        # ACS writes +1.0 for the first half of a pair, +2.0 for the
        # partner that completes it within DESPAWN_DELAY. The scenario's
        # global reward is mirrored to every agent so we collapse the
        # agent axis (max over A) — otherwise each pickup would star N
        # agents instead of one event. The marker is placed at the mean
        # agent position for that tic since the per-agent reward stream
        # carries no info about which agent physically picked the vial.
        try:
            r = batch.get(("next", "agent", "reward")).detach().cpu().numpy()
            if r.ndim == 4 and r.shape[-1] == 1:
                r = r.squeeze(-1)
            if r.ndim == 3:
                rv = r[e].max(axis=-1)  # (T,) one value per tic
                coord_mask = rv >= 1.5
                solo_mask = (rv > 0) & (rv < 1.5)
                mean_x = x[e].mean(axis=-1)
                mean_y = y[e].mean(axis=-1)
                if coord_mask.any():
                    ax.scatter(
                        mean_x[coord_mask], mean_y[coord_mask],
                        marker="*", s=160, c="#2ca02c",
                        edgecolor="k", linewidth=0.7, zorder=4,
                        label=f"coord pickup +2 ({int(coord_mask.sum())})",
                    )
                if solo_mask.any():
                    ax.scatter(
                        mean_x[solo_mask], mean_y[solo_mask],
                        marker="*", s=120, c="#ff7f0e",
                        edgecolor="k", linewidth=0.5, zorder=3,
                        label=f"solo pickup +1 ({int(solo_mask.sum())})",
                    )
        except Exception:
            pass

        for ag in range(A):
            ax.plot(x[e, :, ag], y[e, :, ag], lw=1.0, alpha=0.85, label=f"agent {ag}")
            ax.scatter(x[e, 0, ag], y[e, 0, ag], marker="o", s=60)
            ax.scatter(x[e, -1, ag], y[e, -1, ag], marker="X", s=80)
            if done is not None:
                for t_end in done[e].nonzero()[0]:
                    ax.scatter(
                        x[e, t_end, ag], y[e, t_end, ag],
                        marker="D", s=70, facecolor="white",
                        edgecolor="black", linewidth=1.0, zorder=4,
                    )
        all_x = list(x[e].reshape(-1))
        all_y = list(y[e].reshape(-1))
        for line in blocking + nonblocking:
            all_x.extend([line[0], line[2]])
            all_y.extend([line[1], line[3]])
        for bounds in (self.spawn_bounds, self.agent_bounds):
            if bounds is not None:
                all_x.extend([bounds[0], bounds[1]])
                all_y.extend([bounds[2], bounds[3]])
        for ox, oy in object_points:
            all_x.append(ox)
            all_y.append(oy)
        if all_x and all_y:
            margin = 75.0
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"iter {it} - env {e} - {T} steps "
            "(o=start, X=end, diamond=ep end, green star=+2 coord, orange star=+1 solo)"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2, color="white")

        try:
            import wandb
            if wandb.run is not None:
                table = wandb.Table(columns=["iter", "env", "step", "agent", "x", "y", "done"])
                for t in range(T):
                    step_done = bool(done[e, t]) if done is not None else False
                    for ag in range(A):
                        table.add_data(it, e, t, ag, float(x[e, t, ag]), float(y[e, t, ag]), step_done)
                payload = {
                    self.log_key: wandb.Image(fig),
                    f"{self.log_key}/points": table,
                }
                if object_points:
                    obj_table = wandb.Table(columns=["iter", "env", "object", "x", "y"])
                    for idx, (ox, oy) in enumerate(object_points):
                        obj_table.add_data(it, e, idx, float(ox), float(oy))
                    payload[f"{self.log_key}/objects"] = obj_table
                wandb.log(payload, commit=False)
        finally:
            plt.close(fig)


class PickupCountsCallback(Callback):
    """Log per-batch solo / coop pickup counts to W&B.

    Each rollout step's per-agent reward is classified by magnitude (the
    scenario emits +1.0 for the first vial of a pair and +2.0 for the
    partner that completes it; tiny shaping rewards stay below 0.5):
        reward in [solo_min, coop_min)  -> solo (first-of-pair) pickup
        reward >= coop_min              -> coop (completion) pickup

    Logged keys (committed alongside the experiment's normal W&B logs):
        pickups/solo_count  - total solo pickups across the batch
        pickups/coop_count  - total coop pickups across the batch
        pickups/total_count - solo + coop
        pickups/coop_rate   - coop / (solo + coop), nan if no pickups
        pickups/solo_per_episode - solo pickups divided by episodes ended
        pickups/coop_per_episode - coop pickups divided by episodes ended
    """

    def __init__(self, solo_min: float = 0.5, coop_min: float = 1.5):
        super().__init__()
        self.solo_min = float(solo_min)
        self.coop_min = float(coop_min)

    def on_batch_collected(self, batch):
        try:
            r_td = batch.get(("next", "agent", "reward"))
        except Exception:
            return
        try:
            import torch as _torch
        except Exception:
            return
        r = r_td.detach().to(_torch.float32)
        if r.dim() == 0:
            return
        # Reward shape is (E, T, A, 1) (or (E, T, A) without trailing dim).
        # The scenario's global ACS reward is mirrored to every agent, so
        # reducing across the agent axis with `amax` collapses each pickup
        # event to a single (env, time) cell — without this each pickup
        # would be counted once per agent.
        if r.size(-1) == 1:
            r = r.squeeze(-1)
        if r.dim() >= 3:
            r = r.amax(dim=-1)  # collapse agent axis: (E, T)
        r_flat = r.reshape(-1)
        coop_mask = r_flat >= self.coop_min
        solo_mask = (r_flat >= self.solo_min) & (~coop_mask)
        solo_count = int(solo_mask.sum().item())
        coop_count = int(coop_mask.sum().item())
        total = solo_count + coop_count

        episodes = 0
        try:
            done = None
            for k in (("next", "done"), ("next", "terminated"), ("next", "truncated")):
                if k in batch.keys(include_nested=True):
                    d = batch.get(k)
                    done = d if done is None else (done | d)
            if done is not None:
                episodes = int(done.any(dim=-1).sum().item())
        except Exception:
            pass

        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return

        coop_rate = float("nan") if total == 0 else coop_count / total
        payload = {
            "pickups/solo_count": solo_count,
            "pickups/coop_count": coop_count,
            "pickups/total_count": total,
            "pickups/coop_rate": coop_rate,
        }
        if episodes > 0:
            payload["pickups/solo_per_episode"] = solo_count / episodes
            payload["pickups/coop_per_episode"] = coop_count / episodes
            payload["pickups/total_per_episode"] = total / episodes
        try:
            wandb.log(payload, commit=False)
        except Exception:
            pass


class VideoWandbLogger(Callback):
    """Scan the rollout video directory and upload new mp4s to W&B.

    The mp4s are written from inside the eval env subprocess (by
    ``WorkerVideoLoggerParallelWrapper``). That subprocess cannot call
    ``wandb.log`` directly because ``wandb.init`` is only run in the main
    process. This callback runs in the main process and uploads on:
      * ``on_evaluation_end`` — natural moment, the eval that just produced
        the mp4 has finished.
      * ``on_batch_collected`` — safety net in case the writer hadn't
        flushed by the time eval-end fired.
    Uploads are deduped via ``_uploaded`` so double-firing is harmless.

    Filename convention (must match ``WorkerVideoLoggerParallelWrapper``):
        ``worker_<WW>_pid_<PID>_episode_<EP>.mp4``
    Logged W&B key: ``videos/worker_<WW>/episode_<EP>``
    """

    _MP4_RE = re.compile(
        r"^worker_(?P<worker>\d+)_pid_\d+_episode_(?P<episode>\d+)\.mp4$"
    )

    def __init__(self, video_dir: str | Path | None = None, fps: int = 35):
        super().__init__()
        self._explicit_video_dir = Path(video_dir) if video_dir else None
        self.fps = int(fps)
        self._uploaded: set[str] = set()

    @property
    def video_dir(self) -> Path | None:
        if self._explicit_video_dir is not None:
            return self._explicit_video_dir
        # Resolved lazily because VizdoomExperiment sets DOOM_VIDEO_DIR
        # during setup, which is after this callback is constructed.
        import os as _os
        env_dir = _os.environ.get("DOOM_VIDEO_DIR")
        return Path(env_dir) if env_dir else None

    def on_batch_collected(self, batch):
        self._scan_and_upload()

    def on_evaluation_end(self, rollouts):
        self._scan_and_upload()

    def _scan_and_upload(self):
        video_dir = self.video_dir
        if video_dir is None or not video_dir.is_dir():
            return
        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return

        for mp4 in sorted(video_dir.glob("worker_*_episode_*.mp4")):
            name = mp4.name
            if name in self._uploaded:
                continue
            match = self._MP4_RE.match(name)
            if not match:
                continue
            try:
                if mp4.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            worker_id = int(match.group("worker"))
            episode = int(match.group("episode"))
            key = f"videos/worker_{worker_id:02d}/episode_{episode:06d}"
            try:
                # ``fps`` is ignored by wandb.Video when given a file path
                # (the mp4 already carries its own frame rate), and passing
                # it triggers a wandb WARNING on every upload. Omit it.
                wandb.log(
                    {key: wandb.Video(str(mp4), format="mp4")},
                    commit=False,
                )
                self._uploaded.add(name)
                log_event("video", "uploaded", key=key, path=mp4)
            except Exception as exc:
                log_event(
                    "video", "failed", level="error",
                    op="wandb-upload", key=key, error=exc,
                )


class DebugDiagnostics(Callback):
    """Emit rich-coloured diagnostic prints to debug a non-learning pipeline.

    Checks observations, rewards, actions, advantages, value predictions,
    gradient norms, and policy output distributions. Uses colour coding:
      GREEN  = expectation match / healthy
      YELLOW = warning / suspicious
      RED    = expectation MISMATCH / bug likely
    """

    _TAG = "debug"

    def __init__(self, every_n_iters: int = 1):
        super().__init__()
        self._every = max(1, int(every_n_iters))
        self._iter_count = 0
        self._batch_count = 0

    @staticmethod
    def _ok(msg: str) -> None:
        get_console().print(f"  [bold green]✓ DIAG:[/bold green] [green]{msg}[/green]")

    @staticmethod
    def _warn(msg: str) -> None:
        get_console().print(f"  [bold yellow]⚠ DIAG:[/bold yellow] [yellow]{msg}[/yellow]")

    @staticmethod
    def _fail(msg: str) -> None:
        get_console().print(f"  [bold red]✗ DIAG:[/bold red] [red]{msg}[/red]")

    @staticmethod
    def _info(msg: str) -> None:
        get_console().print(f"  [bold magenta]● DIAG:[/bold magenta] [magenta]{msg}[/magenta]")

    @staticmethod
    def _stat(name: str, t: torch.Tensor) -> str:
        t = t.detach().float()
        return (
            f"{name}: shape={list(t.shape)}, "
            f"min={t.min().item():.4g}, max={t.max().item():.4g}, "
            f"mean={t.mean().item():.4g}, std={t.std().item():.4g}"
        )

    def _should_log(self) -> bool:
        return (self._batch_count % self._every) == 0

    # ------------------------------------------------------------------
    # on_batch_collected: inspect the rollout BEFORE training
    # ------------------------------------------------------------------
    def on_batch_collected(self, batch):
        self._batch_count += 1
        if not self._should_log():
            return
        console = get_console()
        console.rule("[bold magenta]── DEBUG BATCH DIAGNOSTICS ──", style="magenta")

        # 1. Batch structure
        try:
            self._info(f"batch shape: {list(batch.shape)}, numel: {batch.numel()}")
            top_keys = list(batch.keys())
            self._info(f"top-level keys: {top_keys}")
        except Exception as e:
            self._fail(f"batch introspection failed: {e}")

        # 2. Observation / state vector
        for obs_key in [
            ("agent", "state"),
            ("agent", "observation"),
            ("next", "agent", "state"),
            ("next", "agent", "observation"),
        ]:
            try:
                obs = batch.get(obs_key)
                if obs is not None:
                    self._info(self._stat(f"obs {obs_key}", obs))
                    flat = obs.float().reshape(-1)
                    n_zero = (flat == 0).sum().item()
                    pct_zero = 100.0 * n_zero / max(flat.numel(), 1)
                    if pct_zero > 95:
                        self._fail(
                            f"obs {obs_key} is {pct_zero:.1f}% zeros! "
                            "Policy likely sees no useful signal."
                        )
                    elif pct_zero > 50:
                        self._warn(f"obs {obs_key} is {pct_zero:.1f}% zeros")
                    else:
                        self._ok(f"obs {obs_key} has {pct_zero:.1f}% zeros — looks reasonable")

                    # Check if obs varies across batch (envs × time)
                    if obs.dim() >= 3:
                        env_var = obs.float().reshape(obs.shape[0], -1).var(dim=-1).mean().item()
                        time_var = obs.float().reshape(-1, obs.shape[-1]).var(dim=0).mean().item()
                        self._info(f"obs {obs_key} var across-env-features(mean)={env_var:.4g}, "
                                   f"across-time-per-feature(mean)={time_var:.4g}")
                        if env_var < 1e-8 and time_var < 1e-8:
                            self._fail("obs has NO variance — all identical! State is frozen.")
            except Exception:
                pass

        # 3. Reward
        for rkey in [("next", "agent", "reward"), ("next", "reward")]:
            try:
                r = batch.get(rkey)
                if r is not None:
                    self._info(self._stat(f"reward {rkey}", r))
                    nonzero_frac = (r.abs() > 1e-8).float().mean().item()
                    self._info(f"reward {rkey}: {nonzero_frac*100:.1f}% steps have non-zero reward")
                    if nonzero_frac < 0.001:
                        self._warn("reward is almost entirely zero — sparse signal")
                    break
            except Exception:
                pass

        # 4. Actions
        for akey in [("agent", "action")]:
            try:
                a = batch.get(akey)
                if a is not None:
                    self._info(self._stat(f"action {akey}", a))
                    if a.dtype in (torch.int64, torch.int32, torch.long):
                        uniq = a.unique()
                        self._info(f"action unique values ({len(uniq)}): {uniq.tolist()}")
                        if len(uniq) <= 1:
                            self._fail("only ONE unique action value — policy is stuck!")
                    break
            except Exception:
                pass

        # 5. Done / terminated / truncated signals
        for dkey in [
            ("next", "done"), ("next", "terminated"), ("next", "truncated"),
            ("next", "agent", "done"), ("next", "agent", "terminated"),
        ]:
            try:
                d = batch.get(dkey)
                if d is not None:
                    n_true = d.sum().item()
                    total = d.numel()
                    self._info(f"{dkey}: {n_true}/{total} True ({100*n_true/max(total,1):.1f}%)")
            except Exception:
                pass

        # 6. Log probs from collection policy
        for lp_key in [("agent", "log_prob"), ("agent", "sample_log_prob")]:
            try:
                lp = batch.get(lp_key)
                if lp is not None:
                    self._info(self._stat(f"log_prob {lp_key}", lp))
                    mean_lp = lp.mean().item()
                    n_actions = 16
                    uniform_lp = -math.log(n_actions)
                    if abs(mean_lp - uniform_lp) < 0.05:
                        self._warn(
                            f"mean log_prob={mean_lp:.4f} ≈ -ln({n_actions})={uniform_lp:.4f} "
                            "→ policy is UNIFORM (not learning)"
                        )
                    else:
                        self._ok(f"mean log_prob={mean_lp:.4f} differs from uniform={uniform_lp:.4f}")
                    break
            except Exception:
                pass

        # 7. Value predictions (if present from a prior forward pass)
        for vkey in [
            ("agent", "state_value"),
            ("next", "agent", "state_value"),
        ]:
            try:
                v = batch.get(vkey)
                if v is not None:
                    self._info(self._stat(f"value {vkey}", v))
                    break
            except Exception:
                pass

        # 8. Advantage (post process_batch, but might not be here yet)
        for akey in [("agent", "advantage"), ("advantage",)]:
            try:
                adv = batch.get(akey)
                if adv is not None:
                    self._info(self._stat(f"advantage {akey}", adv))
                    if adv.std().item() < 1e-8:
                        self._fail("advantage has ZERO variance — GAE broken or value function constant")
                    break
            except Exception:
                pass

        console.rule(style="magenta")

    # ------------------------------------------------------------------
    # on_train_step: inspect minibatch + gradients DURING training
    # ------------------------------------------------------------------
    def on_train_step(self, subdata, group):
        self._iter_count += 1
        # Only log on the first minibatch of each iter to avoid spam
        if self._iter_count > 1:
            return None
        if not self._should_log():
            return None

        console = get_console()
        console.rule("[bold cyan]── DEBUG TRAIN-STEP DIAGNOSTICS ──", style="cyan")

        # Advantage in the training minibatch
        for akey in [
            (group, "advantage"),
            ("advantage",),
            ("next", group, "advantage"),
        ]:
            try:
                adv = subdata.get(akey)
                if adv is not None:
                    self._info(self._stat(f"train advantage {akey}", adv))
                    adv_std = adv.std().item()
                    if adv_std < 1e-8:
                        self._fail(
                            "train advantage std ≈ 0! GAE is producing constant advantages. "
                            "Value function may be returning identical values for all states."
                        )
                    else:
                        self._ok(f"train advantage std={adv_std:.4g} — non-trivial")
                    break
            except Exception:
                pass

        # Value estimates in the training minibatch
        for vkey in [
            (group, "state_value"),
            ("next", group, "state_value"),
            (group, "value"),
            ("state_value",),
        ]:
            try:
                v = subdata.get(vkey)
                if v is not None:
                    self._info(self._stat(f"train value {vkey}", v))
                    v_std = v.std().item()
                    if v_std < 1e-6:
                        self._warn(f"train value std={v_std:.4g} — critic predicts near-constant!")
                    break
            except Exception:
                pass

        # Value target / returns
        for tkey in [
            (group, "value_target"),
            ("value_target",),
            (group, "returns"),
        ]:
            try:
                vt = subdata.get(tkey)
                if vt is not None:
                    self._info(self._stat(f"train value_target {tkey}", vt))
                    break
            except Exception:
                pass

        # Log-prob ratio (importance weight) — key PPO diagnostic
        for lp_key in [(group, "log_prob"), ("log_prob",)]:
            try:
                lp = subdata.get(lp_key)
                if lp is not None:
                    self._info(self._stat(f"train log_prob {lp_key}", lp))
                    break
            except Exception:
                pass

        # Policy output logits check (gradient norms are now captured by the
        # instrumented _optimizer_loop in VizdoomExperiment._on_setup, which
        # fires BEFORE zero_grad — see PRE-STEP GRADS diagnostics above).
        try:
            e = self.experiment
            for g_name, loss_obj in e.losses.items():
                actor = getattr(loss_obj, "actor_network", None) or getattr(loss_obj, "actor", None)
                if actor is not None:
                    self._check_policy_output(actor, subdata, group)
        except Exception as exc:
            self._warn(f"gradient inspection failed: {exc}")

        console.rule(style="cyan")
        return None

    def _check_policy_output(self, actor, subdata, group):
        """Forward a small slice through the actor to inspect logits."""
        try:
            with torch.no_grad():
                sample = subdata[:1] if subdata.dim() > 0 else subdata.unsqueeze(0)
                out = actor(sample.clone())
                for lkey in [
                    (group, "logits"),
                    (group, "action_value"),
                    ("logits",),
                ]:
                    try:
                        logits = out.get(lkey)
                        if logits is not None:
                            self._info(self._stat(f"policy logits {lkey}", logits))
                            probs = torch.softmax(logits.float(), dim=-1)
                            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
                            n_acts = logits.shape[-1]
                            max_entropy = math.log(n_acts)
                            ratio = entropy / max_entropy if max_entropy > 0 else 0
                            if ratio > 0.98:
                                self._fail(
                                    f"policy entropy={entropy:.4f} / max={max_entropy:.4f} "
                                    f"(ratio={ratio:.4f}) — nearly UNIFORM! "
                                    "Logits are too close to zero."
                                )
                            elif ratio > 0.90:
                                self._warn(
                                    f"policy entropy ratio={ratio:.4f} — still very uniform"
                                )
                            else:
                                self._ok(
                                    f"policy entropy={entropy:.4f}, ratio={ratio:.4f} — "
                                    "distribution is non-uniform"
                                )
                            logit_range = logits.max().item() - logits.min().item()
                            self._info(f"logit range={logit_range:.4g}")
                            if logit_range < 0.01:
                                self._fail(
                                    "logit range < 0.01 — network output is "
                                    "near-constant, cannot differentiate actions"
                                )
                            return
                    except Exception:
                        pass
        except Exception as exc:
            self._warn(f"policy output check failed: {exc}")

    # ------------------------------------------------------------------
    # on_train_end: post-epoch summary
    # ------------------------------------------------------------------
    def on_train_end(self, training_td, group):
        self._iter_count = 0
        if not self._should_log():
            return

        console = get_console()
        console.rule("[bold green]── DEBUG TRAIN-END DIAGNOSTICS ──", style="green")

        td = training_td.detach()

        # Check grad norms that BenchMARL logged
        for key in td.keys():
            k = str(key)
            if "grad_norm" in k:
                vals = td.get(key).float()
                self._info(f"{k}: mean={vals.mean().item():.6g}, "
                           f"min={vals.min().item():.6g}, max={vals.max().item():.6g}")

        # Weight statistics of actor/critic
        try:
            e = self.experiment
            for g_name, loss_obj in e.losses.items():
                actor = getattr(loss_obj, "actor_network", None) or getattr(loss_obj, "actor", None)
                critic = (
                    getattr(loss_obj, "critic_network", None)
                    or getattr(loss_obj, "critic", None)
                )
                for label, net in [("actor", actor), ("critic", critic)]:
                    if net is None:
                        continue
                    all_w = []
                    for p in net.parameters():
                        if p.requires_grad:
                            all_w.append(p.data.float().reshape(-1))
                    if all_w:
                        w = torch.cat(all_w)
                        self._info(
                            f"{g_name}/{label} weights: "
                            f"mean={w.mean().item():.6g}, std={w.std().item():.6g}, "
                            f"min={w.min().item():.6g}, max={w.max().item():.6g}, "
                            f"abs_mean={w.abs().mean().item():.6g}"
                        )
                        if w.abs().max().item() < 1e-6:
                            self._fail(f"{g_name}/{label}: all weights ≈ 0!")
                        elif w.std().item() < 1e-8:
                            self._fail(f"{g_name}/{label}: weight std ≈ 0 — collapsed!")
        except Exception as exc:
            self._warn(f"weight stats failed: {exc}")

        # Inspect the replay buffer content directly for advantage/value
        try:
            e = self.experiment
            buf = e.replay_buffers.get(group)
            if buf is not None:
                storage = buf.storage
                if hasattr(storage, '_storage'):
                    stored = storage._storage
                    for akey in [
                        (group, "advantage"),
                        ("advantage",),
                    ]:
                        try:
                            adv = stored.get(akey)
                            if adv is not None:
                                self._info(self._stat(f"buffer advantage {akey}", adv))
                                adv_std = adv.std().item()
                                if adv_std < 1e-8:
                                    self._fail(
                                        "buffer advantage std ≈ 0! "
                                        "GAE is broken — check value function, "
                                        "reward scale, or terminated/truncated signals."
                                    )
                                break
                        except Exception:
                            pass
                    for vkey in [
                        (group, "state_value"),
                        (group, "value_target"),
                    ]:
                        try:
                            v = stored.get(vkey)
                            if v is not None:
                                self._info(self._stat(f"buffer {vkey}", v))
                                break
                        except Exception:
                            pass
        except Exception:
            pass

        console.rule(style="green")
