import os
from datetime import datetime

import torch
import torch.nn as nn
from tensordict import TensorDict

import torchrl.record.loggers.utils as _torchrl_logger_utils

from doom_marl.log import get_console, log_kv_block, log_panel, log_warn


_TASK_ABBREV = (
    ("remedy_rush_coop_ultracoord", "rrcu"),
    ("remedy_rush_coop_coord_pairs", "rrcp"),
    ("remedy_rush_coop_coord", "rrcc"),
    ("remedy_rush_coop", "rrc"),
    ("remedy_rush_coord", "rrco"),
    ("remedy_rush", "rr"),
    ("health_gathering", "hg"),
    ("pitfall", "pf"),
)


def _abbrev_name(s: str) -> str:
    s = s.replace("_multi_agent", "")
    for full, short in _TASK_ABBREV:
        if full in s:
            s = s.replace(full, short)
            break
    return s


def _short_exp_name(model_name: str, experiment_name: str) -> str:
    parts = [_abbrev_name(model_name)]
    if experiment_name:
        parts.append(experiment_name)
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


_torchrl_logger_utils.generate_exp_name = _short_exp_name
import benchmarl.experiment.experiment as _benchmarl_exp_mod
_benchmarl_exp_mod.generate_exp_name = _short_exp_name


# benchmarl passes the run folder as CSV log_dir AND the run name as exp_name,
# and CSVLogger then joins them — producing an ugly <name>/<name>/ double nest.
# Skip the inner join so CSV files land directly in the run folder.
import os.path as _osp
from torchrl.record.loggers.csv import CSVLogger as _CSVLogger, CSVExperiment as _CSVExp


def _csv_create_experiment_flat(self):
    log_dir = str(self.log_dir)
    if _osp.basename(log_dir) != self.exp_name:
        log_dir = _osp.join(log_dir, self.exp_name)
    return _CSVExp(
        log_dir, video_format=self.video_format, video_fps=self.video_fps
    )


_CSVLogger._create_experiment = _csv_create_experiment_flat

from benchmarl.experiment import Experiment

BASE_ALGO = {
    "contrastivemappo": "mappo",
}


class VizdoomExperiment(Experiment):
    """
    Experiment subclass that injects structured W&B metadata per run.

      project  : set via ExperimentConfig.project_name (from YAML wandb_project)
      group    : clean scenario name (scenario_dir or scenario stripped of _multi_agent)
      job_type : candidate name (baseline, cecl, ...)
      name/id  : {experiment}_{candidate}_seed{S}_{timestamp}
      tags     : [experiment, scenario, candidate, algo, variant?, base_algo?, arch?]
    """

    def __init__(
        self,
        *args,
        experiment_name: str = "",
        candidate: str = "baseline",
        arch: str = "",
        desync_warmup: bool = False,
        extra_tags: list[str] | None = None,
        **kwargs,
    ):
        self._experiment_name = experiment_name
        self._candidate = candidate
        self._arch = arch
        self._desync_warmup = bool(desync_warmup)
        self._extra_tags = extra_tags or []
        super().__init__(*args, **kwargs)

    @staticmethod
    def _count_params(module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    @staticmethod
    def _shape_suffix(sub: nn.Module) -> str:
        """Return a short '[...]' suffix describing in/out dims of common layers.

        Empty string for layers without informative shape info (activations,
        wrappers, …) so the existing param/trainable columns still align.
        """
        # All attribute reads go through getattr so that classes which happen
        # to share a name with a torch.nn module but expose a different
        # attribute surface (e.g. BenchMARL/torchrl wrappers also called
        # ``LSTM``) don't crash the whole layer-tree render.
        def _fmt(parts: list[tuple[str, object | None]]) -> str:
            kept = [(k, v) for k, v in parts if v is not None]
            if not kept:
                return ""
            return "[" + ", ".join(f"{k}={v}" for k, v in kept) + "]"

        cls = sub.__class__.__name__
        g = lambda name: getattr(sub, name, None)  # noqa: E731

        if cls == "Linear":
            return _fmt([("in", g("in_features")), ("out", g("out_features"))])
        if cls in ("Conv1d", "Conv2d", "Conv3d",
                   "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"):
            return _fmt([
                ("in", g("in_channels")), ("out", g("out_channels")),
                ("k", g("kernel_size")), ("s", g("stride")), ("p", g("padding")),
            ])
        if cls in ("LSTM", "GRU", "RNN"):
            return _fmt([
                ("in", g("input_size")), ("hidden", g("hidden_size")),
                ("layers", g("num_layers")),
            ])
        if cls in ("LSTMCell", "GRUCell", "RNNCell"):
            return _fmt([("in", g("input_size")), ("hidden", g("hidden_size"))])
        if cls in ("MaxPool1d", "MaxPool2d", "MaxPool3d",
                   "AvgPool1d", "AvgPool2d", "AvgPool3d"):
            return _fmt([("k", g("kernel_size")), ("s", g("stride")), ("p", g("padding"))])
        if cls in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm"):
            num = g("num_features") or g("normalized_shape")
            return _fmt([("features", num)])
        if cls == "Embedding":
            return _fmt([("num", g("num_embeddings")), ("dim", g("embedding_dim"))])
        if cls == "Dropout":
            return _fmt([("p", g("p"))])
        return ""

    @staticmethod
    def _module_tree_text(module: nn.Module, max_depth: int = 8) -> str:
        """Walk named_modules() and render a layer tree.

        torchinfo trips over BenchMARL/torchrl modules whose attributes return
        TensorDict instances (``str_`` does ``if val`` → triggers
        TensorDictBase.__bool__ → RuntimeError). The default ``str(module)``
        path is also blocked: ``TensorDictModuleBase.__repr__`` is hardcoded
        to ``f"{ClassName}()"`` and hides children.

        We sidestep both by computing param counts ourselves and never calling
        ``str``/``bool`` on module attributes.
        """
        rows: list[tuple[str, str, str]] = [("Layer", "Params", "Trainable")]
        for qualname, sub in module.named_modules():
            depth = qualname.count(".") + (0 if qualname else 0)
            if not qualname:
                depth = 0
            else:
                depth = qualname.count(".") + 1
            if depth > max_depth:
                continue
            indent = "  " * depth
            label = f"{indent}{sub.__class__.__name__}"
            if qualname:
                leaf = qualname.rsplit(".", 1)[-1]
                label += f" ({leaf})"
            try:
                shape = VizdoomExperiment._shape_suffix(sub)
            except Exception:
                shape = ""
            if shape:
                label += f"  {shape}"
            # Direct (non-recursive) parameter count for this module
            direct_total = 0
            direct_trainable = 0
            for p in sub.parameters(recurse=False):
                direct_total += p.numel()
                if p.requires_grad:
                    direct_trainable += p.numel()
            recursive_total = sum(p.numel() for p in sub.parameters())
            recursive_trainable = sum(
                p.numel() for p in sub.parameters() if p.requires_grad
            )
            params_str = (
                f"{direct_total:,}"
                if direct_total == recursive_total
                else f"{direct_total:,} / {recursive_total:,}"
            )
            if recursive_total == 0:
                trainable_str = "--"
            elif recursive_trainable == recursive_total:
                trainable_str = "True"
            elif recursive_trainable == 0:
                trainable_str = "False"
            else:
                trainable_str = "Partial"
            rows.append((label, params_str, trainable_str))

        col0 = max(len(r[0]) for r in rows)
        col1 = max(len(r[1]) for r in rows)
        col2 = max(len(r[2]) for r in rows)
        sep = "─" * (col0 + col1 + col2 + 6)
        out = [sep, f"{rows[0][0]:<{col0}}  {rows[0][1]:>{col1}}  {rows[0][2]:>{col2}}", sep]
        for r in rows[1:]:
            out.append(f"{r[0]:<{col0}}  {r[1]:>{col1}}  {r[2]:>{col2}}")
        out.append(sep)
        return "\n".join(out)

    @staticmethod
    def _print_module_summary(name: str, module: nn.Module):
        from rich.console import Group
        from rich.text import Text

        total, trainable = VizdoomExperiment._count_params(module)
        header = Text()
        header.append("parameters: ", style="dim")
        header.append(f"total={total / 1_000_000:.2f}M ", style="bold cyan")
        header.append(f"({total:,}) ", style="dim")
        header.append(f"trainable={trainable / 1_000_000:.2f}M ", style="bold green")
        header.append(f"({trainable:,})", style="dim")

        try:
            arch_text = Text(VizdoomExperiment._module_tree_text(module))
        except Exception as exc:
            arch_text = Text(f"[layer tree failed: {exc!r}]", style="yellow")

        body = Group(header, Text(""), arch_text)
        log_panel(name, body, style="cyan")

    def _setup_name(self):
        super()._setup_name()
        # Point the in-env video logger at this run's folder so each run's
        # mp4s land alongside its checkpoints. Must happen before
        # _setup_collector spawns parallel env workers, which inherit env vars.
        os.environ["DOOM_VIDEO_DIR"] = str(self.folder_name / "videos")

    def _on_setup(self):
        super()._on_setup()
        console = get_console()
        console.rule("[bold magenta]neural network architecture", style="magenta")
        self._print_module_summary("collection policy", self.policy)
        for group, loss in self.losses.items():
            actor = getattr(loss, "actor_network", None) or getattr(loss, "actor", None)
            critic = (
                getattr(loss, "critic_network", None)
                or getattr(loss, "critic", None)
                or getattr(loss, "qvalue_network", None)
            )
            if actor is not None:
                self._print_module_summary(f"{group} actor for loss", actor)
            if critic is not None:
                self._print_module_summary(f"{group} critic", critic)
        console.rule(style="magenta")

    def _setup_collector(self):
        super()._setup_collector()
        self._install_contrastive_tagger()
        if not self._desync_warmup:
            return
        try:
            self._desynchronize_envs()
        except Exception as exc:
            log_warn(
                f"desync warmup failed, falling back to synchronized resets: {exc!r}",
                tag="VizdoomExperiment",
            )

    def _install_contrastive_tagger(self):
        """Attach a postproc that stamps each frame with (episode_id, step_id).

        These two int64 keys are read by the contrastive losses
        (ContrastiveMappo) to build gap-zone pair masks.
        episode_id comes from torchrl's ``("collector", "traj_ids")`` (already
        unique per trajectory). step_id is a per-env monotonically-increasing
        counter so the same episode straddling two off-policy rollouts still
        has contiguous step ids.

        Always-on: the cost is two int64 tensors per rollout; non-contrastive
        algos just ignore the keys.
        """
        collector = getattr(self, "collector", None)
        if collector is None:
            return

        env = getattr(collector, "env", None)
        n_envs = int(env.batch_size[0]) if env is not None and env.batch_size != () else 1
        env_step_counter = torch.zeros(n_envs, dtype=torch.long)

        existing_postproc = getattr(collector, "postproc", None)

        def _tag(batch):
            if existing_postproc is not None:
                batch = existing_postproc(batch)
            traj_ids = batch.get(("collector", "traj_ids"), default=None)
            if traj_ids is None:
                # Fall back to env-index as a synthetic episode id when the
                # collector does not expose traj_ids (e.g. exotic configs).
                shape = batch.shape
                if len(shape) == 0:
                    return batch
                ep = torch.arange(shape[0], dtype=torch.long).view(
                    shape[0], *([1] * (len(shape) - 1))
                ).expand(shape)
            else:
                ep = traj_ids.to(torch.long).reshape(batch.shape)

            shape = batch.shape
            if len(shape) >= 2:
                rollout_len = shape[-1]
                base = env_step_counter[: shape[0]].view(shape[0], *([1] * (len(shape) - 1)))
                step_ids = base + torch.arange(rollout_len, dtype=torch.long).view(
                    *([1] * (len(shape) - 1)), rollout_len
                )
                env_step_counter[: shape[0]] += rollout_len
            else:
                step_ids = torch.zeros(shape, dtype=torch.long)

            batch.set("contrastive_episode_id", ep)
            batch.set("contrastive_step_id", step_ids)
            return batch

        collector.postproc = _tag

    @torch.no_grad()
    def _desynchronize_envs(self):
        """Stagger per-env episode phase so terminal frames appear in every
        rollout instead of only every ``timeout / steps_per_env`` iterations.

        Per-env warmup length is chosen *adaptively* from the current run
        config so we waste no more frames than necessary.
        """
        collector = getattr(self, "collector", None)
        if collector is None:
            return
        env = getattr(collector, "env", None)
        carrier = getattr(collector, "_carrier", None)
        if env is None or carrier is None or env.batch_size == ():
            log_warn(
                "desync warmup skipped: collector exposes no batched env / carrier "
                "(likely collect_with_grad=true or unsupported torchrl version)",
                tag="VizdoomExperiment",
            )
            return

        n = int(env.batch_size[0])
        # `timeout` in the task config is in vizdoom *game tics* (it's passed
        # straight to `game.set_episode_timeout`). Convert to env steps so it's
        # comparable to `rollout_steps` / loop counters below.
        task_cfg = self.task.config or {}
        timeout_tics = int(task_cfg.get("timeout", self.max_steps or 0))
        skip_frames = max(1, int(task_cfg.get("skip_frames", 1)))
        timeout = max(0, timeout_tics // skip_frames)
        if n <= 1 or timeout <= 1:
            return

        on_policy = self.on_policy
        rollout_steps = int(self.config.collected_frames_per_batch(on_policy))
        steps_per_env_per_iter = max(1, rollout_steps // max(1, n))

        # If a single iter already spans a full episode per env, syncing isn't
        # an issue — skip warmup entirely.
        if steps_per_env_per_iter >= timeout:
            log_kv_block(
                "desync warmup",
                {
                    "status":               "skipped",
                    "reason":               "steps_per_env_per_iter >= timeout",
                    "num_envs":             n,
                    "rollout_steps":        rollout_steps,
                    "steps_per_env / iter": steps_per_env_per_iter,
                    "episode timeout":      timeout,
                },
                style="yellow",
            )
            return

        # Even-spaced offsets so every iter sees ~ rollout_steps / timeout
        # terminal frames, plus a single random shift for run-to-run variety.
        gen = torch.Generator(device="cpu").manual_seed(int(self.seed) + 9973)
        shift = int(torch.randint(0, timeout, (1,), generator=gen).item())
        stride = timeout / n
        offsets = torch.tensor(
            [int((i * stride + shift)) % timeout for i in range(n)],
            dtype=torch.long,
        )

        # We schedule a per-env reset at step (max_off - offsets[i]), then run
        # max_off random-policy steps. After the loop env i has been stepped
        # exactly offsets[i] times since its (scheduled or auto) last reset.
        max_off = int(offsets.max().item())
        if max_off <= 0:
            return
        reset_at = max_off - offsets  # shape [n]

        td = carrier
        # Preserve the bookkeeping subtree torchrl's collector adds to the
        # carrier (e.g. ("collector", "traj_ids")). reset()/step ops can drop
        # it depending on env wrappers; we re-attach it after the warmup loop.
        collector_data = (
            td.get("collector", default=None).clone()
            if td.get("collector", default=None) is not None
            else None
        )
        for k in range(max_off):
            mask = reset_at == k
            if bool(mask.any()):
                reset_td = TensorDict(
                    {"_reset": mask.to(td.device) if td.device is not None else mask},
                    batch_size=td.batch_size,
                )
                reset_out = env.reset(reset_td)
                td.update(reset_out)
            env.rand_action(td)
            _, td = env.step_and_maybe_reset(td)

        if collector_data is not None and td.get("collector", default=None) is None:
            td.set("collector", collector_data)
        collector._carrier = td

        log_kv_block(
            "desync warmup",
            {
                "status":               "applied",
                "num_envs":             n,
                "rollout_steps":        rollout_steps,
                "steps_per_env / iter": steps_per_env_per_iter,
                "episode timeout":      timeout,
                "warmup steps (max)":   max_off,
                "per-env offsets":      offsets.tolist(),
                "expected ep-ends/iter":
                    round(n * steps_per_env_per_iter / timeout, 2),
            },
            style="green",
        )

    def _setup_logger(self):
        task_cfg = self.task.config
        scenario = (
            task_cfg.get("scenario_dir")
            or task_cfg.get("scenario", "").replace("_multi_agent", "")
        )
        variant = task_cfg.get("variant")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Drop the scenario prefix from the experiment_name (typically the
        # YAML stem, e.g. ``remedy_rush_coop_coord_pairs_sequential``) so the
        # run name only carries the variant-distinguishing part. Inject the
        # algorithm name (mappo/ippo/...) so different algos on the same
        # variant don't collide visually in W&B.
        short_exp = self._experiment_name
        if scenario and short_exp.startswith(scenario + "_"):
            short_exp = short_exp[len(scenario) + 1:]
        elif scenario and short_exp == scenario:
            short_exp = ""
        parts = [p for p in (short_exp, self.algorithm_name, self._candidate) if p]
        run_name = "_".join(parts) + f"_seed{self.seed}_{timestamp}"

        tags = [self._experiment_name, scenario, self._candidate, self.algorithm_name]
        if variant:
            tags.append(variant)
        base = BASE_ALGO.get(self.algorithm_name)
        if base:
            tags.append(base)
        if self._arch:
            tags.append(self._arch)
        if self._extra_tags:
            tags.extend(self._extra_tags)

        extra = {
            "group":    scenario,
            "job_type": self._candidate,
            "id":       run_name,
            "name":     run_name,
            "tags":     tags,
        }

        original = self.config.wandb_extra_kwargs
        self.config.wandb_extra_kwargs = {**original, **extra}
        try:
            super()._setup_logger()
        finally:
            self.config.wandb_extra_kwargs = original
