"""
Universal experiment runner.

Usage:
    python run.py <config.yaml> [--seed N] [--candidate NAME] [--set KEY=VALUE ...]

Examples:
    python run.py experiments/remedy_rush_coop_coord_pairs.yaml
    python run.py experiments/remedy_rush_coop_coord_pairs.yaml --candidate cecl
    python run.py experiments/remedy_rush_coop_coord_pairs.yaml --seed 2 --set num_envs=8

Config is loaded with OmegaConf and accessed strictly via attribute style
(``cfg.foo`` / ``cfg.foo.bar``); every key consumed below must be present in
the YAML — there are no implicit defaults.
"""

import sys
from argparse import ArgumentParser
from dataclasses import fields
from pathlib import Path

from doom_marl.utils import install_runtime_hooks

# Install warning filters + signal handlers BEFORE importing torchrl/vizdoom.
install_runtime_hooks()

import torch
from omegaconf import DictConfig, OmegaConf

from benchmarl.algorithms import (
    IppoConfig,
    IqlConfig,
    MappoConfig,
    QmixConfig,
    VdnConfig,
)
# MADDPG/IDDPG omitted: continuous-action only, incompatible with VizdoomTask
# (which exposes a Discrete action space — see doom_marl/task.py).
from benchmarl.experiment import ExperimentConfig
from benchmarl.models import CnnConfig, MlpConfig, SequenceModelConfig
from benchmarl.models.common import ModelConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.lstm import LstmConfig
from torch import nn

from doom_marl.algorithms import ContrastiveMappoConfig
from doom_marl.callbacks import (
    DebugDiagnostics,
    PickupCountsCallback,
    TrajectoryWandbLogger,
    VerboseProgressBar,
    VideoWandbLogger,
)
from doom_marl.experiment import VizdoomExperiment
from doom_marl.log import log_kv_block
from doom_marl.models import HFDinoV3Config, SmallResNetConfig
from doom_marl.task import VizdoomTask
from doom_marl.utils import (
    load_cfg,
    parse_set_overrides,
    resolve_candidate_cfg,
)
from scripts.build_scenario import ensure_built, resolve_base_scenario

# ---------------------------------------------------------------------------
# Algorithm registry — add custom algos from doom_marl/algorithms/ here
# ---------------------------------------------------------------------------
ALGOS = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "contrastive_mappo": ContrastiveMappoConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
    # "my_algo": MyAlgoConfig,
}

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _to_py(node):
    """Convert an OmegaConf node to a plain Python container (list/dict)."""
    return OmegaConf.to_container(node, resolve=True)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_algo_config(cfg: DictConfig):
    """Pick the right ``<algo>_kwargs`` block from the merged candidate cfg.

    Each candidate carries one ``<algo>_kwargs`` block per supported algorithm
    (``mappo_kwargs``, ``ippo_kwargs``, …) so swapping ``--set
    candidates.<name>.algo=ippo`` doesn't require also editing kwargs.
    ``contrastive_mappo`` is a thin wrapper around mappo + a contrastive
    auxiliary loss, so it reuses the candidate's ``mappo_kwargs`` and
    overlays a ``contrastive_kwargs`` block for the auxiliary terms.
    """
    algo_name = cfg.algo
    if algo_name not in ALGOS:
        raise ValueError(f"Unknown algo {algo_name!r}. Available: {list(ALGOS)}")
    algo_cls = ALGOS[algo_name]

    if algo_name == "contrastive_mappo":
        if "mappo_kwargs" not in cfg or "contrastive_kwargs" not in cfg:
            raise ValueError(
                "contrastive_mappo requires both 'mappo_kwargs' (base PPO config) "
                "and 'contrastive_kwargs' (auxiliary loss config) in the candidate."
            )
        kwargs = {**_to_py(cfg.mappo_kwargs), **_to_py(cfg.contrastive_kwargs)}
    else:
        kwargs_key = f"{algo_name}_kwargs"
        if kwargs_key not in cfg:
            raise ValueError(
                f"Missing {kwargs_key!r} in candidate config for algo {algo_name!r}."
            )
        kwargs = _to_py(cfg[kwargs_key])

    return algo_cls(**kwargs)


def _build_backbone(m: DictConfig) -> ModelConfig:
    """Build a single (non-recurrent) backbone ModelConfig from the YAML 'model' block."""
    arch = m.arch
    if arch == "mlp":
        # Pure MLP backbone — used for the `vision_only_obs: false` debug
        # mode, where the policy consumes the ground-truth state vector
        # built by `GroundTruthStateFromInfo` instead of pixels.
        return MlpConfig(
            num_cells=_to_py(m.mlp_cells),
            layer_class=nn.Linear,
            activation_class=nn.ReLU,
        )
    if arch in ("nature_cnn", "cnn", ""):
        return CnnConfig(
            cnn_num_cells=_to_py(m.cnn_cells),
            cnn_kernel_sizes=_to_py(m.cnn_kernels),
            cnn_strides=_to_py(m.cnn_strides),
            cnn_paddings=_to_py(m.cnn_paddings),
            cnn_activation_class=nn.ReLU,
            mlp_num_cells=_to_py(m.mlp_cells),
            mlp_layer_class=nn.Linear,
            mlp_activation_class=nn.ReLU,
        )

    if arch in ("small_resnet", "resnet_small", "torchvision_small_resnet"):
        return SmallResNetConfig(
            feature_dim=m.feature_dim,
            resnet_layers=_to_py(m.resnet_layers),
            stem_kernel_size=m.stem_kernel_size,
            stem_stride=m.stem_stride,
            stem_padding=m.stem_padding,
            use_maxpool=m.use_maxpool,
            mlp_num_cells=_to_py(m.mlp_cells),
            mlp_layer_class=nn.Linear,
            mlp_activation_class=nn.ReLU,
        )

    if arch in ("hf_dinov3", "dinov3", "dinov3_vits16"):
        return HFDinoV3Config(
            model_name=m.model_name,
            image_size=m.image_size,
            freeze_backbone=m.freeze_backbone,
            normalize=m.normalize,
            feature_source=m.feature_source,
            local_files_only=m.local_files_only,
            encoder_batch_size=m.encoder_batch_size,
            mlp_num_cells=_to_py(m.mlp_cells),
            mlp_layer_class=nn.Linear,
            mlp_activation_class=nn.ReLU,
        )

    raise ValueError(
        f"Unknown model arch {arch!r}. Available: mlp, nature_cnn, small_resnet, hf_dinov3. "
        f"For LSTM/temporal aggregation, set model.temporal.kind: lstm instead."
    )


def _build_lstm(rnn_cfg: DictConfig) -> LstmConfig:
    """Build a BenchMARL native LstmConfig head from a YAML rnn block."""
    return LstmConfig(
        hidden_size=rnn_cfg.hidden_size,
        n_layers=rnn_cfg.n_layers,
        bias=rnn_cfg.bias,
        dropout=rnn_cfg.dropout,
        compile=rnn_cfg.compile,
        mlp_num_cells=_to_py(rnn_cfg.mlp_num_cells),
        mlp_layer_class=nn.Linear,
        mlp_activation_class=nn.Tanh,
        mlp_activation_kwargs=None,
        mlp_norm_class=None,
        mlp_norm_kwargs=None,
    )


def _build_gru(rnn_cfg: DictConfig) -> GruConfig:
    """Build a BenchMARL native GruConfig head from a YAML rnn block."""
    return GruConfig(
        hidden_size=rnn_cfg.hidden_size,
        n_layers=rnn_cfg.n_layers,
        bias=rnn_cfg.bias,
        dropout=rnn_cfg.dropout,
        compile=rnn_cfg.compile,
        mlp_num_cells=_to_py(rnn_cfg.mlp_num_cells),
        mlp_layer_class=nn.Linear,
        mlp_activation_class=nn.Tanh,
        mlp_activation_kwargs=None,
        mlp_norm_class=None,
        mlp_norm_kwargs=None,
    )


def build_model_config(cfg: DictConfig) -> tuple[ModelConfig, ModelConfig]:
    """Build actor and critic model configs from the 'model' YAML section.

    Supports an explicit ``model.rnn_type`` override (``"none"`` / ``"lstm"``
    / ``"gru"``).  When absent, the presence of a ``model.lstm`` or
    ``model.gru`` block is used to auto-detect.  Off-policy algos do not
    currently support RNNs; only on-policy algos (MAPPO/IPPO) should use this.
    """
    m = cfg.model
    backbone = _build_backbone(m)

    rnn_type = OmegaConf.select(m, "rnn_type", default=None)
    if rnn_type == "none":
        return backbone, backbone

    if rnn_type == "gru":
        rnn_cfg = m.get("gru", m.get("lstm"))
        rnn = _build_gru(rnn_cfg)
    elif rnn_type == "lstm" or "lstm" in m:
        rnn = _build_lstm(m.lstm)
    elif "gru" in m:
        rnn = _build_gru(m.gru)
    else:
        return backbone, backbone

    backbone_out = m.mlp_cells[-1]
    seq = SequenceModelConfig(
        model_configs=[backbone, rnn],
        intermediate_sizes=[backbone_out],
    )
    return seq, seq


def build_exp_config(cfg: DictConfig, algo_cfg) -> ExperimentConfig:
    rollout_steps = int(cfg.rollout_steps)
    is_on_policy = type(algo_cfg).on_policy()

    exp_cfg = ExperimentConfig.get_from_yaml()

    overrides = {
        "sampling_device": cfg.sampling_device,
        "train_device": cfg.train_device,
        "buffer_device": cfg.buffer_device,
        "share_policy_params": True,
        "parallel_collection": cfg.parallel_collection,
        "max_n_frames": int(cfg.total_steps),
        "lr": float(cfg.lr),
        "gamma": float(cfg.gamma),
        "evaluation": True,
        "render": False,
        "evaluation_interval": rollout_steps * int(cfg.eval.every_n_iters),
        "evaluation_episodes": 1,
        "loggers": ["wandb", "csv"] if cfg.get("use_wandb", True) else ["csv"],
        "project_name": cfg.wandb_project,
        "save_folder": str(OUTPUTS_DIR / "checkpoints"),
        "checkpoint_interval": rollout_steps * int(cfg.checkpoint_every_n_iters),
        "checkpoint_at_end": True,
    }

    if is_on_policy:
        minibatch_size = int(cfg.minibatch_size)
        if minibatch_size > rollout_steps:
            raise ValueError(
                f"minibatch_size ({minibatch_size}) must be <= rollout_steps "
                f"({rollout_steps}) — otherwise only one minibatch runs per epoch."
            )
        overrides.update({
            "on_policy_collected_frames_per_batch": rollout_steps,
            "on_policy_n_envs_per_worker": int(cfg.num_envs),
            "on_policy_n_minibatch_iters": int(cfg.num_epochs),
            "on_policy_minibatch_size": minibatch_size,
        })
    else:
        overrides.update({
            "off_policy_collected_frames_per_batch": rollout_steps,
            "off_policy_n_envs_per_worker": int(cfg.num_envs),
        })
        if "n_optimizer_steps" in cfg:
            overrides["off_policy_n_optimizer_steps"] = int(cfg.n_optimizer_steps)
        if "train_batch_size" in cfg:
            overrides["off_policy_train_batch_size"] = int(cfg.train_batch_size)
        if "memory_size" in cfg:
            overrides["off_policy_memory_size"] = int(cfg.memory_size)
        if "init_random_frames" in cfg:
            overrides["off_policy_init_random_frames"] = int(cfg.init_random_frames)

    if "max_grad_norm" in cfg:
        overrides["clip_grad_norm"] = True
        overrides["clip_grad_val"] = float(cfg.max_grad_norm)

    valid = {f.name for f in fields(ExperimentConfig)}
    for k, v in overrides.items():
        if v is not None and k in valid:
            setattr(exp_cfg, k, v)

    # Keep evaluation interval aligned with the collection horizon
    h = (
        exp_cfg.on_policy_collected_frames_per_batch
        if is_on_policy
        else exp_cfg.off_policy_collected_frames_per_batch
    )
    if h and exp_cfg.evaluation_interval % h != 0:
        exp_cfg.evaluation_interval = ((exp_cfg.evaluation_interval + h - 1) // h) * h

    if cfg.get("use_wandb", True):
        exp_cfg.wandb_extra_kwargs = {
            **getattr(exp_cfg, "wandb_extra_kwargs", {}),
            "config": _to_py(cfg),
        }

    return exp_cfg


def build_task_config(cfg: DictConfig) -> dict:
    if "scenario_dir" in cfg and "variant" in cfg:
        config_file = str(ensure_built(cfg.scenario_dir, cfg.variant))
        scenario = resolve_base_scenario(cfg.scenario_dir)
        scenario_dir = cfg.scenario_dir
        variant = cfg.variant
    else:
        config_file = None
        scenario = cfg.scenario
        scenario_dir = ""
        variant = ""
    try:
        from vizdoom.pettingzoo_wrapper.utils import get_screen_resolution

        get_screen_resolution(cfg.resolution)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ViZDoom resolution {cfg.resolution!r}. "
            "Use a built-in ViZDoom ScreenResolution value such as "
            "'160X120', '320X240', or '512X384'. "
            "Set obs_h/obs_w for the model input resize."
        ) from exc
    return {
        "scenario": scenario,
        "scenario_dir": scenario_dir,
        "variant": variant,
        "config_file": config_file,
        "seed": int(cfg.seed),
        "num_agents": cfg.num_agents,
        "resolution": cfg.resolution,
        "skip_frames": cfg.skip_frames,
        "async_mode": cfg.async_mode,
        "render_mode": cfg.render_mode,
        "host_address": cfg.host_address,
        "netmode": cfg.netmode,
        "ticrate": cfg.ticrate,
        "enable_video": bool(cfg.eval.enable_video),
        "eval_video_every_n_evals": int(cfg.eval.video_every_n_evals),
        "video_fps": cfg.video_fps,
        "quiet_vizdoom_logs": cfg.quiet_vizdoom_logs,
        "vizdoom_init_progress": cfg.vizdoom_init_progress,
        "sampling_device": cfg.sampling_device,
        "daemon": cfg.daemon,
        "obs_h": cfg.obs_h,
        "obs_w": cfg.obs_w,
        "frame_stack": cfg.frame_stack,
        "timeout": int(cfg.timeout),
        "objects_info": bool(cfg.objects_info),
        "object_info_max_objects": int(cfg.object_info_max_objects),
        "action_set": cfg.action_set,
        "vision_only_obs": bool(cfg.vision_only_obs),
        "vizdoom_init_retries": int(
            OmegaConf.select(cfg, "vizdoom_init_retries", default=3)
        ),
        "vizdoom_init_retry_delay_s": float(
            OmegaConf.select(cfg, "vizdoom_init_retry_delay_s", default=2.0)
        ),
        "vizdoom_init_jitter_s": float(
            OmegaConf.select(cfg, "vizdoom_init_jitter_s", default=1.0)
        ),
        "vizdoom_port_min": int(OmegaConf.select(cfg, "vizdoom_port_min", default=20000)),
        "vizdoom_port_max": int(OmegaConf.select(cfg, "vizdoom_port_max", default=29999)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = ArgumentParser(description="Run a doom-marl-cl experiment from a YAML config.")
    ap.add_argument("config", type=Path, help="Path to experiment YAML config file")
    ap.add_argument("--seed", type=int, default=None, help="Override seed from YAML")
    ap.add_argument(
        "--candidate",
        default="baseline",
        metavar="NAME",
        help="Candidate to run (default: 'baseline'). E.g. --candidate cecl",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override any YAML key, e.g. --set num_envs=8 model.arch=nature_cnn",
    )
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"Config file not found: {args.config}")

    cli_overrides = parse_set_overrides(args.set or [])
    if args.seed is not None:
        cli_overrides["seed"] = args.seed

    cfg = load_cfg(args.config, cli_overrides)
    run_cfg = resolve_candidate_cfg(cfg, args.candidate)

    (OUTPUTS_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    algo_cfg = build_algo_config(run_cfg)
    model_cfg, critic_cfg = build_model_config(run_cfg)
    exp_cfg = build_exp_config(run_cfg, algo_cfg)
    log_kv_block(
        "run config",
        {
            "config":          args.config,
            "candidate":       args.candidate,
            "algo":            run_cfg.algo,
            "seed":            run_cfg.seed,
            "train_device":    exp_cfg.train_device,
            "sampling_device": exp_cfg.sampling_device,
            "buffer_device":   exp_cfg.buffer_device,
            "cuda_available":  torch.cuda.is_available(),
        },
        style="cyan",
    )
    task_config = build_task_config(run_cfg)
    task = VizdoomTask(task_config)

    seed = int(run_cfg.seed)
    eval_cfg = run_cfg.eval
    video_every_n_evals = int(eval_cfg.video_every_n_evals)
    traj_every_n_evals = int(eval_cfg.trajectory_plot_every_n_evals)
    if video_every_n_evals < 1 or traj_every_n_evals < 1:
        raise ValueError(
            "eval.video_every_n_evals and eval.trajectory_plot_every_n_evals "
            "must be >= 1 (multiples of eval.every_n_iters)."
        )
    use_wandb = cfg.get("use_wandb", True)
    callbacks = [
        VerboseProgressBar(),
    ]
    if cfg.get("debug_diagnostics", False):
        callbacks.append(DebugDiagnostics(every_n_iters=1))
    if use_wandb:
        callbacks.append(PickupCountsCallback())
    if eval_cfg.enable_trajectory_plot and use_wandb:
        callbacks.append(
            TrajectoryWandbLogger(
                every_n_evals=traj_every_n_evals,
                scenario=task_config["scenario"],
                config_file=task_config["config_file"],
            )
        )
    if eval_cfg.enable_video and use_wandb:
        callbacks.append(VideoWandbLogger(fps=int(run_cfg.video_fps)))
    extra_tags = _to_py(run_cfg.wandb_tags) if "wandb_tags" in run_cfg else []
    experiment = VizdoomExperiment(
        task=task,
        algorithm_config=algo_cfg,
        model_config=model_cfg,
        critic_model_config=critic_cfg,
        seed=seed,
        config=exp_cfg,
        experiment_name=args.config.stem,
        candidate=args.candidate,
        arch=run_cfg.model.arch,
        desync_warmup=bool(OmegaConf.select(run_cfg, "desync_warmup", default=False)),
        extra_tags=extra_tags,
        callbacks=callbacks,
    )

    experiment.run()
    experiment.close()


if __name__ == "__main__":
    main()
