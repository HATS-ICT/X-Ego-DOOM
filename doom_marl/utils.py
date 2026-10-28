"""Shared runtime and config helpers used by the experiment runner.

These utilities have **no dependency on benchmarl or torchrl** so they can be
imported before any heavy imports complete. Specifically, the warning filter
and signal handler installation should happen as early as possible in the
process lifecycle (before importing torchrl / vizdoom).
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Process lifecycle: terminal restore + ViZDoom orphan cleanup + signal handlers
# ---------------------------------------------------------------------------

def _restore_terminal():
    """ViZDoom/pygame leave the TTY in raw mode; ensure it's sane on exit."""
    try:
        if sys.stdin.isatty():
            os.system("stty sane 2>/dev/null")
    except Exception:
        pass


def _kill_orphan_vizdoom():
    """Reap vizdoom binaries owned by this user — they sometimes outlive their
    Python parents when SIGINT hits during a multiplayer net sync.

    The upstream ``kill_stale_vizdoom_processes`` prints a chatty
    ``"kill_stale_vizdoom_processes: killed N orphaned game process(es)"`` to
    stdout per call; we mute it and re-emit a single styled summary instead.
    """
    try:
        import contextlib
        import io

        from vizdoom.pettingzoo_wrapper.base_pettingzoo_env_shm import (
            kill_stale_vizdoom_processes,
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            killed = kill_stale_vizdoom_processes()
        if killed:
            try:
                from doom_marl.log import log_event

                log_event("vizdoom:cleanup", "killed", count=killed)
            except Exception:
                pass
    except Exception:
        pass


def _terminal_signal_handler(signum, _frame):
    _restore_terminal()
    _kill_orphan_vizdoom()
    raise SystemExit(128 + signum)


def install_runtime_hooks() -> None:
    """Install atexit + SIGINT/SIGTERM/SIGHUP handlers and silence noisy TorchRL warnings.

    Call this **once**, as the very first thing in the entry-point script,
    before importing torchrl / benchmarl / vizdoom.
    """
    os.environ.setdefault("RL_WARNINGS", "False")
    warnings.filterwarnings(
        "ignore",
        message=r"The default behavior of TransformedEnv will change.*",
        category=FutureWarning,
        module=r"torchrl\.envs\.transforms\.transforms",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"SyncDataCollector has been deprecated.*",
        category=DeprecationWarning,
        module=r"torchrl\.collectors\._base",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"total_frames .* is not exactly divisible by frames_per_batch.*",
        category=UserWarning,
        module=r"torchrl\.collectors\._single",
    )

    atexit.register(_restore_terminal)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _terminal_signal_handler)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Config loading: YAML + CLI --set overrides + candidate selection
# ---------------------------------------------------------------------------

def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``['key=value', ...]`` using ``yaml.safe_load`` for type conversion."""
    overrides: dict[str, Any] = {}
    for kv in pairs:
        if "=" not in kv:
            from doom_marl.log import log_warn

            log_warn(
                f"ignoring malformed --set value {kv!r} (expected KEY=VALUE)",
                tag="utils",
            )
            continue
        k, _, v = kv.partition("=")
        overrides[k.strip()] = yaml.safe_load(v.strip())
    return overrides


def load_cfg(yaml_path: Path, overrides: dict[str, Any]) -> DictConfig:
    """Load YAML via OmegaConf, apply CLI overrides (dotted paths supported), resolve ``train_device: auto``."""
    import torch

    cfg = OmegaConf.load(yaml_path)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=True)
    if "train_device" in cfg and cfg.train_device == "auto":
        cfg.train_device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def resolve_candidate_cfg(cfg: DictConfig, candidate: str) -> DictConfig:
    """Merge the selected candidate's algo config into the shared config.

    If no ``candidates`` block exists (legacy single-algo YAML), returns ``cfg``
    unchanged.
    """
    if "candidates" not in cfg:
        return cfg
    candidates = cfg.pop("candidates")
    if candidate not in candidates:
        raise ValueError(f"Unknown candidate {candidate!r}. Available: {list(candidates)}")
    return OmegaConf.merge(cfg, candidates[candidate])
