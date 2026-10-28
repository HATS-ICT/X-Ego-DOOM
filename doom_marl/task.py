import json
import math
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from benchmarl.environments import TaskClass
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import Compose, EnvBase, RemoveEmptySpecs, Transform, TransformedEnv
from torchrl.envs.libs.pettingzoo import MarlGroupMapType, PettingZooWrapper
from torchrl.envs.transforms import CatFrames, SelectTransform

from vizdoom.pettingzoo_wrapper import make

from doom_marl.action_wrappers import (
    ForcedForwardTurnOnlyWrapper,
    LeftRightForwardStandWrapper,
    LeftRightForwardWrapper,
)
from doom_marl.log import log_event
from doom_marl.transforms import AHWCToTensorResize
from doom_marl.video import WorkerVideoLoggerParallelWrapper


@dataclass
class _PortReservation:
    port: int
    path: str
    fd: int | None

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def __del__(self) -> None:
        self.release()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_is_stale(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        pid = int(first) if first else -1
    except (OSError, ValueError):
        pid = -1

    if pid > 0 and not _pid_alive(pid):
        return True

    try:
        return time.time() - os.path.getmtime(path) > 24 * 60 * 60
    except OSError:
        return False


def _try_reserve_port_lock(host_address: str, port: int) -> _PortReservation | None:
    lock_dir = os.environ.get(
        "DOOM_PORT_LOCK_DIR",
        os.path.join(tempfile.gettempdir(), "doom_marl_ports"),
    )
    os.makedirs(lock_dir, exist_ok=True)
    safe_host = "".join(c if c.isalnum() else "_" for c in host_address)
    path = os.path.join(lock_dir, f"{safe_host}_{port}.lock")

    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if not _lock_is_stale(path):
                return None
            try:
                os.unlink(path)
            except OSError:
                return None
            continue
        except OSError:
            return None

        try:
            os.write(fd, f"{os.getpid()}\n{time.time():.6f}\n".encode("utf-8"))
        except OSError:
            pass
        return _PortReservation(port=port, path=path, fd=fd)

    return None


def _reserve_udp_port(host_address: str, cfg: Dict[str, Any]) -> _PortReservation:
    # ViZDoom multiplayer uses UDP. Keep our default outside Linux's common
    # ephemeral range (32768-60999) so W&B/client sockets do not race the game
    # host, and keep a small cross-process lock so our own workers do not pick
    # the same port between probe-close and VizDoom bind.
    port_min = int(os.environ.get("DOOM_PORT_MIN", cfg.get("vizdoom_port_min", 20000)))
    port_max = int(os.environ.get("DOOM_PORT_MAX", cfg.get("vizdoom_port_max", 29999)))
    if port_min > port_max:
        raise ValueError(f"Invalid ViZDoom port range: {port_min}-{port_max}")

    span = port_max - port_min + 1
    start = (os.getpid() * 2654435761 + time.time_ns()) % span
    for offset in range(span):
        port = port_min + ((start + offset) % span)
        reservation = _try_reserve_port_lock(host_address, port)
        if reservation is None:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind((host_address, port))
        except OSError:
            reservation.release()
            continue
        return reservation

    raise RuntimeError(
        f"No free UDP ports available for ViZDoom in {host_address}:{port_min}-{port_max}"
    )


def _release_port_when_closed(pz_env, reservation: _PortReservation):
    orig_close = pz_env.close

    def close(*args, **kwargs):
        try:
            return orig_close(*args, **kwargs)
        finally:
            reservation.release()

    pz_env.close = close
    return pz_env


@contextmanager
def _suppress_native_output(enabled: bool):
    """Temporarily silence native stdout/stderr writes from ViZDoom startup."""
    if not enabled:
        yield
        return

    import os
    import sys

    devnull_fd = None
    saved_fds = []
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):
            try:
                saved_fd = os.dup(fd)
                os.dup2(devnull_fd, fd)
                saved_fds.append((fd, saved_fd))
            except OSError:
                pass
        yield
    finally:
        for fd, saved_fd in reversed(saved_fds):
            try:
                os.dup2(saved_fd, fd)
            finally:
                os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def _attach_info_cache(pz_env):
    """Monkey-patch a PettingZoo ParallelEnv so it stashes the last info dict
    on itself as ``last_infos``. Subclassing the env would break TorchRL's
    ``isinstance(env, ParallelEnv)`` check in ``PettingZooWrapper``.

    Also locks the info schema: TorchRL's PettingZooWrapper builds its info
    spec from the keys present in the *first* reset. If the engine emits a
    superset on later resets/steps (e.g. ``get_flat_game_vars`` returns ``{}``
    when ``state is None`` on the first reset but the full var set later),
    the wrapper crashes with ``NoneType.__setitem__`` while filling info.
    We prevent that by locking the per-agent key set to whatever the first
    reset produced and stripping anything extra from subsequent calls.
    """
    pz_env.last_infos = {a: {} for a in getattr(pz_env, "agents", [])}
    orig_reset = pz_env.reset
    orig_step = pz_env.step
    locked_keys: dict[str, set[str]] = {}

    timeout = getattr(pz_env, "_timeout", None)

    def _strip_objects(info):
        # Variable-length 'objects' lists can't be tensor-allocated by torchrl;
        # we keep them in last_infos for downstream transforms.
        return {
            agent: {k: v for k, v in (agent_info or {}).items() if k != "objects"}
            for agent, agent_info in (info or {}).items()
        }

    def _lock_or_filter(info):
        info = _strip_objects(info)
        if not locked_keys:
            for agent, agent_info in info.items():
                locked_keys[agent] = set(agent_info.keys())
            return info
        filtered = {}
        for agent, agent_info in info.items():
            allowed = locked_keys.get(agent)
            if allowed is None:
                filtered[agent] = agent_info
                continue
            filtered[agent] = {
                k: (agent_info[k] if k in agent_info else 0.0) for k in allowed
            }
        return filtered

    def reset(*args, **kwargs):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                obs, info = orig_reset(*args, **kwargs)
                break
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        pz_env.last_infos = info
        return obs, _lock_or_filter(info)

    def step(actions):
        obs, r, term, trunc, info = orig_step(actions)
        pz_env.last_infos = info

        # The upstream ViZDoom PettingZoo wrapper sets `truncated = terminated`
        # whenever `is_episode_finished()` is True (engine-side timeout fires
        # the same flag as a real game-end via ACS). That conflates timeout
        # truncations with terminal states, and torchrl's GAE then zeroes the
        # bootstrap (V(s_T)=0) on every timeout — breaking value learning for
        # any scenario where episodes end primarily via timeout.
        #
        # In async_mode the game advances more tics per make_action call than
        # the nominal skip_frames, so a fixed frame counter underestimates the
        # real elapsed time. Since a configured timeout means the scenario is
        # designed to end via time limit (not agent death), we reclassify ALL
        # episode ends as truncations when timeout is set. Scenarios with true
        # terminal states (death, goal reached) should set timeout to None.
        if timeout is not None and any(term.values()):
            term = {a: False for a in term}
            trunc = {a: True for a in trunc}

        return obs, r, term, trunc, _lock_or_filter(info)

    pz_env.reset = reset
    pz_env.step = step
    return pz_env


def _log_vizdoom_init(
    cfg: Dict[str, Any],
    status: str,
    *,
    port: int,
    elapsed: float | None = None,
    role: str | None = None,
    attempt: str | None = None,
    error: BaseException | None = None,
):
    if not cfg.get("vizdoom_init_progress", True):
        return
    fields: Dict[str, Any] = {
        "seed": cfg["seed"],
        "agents": cfg["num_agents"],
        "resolution": cfg["resolution"],
        "obs": f"{cfg['obs_w']}x{cfg['obs_h']}",
        "port": port,
    }
    if role is not None:
        fields["role"] = role
    if attempt is not None:
        fields["attempt"] = attempt
    if elapsed is not None:
        fields["elapsed"] = f"{elapsed:.1f}s"
    if error is not None:
        fields["error"] = f"{type(error).__name__}: {error}"
    log_event("vizdoom:init", status, pid=True, **fields)


class GameVarsFromInfo(Transform):
    """Copy selected scalar fields out of the cached PettingZoo info dict into
    the rollout TensorDict under ``("agent", <var>)``.

    Reads ``last_infos`` off the underlying PZ env (attached via
    :func:`_attach_info_cache`), traversing ``parent.base_env._env`` to reach
    it.
    """

    def __init__(
        self,
        agent_keys: list[str],
        var_names: tuple[str, ...] = ("POSITION_X", "POSITION_Y", "ANGLE", "HEALTH"),
        group: str = "agent",
    ):
        out_keys = [(group, v) for v in var_names]
        super().__init__(in_keys=[], out_keys=out_keys)
        self._agents = list(agent_keys)
        self._vars = tuple(var_names)
        self._group = group

    def _last_infos(self) -> Dict[str, Dict[str, Any]]:
        env = self.parent
        while hasattr(env, "base_env"):
            env = env.base_env
        pz = getattr(env, "_env", None)
        if pz is None:
            return {}
        return getattr(pz, "last_infos", {}) or {}

    def _build_tensor(self, infos: Dict[str, Dict[str, Any]], var: str, device) -> torch.Tensor:
        vals = [float(infos.get(a, {}).get(var, 0.0)) for a in self._agents]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(-1)

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        infos = self._last_infos()
        if not infos:
            return next_tensordict
        device = next_tensordict.device
        for v in self._vars:
            next_tensordict.set((self._group, v), self._build_tensor(infos, v, device))
        return next_tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        infos = self._last_infos()
        if not infos:
            return tensordict_reset
        device = tensordict_reset.device
        for v in self._vars:
            tensordict_reset.set((self._group, v), self._build_tensor(infos, v, device))
        return tensordict_reset

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        n_agents = len(self._agents)
        for v in self._vars:
            obs_spec[(self._group, v)] = Unbounded(
                shape=torch.Size([n_agents, 1]),
                device=obs_spec.device,
                dtype=torch.float32,
            )
        return obs_spec


class ObjectPositionsFromInfo(Transform):
    """Copy active object coordinates from raw ViZDoom info into tensors."""

    def __init__(
        self,
        agent_keys: list[str],
        max_objects: int = 64,
        group: str = "agent",
    ):
        out_keys = [
            (group, "OBJECT_X"),
            (group, "OBJECT_Y"),
            (group, "OBJECT_PRESENT"),
        ]
        super().__init__(in_keys=[], out_keys=out_keys)
        self._agents = list(agent_keys)
        self._max_objects = int(max_objects)
        self._group = group

    def _last_infos(self) -> Dict[str, Dict[str, Any]]:
        env = self.parent
        while hasattr(env, "base_env"):
            env = env.base_env
        pz = getattr(env, "_env", None)
        if pz is None:
            return {}
        return getattr(pz, "last_infos", {}) or {}

    @staticmethod
    def _xy_from_object(obj) -> tuple[float, float] | None:
        if isinstance(obj, dict):
            x = obj.get("x", obj.get("POSITION_X", obj.get("pos_x")))
            y = obj.get("y", obj.get("POSITION_Y", obj.get("pos_y")))
            if x is None or y is None:
                return None
            return float(x), float(y)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            return float(obj[0]), float(obj[1])
        for x_attr, y_attr in (
            ("x", "y"),
            ("position_x", "position_y"),
            ("POSITION_X", "POSITION_Y"),
        ):
            if hasattr(obj, x_attr) and hasattr(obj, y_attr):
                return float(getattr(obj, x_attr)), float(getattr(obj, y_attr))
        return None

    def _objects(self, infos: Dict[str, Dict[str, Any]]) -> list[tuple[float, float]]:
        for agent in self._agents:
            raw_objects = infos.get(agent, {}).get("objects") or []
            points = []
            for obj in raw_objects[: self._max_objects]:
                xy = self._xy_from_object(obj)
                if xy is not None:
                    points.append(xy)
            if points:
                return points
        return []

    def _build_tensors(self, infos: Dict[str, Dict[str, Any]], device) -> dict[str, torch.Tensor]:
        n_agents = len(self._agents)
        shape = (n_agents, self._max_objects)
        xs = torch.zeros(shape, dtype=torch.float32, device=device)
        ys = torch.zeros(shape, dtype=torch.float32, device=device)
        present = torch.zeros(shape, dtype=torch.float32, device=device)
        for idx, (x, y) in enumerate(self._objects(infos)[: self._max_objects]):
            xs[:, idx] = x
            ys[:, idx] = y
            present[:, idx] = 1.0
        return {"OBJECT_X": xs, "OBJECT_Y": ys, "OBJECT_PRESENT": present}

    def _add_object_tensors(self, tensordict: TensorDictBase) -> TensorDictBase:
        infos = self._last_infos()
        if not infos:
            return tensordict
        for name, value in self._build_tensors(infos, tensordict.device).items():
            tensordict.set((self._group, name), value)
        return tensordict

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self._add_object_tensors(next_tensordict)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._add_object_tensors(tensordict_reset)

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        n_agents = len(self._agents)
        for name in ("OBJECT_X", "OBJECT_Y", "OBJECT_PRESENT"):
            obs_spec[(self._group, name)] = Unbounded(
                shape=torch.Size([n_agents, self._max_objects]),
                device=obs_spec.device,
                dtype=torch.float32,
            )
        return obs_spec


class GroundTruthStateFromInfo(Transform):
    """Build a normalized ground-truth state vector from cached PettingZoo info.

    Produces a single ``("agent", "state")`` tensor of shape
    ``[n_agents, 4 * n_agents + 3 * max_objects]``. Per-agent rows share the
    same global state (every agent sees every agent's pose + every object's
    pose). Used as a debug-friendly low-dimensional alternative to vision —
    drops the visual encoder out of the optimisation problem entirely so we
    can isolate whether the bottleneck is the CNN or the policy/value head.

    Layout per row:
        [agent_0_x/scale, agent_0_y/scale, sin(a_0), cos(a_0), ...,
         agent_{N-1}_x/scale, agent_{N-1}_y/scale, sin(a_{N-1}), cos(a_{N-1}),
         obj_0_x/scale, ..., obj_{M-1}_x/scale,
         obj_0_y/scale, ..., obj_{M-1}_y/scale,
         obj_0_present, ..., obj_{M-1}_present]

    Positions are normalised by ``coord_scale`` (default 1024 — matches the
    typical ACS map-bounds range). Angles are decomposed into ``sin/cos`` so
    the policy doesn't have to learn the wraparound at 0/360 degrees.
    """

    def __init__(
        self,
        agent_keys: list[str],
        max_objects: int = 64,
        coord_scale: float = 1024.0,
        group: str = "agent",
    ):
        out_keys = [(group, "state")]
        super().__init__(in_keys=[], out_keys=out_keys)
        self._agents = list(agent_keys)
        self._max_objects = int(max_objects)
        self._coord_scale = float(coord_scale)
        self._group = group
        self._n_agents = len(self._agents)
        self._state_dim = 4 * self._n_agents + 3 * self._max_objects

    def _last_infos(self) -> Dict[str, Dict[str, Any]]:
        env = self.parent
        while hasattr(env, "base_env"):
            env = env.base_env
        pz = getattr(env, "_env", None)
        if pz is None:
            return {}
        return getattr(pz, "last_infos", {}) or {}

    @staticmethod
    def _xy_from_object(obj) -> tuple[float, float] | None:
        return ObjectPositionsFromInfo._xy_from_object(obj)

    def _build_global_state(self, infos: Dict[str, Dict[str, Any]], device) -> torch.Tensor:
        scale = self._coord_scale
        vec = torch.zeros(self._state_dim, dtype=torch.float32, device=device)

        for i, agent in enumerate(self._agents):
            ai = infos.get(agent, {}) or {}
            x = float(ai.get("POSITION_X", 0.0)) / scale
            y = float(ai.get("POSITION_Y", 0.0)) / scale
            angle_deg = float(ai.get("ANGLE", 0.0))
            angle_rad = angle_deg * math.pi / 180.0
            base = 4 * i
            vec[base + 0] = x
            vec[base + 1] = y
            vec[base + 2] = math.sin(angle_rad)
            vec[base + 3] = math.cos(angle_rad)

        # Objects are global; the upstream PZ wrapper attaches the same list to
        # every agent's info — pick the first non-empty list.
        raw_objects: list = []
        for agent in self._agents:
            raw_objects = (infos.get(agent, {}) or {}).get("objects") or []
            if raw_objects:
                break

        obj_base = 4 * self._n_agents
        x_off = obj_base
        y_off = obj_base + self._max_objects
        p_off = obj_base + 2 * self._max_objects
        for idx, obj in enumerate(raw_objects[: self._max_objects]):
            xy = self._xy_from_object(obj)
            if xy is None:
                continue
            ox, oy = xy
            vec[x_off + idx] = ox / scale
            vec[y_off + idx] = oy / scale
            vec[p_off + idx] = 1.0

        return vec

    def _build_tensor(self, infos: Dict[str, Dict[str, Any]], device) -> torch.Tensor:
        global_state = self._build_global_state(infos, device)
        return global_state.unsqueeze(0).expand(self._n_agents, -1).contiguous()

    def _add_state(self, tensordict: TensorDictBase) -> TensorDictBase:
        infos = self._last_infos()
        if not infos:
            return tensordict
        tensordict.set((self._group, "state"), self._build_tensor(infos, tensordict.device))
        return tensordict

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self._add_state(next_tensordict)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._add_state(tensordict_reset)

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        obs_spec[(self._group, "state")] = Unbounded(
            shape=torch.Size([self._n_agents, self._state_dim]),
            device=obs_spec.device,
            dtype=torch.float32,
        )
        return obs_spec


class VizdoomTask(TaskClass):
    """BenchMARL TaskClass adapter for ViZDoom PettingZoo environments."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name=config["scenario"], config=config)
        proto_env = self.make_env_fn(seed=config["seed"])()
        try:
            self._action_spec = proto_env.action_spec
            self._observation_spec = self._filter_policy_obs(
                proto_env.observation_spec,
                vision_only=bool(config.get("vision_only_obs", False)),
            )
            self._action_mask_spec = getattr(proto_env, "action_mask_spec", None)
            self._info_spec = getattr(proto_env, "info_spec", None)
            self._has_render = True
        finally:
            proto_env.close()

    @staticmethod
    def _filter_policy_obs(obs_spec: Composite, *, vision_only: bool) -> Composite:
        """Return a Composite that advertises a single key to the policy.

        BenchMARL builds the actor/critic ``input_spec`` from
        :meth:`observation_spec`, so dropping every other key here hides them
        from the policy while leaving the corresponding tensors intact in the
        rollout TensorDict (still written by the ``GameVarsFromInfo``,
        ``ObjectPositionsFromInfo`` and ``GroundTruthStateFromInfo``
        transforms). That keeps the trajectory logger working without leaking
        unintended state into the policy network.

        - ``vision_only=True``  → keep only ``("agent", "observation")`` (image).
          This is the standard pixel-based RL setup.
        - ``vision_only=False`` → keep only ``("agent", "state")`` — the
          normalised ground-truth state vector built by
          :class:`GroundTruthStateFromInfo`. Use this debug mode to remove
          the visual encoder from the optimisation problem and check whether
          the policy can solve the task at all given perfect observations.
        """
        agent_spec = obs_spec["agent"]
        keep_key = "observation" if vision_only else "state"
        if keep_key not in agent_spec.keys(True, True):
            raise KeyError(
                f"_filter_policy_obs: required key {keep_key!r} not present "
                f"in agent observation_spec. Available keys: "
                f"{list(agent_spec.keys(True, True))}. "
                f"For vision_only_obs=False, ensure objects_info=true so the "
                "ground-truth state transform is attached."
            )
        filtered_agent = Composite(
            {keep_key: agent_spec[keep_key]},
            shape=agent_spec.shape,
            device=agent_spec.device,
        )
        return Composite(
            {"agent": filtered_agent},
            shape=obs_spec.shape,
            device=obs_spec.device,
        )

    @staticmethod
    def env_name() -> str:
        return "vizdoom"

    def make_env_fn(self, seed: int, is_eval: bool = False):
        """Return a bare zero-arg callable that builds one env when invoked.

        The callable does no work until called, so handing it to ``ParallelEnv``
        lets each worker run its ViZDoom network handshake in parallel rather
        than serializing them in the parent process.

        ``is_eval`` flags envs created for BenchMARL's evaluation loop.
        Video recording is attached *only* to eval envs: with
        ``evaluation_episodes=1`` exactly one eval worker runs, captures the
        full episode every ``evaluation_interval`` frames, and writes one
        mp4 to ``$DOOM_VIDEO_DIR``. Training envs never record — that path
        was retired because it overlapped with eval and bloated runs.
        """
        def _make():
            cfg = self.config
            host_address = cfg["host_address"]

            role = "eval" if is_eval else "train"
            max_attempts = 1 + max(0, int(cfg.get("vizdoom_init_retries", 3)))
            retry_delay_s = max(0.0, float(cfg.get("vizdoom_init_retry_delay_s", 2.0)))
            init_jitter_s = max(0.0, float(cfg.get("vizdoom_init_jitter_s", 1.0)))

            pz_env = None
            env = None
            for attempt_idx in range(max_attempts):
                pz_env = None
                env = None
                if attempt_idx > 0 and retry_delay_s > 0:
                    time.sleep(retry_delay_s * attempt_idx)
                if init_jitter_s > 0:
                    jitter = (
                        (os.getpid() * 1103515245 + time.time_ns()) % 1_000_000
                    ) / 1_000_000
                    time.sleep(jitter * init_jitter_s)

                port_reservation = _reserve_udp_port(host_address, cfg)
                port = port_reservation.port
                start = time.perf_counter()
                attempt = f"{attempt_idx + 1}/{max_attempts}"
                _log_vizdoom_init(cfg, "starting", port=port, role=role, attempt=attempt)
                try:
                    with _suppress_native_output(cfg.get("quiet_vizdoom_logs", True)):
                        pz_env = make(
                            scenario=cfg["scenario"],
                            config_file=cfg["config_file"],
                            num_agents=cfg["num_agents"],
                            resolution=cfg["resolution"],
                            timeout=int(cfg["timeout"]),
                            skip_frames=cfg["skip_frames"],
                            async_mode=cfg["async_mode"],
                            render_mode=cfg["render_mode"],
                            host_address=host_address,
                            port=port,
                            netmode=cfg["netmode"],
                            ticrate=cfg["ticrate"],
                            use_multi_binary_action_space=False,
                            seed=seed,
                            enable_video=False,
                            record_every=0,
                            video_fps=cfg["video_fps"],
                            daemon=cfg["daemon"],
                            objects_info=cfg.get("objects_info", False),
                        )
                        pz_env = _release_port_when_closed(pz_env, port_reservation)
                        video_dir = os.environ.get("DOOM_VIDEO_DIR")
                        if cfg["enable_video"] and video_dir and is_eval:
                            # One eval env, one episode per eval call. Recording
                            # cadence is `eval.video_every_n_evals` evaluations
                            # (= every Nth call to the eval loop), on top of the
                            # `evaluation_interval` throttle.
                            pz_env = WorkerVideoLoggerParallelWrapper(
                                pz_env,
                                every_n_episodes=int(
                                    cfg.get("eval_video_every_n_evals", 1)
                                ),
                                fps=cfg["video_fps"],
                                video_dir=video_dir,
                                max_workers=1,
                                scenario=cfg["scenario"],
                                config_file=cfg["config_file"],
                                skip_frames=cfg.get("skip_frames", 1),
                            )
                        action_set = cfg.get("action_set", "full")
                        if action_set == "turn_only_forward":
                            pz_env = ForcedForwardTurnOnlyWrapper(pz_env)
                        elif action_set == "left_right_forward_stand":
                            pz_env = LeftRightForwardStandWrapper(pz_env)
                        elif action_set == "left_right_forward":
                            pz_env = LeftRightForwardWrapper(pz_env)
                        elif action_set != "full":
                            raise ValueError(
                                f"Unknown action_set {action_set!r}. "
                                "Expected 'full', 'turn_only_forward', "
                                "'left_right_forward', or "
                                "'left_right_forward_stand'."
                            )
                        pz_env = _attach_info_cache(pz_env)
                        env = PettingZooWrapper(env=pz_env)
                except Exception as exc:
                    if pz_env is not None:
                        try:
                            pz_env.close()
                        except Exception:
                            port_reservation.release()
                    else:
                        port_reservation.release()
                    _log_vizdoom_init(
                        cfg,
                        "failed",
                        port=port,
                        elapsed=time.perf_counter() - start,
                        role=role,
                        attempt=attempt,
                        error=exc,
                    )
                    if attempt_idx == max_attempts - 1:
                        raise
                    continue

                _log_vizdoom_init(
                    cfg,
                    "ready",
                    port=port,
                    elapsed=time.perf_counter() - start,
                    role=role,
                    attempt=attempt,
                )
                break

            if pz_env is None or env is None:
                raise RuntimeError("ViZDoom environment construction failed without an exception")
            agent_keys = list(
                getattr(pz_env, "possible_agents", None)
                or getattr(pz_env, "agents", None)
                or [f"agent_{i}" for i in range(cfg["num_agents"])]
            )
            obs_h = cfg["obs_h"]
            obs_w = cfg["obs_w"]
            n_stack = cfg["frame_stack"]
            transforms = [
                SelectTransform(("agent", "observation"), ("agent", "info")),
                AHWCToTensorResize(
                    key=("agent", "observation"), h=obs_h, w=obs_w, mode="bilinear"
                ),
            ]
            if n_stack > 1:
                # reset_key fires at episode boundaries (all agents together) — correct for
                # cooperative no-death scenarios. For per-agent deaths (respawn/deathmatch),
                # switch to reset_key=("agent", "terminated") so each agent's buffer flushes
                # independently; otherwise the centralized critic sees stale frames for dead agents.
                transforms.append(
                    CatFrames(
                        N=n_stack,
                        dim=-1,
                        in_keys=[("agent", "observation")],
                        out_keys=[("agent", "observation")],
                        padding="same",
                        reset_key=("agent", "_reset"),
                    )
                )
            transforms.append(GameVarsFromInfo(agent_keys))
            if cfg.get("objects_info", False):
                transforms.append(
                    ObjectPositionsFromInfo(
                        agent_keys,
                        max_objects=cfg.get("object_info_max_objects", 64),
                    )
                )
                transforms.append(
                    GroundTruthStateFromInfo(
                        agent_keys,
                        max_objects=cfg.get("object_info_max_objects", 64),
                        coord_scale=float(cfg.get("state_coord_scale", 1024.0)),
                    )
                )
            transforms.append(RemoveEmptySpecs())
            env = TransformedEnv(env, Compose(*transforms))
            return env.to(cfg["sampling_device"])

        return _make

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: int | None, device=None):
        # BenchMARL's Experiment._setup_task calls get_env_fun TWICE per run:
        # the first call builds the evaluation env (test_env), the second
        # builds the training env factory. We tag the first call as eval so
        # that the video-recording wrapper is only attached to training envs.
        is_eval = not getattr(self, "_eval_env_built", False)
        if is_eval:
            self._eval_env_built = True
        return self.make_env_fn(seed, is_eval=is_eval)

    def action_spec(self, env: EnvBase) -> Composite:
        return self._action_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        return self._observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return self._action_mask_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return self._info_spec

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def group_map(self, env=None):
        if env is not None:
            gm = getattr(env, "group_map", None)
            if isinstance(gm, dict) and gm:
                return gm
            agents = getattr(env, "possible_agents", None)
            if agents:
                return {"agent": list(agents)}
        n = int(self.config["num_agents"])
        return {"agent": [f"agent_{i}" for i in range(n)]}

    def has_render(self, env: EnvBase) -> bool:
        return self._has_render

    def max_steps(self, env: EnvBase) -> int:
        return int(self.config["timeout"])

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True
