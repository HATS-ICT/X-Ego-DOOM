from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ParallelEnv

from doom_marl.callbacks import _parse_wad_map
from doom_marl.log import log_event


_AGENT_COLORS = (
    (0, 215, 255),
    (184, 79, 255),
    (255, 200, 0),
    (53, 107, 255),
)


class WorkerVideoLoggerParallelWrapper(wrappers.BaseParallelWrapper):
    """Record selected PettingZoo worker episodes with unique W&B keys.

    The ViZDoom wheel's recorder counts episodes per environment process, but
    all processes log to the same W&B key. This wrapper claims a small number
    of recorder slots per run and includes the slot id in filenames and keys.
    """

    def __init__(
        self,
        env: ParallelEnv,
        *,
        every_n_episodes: int = 50,
        fps: int = 35,
        max_frames: int = 1000,
        video_dir: str | os.PathLike[str] | None = None,
        max_workers: int = 1,
        scenario: str | None = None,
        config_file: str | os.PathLike[str] | None = None,
        skip_frames: int = 1,
    ):
        super().__init__(env)
        self.every_n = int(every_n_episodes)
        self.fps = int(fps)
        self.max_frames = int(max_frames)
        self.video_dir = Path(video_dir) if video_dir else None
        self.max_workers = max(0, int(max_workers))
        self.scenario = scenario
        self.config_file = config_file
        self.skip_frames = max(1, int(skip_frames))

        self._ep_idx = 0
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._positions: dict[str, list[tuple[float, float, float]]] = {}
        self._objects: list[tuple[float, float]] = []
        self._object_signature: tuple[tuple[float, float], ...] | None = None
        self._pending_reset_obs = None
        self._pending_reset_info = None
        # Per-episode running totals for the on-screen overlay. Cumulative
        # reward across all agents in the current episode, and a count of
        # discrete pickups (a step whose reward jump exceeds PICKUP_REWARD_MIN
        # is treated as one item collected — distinguishes the +1/+2 pickup
        # signal from continuous shaping like the per-tic facing bonus).
        self._ep_reward: float = 0.0
        self._items_collected: int = 0
        self._items_solo: int = 0
        self._items_coop: int = 0
        self._ep_frame: int = 0
        self._map_data = _parse_wad_map(scenario, config_file)
        self._map_bounds = self._compute_map_bounds()

        # Slot claim is DEFERRED until the first real step() call. TorchRL's
        # ParallelEnv builds a transient "dummy" env in the main process at
        # construction time purely to introspect specs — that env calls
        # reset() but never steps, then is discarded without close(). If we
        # claimed the slot eagerly in __init__, that dummy would steal slot 0
        # forever and starve every actual rollout subprocess of recording.
        # By deferring to step(), only envs that actually run can claim.
        self._worker_id: int | None = None
        self._slot_attempted = False
        self._enabled = self.every_n > 0 and self.max_workers > 0 and self.video_dir is not None
        if not self._enabled and self.every_n > 0:
            log_event(
                "video", "disabled", pid=True,
                reason="not configured",
                max_workers=self.max_workers, dir=self.video_dir,
            )

    def _ensure_slot_claimed(self):
        """Lazily claim a recording slot on first real step()."""
        if self._slot_attempted or not self._enabled:
            return
        self._slot_attempted = True
        self._worker_id = self._claim_worker_slot()
        if self._worker_id is None:
            self._enabled = False
            log_event(
                "video", "disabled", pid=True,
                reason="no free slot", max_workers=self.max_workers,
            )
        else:
            log_event(
                "video", "enabled", pid=True,
                worker=f"{self._worker_id:02d}",
                every_n_episodes=self.every_n,
                dir=self.video_dir,
            )

    def _claim_worker_slot(self) -> int | None:
        if self.max_workers <= 0 or self.video_dir is None:
            return None
        slots_dir = self.video_dir / ".record_worker_slots"
        slots_dir.mkdir(parents=True, exist_ok=True)
        for worker_id in range(self.max_workers):
            slot_dir = slots_dir / f"{worker_id:02d}"
            try:
                slot_dir.mkdir()
            except FileExistsError:
                continue
            (slot_dir / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
            self.video_dir.mkdir(parents=True, exist_ok=True)
            return worker_id
        return None

    def _maybe_enable(self):
        self._ep_idx += 1
        self._recording = self._enabled and (self._ep_idx % self.every_n == 0)
        self._frames.clear()
        self._positions.clear()
        self._objects.clear()
        self._ep_reward = 0.0
        self._items_collected = 0
        self._items_solo = 0
        self._items_coop = 0
        self._ep_frame = 0
        self._object_signature = None

    def _compute_map_bounds(self) -> tuple[float, float, float, float]:
        lines = (
            self._map_data.get("blocking_lines", [])
            + self._map_data.get("nonblocking_lines", [])
        )
        if not lines:
            return -700.0, 1200.0, -300.0, 1400.0
        xs = [line[0] for line in lines] + [line[2] for line in lines]
        ys = [line[1] for line in lines] + [line[3] for line in lines]
        return min(xs) - 50.0, max(xs) + 50.0, min(ys) - 50.0, max(ys) + 50.0

    @staticmethod
    def _tile_ahwc(per_agent_ahwc: np.ndarray) -> np.ndarray:
        agent_count, height, width, channels = per_agent_ahwc.shape
        cols = int(math.ceil(math.sqrt(agent_count)))
        rows = int(math.ceil(agent_count / cols))
        canvas = np.zeros(
            (rows * height, cols * width, channels),
            dtype=per_agent_ahwc.dtype,
        )
        for index in range(agent_count):
            row, col = divmod(index, cols)
            canvas[
                row * height : (row + 1) * height,
                col * width : (col + 1) * width,
            ] = per_agent_ahwc[index]
        return canvas

    def _record_infos(self, info_dict):
        for agent in list(self.agents):
            info = (info_dict or {}).get(agent, {}) or {}
            x_val = float(info.get("POSITION_X", 0.0))
            y_val = float(info.get("POSITION_Y", 0.0))
            angle_val = float(info.get("ANGLE", 0.0))
            self._positions.setdefault(agent, []).append((x_val, y_val, angle_val))

        objects = self._extract_objects(info_dict)
        self._objects = objects
        self._object_signature = self._object_signature_from_points(objects)

    def _extract_objects(self, info_dict) -> list[tuple[float, float]]:
        objects = []
        if not info_dict:
            return objects
        first_info = next(iter(info_dict.values()), {}) or {}
        for obj in first_info.get("objects", []) or []:
            xy = self._object_xy(obj)
            if xy is not None:
                objects.append(xy)
        return objects

    @staticmethod
    def _object_signature_from_points(
        objects: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...] | None:
        if not objects:
            return None
        return tuple(sorted((round(x, 1), round(y, 1)) for x, y in objects))

    def _object_signature_from_info(self, info_dict):
        return self._object_signature_from_points(self._extract_objects(info_dict))

    @staticmethod
    def _object_xy(obj) -> tuple[float, float] | None:
        if isinstance(obj, dict):
            x_val = obj.get("x", obj.get("POSITION_X", obj.get("pos_x")))
            y_val = obj.get("y", obj.get("POSITION_Y", obj.get("pos_y")))
            if x_val is None or y_val is None:
                return None
            return float(x_val), float(y_val)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            return float(obj[0]), float(obj[1])
        return None

    def _world_to_panel(self, x_val: float, y_val: float, size: int) -> tuple[int, int]:
        xlo, xhi, ylo, yhi = self._map_bounds
        scale = min((size - 24) / max(xhi - xlo, 1.0), (size - 24) / max(yhi - ylo, 1.0))
        x_pad = (size - (xhi - xlo) * scale) * 0.5
        y_pad = (size - (yhi - ylo) * scale) * 0.5
        px = int(round(x_pad + (x_val - xlo) * scale))
        py = int(round(size - (y_pad + (y_val - ylo) * scale)))
        return px, py

    def _draw_minimap(self, size: int) -> np.ndarray:
        import cv2

        panel = np.full((size, size, 3), (23, 27, 35), dtype=np.uint8)
        blocking = self._map_data.get("blocking_lines", [])
        nonblocking = self._map_data.get("nonblocking_lines", [])

        for lines, color, thickness in (
            (nonblocking, (78, 85, 96), 1),
            (blocking, (205, 211, 217), 1),
        ):
            for x1, y1, x2, y2 in lines:
                cv2.line(
                    panel,
                    self._world_to_panel(x1, y1, size),
                    self._world_to_panel(x2, y2, size),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        for ox, oy in self._objects:
            cv2.rectangle(
                panel,
                tuple(v - 2 for v in self._world_to_panel(ox, oy, size)),
                tuple(v + 2 for v in self._world_to_panel(ox, oy, size)),
                (204, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )

        for agent_idx, agent in enumerate(list(self.agents)):
            color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
            trail = self._positions.get(agent, [])
            if not trail:
                continue
            recent = trail[-20:]
            points = [self._world_to_panel(x, y, size) for x, y, _ in recent]
            for p1, p2 in zip(points, points[1:]):
                cv2.line(panel, p1, p2, color, 2, lineType=cv2.LINE_AA)

            x_val, y_val, angle = trail[-1]
            center = self._world_to_panel(x_val, y_val, size)
            heading_len = max(12, size // 14)
            angle_rad = math.radians(angle)
            tip = (
                int(round(center[0] + math.cos(angle_rad) * heading_len)),
                int(round(center[1] - math.sin(angle_rad) * heading_len)),
            )
            cv2.arrowedLine(
                panel,
                center,
                tip,
                color,
                2,
                tipLength=0.45,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(panel, center, max(2, size // 90), color, -1, lineType=cv2.LINE_AA)
            cv2.circle(panel, center, max(3, size // 75), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        tick = self._ep_frame * self.skip_frames
        cv2.putText(
            panel,
            (
                f"R {self._ep_reward:.1f}  "
                f"solo {self._items_solo}  coop {self._items_coop}  "
                f"f {self._ep_frame}  t {tick}"
            ),
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _compose_frame(self, pov_frame: np.ndarray) -> np.ndarray:
        import cv2

        panel_size = max(240, pov_frame.shape[1])
        minimap = self._draw_minimap(panel_size)
        if pov_frame.shape[0] != panel_size:
            new_width = max(1, int(round(pov_frame.shape[1] * panel_size / pov_frame.shape[0])))
            if new_width % 2:
                new_width += 1
            pov_frame = cv2.resize(pov_frame, (new_width, panel_size), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([pov_frame, minimap], axis=1)

    @staticmethod
    def _encode_safe_frame(frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        # Pad up to a multiple of 16 so imageio-ffmpeg won't auto-resize and
        # spam IMAGEIO FFMPEG_WRITER WARNING on every video write.
        block = 16
        pad_bottom = (-height) % block
        pad_right = (-width) % block
        if pad_bottom or pad_right:
            frame = np.pad(
                frame,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="edge",
            )
        return frame

    def _push_frame_from_obs(self, obs_dict, info_dict=None):
        if not self._recording or not obs_dict:
            return
        self._record_infos(info_dict)
        frames = []
        for agent in list(self.agents):
            frame = obs_dict[agent]
            if frame.dtype.kind == "f":
                frame = np.clip(frame, 0.0, 1.0)
                frame = (frame * 255.0).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8, copy=False)
            frames.append(frame)
        tiled = self._tile_ahwc(np.stack(frames, axis=0))
        self._frames.append(self._compose_frame(tiled))
        self._ep_frame += 1
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)

    def _video_path(self) -> Path:
        assert self.video_dir is not None
        assert self._worker_id is not None
        return self.video_dir / (
            f"worker_{self._worker_id:02d}_pid_{os.getpid()}_"
            f"episode_{self._ep_idx:06d}.mp4"
        )

    def _write_mp4(self, arr_thwc: np.ndarray) -> Path | None:
        if self.video_dir is None or self._worker_id is None:
            return None
        path = self._video_path()
        # Write to a sibling temp file then atomic-rename, so a main-process
        # uploader scanning the dir never sees a half-written mp4. The temp
        # file MUST keep the `.mp4` suffix (imageio's FFMPEG plugin validates
        # the URI extension and rejects `.part`), so we insert `.part` before
        # `.mp4` instead of appending after it.
        part_path = path.with_name(f"{path.stem}.part{path.suffix}")
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(
                str(part_path),
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=16,
                # Don't pass ``-pix_fmt yuv420p`` here: imageio-ffmpeg already
                # injects it for libx264, and a duplicate triggers a noisy
                # "Multiple -pix_fmt options specified" warning per write.
                ffmpeg_params=["-movflags", "+faststart"],
            ) as writer:
                for frame in arr_thwc:
                    writer.append_data(self._encode_safe_frame(frame))
            os.replace(part_path, path)
            return path
        except Exception as exc:
            log_event("video", "failed", level="error", op="mp4-write", path=path, error=exc)
            return None

    def _finalize(self):
        if not self._recording or not self._frames:
            return
        arr_thwc = np.stack(self._frames, axis=0)
        path = self._write_mp4(arr_thwc)
        # NOTE: do NOT call wandb.log from here. This wrapper runs inside a
        # torchrl ParallelEnv subprocess where wandb.init() was never invoked
        # (wandb is initialised only in the main process). Uploading to W&B
        # is done by the main-process VideoWandbLogger callback in
        # doom_marl/callbacks.py, which scans this directory each iter and
        # logs new mp4s under the same key scheme used here.
        if path is not None:
            log_event(
                "video", "saved",
                worker=f"{self._worker_id:02d}",
                episode=f"{self._ep_idx:06d}",
                path=path,
            )
        self._frames.clear()
        self._recording = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._pending_reset_obs = obs
        self._pending_reset_info = info
        # Slot claim hasn't happened yet during the very first reset because
        # it is deferred until step(). The reset frame is cached and flushed
        # after the first real worker successfully claims a slot.
        if self._worker_id is not None:
            self._maybe_enable()
            self._push_frame_from_obs(obs, info)
            self._pending_reset_obs = None
            self._pending_reset_info = None
        return obs, info

    def step(self, actions):
        self._ensure_slot_claimed()
        obs, rewards, term, trunc, info = self.env.step(actions)
        self.agents = self.env.agents[:]
        if self._worker_id is not None:
            # If this is the first ever step on the recording worker, mirror
            # what reset() would have done so the rest of episode 1 records.
            if not self._recording and self._enabled and self._ep_idx == 0:
                self._maybe_enable()
                if self._pending_reset_obs is not None:
                    self._push_frame_from_obs(
                        self._pending_reset_obs,
                        self._pending_reset_info,
                    )
                    self._pending_reset_obs = None
                    self._pending_reset_info = None
            self._accumulate_reward(rewards, info)
            self._push_frame_from_obs(obs, info)
            if any(term.values()) or any(trunc.values()):
                self._finalize()
        return obs, rewards, term, trunc, info

    # Reward jumps at or above this threshold are counted as a discrete
    # item pickup (the scenario emits +1 for solo and +2 for coop pickups,
    # while continuous shaping like the per-tic facing bonus is ~0.005).
    _PICKUP_REWARD_MIN = 0.5
    # A step whose reward sum is at or above this threshold is treated as
    # a coop (partner) pickup; below it (but above _PICKUP_REWARD_MIN) is
    # a solo (first-of-pair) pickup. The scenario emits +1.0 / +2.0, so
    # 1.5 cleanly separates them and absorbs any tiny shaping reward
    # accumulated in the same step.
    _COOP_REWARD_MIN = 1.5

    def _accumulate_reward(self, rewards, info_dict=None):
        if not self._recording or not rewards:
            return
        # The scenario writes its reward into a shared ACS global, which
        # ViZDoom mirrors to every agent's per-step reward identically. So
        # rewards.values() are duplicates of the same global delta. Take
        # the max (== any single agent) instead of summing — otherwise a
        # real +1.0 solo pickup would inflate to +N and get misclassified
        # as a coop event with N agents.
        step_values = []
        for value in rewards.values():
            try:
                step_values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not step_values:
            return
        step_value = max(step_values)
        current_signature = self._object_signature_from_info(info_dict)
        if (
            step_value >= self._PICKUP_REWARD_MIN
            and self._object_signature is not None
            and current_signature is not None
            and current_signature == self._object_signature
        ):
            step_value = 0.0

        self._ep_reward += step_value
        if step_value >= self._PICKUP_REWARD_MIN:
            self._items_collected += 1
            if step_value >= self._COOP_REWARD_MIN:
                self._items_coop += 1
            else:
                self._items_solo += 1

    def close(self):
        if self._worker_id is not None:
            self._finalize()
            self._release_worker_slot()
        return self.env.close()

    def _release_worker_slot(self):
        """Release this worker's slot directory so a future env can take it.

        Without this, an early-discarded env would permanently hold the slot.
        Combined with the lazy claim in step(), this makes the slot system
        robust to torchrl's transient spec-introspection envs.
        """
        if self._worker_id is None or self.video_dir is None:
            return
        slot_dir = self.video_dir / ".record_worker_slots" / f"{self._worker_id:02d}"
        try:
            (slot_dir / "pid.txt").unlink(missing_ok=True)
            slot_dir.rmdir()
        except OSError:
            pass
        self._worker_id = None
        self._enabled = False
