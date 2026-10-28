from __future__ import annotations

import multiprocessing as mp
import random
import socket
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doom_marl.utils import load_cfg, parse_set_overrides  # noqa: E402
from scripts.build_scenario import ensure_built, resolve_base_scenario  # noqa: E402


def _parse_args():
    ap = ArgumentParser(description="Smoke-test ViZDoom scenarios before training.")
    ap.add_argument("config", type=Path, help="Path to experiment YAML config file")
    ap.add_argument("--seed", type=int, default=None, help="Override seed from YAML")
    ap.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override any YAML key, e.g. --set scenario=foo_multi_agent skip_frames=1",
    )
    ap.add_argument(
        "--launcher",
        choices=("parallel", "human-multi", "single"),
        default="parallel",
        help="parallel: exact PettingZoo stack; human-multi: one human-controlled agent in multiplayer; single: one local DoomGame window.",
    )
    ap.add_argument(
        "--human-agent",
        type=int,
        default=0,
        help="Agent index to control in human-multi mode.",
    )
    ap.add_argument(
        "--human-mode",
        choices=("spectator", "async_spectator"),
        default="async_spectator",
        help="ViZDoom mode for the human-controlled instance.",
    )
    ap.add_argument(
        "--peer-policy",
        choices=("noop", "random"),
        default="noop",
        help="Background policy for non-human agents in human-multi mode.",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes to run before exiting.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Hard cap on loop iterations per episode for smoke tests.",
    )
    ap.add_argument(
        "--debug-objects",
        action="store_true",
        help="Enable objects info in direct DoomGame launchers.",
    )
    ap.add_argument(
        "--debug-automap",
        action="store_true",
        help="Enable automap buffer in direct DoomGame launchers.",
    )
    ap.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="human",
        help="Render mode for the exact-stack parallel launcher.",
    )
    ap.add_argument(
        "--record-video",
        type=Path,
        metavar="PATH",
        help="Save video of parallel run to this path (forces --render-mode rgb_array).",
    )
    return ap.parse_args()


def _pick_free_port(host_address: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host_address, 0))
        return s.getsockname()[1]


def _load_runtime_cfg(args) -> dict:
    overrides = parse_set_overrides(args.set or [])
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_cfg(args.config, overrides)
    if "scenario_dir" in cfg and "variant" in cfg:
        # Same resolution as run.py::build_task_config: build the variant WAD
        # and point ViZDoom at its generated .cfg.
        cfg.config_file = str(ensure_built(cfg.scenario_dir, cfg.variant))
        cfg.scenario = resolve_base_scenario(cfg.scenario_dir)
    return cfg


def _resolve_scenario_cfg(scenario: Optional[str] = None, config_file: Optional[str] = None) -> str:
    if config_file is not None:
        return str(Path(config_file).expanduser().resolve())

    import vizdoom as vzd

    if scenario is None:
        raise ValueError("Either scenario or config_file must be provided.")
    return str((Path(vzd.scenarios_path) / f"{scenario}.cfg").resolve())


def _build_action_template(game) -> list[float]:
    import vizdoom as vzd

    action: list[float] = []
    for button in game.get_available_buttons():
        action.append(0.0 if vzd.is_delta_button(button) else 0.0)
    return action


def _sample_action(game, rng: random.Random, policy: str) -> list[float]:
    import vizdoom as vzd

    if policy == "noop":
        return _build_action_template(game)

    action: list[float] = []
    for button in game.get_available_buttons():
        if vzd.is_delta_button(button):
            action.append(float(rng.choice((-1, 0, 1))))
        else:
            action.append(float(rng.choice((0, 1))))
    return action


def _configure_game(
    *,
    config_path: str,
    resolution: str,
    ticrate: int,
    timeout: Optional[int],
    seed: Optional[int],
    host_address: str,
    port: int,
    netmode: int,
    agent_idx: int,
    num_agents: int,
    is_host: bool,
    window_visible: bool,
    mode_name: str,
    debug_objects: bool,
    debug_automap: bool,
):
    import vizdoom as vzd
    from vizdoom.pettingzoo_wrapper.utils import get_screen_resolution

    mode_map = {
        "player": vzd.Mode.PLAYER,
        "async_player": vzd.Mode.ASYNC_PLAYER,
        "spectator": vzd.Mode.SPECTATOR,
        "async_spectator": vzd.Mode.ASYNC_SPECTATOR,
    }

    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_ticrate(int(ticrate))
    game.set_window_visible(bool(window_visible))
    game.set_mode(mode_map[mode_name])
    if timeout is not None:
        game.set_episode_timeout(int(timeout))
    if seed is not None:
        game.set_seed(int(seed))
    if debug_objects:
        game.set_objects_info_enabled(True)
    if debug_automap:
        game.set_automap_buffer_enabled(True)
    if is_host:
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode {int(netmode)} +sv_spawnfarthest 1"
        )
    else:
        game.add_game_args(
            f"-join {host_address} -port {port} -netmode {int(netmode)}"
        )
    game.add_game_args(f"+name Player{agent_idx} +colorset {agent_idx}")
    game.add_game_args(f"+playernumber {agent_idx}")
    return game


def _run_single(cfg: dict, args) -> None:
    config_path = _resolve_scenario_cfg(cfg.get("scenario"), cfg.get("config_file"))
    game = _configure_game(
        config_path=config_path,
        resolution=cfg.get("resolution", "160X120"),
        ticrate=cfg.get("ticrate", 35),
        timeout=cfg.get("timeout"),
        seed=cfg.get("seed"),
        host_address=cfg.get("host_address", "127.0.0.1"),
        port=_pick_free_port(cfg.get("host_address", "127.0.0.1")),
        netmode=cfg.get("netmode", 0),
        agent_idx=0,
        num_agents=1,
        is_host=True,
        window_visible=True,
        mode_name=args.human_mode,
        debug_objects=args.debug_objects,
        debug_automap=args.debug_automap,
    )

    try:
        game.init()
        for episode in range(args.episodes):
            print(f"[single] episode {episode + 1}/{args.episodes}")
            game.new_episode()
            steps = 0
            while not game.is_episode_finished() and steps < args.max_steps:
                state = game.get_state()
                if state is not None and steps % 35 == 0:
                    print(
                        f"[single] tic={state.number} vars={list(state.game_variables) if state.game_variables is not None else []}"
                    )
                game.advance_action()
                steps += 1
            print(f"[single] total_reward={game.get_total_reward():.3f}")
            time.sleep(0.5)
    finally:
        game.close()


def _parallel_noop_action(space) -> Any:
    import numpy as np
    from gymnasium import spaces

    if isinstance(space, spaces.Discrete):
        return 0
    if isinstance(space, spaces.MultiBinary):
        return np.zeros(space.shape, dtype=np.int8)
    if isinstance(space, spaces.MultiDiscrete):
        return np.zeros_like(space.nvec, dtype=np.int64)
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Dict):
        return {k: _parallel_noop_action(v) for k, v in space.spaces.items()}
    raise TypeError(f"Unsupported action space: {type(space)}")


def _parallel_random_action(space, rng) -> Any:
    try:
        return space.sample(mask=None)
    except TypeError:
        return space.sample()


def _run_parallel(cfg: dict, args) -> None:
    from vizdoom.pettingzoo_wrapper import make

    if args.record_video:
        args.render_mode = "rgb_array"
        video_frames = []

    host_address = cfg.get("host_address", "127.0.0.1")
    env = make(
        scenario=cfg["scenario"],
        config_file=cfg.get("config_file"),
        num_agents=cfg["num_agents"],
        resolution=cfg.get("resolution", "160X120"),
        timeout=cfg.get("timeout"),
        skip_frames=cfg.get("skip_frames", 1),
        async_mode=cfg.get("async_mode", True),
        render_mode=args.render_mode,
        host_address=host_address,
        port=_pick_free_port(host_address),
        netmode=cfg.get("netmode", 1),
        ticrate=cfg.get("ticrate", 35),
        use_multi_binary_action_space=False,
        seed=cfg.get("seed"),
        enable_video=False,
        daemon=cfg.get("daemon", True),
    )

    rng = random.Random(cfg.get("seed", 0))
    try:
        for episode in range(args.episodes):
            obs, infos = env.reset(seed=cfg.get("seed"))
            print(f"[parallel] episode {episode + 1}/{args.episodes} agents={list(obs)}")
            for agent, info in infos.items():
                keys = ", ".join(sorted(info.keys())[:8])
                print(f"[parallel] {agent} info keys: {keys}")

            for step in range(args.max_steps):
                actions = {}
                for agent in env.agents:
                    space = env.action_space(agent)
                    if args.peer_policy == "random":
                        actions[agent] = _parallel_random_action(space, rng)
                    else:
                        actions[agent] = _parallel_noop_action(space)

                _, rewards, terminations, truncations, infos = env.step(actions)
                if args.render_mode == "human":
                    env.render()
                elif args.record_video:
                    frame = env.render()
                    if frame is not None:
                        video_frames.append(frame)
                if step % 35 == 0:
                    reward_str = ", ".join(
                        f"{agent}={rewards[agent]:.3f}" for agent in env.agents
                    )
                    print(f"[parallel] step={step} rewards: {reward_str}")
                if any(terminations.values()) or any(truncations.values()):
                    print(f"[parallel] finished after {step + 1} steps")
                    break

        if args.record_video and video_frames:
            import imageio
            print(f"[parallel] saving video to {args.record_video}...")
            args.record_video.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(str(args.record_video), fps=35, codec="libx264", quality=8) as writer:
                for frame in video_frames:
                    writer.append_data(frame)
            print("[parallel] video saved!")
    finally:
        env.close()


def _background_multiplayer_worker(
    *,
    config_path: str,
    resolution: str,
    ticrate: int,
    timeout: Optional[int],
    seed: Optional[int],
    host_address: str,
    port: int,
    netmode: int,
    agent_idx: int,
    num_agents: int,
    is_host: bool,
    mode_name: str,
    window_visible: bool,
    policy: str,
    episodes: int,
    max_steps: int,
    skip_frames: int,
    debug_objects: bool,
    debug_automap: bool,
) -> None:
    from vizdoom.pettingzoo_wrapper.utils import get_flat_game_vars

    game = _configure_game(
        config_path=config_path,
        resolution=resolution,
        ticrate=ticrate,
        timeout=timeout,
        seed=seed,
        host_address=host_address,
        port=port,
        netmode=netmode,
        agent_idx=agent_idx,
        num_agents=num_agents,
        is_host=is_host,
        window_visible=window_visible,
        mode_name=mode_name,
        debug_objects=debug_objects,
        debug_automap=debug_automap,
    )

    rng = random.Random((seed or 0) + agent_idx)
    try:
        game.init()
        available_game_vars = game.get_available_game_variables()
        for episode in range(episodes):
            game.new_episode()
            if game.is_player_dead():
                game.respawn_player()
            print(f"[human-multi] agent={agent_idx} episode={episode + 1}/{episodes}")
            
            import time
            last_time = time.time()
            
            for step in range(max_steps):
                if game.is_episode_finished():
                    break
                if "spectator" in mode_name:
                    game.advance_action(skip_frames)
                else:
                    game.make_action(_sample_action(game, rng, policy), skip_frames)

                if game.is_player_dead():
                    game.respawn_player()

                state = game.get_state()
                if state is not None and step % 35 == 0:
                    info = get_flat_game_vars(state, available_game_vars)
                    curr_time = time.time()
                    elapsed = curr_time - last_time
                    fps = 35 / elapsed if elapsed > 0 and step > 0 else 0.0
                    print(
                        f"[human-multi] agent={agent_idx} step={step} vars={info} | last_35_steps_time={elapsed:.3f}s (FPS={fps:.1f})"
                    )
                    last_time = curr_time
            print(
                f"[human-multi] agent={agent_idx} total_reward={game.get_total_reward():.3f}"
            )
    finally:
        game.close()


def _run_human_multi(cfg: dict, args) -> None:
    config_path = _resolve_scenario_cfg(cfg.get("scenario"), cfg.get("config_file"))
    host_address = cfg.get("host_address", "127.0.0.1")
    num_agents = int(cfg["num_agents"])
    human_agent = int(args.human_agent)
    if not 0 <= human_agent < num_agents:
        raise ValueError(f"--human-agent must be in [0, {num_agents - 1}]")

    port = _pick_free_port(host_address)
    procs: list[mp.Process] = []

    for agent_idx in range(num_agents):
        is_human = agent_idx == human_agent
        proc = mp.Process(
            target=_background_multiplayer_worker,
            kwargs=dict(
                config_path=config_path,
                resolution=cfg.get("resolution", "160X120"),
                ticrate=cfg.get("ticrate", 35),
                timeout=cfg.get("timeout"),
                seed=None if cfg.get("seed") is None else int(cfg["seed"]) + agent_idx,
                host_address=host_address,
                port=port,
                netmode=cfg.get("netmode", 1),
                agent_idx=agent_idx,
                num_agents=num_agents,
                is_host=(agent_idx == 0),
                mode_name=args.human_mode if is_human else "async_player",
                window_visible=is_human,
                policy="noop" if is_human else args.peer_policy,
                episodes=args.episodes,
                max_steps=args.max_steps,
                skip_frames=cfg.get("skip_frames", 1),
                debug_objects=args.debug_objects and is_human,
                debug_automap=args.debug_automap and is_human,
            ),
            daemon=False,
        )
        proc.start()
        procs.append(proc)
        time.sleep(0.4)

    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode not in (0, None):
            exit_code = proc.exitcode

    if exit_code:
        raise SystemExit(exit_code)


def main() -> None:
    args = _parse_args()
    cfg = _load_runtime_cfg(args)

    if args.launcher == "parallel":
        _run_parallel(cfg, args)
    elif args.launcher == "human-multi":
        _run_human_multi(cfg, args)
    else:
        _run_single(cfg, args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
