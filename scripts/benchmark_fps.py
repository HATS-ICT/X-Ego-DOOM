import argparse
import multiprocessing as mp
import sys
import termios
import time
from pathlib import Path

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import RandomPolicy

# Ensure we can import from the root directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doom_marl.task import VizdoomTask
from run import build_task_config, load_cfg, parse_set_overrides


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS of parallel ViZDoom environments.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config file")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments to run")
    parser.add_argument("--steps", type=int, default=20000, help="Total environment steps to collect")
    parser.add_argument("--batch-size", type=int, default=200, help="Steps per env per collection batch")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", default=[], help="Override YAML settings")
    args = parser.parse_args()

    overrides = parse_set_overrides(args.set)
    cfg = load_cfg(args.config, overrides)

    cfg.setdefault("scenario", "remedy_rush_coop_multi_agent")
    cfg.setdefault("num_agents", 2)
    cfg.setdefault("resolution", "160X120")
    cfg.setdefault("skip_frames", 4)
    cfg.setdefault("seed", 0)
    cfg.setdefault("async_mode", True)
    cfg.setdefault("host_address", "127.0.0.1")
    cfg.setdefault("netmode", 1)
    cfg.setdefault("ticrate", 35)
    cfg.setdefault("daemon", True)
    cfg.setdefault("render_mode", "rgb_array")
    cfg.setdefault("obs_h", 72)
    cfg.setdefault("obs_w", 96)
    cfg.setdefault("frame_stack", 1)
    cfg.setdefault("timeout", 1000)
    cfg.setdefault("enable_video", True)
    cfg.setdefault("record_every_n_episodes_per_worker", 100)
    cfg.setdefault("video_fps", 35)
    cfg.setdefault("sampling_device", "cpu")
    cfg["num_envs"] = args.num_envs

    # Force configurations for maximum headless speed
    cfg["enable_video"] = False
    cfg["render_mode"] = "rgb_array"
    cfg["record_every_n_episodes_per_worker"] = 0

    cfg["enable_video"] = False
    cfg["render_mode"] = "rgb_array"
    cfg["record_every_n_episodes_per_worker"] = 0
    task = VizdoomTask(task_cfg)

    print(f"[{args.config.name}] Setting up {args.num_envs} parallel environments...")

    # Bare callables — ParallelEnv will only invoke them inside worker
    # subprocesses, so the 8 ViZDoom network handshakes happen concurrently
    # instead of being serialized in the main process.
    env_fns = [task.make_env_fn(seed=i) for i in range(args.num_envs)]
    
    parallel_env = ParallelEnv(
        num_workers=args.num_envs,
        create_env_fn=env_fns,
    )
    
    # Force start to catch initialization errors early
    parallel_env.start()
    
    print("Environments initialized! Building Zero policy to prevent agent deaths...")
    
    # We use a Zero action policy instead of Random. 
    # Random actions cause agents to kill each other, ending the episode in seconds.
    # If the episode ends, TorchRL calls env.reset(). 
    # If the ViZDoom wrapper completely restarts the engine on reset(), 
    # doing this across 8 parallel envs will cause an endless loop of slow networking re-initializations.
    class ZeroPolicy:
        def __init__(self, spec):
            self.spec = spec
        def __call__(self, td):
            td.update(self.spec.zero())
            return td
            
    policy = ZeroPolicy(parallel_env.action_spec)

    frames_per_batch = args.num_envs * args.batch_size
    print(f"Starting benchmark... (Target: {args.steps} total steps)")
    print(f"Batch size: {frames_per_batch} steps per batch ({args.batch_size} per env)")
    
    collector = SyncDataCollector(
        create_env_fn=parallel_env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=args.steps,
        device="cpu",
    )

    start_time = time.time()
    total_env_steps = 0
    num_agents = int(cfg.get("num_agents", 2))

    for batch in collector:
        # batch.numel() counts the leading dimensions (num_envs * time_steps)
        # This is the number of environment transitions.
        env_steps_in_batch = batch.numel()
        total_env_steps += env_steps_in_batch
        total_agent_steps = total_env_steps * num_agents
        
        elapsed = time.time() - start_time
        env_fps = total_env_steps / elapsed
        agent_fps = total_agent_steps / elapsed
        
        print(f"Collected {total_env_steps}/{args.steps} env steps | Elapsed: {elapsed:.2f}s | "
              f"Env FPS: {env_fps:.0f} | Agent FPS: {agent_fps:.0f}")

    end_time = time.time()
    duration = end_time - start_time
    total_agent_steps = total_env_steps * num_agents

    print("\n" + "="*50)
    print("Benchmark Complete!")
    print("="*50)
    print(f"Total Envs:   {args.num_envs}")
    print(f"Agents/Env:   {num_agents}")
    print("-" * 50)
    print(f"Total Env Steps:   {total_env_steps}")
    print(f"Total Agent Steps: {total_agent_steps}")
    print(f"Total Time:        {duration:.2f} seconds")
    print("-" * 50)
    print(f"Average Env FPS:   {total_env_steps / duration:.2f} steps/sec")
    print(f"Average Agent FPS: {total_agent_steps / duration:.2f} steps/sec")
    print("="*50)

    # Cleanly shut down parallel workers. Guard close() because torchrl's
    # ParallelEnv raises if it's already been torn down (e.g. after the
    # collector closed it during iteration cleanup).
    try:
        if not getattr(parallel_env, "is_closed", False):
            parallel_env.close()
    except RuntimeError:
        pass


def _restore_terminal(fd: int, attrs):
    if attrs is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except (termios.error, OSError):
        pass


if __name__ == "__main__":
    # Required for parallel environments on Windows
    mp.set_start_method("spawn", force=True)

    # pygame/SDL (pulled in by ViZDoom) can leave the controlling terminal in
    # a no-echo state if a child crashes during shutdown. Snapshot termios
    # up front and restore on exit so the user's shell stays usable.
    _tty_fd = sys.stdin.fileno() if sys.stdin.isatty() else -1
    _tty_attrs = None
    if _tty_fd >= 0:
        try:
            _tty_attrs = termios.tcgetattr(_tty_fd)
        except (termios.error, OSError):
            _tty_attrs = None
    try:
        main()
    finally:
        _restore_terminal(_tty_fd, _tty_attrs)
