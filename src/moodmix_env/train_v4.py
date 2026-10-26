"""PPO training for DJSet-v4 (stem-level remix)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .env_v4 import MoodMixV4Env
from .logging_v1 import TrainingLogger


def make_env(data_dir: str | Path | None = None,
             max_steps: int = 128,
             compatible_only: bool = False,
             seed: int = 0) -> Callable[[], MoodMixV4Env]:
    def _init():
        env = MoodMixV4Env(data_dir=data_dir, max_steps=max_steps,
                           compatible_only=compatible_only)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model: PPO, episodes: int = 10, seed: int = 10_000,
             data_dir: str | Path | None = None,
             max_steps: int = 128) -> dict[str, float]:
    env = MoodMixV4Env(data_dir=data_dir, max_steps=max_steps)
    returns, finals, overlaps, jumps = [], [], [], []
    components: dict[str, list[float]] = {}
    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
        finals.append(float(env.fader))
        overlaps.append(float(np.mean(env._low_overlaps)))
        jumps.append(float(np.mean(env._level_jumps)))
        for name, value in env.episode_components.items():
            components.setdefault(name, []).append(float(value))
    env.close()
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "final_fader": float(np.mean(finals)),
        "low_overlap": float(np.mean(overlaps)),
        "level_jump": float(np.mean(jumps)),
        **{f"component_{name}": float(np.mean(values))
           for name, values in components.items()},
    }


def train(timesteps: int = 1_000_000,
          n_envs: int = 8,
          n_steps: int = 256,
          batch_size: int = 256,
          seed: int = 0,
          output: str | Path = "models/ppo_v4",
          resume: str | Path | None = None,
          checkpoint_freq: int = 100_000,
          checkpoint_dir: str | Path = "checkpoints_v4",
          data_dir: str | Path | None = None,
          max_steps: int = 128,
          eval_episodes: int = 10,
          verbose: int = 1,
          tensorboard_log: str | Path | None = None,
          device: str = "cpu",
          learning_rate: float = 3e-4,
          ent_coef: float = 0.002,
          vec_env: str = "dummy",
          compatible_only: bool = False,
          log_dir: str | Path | None = "runs/training",
          log_eval_freq: int = 100_000,
          log_eval_episodes: int = 5,
          log_save_every: int = 100_000) -> str:
    env_fns = [make_env(data_dir=data_dir, max_steps=max_steps,
                        compatible_only=compatible_only, seed=seed + index)
               for index in range(n_envs)]
    vec_cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    vec = VecMonitor(vec_cls(env_fns))
    vec.seed(seed)

    if resume:
        model = PPO.load(str(resume), env=vec, device=device)
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
        reset_timesteps = False
    else:
        model = PPO(
            "MlpPolicy", vec,
            n_steps=n_steps, batch_size=batch_size,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            learning_rate=learning_rate, ent_coef=ent_coef,
            clip_range=0.2, seed=seed, device=device, verbose=verbose,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs=dict(net_arch=[256, 128]),
        )
        reset_timesteps = True

    callbacks = []
    if checkpoint_freq > 0:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=max(1, checkpoint_freq // n_envs),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_v4",
            verbose=verbose,
        ))

    log_path = None
    if log_dir:
        log_path = Path(log_dir) / Path(output).stem
        callbacks.append(TrainingLogger(
            log_path, data_dir=data_dir, max_steps=max_steps,
            eval_freq=log_eval_freq, eval_episodes=log_eval_episodes,
            save_every=log_save_every, verbose=verbose,
            env_cls=MoodMixV4Env,
        ))

    started = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks or None,
                reset_num_timesteps=reset_timesteps,
                tb_log_name="ppo_v4")
    elapsed = time.time() - started

    base = Path(output)
    if base.suffix == ".zip":
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(base))
    metrics = evaluate(model, episodes=eval_episodes, data_dir=data_dir,
                       max_steps=max_steps)
    meta = {
        "model": str(base.with_suffix(".zip")),
        "timesteps": int(timesteps),
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "data_dir": str(data_dir) if data_dir else None,
        "compatible_only": bool(compatible_only),
        "elapsed_seconds": round(elapsed, 2),
        "log_dir": str(log_path) if log_path else None,
        "eval": metrics,
    }
    (base.parent / f"{base.name}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    vec.close()
    return str(base.with_suffix(".zip"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on DJSet-v4")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/ppo_v4"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints_v4"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vec-env", choices=("dummy", "subproc"),
                        default="dummy")
    parser.add_argument("--compatible-only", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/training"),
                        help="where CSV/plots/report are written")
    parser.add_argument("--log-eval-freq", type=int, default=100_000)
    parser.add_argument("--log-eval-episodes", type=int, default=5)
    parser.add_argument("--log-save-every", type=int, default=100_000)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    print(f"training DJSet-v4 PPO for {args.timesteps:,} steps "
          f"with {args.n_envs} envs", flush=True)
    path = train(
        timesteps=args.timesteps, n_envs=args.n_envs,
        n_steps=args.n_steps, batch_size=args.batch_size, seed=args.seed,
        output=args.output, resume=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir, data_dir=args.data_dir,
        max_steps=args.max_steps, eval_episodes=args.eval_episodes,
        verbose=args.verbose, tensorboard_log=args.tensorboard_log,
        device=args.device, learning_rate=args.learning_rate,
        ent_coef=args.ent_coef, vec_env=args.vec_env,
        compatible_only=args.compatible_only,
        log_dir=args.log_dir, log_eval_freq=args.log_eval_freq,
        log_eval_episodes=args.log_eval_episodes,
        log_save_every=args.log_save_every,
    )
    print(f"model: {path}")
    meta = Path(path).with_name(Path(path).stem + "_meta.json")
    if meta.exists():
        print(meta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
