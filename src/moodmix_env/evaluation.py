"""Evaluate policies on identical seeds and report transition diagnostics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .env import MoodMixEnv
from .policies import smoothstep

Policy = Callable[[dict, dict], list[int]]

OBS_ORDER = ("context", "deck_a", "deck_b", "mixer")


def model_policy(model) -> Policy:
    """Wrap a Stable-Baselines3 model; FlattenObservation uses sorted keys."""

    def _policy(obs: dict, info: dict) -> list[int]:
        flat = np.concatenate([obs[key] for key in OBS_ORDER]).astype(np.float32)
        action, _ = model.predict(flat, deterministic=True)
        return action

    return _policy


def random_policy(env: MoodMixEnv) -> Policy:
    def _policy(obs: dict, info: dict) -> list[int]:
        return env.action_space.sample()

    return _policy


def run_episode(
    env: MoodMixEnv, policy: Policy, seed: int = 0
) -> tuple[float, dict[str, float]]:
    """Run one transition and return total reward plus diagnostics."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    deviation = 0.0
    synced = 0
    steps = 0
    while True:
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        progress = info["step"] / info["max_steps"]
        target = smoothstep((progress - 0.12) / 0.68)
        deviation += abs(info["crossfader"] - target)
        synced += int(info["sync"])
        steps += 1
        if terminated or truncated:
            break
    return total, {
        "final_fader": float(info["crossfader"]),
        "completed": 1.0 if info["crossfader"] > 0.9 else 0.0,
        "mean_deviation": deviation / max(1, steps),
        "sync_fraction": synced / max(1, steps),
    }


def evaluate_policy(
    policy: Policy,
    episodes: int = 10,
    seed: int = 1000,
    max_steps: int = 64,
    data_dir: str | None = None,
    reward_profile: str = "expressive",
) -> dict[str, float]:
    env = MoodMixEnv(data_dir=data_dir, max_steps=max_steps,
                     reward_profile=reward_profile)
    totals: list[float] = []
    metrics: list[dict[str, float]] = []
    for episode in range(episodes):
        total, result = run_episode(env, policy, seed=seed + episode)
        totals.append(total)
        metrics.append(result)
    env.close()
    return {
        "episodes": float(episodes),
        "mean": float(np.mean(totals)) if totals else 0.0,
        "std": float(np.std(totals)) if totals else 0.0,
        "final_fader": float(np.mean([m["final_fader"] for m in metrics]))
        if metrics else 0.0,
        "completed": float(np.mean([m["completed"] for m in metrics]))
        if metrics else 0.0,
        "mean_deviation": float(np.mean([m["mean_deviation"] for m in metrics]))
        if metrics else 0.0,
        "sync_fraction": float(np.mean([m["sync_fraction"] for m in metrics]))
        if metrics else 0.0,
    }
