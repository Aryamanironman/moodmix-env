"""Smoke tests for DJSet-v3 (drum stems and echo sends)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moodmix_env.env_v3 import MoodMixV3Env
from moodmix_env.expert_v3 import expert_v3_action
from moodmix_env.render_v3 import render_mix

ROOT = Path(__file__).resolve().parents[1]
PAIR = {"track_a": "1. Blinding Lights", "track_b": "16. The Night We Met"}


def _requires_crate() -> bool:
    return (ROOT / "crate" / "manifest_v2.json").exists()


def test_v3_spaces_and_expert_actions():
    if not _requires_crate():
        return
    env = MoodMixV3Env(data_dir=ROOT / "crate", max_steps=8)
    obs, _ = env.reset(seed=0, options=PAIR)
    assert obs.shape == (48,)
    assert env.action_space.shape == (12,)
    total = 0.0
    for _ in range(8):
        action = expert_v3_action(env)
        assert action.shape == (12,)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (48,)
        assert np.isfinite(reward)
        assert {"drum_craft", "drum_swap", "echo_use"} <= set(
            info["reward_components"])
        total += float(reward)
        if terminated or truncated:
            break
    assert total > 0.0
    env.close()


def test_v3_render_has_drums_and_a_tail():
    if not _requires_crate():
        return
    env = MoodMixV3Env(data_dir=ROOT / "crate", max_steps=8)
    env.reset(seed=0, options=PAIR)
    for _ in range(8):
        _, _, terminated, truncated, _ = env.step(expert_v3_action(env))
        if terminated or truncated:
            break
    mix, sr = render_mix(env)
    assert mix.ndim == 2 and mix.shape[1] == 2
    assert np.isfinite(mix).all()
    assert float(np.max(np.abs(mix))) > 0.01
    assert len(mix) / sr > 4.0
    env.close()
