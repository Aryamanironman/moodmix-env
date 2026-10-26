"""Smoke tests for DJSet-v1 (continuous controls, real beat grid)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moodmix_env.env_v1 import MoodMixV1Env
from moodmix_env.expert_v1 import expert_action

ROOT = Path(__file__).resolve().parents[1]
PAIR = {"track_a": "1. Blinding Lights", "track_b": "16. The Night We Met"}


def _requires_crate() -> bool:
    return (ROOT / "crate" / "manifest_v2.json").exists()


def test_v1_spaces_and_reset():
    if not _requires_crate():
        return
    env = MoodMixV1Env(data_dir=ROOT / "crate", max_steps=16)
    obs, info = env.reset(seed=0)
    assert obs.shape == (39,)
    assert env.action_space.shape == (8,)
    assert env.observation_space.contains(obs)
    assert info["deck_a"]["id"] != info["deck_b"]["id"]
    assert 0 <= env.cue_a <= len(env.track_a["beat_times"]) - 17
    assert 0 <= env.cue_b <= len(env.track_b["beat_times"]) - 17
    env.close()


def test_v1_expert_beats_a_parked_fader():
    if not _requires_crate():
        return
    env = MoodMixV1Env(data_dir=ROOT / "crate", max_steps=16)
    env.reset(seed=0, options=PAIR)
    expert_total = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(expert_action(env))
        expert_total += float(reward)
        if terminated or truncated:
            break

    parked = env.reset(seed=0, options=PAIR)[0]
    parked_total = 0.0
    while True:
        parked, reward, terminated, truncated, _ = env.step(
            np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
                     dtype=np.float32))
        parked_total += float(reward)
        if terminated or truncated:
            break
    assert expert_total > parked_total + 4.0, (expert_total, parked_total)
    assert env.fader < 0.95


def test_v1_random_actions_stay_inside_the_observation_space():
    if not _requires_crate():
        return
    env = MoodMixV1Env(data_dir=ROOT / "crate", max_steps=8)
    obs, _ = env.reset(seed=3)
    for _ in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()
