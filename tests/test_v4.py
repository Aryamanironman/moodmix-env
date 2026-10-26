"""Smoke test for the DJSet-v4 stem-remix environment."""

from __future__ import annotations

import numpy as np

from moodmix_env.env_v4 import MoodMixV4Env
from moodmix_env.expert_v4 import expert_v4_action


def test_v4_spaces_and_expert(tmp_path):
    beats = 128
    energy = np.linspace(0.05, 0.20, beats, dtype=np.float32)
    feature_path = tmp_path / "features.npz"
    np.savez(
        feature_path,
        beat_seconds=0.464,
        beats=beats,
        a_vocal=energy,
        b_drums=energy,
        b_bass=energy,
        b_other=energy,
        b_inst=energy,
    )
    env = MoodMixV4Env(feature_path=feature_path, max_steps=beats)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (14,)
    assert env.action_space.shape == (6,)
    total = 0.0
    for _ in range(beats):
        action = expert_v4_action(env)
        assert action.shape == (6,)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (14,)
        assert np.isfinite(reward)
        assert {"mashup", "structure", "echo_use"} <= set(
            info["reward_components"])
        total += float(reward)
        if terminated or truncated:
            break
    assert total > 0.0
    assert "structure" in env.episode_components
    env.close()
