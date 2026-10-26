"""Smoke tests for the MoodMix DJ environment."""

from __future__ import annotations

import numpy as np

from moodmix_env import MOODS, MoodMixEnv


def test_spaces_and_reset():
    env = MoodMixEnv()
    obs, info = env.reset(seed=0)
    assert set(obs) == {"deck_a", "deck_b", "mixer", "context"}
    for key, value in obs.items():
        assert env.observation_space[key].contains(value), key
    assert info["mood"] in MOODS
    assert info["deck_a"]["mood"] == info["mood"]
    env.close()


def test_step_returns_valid_tuple():
    env = MoodMixEnv()
    env.reset(seed=1)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    for key, value in obs.items():
        assert env.observation_space[key].contains(value), key
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_components" in info


def test_episode_terminates_after_max_steps():
    env = MoodMixEnv(max_steps=8)
    env.reset(seed=2)
    for _ in range(7):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert not terminated
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert terminated
    assert info["step"] == 8
    assert "mix_quality" in info["episode_components"]


def test_crossfader_action_moves_gain():
    env = MoodMixEnv()
    env.reset(seed=3)
    action = env.action_space.sample()
    action[0] = 0
    _, _, _, _, info = env.step(action)
    assert info["gain_a"] > 0.99 and info["gain_b"] < 0.01
    action[0] = 20
    _, _, _, _, info = env.step(action)
    assert info["gain_b"] > 0.99 and info["gain_a"] < 0.01


def test_observation_hides_the_scripted_target():
    """The target curve is a reward term, not an observation.

    Feeding it to the policy told it which move to make; the agent has to
    learn the move from deck state and progress instead.
    """
    env = MoodMixEnv(max_steps=64)
    obs, _ = env.reset(seed=3)
    assert obs["context"].shape == (12,)
    assert env.observation_space["context"].shape == (12,)
    assert env.observation_space["context"].contains(obs["context"])
    # Deck and mixer groups are unchanged; total observation is 21+21+9+12.
    assert sum(value.shape[0] for value in obs.values()) == 63
    env.step([0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert env._observation()["context"].shape == (12,)


def test_sync_aligns_phase():
    env = MoodMixEnv()
    env.reset(seed=4)
    action = env.action_space.sample()
    action[7] = 1
    _, _, _, _, info = env.step(action)
    assert info["sync"] is True
    # After stepping, both phases advance together while sync is held.
    obs = env._observation()
    assert obs["deck_a"][3] == obs["deck_b"][3] or abs(
        obs["deck_a"][3] - obs["deck_b"][3]
    ) < 1e-6


def test_avoid_tracks_is_honoured_when_a_choice_remains():
    env = MoodMixEnv(max_steps=8)
    _, info = env.reset(
        seed=0,
        options={"mood": "relaxed", "track_a": "relaxed_drift",
                 "avoid_tracks": ["focused_flow"]},
    )
    assert info["deck_b"]["id"] == "emotional_rain"


def test_seed_reproducible():
    a = MoodMixEnv().reset(seed=9)[1]
    b = MoodMixEnv().reset(seed=9)[1]
    assert a["mood"] == b["mood"]
    assert a["deck_a"]["id"] == b["deck_a"]["id"]
    assert a["deck_b"]["id"] == b["deck_b"]["id"]


def _score(env, policy, seed=5):
    obs, info = env.reset(seed=seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(policy(info))
        total += float(reward)
        if terminated or truncated:
            return total, info


def test_completed_transition_beats_parking():
    env = MoodMixEnv(max_steps=64)

    def parking(info):
        # Park the fader mid-blend forever: an easy reward-hack.
        return [10, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    def transition(info):
        u = min(max((info["step"] / info["max_steps"] - 0.12) / 0.68, 0.0), 1.0)
        target = u * u * (3.0 - 2.0 * u)
        sync = 1 if info["step"] == 0 else 0
        return [int(round(target * 20)), 1, 1, 1, 1, 1, 1, sync, 0, 0]

    parked, _ = _score(env, parking)
    finished, info = _score(env, transition)
    assert info["crossfader"] > 0.9
    assert finished > parked + 5.0, (finished, parked)
