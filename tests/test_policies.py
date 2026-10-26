"""Tests for the shared heuristic policy and evaluation harness."""

from __future__ import annotations

from moodmix_env import MoodMixEnv
from moodmix_env.evaluation import evaluate_policy, run_episode
from moodmix_env.policies import heuristic_action


def test_heuristic_action_finishes_transition():
    env = MoodMixEnv(max_steps=64)
    obs, info = env.reset(seed=5)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(
            heuristic_action(obs, info)
        )
        total += float(reward)
        if terminated or truncated:
            break
    assert info["crossfader"] > 0.9
    # Scale is set by the expressive profile; the craft terms add to the
    # completion bonus, so a performance clears 15 with room to spare.
    assert total > 15.0


def test_run_episode_reports_metrics():
    env = MoodMixEnv(max_steps=64)
    total, metrics = run_episode(env, heuristic_action, seed=7)
    assert total > 15.0
    assert metrics["completed"] == 1.0
    assert 0.0 <= metrics["sync_fraction"] <= 1.0
    assert metrics["mean_deviation"] < 0.1


def test_evaluate_policy_aggregates():
    result = evaluate_policy(heuristic_action, episodes=2, seed=10)
    assert result["episodes"] == 2
    # The absolute scale moved when the arc-shaped energy term and the route
    # charge landed (the hand-written DJ now scores ~14.7 where it used to
    # score ~16), so the check that matters is the one against a parked fader:
    # the same seeds, the same episodes, no craft.
    assert result["mean"] > 13.0
    assert result["completed"] == 1.0


def test_heuristic_beats_parking_on_the_same_seeds():
    def parked(_obs, info):
        return [10, 1, 1, 1, 1, 1, 1, 1 if info["step"] == 0 else 0, 0, 0]

    human = evaluate_policy(heuristic_action, episodes=3, seed=10)
    idle = evaluate_policy(parked, episodes=3, seed=10)
    assert human["completed"] == 1.0
    assert idle["completed"] == 0.0
    assert human["mean"] > idle["mean"] + 5.0, (human["mean"], idle["mean"])
