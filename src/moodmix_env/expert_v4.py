"""Stem-arrangement expert for DJSet-v4."""

from __future__ import annotations

from .env_v4 import MoodMixV4Env


def expert_v4_action(env: MoodMixV4Env):
    phase = env.step_count / max(1, env.max_steps)
    return env._target(phase)
