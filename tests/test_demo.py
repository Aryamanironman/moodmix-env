"""Tests for the demo helpers (pair choice and the set tail)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from demo import forgiving_pairs, render_tail  # noqa: E402
from moodmix_env import MoodMixEnv  # noqa: E402
from moodmix_env.env import _key_distance  # noqa: E402


def test_forgiving_pairs_are_close_in_tempo_and_key():
    env = MoodMixEnv()
    pairs = forgiving_pairs(env, limit=4)
    assert len(pairs) >= 2
    by_id = {track["id"]: track for track in env.tracks}
    for a, b in pairs:
        assert abs(float(by_id[b]["bpm"]) / float(by_id[a]["bpm"]) - 1.0) <= 0.05 + 1e-9
        assert _key_distance(by_id[a], by_id[b]) <= 5
    env.close()


def test_render_tail_has_one_bar_of_audible_deck_b():
    env = MoodMixEnv(max_steps=8)
    env.reset(seed=0, options={"track_a": "energetic_pulse",
                               "track_b": "party_strobe"})
    for _ in range(8):
        env.step([20, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    tail = render_tail(env, 22050, bars=1)
    expected = int(round(4 * 60.0 / 128.0 * 22050))
    assert len(tail) == expected
    assert np.max(np.abs(tail)) > 0.0
    env.close()
