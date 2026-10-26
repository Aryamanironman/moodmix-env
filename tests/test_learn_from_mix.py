"""Tests for turning a recorded transition into a policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import learn_from_mix as learning  # noqa: E402
from moodmix_env.env import MoodMixEnv  # noqa: E402

DEMO = ROOT / "data" / "demo"


def _trace(faders: list[float], eq_a: list[list[int]] | None = None) -> dict:
    steps = []
    for i, fader in enumerate(faders):
        steps.append({
            "step": i + 1,
            "crossfader": fader,
            "eq_a": eq_a[i] if eq_a else [1, 1, 1],
            "eq_b": [1, 1, 1],
        })
    return {"steps": steps}


def test_find_track_ids_matches_names_and_stems(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tracks": [
        {"id": "alpha", "file": "1. Alpha.mp3"},
        {"id": "beta", "file": "2. Beta.mp3"},
    ]}))

    ids = learning.find_track_ids(
        manifest, [tmp_path / "1. Alpha.mp3", tmp_path / "2. Beta.mp3"]
    )
    assert ids == ["alpha", "beta"]

    # A path elsewhere on disk still resolves by file name.
    ids = learning.find_track_ids(manifest, [Path("/elsewhere/1. Alpha.mp3"),
                                            Path("/elsewhere/2. Beta.mp3")])
    assert ids == ["alpha", "beta"]


def test_find_track_ids_rejects_unknown_and_duplicate(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tracks": [
        {"id": "alpha", "file": "1. Alpha.mp3"},
    ]}))
    with pytest.raises(ValueError, match="not in"):
        learning.find_track_ids(manifest, [tmp_path / "nope.mp3",
                                          tmp_path / "1. Alpha.mp3"])
    with pytest.raises(ValueError, match="two different tracks"):
        learning.find_track_ids(manifest, [tmp_path / "1. Alpha.mp3",
                                          tmp_path / "1. Alpha.mp3"])


def test_actions_from_trace_resamples_to_the_requested_length():
    trace = _trace([0.0, 0.25, 0.5, 0.75, 1.0])
    actions = learning.actions_from_trace(trace, steps=21, sync=True)

    assert actions.shape == (21, 10)
    assert actions[0, 0] == 0
    assert actions[-1, 0] == 20
    assert np.all(np.diff(actions[:, 0]) >= 0)          # monotone fader
    assert set(np.unique(actions[:, 0])) <= set(range(21))
    assert actions[0, 7] == 1 and actions[1:, 7].sum() == 0
    assert actions[:, 8:].sum() == 0                    # drums/filter untouched


def test_actions_from_trace_keeps_eq_and_omits_sync_when_tempi_match():
    eq = [[0, 1, 2]] * 4
    actions = learning.actions_from_trace(_trace([0.0, 0.4, 0.8, 1.0], eq),
                                          steps=4, sync=False)
    assert actions[:, 1:4].tolist() == eq
    assert actions[:, 4:7].tolist() == [[1, 1, 1]] * 4
    assert actions[:, 7].sum() == 0


def test_actions_from_trace_rejects_empty_trace():
    with pytest.raises(ValueError):
        learning.actions_from_trace({"steps": []}, steps=8)


def test_replay_pins_the_pair_and_collects_one_observation_per_action():
    env = MoodMixEnv(data_dir=DEMO, max_steps=4)
    ids = [env.tracks[0]["id"], env.tracks[1]["id"]]
    actions = np.zeros((4, 10), dtype=np.int64)
    actions[:, 0] = [0, 5, 10, 20]

    observations, used = learning.replay(env, actions, ids, seed=0)

    assert observations.shape == (4, 63)        # 21 + 21 + 9 + 12 features
    assert used.shape == (4, 10)
    assert env.track_a["id"] == ids[0]
    assert env.track_b["id"] == ids[1]
    assert env.step_count == 4
    env.close()


def test_pinning_an_unknown_track_is_an_error():
    env = MoodMixEnv(data_dir=DEMO, max_steps=4)
    with pytest.raises(ValueError, match="unknown track id"):
        env.reset(seed=0, options={"track_a": "does not exist"})
    env.close()


def test_actions_agreement_is_exact_for_identical_controls():
    actions = learning.actions_from_trace(_trace([0.0, 0.5, 1.0]), steps=9)
    scores = learning.actions_agreement(actions, actions)
    assert scores["fader_mae_bins"] == 0.0
    assert scores["fader_exact"] == 1.0
    assert scores["eq_band_accuracy"] == 1.0
    assert scores["steps"] == 9


def test_actions_agreement_measures_a_deviating_policy():
    target = learning.actions_from_trace(_trace([0.0, 0.5, 1.0]), steps=4)
    taken = target.copy()
    taken[1:, 0] = 20                    # parks the fader on deck B
    scores = learning.actions_agreement(taken, target)
    assert scores["fader_mae_bins"] >= 5.0
    assert 0.0 <= scores["fader_exact"] < 1.0
