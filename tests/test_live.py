"""Live booth tests: dry decks and the live render endpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moodmix_env.env_v1 import MoodMixV1Env
from moodmix_env.expert_v1 import expert_action
from moodmix_env.render_v2 import prepare_decks
from moodmix_env.webapp_v1 import create_app

ROOT = Path(__file__).resolve().parents[1]
PAIR = {"track_a": "1. Blinding Lights", "track_b": "16. The Night We Met"}


def _has_crate() -> bool:
    return (ROOT / "crate" / "manifest_v2.json").exists()


def _run_expert(env: MoodMixV1Env) -> None:
    env.reset(seed=0, options=PAIR)
    while True:
        _, _, terminated, truncated, _ = env.step(expert_action(env))
        if terminated or truncated:
            return


def test_prepare_decks_are_aligned_and_headroomed():
    if not _has_crate():
        return
    env = MoodMixV1Env(data_dir=ROOT / "crate", max_steps=32)
    _run_expert(env)
    a, b = prepare_decks(env)
    assert a.shape == b.shape
    assert a.ndim == 2 and a.shape[1] == 2
    expected = (env.track_a["beat_times"][env.cue_a + env.step_count]
                - env.track_a["beat_times"][env.cue_a])
    assert abs(a.shape[0] / 44100 - expected) < 0.05
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) <= 0.81
    assert float(np.sqrt(np.mean(a ** 2))) > 0.01
    assert float(np.sqrt(np.mean(b ** 2))) > 0.01
    assert abs(float(np.sqrt(np.mean(a ** 2)))
               - float(np.sqrt(np.mean(b ** 2)))) < 0.10
    env.close()


def test_live_endpoint_returns_two_downloadable_decks(tmp_path: Path):
    if not _has_crate():
        return
    app = create_app(data_dir=ROOT / "crate", model_path=None,
                     runs_dir=tmp_path / "runs")
    client = app.test_client()
    response = client.post("/api/render", json={
        **PAIR,
        "policy": "expert",
        "best_cues": True,
        "seed": 0,
        "live": True,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["live"] is True
    assert data["deck_a_url"] and data["deck_b_url"]
    assert data["audio_url"] is None
    for url in (data["deck_a_url"], data["deck_b_url"]):
        audio = client.get(url)
        assert audio.status_code == 200
        assert len(audio.data) > 100_000


def test_full_endpoint_streams_songs_and_a_transition(tmp_path: Path):
    if not _has_crate():
        return
    app = create_app(data_dir=ROOT / "crate", model_path=None,
                     runs_dir=tmp_path / "runs")
    client = app.test_client()
    response = client.post("/api/render", json={
        **PAIR,
        "policy": "expert",
        "best_cues": True,
        "seed": 0,
        "live": True,
        "full": True,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["full"] is True
    assert data["duration"] > 100.0
    assert data["transition_start"] > 0.0
    assert data["transition_end"] > data["transition_start"]
    assert data["deck_b_gain"] >= 0.5
    assert len(data["wave_a"]) == 320
    assert len(data["wave_b"]) == 320
    source = client.get(data["deck_a_url"])
    padded = client.get(data["deck_b_url"])
    assert source.status_code == 200
    assert padded.status_code == 200
    assert len(source.data) > 1_000_000
    assert len(padded.data) > 1_000_000
    for key in ("deck_a_drums_url", "deck_a_tonal_url",
                "deck_b_drums_url", "deck_b_tonal_url"):
        assert data[key]
        stem = client.get(data[key])
        assert stem.status_code == 200
        assert len(stem.data) > 100_000
