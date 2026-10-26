"""Tests for the local MoodMix demo web app."""

from __future__ import annotations

import json
from pathlib import Path

from moodmix_env.webapp import create_app

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def test_index_serves_controller():
    app = create_app(model_path=None, web_dir=WEB_DIR)
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"MoodMix" in response.data


def test_episode_endpoint_returns_trace_and_audio(tmp_path: Path):
    app = create_app(model_path=None, runs_dir=tmp_path / "runs", web_dir=WEB_DIR)
    client = app.test_client()
    response = client.get("/api/episode?mood=party&policy=human&seed=3")
    assert response.status_code == 200
    data = response.get_json()
    assert data["mood"] == "party"
    assert len(data["steps"]) == 64
    assert data["steps"][0]["bar"] == 1
    assert data["steps"][-1]["fader"] > 0.9
    assert data["summary"]["total"] > 15.0
    # The reward panel is built from these keys, so the craft terms have to
    # reach the browser under the default profile.
    assert {"bass_swap", "eq_craft", "pad_craft", "phrase"} <= set(
        data["summary"]["components"]
    )
    assert "bass_swap" in data["steps"][0]["components"]
    assert data["audio_url"].endswith(".wav")

    audio = client.get(data["audio_url"])
    assert audio.status_code == 200
    assert len(audio.data) > 1000


def test_set_endpoint_chains_transitions(tmp_path: Path):
    app = create_app(model_path=None, runs_dir=tmp_path / "runs", web_dir=WEB_DIR)
    client = app.test_client()
    response = client.get("/api/set?mood=relaxed&policy=human&seed=4&transitions=3")
    assert response.status_code == 200
    data = response.get_json()

    transitions = data["set"]["transitions"]
    assert len(transitions) == 3
    # A set hands over: deck B of one transition opens the next.
    for previous, following in zip(transitions, transitions[1:]):
        assert following["track_a"] == previous["track_b"]
    # ...and does not immediately run the same pair backwards.
    pairs = [(t["track_a"], t["track_b"]) for t in transitions]
    assert all(b != (a[1], a[0]) for a, b in zip(pairs, pairs[1:]))

    assert len(data["steps"]) == 64 * 3
    times = [step["t"] for step in data["steps"]]
    assert times == sorted(times)
    assert [step["transition"] for step in data["steps"]][:2] == [0, 0]

    audio = client.get(data["audio_url"])
    assert audio.status_code == 200
    assert len(audio.data) > 100_000


def test_set_length_is_clamped_and_validated(tmp_path: Path):
    app = create_app(model_path=None, runs_dir=tmp_path / "runs", web_dir=WEB_DIR)
    client = app.test_client()
    clamped = client.get("/api/set?mood=relaxed&policy=human&seed=1&transitions=99")
    assert clamped.status_code == 200
    assert clamped.get_json()["set"]["count"] == 8
    assert client.get("/api/set?transitions=two").status_code == 400


def test_invalid_policy_is_rejected(tmp_path: Path):
    app = create_app(model_path=None, runs_dir=tmp_path / "runs", web_dir=WEB_DIR)
    client = app.test_client()
    response = client.get("/api/episode?policy=hacker")
    assert response.status_code == 400


def test_health_reports_playable_moods():
    app = create_app(model_path=None, web_dir=WEB_DIR)
    health = app.test_client().get("/api/health").get_json()
    assert health["available_moods"] == [
        "energetic", "happy", "relaxed", "emotional", "focused", "party"
    ]


def test_mood_with_no_tracks_is_rejected(tmp_path: Path):
    demo = Path(__file__).resolve().parents[1] / "data" / "demo"
    manifest = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    manifest["tracks"] = [t for t in manifest["tracks"]
                          if t["mood"] in ("relaxed", "focused")]
    crate = tmp_path / "crate"
    crate.mkdir()
    (crate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    app = create_app(model_path=None, runs_dir=tmp_path / "runs",
                     web_dir=WEB_DIR, data_dir=crate)
    client = app.test_client()
    health = client.get("/api/health").get_json()
    assert health["available_moods"] == ["relaxed", "focused"]

    response = client.get("/api/episode?mood=party")
    assert response.status_code == 400
    body = response.get_json()
    assert "no party tracks" in body["error"]
    assert body["available_moods"] == ["relaxed", "focused"]
