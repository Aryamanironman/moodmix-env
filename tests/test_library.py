"""Tests for turning a local audio folder into an environment manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from moodmix_env.library import analyze_track, build_manifest, fold_tempo


def _click_track(path: Path, seconds: float = 8.0, sr: int = 22050) -> None:
    t = np.arange(int(seconds * sr)) / sr
    y = 0.4 * np.sin(2 * np.pi * 110.0 * t)
    click = np.zeros_like(y)
    for beat in np.arange(0, seconds, 0.5):  # 120 BPM
        start = int(beat * sr)
        length = min(int(0.05 * sr), len(click) - start)
        if length > 0:
            click[start:start + length] += 0.8 * np.hanning(length)
    sf.write(path, np.clip(y + click, -1, 1), sr)


def test_fold_tempo_removes_octave_errors():
    assert fold_tempo(172.3) == pytest.approx(86.15)
    assert fold_tempo(63.0) == pytest.approx(63.0)
    assert fold_tempo(128.0) == pytest.approx(128.0)
    assert fold_tempo(45.0) == pytest.approx(45.0)
    assert fold_tempo(0.0) == pytest.approx(120.0)
    assert fold_tempo(float("nan")) == pytest.approx(120.0)


def test_analyze_track_returns_env_fields(tmp_path: Path):
    path = tmp_path / "test_click.wav"
    _click_track(path)
    track = analyze_track(path, mood="focused")
    assert track["mood"] == "focused"
    assert track["duration"] > 7.5
    assert track["bpm"] > 0
    assert track["key"] in (
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    )
    assert len(track["band_curve"]) > 4
    assert len(track["band_curve"][0]) == 3
    assert len(track["sections"]) == max(1, len(track["band_curve"]) // 4)


def test_build_manifest_writes_loadable_json(tmp_path: Path):
    _click_track(tmp_path / "a.wav")
    out = tmp_path / "manifest.json"
    manifest = build_manifest(tmp_path, out, moods={"a": "relaxed"})
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded["tracks"]) == 1
    assert loaded["tracks"][0]["mood"] == "relaxed"
    assert manifest["tracks"][0]["file"] == "a.wav"
