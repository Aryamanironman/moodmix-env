"""Rendering must survive a real crate, where files disagree on sample rate.

The browser demo refused to render anything unless both decks were the same
rate, which is true of the generated demo crate and false of a folder of
downloaded songs - so the demo could not be pointed at real audio at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from moodmix_env import MoodMixEnv
from moodmix_env.policies import heuristic_action
from moodmix_env.render_audio import render_transition


def _crate(tmp_path: Path, rates: tuple[int, ...] = (44100, 48000)) -> Path:
    tracks = []
    for index, sr in enumerate(rates):
        name = f"t{index}.wav"
        t = np.arange(int(sr * 8)) / sr
        tone = 0.2 * np.sin(2 * np.pi * (110 * (index + 1)) * t)
        sf.write(tmp_path / name, tone.astype(np.float32), sr)
        tracks.append({
            "id": f"t{index}",
            "mood": "relaxed" if index == 0 else "emotional",
            "file": name,
            "bpm": 100,
            "key": "F",
            "band_curve": [[0.4, 0.3, 0.2]] * 96,
            "sections": ["intro", "build", "drop", "outro"] * 6,
        })
    (tmp_path / "manifest.json").write_text(json.dumps({"tracks": tracks}),
                                            encoding="utf-8")
    return tmp_path


def _play(env: MoodMixEnv) -> None:
    obs, info = env.reset(seed=0, options={"track_a": "t0", "track_b": "t1"})
    while True:
        obs, reward, terminated, truncated, info = env.step(
            heuristic_action(obs, info)
        )
        if terminated or truncated:
            return


def test_render_mixes_tracks_with_different_sample_rates(tmp_path: Path):
    env = MoodMixEnv(data_dir=_crate(tmp_path), max_steps=8)
    _play(env)

    out = render_transition(env, tmp_path / "mix.wav")
    assert out.exists() and out.stat().st_size > 1000
    # Deck A sets the rate; deck B is resampled onto it.
    assert sf.info(str(out)).samplerate == 44100


def test_render_still_accepts_an_explicit_rate(tmp_path: Path):
    env = MoodMixEnv(data_dir=_crate(tmp_path), max_steps=4)
    _play(env)

    out = render_transition(env, tmp_path / "mix48.wav", sr=48000)
    assert sf.info(str(out)).samplerate == 48000
