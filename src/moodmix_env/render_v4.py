"""Render a DJSet-v4 stem arrangement to audio."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .render_v2 import _apply_filter, finalise

SR = 44100
BEAT = 0.6966
BARS = 32
BEATS = BARS * 4
STEM_CUE = 78.0
STEM_DIR = Path(__file__).resolve().parents[2] / "runs" / "v1" / "remix_stems"


def _read(path: Path, start: float, beats: int) -> np.ndarray:
    duration = beats * BEAT
    with sf.SoundFile(str(path)) as handle:
        handle.seek(int(round(start * SR)))
        data = handle.read(int(round(duration * SR)), dtype="float32",
                           always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    want = int(round(duration * SR))
    if len(data) < want:
        data = np.pad(data, ((0, want - len(data)), (0, 0)))
    return data[:want].astype(np.float32)


def _curve(actions: np.ndarray, column: int, n: int) -> np.ndarray:
    times = np.arange(len(actions), dtype=np.float64) * BEAT
    values = actions[:, column].astype(np.float64)
    return np.interp(np.arange(n) / SR, times, values).astype(np.float32)


def _echo(audio: np.ndarray, wet: np.ndarray, delay: float,
          feedback: float = 0.30, taps: int = 3) -> np.ndarray:
    if float(np.max(wet)) <= 1e-4:
        return np.zeros_like(audio)
    shift = max(1, int(round(delay * SR)))
    out = np.zeros_like(audio)
    for tap in range(1, taps + 1):
        offset = shift * tap
        if offset >= len(audio):
            break
        out[offset:] += (feedback ** (tap - 1)) * audio[:-offset]
    return out * wet[:, None]


def _gate_echo(values: np.ndarray, max_beats: float = 4.0) -> np.ndarray:
    out = np.zeros_like(values)
    active = False
    start = 0
    for index, value in enumerate(values):
        if value > 0.12:
            if not active:
                active = True
                start = index
            if index - start <= max_beats:
                out[index] = value
        else:
            active = False
    return out


def render_actions(actions: np.ndarray, out_path: str | Path,
                   stem_dir: Path = STEM_DIR) -> Path:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError("expected an (n, 6) stem action array")
    a_vocal = _read(stem_dir / "A_vocals.wav", STEM_CUE, BEATS)
    b_drums = _read(stem_dir / "B_drums.wav", STEM_CUE, BEATS)
    b_bass = _read(stem_dir / "B_bass.wav", STEM_CUE, BEATS)
    b_other = _read(stem_dir / "B_other.wav", STEM_CUE, BEATS)
    n = len(a_vocal)

    vocal = _curve(actions, 0, n)
    drums = _curve(actions, 1, n)
    bass = _curve(actions, 2, n)
    other = _curve(actions, 3, n)
    wet_actions = actions.copy()
    wet_actions[:, 4] = _gate_echo(actions[:, 4])
    wet = _curve(wet_actions, 4, n)
    filt = _curve(actions, 5, n)

    b_inst = (b_drums * drums[:, None] + b_bass * bass[:, None]
              + b_other * other[:, None])
    f = (filt - 0.5) * 2.0
    lp_cut = np.where(f < 0.0, 20000.0 * (200.0 / 20000.0) ** (-f), 20000.0)
    hp_cut = np.where(f > 0.0, 20.0 * (8000.0 / 20.0) ** f, 20.0)
    b_inst = _apply_filter(b_inst, lp_cut.astype(np.float32), SR, False)
    b_inst = _apply_filter(b_inst, hp_cut.astype(np.float32), SR, True)

    dry = a_vocal * vocal[:, None] + b_inst
    mix = dry + _echo(a_vocal * vocal[:, None], wet * 0.65, 0.75 * BEAT)

    # Drop impact at the 80th beat.
    drop_at = int(20 * 4 * BEAT * SR)
    impact_len = min(int(2.0 * SR), len(mix) - drop_at)
    if impact_len > 0:
        times = np.arange(impact_len, dtype=np.float32) / SR
        thump = np.sin(2.0 * np.pi * 55.0 * times) * np.exp(-times * 3.5)
        burst = np.random.default_rng(7).standard_normal(
            impact_len).astype(np.float32) * np.exp(-times * 7.0)
        burst = _apply_filter(
            burst[:, None], np.full(impact_len, 1800.0, dtype=np.float32),
            SR, True)[:, 0]
        mix[drop_at:drop_at + impact_len] += \
            (0.20 * thump + 0.07 * burst)[:, None]

    # Stronger drop production: sub, crash, reverse swell and drum fill.
    rng = np.random.default_rng(11)
    sub_len = min(int(1.8 * SR), len(mix) - drop_at)
    if sub_len > 0:
        times = np.arange(sub_len, dtype=np.float32) / SR
        sweep = 62.0 - 26.0 * np.minimum(times / 0.5, 1.0)
        sub = np.sin(2.0 * np.pi * np.cumsum(sweep) / SR) * np.exp(-times * 2.6)
        mix[drop_at:drop_at + sub_len] += (0.18 * sub)[:, None]
    crash_len = min(int(2.8 * SR), len(mix) - drop_at)
    if crash_len > 0:
        times = np.arange(crash_len, dtype=np.float32) / SR
        crash = rng.standard_normal(crash_len).astype(np.float32) * np.exp(-times * 2.2)
        crash = _apply_filter(
            crash[:, None], np.full(crash_len, 2600.0, dtype=np.float32),
            SR, True)[:, 0]
        mix[drop_at:drop_at + crash_len] += (0.11 * crash)[:, None]
    reverse_start = drop_at - int(4 * BEAT * SR)
    reverse_len = drop_at - reverse_start
    if reverse_len > 0:
        times = np.arange(reverse_len, dtype=np.float32) / SR
        swell = (times / max(times[-1], 1e-3)) ** 2
        reverse = rng.standard_normal(reverse_len).astype(np.float32) * swell
        reverse = _apply_filter(
            reverse[:, None],
            np.full(reverse_len, 3200.0, dtype=np.float32), SR, True)[:, 0]
        mix[reverse_start:drop_at] += (0.08 * reverse)[:, None]
    roll_start = drop_at - int(2 * BEAT * SR)
    hit_at = int(19.0 * 4 * BEAT * SR)
    hit = b_drums[hit_at:hit_at + int(0.06 * SR)]
    if len(hit) > 0:
        for k in range(8):
            at = roll_start + int(k * (drop_at - roll_start) / 8)
            gain = 0.30 + 0.50 * (k / 7.0)
            end = min(len(mix), at + len(hit))
            mix[at:end] += hit[:end - at] * gain
    out = finalise(mix, SR)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, SR, subtype="PCM_16")
    mp3 = out_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(out_path), "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)],
        check=True,
    )
    return out_path
