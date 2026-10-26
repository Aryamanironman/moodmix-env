"""Tests for automatic mix-to-track alignment."""

from __future__ import annotations

import numpy as np
import pytest

from moodmix_env.alignment import ALIGN_HOP, align, refine_seconds

SR = 22050


def _melody(seconds: float = 30.0, sr: int = SR) -> np.ndarray:
    """A chromatic melody: distinctive enough for chroma-based alignment."""
    rng = np.random.default_rng(0)
    notes = rng.integers(0, 12, size=int(seconds * 2) + 1)
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    step = int(0.5 * sr)
    t = np.arange(step) / sr
    for i, note in enumerate(notes):
        start = i * step
        if start >= len(out):
            break
        freq = 220.0 * 2 ** (note / 12)
        chunk = 0.4 * np.sin(2 * np.pi * freq * t)
        end = min(len(out), start + step)
        out[start:end] += chunk[:end - start]
    return out


def test_align_finds_entry_point_and_tempo():
    track = _melody(30.0)
    start = 8.0
    mix = track[int(start * SR):int((start + 12.0) * SR)]
    result = align(track, mix, sr=SR)
    offset = result.intercept * result.hop / SR
    assert abs(offset - start) < 1.0
    assert abs(result.slope - 1.0) < 0.05


def test_align_survives_a_silent_lead_in():
    # A mix can open with digital silence (deck B not yet cued).  Those frames
    # have no pitch class, and a NaN in the chroma matrix used to break DTW
    # outright; the flat-vector substitution must keep the result finite.
    track = _melody(30.0)
    body = track[int(4.0 * SR):int(14.0 * SR)]
    mix = np.concatenate([np.zeros(int(3.0 * SR), dtype=np.float32), body])

    result = align(track, mix, sr=SR)
    offset = result.intercept * result.hop / SR
    assert np.isfinite(offset)
    assert np.isfinite(result.score)
    assert 0.0 <= offset < 30.0


def test_align_accepts_a_focus_mask_built_from_samples():
    # Callers build masks from sample counts, which is one frame off the chroma
    # matrix; a dropped mask sends the offset search over the whole track.
    track = _melody(30.0)
    start = 8.0
    mix = track[int(start * SR):int((start + 12.0) * SR)]
    n_frames = int(np.ceil(len(mix) / ALIGN_HOP))
    mask = np.zeros(n_frames, dtype=bool)
    mask[: n_frames // 3] = True

    padded = align(track, mix, sr=SR, focus=mask)
    exact = align(track, mix, sr=SR, focus=mask[: (len(mix) // ALIGN_HOP) - 1])
    assert abs(padded.intercept - exact.intercept) < 2.0


def test_align_honours_a_known_playback_rate():
    track = _melody(30.0)
    mix = track[int(8.0 * SR):int(20.0 * SR)]
    result = align(track, mix, sr=SR, rate=1.04)
    assert result.slope == pytest.approx(1.04)


def test_refine_seconds_improves_coarse_alignment():
    track = _melody(30.0)
    start = 8.0
    mix = track[int(start * SR):int((start + 12.0) * SR)]
    refined = refine_seconds(track, mix, SR, 1.0, start + 0.7, hop=ALIGN_HOP)
    assert abs(refined - start) < 0.25
