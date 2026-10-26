"""Tests for recovering mixer controls from a mixed signal."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recover_transition as recovery  # noqa: E402


def test_eq_bin_maps_gain_multipliers():
    assert recovery._eq_bin(0.126) == 0      # -18 dB cut
    assert recovery._eq_bin(1.0) == 1        # neutral
    assert recovery._eq_bin(1.995) == 2      # +6 dB boost


def test_eq_from_gains_removes_fader_gain():
    # -18 dB cut on low, neutral elsewhere, at half fader travel.
    fader_gain = float(np.sin(np.pi / 4))
    gains = [0.126 * fader_gain, fader_gain, fader_gain]
    assert recovery._eq_from_gains(gains, fader_gain) == [0, 1, 1]

    # +6 dB boost on the high band
    gains = [fader_gain, fader_gain, 1.995 * fader_gain]
    assert recovery._eq_from_gains(gains, fader_gain) == [1, 1, 2]


def test_solve_nonneg_matches_known_mix():
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(size=400))
    b = np.abs(rng.normal(size=400))
    mix = 0.7 * a + 0.3 * b
    ga, gb = recovery._solve_nonneg(mix, a, b)
    assert abs(ga - 0.7) < 1e-6
    assert abs(gb - 0.3) < 1e-6


def test_solve_nonneg_clips_negative_gain():
    a = np.ones(200)
    b = np.linspace(1, 2, 200)
    ga, gb = recovery._solve_nonneg(2.0 * b, a, b)
    assert ga < 1e-6      # numerical noise, not a real contribution
    assert gb > 0


def test_estimate_faders_interpolates_silent_beats():
    # Deck A alone, a silent beat, then deck B alone.
    raw = [([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
           ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
           ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])]
    ref = [1.0, 1.0, 1.0]
    faders = recovery._estimate_faders(raw, ref, ref)
    assert faders[0] < 0.1
    assert faders[-1] > 0.9
    assert 0.1 <= faders[1] <= 0.9      # interpolated, not a 0.5 fallback spike


def test_compare_scores_against_truth():
    trace = {"steps": [{"crossfader": 0.0, "eq_a": [1, 1, 1], "eq_b": [1, 1, 1]},
                       {"crossfader": 1.0, "eq_a": [0, 1, 1], "eq_b": [1, 1, 2]}]}
    truth = [{"crossfader": 0.0, "eq_a": [1, 1, 1], "eq_b": [1, 1, 1]},
             {"crossfader": 1.0, "eq_a": [1, 1, 1], "eq_b": [1, 1, 1]}]
    scores = recovery.compare(trace, truth)
    assert scores["fader_mae_bins"] == 0.0
    assert scores["eq_accuracy"] == 0.5
    assert scores["eq_band_accuracy"] == 5 / 6
