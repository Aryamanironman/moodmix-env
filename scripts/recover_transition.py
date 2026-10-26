"""Recover mixer controls from a DJ mix, with no manual cue points.

A mix is just audio: the fader and EQ moves are hidden inside it.  This script

1. aligns each source track to the mix (subsequence DTW, `moodmix_env.alignment`),
   which recovers where the track enters and how far it was pitched in tempo;
2. models each band of the mix as a non-negative combination of the two decks'
   spectra, beat by beat;
3. turns the fitted gains into a crossfader position and per-deck EQ moves.

That is the bridge needed to learn from real DJ sets, where only the finished
mix exists.

    python scripts/recover_transition.py --mix set.wav \
        --track-a a.mp3 --track-b b.mp3 --bpm 124 --out trace.json

Pass `--truth trace.json` (an env trace with `crossfader` / `eq_a` / `eq_b`) to
score the recovery.  `--offset-a` / `--offset-b` force a manual entry point for
a deck whose alignment looks wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.alignment import ALIGN_HOP, align, refine_seconds  # noqa: E402

N_FFT = 2048
HOP = 512
FADER_BINS = 21
BAND_EDGES = (250.0, 2000.0)


def _load(path: Path, sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def _bands(sr: int) -> list[np.ndarray]:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    return [
        freqs < BAND_EDGES[0],
        (freqs >= BAND_EDGES[0]) & (freqs < BAND_EDGES[1]),
        freqs >= BAND_EDGES[1],
    ]


def _solve_nonneg(mix: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Least-squares gains for mix ~= ga*a + gb*b with ga, gb >= 0."""
    matrix = np.stack([a, b], axis=1)
    coef, *_ = np.linalg.lstsq(matrix, mix, rcond=None)
    ga, gb = float(coef[0]), float(coef[1])
    if ga < 0 and gb < 0:
        return 0.0, 0.0
    if ga < 0:
        denom = float(b @ b)
        return 0.0, float(max(0.0, float(mix @ b) / denom)) if denom else 0.0
    if gb < 0:
        denom = float(a @ a)
        return (float(max(0.0, float(mix @ a) / denom)) if denom else 0.0), 0.0
    return ga, gb


def _eq_bin(ratio: float) -> int:
    if ratio < 0.5:
        return 0        # cut  (-18 dB = 0.126)
    if ratio > 1.5:
        return 2        # boost (+6 dB = 1.995)
    return 1            # neutral


def _eq_from_gains(gains: list[float], fader_gain: float) -> list[int]:
    """Turn band gains into EQ positions.

    The least-squares gain is already an absolute amplitude relative to the
    source, so dividing by the equal-power fader gain leaves the EQ multiplier
    itself.  A solo-section reference must not be used: a boost still engaged
    when that deck ends up alone would cancel itself out.
    """
    scale = max(fader_gain, 0.05)
    return [_eq_bin(g / scale) for g in gains]


def _estimate_faders(raw: list[tuple[list[float], list[float]]],
                      ref_a: list[float], ref_b: list[float]) -> np.ndarray:
    """Crossfader per beat from calibrated per-band gains.

    Each band votes on the fader after dividing out the source's own spectral
    balance, and the median vote wins.  Beats quieter than 5% of the median
    energy (an intro, a breakdown, the fade at the edges) carry no information
    and are interpolated from their neighbours instead of guessing 0.5.
    """
    energy = np.array([sum(ga) + sum(gb) for ga, gb in raw], dtype=float)
    reference_energy = float(np.median(energy[energy > 0])) if np.any(energy > 0) else 0.0
    votes = []
    for gains_a, gains_b in raw:
        bands = []
        for i in range(3):
            fa = gains_a[i] / ref_a[i] if ref_a[i] > 1e-9 else 0.0
            fb = gains_b[i] / ref_b[i] if ref_b[i] > 1e-9 else 0.0
            if fa + fb > 1e-9:
                bands.append(2.0 / np.pi * np.arctan2(fb, fa))
        votes.append(float(np.median(bands)) if bands else np.nan)

    faders = np.array(votes, dtype=float)
    if reference_energy > 0:
        weak = energy < 0.05 * reference_energy
        faders[weak] = np.nan
    valid = np.isfinite(faders)
    if valid.sum() == 0:
        return np.full(len(faders), 0.5)
    if not valid.all():
        idx = np.arange(len(faders))
        faders = np.interp(idx, idx[valid], faders[valid])
    if len(faders) >= 3:
        padded = np.pad(faders, 1, mode="edge")
        faders = np.array([np.median(padded[i:i + 3]) for i in range(len(faders))])
    return faders


def _solo_reference(raw: list[tuple[list[float], list[float]]], deck: int) -> list[float]:
    """Per-band gain while this deck is essentially alone in the mix."""
    solo = []
    for gains_a, gains_b in raw:
        total_a, total_b = sum(gains_a), sum(gains_b)
        other = total_b if deck == 0 else total_a
        own = total_a if deck == 0 else total_b
        if own > 3.0 * other and own > 1e-9:
            solo.append(gains_a if deck == 0 else gains_b)
    if solo:
        return list(np.median(np.array(solo), axis=0))
    best = max(raw, key=lambda pair: sum(pair[deck]))
    return list(best[deck])


def _slice(spec: np.ndarray, mask: np.ndarray, start: float,
           length: int) -> np.ndarray:
    """Average a band over a frame window, zero-padded outside the track."""
    begin = int(round(start))
    lo = max(0, begin)
    hi = min(spec.shape[1], begin + length)
    if hi <= lo:
        return np.zeros(int(mask.sum()), dtype=float)
    block = spec[mask][:, lo:hi].sum(axis=1)
    return block / max(1, length)


def _band_gains(mix_spec, a_spec, b_spec, al_a, al_b, bands,
                frames_per_beat: int, total_steps: int):
    """Per-beat per-band gains for both decks under the given alignments."""
    hop_ratio = ALIGN_HOP / HOP
    raw: list[tuple[list[float], list[float]]] = []
    for step in range(total_steps):
        start = step * frames_per_beat
        end = min(mix_spec.shape[1], start + frames_per_beat)
        length = end - start
        if length <= 0:
            break
        a_start = al_a.source_frame(start / hop_ratio) * hop_ratio
        b_start = al_b.source_frame(start / hop_ratio) * hop_ratio
        gains_a, gains_b = [], []
        for mask in bands:
            mix_band = mix_spec[mask, start:end].mean(axis=1)
            a_band = _slice(a_spec, mask, a_start, length)
            b_band = _slice(b_spec, mask, b_start, length)
            ga, gb = _solve_nonneg(mix_band, a_band, b_band)
            gains_a.append(ga)
            gains_b.append(gb)
        raw.append((gains_a, gains_b))
    return raw


def _focus_mask(raw, deck: int, frames_per_beat: int, n_frames: int) -> np.ndarray:
    """Mix frames where this deck clearly dominates the blend."""
    mask = np.zeros(n_frames, dtype=bool)
    for step, (gains_a, gains_b) in enumerate(raw):
        total_a, total_b = sum(gains_a), sum(gains_b)
        own, other = (total_a, total_b) if deck == 0 else (total_b, total_a)
        if own > 2.0 * other and own > 1e-9:
            lo = step * frames_per_beat
            hi = min(n_frames, lo + frames_per_beat)
            mask[lo:hi] = True
    return mask


def recover(
    mix_path: Path,
    track_a: Path,
    track_b: Path,
    bpm: float,
    offset_a: float | None = None,
    offset_b: float | None = None,
    sr: int = 22050,
    steps: int | None = None,
    bpm_a: float | None = None,
    bpm_b: float | None = None,
) -> dict:
    mix = _load(mix_path, sr)
    a = _load(track_a, sr)
    b = _load(track_b, sr)

    # Alignment maps mix time -> source time.  A transition opens on deck A and
    # closes on deck B, so seed each alignment on its own end of the mix: a
    # track that is silent for most of the recording otherwise matches whatever
    # happens to be loud later.  A manual offset pins the entry point instead
    # and assumes the track plays at its own tempo.
    n_align_frames = int(np.ceil(len(mix) / ALIGN_HOP))
    opening = np.zeros(n_align_frames, dtype=bool)
    opening[:max(1, n_align_frames // 4)] = True
    closing = np.zeros(n_align_frames, dtype=bool)
    closing[int(0.75 * n_align_frames):] = True
    # A synced deck plays at mix tempo / its own tempo, and chroma cannot
    # resolve a few percent of pitch shift on a short clip.  When the source
    # tempi are known, hand the alignment the rate instead of letting it guess.
    rate_a = bpm / bpm_a if bpm_a else None
    rate_b = bpm / bpm_b if bpm_b else None
    al_a = align(a, mix, sr=sr, focus=opening, rate=rate_a)
    al_b = align(b, mix, sr=sr, focus=closing, rate=rate_b)
    if offset_a is not None:
        al_a.slope, al_a.intercept = 1.0, offset_a * sr / al_a.hop
    else:
        fine_a = refine_seconds(a, mix, sr, al_a.slope,
                                al_a.intercept * al_a.hop / sr, focus=opening)
        al_a.intercept = fine_a * sr / al_a.hop
    if offset_b is not None:
        al_b.slope, al_b.intercept = 1.0, offset_b * sr / al_b.hop
    else:
        fine_b = refine_seconds(b, mix, sr, al_b.slope,
                                al_b.intercept * al_b.hop / sr, focus=closing)
        al_b.intercept = fine_b * sr / al_b.hop

    mix_spec = np.abs(librosa.stft(mix, n_fft=N_FFT, hop_length=HOP))
    a_spec = np.abs(librosa.stft(a, n_fft=N_FFT, hop_length=HOP))
    b_spec = np.abs(librosa.stft(b, n_fft=N_FFT, hop_length=HOP))
    bands = _bands(sr)
    beat_seconds = 60.0 / bpm
    frames_per_beat = max(1, int(round(beat_seconds * sr / HOP)))
    total_steps = steps or int(np.floor(mix_spec.shape[1] / frames_per_beat))

    # A second, gain-weighted re-alignment was tried and made results worse:
    # on a repetitive track the correlation peaks on a later, similar phrase
    # (7.15s instead of 0s on the validation mix), so the structural seed is
    # kept as the single source of truth.
    raw = _band_gains(mix_spec, a_spec, b_spec, al_a, al_b, bands,
                      frames_per_beat, total_steps)

    ref_a = _solo_reference(raw, deck=0)
    ref_b = _solo_reference(raw, deck=1)
    faders = _estimate_faders(raw, ref_a, ref_b)

    trace: list[dict] = []
    for step, (gains_a, gains_b) in enumerate(raw):
        total_a = float(sum(gains_a))
        total_b = float(sum(gains_b))
        fader = float(np.clip(faders[step], 0.0, 1.0))
        eq_a = _eq_from_gains(gains_a, float(np.cos(fader * np.pi / 2)))
        eq_b = _eq_from_gains(gains_b, float(np.sin(fader * np.pi / 2)))
        trace.append({
            "step": step + 1,
            "crossfader": round(fader, 4),
            "fader_bin": int(round(fader * (FADER_BINS - 1))),
            "eq_a": eq_a,
            "eq_b": eq_b,
            "gain_a": round(total_a, 6),
            "gain_b": round(total_b, 6),
        })
    return {
        "bpm": bpm,
        "bands_hz": [[0, BAND_EDGES[0]], [BAND_EDGES[0], BAND_EDGES[1]],
                     [BAND_EDGES[1], sr / 2]],
        "alignment": {
            "a": {"slope": round(al_a.slope, 5),
                  "offset_seconds": round(al_a.intercept * al_a.hop / sr, 3),
                  "dtw_cost": round(al_a.score, 5) if np.isfinite(al_a.score) else None},
            "b": {"slope": round(al_b.slope, 5),
                  "offset_seconds": round(al_b.intercept * al_b.hop / sr, 3),
                  "dtw_cost": round(al_b.score, 5) if np.isfinite(al_b.score) else None},
        },
        "steps": trace,
    }


def compare(trace: dict, truth: list[dict]) -> dict:
    n = min(len(trace["steps"]), len(truth))
    est = np.array([trace["steps"][i]["crossfader"] for i in range(n)])
    ref = np.array([float(truth[i]["crossfader"]) for i in range(n)])
    bins = np.abs(est - ref) * (FADER_BINS - 1)
    eq_vectors = 0
    eq_bands = 0
    eq_total = 0
    band_total = 0
    for i in range(n):
        for key in ("eq_a", "eq_b"):
            if key not in truth[i]:
                continue
            got = list(trace["steps"][i][key])
            want = list(truth[i][key])
            eq_vectors += int(got == want)
            eq_total += 1
            eq_bands += sum(int(a == b) for a, b in zip(got, want))
            band_total += len(want)
    return {
        "steps": n,
        "fader_mae_bins": float(bins.mean()),
        "fader_p90_bins": float(np.percentile(bins, 90)),
        "fader_corr": float(np.corrcoef(est, ref)[0, 1]) if n > 2 else float("nan"),
        "eq_accuracy": eq_vectors / eq_total if eq_total else float("nan"),
        "eq_band_accuracy": eq_bands / band_total if band_total else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--track-a", type=Path, required=True)
    parser.add_argument("--track-b", type=Path, required=True)
    parser.add_argument("--bpm", type=float, required=True,
                        help="tempo of the mix, used to slice it into beats")
    parser.add_argument("--offset-a", type=float, default=None,
                        help="force entry point in seconds instead of aligning deck A")
    parser.add_argument("--offset-b", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <mix>.trace.json")
    parser.add_argument("--truth", type=Path, default=None,
                        help="env trace JSON to score the recovery against")
    args = parser.parse_args()

    trace = recover(
        args.mix, args.track_a, args.track_b, args.bpm,
        offset_a=args.offset_a, offset_b=args.offset_b,
        sr=args.sample_rate, steps=args.steps,
    )
    out = args.out or args.mix.with_suffix(".trace.json")
    out.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    alignment = trace["alignment"]
    print(f"aligned A at {alignment['a']['offset_seconds']:.2f}s "
          f"(rate {alignment['a']['slope']:.3f}), "
          f"B at {alignment['b']['offset_seconds']:.2f}s "
          f"(rate {alignment['b']['slope']:.3f})")
    print(f"recovered {len(trace['steps'])} beats -> {out}")

    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        if isinstance(truth, dict) and "steps" in truth:
            truth = truth["steps"]
        scores = compare(trace, truth)
        print(f"fader error: {scores['fader_mae_bins']:.2f} bins mean, "
              f"{scores['fader_p90_bins']:.2f} p90  (of {FADER_BINS} positions)")
        print(f"fader correlation: {scores['fader_corr']:.3f}")
        print(f"EQ: {scores['eq_accuracy']:.1%} of settings exact, "
              f"{scores['eq_band_accuracy']:.1%} of individual bands correct "
              f"over {scores['steps']} beats")


if __name__ == "__main__":
    main()
