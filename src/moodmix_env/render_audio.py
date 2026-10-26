"""Render a finished MoodMix episode to a WAV file.

The simulation works on features, but the same control trace can be applied to
the demo audio so you can hear the transition.  This is deliberately simple:
3-band one-pole EQ, equal-power crossfader, filter switch and synthesized drum
hits.  It is an evaluation/demo utility, not a mastering chain.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

from .env import EQ_GAINS


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Re-rate a signal, so a crate of real songs can be mixed at all.

    A generated demo crate is one sample rate by construction; a folder of
    downloaded tracks is not, and refusing to render unless every file matches
    made the browser demo unusable on a real crate.
    """
    if sr_in == sr_out or len(audio) < 2:
        return audio
    factor = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // factor, int(sr_in) // factor
    return resample_poly(audio, up, down).astype(np.float32)


def _read_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, file_sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr is not None and int(file_sr) != int(sr):
        audio = _resample(audio, int(file_sr), int(sr))
        file_sr = int(sr)
    return audio, int(file_sr)


class _BandSplit:
    """Two one-pole low-passes split a signal into low / mid / high."""

    def __init__(self, sr: int) -> None:
        self.a_low = 1.0 - np.exp(-2.0 * np.pi * 250.0 / sr)
        self.a_high = 1.0 - np.exp(-2.0 * np.pi * 2000.0 / sr)
        self.zi_low = np.zeros(1, dtype=np.float64)
        self.zi_high = np.zeros(1, dtype=np.float64)

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low, self.zi_low = lfilter(
            [self.a_low], [1.0, self.a_low - 1.0], x, zi=self.zi_low
        )
        low_mid, self.zi_high = lfilter(
            [self.a_high], [1.0, self.a_high - 1.0], x, zi=self.zi_high
        )
        return low, low_mid - low, x - low_mid


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff / sr)
    return lfilter([a], [1.0, a - 1.0], x).astype(np.float32)


def _normalise(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak).astype(np.float32)


def _drum_hit(kind: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A soft, filtered pad hit.

    Differentiated white noise is the obvious way to synthesise a snare and it
    is also the way to make a mix sound broken: measured over a rendered
    transition it accounted for every sample-to-sample jump above 0.5, while
    the music underneath had none.  The noise is low-passed and the hits are
    normalised so the pad sits under the track instead of on top of it.
    """
    if kind == 1:  # kick
        t = np.arange(int(0.32 * sr)) / sr
        freq = 45.0 + 95.0 * np.exp(-t / 0.035)
        return _normalise(np.sin(2 * np.pi * np.cumsum(freq) / sr)
                          * np.exp(-t / 0.16))
    if kind == 2:  # snare
        t = np.arange(int(0.24 * sr)) / sr
        noise = _lowpass(rng.standard_normal(len(t)), sr, 5000.0)
        body = 0.5 * np.sin(2 * np.pi * 185.0 * t)
        return _normalise((0.6 * noise + body) * np.exp(-t / 0.07))
    if kind == 3:  # hat
        t = np.arange(int(0.08 * sr)) / sr
        noise = _lowpass(rng.standard_normal(len(t)), sr, 9000.0)
        return _normalise(noise * np.exp(-t / 0.018))
    if kind == 4:  # clap: three short noise bursts
        t = np.arange(int(0.28 * sr)) / sr
        noise = _lowpass(rng.standard_normal(len(t)), sr, 6000.0)
        env = np.zeros(len(t))
        for delay in (0.0, 0.012, 0.024):
            env += np.exp(-np.maximum(t - delay, 0.0) / 0.018)
        return _normalise(noise * env)
    return np.zeros(1, dtype=np.float32)


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _source_slice(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(1, dtype=np.float32)
    if start >= len(audio):
        return np.zeros(length, dtype=np.float32)
    chunk = audio[max(0, start):start + length]
    if len(chunk) < length:
        chunk = np.pad(chunk, (0, length - len(chunk)))
    return chunk.astype(np.float32)


def _stretch(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio
    if len(audio) < 2:
        return np.zeros(length, dtype=np.float32)
    x = np.linspace(0.0, len(audio) - 1.0, length)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Change tempo while preserving pitch.

    Deck B is stretched once for the whole transition.  Stretching per beat
    instead makes every block boundary a phase discontinuity, which is what
    turns a beatmatched mix into a wobbly, artifact-ridden one.
    """
    if abs(rate - 1.0) < 1e-3 or len(audio) < 2:
        return audio
    try:
        import librosa
    except ImportError:  # keep the demo runnable without the library extra
        return _stretch(audio, max(2, int(round(len(audio) / rate))))
    return librosa.effects.time_stretch(audio, rate=float(rate)).astype(np.float32)


def render_mix(env, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Render one episode to samples, before fades and peak normalisation.

    Returning samples rather than a file is what lets a set be one continuous
    mix: normalising each transition on its own would put a loudness step at
    every hand-over, which is exactly the seam a DJ set must not have.
    """
    if not env.trace:
        raise ValueError("no episode to render: run at least one step first")

    audio_a, sr_a = _read_mono(env.data_dir / env.track_a["file"])
    # Deck A sets the rate; deck B is resampled onto it when the two files
    # disagree (any real crate will).
    sr = int(sr) if sr else sr_a
    if sr_a != sr:
        audio_a = _resample(audio_a, sr_a, sr)
        sr_a = sr
    audio_b, sr_b = _read_mono(env.data_dir / env.track_b["file"], sr=sr)
    sr = sr_a

    bpm_a = float(env.track_a["bpm"])
    bpm_b_natural = float(env.track_b["bpm"])
    beat_a = 60.0 / bpm_a
    # Put deck B on deck A's beat grid before slicing.  The environment
    # advances pos_b in deck-A beats once synced, so this mapping is exact.
    audio_b = _time_stretch(audio_b, bpm_b_natural / bpm_a)

    # Sample offsets are cumulative, not `beat * i`: at 95.7 BPM a beat is
    # 0.6266 s, which is not a whole number of samples, so rounding every block
    # independently repeats or drops a sample at each boundary.
    offsets = [int(round(i * beat_a * sr)) for i in range(len(env.trace) + 1)]
    total = offsets[-1]

    # Loudness trim: a DJ matches the incoming deck to the outgoing one before
    # bringing it in.  Without it the crossfade itself changes the level in
    # steps whenever the two tracks were mastered differently.
    span_a = int(round(float(env.trace[0]["pos_a"]) * beat_a * sr))
    span_b = int(round(float(env.trace[0]["pos_b"]) * beat_a * sr))
    level_a = _rms(audio_a[span_a:span_a + total])
    level_b = _rms(audio_b[span_b:span_b + total])
    if level_b > 1e-4 and level_a > 1e-4:
        audio_b = audio_b * float(np.clip(level_a / level_b, 0.25, 4.0))

    mix_out = np.zeros(total, dtype=np.float32)

    band_a = _BandSplit(sr)
    band_b = _BandSplit(sr)
    rng = np.random.default_rng(0)
    prev_x = 0.0
    prev_eq_a = np.ones(3, dtype=np.float32)
    prev_eq_b = np.ones(3, dtype=np.float32)
    prev_filter = 0

    for i, t in enumerate(env.trace):
        lo, hi = offsets[i], offsets[i + 1]
        block = hi - lo
        ramp = np.linspace(0.0, 1.0, block, endpoint=False,
                           dtype=np.float32)[:, None]
        pos_a = float(t["pos_a"])
        pos_b = float(t["pos_b"])
        start_a = int(round(pos_a * beat_a * sr))
        src_a = _source_slice(audio_a, start_a, block)
        src_b = _source_slice(audio_b, int(round(pos_b * beat_a * sr)), block)

        low_a, mid_a, high_a = band_a.process(src_a)
        low_b, mid_b, high_b = band_b.process(src_b)
        eq_a = EQ_GAINS[[int(v) for v in t["eq_a"]]]
        eq_b = EQ_GAINS[[int(v) for v in t["eq_b"]]]
        gain_a_eq = prev_eq_a + (eq_a - prev_eq_a) * ramp
        gain_b_eq = prev_eq_b + (eq_b - prev_eq_b) * ramp

        # Filter moves are ramped too, so a switch does not click.
        filt = int(t["filter"])
        lp = 0.15 if filt == 1 else 1.0
        hp = 0.12 if filt == 2 else 1.0
        lp_prev = 0.15 if prev_filter == 1 else 1.0
        hp_prev = 0.12 if prev_filter == 2 else 1.0
        lp_ramp = lp_prev + (lp - lp_prev) * ramp
        hp_ramp = hp_prev + (hp - hp_prev) * ramp
        high_b = high_b[:, None] * lp_ramp
        low_b = low_b[:, None] * hp_ramp

        fader = np.linspace(prev_x, float(t["crossfader"]), block, endpoint=False)
        gain_a = np.cos(fader * np.pi / 2)
        gain_b = np.sin(fader * np.pi / 2)
        block_mix = (
            gain_a * (low_a * gain_a_eq[:, 0] + mid_a * gain_a_eq[:, 1]
                      + high_a * gain_a_eq[:, 2])
            + gain_b * (low_b[:, 0] * gain_b_eq[:, 0] + mid_b * gain_b_eq[:, 1]
                        + high_b[:, 0] * gain_b_eq[:, 2])
        )

        drum = int(t["drum"])
        if drum:
            hit = _drum_hit(drum, sr, rng)[:block]
            block_mix[:len(hit)] += 0.18 * hit

        mix_out[lo:hi] = block_mix
        prev_x = float(t["crossfader"])
        prev_eq_a = eq_a
        prev_eq_b = eq_b
        prev_filter = filt

    return mix_out, sr


def finalise(mix: np.ndarray, sr: int, fade_in: bool = True,
             fade_out: bool = True) -> np.ndarray:
    """Soft-limit, fade the ends and set the peak."""
    mix = np.tanh(np.asarray(mix, dtype=np.float32) * 1.15)
    fade = max(1, int(0.02 * sr))
    if fade_in and len(mix) > fade:
        mix[:fade] *= np.linspace(0, 1, fade)
    if fade_out and len(mix) > fade:
        mix[-fade:] *= np.linspace(1, 0, fade)
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.95).astype(np.float32)


def write_wav(audio: np.ndarray, sr: int, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, int(sr), subtype="PCM_16")
    return out_path


def render_transition(env, out_path: str | Path, sr: int | None = None) -> Path:
    """Apply env.trace to the two tracks and write a mono WAV."""
    mix, sr = render_mix(env, sr)
    return write_wav(finalise(mix, sr), sr, out_path)


def render_set(mixes: list[np.ndarray], sr: int, out_path: str | Path) -> Path:
    """Join rendered transitions into one file with no seam between them."""
    if not mixes:
        raise ValueError("a set needs at least one rendered transition")
    joined = np.concatenate([np.asarray(m, dtype=np.float32) for m in mixes])
    return write_wav(finalise(joined, sr), sr, out_path)
