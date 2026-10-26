"""Beat-anchored audio rendering for DJSet-v1.

Differences from the v0 renderer that matter audibly:

* every position is a beat **index** into the track's real beat grid, so a
  track whose first beat is 19 s in no longer plays 19 s out of sync;
* deck B is time-stretched by ``median(B beat) / median(A beat)`` (the old
  ratio was inverted), then each beat is anchored to A's beat;
* the three bands are a Linkwitz-Riley 4th-order crossover, not one-pole
  filters, and the filter is a 2-pole state-variable filter with a smooth
  cutoff curve;
* deck B is loudness-matched to A over the whole transition, and the output
  is soft-limited and faded at the edges.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from numba import njit
from scipy.signal import butter, resample_poly, sosfilt

EQ_MIN_DB = -18.0
EQ_MAX_DB = 6.0
FILTER_Q = 0.8


# ------------------------------------------------------------------ audio io
def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(audio) < 2:
        return audio
    factor = gcd(int(sr_in), int(sr_out))
    return resample_poly(audio, int(sr_out) // factor,
                         int(sr_in) // factor, axis=0).astype(np.float32)


def load_track(path: str | Path, sr: int = 44100) -> np.ndarray:
    """Load a track as float32 ``(samples, channels)`` at ``sr``."""
    audio, file_sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32)
    if int(file_sr) != int(sr):
        audio = _resample(audio, int(file_sr), int(sr))
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return np.ascontiguousarray(audio)


def _slice(audio: np.ndarray, sr: int, start: float,
           duration: float) -> np.ndarray:
    length = max(1, int(round(duration * sr)))
    i0 = max(0, int(round(start * sr)))
    chunk = audio[i0:i0 + length]
    if len(chunk) < length:
        chunk = np.pad(chunk, ((0, length - len(chunk)), (0, 0)))
    return chunk.astype(np.float32)


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio
    if len(audio) < 2:
        return np.zeros((length, audio.shape[1]), dtype=np.float32)
    x = np.linspace(0.0, len(audio) - 1.0, length)
    out = np.empty((length, audio.shape[1]), dtype=np.float32)
    for channel in range(audio.shape[1]):
        out[:, channel] = np.interp(x, np.arange(len(audio)), audio[:, channel])
    return out


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder stretch, one channel at a time.  ``rate > 1`` speeds up."""
    if abs(rate - 1.0) < 1e-4 or len(audio) < 4:
        return audio
    out = np.empty((max(2, int(round(len(audio) / rate))), audio.shape[1]),
                   dtype=np.float32)
    for channel in range(audio.shape[1]):
        stretched = librosa.effects.time_stretch(
            np.ascontiguousarray(audio[:, channel]), rate=float(rate))
        out[:, channel] = _fit_length(
            stretched[:, None], len(out))[:, 0]
    return out


# -------------------------------------------------------------- processing
def _split(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three bands via two Linkwitz-Riley 4th-order splits.

    The sum of an LR4 low and high is allpass, so splitting the high side
    again keeps low + mid + high phase-coherent instead of comb-filtering at
    the 250 Hz / 2 kHz crossovers.
    """
    nyq = sr / 2.0

    def lr(signal: np.ndarray, fc: float, high: bool) -> np.ndarray:
        sos = butter(2, fc / nyq, btype="high" if high else "low",
                     output="sos")
        y = sosfilt(sos, signal, axis=0)
        return sosfilt(sos, y, axis=0).astype(np.float32)

    low = lr(x, 250.0, False)
    rest = lr(x, 250.0, True)
    mid = lr(rest, 2000.0, False)
    high = lr(rest, 2000.0, True)
    return low, mid, high


@njit(cache=True)
def _svf(x: np.ndarray, cutoff: np.ndarray, sr: float,
         highpass: bool, q: float) -> np.ndarray:
    """TPT state-variable filter with a per-sample cutoff."""
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(n):
        fc = cutoff[i]
        if fc < 20.0:
            fc = 20.0
        elif fc > sr * 0.45:
            fc = sr * 0.45
        g = np.tan(np.pi * fc / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if highpass:
            out[i] = x[i] - k * v1 - v2
        else:
            out[i] = v2
    return out


def _apply_filter(x: np.ndarray, cutoff: np.ndarray, sr: int,
                  highpass: bool) -> np.ndarray:
    out = np.empty_like(x)
    for channel in range(x.shape[1]):
        out[:, channel] = _svf(np.ascontiguousarray(x[:, channel]),
                               cutoff, float(sr), highpass, FILTER_Q)
    return out


def _rms_envelope(x: np.ndarray, sr: int, frame: int = 2048,
                  hop: int = 512) -> np.ndarray:
    mono = x.mean(axis=1) if x.ndim == 2 else x
    pad = max(0, frame - len(mono))
    if pad:
        mono = np.pad(mono, (0, pad))
    n = 1 + (len(mono) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = mono[idx]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12).astype(np.float32)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 2:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _warp(audio: np.ndarray, src_beats: np.ndarray,
          dst_beats: np.ndarray, sr: int) -> np.ndarray:
    """Map every output beat onto the corresponding source beat."""
    n = len(audio)
    t = np.arange(n, dtype=np.float64) / sr
    src_t = np.interp(t, dst_beats, src_beats) * sr
    src_idx = np.clip(src_t, 0, n - 1)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = (src_idx - lo)[:, None]
    return ((1.0 - frac) * audio[lo] + frac * audio[hi]).astype(np.float32)


# ------------------------------------------------------------------- render
def _eq_curve(action: np.ndarray) -> np.ndarray:
    action = np.clip(action, 0.0, 1.0)
    db = np.where(action <= 0.5, (action - 0.5) * 2.0 * EQ_MIN_DB,
                  (action - 0.5) * 2.0 * EQ_MAX_DB)
    return (10.0 ** (db / 20.0)).astype(np.float32)


def _filter_curves(filter_action: np.ndarray, sr: int) -> tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]:
    f = np.clip(filter_action, -1.0, 1.0)
    lp_cutoff = 20000.0 * (200.0 / 20000.0) ** np.maximum(-f, 0.0)
    hp_cutoff = 20.0 * (8000.0 / 20.0) ** np.maximum(f, 0.0)
    highpass = (f > 0).astype(np.float32)
    return lp_cutoff.astype(np.float32), hp_cutoff.astype(np.float32), highpass


def render_mix(env, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Render one DJSet-v1 episode to float32 stereo samples."""
    track_a = env.track_a
    track_b = env.track_b
    cue_a = int(env.cue_a)
    cue_b = int(env.cue_b)
    trace = env.trace
    if not trace:
        raise ValueError("no episode to render: run at least one step first")
    if "action" not in trace[0]:
        raise ValueError("DJSet-v1 traces need per-step 'action' arrays")

    n_steps = len(trace)
    ta = np.asarray(track_a["beat_times"], dtype=np.float64)
    tb = np.asarray(track_b["beat_times"], dtype=np.float64)
    beat_a = float(np.median(np.diff(ta)))
    beat_b = float(np.median(np.diff(tb)))
    rate = beat_b / beat_a                 # >1 speeds deck B up

    start_a = float(ta[cue_a])
    end_index = min(cue_a + n_steps, len(ta) - 1)
    duration = float(ta[end_index] - start_a)
    if duration <= 0.1:
        raise ValueError("transition has no duration")

    audio_a = load_track(env.data_dir / track_a["file"], sr)
    audio_b = load_track(env.data_dir / track_b["file"], sr)
    deck_a = _slice(audio_a, sr, start_a, duration)
    need_b = duration * rate
    deck_b = _slice(audio_b, sr, float(tb[cue_b]), need_b)
    deck_b = _time_stretch(deck_b, rate)
    deck_b = _fit_length(deck_b, len(deck_a))

    dst_rel = ta[cue_a:end_index + 1] - start_a
    src_rel = (tb[cue_b:cue_b + len(dst_rel)] - tb[cue_b]) / rate
    src_rel = src_rel - src_rel[0]
    if len(src_rel) == len(dst_rel) and np.all(np.diff(src_rel) > 0):
        if float(np.max(np.abs(src_rel - dst_rel))) > 0.004:
            deck_b = _warp(deck_b, src_rel, dst_rel, sr)

    n = len(deck_a)
    actions = np.asarray([np.asarray(t["action"], dtype=np.float32)
                          for t in trace], dtype=np.float32)
    action_t = dst_rel[:len(actions)]
    sample_t = np.arange(n, dtype=np.float64) / sr

    def curve(column: int) -> np.ndarray:
        values = np.interp(sample_t, action_t, actions[:, column])
        return values.astype(np.float32)

    fader = curve(0)
    eq_a = [_eq_curve(curve(1 + i)) for i in range(3)]
    eq_b = [_eq_curve(curve(4 + i)) for i in range(3)]
    filt = curve(7)

    a_low, a_mid, a_high = _split(deck_a, sr)
    b_low, b_mid, b_high = _split(deck_b, sr)
    a_mix = (a_low * eq_a[0][:, None] + a_mid * eq_a[1][:, None]
             + a_high * eq_a[2][:, None])
    b_mix = (b_low * eq_b[0][:, None] + b_mid * eq_b[1][:, None]
             + b_high * eq_b[2][:, None])

    # Loudness: match deck B to deck A's rolling RMS, then smooth, so the
    # crossfader does not also change the mix's loudness.
    env_a = _smooth(_rms_envelope(deck_a, sr), 24)
    env_b = _smooth(_rms_envelope(deck_b, sr), 24)
    match = np.clip(env_a / (env_b + 1e-6), 0.5, 2.0)
    match = _smooth(match, 32)
    match_t = np.linspace(0.0, duration, len(match))
    match_curve = np.interp(sample_t, match_t, match).astype(np.float32)
    b_mix = b_mix * match_curve[:, None]

    lp_cutoff, hp_cutoff, highpass = _filter_curves(filt, sr)
    lp = _apply_filter(b_mix, lp_cutoff, sr, False)
    hp = _apply_filter(b_mix, hp_cutoff, sr, True)
    b_filt = np.where((filt <= 0.0)[:, None], lp, hp).astype(np.float32)

    gain_a = np.cos(fader * np.pi / 2.0)[:, None]
    gain_b = np.sin(fader * np.pi / 2.0)[:, None]
    mix = gain_a * a_mix + gain_b * b_filt

    # Slow master gain riding: a DJ keeps the room level steady while tracks
    # change loudness.  The 1.5 s smoothing leaves transients alone and only
    # corrects the macro level, clamped to +/-3 dB.
    mono_env = _smooth(_rms_envelope(mix, sr), 128)
    target = float(np.median(mono_env)) if len(mono_env) else 0.0
    if target > 1e-5:
        ride = np.clip(target / (mono_env + 1e-6), 0.7, 1.5)
        ride = _smooth(ride, 96)
        ride_t = np.linspace(0.0, duration, len(ride))
        mix = mix * np.interp(sample_t, ride_t, ride)[:, None].astype(
            np.float32)
    return mix.astype(np.float32), sr


def finalise(mix: np.ndarray, sr: int, fade_in: bool = True,
             fade_out: bool = True) -> np.ndarray:
    mix = np.tanh(np.asarray(mix, dtype=np.float32) * 1.05)
    fade = max(1, int(0.02 * sr))
    if fade_in and len(mix) > fade:
        mix[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    if fade_out and len(mix) > fade:
        mix[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.95).astype(np.float32)


def write_wav(audio: np.ndarray, sr: int, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, int(sr), subtype="PCM_16")
    return out_path


def render_transition(env, out_path: str | Path, sr: int = 44100) -> Path:
    mix, sr = render_mix(env, sr)
    return write_wav(finalise(mix, sr), sr, out_path)


def join_mixes(mixes: list[np.ndarray], sr: int, gap: float = 0.0) -> np.ndarray:
    if not mixes:
        raise ValueError("a set needs at least one transition")
    pieces = []
    silence = np.zeros((int(gap * sr), 2), dtype=np.float32)
    for index, mix in enumerate(mixes):
        if index and gap:
            pieces.append(silence)
        pieces.append(mix)
    return np.concatenate(pieces, axis=0)
