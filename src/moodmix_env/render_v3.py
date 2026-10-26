"""Beat-anchored audio rendering for DJSet-v3.

The v2 renderer plus drum-stem gain and beat-synced echo sends.  Drums are
the percussive component from an HPSS split, so the model can drop or swap
the drum pattern without touching the melodic and vocal content.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .render_v2 import (
    _apply_filter, _eq_curve, _filter_curves, _fit_length, _rms_envelope,
    _slice,
    _smooth, _smooth_steps, _split, _time_stretch, _warp, finalise, load_track,
    write_wav,
)


def _drum_split(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a stereo deck into percussive and harmonic parts."""
    drums = np.zeros_like(audio, dtype=np.float32)
    tonal = np.zeros_like(audio, dtype=np.float32)
    for channel in range(audio.shape[1]):
        percussive, harmonic = librosa.effects.hpss(
            np.ascontiguousarray(audio[:, channel]), margin=2.0)
        drums[:, channel] = _fit_length(percussive[:, None],
                                        len(audio))[:, 0]
        tonal[:, channel] = _fit_length(harmonic[:, None],
                                        len(audio))[:, 0]
    return drums, tonal


def _echo_taps(audio: np.ndarray, sr: int, delay_seconds: float,
               wet: np.ndarray, feedback: float = 0.34,
               taps: int = 4, tail_seconds: float = 1.5
               ) -> np.ndarray:
    """Beat-synced multi-tap echo with a short tail past the end."""
    if float(np.max(wet)) <= 1e-4:
        return np.zeros_like(audio)
    delay = max(1, int(round(delay_seconds * sr)))
    tail = max(1, int(round(tail_seconds * sr)))
    padded = np.pad(audio, ((0, tail), (0, 0)))
    wet_padded = np.pad(wet, (0, tail), mode="edge")
    out = np.zeros_like(padded)
    for tap in range(1, taps + 1):
        shift = delay * tap
        if shift >= len(padded):
            break
        out[shift:] += (feedback ** (tap - 1)) * padded[:-shift]
    return (out * wet_padded[:, None] * 0.65).astype(np.float32)


def render_mix(env, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Render one DJSet-v3 episode to float32 stereo samples."""
    track_a = env.track_a
    track_b = env.track_b
    cue_a = int(env.cue_a)
    cue_b = int(env.cue_b)
    trace = env.trace
    if not trace:
        raise ValueError("no episode to render: run at least one step first")
    if "action" not in trace[0]:
        raise ValueError("DJSet-v3 traces need per-step 'action' arrays")

    n_steps = len(trace)
    ta = np.asarray(track_a["beat_times"], dtype=np.float64)
    tb = np.asarray(track_b["beat_times"], dtype=np.float64)
    beat_a = float(np.median(np.diff(ta)))
    beat_b = float(np.median(np.diff(tb)))
    rate = beat_b / beat_a

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
    drum_a = curve(8)
    drum_b = curve(9)
    echo_a = curve(10)
    echo_b = curve(11)

    vocal_a = np.asarray(track_a["features"].get(
        "vocal", np.zeros(len(ta))), dtype=np.float32)
    vocal_b = np.asarray(track_b["features"].get(
        "vocal", np.zeros(len(tb))), dtype=np.float32)
    step_idx = np.arange(len(actions))
    beat_idx_a = np.clip(cue_a + step_idx, 0, len(vocal_a) - 1)
    beat_idx_b = np.clip(cue_b + step_idx, 0, len(vocal_b) - 1)
    va = np.interp(sample_t, action_t,
                   _smooth_steps(vocal_a[beat_idx_a])).astype(np.float32)
    vb = np.interp(sample_t, action_t,
                   _smooth_steps(vocal_b[beat_idx_b])).astype(np.float32)
    both = np.minimum(va, vb)
    duck_a = np.clip(1.0 - 0.45 * both
                     - 0.25 * np.clip(vb - va, 0.0, 1.0), 0.25, 1.0)
    duck_b = np.clip(1.0 - 0.30 * both
                     - 0.20 * np.clip(va - vb, 0.0, 1.0), 0.25, 1.0)

    a_drums, a_tonal = _drum_split(deck_a, sr)
    b_drums, b_tonal = _drum_split(deck_b, sr)
    deck_a = a_tonal + a_drums * drum_a[:, None]
    deck_b = b_tonal + b_drums * drum_b[:, None]

    a_low, a_mid, a_high = _split(deck_a, sr)
    b_low, b_mid, b_high = _split(deck_b, sr)
    a_mid = a_mid * duck_a[:, None]
    a_high = a_high * duck_a[:, None]
    b_mid = b_mid * duck_b[:, None]
    b_high = b_high * duck_b[:, None]
    a_mix = (a_low * eq_a[0][:, None] + a_mid * eq_a[1][:, None]
             + a_high * eq_a[2][:, None])
    b_mix = (b_low * eq_b[0][:, None] + b_mid * eq_b[1][:, None]
             + b_high * eq_b[2][:, None])

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
    a_post = gain_a * a_mix
    b_post = gain_b * b_filt
    dry = a_post + b_post

    delay_seconds = max(0.08, beat_a * 0.75)
    echo = (_echo_taps(a_post, sr, delay_seconds, echo_a)
            + _echo_taps(b_post, sr, delay_seconds, echo_b))
    mix = np.pad(dry, ((0, len(echo) - len(dry)), (0, 0))) + echo

    sample_pad = np.arange(len(mix), dtype=np.float64) / sr
    mono_env = _smooth(_rms_envelope(mix, sr), 128)
    target = float(np.median(mono_env)) if len(mono_env) else 0.0
    if target > 1e-5:
        ride = np.clip(target / (mono_env + 1e-6), 0.7, 1.5)
        ride = _smooth(ride, 96)
        ride_t = np.linspace(0.0, len(mix) / sr, len(ride))
        mix = mix * np.interp(sample_pad, ride_t, ride)[:, None].astype(
            np.float32)
    return mix.astype(np.float32), sr


def render_transition(env, out_path: str | Path, sr: int = 44100) -> Path:
    mix, sample_rate = render_mix(env, sr)
    return write_wav(finalise(mix, sample_rate), sample_rate, out_path)
