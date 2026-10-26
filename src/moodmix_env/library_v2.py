"""Beat-accurate library analysis for the DJSet-v1 environment.

The v1 manifest keeps the actual beat times, the downbeat phase and per-beat
features.  The old manifest stored only a BPM and a beat index, so rendering
had to assume the first beat sat at t=0 and that every beat was exactly
60/BPM long.  On a real crate that is false twice over: a track can have its
first detected beat 19 seconds in, and a tracker can report 184 BPM for a
63 BPM ballad.  Mixing from those assumptions is what made the output drift.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")
KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
HOP = 512
N_FFT = 2048
SR = 22050
BEATS_PER_BAR = 4
PHRASE_BEATS = 32          # eight bars

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def fold_tempo(bpm: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return 120.0
    while bpm >= 160.0:
        bpm /= 2.0
    return float(bpm)


def _infer_mood(bpm: float, mode: str, energy: float) -> str:
    if bpm >= 124:
        return "party" if energy > 0.75 else "energetic"
    if bpm >= 112:
        return "happy" if mode == "major" else "focused"
    if bpm >= 100:
        return "focused"
    if bpm >= 90:
        return "relaxed" if mode == "major" else "emotional"
    return "relaxed"


def _key_and_mode(chroma: np.ndarray) -> tuple[str, str]:
    profile = chroma.mean(axis=1)
    if not np.isfinite(profile).all() or float(profile.std()) < 1e-9:
        return "C", "major"
    profile = (profile - profile.mean()) / (profile.std() + 1e-9)
    best_key, best_mode, best_score = 0, "major", -np.inf
    for key in range(12):
        rolled = np.roll(profile, -key)
        for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            t = (template - template.mean()) / (template.std() + 1e-9)
            score = float(np.dot(rolled, t) / len(t))
            if score > best_score:
                best_key, best_mode, best_score = key, mode, score
    return KEY_NAMES[best_key], best_mode


def _per_beat(values: np.ndarray, frames: np.ndarray,
              n_frames: int) -> np.ndarray:
    """Mean of a frame-level curve over each beat interval."""
    starts = np.clip(frames, 0, n_frames - 1)
    ends = np.clip(np.r_[frames[1:], n_frames], starts + 1, n_frames)
    out = np.zeros(len(starts), dtype=np.float32)
    for i, (start, end) in enumerate(zip(starts, ends)):
        out[i] = float(np.mean(values[int(start):int(end)]))
    return out


def _normalise(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    reference = float(np.percentile(curve, 95)) if len(curve) else 0.0
    if reference <= 1e-9:
        return np.zeros_like(curve)
    return np.clip(curve / reference, 0.0, 1.0)


def _smooth_frames(values: np.ndarray, width: int = 9) -> np.ndarray:
    if width <= 1 or len(values) < 2:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _parse_lrc(path: Path) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return events
    for line in raw.splitlines():
        stamps = re.findall(r"\[(\d{1,2}):(\d{2}(?:\.\d+)?)\]", line)
        text = re.sub(r"\[[^\]]*\]", "", line).strip()
        if not stamps or not text:
            continue
        for minutes, seconds in stamps:
            events.append((int(minutes) * 60.0 + float(seconds), text))
    return sorted(events)


def _vocal_curve(y: np.ndarray, y_harm: np.ndarray, sr: int,
                 spec: np.ndarray, beat_frames: np.ndarray,
                 n_frames: int, centroid_frames: np.ndarray,
                 freqs: np.ndarray) -> np.ndarray:
    """Estimate lead-vocal activity from the harmonic, mid-band, tonal content.

    This is not a separator. It is a cheap proxy for "a lead voice is likely
    here": harmonic fraction of the 200-4000 Hz band, low spectral flatness
    (tonal rather than noisy), and a centroid in the vocal range. An .lrc
    sidecar overrides it with real lyric timings when one exists.
    """
    spec_harm = np.abs(librosa.stft(y_harm, n_fft=N_FFT, hop_length=HOP))
    flatness = librosa.feature.spectral_flatness(S=spec)[0]
    mid = (freqs >= 200.0) & (freqs <= 4000.0)
    harmonic_mid = spec_harm[mid].sum(axis=0)
    total_mid = spec[mid].sum(axis=0)
    vocal_ratio = harmonic_mid / (total_mid + 1e-9)
    tonal = 1.0 - np.clip(flatness * 6.0, 0.0, 1.0)
    centroid_hz = centroid_frames * (sr / 2.0)
    weight = np.exp(-((centroid_hz - 1200.0) / 1500.0) ** 2)
    frames = _smooth_frames(vocal_ratio * tonal * weight, 9)
    return _normalise(_per_beat(frames, beat_frames, n_frames))


def _downbeat_phase(y: np.ndarray, sr: int,
                    beat_frames: np.ndarray) -> int:
    """Pick which of the four beat phases is the bar's downbeat.

    A downbeat usually carries the strongest low-frequency attack, so average
    low-band onset strength at each phase and take the maximum.  It is a
    heuristic, but it is stable enough to place every cue on a bar line.
    """
    if len(beat_frames) < 8:
        return 0
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP, aggregate=np.median)
    # Low band only: the kick owns the downbeat.
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    low = spec[freqs < 200].sum(axis=0)
    flux = np.r_[0.0, np.diff(low)]
    flux = np.maximum(flux, 0.0)
    _ = onset, onset_env
    scores = []
    for phase in range(4):
        idx = beat_frames[phase::4]
        idx = idx[(idx >= 0) & (idx < len(flux))]
        scores.append(float(np.mean(flux[idx])) if len(idx) else 0.0)
    return int(np.argmax(scores))


def _sections(beat_rms: np.ndarray, phase: int) -> tuple[list[float], list[str]]:
    n_bars = max(1, (len(beat_rms) - phase) // BEATS_PER_BAR)
    bar_energy: list[float] = []
    for bar in range(n_bars):
        start = phase + bar * BEATS_PER_BAR
        chunk = beat_rms[start:start + BEATS_PER_BAR]
        bar_energy.append(float(np.mean(chunk)) if len(chunk) else 0.0)
    peak = max(bar_energy) or 1.0
    bar_energy = [value / peak for value in bar_energy]

    labels: list[str] = []
    for bar, energy in enumerate(bar_energy):
        if bar < 4:
            labels.append("intro")
        elif bar >= n_bars - 4:
            labels.append("outro")
        elif energy >= 0.72:
            labels.append("drop")
        elif energy <= 0.28:
            labels.append("breakdown")
        else:
            labels.append("build")
    return bar_energy, labels


def _cues(beat_times: np.ndarray, phase: int, beat_rms: np.ndarray,
          labels: list[str], bar_energy: list[float]) -> list[dict]:
    cues: list[dict] = []
    last = len(beat_times) - PHRASE_BEATS
    for beat in range(0, max(0, last) + 1):
        if (beat - phase) % PHRASE_BEATS:
            continue
        bar = (beat - phase) // BEATS_PER_BAR
        section = labels[bar] if 0 <= bar < len(labels) else "build"
        window = beat_rms[beat:beat + PHRASE_BEATS]
        cues.append({
            "beat": int(beat),
            "bar": int(bar),
            "section": section,
            "energy": round(float(np.mean(window)) if len(window) else 0.0, 4),
            "phrase": int(beat // PHRASE_BEATS),
        })
    return cues


def analyze_track(path: str | Path, mood: str | None = None,
                  sr: int = SR) -> dict:
    path = Path(path)
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = len(y) / sr
    if duration < 4.0:
        raise ValueError(f"{path.name} is too short ({duration:.1f}s)")

    tempo_raw, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP, trim=False)
    bpm = float(np.atleast_1d(tempo_raw)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)

    # Tempo octave.  A tracker that reports 172 BPM for an 86 BPM song puts a
    # beat on every half-beat; halving only the number leaves the grid twice as
    # fast as the number says, which is exactly how an 86 BPM track ended up
    # asking to be stretched by 2x.  Fold the grid itself, choosing the offset
    # whose beats carry the strongest attacks.
    if bpm >= 150.0 and len(beat_frames) >= 16:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        while bpm >= 150.0 and len(beat_frames) >= 16:
            scores = []
            for offset in (0, 1):
                idx = beat_frames[offset::2]
                idx = idx[idx < len(onset_env)]
                scores.append(float(np.mean(onset_env[idx]))
                              if len(idx) else 0.0)
            offset = int(np.argmax(scores))
            beat_frames = beat_frames[offset::2]
            bpm /= 2.0
    bpm = max(30.0, min(199.0, bpm))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    if len(beat_times) < 8:
        beat_times = np.arange(0.0, duration, 60.0 / bpm)
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode = _key_and_mode(chroma)

    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    masks = (freqs < 250, (freqs >= 250) & (freqs < 2000), freqs >= 2000)
    band_frames = np.stack([spec[mask].mean(axis=0) for mask in masks])
    rms_frames = librosa.feature.rms(y=y, frame_length=N_FFT,
                                     hop_length=HOP)[0]
    onset_frames = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    centroid_frames = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0] / (sr / 2)
    y_perc, y_harm = librosa.effects.hpss(y)
    perc_frames = librosa.feature.rms(y=y_perc, frame_length=N_FFT,
                                      hop_length=HOP)[0]
    harm_frames = librosa.feature.rms(y=y_harm, frame_length=N_FFT,
                                      hop_length=HOP)[0]

    n_frames = spec.shape[1]
    bands = np.stack([_normalise(_per_beat(channel, beat_frames, n_frames))
                      for channel in band_frames])
    rms_curve = _normalise(_per_beat(rms_frames, beat_frames, n_frames))
    onset_curve = _normalise(_per_beat(onset_frames, beat_frames, n_frames))
    perc_curve = _normalise(_per_beat(perc_frames, beat_frames, n_frames))
    harm_curve = _normalise(_per_beat(harm_frames, beat_frames, n_frames))
    centroid_curve = _normalise(
        _per_beat(centroid_frames, beat_frames, n_frames))
    vocal_curve = _vocal_curve(y, y_harm, sr, spec, beat_frames, n_frames,
                              centroid_frames, freqs)
    lyric_lines = [""] * len(beat_times)
    lrc_path = path.with_suffix(".lrc")
    if lrc_path.exists():
        events = _parse_lrc(lrc_path)
        for i, (start, text) in enumerate(events):
            end = (events[i + 1][0] if i + 1 < len(events)
                   else min(duration, start + 8.0))
            mask = (beat_times >= start) & (beat_times < end)
            for index in np.where(mask)[0]:
                lyric_lines[int(index)] = text
            if mask.any():
                vocal_curve[mask] = 1.0
    vocal_source = "lrc" if any(lyric_lines) else "audio"

    phase = _downbeat_phase(y, sr, beat_frames)
    bar_energy, labels = _sections(rms_curve, phase)
    cues = _cues(beat_times, phase, rms_curve, labels, bar_energy)
    if not cues:
        cues = [{"beat": 0, "bar": 0, "section": "intro",
                 "energy": float(np.mean(rms_curve)) if len(rms_curve) else 0.0,
                 "phrase": 0}]

    energy = float(np.clip(np.mean(rms_frames) /
                           (np.percentile(rms_frames, 95) + 1e-9), 0.0, 1.0))
    musical_bpm = bpm
    return {
        "id": path.stem,
        "file": path.name,
        "mood": mood or _infer_mood(musical_bpm, mode, energy),
        "bpm": round(float(bpm), 3),
        "musical_bpm": round(float(musical_bpm), 3),
        "beat_seconds": round(60.0 / float(bpm), 5),
        "key": key,
        "mode": mode,
        "root_midi": 48 + KEY_NAMES.index(key),
        "duration": round(duration, 3),
        "energy": round(energy, 4),
        "downbeat_phase": int(phase),
        "beat_times": [round(float(t), 5) for t in beat_times],
        "features": {
            "low": np.round(bands[0], 4).tolist(),
            "mid": np.round(bands[1], 4).tolist(),
            "high": np.round(bands[2], 4).tolist(),
            "rms": np.round(rms_curve, 4).tolist(),
            "onset": np.round(onset_curve, 4).tolist(),
            "percussive": np.round(perc_curve, 4).tolist(),
            "harmonic": np.round(harm_curve, 4).tolist(),
            "centroid": np.round(centroid_curve, 4).tolist(),
            "vocal": np.round(vocal_curve, 4).tolist(),
        },
        "lyrics": lyric_lines,
        "vocal_source": vocal_source,
        "bar_energy": [round(float(v), 4) for v in bar_energy],
        "sections": labels,
        "cues": cues,
    }


def build_manifest(audio_dir: str | Path, out_path: str | Path,
                   moods: dict[str, str] | None = None,
                   sr: int = SR, verbose: bool = True,
                   progress: Callable[[int, int, str], None] | None = None
                   ) -> dict:
    audio_dir = Path(audio_dir)
    out_path = Path(out_path)
    moods = moods or {}
    files = sorted(p for p in audio_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise FileNotFoundError(f"no audio files found in {audio_dir}")

    tracks = []
    total = len(files)
    for index, path in enumerate(files, 1):
        if progress:
            progress(index, total, path.name)
        if verbose:
            print(f"analyzing {path.name} ...", flush=True)
        tracks.append(analyze_track(path, mood=moods.get(path.stem), sr=sr))
    if progress:
        progress(total, total, "done")
    manifest = {
        "version": 2,
        "sample_rate": sr,
        "beats_per_bar": BEATS_PER_BAR,
        "phrase_beats": PHRASE_BEATS,
        "tracks": tracks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
