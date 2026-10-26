"""Turn a folder of real audio files into the manifest MoodMixEnv reads.

This is the bridge for "my own library": point it at MP3/WAV/FLAC files you
own, and it writes bpm, key, per-beat band energy and rough sections.  It does
not download anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")
KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
HOP = 512
N_FFT = 2048

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def fold_tempo(bpm: float) -> float:
    """Halve tempi that a beat tracker double-counted.

    Trackers routinely report double the musical tempo (an 86 BPM song read as
    172), which corrupts the mood label, the observation's BPM feature and the
    mixdown's stretch ratio.  Only *halving* is applied: slow songs are
    detected correctly often enough that doubling them (63 -> 126) would break
    ballads, while >160 BPM readings are almost always double-time.
    """
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


def _beat_curve(y: np.ndarray, sr: int, beat_times: np.ndarray) -> np.ndarray:
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    masks = (freqs < 250, (freqs >= 250) & (freqs < 2000), freqs >= 2000)
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
    curve = np.zeros((len(beat_times), 3), dtype=np.float32)
    for i, start in enumerate(frames):
        end = frames[i + 1] if i + 1 < len(frames) else stft.shape[1]
        start = int(np.clip(start, 0, stft.shape[1] - 1))
        end = int(np.clip(max(end, start + 1), start + 1, stft.shape[1]))
        chunk = stft[:, start:end]
        curve[i] = [float(chunk[mask].mean()) if np.any(mask) else 0.0
                    for mask in masks]
    peak = float(curve.max()) or 1.0
    return np.clip(curve / peak, 0.0, 1.0)


def _sections(y: np.ndarray, sr: int, beat_times: np.ndarray) -> tuple[list[str], list[float]]:
    n_bars = max(1, len(beat_times) // 4)
    energies: list[float] = []
    for bar in range(n_bars):
        start_beat = bar * 4
        end_beat = min(start_beat + 4, len(beat_times))
        start = int(beat_times[start_beat] * sr) if start_beat < len(beat_times) else 0
        end = int(beat_times[end_beat] * sr) if end_beat < len(beat_times) else len(y)
        chunk = y[start:max(start + 1, end)]
        energies.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)
    peak = max(energies) or 1.0
    energies = [min(1.0, value / peak) for value in energies]

    labels: list[str] = []
    for bar, energy in enumerate(energies):
        if bar == 0:
            labels.append("intro")
        elif bar == n_bars - 1:
            labels.append("outro")
        elif energy >= 0.75:
            labels.append("drop")
        elif energy <= 0.25:
            labels.append("breakdown")
        else:
            labels.append("build")
    return labels, [round(value, 4) for value in energies]


def analyze_track(path: str | Path, mood: str | None = None,
                  sr: int = 22050) -> dict:
    path = Path(path)
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = len(y) / sr
    if duration < 2.0:
        raise ValueError(f"{path.name} is too short ({duration:.1f}s)")

    tempo_raw, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP
    )
    bpm = fold_tempo(float(np.atleast_1d(tempo_raw)[0]))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    if len(beat_times) < 4:
        beat_times = np.arange(0.0, duration, 60.0 / bpm)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode = _key_and_mode(chroma)
    curve = _beat_curve(y, sr, beat_times)
    sections, energy_curve = _sections(y, sr, beat_times)

    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.clip(np.mean(rms) / (np.percentile(rms, 95) + 1e-9), 0.0, 1.0))
    inferred = _infer_mood(bpm, mode, energy)

    return {
        "id": path.stem,
        "mood": mood or inferred,
        "file": path.name,
        "bpm": round(bpm, 2),
        "key": key,
        "root_midi": 48 + KEY_NAMES.index(key),
        "mode": mode,
        "energy": round(energy, 4),
        "duration": round(duration, 3),
        "beat_seconds": round(60.0 / bpm, 5),
        "band_curve": np.round(curve, 3).tolist(),
        "energy_curve": energy_curve,
        "sections": sections,
    }


def build_manifest(
    audio_dir: str | Path,
    out_path: str | Path,
    moods: dict[str, str] | None = None,
    sr: int = 22050,
) -> dict:
    audio_dir = Path(audio_dir)
    out_path = Path(out_path)
    moods = moods or {}
    files = sorted(p for p in audio_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise FileNotFoundError(f"no audio files found in {audio_dir}")

    tracks = []
    for path in files:
        print(f"analyzing {path.name} ...", flush=True)
        tracks.append(analyze_track(path, mood=moods.get(path.stem), sr=sr))

    manifest = {
        "sample_rate": sr,
        "beats_per_bar": 4,
        "bars": max(len(t["sections"]) for t in tracks),
        "beats": max(len(t["band_curve"]) for t in tracks),
        "tracks": tracks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
