"""Build per-beat stem features for the v4 remix environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
STEMS = ROOT / "runs" / "v1" / "remix_stems"
OUT = ROOT / "runs" / "v1" / "remix_features.npz"
SR = 44100
BEAT = 0.6966
BARS = 32
BEATS = BARS * 4
A_CUE = 78.0
B_CUE = 78.0


def read_segment(path: Path, start: float, beats: int) -> np.ndarray:
    duration = beats * BEAT
    with sf.SoundFile(str(path)) as handle:
        handle.seek(int(round(start * SR)))
        data = handle.read(int(round(duration * SR)), dtype="float32",
                           always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data


def beat_rms(audio: np.ndarray) -> np.ndarray:
    values = []
    for index in range(BEATS):
        start = int(round(index * BEAT * SR))
        end = int(round((index + 1) * BEAT * SR))
        chunk = audio[start:end]
        values.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    stems = {
        "a_vocal": read_segment(STEMS / "A_vocals.wav", A_CUE, BEATS),
        "b_drums": read_segment(STEMS / "B_drums.wav", B_CUE, BEATS),
        "b_bass": read_segment(STEMS / "B_bass.wav", B_CUE, BEATS),
        "b_other": read_segment(STEMS / "B_other.wav", B_CUE, BEATS),
        "b_inst": read_segment(STEMS / "B_instrumental.wav", B_CUE, BEATS),
    }
    features = {name: beat_rms(audio) for name, audio in stems.items()}
    np.savez_compressed(OUT, beat_seconds=BEAT, beats=BEATS, **features)
    print(f"wrote {OUT}")
    for name, values in features.items():
        print(f"{name:8} max {values.max():.3f} mean {values.mean():.3f}")


if __name__ == "__main__":
    main()
