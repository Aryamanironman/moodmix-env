"""Generate the self-contained demo crate for MoodMix.

Every track is synthesised from scratch with numpy, so the demo has no
licensing baggage and runs offline.  The generator also writes the manifest
the environment reads: tempo, key, per-beat band energies and song sections.

Run:
    python scripts/make_demo_library.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
BEATS_PER_BAR = 4
BARS = 24
BEATS = BARS * BEATS_PER_BAR
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"

MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)

# Per-mood arrangement.  root is a MIDI note; levels are relative stem gains.
MOODS = {
    "energetic": dict(
        id="energetic_pulse", bpm=128, root=45, mode=MINOR,
        kick=0.95, snare=0.55, hat=0.30, bass=0.80, pad=0.30, arp=0.45,
        swing=0.0, energy=0.95,
    ),
    "happy": dict(
        id="happy_bounce", bpm=122, root=48, mode=MAJOR,
        kick=0.70, snare=0.40, hat=0.28, bass=0.55, pad=0.55, arp=0.65,
        swing=0.06, energy=0.80,
    ),
    "relaxed": dict(
        id="relaxed_drift", bpm=100, root=41, mode=MAJOR,
        kick=0.32, snare=0.16, hat=0.12, bass=0.38, pad=0.85, arp=0.28,
        swing=0.14, energy=0.35,
    ),
    "emotional": dict(
        id="emotional_rain", bpm=90, root=50, mode=MINOR,
        kick=0.38, snare=0.22, hat=0.10, bass=0.46, pad=0.95, arp=0.30,
        swing=0.12, energy=0.45,
    ),
    "focused": dict(
        id="focused_flow", bpm=110, root=43, mode=MINOR,
        kick=0.55, snare=0.30, hat=0.34, bass=0.58, pad=0.62, arp=0.50,
        swing=0.03, energy=0.60,
    ),
    "party": dict(
        id="party_strobe", bpm=126, root=40, mode=MINOR,
        kick=1.00, snare=0.62, hat=0.38, bass=0.88, pad=0.35, arp=0.58,
        swing=0.0, energy=1.00,
    ),
}


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def env_ad(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    a = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    return a * np.exp(-t / max(decay, 1e-4))


def kick(dur: float = 0.38) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    freq = 45.0 + 95.0 * np.exp(-t / 0.035)
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * np.exp(-t / 0.16)


def snare(rng: np.random.Generator, dur: float = 0.26) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n), prepend=0.0)
    body = 0.35 * np.sin(2.0 * np.pi * 185.0 * t)
    return (0.75 * noise + body) * np.exp(-t / 0.075)


def hat(rng: np.random.Generator, dur: float = 0.07) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n), n=2, prepend=[0.0, 0.0])
    return noise * np.exp(-t / 0.018)


def bass_note(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.28 * np.sin(2 * np.pi * 2 * freq * t)
    wave = np.tanh(1.8 * wave) * 0.7
    return wave * env_ad(t, 0.012, dur * 0.55)


def chord(notes: list[float], dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for i, midi in enumerate(notes):
        f = midi_hz(midi)
        detune = 1.0 + (i - len(notes) / 2) * 0.0015
        out += np.sin(2 * np.pi * f * detune * t) * (0.9 ** i)
    return out * env_ad(t, 0.18, dur * 0.8) / len(notes)


def pluck(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    wave = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t)
    return wave * np.exp(-t / (dur * 0.35)) * env_ad(t, 0.004, dur)


def add_at(buffer: np.ndarray, start: int, signal: np.ndarray, gain: float) -> None:
    if start >= len(buffer) or gain == 0.0:
        return
    end = min(len(buffer), start + len(signal))
    buffer[start:end] += gain * signal[: end - start]


def triad(root: float, mode: tuple[int, ...], degree: int) -> list[float]:
    """Build a diatonic triad on a scale degree (0-6)."""
    idx = [degree % 7, (degree + 2) % 7, (degree + 4) % 7]
    octaves = [degree // 7, (degree + 2) // 7, (degree + 4) // 7]
    return [root + mode[i] + 12 * o for i, o in zip(idx, octaves)]


def render(spec: dict, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    beat = 60.0 / spec["bpm"]
    samples = int(BEATS * beat * SR) + int(2.0 * SR)
    out = np.zeros(samples)

    kick_sig = kick()
    snare_sig = snare(rng)
    hat_sig = hat(rng)
    root = spec["root"]
    mode = spec["mode"]
    progression = (0, 5, 3, 4)  # i - VI - IV - V style movement

    for b in range(BEATS):
        start_s = b * beat
        bar = b // BEATS_PER_BAR
        beat_in_bar = b % BEATS_PER_BAR
        swing = spec["swing"] * beat * 0.5

        # Four-on-the-floor for most moods; relaxed/emotional get a sparser kick.
        sparse = spec["mood_name"] in ("relaxed", "emotional")
        if (not sparse) or beat_in_bar in (0, 2):
            add_at(out, int(start_s * SR), kick_sig, spec["kick"] * 0.55)

        if beat_in_bar in (1, 3):
            add_at(out, int(start_s * SR), snare_sig, spec["snare"] * 0.45)

        # Eighth-note hats, lightly swung.
        add_at(out, int(start_s * SR), hat_sig, spec["hat"] * 0.30)
        add_at(out, int((start_s + beat * 0.5 + swing) * SR), hat_sig,
               spec["hat"] * 0.20)

        # Bass: root on 1 and 3, fifth on 2 and 4.
        degree = 0 if beat_in_bar in (0, 2) else 4
        note = root + mode[degree % 7]
        add_at(out, int(start_s * SR), bass_note(midi_hz(note), beat * 0.92),
               spec["bass"] * 0.38)

        # One sustained chord per bar.
        if beat_in_bar == 0:
            notes = triad(root + 12, mode, progression[bar % 4])
            add_at(out, int(start_s * SR), chord(notes, beat * 3.7),
                   spec["pad"] * 0.30)

        # Arpeggio on eighth notes, one octave up.
        for k in range(2):
            step = bar * 2 + beat_in_bar // 2 + k
            note = root + 24 + mode[(step * 2) % 7]
            add_at(out, int((start_s + k * beat * 0.5 + swing) * SR),
                   pluck(midi_hz(note), beat * 0.42), spec["arp"] * 0.22)

    out = np.tanh(out * 1.15) * 0.92
    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.89
    return out.astype(np.float32), beat


def band_curve(audio: np.ndarray, beat: float) -> np.ndarray:
    """Per-beat low/mid/high energy, normalised to the loudest beat."""
    n_beats = BEATS
    curve = np.zeros((n_beats, 3), dtype=np.float32)
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    low = freqs < 250
    mid = (freqs >= 250) & (freqs < 2000)
    high = freqs >= 2000

    for b in range(n_beats):
        centre = int((b + 0.5) * beat * SR)
        lo = max(0, centre - 1024)
        hi = min(len(audio), centre + 1024)
        window = audio[lo:hi] * np.hanning(max(1, hi - lo))
        if len(window) < 2048:
            window = np.pad(window, (0, 2048 - len(window)))
        mag = np.abs(np.fft.rfft(window))
        curve[b] = (mag[low].mean(), mag[mid].mean(), mag[high].mean())

    curve /= max(float(curve.max()), 1e-6)
    return np.clip(curve, 0.0, 1.0)


def energy_curve(audio: np.ndarray, beat: float) -> list[float]:
    per_bar = []
    bar_len = int(4 * beat * SR)
    for bar in range(BARS):
        chunk = audio[bar * bar_len:(bar + 1) * bar_len]
        per_bar.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)
    peak = max(per_bar) or 1.0
    return [round(v / peak, 4) for v in per_bar]


def section_for_bar(bar: int) -> str:
    if bar < 4:
        return "intro"
    if bar < 8:
        return "build"
    if bar < 16:
        return "drop"
    if bar < 20:
        return "breakdown"
    return "outro"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"sample_rate": SR, "beats_per_bar": BEATS_PER_BAR,
                "bars": BARS, "beats": BEATS, "tracks": []}

    for i, (mood, base) in enumerate(MOODS.items()):
        spec = dict(base)
        spec["mood_name"] = mood
        audio, beat = render(spec, seed=1000 + i)
        path = OUT_DIR / f"{spec['id']}.wav"
        sf.write(path, audio, SR, subtype="PCM_16")

        curve = band_curve(audio, beat)
        manifest["tracks"].append({
            "id": spec["id"],
            "mood": mood,
            "file": path.name,
            "bpm": spec["bpm"],
            "key": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
                    "A", "A#", "B"][spec["root"] % 12],
            "root_midi": spec["root"],
            "mode": "minor" if spec["mode"] == MINOR else "major",
            "energy": spec["energy"],
            "duration": round(len(audio) / SR, 3),
            "beat_seconds": round(beat, 5),
            "band_curve": np.round(curve, 3).tolist(),
            "energy_curve": energy_curve(audio, beat),
            "sections": [section_for_bar(b) for b in range(BARS)],
        })
        print(f"wrote {path.name:24s} {mood:10s} {spec['bpm']:3d} BPM "
              f"{len(audio) / SR:5.1f}s")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("*.wav")) / 1e6
    print(f"\nmanifest: {manifest_path}")
    print(f"demo crate: {total_mb:.1f} MB, {len(manifest['tracks'])} tracks")


if __name__ == "__main__":
    main()
