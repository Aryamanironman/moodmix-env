"""Build a hand-authored remix prototype from two Demucs stem sets.

    python scripts/make_remix_prototype.py

The arrangement is 32 bars at the crate tempo:

    0-4 bars   B instrumental, low-pass sweep intro
    4-16 bars  A vocal over B instrumental
    16-20 bars build: B drums + B other, looped A vocal, noise riser
    20-32 bars drop: full B instrumental + A vocal, echo tail

A vocal is pitched up one semitone so F minor sits in B's F sharp minor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.render_v2 import (  # noqa: E402
    _apply_filter, _rms_envelope, _smooth, finalise,
)

STEMS = ROOT / "runs" / "v1" / "remix_stems"
OUT = ROOT / "runs" / "v1"
SR = 44100
BEAT = 0.6966
BAR = 4.0 * BEAT
BARS = 32
DURATION = BARS * BAR
A_CUE = 78.0
B_CUE = 78.0


def read_segment(path: Path, start: float, duration: float) -> np.ndarray:
    with sf.SoundFile(str(path)) as handle:
        handle.seek(int(round(start * SR)))
        data = handle.read(int(round(duration * SR)), dtype="float32",
                           always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    want = int(round(duration * SR))
    if len(data) < want:
        data = np.pad(data, ((0, want - len(data)), (0, 0)))
    return data[:want].astype(np.float32)


def fade_edges(audio: np.ndarray, ms: float = 10.0) -> np.ndarray:
    n = min(len(audio), int(SR * ms / 1000.0))
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def echo(audio: np.ndarray, delay: float, wet: float,
         feedback: float = 0.30, taps: int = 3) -> np.ndarray:
    shift = max(1, int(round(delay * SR)))
    out = np.zeros_like(audio)
    for tap in range(1, taps + 1):
        offset = shift * tap
        if offset >= len(audio):
            break
        out[offset:] += (feedback ** (tap - 1)) * audio[:-offset]
    return out * wet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-stems", type=Path, default=STEMS)
    parser.add_argument("--b-stems", type=Path, default=STEMS)
    parser.add_argument("--a-cue", type=float, default=35.0)
    parser.add_argument("--b-cue", type=float, default=35.0)
    parser.add_argument("--a-pitch", type=float, default=1.0,
                        help="semitones to shift the A vocal")
    parser.add_argument("--b-drop-cue", type=float, default=None,
                        help="B seconds for the post-drop section")
    parser.add_argument("--out", default="remix_prototype_v2")
    parser.add_argument("--title",
                        default="Blinding Lights vocal over The Night We Met")
    parser.add_argument("--enhance-drop", action="store_true")
    args = parser.parse_args()

    a_vocal = read_segment(args.a_stems / "A_vocals.wav",
                           args.a_cue, DURATION)
    b_inst = read_segment(args.b_stems / "B_instrumental.wav",
                          args.b_cue, DURATION)
    b_drums = read_segment(args.b_stems / "B_drums.wav",
                           args.b_cue, DURATION)
    b_other = read_segment(args.b_stems / "B_other.wav",
                           args.b_cue, DURATION)
    n = len(b_inst)

    # Pitch the A vocal up one semitone so F minor matches B's F sharp minor.
    if abs(args.a_pitch) > 1e-4:
        raw = args.a_stems / "A_vocal_segment.wav"
        pitched_path = args.a_stems / "A_vocal_pitched.wav"
        sf.write(str(raw), a_vocal, SR, subtype="PCM_16")
        factor = 2.0 ** (args.a_pitch / 12.0)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(raw), "-af", f"rubberband=pitch={factor:.6f}",
             "-c:a", "pcm_s16le", str(pitched_path)],
            check=True,
        )
        a_vocal = read_segment(pitched_path, 0.0, DURATION)

    # B arrangement.
    intro_n = int(4 * BAR * SR)
    cutoff = np.full(n, 20000.0, dtype=np.float32)
    cutoff[:intro_n] = np.geomspace(220.0, 20000.0, intro_n).astype(np.float32)
    b_filtered = _apply_filter(b_inst, cutoff, SR, False)

    other_hp = _apply_filter(
        b_other, np.full(n, 220.0, dtype=np.float32), SR, True)
    build = b_drums + other_hp
    gap0, gap1 = int(19.5 * BAR * SR), int(20.0 * BAR * SR)
    build[gap0:gap1] *= 0.0

    b_track = np.zeros_like(b_inst)
    b_track[:intro_n] = b_filtered[:intro_n] * 0.75
    b_track[intro_n:int(16 * BAR * SR)] = \
        b_inst[intro_n:int(16 * BAR * SR)] * 0.75
    b_track[int(16 * BAR * SR):int(20 * BAR * SR)] = \
        build[int(16 * BAR * SR):int(20 * BAR * SR)] * (0.72 if args.enhance_drop else 0.85)
    if args.b_drop_cue is not None:
        drop_rel = max(0, int((args.b_drop_cue - args.b_cue) * SR))
        drop_len = int(12 * BAR * SR)
        drop_end = min(len(b_inst), drop_rel + drop_len)
        b_track[int(20 * BAR * SR):int(20 * BAR * SR) + drop_end - drop_rel] =             b_inst[drop_rel:drop_end] * (0.94 if args.enhance_drop else 0.82)
    else:
        b_track[int(20 * BAR * SR):] = b_inst[int(20 * BAR * SR):] * 0.78

    # A vocal arrangement.
    vocal = np.zeros_like(a_vocal)
    start = int(4 * BAR * SR)
    end = int(16 * BAR * SR)
    line = fade_edges(a_vocal[start:end].copy())
    vocal[start:end] = line * 1.55

    loop = a_vocal[int(8 * BAR * SR):int(10 * BAR * SR)].copy()
    loop_tile = fade_edges(np.tile(loop, (2, 1)))
    loop_start = int(16 * BAR * SR)
    loop_end = int(20 * BAR * SR)
    vocal[loop_start:loop_end] = loop_tile[:loop_end - loop_start] * (1.34 if args.enhance_drop else 1.45)

    outro = fade_edges(a_vocal[int(20 * BAR * SR):].copy())
    vocal[int(20 * BAR * SR):] = outro * 1.55

    # Echo only through the build and outro.
    echo_mask = np.zeros(n, dtype=np.float32)
    echo_mask[int(15 * BAR * SR):int(20 * BAR * SR)] = 1.0
    echo_mask[int(28 * BAR * SR):] = 1.0
    vocal_echo = echo(vocal, 0.75 * BEAT, 0.22)
    vocal_echo *= echo_mask[:, None]

    # Noise riser through the build.
    rng = np.random.default_rng(7)
    build_n = int(4 * BAR * SR)
    noise = rng.standard_normal((build_n, 2)).astype(np.float32)
    noise = _apply_filter(
        noise, np.full(build_n, 2500.0, dtype=np.float32), SR, True)
    rise = np.linspace(0.0, 1.0, build_n, dtype=np.float32) ** 2
    riser = np.zeros_like(b_inst)
    rise_start = int(16 * BAR * SR)
    rise_end = int(20 * BAR * SR)
    rise_count = rise_end - rise_start
    rise_count = min(rise_count, len(noise))
    riser[rise_start:rise_start + rise_count] = noise[:rise_count] * (rise[:rise_count] * 0.055)[:, None]

    # Duck B under the vocal so the mashup stays intelligible.
    env = _rms_envelope(vocal, SR)
    env_t = np.linspace(0.0, n / SR, len(env))
    env_full = np.interp(np.arange(n) / SR, env_t, env).astype(np.float32)
    env_full = _smooth(env_full, 64)
    duck = 1.0 - 0.22 * np.clip(env_full / 0.12, 0.0, 1.0)

    mix = b_track * duck[:, None] + vocal + vocal_echo + riser

    drop_at = int(20 * BAR * SR)
    impact_len = min(int(2.0 * SR), len(mix) - drop_at)
    if impact_len > 0:
        times = np.arange(impact_len, dtype=np.float32) / SR
        thump = np.sin(2.0 * np.pi * 55.0 * times) * np.exp(-times * 3.5)
        burst = rng.standard_normal(impact_len).astype(np.float32)             * np.exp(-times * 7.0)
        burst = _apply_filter(
            burst[:, None], np.full(impact_len, 1800.0, dtype=np.float32),
            SR, True)[:, 0]
        impact = (0.20 * thump + 0.07 * burst)[:, None]
        mix[drop_at:drop_at + impact_len] += impact

    if args.enhance_drop:
        # Bigger sub, crash, reverse swell and a drum roll into the hit.
        sub_len = min(int(1.8 * SR), len(mix) - drop_at)
        if sub_len > 0:
            times = np.arange(sub_len, dtype=np.float32) / SR
            sweep = 62.0 - 26.0 * np.minimum(times / 0.5, 1.0)
            sub = np.sin(2.0 * np.pi * np.cumsum(sweep) / SR)                 * np.exp(-times * 2.6)
            mix[drop_at:drop_at + sub_len] += (0.18 * sub)[:, None]
        crash_len = min(int(2.8 * SR), len(mix) - drop_at)
        if crash_len > 0:
            times = np.arange(crash_len, dtype=np.float32) / SR
            crash = rng.standard_normal(crash_len).astype(np.float32)                 * np.exp(-times * 2.2)
            crash = _apply_filter(
                crash[:, None], np.full(crash_len, 2600.0, dtype=np.float32),
                SR, True)[:, 0]
            mix[drop_at:drop_at + crash_len] += (0.11 * crash)[:, None]
        reverse_start = drop_at - int(1.0 * BAR * SR)
        reverse_len = drop_at - reverse_start
        if reverse_len > 0:
            times = np.arange(reverse_len, dtype=np.float32) / SR
            swell = (times / max(times[-1], 1e-3)) ** 2
            reverse = rng.standard_normal(reverse_len).astype(np.float32)                 * swell
            reverse = _apply_filter(
                reverse[:, None],
                np.full(reverse_len, 3200.0, dtype=np.float32), SR, True)[:, 0]
            mix[reverse_start:drop_at] += (0.08 * reverse)[:, None]
        roll_start = drop_at - int(0.5 * BAR * SR)
        roll_end = drop_at
        hit_at = int(19.0 * BAR * SR)
        hit = b_drums[hit_at:hit_at + int(0.06 * SR)]
        if len(hit) > 0 and roll_end > roll_start:
            for k in range(8):
                at = roll_start + int(k * (roll_end - roll_start) / 8)
                gain = 0.30 + 0.50 * (k / 7.0)
                end = min(len(mix), at + len(hit))
                mix[at:end] += hit[:end - at] * gain

    out = finalise(mix, SR)

    wav = OUT / f"{args.out}.wav"
    mp3 = OUT / f"{args.out}.mp3"
    sf.write(str(wav), out, SR, subtype="PCM_16")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)],
        check=True,
    )

    sections = [
        {"start": 0.0, "end": 4 * BAR, "label": "B intro, low-pass sweep"},
        {"start": 4 * BAR, "end": 16 * BAR, "label": "A vocal over B instrumental"},
        {"start": 16 * BAR, "end": 20 * BAR, "label": "build, looped vocal, riser"},
        {"start": 20 * BAR, "end": 32 * BAR, "label": "drop and outro"},
    ]
    meta = {
        "title": args.title,
        "bars": BARS,
        "beat_seconds": BEAT,
        "duration": round(len(out) / SR, 3),
        "stems": "Demucs htdemucs, four stems",
        "sections": sections,
        "files": {"wav": str(wav.relative_to(ROOT)),
                  "mp3": str(mp3.relative_to(ROOT))},
    }
    (OUT / f"{args.out}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    rms = []
    for bar in range(0, BARS, 4):
        chunk = out[int(bar * BAR * SR):int((bar + 4) * BAR * SR)]
        rms.append(round(float(np.sqrt(np.mean(chunk ** 2))), 3))
    print(f"wrote {wav} ({len(out) / SR:.1f}s)")
    print(f"wrote {mp3}")
    print("4-bar RMS", rms)
    print("peak", round(float(np.max(np.abs(out))), 3))


if __name__ == "__main__":
    main()
