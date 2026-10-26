"""Render DJSet-v1 transitions: one, a set, stochastic samples, or a compare.

    python scripts/demo_v1.py --model models/ppo_v1.zip --compare
    python scripts/demo_v1.py --model models/ppo_v1.zip --samples 6
    python scripts/demo_v1.py --model models/ppo_v1.zip --set --transitions 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import librosa
import numpy as np

from moodmix_env.env_v1 import MoodMixV1Env
from moodmix_env.expert_v1 import expert_action
from moodmix_env.render_v2 import finalise, render_mix, write_wav


def episode(env: MoodMixV1Env, policy, options: dict | None = None,
            seed: int | None = None) -> dict:
    obs, info = env.reset(seed=seed, options=options or {})
    total = 0.0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return {
        "track_a": env.track_a["id"], "track_b": env.track_b["id"],
        "cue_a": int(env.cue_a), "cue_b": int(env.cue_b),
        "reward": float(total),
        "final_fader": float(env.fader),
        "low_overlap": float(np.mean(env._low_overlaps)),
        "level_jump": float(np.mean(env._level_jumps)),
        "components": {key: float(value)
                       for key, value in env.episode_components.items()},
    }


def audio_metrics(path: Path, beats: np.ndarray) -> dict:
    y, sr = librosa.load(path, sr=44100, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr,
                                   hop_length=256)
    peaks = librosa.util.peak_pick(onset, pre_max=3, post_max=3, pre_avg=3,
                                   post_avg=3, delta=0.2, wait=2)
    peak_times = times[peaks]
    hits = []
    if len(peak_times):
        for beat in beats:
            offset = float(peak_times[np.argmin(np.abs(peak_times - beat))]
                           - beat)
            hits.append(abs(offset) <= 0.07)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20.0 * np.log10(rms + 1e-9)
    jumps = np.abs(np.diff(db))
    return {
        "seconds": round(len(y) / sr, 2),
        "beat_hit_fraction": round(float(np.mean(hits)) if hits else 0.0, 3),
        "rms_p99_jump_db": round(float(np.percentile(jumps, 99)), 2),
        "rms_max_jump_db": round(float(np.max(jumps)), 2),
    }


def crossfade(a: np.ndarray, b: np.ndarray, sr: int,
              seconds: float = 0.4) -> np.ndarray:
    n = min(int(seconds * sr), len(a), len(b))
    if n <= 1:
        return np.concatenate([a, b], axis=0)
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return np.concatenate([a[:-n], a[-n:] * (1 - fade) + b[:n] * fade,
                           b[n:]], axis=0)


def render(env: MoodMixV1Env, path: Path, sr: int = 44100) -> dict:
    mix, sr = render_mix(env, sr)
    mix = finalise(mix, sr)
    write_wav(mix, sr, path)
    ta = env.track_a["beat_times"]
    start = env.cue_a
    end = min(start + env.step_count + 1, len(ta))
    beats = ta[start:end] - ta[start]
    return audio_metrics(path, beats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "crate",
                        help="folder with manifest_v2.json")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track-a", default=None)
    parser.add_argument("--track-b", default=None)
    parser.add_argument("--cue-a", type=int, default=None)
    parser.add_argument("--cue-b", type=int, default=None)
    parser.add_argument("--transitions", type=int, default=4)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "v1")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = MoodMixV1Env(data_dir=args.data_dir, max_steps=args.steps)
    model = None
    if args.model is not None:
        from stable_baselines3 import PPO
        model = PPO.load(str(args.model), device="cpu")

    def model_policy(deterministic: bool = True):
        def _policy(obs):
            action, _ = model.predict(obs, deterministic=deterministic)
            return action
        return _policy

    if args.samples:
        if model is None:
            raise SystemExit("--samples needs --model")
        # The clone was fitted with a wide exploration sigma; sampling at that
        # width is mostly noise, not a distribution of performances.  Narrow it
        # for the demo so each rollout is a plausible performance.
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.data.fill_(-2.5)
        options = {"track_a": args.track_a, "track_b": args.track_b,
                   "cue_a": args.cue_a, "cue_b": args.cue_b}
        options = {key: value for key, value in options.items()
                   if value is not None}
        _, info = env.reset(seed=args.seed, options=options)
        fixed = {"track_a": env.track_a["id"], "track_b": env.track_b["id"],
                 "cue_a": env.cue_a, "cue_b": env.cue_b}
        rows = []
        for index in range(args.samples):
            result = episode(env, model_policy(False), fixed,
                             seed=args.seed + index)
            path = args.out_dir / f"sample_{index + 1:02d}.wav"
            result["audio"] = render(env, path)
            result["file"] = str(path)
            rows.append(result)
            print(f"sample {index + 1:02d}  {result['track_a']} -> "
                  f"{result['track_b']}  reward {result['reward']:+.2f}  "
                  f"fader {result['final_fader']:.2f}")
        (args.out_dir / "samples.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        print(f"labels {args.out_dir / 'samples.json'}")
        return

    if args.compare:
        if model is None:
            raise SystemExit("--compare needs --model")
        options = {"track_a": args.track_a, "track_b": args.track_b,
                   "cue_a": args.cue_a, "cue_b": args.cue_b}
        options = {key: value for key, value in options.items()
                   if value is not None}
        _, info = env.reset(seed=args.seed, options=options)
        fixed = {"track_a": env.track_a["id"], "track_b": env.track_b["id"],
                 "cue_a": env.cue_a, "cue_b": env.cue_b}
        results = {}
        clips = []
        for name, policy in (("expert", lambda obs: expert_action(env)),
                             ("model", model_policy(True))):
            result = episode(env, policy, fixed, seed=args.seed)
            path = args.out_dir / f"compare_{name}.wav"
            result["audio"] = render(env, path)
            result["file"] = str(path)
            results[name] = result
            mix, sr = render_mix(env, 44100)
            clips.append(finalise(mix, sr))
            print(f"{name:6s} reward {result['reward']:+.2f}  "
                  f"fader {result['final_fader']:.2f}  "
                  f"low overlap {result['low_overlap']:.3f}  "
                  f"level jump {result['level_jump']:.4f}  "
                  f"beat hits {result['audio']['beat_hit_fraction']:.2f}")
        gap = np.zeros((int(0.4 * sr), 2), dtype=np.float32)
        write_wav(np.concatenate([clips[0], gap, clips[1]]), sr,
                  args.out_dir / "compare_side_by_side.wav")
        print("pair:", results["model"]["track_a"], "->",
              results["model"]["track_b"])
        return

    # A short set: independent compatible transitions, crossfaded at the seams.
    if model is None:
        raise SystemExit("--set needs --model")
    pieces = []
    rows = []
    played: list[str] = []
    for index in range(max(2, min(8, args.transitions))):
        options = {"avoid_tracks": played[-2:]}
        result = episode(env, model_policy(True), options,
                         seed=args.seed + index)
        path = args.out_dir / f"set_part_{index + 1:02d}.wav"
        result["audio"] = render(env, path)
        result["file"] = str(path)
        rows.append(result)
        played.extend([result["track_a"], result["track_b"]])
        mix, sr = render_mix(env, 44100)
        mix = finalise(mix, sr)
        pieces.append(mix)
        print(f"{index + 1}. {result['track_a']} -> {result['track_b']}  "
              f"reward {result['reward']:+.2f}  "
              f"beat hits {result['audio']['beat_hit_fraction']:.2f}")
    joined = pieces[0]
    for piece in pieces[1:]:
        joined = crossfade(joined, piece, sr)
    out = args.out_dir / f"set_{len(pieces)}transitions.wav"
    write_wav(finalise(joined, sr), sr, out)
    (args.out_dir / "set.json").write_text(json.dumps(rows, indent=2),
                                           encoding="utf-8")
    print(f"set: {out}  ({len(joined) / sr:.1f}s)")


if __name__ == "__main__":
    main()
