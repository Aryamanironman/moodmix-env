"""Mix any two analysed songs with DJSet-v1.

    # first time on a folder
    python scripts/prepare_library_v2.py --audio-dir "D:\\Music" --out "D:\\Music\\manifest_v2.json"

    python scripts/mix_songs.py --data-dir crate \
        --track-a "1. Blinding Lights.mp3" --track-b "10. STAY (with Justin Bieber).mp3" \
        --model models/ppo_v1.zip --out runs/v1/blend.wav

The policy only sees relative features (progress, band levels, phrase position,
tempo ratio, key distance), so a checkpoint trained on one crate applies to a
new one.  `--policy expert` renders the hand-written reference instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from moodmix_env.env_v1 import MoodMixV1Env
from moodmix_env.expert_v1 import expert_action
from moodmix_env.render_v2 import finalise, render_mix, write_wav


def resolve(env: MoodMixV1Env, value: str) -> str:
    wanted = Path(value).name.lower()
    stem = Path(value).stem.lower()
    for track in env.tracks:
        if value in (track["id"], track["file"]):
            return track["id"]
        if track["file"].lower() == wanted or track["id"].lower() == stem:
            return track["id"]
    known = ", ".join(track["id"] for track in env.tracks[:6])
    raise SystemExit(f"unknown track {value!r}; try one of: {known} ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="folder containing manifest_v2.json")
    parser.add_argument("--track-a", required=True)
    parser.add_argument("--track-b", required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--policy", choices=("model", "expert", "random"),
                        default="model")
    parser.add_argument("--cue-a", type=int, default=None)
    parser.add_argument("--cue-b", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-cues", action="store_true",
                        help="try the strongest phrase pairs and keep the "
                             "steadiest blend")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "v1" / "mix.wav")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    env = MoodMixV1Env(data_dir=args.data_dir, max_steps=args.steps)
    track_a = resolve(env, args.track_a)
    track_b = resolve(env, args.track_b)

    if args.policy == "model":
        if args.model is None:
            raise SystemExit("--policy model needs --model")
        from stable_baselines3 import PPO
        model = PPO.load(str(args.model), device="cpu")

        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
    elif args.policy == "expert":
        def policy(obs):
            return expert_action(env)
    else:
        def policy(obs):
            return env.action_space.sample()

    options = {"track_a": track_a, "track_b": track_b}
    if args.cue_a is not None:
        options["cue_a"] = args.cue_a
    if args.cue_b is not None:
        options["cue_b"] = args.cue_b
    if args.best_cues and args.cue_a is None and args.cue_b is None:
        from moodmix_env.select_v1 import best_cues
        picked = best_cues(env, policy, track_a, track_b, seed=args.seed)
        options["cue_a"] = picked["cue_a"]
        options["cue_b"] = picked["cue_b"]
        print(f"best cues {picked['cue_a']} -> {picked['cue_b']}  "
              f"predicted score {picked['score']:.3f}")
    obs, info = env.reset(seed=args.seed, options=options)
    print(f"{info['deck_a']['id']} [{info['deck_a']['section']}] -> "
          f"{info['deck_b']['id']} [{info['deck_b']['section']}]")
    print(f"bpm {info['deck_a']['bpm']} -> {info['deck_b']['bpm']}  "
          f"keys {info['deck_a']['key']} -> {info['deck_b']['key']}  "
          f"cues {info['cue_a']} -> {info['cue_b']}")

    total = 0.0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break

    mix, sr = render_mix(env, args.sample_rate)
    write_wav(finalise(mix, sr), sr, args.out)
    report = {
        "track_a": track_a,
        "track_b": track_b,
        "cue_a": info["cue_a"],
        "cue_b": info["cue_b"],
        "steps": env.step_count,
        "reward": round(total, 3),
        "final_fader": round(float(env.fader), 3),
        "low_overlap": round(float(np.mean(env._low_overlaps)), 4),
        "mean_level_jump": round(float(np.mean(env._level_jumps)), 4),
        "components": {key: round(float(value), 3)
                       for key, value in env.episode_components.items()},
        "wav": str(args.out),
        "duration_seconds": round(len(mix) / sr, 2),
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"audio: {args.out}")


if __name__ == "__main__":
    main()
