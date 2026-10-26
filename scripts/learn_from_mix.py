"""Learn a policy from a recorded DJ transition.

`recover_transition.py` turns a finished mix into fader/EQ moves.  This script
closes the loop: those moves become the expert a policy is cloned from, so the
agent imitates a real performance instead of the hand-written baseline.

    python scripts/learn_from_mix.py --mix set.wav \
        --track-a a.mp3 --track-b b.mp3 --bpm 124 \
        --data-dir crate --base models/ppo_100k.zip \
        --output models/ppo_mix.zip

How it works:

1. recover the crossfader and EQ from the mix (automatic alignment, no cue
   points, playback rate locked to the tempi in the crate manifest);
2. replay that action sequence in the environment on the same pair of tracks,
   so every observation is the one the mixer produced at that beat;
3. fit the policy's crossfader, EQ and sync heads to those actions with
   cross-entropy.  The filter and drum pads are not recoverable from audio, so
   their heads are left alone;
4. report how tightly the cloned policy tracks the recorded moves, and what the
   environment's own reward makes of it.

The same action sequence is replayed on extra random pairs (`--augment`), which
stops the clone from memorising one transition.

Honest limits: a mix only reveals fader and EQ, the environment's reward encodes
its own smoothstep ideal (so imitating a very different DJ can score *lower*
than PPO), and an EQ low-cut is indistinguishable from the high-pass filter in
the recovered signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moodmix_env.env import MoodMixEnv  # noqa: E402
from moodmix_env.evaluation import model_policy  # noqa: E402
from moodmix_env.policies import heuristic_action  # noqa: E402
from pretrain_human import _flatten, fit_policy  # noqa: E402
from recover_transition import FADER_BINS, compare, recover  # noqa: E402

# Crossfader + the three EQ bands on each deck, plus sync.  Sync is not in the
# audio, but the recovered playback rate says whether the deck was beatmatched,
# so it is supervised too.  The filter and drum pads remain unsupervised: an EQ
# low-cut and the high-pass filter look identical in a finished mix.
RECOVERABLE_DIMS = (0, 1, 2, 3, 4, 5, 6, 7)


def find_track_ids(manifest_path: Path, paths: list[Path]) -> list[str]:
    """Map source files to the ids the environment uses for them."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_name = {track["file"]: track["id"] for track in manifest["tracks"]}
    by_stem = {Path(track["file"]).stem: track["id"] for track in manifest["tracks"]}
    ids: list[str] = []
    for path in paths:
        path = Path(path)
        found = by_name.get(path.name) or by_stem.get(path.stem)
        if found is None:
            known = ", ".join(sorted(by_stem)[:5])
            raise ValueError(
                f"{path.name} is not in {manifest_path} (try {known}, ...). "
                f"Analyse it first: python scripts/analyze_library.py "
                f"--audio-dir {path.parent} --out {manifest_path.parent}"
            )
        ids.append(found)
    if len(set(ids)) < 2:
        raise ValueError("a transition needs two different tracks")
    return ids


def actions_from_trace(trace: dict, steps: int = 64, sync: bool = True) -> np.ndarray:
    """Resample a recovered trace to `steps` beats and encode env actions."""
    recovered = trace["steps"]
    if not recovered:
        raise ValueError("the recovered trace has no beats")
    source = np.arange(len(recovered), dtype=float)
    target = np.linspace(0.0, len(recovered) - 1.0, steps)

    faders = np.interp(target, source,
                       [float(s["crossfader"]) for s in recovered])
    bins = np.clip(np.round(faders * (FADER_BINS - 1)), 0, FADER_BINS - 1)

    eq_keys = ("eq_a", "eq_b")
    eq = {}
    for key in eq_keys:
        bands = np.array([list(s[key]) for s in recovered], dtype=int)
        nearest = np.round(target).astype(int)
        eq[key] = bands[nearest]

    actions = np.zeros((steps, 10), dtype=np.int64)
    actions[:, 0] = bins.astype(np.int64)
    actions[:, 1:4] = eq["eq_a"]
    actions[:, 4:7] = eq["eq_b"]
    # Beatmatching is a precondition of the blend, not something the fader curve
    # says, so it is engaged on the first beat when the tempi differ.
    actions[0, 7] = 1 if sync else 0
    return actions


def replay(env: MoodMixEnv, actions: np.ndarray, ids: list[str],
           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Observation/action pairs for a recorded sequence on a pinned pair."""
    obs, info = env.reset(seed=seed, options={"track_a": ids[0],
                                             "track_b": ids[1]})
    observations: list[np.ndarray] = []
    used: list[np.ndarray] = []
    for action in actions:
        observations.append(_flatten(obs))
        used.append(np.asarray(action, dtype=np.int64))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return np.stack(observations), np.stack(used)


def collect_imitation_data(
    data_dir: str | Path | None,
    ids: list[str],
    actions: np.ndarray,
    augment: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs from the recorded transition plus copies on random pairs."""
    steps = len(actions)
    env = MoodMixEnv(data_dir=data_dir, max_steps=steps)
    obs, act = replay(env, actions, ids, seed=seed)
    observations = [obs]
    targets = [act]
    for episode in range(max(1, augment)):
        obs, info = env.reset(seed=seed + 1 + episode)
        rows, used = [], []
        for action in actions:
            rows.append(_flatten(obs))
            used.append(np.asarray(action, dtype=np.int64))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        observations.append(np.stack(rows))
        targets.append(np.stack(used))
    env.close()
    return np.concatenate(observations), np.concatenate(targets)


def actions_agreement(taken: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """How closely a policy reproduces the recorded controls."""
    taken = np.asarray(taken, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    n = min(len(taken), len(target))
    if n == 0:
        return {"steps": 0, "fader_mae_bins": float("nan"),
                "fader_exact": float("nan"), "eq_band_accuracy": float("nan")}
    bins = np.abs(taken[:n, 0] - target[:n, 0])
    eq_taken, eq_target = taken[:n, 1:7], target[:n, 1:7]
    return {
        "steps": n,
        "fader_mae_bins": float(bins.mean()),
        "fader_exact": float((bins == 0).mean()),
        "eq_band_accuracy": float((eq_taken == eq_target).mean()),
    }


def run_policy(policy, env: MoodMixEnv, ids: list[str], steps: int,
               seed: int = 0) -> tuple[float, np.ndarray]:
    """Total reward and controls for one pinned transition."""
    obs, info = env.reset(seed=seed, options={"track_a": ids[0],
                                             "track_b": ids[1]})
    total = 0.0
    taken: list[np.ndarray] = []
    for _ in range(steps):
        action = np.asarray(policy(obs, info), dtype=np.int64)
        taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total, np.stack(taken)


def learn(
    mix_path: Path,
    track_a: Path,
    track_b: Path,
    bpm: float,
    data_dir: str | Path = ROOT / "crate",
    base: str | Path = ROOT / "models" / "ppo_100k.zip",
    output: str | Path = ROOT / "models" / "ppo_mix.zip",
    steps: int = 64,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    augment: int = 8,
    seed: int = 0,
    truth: Path | None = None,
    offset_a: float | None = None,
    offset_b: float | None = None,
    trace_out: Path | None = None,
    sample_rate: int = 22050,
    verbose: bool = True,
) -> tuple[Path, dict]:
    if steps < 2:
        raise ValueError("steps must be at least 2")

    manifest_path = Path(data_dir) / "manifest.json"
    ids = find_track_ids(manifest_path, [track_a, track_b])
    bpm_a = _manifest_bpm(manifest_path, ids[0])
    bpm_b = _manifest_bpm(manifest_path, ids[1])

    trace = recover(mix_path, track_a, track_b, bpm,
                    offset_a=offset_a, offset_b=offset_b,
                    sr=sample_rate, steps=steps, bpm_a=bpm_a, bpm_b=bpm_b)
    if trace_out:
        Path(trace_out).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    actions = actions_from_trace(trace, steps=steps, sync=abs(bpm_b - bpm_a) > 0.005 * bpm_a)

    observations, targets = collect_imitation_data(
        data_dir, ids, actions, augment=augment, seed=seed
    )

    from stable_baselines3 import PPO

    model = PPO.load(str(base), device="cpu")
    if tuple(model.observation_space.shape) != (63,):
        raise ValueError(
            f"{base} expects observation {model.observation_space.shape}, "
            "but the environment now emits (63,) floats (the scripted target "
            "curve was removed from context). Retrain the base checkpoint "
            "with scripts/train.py, or pass a 63-dim one with --base."
        )
    accuracy = fit_policy(
        model, observations, targets,
        epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed, dims=RECOVERABLE_DIMS,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))

    env = MoodMixEnv(data_dir=data_dir, max_steps=steps)
    learned = model_policy(model)
    reward_mix, taken = run_policy(learned, env, ids, steps, seed=seed)
    agreement = actions_agreement(taken, actions)
    reward_human, _ = run_policy(heuristic_action, env, ids, steps, seed=seed)
    env.close()

    alignment = trace["alignment"]
    meta = {
        "model": str(output),
        "mix": str(mix_path),
        "tracks": ids,
        "bpm": float(bpm),
        "steps": int(steps),
        "augment": int(augment),
        "epochs": int(epochs),
        "base": str(base),
        "seed": int(seed),
        "supervised_dims": list(RECOVERABLE_DIMS),
        "bc_accuracy": float(accuracy),
        "imitation": agreement,
        "alignment": alignment,
        "eval_reward_mix": float(reward_mix),
        "eval_reward_human": float(reward_human),
    }
    if truth:
        truth_data = json.loads(Path(truth).read_text(encoding="utf-8"))
        if isinstance(truth_data, dict) and "steps" in truth_data:
            truth_data = truth_data["steps"]
        meta["recovery_vs_truth"] = compare(trace, truth_data)
    meta_path = output.parent / f"{output.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"tracks: {ids[0]} -> {ids[1]}  ({bpm_a} -> {bpm_b} BPM, "
              f"mix {bpm} BPM)")
        print(f"aligned A at {alignment['a']['offset_seconds']:.2f}s "
              f"(rate {alignment['a']['slope']:.3f}), "
              f"B at {alignment['b']['offset_seconds']:.2f}s "
              f"(rate {alignment['b']['slope']:.3f})")
        if "recovery_vs_truth" in meta:
            scores = meta["recovery_vs_truth"]
            print(f"recovery vs truth: {scores['fader_mae_bins']:.2f} bins, "
                  f"correlation {scores['fader_corr']:.3f}, "
                  f"EQ {scores['eq_band_accuracy']:.1%} of bands")
        print(f"cloned policy: {accuracy:.1%} head accuracy over "
              f"{len(observations)} observations")
        print(f"tracks the recording to {agreement['fader_mae_bins']:.2f} bins "
              f"mean, {agreement['fader_exact']:.1%} of beats exact")
        print(f"env reward on that transition: mix policy {reward_mix:+.2f} vs "
              f"human {reward_human:+.2f}")
        print(f"\nsaved {output}\nmeta   {meta_path}")
    return output, meta


def _manifest_bpm(manifest_path: Path, track_id: str) -> float:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for track in manifest["tracks"]:
        if track["id"] == track_id:
            return float(track["bpm"])
    raise ValueError(f"unknown track id {track_id!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mix", type=Path, required=True,
                        help="recording of the transition (wav/mp3/flac)")
    parser.add_argument("--track-a", type=Path, required=True)
    parser.add_argument("--track-b", type=Path, required=True)
    parser.add_argument("--bpm", type=float, required=True,
                        help="tempo of the mix, used to slice it into beats")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "crate",
                        help="crate whose manifest describes the two tracks")
    parser.add_argument("--base", type=Path, default=ROOT / "models" / "ppo_100k.zip",
                        help="checkpoint whose policy is cloned")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ppo_mix.zip")
    parser.add_argument("--steps", type=int, default=64,
                        help="beats in the resulting episode (the demo uses 64)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--augment", type=int, default=8,
                        help="extra random track pairs to replay the moves on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truth", type=Path, default=None,
                        help="env trace JSON to score the recovery against")
    parser.add_argument("--trace-out", type=Path, default=None,
                        help="also write the recovered trace here")
    parser.add_argument("--offset-a", type=float, default=None)
    parser.add_argument("--offset-b", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    learn(
        args.mix, args.track_a, args.track_b, args.bpm,
        data_dir=args.data_dir, base=args.base, output=args.output,
        steps=args.steps, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, augment=args.augment, seed=args.seed,
        truth=args.truth, offset_a=args.offset_a, offset_b=args.offset_b,
        trace_out=args.trace_out, sample_rate=args.sample_rate,
    )
    print(f"next: python scripts/serve_demo.py --model {args.output}")


if __name__ == "__main__":
    main()
