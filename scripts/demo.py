"""Run one MoodMix transition, a set, or a comparison in the terminal.

    python scripts/demo.py
    python scripts/demo.py --mood party --policy demo --speed 0.08
    python scripts/demo.py --policy random --no-live --steps 32

    # trained policy
    python scripts/demo.py --model models/ppo_final.zip --no-live
    python scripts/demo.py --model models/ppo_final.zip --set --transitions 4
    python scripts/demo.py --model models/ppo_final.zip --samples 6
    python scripts/demo.py --model models/ppo_final.zip --compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.live import Live
from rich.table import Table

from moodmix_env import MOODS, MoodMixEnv
from moodmix_env.env import COMPATIBLE_MOODS, EQ_GAINS, _key_distance
from moodmix_env.evaluation import OBS_ORDER, model_policy, random_policy
from moodmix_env.policies import heuristic_action
from moodmix_env.render_audio import (
    _BandSplit, _read_mono, _rms, _source_slice, _time_stretch,
    finalise, render_mix, write_wav,
)
from moodmix_env.viz import dashboard

CONSOLE = Console()


# --------------------------------------------------------------- policies
def _flatten(obs: dict) -> np.ndarray:
    return np.concatenate([obs[key] for key in OBS_ORDER]).astype(np.float32)


def stochastic_model_policy(model):
    """Rollouts from the policy distribution, not the argmax.

    A demo of one deterministic path shows what the agent can do; a handful of
    stochastic ones show what it *learned*, including the variance.
    """
    def _policy(obs, info):
        action, _ = model.predict(_flatten(obs), deterministic=False)
        return action
    return _policy


def _load_model(path: Path | None):
    if path is None:
        return None
    from stable_baselines3 import PPO
    return PPO.load(str(path))


def _require_model(args, model):
    if model is None:
        CONSOLE.print("[red]this mode needs --model PATH[/red]")
        raise SystemExit(2)
    if tuple(model.observation_space.shape) != (63,):
        CONSOLE.print(
            f"[red]model expects observation {model.observation_space.shape}, "
            f"the environment is (63,)[/red]. Pre-Phase-4 checkpoints are "
            "incompatible; retrain with scripts/train.py."
        )
        raise SystemExit(2)


# ---------------------------------------------------------------- pairs
def forgiving_pairs(env: MoodMixEnv, limit: int = 5,
                    max_bpm_ratio: float = 0.05,
                    max_key_distance: int = 5) -> list[tuple[str, str]]:
    """Pairs a DJ would not have to fight: tempo-close and key-close.

    Drum-only intros/outros are already how the demo crate is built (deck B
    starts in its intro, deck A in its drop), so tempo and key are the two
    choices left.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a in env.tracks:
        for b in env.tracks:
            if a["id"] == b["id"]:
                continue
            ratio = float(b["bpm"]) / float(a["bpm"])
            if abs(ratio - 1.0) > max_bpm_ratio:
                continue
            if _key_distance(a, b) > max_key_distance:
                continue
            if (b["mood"] != a["mood"]
                    and b["mood"] not in COMPATIBLE_MOODS.get(a["mood"], ())):
                continue
            key = (a["id"], b["id"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs[:max(1, limit)]


def _pair_options(args, env: MoodMixEnv) -> dict:
    if args.track_a and args.track_b:
        return {"track_a": args.track_a, "track_b": args.track_b}
    if args.track_a:
        return {"track_a": args.track_a}
    pairs = forgiving_pairs(env, limit=12)
    if args.mood:
        preferred = [p for p in pairs
                     if env.tracks[[t["id"] for t in env.tracks].index(p[0])]["mood"]
                     == args.mood]
        pairs = preferred or pairs
    if not pairs:
        return {}
    return {"track_a": pairs[0][0], "track_b": pairs[0][1]}


# ------------------------------------------------------------- episodes
def run_episode(env: MoodMixEnv, policy, options: dict | None = None,
                seed: int | None = None) -> dict:
    obs, info = env.reset(seed=seed, options=options or {})
    rewards: list[float] = []
    sync_beats = 0
    faders: list[float] = []
    while True:
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        sync_beats += int(info["sync"])
        faders.append(float(info["crossfader"]))
        if terminated or truncated:
            break
    deltas = np.diff(faders)
    signs = np.sign(deltas[deltas != 0])
    reversals = int(np.sum(signs[1:] * signs[:-1] < 0))
    return {
        "mood": info["mood"],
        "track_a": env.track_a["id"],
        "track_b": env.track_b["id"],
        "steps": env.step_count,
        "total": float(sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "final_fader": float(faders[-1]),
        "sync_fraction": sync_beats / max(1, env.step_count),
        "reversals": reversals,
        "components": dict(env.episode_components),
    }


# ---------------------------------------------------------------- render
def _crossfade(a: np.ndarray, b: np.ndarray, sr: int,
               seconds: float = 0.35) -> np.ndarray:
    """Join two clips without a click at the seam."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(int(seconds * sr), len(a), len(b))
    if n <= 1:
        return np.concatenate([a, b])
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.concatenate([a[:-n], a[-n:] * (1 - fade) + b[:n] * fade, b[n:]])


def render_tail(env: MoodMixEnv, sr: int, bars: int = 2) -> np.ndarray:
    """The outgoing deck's next few bars, so a set breathes between mixes.

    The tail is deck B, the same track the next transition hands over to, at
    the tempo and EQ the transition left it on.
    """
    beat_a = 60.0 / float(env.track_a["bpm"])
    audio, _ = _read_mono(env.data_dir / env.track_b["file"], sr=sr)
    audio = _time_stretch(audio, float(env.track_b["bpm"]) /
                          float(env.track_a["bpm"]))
    length = max(1, int(round(bars * 4 * beat_a * sr)))
    last = env.trace[-1] if env.trace else {"pos_b": 0.0, "eq_b": (1, 1, 1),
                                            "filter": 0.0}
    start = int(round(float(last["pos_b"]) * beat_a * sr))
    low, mid, high = _BandSplit(sr).process(_source_slice(audio, start, length))
    eq = EQ_GAINS[[int(v) for v in last["eq_b"]]]
    filt = int(last["filter"])
    if filt == 1:
        high = high * 0.15
    elif filt == 2:
        low = low * 0.12
    return (low * eq[0] + mid * eq[1] + high * eq[2]).astype(np.float32)


def _level_match(tail: np.ndarray, mix: np.ndarray) -> np.ndarray:
    reference = _rms(mix[-len(tail):]) or _rms(mix)
    level = _rms(tail)
    if level > 1e-4 and reference > 1e-4:
        tail = tail * float(np.clip(reference / level, 0.25, 4.0))
    return tail.astype(np.float32)


# ----------------------------------------------------------------- modes
def mode_set(args, env: MoodMixEnv, model) -> None:
    _require_model(args, model)
    policy = model_policy(model)
    pairs = forgiving_pairs(env, limit=max(2, args.transitions))
    if not pairs:
        CONSOLE.print("[red]no forgiving pairs in this crate[/red]")
        raise SystemExit(2)
    pairs = pairs[:args.transitions]

    pieces: list[np.ndarray] = []
    sample_rate = 44100
    rows = []
    for index, (a, b) in enumerate(pairs):
        result = run_episode(env, policy, {"track_a": a, "track_b": b},
                             seed=args.seed if args.seed is not None
                             else 1000 + index)
        mix, sample_rate = render_mix(env, sample_rate)
        result["duration_seconds"] = round(len(mix) / sample_rate, 2)
        rows.append(result)
        pieces.append(mix)
        if index < len(pairs) - 1:
            pieces.append(_level_match(render_tail(env, sample_rate), mix))

    joined = pieces[0]
    for piece in pieces[1:]:
        joined = _crossfade(joined, piece, sample_rate)
    out = args.out_dir / f"set_{len(pairs)}transitions.wav"
    write_wav(finalise(joined, sample_rate), sample_rate, out)

    table = Table(title="MoodMix set", border_style="bright_magenta")
    for column in ("#", "deck A", "deck B", "reward", "fader", "sync", "rev", "secs"):
        table.add_column(column, justify="right" if column != "#" else "left")
    for index, row in enumerate(rows, 1):
        table.add_row(str(index), row["track_a"], row["track_b"],
                      f"{row['total']:+.2f}", f"{row['final_fader']:.2f}",
                      f"{row['sync_fraction'] * 100:.0f}%",
                      str(row["reversals"]), f"{row['duration_seconds']:.1f}")
    CONSOLE.print(table)
    CONSOLE.print(f"set audio: [cyan]{out}[/cyan]  "
                  f"({len(joined) / sample_rate:.1f}s)")


def mode_samples(args, env: MoodMixEnv, model) -> None:
    _require_model(args, model)
    policy = stochastic_model_policy(model)
    options = _pair_options(args, env)
    out_dir = args.out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(args.samples):
        result = run_episode(env, policy, options,
                             seed=(args.seed + index) if args.seed is not None
                             else 1000 + index)
        name = (f"{result['track_a']}__{result['track_b']}__sample{index + 1}"
                ".wav")
        mix, sr = render_mix(env)
        write_wav(finalise(mix, sr), sr, out_dir / name)
        result["file"] = str(out_dir / name)
        rows.append(result)

    table = Table(title=f"{args.samples} stochastic rollouts of one pair",
                  border_style="bright_magenta")
    for column in ("sample", "reward", "fader", "sync", "rev", "file"):
        table.add_column(column, justify="left" if column == "file" else "right")
    for index, row in enumerate(rows, 1):
        table.add_row(f"{index:02d}", f"{row['total']:+.2f}",
                      f"{row['final_fader']:.2f}",
                      f"{row['sync_fraction'] * 100:.0f}%",
                      str(row["reversals"]), Path(row["file"]).name)
    CONSOLE.print(table)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    CONSOLE.print(f"labels: [cyan]{manifest}[/cyan]")


def _metric_row(name: str, result: dict) -> tuple[str, ...]:
    return (name, f"{result['total']:+.2f}",
            f"{result['final_fader']:.2f}",
            f"{result['sync_fraction'] * 100:.0f}%",
            str(result["reversals"]), f"{result['mean_reward']:+.3f}")


def mode_compare(args, env: MoodMixEnv, model) -> None:
    _require_model(args, model)
    options = _pair_options(args, env)
    results = {}
    clips = []
    for name, policy in (("human", heuristic_action),
                         ("model", model_policy(model))):
        results[name] = run_episode(env, policy, options, seed=args.seed)
        mix, sr = render_mix(env)
        results[name]["file"] = str(args.out_dir / f"compare_{name}.wav")
        write_wav(finalise(mix, sr), sr, args.out_dir / f"compare_{name}.wav")
        clips.append(finalise(mix, sr))

    gap = np.zeros(int(0.4 * sr), dtype=np.float32)
    side_by_side = np.concatenate([clips[0], gap, clips[1]])
    out = args.out_dir / "compare_side_by_side.wav"
    write_wav(side_by_side, sr, out)

    table = Table(title="Same pair: human vs trained policy",
                  border_style="bright_magenta")
    for column in ("policy", "reward", "fader", "sync", "rev", "per step"):
        table.add_column(column, justify="left" if column == "policy" else "right")
    for name in ("human", "model"):
        table.add_row(*_metric_row(name, results[name]))
    CONSOLE.print(table)
    CONSOLE.print(f"human first, then model, 0.4s gap: [cyan]{out}[/cyan]")


# ----------------------------------------------------------- single mode
def print_summary(info: dict, history: list[float], policy: str) -> None:
    components = info.get("episode_components", {})
    table = Table(title=f"{policy.upper()} POLICY - episode summary",
                  border_style="bright_magenta")
    table.add_column("signal")
    table.add_column("total", justify="right")
    table.add_column("per step", justify="right")
    steps = max(1, info["step"])
    for name, value in components.items():
        table.add_row(name.replace("_", " "), f"{value:+.2f}", f"{value / steps:+.3f}")
    table.add_section()
    table.add_row("TOTAL", f"{sum(components.values()):+.2f}", "")
    table.add_row("MEAN REWARD", f"{sum(history) / len(history):+.3f}", "")
    table.add_row("final gain B", f"{info['gain_b']:.2f}", "")
    table.add_row("final phase sync", "yes" if info["sync"] else "no", "")
    CONSOLE.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mood", choices=MOODS, default=None)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder containing manifest.json (default: demo crate)")
    parser.add_argument("--model", type=Path, default=None,
                        help="trained .zip for the model/set/samples/compare modes")
    parser.add_argument("--policy", choices=("demo", "random", "model"),
                        default="demo")
    parser.add_argument("--track-a", default=None, help="pin deck A by track id")
    parser.add_argument("--track-b", default=None, help="pin deck B by track id")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--speed", type=float, default=0.09,
                        help="seconds between beats (0 for instant)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--audio", action="store_true",
                        help="render the transition to runs/<mood>_<policy>.wav")
    parser.add_argument("--set", dest="set_mode", action="store_true",
                        help="render 3-5 transitions into one set file")
    parser.add_argument("--transitions", type=int, default=4,
                        help="transitions in --set mode (2-8)")
    parser.add_argument("--samples", type=int, default=0,
                        help="render N stochastic rollouts of one pair")
    parser.add_argument("--compare", action="store_true",
                        help="human vs trained policy on one pair")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "demo")
    args = parser.parse_args()
    args.transitions = max(2, min(8, args.transitions))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = MoodMixEnv(data_dir=args.data_dir, mood=args.mood, max_steps=args.steps)
    model = _load_model(args.model)

    if args.set_mode:
        mode_set(args, env, model)
        return
    if args.samples:
        mode_samples(args, env, model)
        return
    if args.compare:
        mode_compare(args, env, model)
        return

    if args.policy == "model":
        _require_model(args, model)
        policy = model_policy(model)
    elif args.policy == "random":
        policy = random_policy(env)
    else:
        policy = heuristic_action

    obs, info = env.reset(seed=args.seed, options=_pair_options(args, env))
    history: list[float] = []
    actions: list[list[int]] = []

    live = None if args.no_live else Live(
        dashboard(info, history, args.policy),
        console=CONSOLE, refresh_per_second=20,
    )
    if live is not None:
        live.start()
    try:
        done = False
        while not done:
            action = policy(obs, info)
            actions.append([int(v) for v in action])
            obs, reward, terminated, truncated, info = env.step(action)
            history.append(float(reward))
            done = terminated or truncated
            if live is not None:
                live.update(dashboard(info, history, args.policy))
                if args.speed > 0:
                    time.sleep(args.speed)
            elif args.no_live:
                env.render()
    finally:
        if live is not None:
            live.stop()

    info["episode_components"] = env.episode_components
    info["step"] = env.step_count
    print_summary(info, history, args.policy)

    if args.audio:
        from moodmix_env.render_audio import render_transition
        wav = ROOT / "runs" / f"{info['mood']}_{args.policy}.wav"
        render_transition(env, wav)
        CONSOLE.print(f"audio mixdown: [cyan]{wav}[/cyan]")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({
            "mood": info["mood"],
            "policy": args.policy,
            "deck_a": info["deck_a"],
            "deck_b": info["deck_b"],
            "actions": actions,
            "rewards": history,
            "components": env.episode_components,
        }, indent=2), encoding="utf-8")
        CONSOLE.print(f"saved [cyan]{args.save}[/cyan]")


if __name__ == "__main__":
    main()
