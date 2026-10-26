"""Ablation runner for the reward and sync fixes.

Trains the same short PPO run with one piece switched off at a time, logs the
episodic return against environment steps to CSV, plots the curves, and prints
one summary table.

    python scripts/ablate.py --timesteps 100000 --out-dir runs/ablation
    python scripts/ablate.py --variants full,no_sync_fix --timesteps 20000

Each run is sequential: the machine is one laptop, not a cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from moodmix_env.env import (
    SYNC_ENGAGE_BONUS, MoodMixEnv, _energy_arc, _smoothstep,
)
from moodmix_env.evaluation import model_policy
from moodmix_env.train import train

VARIANTS: dict[str, dict] = {
    "full": dict(binary_sync=True, sync_bonus=SYNC_ENGAGE_BONUS,
                 term_scales={}),
    "no_sync_fix": dict(binary_sync=False, sync_bonus=0.0, term_scales={}),
    "no_bass_swap": dict(binary_sync=True, sync_bonus=SYNC_ENGAGE_BONUS,
                         term_scales={"bass_swap": 0.0}),
    "no_pad_filter": dict(binary_sync=True, sync_bonus=SYNC_ENGAGE_BONUS,
                          term_scales={"pad_craft": 0.0,
                                       "filter_craft": 0.0}),
}


class ReturnLogger(BaseCallback):
    """Collect VecMonitor's per-episode returns with the step they ended at."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.rows.append((int(self.num_timesteps),
                                  float(episode["r"])))
        return True


def _shape_share(env: MoodMixEnv) -> float:
    """Share of the episode's reward that came from tracking the curve.

    The expressive profile keeps the authored envelope only inside the
    terminal mix_quality (0.20 of its quality score); this isolates that slice
    instead of counting the whole terminal bonus as curve-following.
    """
    frames = env.trace
    if not frames:
        return 0.0
    shape = float(np.mean([t["shape"] for t in frames]))
    music = float(np.mean([
        0.35 * t["align"] + 0.25 * t["tempo"]
        + 0.20 * t["harmonic"] + 0.20 * t["spectral"] for t in frames
    ]))
    fader = float(env.trace[-1]["crossfader"])
    gain_b = float(np.sin(fader * np.pi / 2))
    ending = 1.0 if fader >= 0.9 and gain_b > 0.8 else (0.5 if fader > 0.6 else 0.0)
    span, _, arc = _energy_arc(
        np.asarray([t["energy"] for t in frames], dtype=np.float64))
    expression = float(np.clip(span / 0.25, 0.0, 1.0)) * arc
    jumps = float(np.mean([1.0 if t["jump"] > 0.30 else 0.0 for t in frames]))
    raw = (0.30 * music + 0.20 * shape + 0.25 * ending
           + 0.25 * expression - 0.15 * jumps)
    if raw <= 1e-9:
        return 0.0
    quality = float(np.clip(raw, 0.0, 1.0))
    shape_part = 10.0 * quality * (0.20 * shape / raw)
    total = float(sum(env.episode_components.values()))
    return shape_part / total if total else 0.0


def evaluate_model(path: str | Path, episodes: int = 10,
                   seed: int = 10_000) -> dict[str, float]:
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    policy = model_policy(model)
    # Evaluate in the four-action environment: a binary policy only emits 0/1,
    # while a no-sync-fix policy must be allowed to emit the release/nudge it
    # learned.  Clipping those to 1 would silently rewrite its behaviour.
    env = MoodMixEnv(max_steps=64, reward_profile="expressive",
                     binary_sync=False)
    returns, aligns, sync_fraction, shares = [], [], [], []
    gate_kept = gate_potential = 0.0
    reversals = []

    for index in range(episodes):
        obs, info = env.reset(seed=seed + index)
        total = 0.0
        faders, syncs = [], []
        steps = 0
        while True:
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            faders.append(float(info["crossfader"]))
            syncs.append(int(info["sync"]))
            steps += 1
            # Diagnostic only: expressive has no gate.  This measures how far
            # the policy's fader sits from the authored curve, on the curve
            # profile's weights, so the columns stay comparable with old runs.
            target = _smoothstep((info["step"] / info["max_steps"] - 0.12) / 0.68)
            window = float(np.clip(
                1.0 - abs(info["crossfader"] - target) / 0.18, 0.0, 1.0))
            frame = env.trace[-1]
            potential = (0.22 * frame["align"] + 0.13 * frame["tempo"]
                         + 0.13 * frame["harmonic"]
                         + 0.15 * frame["spectral"])
            gate_potential += potential
            gate_kept += potential * window
            if terminated or truncated:
                break
        deltas = np.diff(faders)
        signs = np.sign(deltas[deltas != 0])
        reversals.append(int(np.sum(signs[1:] * signs[:-1] < 0)))
        aligns.append(float(np.mean([t["align"] for t in env.trace])))
        sync_fraction.append(float(np.mean(syncs)))
        shares.append(_shape_share(env))
        returns.append(total)

    return {
        "final_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "beat_alignment": float(np.mean(aligns)),
        "gate_withhold_pct": 100.0 * (1.0 - gate_kept / gate_potential)
        if gate_potential else 0.0,
        "smoothstep_share_pct": 100.0 * float(np.mean(shares)),
        "sync_pct": 100.0 * float(np.mean(sync_fraction)),
        "reversals": float(np.mean(reversals)),
    }


def write_plot(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib is not installed", flush=True)
        return
    variants = sorted({row["variant"] for row in rows})
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for variant in variants:
        points = sorted((row["timestep"], row["return"])
                        for row in rows if row["variant"] == variant)
        if not points:
            continue
        x = np.asarray([p[0] for p in points], dtype=float)
        y = np.asarray([p[1] for p in points], dtype=float)
        if len(y) >= 7:
            window = max(3, len(y) // 10)
            y = np.convolve(y, np.ones(window) / window, mode="valid")
            x = x[window - 1:]
        ax.plot(x, y, label=variant, linewidth=1.4)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (smoothed)")
    ax.set_title("MoodMix ablation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[plot] {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10,
                        help="deterministic evaluation episodes per variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"comma-separated subset of {', '.join(VARIANTS)}")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "ablation")
    args = parser.parse_args()

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    summary: list[dict] = []
    for name in names:
        config = VARIANTS[name]
        print(f"\n=== {name} ===", flush=True)
        logger = ReturnLogger()
        model_path = train(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            output=args.out_dir / name,
            checkpoint_freq=0,
            eval_episodes=0,
            verbose=0,
            tensorboard_log=None,
            device=args.device,
            data_dir=None,
            reward_profile="expressive",
            binary_sync=config["binary_sync"],
            sync_bonus=config["sync_bonus"],
            term_scales=config["term_scales"],
            extra_callbacks=[logger],
        )
        for timestep, value in logger.rows:
            rows.append({"variant": name, "timestep": timestep,
                         "return": value})
        metrics = evaluate_model(model_path, episodes=args.episodes)
        metrics["variant"] = name
        metrics["timesteps"] = args.timesteps
        metrics["model"] = str(model_path)
        summary.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    csv_path = args.out_dir / "returns.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "timestep", "return"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "summary.csv"
    columns = ["variant", "timesteps", "final_return", "return_std",
               "beat_alignment", "gate_withhold_pct", "smoothstep_share_pct",
               "sync_pct", "reversals", "model"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key) for key in columns})

    write_plot(rows, args.out_dir / "returns.png")

    print("\n" + "=" * 78)
    print(f"{'variant':16s} {'return':>8s} {'align':>7s} {'gate%':>7s} "
          f"{'curve%':>7s} {'sync%':>7s} {'rev':>5s}")
    for row in summary:
        print(f"{row['variant']:16s} {row['final_return']:+8.2f} "
              f"{row['beat_alignment']:7.3f} {row['gate_withhold_pct']:7.1f} "
              f"{row['smoothstep_share_pct']:7.1f} {row['sync_pct']:7.1f} "
              f"{row['reversals']:5.1f}")
    print(f"\nreturns {csv_path}\nsummary {summary_path}")


if __name__ == "__main__":
    main()
