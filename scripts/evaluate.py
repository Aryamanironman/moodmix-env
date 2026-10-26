"""Compare a trained PPO model against the hand-written DJ and random play.

    python scripts/evaluate.py --model models/ppo_100k.zip --episodes 10
    python scripts/evaluate.py --model models/ppo_100k.zip --audio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO

from moodmix_env import MoodMixEnv
from moodmix_env.env import REWARD_PROFILES
from moodmix_env.evaluation import evaluate_policy, model_policy, random_policy
from moodmix_env.policies import heuristic_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder containing manifest.json (default: demo crate)")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES,
                        default="expressive",
                        help="which reward to score with; curve reproduces the "
                             "numbers in older notes")
    parser.add_argument("--audio", action="store_true",
                        help="render the model's first evaluation episode")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "model_eval.wav")
    args = parser.parse_args()

    model = PPO.load(str(args.model))
    console = Console()

    env = MoodMixEnv(data_dir=args.data_dir, max_steps=64,
                     reward_profile=args.reward_profile)
    policies = {
        "PPO": model_policy(model),
        "human": heuristic_action,
        "random": random_policy(env),
    }
    results = {name: evaluate_policy(policy, episodes=args.episodes,
                                     seed=args.seed, data_dir=args.data_dir,
                                     reward_profile=args.reward_profile)
               for name, policy in policies.items()}

    table = Table(title=f"MoodMix policy comparison ({args.reward_profile})",
                  border_style="bright_magenta")
    for column in ("policy", "reward", "std", "completed", "final fader",
                   "curve error", "sync"):
        table.add_column(column, justify="right" if column != "policy" else "left")
    for name, result in results.items():
        table.add_row(
            name,
            f"{result['mean']:+.2f}",
            f"{result['std']:.2f}",
            f"{result['completed'] * 100:.0f}%",
            f"{result['final_fader']:.2f}",
            f"{result['mean_deviation']:.3f}",
            f"{result['sync_fraction'] * 100:.0f}%",
        )
    console.print(table)

    ppo = results["PPO"]["mean"]
    human = results["human"]["mean"]
    if ppo > human:
        console.print(f"[bold green]PPO beats the human baseline by "
                      f"{ppo - human:+.2f}[/bold green]")
    else:
        console.print(f"[bold yellow]PPO is {human - ppo:.2f} below the human "
                      f"baseline[/bold yellow]")

    if args.audio:
        from moodmix_env.render_audio import render_transition
        env = MoodMixEnv(data_dir=args.data_dir, max_steps=64)
        obs, info = env.reset(seed=args.seed)
        policy = model_policy(model)
        while True:
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        render_transition(env, args.output)
        console.print(f"audio mixdown: [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    main()
