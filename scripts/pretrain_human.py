"""Warm-start the PPO policy by imitating the hand-written DJ.

PPO learns a transition from scratch by trial and error, which is why it can
score well while still moving the fader unlike a DJ.  This script collects
rollouts of the human baseline (`moodmix_env.policies.heuristic_action`) on a
crate and fits the policy network to those actions with cross-entropy, then
saves a checkpoint that `scripts/train.py --resume` can keep training from.

    python scripts/pretrain_human.py --model models/ppo_1m.zip \
        --data-dir crate --episodes 200 --output models/ppo_bc.zip

The expert here is the hand-written baseline, not recordings of real DJ sets;
swap `expert_policy` if you add a dataset of real transitions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.env import MoodMixEnv  # noqa: E402
from moodmix_env.policies import heuristic_action  # noqa: E402

OBS_ORDER = ("context", "deck_a", "deck_b", "mixer")


def _flatten(obs: dict) -> np.ndarray:
    return np.concatenate([obs[key] for key in OBS_ORDER]).astype(np.float32)


def collect_expert(
    data_dir: str | None,
    episodes: int,
    max_steps: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    env = MoodMixEnv(data_dir=data_dir, max_steps=max_steps)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        while True:
            action = heuristic_action(obs, info)
            observations.append(_flatten(obs))
            actions.append(np.asarray(action, dtype=np.int64))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    env.close()
    return np.stack(observations), np.stack(actions)


def fit_policy(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
    dims: Sequence[int] | None = None,
) -> float:
    """Cross-entropy fit of the policy heads to `actions`.

    `dims` restricts the loss to some control dims.  A signal recovered from a
    mix only contains the crossfader and EQ -- the filter, drum pads and sync
    button are not visible in the finished audio -- so those heads are left
    untouched rather than trained on invented labels.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = model.policy
    nvec = [int(v) for v in model.action_space.nvec]
    heads_trained = list(range(len(nvec))) if dims is None else [int(d) for d in dims]
    if not heads_trained:
        raise ValueError("dims must select at least one control")
    features = torch.as_tensor(observations)
    targets = torch.as_tensor(actions)

    params = list(policy.mlp_extractor.parameters()) + list(policy.action_net.parameters())
    optimiser = torch.optim.Adam(params, lr=learning_rate)
    policy.train()

    n = len(features)
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        correct = np.zeros(len(heads_trained), dtype=np.int64)
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_x = features[idx]
            batch_y = targets[idx]
            latent = policy.mlp_extractor.forward_actor(batch_x)
            logits = policy.action_net(latent)
            split = torch.split(logits, nvec, dim=1)
            loss = sum(
                nn.functional.cross_entropy(split[head], batch_y[:, head])
                for head in heads_trained
            ) / len(heads_trained)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += float(loss) * len(idx)
            with torch.no_grad():
                for slot, head in enumerate(heads_trained):
                    correct[slot] += int(
                        (split[head].argmax(dim=1) == batch_y[:, head]).sum()
                    )
            seen += len(idx)
        accuracy = correct / max(1, seen)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {total_loss / max(1, seen):.4f}  "
              f"mean dim-accuracy {accuracy.mean():.3f}")
    policy.eval()
    return float(accuracy.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "ppo_1m.zip",
                        help="checkpoint whose policy is imitated and saved")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "crate")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ppo_bc.zip")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(str(args.model), device="cpu")
    observations, actions = collect_expert(
        args.data_dir, args.episodes, args.max_steps, seed=args.seed
    )
    print(f"expert samples: {len(observations)}  action dims: {actions.shape[1]}")

    accuracy = fit_policy(
        model, observations, actions,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"\nsaved {args.output}  (agreement with the human policy {accuracy:.1%})")
    print("next: python scripts/train.py --resume", args.output,
          "--data-dir", args.data_dir, "--timesteps 600000")


if __name__ == "__main__":
    main()
