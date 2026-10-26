"""Tests for the PPO training entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from moodmix_env.train import evaluate, make_env, train


def test_make_env_flattens_observation():
    env = make_env(max_steps=8)()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (63,)
    assert env.observation_space.shape == (63,)
    assert env.action_space.nvec.tolist() == [21, 3, 3, 3, 3, 3, 3, 4, 5, 3]
    env.close()


def test_make_env_binary_sync_collapses_the_sync_head():
    env = make_env(max_steps=8, binary_sync=True)()
    assert env.action_space.nvec.tolist() == [21, 3, 3, 3, 3, 3, 3, 2, 5, 3]
    env.close()


def test_make_env_accepts_data_dir():
    demo_dir = Path(__file__).resolve().parents[1] / "data" / "demo"
    env = make_env(max_steps=8, data_dir=demo_dir)()
    obs, info = env.reset(seed=0)
    assert obs.shape == (63,)
    assert info["deck_a"]["id"]
    env.close()


def test_train_smoke_and_evaluate(tmp_path: Path):
    model_path = train(
        timesteps=64, n_envs=2, n_steps=32, batch_size=32, seed=0,
        output=tmp_path / "ppo_smoke", checkpoint_freq=0, verbose=0,
        eval_episodes=1, tensorboard_log=None,
    )
    assert Path(model_path).exists()
    model = PPO.load(model_path)
    env = make_env(max_steps=64)()
    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(action)
    mean, std = evaluate(model, episodes=1, seed=2)
    assert np.isfinite(mean)
    assert std >= 0.0


def test_train_resumes_from_checkpoint(tmp_path: Path):
    first = train(
        timesteps=64, n_envs=1, n_steps=32, batch_size=32, seed=0,
        output=tmp_path / "first", checkpoint_freq=0, verbose=0,
        eval_episodes=0, tensorboard_log=None,
    )
    second = train(
        timesteps=64, n_envs=1, n_steps=32, batch_size=32, seed=0,
        output=tmp_path / "second", resume=first, checkpoint_freq=0,
        verbose=0, eval_episodes=0, tensorboard_log=None,
        learning_rate=1e-4,
    )
    assert Path(second).exists()
    assert Path(second).stat().st_size > 0
    resumed = PPO.load(second)
    assert float(resumed.learning_rate) == 1e-4


def test_train_subproc_smoke(tmp_path: Path):
    path = train(
        timesteps=64, n_envs=2, n_steps=32, batch_size=32, seed=0,
        output=tmp_path / "subproc", checkpoint_freq=0, verbose=0,
        eval_episodes=0, tensorboard_log=None, vec_env="subproc",
    )
    assert Path(path).exists()


def test_checkpoint_freq_is_environment_steps(tmp_path: Path):
    checkpoint_dir = tmp_path / "ckpt"
    train(
        timesteps=64, n_envs=2, n_steps=32, batch_size=32, seed=0,
        output=tmp_path / "model", checkpoint_freq=32,
        checkpoint_dir=checkpoint_dir, verbose=0, eval_episodes=0,
        tensorboard_log=None,
    )
    files = list(checkpoint_dir.glob("*.zip"))
    assert files, "checkpoint callback wrote nothing"
    timesteps = [int(path.stem.split("_")[-2]) for path in files]
    assert min(timesteps) <= 32, timesteps
