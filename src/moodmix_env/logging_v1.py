"""Training records for DJSet-v1: CSV, curves and an HTML report.

The logger writes during training (every ``save_every`` steps), so a killed
run still leaves a usable record.  It logs:

* every finished training episode (return, length),
* a deterministic evaluation every ``eval_freq`` steps (return, low-band
  overlap, level jump, mix quality, vocal terms),
* the PPO losses at each evaluation,
* ``curves.png`` and ``report.html`` / ``report.md``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .env_v1 import MoodMixV1Env

LOSS_KEYS = ("train/policy_loss", "train/value_loss", "train/entropy_loss",
             "train/approx_kl", "train/explained_variance")


class TrainingLogger(BaseCallback):
    def __init__(self, out_dir: str | Path, data_dir: str | Path | None = None,
                 max_steps: int = 32, eval_freq: int = 100_000,
                 eval_episodes: int = 5, save_every: int = 100_000,
                 verbose: int = 1, env_cls=MoodMixV1Env) -> None:
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_steps = int(max_steps)
        self.env_cls = env_cls
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.save_every = int(save_every)
        self.next_eval = self.eval_freq
        self.next_save = self.save_every
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []
        self.loss_rows: list[dict] = []
        self.started = time.time()

    # ------------------------------------------------------------- collection
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.train_rows.append({
                    "timestep": int(self.num_timesteps),
                    "return": float(episode["r"]),
                    "length": int(episode["l"]),
                })
        if self.num_timesteps >= self.next_eval:
            self._run_eval()
            self.next_eval = ((self.num_timesteps // self.eval_freq) + 1)                 * self.eval_freq
        if self.num_timesteps >= self.next_save:
            self.write()
            self.next_save = ((self.num_timesteps // self.save_every) + 1)                 * self.save_every
        return True

    def _losses(self) -> dict[str, float]:
        values = getattr(self.model.logger, "name_to_value", {}) or {}
        return {key: float(values.get(key, float("nan"))) for key in LOSS_KEYS}

    def _run_eval(self) -> None:
        env = self.env_cls(data_dir=self.data_dir, max_steps=self.max_steps)
        returns: list[float] = []
        finals: list[float] = []
        overlaps: list[float] = []
        jumps: list[float] = []
        components: dict[str, list[float]] = {}
        for episode in range(self.eval_episodes):
            obs, _ = env.reset(seed=20_000 + episode)
            total = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            finals.append(float(env.fader))
            overlaps.append(float(np.mean(env._low_overlaps)))
            jumps.append(float(np.mean(env._level_jumps)))
            for name, value in env.episode_components.items():
                components.setdefault(name, []).append(float(value))
        env.close()
        row = {
            "timestep": int(self.num_timesteps),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "final_fader": float(np.mean(finals)),
            "low_overlap": float(np.mean(overlaps)),
            "level_jump": float(np.mean(jumps)),
            **{f"component_{name}": float(np.mean(values))
               for name, values in components.items()},
        }
        losses = self._losses()
        row.update(losses)
        self.eval_rows.append(row)
        self.loss_rows.append({"timestep": int(self.num_timesteps), **losses})
        if self.verbose:
            print(f"[log] {self.num_timesteps:>9,d}  "
                  f"eval {row['mean_return']:+.3f}  "
                  f"low {row['low_overlap']:.3f}  "
                  f"jump {row['level_jump']:.3f}", flush=True)

    # --------------------------------------------------------------- writing
    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv("train_returns.csv",
                        ["timestep", "return", "length"], self.train_rows)
        eval_columns = sorted({key for row in self.eval_rows for key in row})
        self._write_csv("eval.csv", eval_columns, self.eval_rows)
        self._write_csv("losses.csv", ["timestep", *LOSS_KEYS], self.loss_rows)
        self._plot()
        self._report()

    def _write_csv(self, name: str, columns: list[str],
                   rows: list[dict]) -> None:
        if not columns:
            columns = ["timestep"]
        with (self.out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _plot(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        ax = axes[0, 0]
        if self.train_rows:
            x = np.asarray([row["timestep"] for row in self.train_rows])
            y = np.asarray([row["return"] for row in self.train_rows])
            ax.scatter(x, y, s=6, alpha=0.25, color="#b45309")
            if len(y) >= 20:
                window = max(5, len(y) // 40)
                smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                ax.plot(x[window - 1:], smooth, color="#b45309", linewidth=1.6)
        ax.set_title("training episode return")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.2)

        ax = axes[0, 1]
        if self.eval_rows:
            x = [row["timestep"] for row in self.eval_rows]
            y = [row["mean_return"] for row in self.eval_rows]
            e = [row["std_return"] for row in self.eval_rows]
            ax.errorbar(x, y, yerr=e, marker="o", markersize=3,
                        color="#a84a06", linewidth=1.4, capsize=2)
        ax.set_title("deterministic eval return")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.2)

        ax = axes[0, 2]
        if self.eval_rows:
            x = [row["timestep"] for row in self.eval_rows]
            ax.plot(x, [row["low_overlap"] for row in self.eval_rows],
                    marker="o", markersize=2, label="low overlap",
                    color="#0e7490")
            ax.plot(x, [row["level_jump"] for row in self.eval_rows],
                    marker="o", markersize=2, label="level jump",
                    color="#b91c1c")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
        ax.set_title("mix continuity")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.2)

        ax = axes[1, 0]
        if self.eval_rows:
            x = [row["timestep"] for row in self.eval_rows]
            for name, color in (("component_mix_quality", "#15803d"),
                                ("component_vocal_clash", "#7c3aed"),
                                ("component_vocal_space", "#a21caf")):
                values = [row.get(name, 0.0) for row in self.eval_rows]
                ax.plot(x, values, marker="o", markersize=2, label=name.replace("component_", ""),
                        color=color, linewidth=1.2)
            ax.legend(fontsize=7)
        ax.set_title("mix quality and vocal terms")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.2)

        ax = axes[1, 1]
        if self.eval_rows:
            x = [row["timestep"] for row in self.eval_rows]
            ax.plot(x, [row.get("train/policy_loss", np.nan) for row in self.eval_rows],
                    label="policy", color="#1d4ed8")
            ax.plot(x, [row.get("train/value_loss", np.nan) for row in self.eval_rows],
                    label="value", color="#b45309")
            ax.legend(fontsize=7)
        ax.set_title("PPO losses")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.2)

        ax = axes[1, 2]
        if self.eval_rows:
            x = [row["timestep"] for row in self.eval_rows]
            ax.plot(x, [row.get("train/entropy_loss", np.nan) for row in self.eval_rows],
                    label="entropy", color="#0f766e")
            ax.plot(x, [row.get("train/approx_kl", np.nan) for row in self.eval_rows],
                    label="approx KL", color="#7c3aed")
            ax.legend(fontsize=7)
        ax.set_title("exploration / update size")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.2)

        fig.suptitle(f"DJSet-v1 training  ·  {self.out_dir.name}", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.out_dir / "curves.png", dpi=140)
        plt.close(fig)

    def _report(self) -> None:
        latest = self.eval_rows[-1] if self.eval_rows else {}
        config = {
            "run": self.out_dir.name,
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "max_steps": self.max_steps,
            "eval_freq": self.eval_freq,
            "eval_episodes": self.eval_episodes,
            "episodes_logged": len(self.train_rows),
            "elapsed_seconds": round(time.time() - self.started, 1),
        }
        (self.out_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8")
        lines = [
            f"# Training run: {self.out_dir.name}",
            "",
            f"- data: `{config['data_dir']}`",
            f"- episodes logged: {config['episodes_logged']}",
            f"- elapsed: {config['elapsed_seconds']} s",
            "",
            "## Latest deterministic eval",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for key in ("timestep", "mean_return", "std_return", "final_fader",
                    "low_overlap", "level_jump", "component_mix_quality",
                    "component_vocal_clash", "component_vocal_space"):
            if key in latest:
                value = latest[key]
                lines.append(f"| {key} | {value:.4f} |"
                             if isinstance(value, (int, float))
                             else f"| {key} | {value} |")
        lines += ["", "![training curves](curves.png)", ""]
        (self.out_dir / "report.md").write_text("\n".join(lines),
                                                encoding="utf-8")
        rows = "".join(
            f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
            if isinstance(value, (int, float)) else
            f"<tr><td>{key}</td><td>{value}</td></tr>"
            for key, value in latest.items()
        )
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{self.out_dir.name} training</title>
<style>
body{{background:#f5f6f8;color:#151a22;font-family:system-ui,sans-serif;
margin:0;padding:28px;line-height:1.5}}
.wrap{{max-width:1100px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 4px}} .sub{{color:#5b6472;font-family:ui-monospace;
font-size:.8rem;margin-bottom:20px}} img{{width:100%;border:1px solid #d7dde5;
background:#fff}} table{{border-collapse:collapse;margin-top:16px;font-size:.85rem}}
td,th{{border-bottom:1px solid #d7dde5;padding:6px 12px;text-align:left}}
pre{{background:#fff;border:1px solid #d7dde5;padding:12px;overflow:auto;font-size:.78rem}}
</style></head><body><div class="wrap">
<h1>DJSet-v1 training run</h1>
<div class="sub">{config['data_dir']} · {config['episodes_logged']} episodes · {config['elapsed_seconds']} s</div>
<img src="curves.png" alt="training curves">
<table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h2>Config</h2><pre>{json.dumps(config, indent=2)}</pre>
</div></body></html>"""
        (self.out_dir / "report.html").write_text(html, encoding="utf-8")

    def _on_training_end(self) -> None:
        self.write()
