"""DJSet-v3: a remix surface with drum stems and echo sends.

The control surface extends DJSet-v1 with four new continuous controls:

    drum A (1), drum B (1), echo A (1), echo B (1)

Drum gain acts on the percussive component of each deck.  Echo is a
beat-synced send.  The original fader, EQ and filter controls keep their
meaning, so a v1 transition still works as the first eight dimensions.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env_v1 import MoodMixV1Env, _overlap


ECHO_BUDGET = 6
DRUM_SWAP_COOLDOWN = 8


class MoodMixV3Env(MoodMixV1Env):
    """Continuous DJ transition plus drum-stem and echo controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                         dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(48,), dtype=np.float32)

    def _init_state(self):
        super()._init_state()
        self.drum_a = 1.0
        self.drum_b = 1.0
        self.echo_a = 0.0
        self.echo_b = 0.0
        self._prev_drum_a = 1.0
        self._prev_drum_b = 1.0
        self._drum_owner = 0
        self._drum_cooldown = 0
        self._drum_clashes: list[float] = []
        self._echo_beats_a = 0
        self._echo_beats_b = 0

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1),
                         self.action_space.low, self.action_space.high)
        if action.shape != (12,):
            raise ValueError(f"expected 12 controls, got {action.shape}")

        self._prev_fader = self.fader
        self._prev_filter = self.filter
        self._prev_eq_a = self.eq_a.copy()
        self._prev_eq_b = self.eq_b.copy()
        self._prev_drum_a = self.drum_a
        self._prev_drum_b = self.drum_b
        self.fader = float(action[0])
        self.eq_a = action[1:4].astype(np.float32)
        self.eq_b = action[4:7].astype(np.float32)
        self.filter = float(action[7])
        self.drum_a = float(action[8])
        self.drum_b = float(action[9])
        self.echo_a = float(action[10])
        self.echo_b = float(action[11])

        components, detail = self._reward_components(action)
        reward = float(sum(components.values()))
        self.trace.append({"action": action.copy(), **detail})
        for name, value in components.items():
            self.episode_components[name] = \
                self.episode_components.get(name, 0.0) + float(value)
        self.last_components = components
        self.last_reward = reward

        self.step_count += 1
        self._swap_cooldown = max(0, self._swap_cooldown - 1)
        self._drum_cooldown = max(0, self._drum_cooldown - 1)
        terminated = self.step_count >= self.max_steps
        if terminated:
            bonus = self._end_bonus()
            components.update(bonus)
            reward += float(sum(bonus.values()))
            for name, value in bonus.items():
                self.episode_components[name] = \
                    self.episode_components.get(name, 0.0) + float(value)
            self.last_components = components

        obs = self._observation()
        info = self._info()
        info["reward_components"] = components
        return obs, reward, bool(terminated), False, info

    def _percussive(self, track: dict, beat: int) -> float:
        curve = track["features"].get("percussive")
        if curve is None:
            return 0.0
        index = int(np.clip(beat, 0, len(curve) - 1))
        return float(curve[index])

    def _reward_components(self, action: np.ndarray
                           ) -> tuple[dict[str, float], dict[str, float]]:
        base, detail = super()._reward_components(action[:8])
        beat_a = self.cue_a + self.step_count
        beat_b = self.cue_b + self.step_count
        perc_a = self._percussive(self.track_a, beat_a)
        perc_b = self._percussive(self.track_b, beat_b)
        gain_a = float(np.cos(self.fader * np.pi / 2.0))
        gain_b = float(np.sin(self.fader * np.pi / 2.0))
        drum_a = gain_a * self.drum_a * perc_a
        drum_b = gain_b * self.drum_b * perc_b
        blend = float(detail["blend"])
        downbeat = bool(detail["downbeat"])
        phrase_boundary = bool(detail["phrase_boundary"])

        drum_clash = _overlap(drum_a, drum_b) * blend
        total_drum = drum_a + drum_b
        dominance = abs(drum_a - drum_b) / (total_drum + 1e-6)
        drum_craft = 0.10 * dominance * blend * (1.0 - drum_clash)

        owner = 1 if drum_a > drum_b else -1
        drum_swap = 0.0
        if blend > 0.2 and self._drum_cooldown == 0:
            if self._drum_owner != 0 and owner != self._drum_owner:
                drum_swap = 0.15 * (1.5 if downbeat else 1.0)
                self._drum_cooldown = DRUM_SWAP_COOLDOWN
            self._drum_owner = owner

        drum_delta = max(abs(self.drum_a - self._prev_drum_a),
                         abs(self.drum_b - self._prev_drum_b))
        drum_cut = 0.0
        if drum_delta > 0.5:
            drum_cut = -0.10 if downbeat else -0.25

        fresh_a = float(np.clip(1.0 - self._echo_beats_a / ECHO_BUDGET,
                                0.0, 1.0))
        fresh_b = float(np.clip(1.0 - self._echo_beats_b / ECHO_BUDGET,
                                0.0, 1.0))
        echo_use = 0.08 * (self.echo_a * fresh_a + self.echo_b * fresh_b)
        if phrase_boundary:
            echo_use *= 1.5
        echo_mud = -0.08 * min(self.echo_a, self.echo_b) * blend
        echo_hold = -0.10 * max(
            self.echo_a if fresh_a <= 1e-6 else 0.0,
            self.echo_b if fresh_b <= 1e-6 else 0.0)
        if self.echo_a > 0.05:
            self._echo_beats_a += 1
        else:
            self._echo_beats_a = max(0, self._echo_beats_a - 2)
        if self.echo_b > 0.05:
            self._echo_beats_b += 1
        else:
            self._echo_beats_b = max(0, self._echo_beats_b - 2)

        self._drum_clashes.append(drum_clash)
        components = {
            **base,
            "drum_craft": drum_craft,
            "drum_swap": drum_swap,
            "drum_clash": -0.10 * drum_clash,
            "drum_cut": drum_cut,
            "echo_use": echo_use,
            "echo_mud": echo_mud,
            "echo_hold": echo_hold,
        }
        detail = {
            **detail,
            "drum_a": self.drum_a,
            "drum_b": self.drum_b,
            "echo_a": self.echo_a,
            "echo_b": self.echo_b,
            "drum_a_eff": drum_a,
            "drum_b_eff": drum_b,
            "drum_clash": drum_clash,
            "drum_swap": drum_swap,
            "echo_use": echo_use,
            "echo_hold": echo_hold,
        }
        return components, detail

    def _end_bonus(self) -> dict[str, float]:
        bonus = super()._end_bonus()
        if self._drum_clashes:
            clash = float(np.mean(self._drum_clashes))
            blend_fraction = (float(np.mean(np.asarray(self._blends) > 0.15))
                              if self._blends else 0.0)
            bonus["drum_quality"] = (0.6 * blend_fraction
                                     * (1.0 - float(np.clip(clash, 0, 1))))
        return bonus

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        beat_a = self.cue_a + self.step_count
        beat_b = self.cue_b + self.step_count
        perc_a = self._percussive(self.track_a, beat_a)
        perc_b = self._percussive(self.track_b, beat_b)
        gain_a = float(np.cos(self.fader * np.pi / 2.0))
        gain_b = float(np.sin(self.fader * np.pi / 2.0))
        drum_a = gain_a * self.drum_a * perc_a
        drum_b = gain_b * self.drum_b * perc_b
        extra = np.array([
            self.drum_a, self.drum_b,
            drum_a, drum_b,
            _overlap(drum_a, drum_b),
            self.echo_a, self.echo_b,
        ], dtype=np.float32)
        return np.clip(np.concatenate([base, extra]), 0.0, 1.0)

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info.update({
            "drum_a": self.drum_a,
            "drum_b": self.drum_b,
            "echo_a": self.echo_a,
            "echo_b": self.echo_b,
        })
        return info

    def render(self):
        super().render()
        info = self._info()
        print(f"  drums A {info['drum_a']:.2f} B {info['drum_b']:.2f}  "
              f"echo A {info['echo_a']:.2f} B {info['echo_b']:.2f}")


gym.register(
    id="MoodMix/DJSet-v3",
    entry_point="moodmix_env.env_v3:MoodMixV3Env",
    max_episode_steps=None,
    kwargs={"max_steps": 32},
)
