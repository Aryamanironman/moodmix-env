"""MoodMix DJ-set environment.

The agent performs one transition, one beat per step.  It works a small DJ
surface: crossfader, three-band EQ per deck, tempo sync/nudge, a filter switch
and four drum pads.  Observations are pre-analysed features (no audio decoding
during training); render() and the demo script paint the control surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MOODS = ("energetic", "happy", "relaxed", "emotional", "focused", "party")
KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DRUM_NAMES = ("none", "kick", "snare", "hat", "clap")

# Linear gains for the three EQ positions: cut, neutral, boost.
EQ_GAINS = np.array([10 ** (-18 / 20), 1.0, 10 ** (6 / 20)], dtype=np.float32)

# Two reward designs.  "curve" scores agreement with an authored smoothstep
# fader envelope; "expressive" pays for the craft moves (bass swap, filter use,
# pad work, phrase timing) and treats the envelope as weak shaping.
REWARD_PROFILES = ("expressive", "curve")

# Engaging sync is a latch: one discrete action, then a flat tempo/alignment
# benefit for the rest of the transition.  Under PPO the engage step therefore
# carries almost none of the credit for that plateau, and the sync head can
# random-walk to "off" entirely.  A one-time bonus at the exact step the latch
# closes gives the action an immediate advantage.  It is paid once per
# episode, only on the off->on transition, and is a constructor argument so an
# ablation can switch it off.
SYNC_ENGAGE_BONUS = 0.5

CONTROL_NAMES = (
    "crossfader", "eq_a_low", "eq_a_mid", "eq_a_high",
    "eq_b_low", "eq_b_mid", "eq_b_high", "sync", "drum", "filter",
)

COMPATIBLE_MOODS = {
    "energetic": ("party", "focused"),
    "happy": ("focused", "party"),
    "relaxed": ("emotional", "focused"),
    "emotional": ("relaxed", "focused"),
    "focused": ("happy", "relaxed", "energetic"),
    "party": ("energetic", "happy"),
}


def _default_data_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / "data" / "demo", Path.cwd() / "data" / "demo"):
        if (candidate / "manifest.json").exists():
            return candidate
    return here.parents[2] / "data" / "demo"


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _wrap_phase(value: float) -> float:
    return float(value % 1.0)


def _energy_arc(energy: np.ndarray, window: int = 4) -> tuple[float, int, float]:
    """Describe the shape of an output-energy curve.

    Returns `(span, peaks, arc)`.  Standard deviation is the obvious measure of
    "did the energy move" and the wrong one: a policy can maximise it by raking
    the crossfader, which is a stutter, not an arc.  A build-and-release has one
    peak, a rake has several, so the peaks decide whether the span counts.
    """
    curve = np.asarray(energy, dtype=np.float64)
    usable = len(curve) - len(curve) % window
    if usable >= window:
        curve = curve[:usable].reshape(-1, window).mean(axis=1)
    if len(curve) < 3:
        return 0.0, 0, 1.0
    span = float(np.percentile(curve, 90) - np.percentile(curve, 10))
    span = span / (float(np.max(curve)) + 1e-6)
    peaks = sum(1 for i in range(1, len(curve) - 1)
                if curve[i] > curve[i - 1] and curve[i] > curve[i + 1])
    arc = 1.0 if peaks <= 1 else max(0.0, 1.0 - 0.3 * (peaks - 1))
    return span, peaks, arc


def _key_index(track: dict[str, Any]) -> int:
    try:
        return KEY_NAMES.index(track["key"])
    except (ValueError, KeyError):
        return 0


def _key_distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    d = abs(_key_index(a) - _key_index(b))
    return min(d, 12 - d)


class MoodMixEnv(gym.Env):
    """A live DJ transition environment with a physical-style control surface."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        data_dir: str | Path | None = None,
        mood: str | None = None,
        max_steps: int = 64,
        step_beats: int = 1,
        render_mode: str | None = None,
        reward_profile: str = "expressive",
        sync_bonus: float = SYNC_ENGAGE_BONUS,
        binary_sync: bool = False,
        term_scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if mood is not None and mood not in MOODS:
            raise ValueError(f"mood must be one of {MOODS}, got {mood!r}")
        if step_beats != 1:
            raise ValueError("the demo surface currently steps one beat at a time")
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(
                f"reward_profile must be one of {REWARD_PROFILES}, "
                f"got {reward_profile!r}"
            )
        self.reward_profile = reward_profile
        if sync_bonus < 0.0:
            raise ValueError("sync_bonus cannot be negative")
        self.sync_bonus = float(sync_bonus)
        # Binary sync is the latch the control surface actually is: press once
        # and deck B stays beatmatched.  Four sync states (none/sync/nudge/
        # release) give the policy two ways to lose the lock for no reward.
        self.binary_sync = bool(binary_sync)
        # Per-term multipliers for ablations (1.0 = unchanged).  An unknown key
        # is harmless: it has no component to scale.
        self.term_scales = {name: float(value)
                            for name, value in (term_scales or {}).items()}
        # Style memory lives across episodes so a run can score novelty: a set
        # that plays the same transition every time is not a performance.
        self._style_memory: list[np.ndarray] = []

        self.data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run scripts/make_demo_library.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.tracks: list[dict[str, Any]] = manifest["tracks"]
        if len(self.tracks) < 2:
            raise ValueError(
                f"{manifest_path} has {len(self.tracks)} track(s); a transition "
                "needs at least 2. Add more audio and re-run "
                "scripts/analyze_library.py."
            )
        for track in self.tracks:
            track["band_curve"] = np.asarray(track["band_curve"], dtype=np.float32)

        self.requested_mood = mood
        self.max_steps = int(max_steps)
        self.step_beats = int(step_beats)
        self.render_mode = render_mode

        # 21 crossfader positions x 3 EQ positions per band per deck,
        # 4 sync states, 5 drum pads, 3 filter states.
        sync_actions = 2 if self.binary_sync else 4
        self.action_space = spaces.MultiDiscrete(
            [21, 3, 3, 3, 3, 3, 3, sync_actions, 5, 3])
        self.observation_space = spaces.Dict({
            "deck_a": spaces.Box(0.0, 1.0, shape=(21,), dtype=np.float32),
            "deck_b": spaces.Box(0.0, 1.0, shape=(21,), dtype=np.float32),
            "mixer": spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32),
            "context": spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32),
        })

        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self) -> None:
        self.mood: str = MOODS[0]
        self.track_a: dict[str, Any] = self.tracks[0]
        self.track_b: dict[str, Any] = self.tracks[0]
        self.phase_a = 0.0
        self.phase_b = 0.0
        self.pos_a = 0.0
        self.pos_b = 0.0
        self.bpm_eff_b = 120.0
        self.crossfader = 0.0
        self.prev_crossfader = 0.0
        self.eq_a = [1, 1, 1]
        self.eq_b = [1, 1, 1]
        self.sync = False
        self._sync_engaged = False
        self._sync_bonus_paid = False
        self.filter = 0
        self.last_drum = 0
        self.step_count = 0
        self.last_reward = 0.0
        self.last_components: dict[str, float] = {}
        self.episode_components: dict[str, float] = {}
        self.events: list[str] = []
        self.trace: list[dict[str, float]] = []
        self._prev_eq_a = [1, 1, 1]
        self._prev_eq_b = [1, 1, 1]
        self._prev_filter = 0
        self._filter_age = 99
        self._filter_beats = 0
        self._pad_history: list[int] = []
        self._fader_signs: list[float] = []
        self._route_signs: list[float] = []
        self._route_turns = 0
        self._cut_downbeat = False
        self._swap_cooldown = 0
        self._last_gesture_step = -99
        self._gesture_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._init_state()

        options = options or {}
        mood = options.get("mood", self.requested_mood)
        self.mood = mood if mood in MOODS else str(self.np_random.choice(MOODS))

        same = [t for t in self.tracks if t["mood"] == self.mood]
        self.track_a = same[int(self.np_random.integers(len(same)))] if same else \
            self.tracks[int(self.np_random.integers(len(self.tracks)))]

        # Pin deck A before choosing a partner.  Pinning used to happen after
        # the partner was sampled, so the choice was made against a random
        # track and could land on the pinned one - a set hands its last deck B
        # in as the next deck A and hit that every time.
        by_id = {track["id"]: track for track in self.tracks}
        pinned_a = options.get("track_a")
        if pinned_a is not None:
            if pinned_a not in by_id:
                raise ValueError(f"unknown track id {pinned_a!r}")
            self.track_a = by_id[pinned_a]

        pool = [t for t in self.tracks if t["mood"] in COMPATIBLE_MOODS[self.mood]]
        pool = [t for t in pool if t["id"] != self.track_a["id"]] or pool
        if not pool:
            pool = [t for t in self.tracks if t["id"] != self.track_a["id"]]
        # A DJ picks a partner that is already close in tempo: stretching a
        # track by 40% is audible even with a good time-stretcher.  Tempo
        # closeness beats mood compatibility, but a compatible mood wins among
        # equally close candidates.
        bpm_a = float(self.track_a["bpm"])
        candidates = [t for t in self.tracks if t["id"] != self.track_a["id"]]

        def _ratio(track: dict[str, Any]) -> float:
            return float(track["bpm"]) / bpm_a

        def _fits_mood(track: dict[str, Any]) -> bool:
            return (track["mood"] == self.mood
                    or track["mood"] in COMPATIBLE_MOODS[self.mood])

        close = [t for t in candidates if 0.92 <= _ratio(t) <= 1.08]
        preferred = [t for t in close if _fits_mood(t)] or close
        if preferred:
            pool = preferred
        elif pool:
            nearest = min(abs(_ratio(t) - 1.0) for t in pool)
            pool = [t for t in pool if abs(_ratio(t) - 1.0) <= nearest + 0.03]
        else:
            pool = candidates
        # A set hands deck B over as the next deck A; without this the same
        # pair can come straight back reversed, which is not a set, it is a
        # loop.
        avoid = {str(track) for track in (options.get("avoid_tracks") or ())}
        if avoid:
            fresh = [t for t in pool if t["id"] not in avoid]
            if not fresh:
                # Nothing left in this mood's pool: widen to any track the set
                # has not just played, rather than ping-ponging between two.
                fresh = [t for t in self.tracks
                         if t["id"] not in avoid and t["id"] != self.track_a["id"]]
            if fresh:
                pool = fresh
        self.track_b = pool[int(self.np_random.integers(len(pool)))]

        # Pinning deck B replays a specific transition (for imitation from a
        # recording) instead of sampling one.
        pinned_b = options.get("track_b")
        if pinned_b is not None:
            if pinned_b not in by_id:
                raise ValueError(f"unknown track id {pinned_b!r}")
            self.track_b = by_id[pinned_b]
        if self.track_a["id"] == self.track_b["id"]:
            raise ValueError("a transition needs two different tracks")

        # Start deck A in its drop, deck B at the top of the record.
        beats_a = len(self.track_a["band_curve"])
        self.pos_a = float(max(0, min(32, beats_a - self.max_steps)))
        self.pos_b = 0.0
        self.phase_a = 0.0
        self.phase_b = 0.25  # deliberately out of phase: the agent must sync
        self.bpm_eff_b = float(self.track_b["bpm"])

        self.events = [f"loaded {self.track_a['id']} -> {self.track_b['id']}",
                       "deck B is 1/4 beat out of phase"]
        obs = self._observation()
        info = self._info()
        return obs, info

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape != (10,):
            raise ValueError(f"expected 10 controls, got {action.shape}")
        action = np.clip(action, 0, self.action_space.nvec - 1)

        (fader_bin, eq_al, eq_am, eq_ah, eq_bl, eq_bm, eq_bh,
         sync_cmd, drum, filt) = (int(v) for v in action)

        # Remember the previous control positions so phrase/timing rewards can
        # see the move that was made, not just the state it landed in.
        self._prev_eq_a = list(self.eq_a)
        self._prev_eq_b = list(self.eq_b)
        self._prev_filter = self.filter

        self.prev_crossfader = self.crossfader
        self.crossfader = fader_bin / 20.0
        self.eq_a = [eq_al, eq_am, eq_ah]
        self.eq_b = [eq_bl, eq_bm, eq_bh]
        self.filter = filt
        self.last_drum = drum

        # The engage bonus is an edge, not a level: it pays on the step the
        # latch closes and never again, however long sync is held.
        was_synced = self.sync
        if sync_cmd == 1 and not self.sync:
            self.sync = True
            self.phase_b = self.phase_a
            self.events.append("SYNC: deck B beatmatched")
        elif self.binary_sync:
            # Binary latch: 0 is a no-op and there is no release.  Releasing
            # mid-transition only loses the tempo/alignment payoff.
            pass
        elif sync_cmd == 2:
            diff = ((self.phase_a - self.phase_b + 0.5) % 1.0) - 0.5
            self.phase_b = _wrap_phase(self.phase_b + 0.5 * diff)
            self.sync = False
            self.events.append("NUDGE: phase corrected")
        elif sync_cmd == 3:
            self.sync = False
        # Pay the bonus on the first off->on edge only.  Paying every edge
        # would let a policy farm it by releasing and re-engaging each beat.
        edge = bool(self.sync and not was_synced)
        self._sync_engaged = bool(edge and not self._sync_bonus_paid)
        if self._sync_engaged:
            self._sync_bonus_paid = True

        if filt == 1:
            self.events.append("FILTER: low-pass on deck B")
        elif filt == 2:
            self.events.append("FILTER: high-pass on deck B")

        components, detail = self._reward_components(action)
        if self.term_scales:
            components = {name: value * self.term_scales.get(name, 1.0)
                          for name, value in components.items()}
        reward = float(sum(components.values()))
        self.last_components = components
        self.trace.append(detail)
        for name, value in components.items():
            self.episode_components[name] = self.episode_components.get(name, 0.0) + value
        self.last_reward = reward

        fader_move = self.crossfader - self.prev_crossfader
        if abs(fader_move) > 0.30:
            self._cut_downbeat = int(round(self.pos_a)) % 4 == 0
        self._pad_history.append(drum)
        self._pad_history = self._pad_history[-8:]
        if self.filter != self._prev_filter:
            self._filter_age = 0
        else:
            self._filter_age = min(self._filter_age + 1, 999)
        # Cumulative, never reset by switching the filter off: toggling it to
        # look "fresh" again must not buy a new budget.
        self._filter_beats += 1 if self.filter else 0
        self._swap_cooldown = max(0, self._swap_cooldown - 1)

        # Advance the transport one beat.
        self.step_count += 1
        ratio = 1.0 if self.sync else self.bpm_eff_b / float(self.track_a["bpm"])
        self.phase_a = _wrap_phase(self.phase_a + 1.0)
        self.phase_b = _wrap_phase(self.phase_b + ratio)
        self.pos_a += 1.0
        self.pos_b += ratio

        terminated = self.step_count >= self.max_steps
        if terminated:
            bonus = self._end_bonus()
            if self.term_scales:
                bonus = {name: value * self.term_scales.get(name, 1.0)
                         for name, value in bonus.items()}
            components.update(bonus)
            reward += float(sum(bonus.values()))
            for name, value in bonus.items():
                self.episode_components[name] = (
                    self.episode_components.get(name, 0.0) + value
                )
            self.last_components = components

        obs = self._observation()
        info = self._info()
        info["reward_components"] = components
        return obs, reward, bool(terminated), False, info

    # ----------------------------------------------------------------- reward
    def _reward_components(self, action: np.ndarray
                           ) -> tuple[dict[str, float], dict[str, float]]:
        """Score one beat with the selected reward design."""
        if self.reward_profile == "curve":
            return self._reward_curve(int(action[8]))
        return self._reward_expressive(action)

    def _reward_curve(self, drum: int) -> tuple[dict[str, float], dict[str, float]]:
        """Legacy design: agreement with the authored smoothstep envelope."""
        bands_a = self._bands(self.track_a, self.pos_a)
        bands_b = self._bands(self.track_b, self.pos_b)
        eq_a = np.asarray([EQ_GAINS[i] for i in self.eq_a], dtype=np.float32)
        eq_b = np.asarray([EQ_GAINS[i] for i in self.eq_b], dtype=np.float32)

        gain_a = float(np.cos(self.crossfader * np.pi / 2))
        gain_b = float(np.sin(self.crossfader * np.pi / 2))
        mix_a = gain_a * bands_a * eq_a
        mix_b = gain_b * bands_b * eq_b
        if self.filter == 1:       # low-pass: pull deck B's top end down
            mix_b[2] *= 0.15
        elif self.filter == 2:     # high-pass: pull deck B's bass away
            mix_b[0] *= 0.12

        blend = float(np.clip(min(gain_a, gain_b) / 0.7071, 0.0, 1.0))

        diff = abs(_wrap_phase(self.phase_a - self.phase_b))
        diff = min(diff, 1.0 - diff)
        align = float(np.clip(1.0 - diff / 0.25, 0.0, 1.0))

        bpm_diff = abs(self.bpm_eff_b - float(self.track_a["bpm"]))
        tempo = 1.0 if self.sync else float(np.clip(1.0 - bpm_diff / 12.0, 0.0, 1.0))

        key_dist = _key_distance(self.track_a, self.track_b)
        harmonic = 1.0 - key_dist / 6.0

        low_a, low_b = float(mix_a[0]), float(mix_b[0])
        mid_a, mid_b = float(mix_a[1]), float(mix_b[1])
        low_overlap = min(low_a, low_b) / (max(low_a, low_b) + 1e-6) if min(low_a, low_b) > 0.02 else 0.0
        mid_overlap = min(mid_a, mid_b) / (max(mid_a, mid_b) + 1e-6) if min(mid_a, mid_b) > 0.02 else 0.0
        spectral = 1.0 - (0.7 * low_overlap + 0.3 * mid_overlap)

        progress = self.step_count / max(1, self.max_steps)
        target = _smoothstep((progress - 0.12) / 0.68)
        shape = 1.0 - abs(self.crossfader - target)
        # Only pay for mixing quality while the fader is actually following
        # the intended transition curve.  Without this gate a policy can park
        # the fader mid-blend forever and farm beat/key rewards.
        window = float(np.clip(1.0 - abs(self.crossfader - target) / 0.18, 0.0, 1.0))
        music_gate = blend * window

        mood_match = self._mood_match()
        beat_dist = min(self.phase_a, 1.0 - self.phase_a)
        drum_reward = 0.0
        if drum != 0:
            drum_reward = 0.06 if beat_dist < 0.13 else 0.01
            if drum == self.last_drum and self.last_drum != 0:
                drum_reward -= 0.01
        jump = abs(self.crossfader - self.prev_crossfader)
        smooth = -0.05 if jump > 0.30 else 0.0
        extreme = 0.0
        if self.eq_a[0] == 2 and self.eq_b[0] == 2:
            extreme -= 0.04
        if all(v == 0 for v in self.eq_a) or all(v == 0 for v in self.eq_b):
            extreme -= 0.03

        components = {
            "beat_alignment": 0.22 * align * music_gate,
            "tempo": 0.13 * tempo * music_gate,
            "harmonic": 0.13 * harmonic * music_gate,
            "spectral": 0.15 * spectral * music_gate,
            "transition_shape": 0.25 * shape,
            "mood": 0.05 * mood_match,
            "drum": drum_reward,
            "smoothness": smooth + extreme,
        }
        detail = {
            "blend": blend, "align": align, "tempo": tempo,
            "harmonic": harmonic, "spectral": spectral, "shape": shape,
            "window": window, "target": target,
            "jump": jump, "crossfader": self.crossfader,
            "eq_a": tuple(self.eq_a), "eq_b": tuple(self.eq_b),
            "filter": float(self.filter), "drum": float(drum),
            "sync": float(self.sync),
            "pos_a": self.pos_a, "pos_b": self.pos_b,
            "bpm_a": float(self.track_a["bpm"]),
            "bpm_b_eff": float(self.bpm_eff_b),
        }
        return components, detail

    def _reward_expressive(self, action: np.ndarray
                           ) -> tuple[dict[str, float], dict[str, float]]:
        """Pay for the moves a DJ makes, not for following a curve.

        Every term here is visible in the features the environment already
        computes, and each one maps to a real gesture:

        * `bass_swap` - during the blend only one deck owns the low end, and
          the swap is worth more on a phrase downbeat.
        * `filter_craft` - the filter is engaged where it does something (a
          high-pass on deck B while B still carries the bass, a low-pass while
          B is bright), plus a small credit for touching it at all.
        * `pad_craft` - hits on the beat, pad variety, and a fill into the next
          eight-bar phrase.
        * `phrase` - cuts and filter moves land on downbeats, with a
          deliberate-off-phrase allowance instead of a hard rule.
        * `sync_engage` - a one-time bonus on the step the sync latch closes,
          because the latch's payoff (tempo and alignment) is a flat plateau
          that the engage action would otherwise not be credited for.
        * `busy_hands` - the fifth gesture in a transition is charged, so a
          performance of two good moves beats one of sixteen nervous ones.
        * `jitter` - the only smoothness term left, and it penalises wobble
          (reversing the fader inside four beats), never speed.
        """
        bands_a = self._bands(self.track_a, self.pos_a)
        bands_b = self._bands(self.track_b, self.pos_b)
        eq_a = np.asarray([EQ_GAINS[i] for i in self.eq_a], dtype=np.float32)
        eq_b = np.asarray([EQ_GAINS[i] for i in self.eq_b], dtype=np.float32)

        gain_a = float(np.cos(self.crossfader * np.pi / 2))
        gain_b = float(np.sin(self.crossfader * np.pi / 2))
        mix_a = gain_a * bands_a * eq_a
        mix_b = gain_b * bands_b * eq_b
        if self.filter == 1:       # low-pass: pull deck B's top end down
            mix_b[2] *= 0.15
        elif self.filter == 2:     # high-pass: pull deck B's bass away
            mix_b[0] *= 0.12

        blend = float(np.clip(min(gain_a, gain_b) / 0.7071, 0.0, 1.0))
        diff = abs(_wrap_phase(self.phase_a - self.phase_b))
        diff = min(diff, 1.0 - diff)
        align = float(np.clip(1.0 - diff / 0.25, 0.0, 1.0))
        bpm_diff = abs(self.bpm_eff_b - float(self.track_a["bpm"]))
        tempo = 1.0 if self.sync else float(np.clip(1.0 - bpm_diff / 12.0, 0.0, 1.0))
        harmonic = 1.0 - _key_distance(self.track_a, self.track_b) / 6.0

        # Separation the tracks already had is not a decision, so both the EQ
        # and the bass-swap terms score the *improvement* over the same mix
        # with the EQ bypassed.  A policy that parks the fader and touches
        # nothing earns nothing for these.
        raw_a = gain_a * bands_a
        raw_b = gain_b * bands_b

        def _overlap(x: np.ndarray, y: np.ndarray) -> float:
            lo, hi = min(x, y), max(x, y)
            return float(lo / (hi + 1e-6)) if lo > 0.02 else 0.0

        def _separation(x: np.ndarray, y: np.ndarray) -> float:
            """How far apart two decks are across all three bands."""
            return 1.0 - (0.5 * _overlap(float(x[0]), float(y[0]))
                          + 0.3 * _overlap(float(x[1]), float(y[1]))
                          + 0.2 * _overlap(float(x[2]), float(y[2])))

        low_a, low_b = float(mix_a[0]), float(mix_b[0])
        spectral = _separation(mix_a, mix_b)
        spectral_raw = _separation(raw_a, raw_b)
        spectral_gain = float(np.clip(spectral - spectral_raw, 0.0, 1.0))

        beat = int(round(self.pos_a)) % 4 == 0
        downbeat = 1.5 if beat else 1.0

        # Bass swap: during the blend one deck owns the low end, and the credit
        # is for the exclusivity the EQ created, not for what the tracks sound
        # like on their own.  A swap is an *event*: it pays only while the low
        # EQ is being held, and only once per two bars.  Without that a policy
        # learns to flick the low band on and off every beat and farms the term.
        low_moved = (self.eq_a[0] != self._prev_eq_a[0]
                     or self.eq_b[0] != self._prev_eq_b[0])
        swap = 0.0
        if (blend > 0.2 and max(low_a, low_b) > 0.02
                and not low_moved and self._swap_cooldown == 0):
            swap_eq = 1.0 - _overlap(low_a, low_b)
            swap_raw = 1.0 - _overlap(float(raw_a[0]), float(raw_b[0]))
            swap = float(np.clip(swap_eq - swap_raw, 0.0, 1.0))
            if swap > 0.05:
                self._swap_cooldown = 8

        # Filter craft: engage it where it removes something that is there, and
        # only while it is fresh.  The budget is ten filter beats per
        # transition and does not refill, so neither leaving the filter on all
        # episode nor flicking it off and on every beat keeps paying; the move
        # that earns is one sweep, in a place it does something.
        filt_score = 0.0
        if self.filter and float(gain_b) > 0.02:
            if self.filter == 2:
                value = float(np.clip(
                    float(bands_b[0]) / (float(bands_a[0]) + 1e-3), 0.0, 1.0))
            else:
                value = float(np.clip(
                    float(bands_b[2]) / (float(bands_a[2]) + 1e-3), 0.0, 1.0))
            fresh = float(np.clip(1.0 - self._filter_beats / 10.0, 0.0, 1.0))
            filt_score = value * fresh
        if self.filter != self._prev_filter and self._filter_age >= 4:
            reach = 0.35 if self.filter else 0.0
            filt_score = float(np.clip(filt_score + reach, 0.0, 1.0))

        # Pad craft: restraint and placement.  `phase_a` wraps to 0 every beat,
        # so "on the beat" is true of any hit by construction - paying for that
        # pays for spamming.  Credit the downbeat, the fill into the next
        # eight-bar phrase, and variety instead.
        drum = int(action[8])
        pad_score = 0.0
        if drum:
            if int(round(self.pos_a)) % 4 == 0:
                pad_score += 0.4
            if int(round(self.pos_a)) % 32 == 31 and drum in (2, 3, 4):
                pad_score += 0.3
            # Variety compares against the pads actually played, not against
            # the zeros the history records on every silent beat.  With raw
            # history a repeated pad always looked "new", because the two
            # previous entries were 0,0.
            recent = [pad for pad in self._pad_history[-4:] if pad]
            if drum not in recent:
                pad_score += 0.3
            pad_score = float(np.clip(pad_score, 0.0, 1.0))

        # Phrase craft: a cut or a filter move pays when it lands on a downbeat,
        # pays a third as much off it, and pays nothing if another gesture came
        # in the last four beats.  The cool-down is what stops "gesture every
        # beat" from being the winning strategy; the low band is scored by
        # bass_swap and eq_craft instead, so it is not a gesture here.
        fader_delta = self.crossfader - self.prev_crossfader
        gestures = 0.0
        if abs(fader_delta) > 0.30:
            gestures += 1.0
        if self.filter != self._prev_filter:
            gestures += 1.0
        # Four gestures is a performance; the fifth is fidgeting, and pays a
        # small penalty so "do something every bar" cannot out-earn two moves
        # in the right places.
        ready = self.step_count - self._last_gesture_step >= 4
        phrase = 0.0
        busy = 0.0
        if gestures and ready:
            self._last_gesture_step = self.step_count
            self._gesture_count += 1
            if self._gesture_count <= 4:
                phrase = 1.0 if beat else 0.35
            else:
                busy = -0.34

        # Jitter, not speed: reversing the fader twice in four beats is the
        # robot look; a hard cut is a legal, deliberate move.
        if abs(fader_delta) > 1e-9:
            self._fader_signs.append(float(np.sign(fader_delta)))
            self._fader_signs = self._fader_signs[-4:]
        # A rendered comparison settled this: the policy that scored well still
        # reversed the fader 13 times in 64 beats (the hand-written DJ: zero),
        # and that pumping is what sounds unsmooth, whatever the reward says.
        # Speed stays free; repeated direction changes do not.
        flips = sum(1 for a, b in zip(self._fader_signs[:-1], self._fader_signs[1:])
                    if a * b < 0)
        jitter = -0.12 * (flips - 1) if flips >= 2 else 0.0

        # Route: a transition crosses once, or cuts once.  A cut after a blend
        # is one reversal and costs nothing; raking the fader across the range
        # eight times is fifteen moves and is charged as it happens.  The same
        # charge at the end of the episode was capped out and ignored, because
        # a terminal penalty 64 steps away barely reaches the policy's credit.
        route = 0.0
        if abs(fader_delta) >= 0.15:
            move_sign = float(np.sign(fader_delta))
            if self._route_signs and move_sign * self._route_signs[-1] < 0:
                self._route_turns += 1
                if self._route_turns > 1:
                    route = -0.7
            self._route_signs.append(move_sign)

        extreme = 0.0
        if all(v == 0 for v in self.eq_a) or all(v == 0 for v in self.eq_b):
            extreme -= 0.03

        components = {
            "beat_alignment": 0.02 * align,
            "tempo": 0.02 * tempo,
            # One-time credit for closing the latch, so the engage action is
            # not credited only through the flat plateau that follows it.
            "sync_engage": self.sync_bonus * (1.0 if self._sync_engaged else 0.0),
            "eq_craft": 0.14 * spectral_gain,
            "bass_swap": 0.20 * swap * downbeat,
            "filter_craft": 0.10 * filt_score,
            "pad_craft": 0.10 * pad_score,
            "phrase": 0.12 * phrase,
            "busy_hands": 0.12 * busy,
            "jitter": jitter + extreme,
            "route": route,
        }
        energy = float(np.sum(mix_a) + np.sum(mix_b))
        shape = 1.0 - abs(self.crossfader - _smoothstep(
            (self.step_count / max(1, self.max_steps) - 0.12) / 0.68))
        detail = {
            "blend": blend, "align": align, "tempo": tempo,
            "harmonic": harmonic, "spectral": spectral, "shape": shape,
            "jump": abs(fader_delta),
            "crossfader": self.crossfader,
            "eq_a": tuple(self.eq_a), "eq_b": tuple(self.eq_b),
            "filter": float(self.filter), "drum": float(drum),
            "sync": float(self.sync),
            "sync_engage": float(self._sync_engaged),
            "pos_a": self.pos_a, "pos_b": self.pos_b,
            "bpm_a": float(self.track_a["bpm"]),
            "bpm_b_eff": float(self.bpm_eff_b),
            "energy": energy, "bass_swap": swap, "filter_craft": filt_score,
            "pad_craft": pad_score, "phrase": phrase,
            "busy_hands": busy, "downbeat": float(beat),
        }
        return components, detail

    def _style_profile(self) -> np.ndarray:
        """Coarse summary of how this transition was played."""
        if not self.trace:
            return np.zeros(15, dtype=np.float64)
        faders = np.asarray([t["crossfader"] for t in self.trace], dtype=np.float64)
        quarters = [float(faders[i * len(faders) // 4:(i + 1) * len(faders) // 4].mean())
                    for i in range(4)]
        eqs = np.asarray([[t["eq_a"][0] / 2.0, t["eq_b"][0] / 2.0,
                           t["eq_a"][1] / 2.0, t["eq_b"][1] / 2.0]
                          for t in self.trace], dtype=np.float64).mean(axis=0)
        pads = np.bincount([int(t["drum"]) for t in self.trace], minlength=5)[:5]
        pads = pads / max(1, len(self.trace))
        filt = float(np.mean([1.0 if t["filter"] > 0 else 0.0 for t in self.trace]))
        return np.concatenate([quarters, eqs, pads, [filt]])

    def _novelty(self) -> float:
        """How unlike the transitions this env has already played this one is."""
        vector = self._style_profile()
        if not np.any(vector):
            return 0.0
        if not self._style_memory:
            self._style_memory.append(vector)
            return 1.0
        sims = [float(np.dot(vector, old) /
                      (np.linalg.norm(vector) * np.linalg.norm(old) + 1e-9))
                for old in self._style_memory]
        self._style_memory.append(vector)
        self._style_memory = self._style_memory[-32:]
        return float(np.clip(1.0 - max(sims), 0.0, 1.0))

    def _end_bonus(self) -> dict[str, float]:
        """Score the transition as a whole, not just the final frame.

        A razor cut has no frames in the blend zone.  The old fallback read
        that as "shape 0, every frame a jump", so a clean cut scored the same
        as a broken one.  Shape and jump fraction are measured from the trace
        unconditionally now, so a cut is judged on the transition it made.
        """
        gain_b = float(np.sin(self.crossfader * np.pi / 2))
        frames = self.trace
        if not frames:
            return {}
        shape = float(np.mean([t["shape"] for t in frames]))
        jumps = float(np.mean([1.0 if t["jump"] > 0.30 else 0.0 for t in frames]))
        music = float(np.mean([
            0.35 * t["align"] + 0.25 * t["tempo"]
            + 0.20 * t["harmonic"] + 0.20 * t["spectral"] for t in frames
        ]))

        if self.crossfader >= 0.9 and gain_b > 0.8:
            ending = 1.0
        elif self.crossfader > 0.6:
            ending = 0.5
        else:
            ending = 0.0

        if self.reward_profile == "curve":
            quality = 0.45 * music + 0.35 * shape + 0.20 * ending - 0.20 * jumps
            bonus = {"mix_quality": 12.0 * float(np.clip(quality, 0.0, 1.0))}
        else:
            # Shape of the energy curve, not its variance.  Standard deviation
            # is maximised by pumping the fader, which is how a policy that
            # raked A to B and back eight times still scored 0.97 here; a build
            # and release has one peak, a rake has several.  Counting peaks
            # separates the two, and the span keeps the arc worth having.
            energy = np.asarray([t["energy"] for t in frames], dtype=np.float64)
            span, _, arc = _energy_arc(energy)
            expression = float(np.clip(span / 0.25, 0.0, 1.0)) * arc
            quality = (0.30 * music + 0.20 * shape + 0.25 * ending
                       + 0.25 * expression - 0.15 * jumps)
            bonus = {"mix_quality": 10.0 * float(np.clip(quality, 0.0, 1.0))}
            if (not any(t["blend"] > 0.12 for t in frames)
                    and ending >= 0.5 and self._cut_downbeat):
                bonus["cut_credit"] = 2.0
            novelty = self._novelty()
            if novelty > 0.0:
                bonus["novelty"] = 0.6 * novelty

        if self.reward_profile == "curve":
            if ending < 1.0:
                bonus["end_transition"] = -0.5
        else:
            # Handing the set over is the one non-negotiable; a parked fader
            # is not a transition however tasteful the rest of it looks.
            bonus["end_transition"] = 2.0 if ending >= 1.0 else -2.0
        if self._mood_match() >= 0.6:
            bonus["end_mood"] = 0.1
        return bonus

    def _mood_match(self) -> float:
        if self.track_b["mood"] == self.mood:
            return 1.0
        if self.track_b["mood"] in COMPATIBLE_MOODS[self.mood]:
            return 0.6
        return 0.2

    # ------------------------------------------------------------ observation
    def _bands(self, track: dict[str, Any], pos: float) -> np.ndarray:
        curve = track["band_curve"]
        idx = int(np.clip(pos, 0, len(curve) - 1))
        return curve[idx].astype(np.float32)

    def _deck_vector(self, track: dict[str, Any], pos: float, phase: float,
                     eq_state: list[int], gain: float) -> np.ndarray:
        bands = self._bands(track, pos)
        bpm_norm = float(np.clip((track["bpm"] - 60) / 140, 0.0, 1.0))
        key = np.zeros(12, dtype=np.float32)
        key[_key_index(track)] = 1.0
        pos_norm = float(np.clip(pos / max(1, len(track["band_curve"])), 0.0, 1.0))
        bar_phase = (pos % 4) / 4.0
        vec = np.concatenate([
            bands,
            [(np.sin(2 * np.pi * phase) + 1) / 2, (np.cos(2 * np.pi * phase) + 1) / 2],
            [bar_phase, bpm_norm],
            key,
            [float(np.clip(gain, 0, 1)), pos_norm],
        ]).astype(np.float32)
        return np.clip(vec, 0.0, 1.0)

    def _observation(self) -> dict[str, np.ndarray]:
        gain_a = float(np.cos(self.crossfader * np.pi / 2))
        gain_b = float(np.sin(self.crossfader * np.pi / 2))
        deck_a = self._deck_vector(self.track_a, self.pos_a, self.phase_a, self.eq_a, gain_a)
        deck_b = self._deck_vector(self.track_b, self.pos_b, self.phase_b, self.eq_b, gain_b)

        mixer = np.array([
            self.crossfader,
            self.filter / 2.0,
            self.eq_a[0] / 2.0, self.eq_a[1] / 2.0, self.eq_a[2] / 2.0,
            self.eq_b[0] / 2.0, self.eq_b[1] / 2.0, self.eq_b[2] / 2.0,
            1.0 if self.sync else 0.0,
        ], dtype=np.float32)

        mood_onehot = np.zeros(6, dtype=np.float32)
        mood_onehot[MOODS.index(self.mood)] = 1.0
        progress = self.step_count / max(1, self.max_steps)
        bpm_diff = abs(self.bpm_eff_b - float(self.track_a["bpm"])) / 40.0
        key_dist = _key_distance(self.track_a, self.track_b) / 6.0
        blend = float(np.clip(min(gain_a, gain_b) / 0.7071, 0, 1))
        # The authored smoothstep is deliberately absent here.  It used to be
        # an observation, which told the policy what move to make instead of
        # letting it learn one; it survives only as the weak shape term inside
        # the expressive terminal score.
        context = np.concatenate([
            mood_onehot,
            [progress, 1.0 - progress,
             float(np.clip(bpm_diff, 0, 1)),
             float(np.clip(key_dist, 0, 1)),
             blend,
             self.last_drum / 4.0],
        ]).astype(np.float32)

        return {"deck_a": deck_a, "deck_b": deck_b,
                "mixer": np.clip(mixer, 0, 1),
                "context": np.clip(context, 0, 1)}

    # -------------------------------------------------------------------- info
    def _section(self, track: dict[str, Any], pos: float) -> str:
        bar = int(np.clip(pos // 4, 0, len(track["sections"]) - 1))
        return track["sections"][bar]

    def _info(self) -> dict[str, Any]:
        gain_a = float(np.cos(self.crossfader * np.pi / 2))
        gain_b = float(np.sin(self.crossfader * np.pi / 2))
        return {
            "mood": self.mood,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "bar": self.step_count // 4 + 1,
            "crossfader": self.crossfader,
            "gain_a": gain_a,
            "gain_b": gain_b,
            "eq_a": list(self.eq_a),
            "eq_b": list(self.eq_b),
            "filter": self.filter,
            "sync": self.sync,
            "drum": DRUM_NAMES[self.last_drum],
            "deck_a": {
                "id": self.track_a["id"], "mood": self.track_a["mood"],
                "bpm": self.track_a["bpm"], "key": self.track_a["key"],
                "section": self._section(self.track_a, self.pos_a),
                "position": round(self.pos_a, 2),
            },
            "deck_b": {
                "id": self.track_b["id"], "mood": self.track_b["mood"],
                "bpm": self.track_b["bpm"], "key": self.track_b["key"],
                "section": self._section(self.track_b, self.pos_b),
                "position": round(self.pos_b, 2),
            },
            "reward": self.last_reward,
            "reward_profile": self.reward_profile,
            "reward_components": self.last_components,
            "episode_components": dict(self.episode_components),
            "last_event": self.events[-1] if self.events else "",
        }

    def render(self):
        info = self._info()
        a, b = info["deck_a"], info["deck_b"]
        print(f"\n[{info['mood'].upper()}] bar {info['bar']:02d}  "
              f"{a['id']} ({a['section']})  ->  {b['id']} ({b['section']})")
        print(f"  A {a['bpm']} BPM {a['key']:>2}   fader "
              f"{'|' * int(info['crossfader'] * 20):<20}   "
              f"B {b['bpm']} BPM {b['key']:>2}")
        print(f"  EQ A {info['eq_a']}  EQ B {info['eq_b']}  "
              f"sync={info['sync']}  filter={info['filter']}  drum={info['drum']}")
        print(f"  reward {info['reward']:+.3f}  {info['reward_components']}")

    def close(self):
        pass


gym.register(
    id="MoodMix/DJSet-v0",
    entry_point="moodmix_env.env:MoodMixEnv",
    max_episode_steps=None,
    kwargs={"max_steps": 64},
)
