"""Tests for the expressive reward profile and the cut fix.

The two designs are checked separately: `curve` scores agreement with the
authored smoothstep envelope, `expressive` pays for craft moves.  The tests
that matter here are the ones that used to fail - a razor cut scoring zero, and
a parked fader collecting craft points for free.
"""

from __future__ import annotations

import numpy as np
import pytest

import numpy as np

from moodmix_env import MoodMixEnv
from moodmix_env.env import SYNC_ENGAGE_BONUS, _energy_arc

# A pinned pair from the committed demo crate, so these tests never depend on
# whatever else is in someone's music folder.
PAIR = {"track_a": "relaxed_drift", "track_b": "emotional_rain"}


def _action(fader, eq_a=(1, 1, 1), eq_b=(1, 1, 1), sync=0, drum=0, filt=0):
    return [fader, *eq_a, *eq_b, sync, drum, filt]


def _play(profile, script, seed=5, steps=64):
    env = MoodMixEnv(max_steps=steps, reward_profile=profile)
    obs, info = env.reset(seed=seed, options=PAIR)
    while True:
        obs, reward, terminated, truncated, info = env.step(script(info))
        if terminated or truncated:
            break
    return sum(env.episode_components.values()), dict(env.episode_components), env


def _target(info):
    u = min(max((info["step"] / info["max_steps"] - 0.12) / 0.68, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _follower(info):
    return _action(int(round(_target(info) * 20)),
                   sync=1 if info["step"] == 0 else 0)


def _cut(info):
    return _action(0 if info["step"] < 32 else 20,
                   sync=1 if info["step"] == 0 else 0)


def test_razor_cut_earns_mix_quality():
    """A cut has no blended frames; it must still score as a transition."""
    _, components, env = _play("curve", _cut)
    assert components["mix_quality"] > 5.0
    assert env.trace[-1]["crossfader"] == 1.0
    # ...and it must not be charged a jump on every frame.
    assert components["mix_quality"] > 9.0


def test_cut_is_not_structurally_worse_than_a_blend():
    cut, _, _ = _play("expressive", _cut)
    blend, _, _ = _play("expressive", _follower)
    assert cut > blend - 2.0, (cut, blend)


def test_one_beat_cut_is_not_penalised():
    """A cut is one deliberate move, not a wobble: speed costs nothing."""
    _, parts, _ = _play("expressive", _cut)
    assert parts["jitter"] == pytest.approx(0.0, abs=1e-9)
    assert parts["route"] == pytest.approx(0.0, abs=1e-9)


def test_offbeat_cut_is_not_penalised_for_speed():
    """The same full-scale cut, one beat off the downbeat, costs no more.

    Phrase timing may pay less off the beat, but the fader itself must not
    be charged for moving fast.
    """
    _, parts, _ = _play(
        "expressive",
        lambda info: _action(0 if info["step"] < 31 else 20,
                             sync=1 if info["step"] == 0 else 0),
    )
    assert parts["jitter"] == pytest.approx(0.0, abs=1e-9)
    assert parts["route"] == pytest.approx(0.0, abs=1e-9)


def test_switch_knobs_are_not_scored_on_the_fader_scale():
    """EQ is a 3-position switch; toggling it is not a full-scale jump."""
    def churn(info):
        value = 0 if info["step"] % 2 else 2
        return _action(int(round(_target(info) * 20)), eq_a=(value, 1, 1),
                       sync=1 if info["step"] == 0 else 0)

    _, parts, _ = _play("expressive", churn)
    assert parts["jitter"] == pytest.approx(0.0, abs=1e-9)
    assert parts["route"] == pytest.approx(0.0, abs=1e-9)


def test_bass_swap_pays_more_than_flat_eq():
    def swapped(info):
        fader = int(round(_target(info) * 20))
        sync = 1 if info["step"] == 0 else 0
        if info["step"] < 32:      # deck A owns the low end
            return _action(fader, eq_b=(0, 1, 1), sync=sync)
        return _action(fader, eq_a=(0, 1, 1), eq_b=(1, 1, 1), sync=sync)

    flat, _, _ = _play("expressive", _follower)
    swap, components, _ = _play("expressive", swapped)
    assert components["bass_swap"] > 0.0
    assert swap > flat + 1.0, (swap, flat)


def test_flat_eq_earns_no_eq_craft():
    """Separation the tracks already had is not a decision."""
    _, components, _ = _play("expressive", lambda info: _action(10, sync=1))
    assert components.get("eq_craft", 0.0) == pytest.approx(0.0, abs=1e-9)
    assert components.get("bass_swap", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_filter_craft_fires_when_deck_b_is_audible():
    def script(filt):
        def play(info):
            return _action(13, sync=1 if info["step"] == 0 else 0, filt=filt)
        return play

    _, with_filter, _ = _play("expressive", script(2))
    _, without, _ = _play("expressive", script(0))
    assert with_filter["filter_craft"] > 0.0
    assert without.get("filter_craft", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_parking_is_not_rewarded_for_craft():
    parked, components, _ = _play("expressive", lambda info: _action(10, sync=1))
    finished, _, _ = _play("expressive", _follower)
    assert finished > parked + 5.0, (finished, parked)
    assert components.get("end_transition", 0.0) < 0.0


def test_repeated_transitions_lose_novelty():
    env = MoodMixEnv(max_steps=64)
    novelty = []
    for _ in range(2):
        obs, info = env.reset(seed=5, options=PAIR)
        while True:
            obs, reward, terminated, truncated, info = env.step(_follower(info))
            if terminated or truncated:
                break
        novelty.append(env.episode_components.get("novelty", 0.0))
    assert novelty[0] > novelty[1]
    assert novelty[1] == pytest.approx(0.0, abs=1e-9)


def test_flickering_the_low_band_does_not_beat_one_clean_swap():
    """A trained policy learned to toggle the low EQ every beat to farm this."""
    def clean(info):
        fader = int(round(_target(info) * 20))
        sync = 1 if info["step"] == 0 else 0
        if info["step"] < 32:
            return _action(fader, eq_b=(0, 1, 1), sync=sync)
        return _action(fader, sync=sync)

    def flicker(info):
        fader = int(round(_target(info) * 20))
        sync = 1 if info["step"] == 0 else 0
        low = info["step"] % 2
        return _action(fader, eq_b=(low, 1, 1), sync=sync)

    _, clean_parts, _ = _play("expressive", clean)
    _, flicker_parts, _ = _play("expressive", flicker)
    assert clean_parts["bass_swap"] > 0.0
    assert clean_parts["bass_swap"] >= flicker_parts["bass_swap"]


def test_filter_pays_for_sweeping_not_for_leaving_it_on():
    def held(info):
        return _action(int(round(_target(info) * 20)), filt=2,
                       sync=1 if info["step"] == 0 else 0)

    def swept(info):
        fader = int(round(_target(info) * 20))
        return _action(fader, filt=2 if 8 <= info["step"] < 20 else 0,
                       sync=1 if info["step"] == 0 else 0)

    _, held_parts, _ = _play("expressive", held)
    _, swept_parts, _ = _play("expressive", swept)
    assert swept_parts["filter_craft"] > held_parts["filter_craft"]


def test_pad_variety_pays_more_than_repeating_one_pad():
    """Repeating the same pad must not collect the novelty bonus."""
    def repeated(info):
        hit = 1 if info["step"] % 4 == 0 else 0
        return _action(int(round(_target(info) * 20)), drum=hit,
                       sync=1 if info["step"] == 0 else 0)

    def varied(info):
        pads = (1, 2, 3, 4)
        hit = pads[(info["step"] // 4) % 4] if info["step"] % 4 == 0 else 0
        return _action(int(round(_target(info) * 20)), drum=hit,
                       sync=1 if info["step"] == 0 else 0)

    _, same_parts, _ = _play("expressive", repeated)
    _, varied_parts, _ = _play("expressive", varied)
    assert varied_parts["pad_craft"] > same_parts["pad_craft"] + 0.2


def test_toggling_the_filter_does_not_refill_its_budget():
    def churn(info):
        fader = int(round(_target(info) * 20))
        return _action(fader, filt=1 + info["step"] % 2,
                       sync=1 if info["step"] == 0 else 0)

    _, churn_parts, _ = _play("expressive", churn)
    _, swept_parts, _ = _play(
        "expressive",
        lambda info: _action(int(round(_target(info) * 20)),
                             filt=2 if 8 <= info["step"] < 18 else 0,
                             sync=1 if info["step"] == 0 else 0),
    )
    assert swept_parts["filter_craft"] > 0.0
    assert churn_parts["filter_craft"] <= swept_parts["filter_craft"]


def test_gesture_spam_does_not_beat_two_placed_gestures():
    def spammed(info):
        fader = int(round(_target(info) * 20))
        return _action(fader, filt=1 + info["step"] % 2,
                       sync=1 if info["step"] == 0 else 0)

    def placed(info):
        fader = int(round(_target(info) * 20))
        return _action(fader, filt=2 if info["step"] in (32, 48) else 0,
                       sync=1 if info["step"] == 0 else 0)

    _, spam_parts, _ = _play("expressive", spammed)
    _, placed_parts, _ = _play("expressive", placed)

    def craft(parts):
        return parts["phrase"] + parts.get("busy_hands", 0.0)

    assert placed_parts["phrase"] > 0.0
    assert craft(placed_parts) > craft(spam_parts), (craft(placed_parts), craft(spam_parts))


def test_energy_arc_counts_peaks_not_variance():
    """Variance is maximised by pumping the fader; an arc is not."""
    rise = np.linspace(0.2, 1.0, 64)
    span, peaks, arc = _energy_arc(rise)
    assert peaks <= 1 and arc == 1.0 and span > 0.5

    saw = np.tile(np.concatenate([np.linspace(0.2, 1.0, 8),
                                  np.linspace(1.0, 0.2, 8)]), 4)
    span, peaks, arc = _energy_arc(saw)
    assert peaks >= 3 and arc < 0.4
    # ...and the saw has the *higher* variance the old term paid for.
    assert saw.std() > rise.std()


def test_raking_the_fader_back_and_forth_is_charged():
    """The policy learned to sweep A to B and back every eight beats."""
    def rake(info):
        phase = (info["step"] // 8) % 2
        fader = (info["step"] % 8) * 3 if phase == 0 else 20 - (info["step"] % 8) * 3
        return _action(int(min(20, max(0, fader))),
                       sync=1 if info["step"] == 0 else 0)

    blend, _, _ = _play("expressive", _follower)
    raked, parts, _ = _play("expressive", rake)
    assert parts.get("route", 0.0) < 0.0
    assert blend > raked + 2.0, (blend, raked)


def test_a_single_crossing_has_no_route_penalty():
    _, parts, _ = _play("expressive", _follower)
    assert parts.get("route", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_parking_at_085_is_not_a_handover():
    _, components, env = _play(
        "expressive", lambda info: _action(15 if info["step"] < 60 else 17)
    )
    assert env.trace[-1]["crossfader"] == pytest.approx(0.85)
    assert components["end_transition"] < 0.0


def test_sync_engage_pays_once_on_the_engage_step():
    """The latch has no edge credit of its own; the bonus supplies it."""
    env = MoodMixEnv(max_steps=8, reward_profile="expressive")
    env.reset(seed=5, options=PAIR)

    _, _, _, _, info = env.step(_action(0, sync=1))
    assert info["sync"] is True
    assert info["reward_components"]["sync_engage"] == pytest.approx(
        SYNC_ENGAGE_BONUS)

    # Holding the latch must not pay again.
    _, _, _, _, info = env.step(_action(0, sync=1))
    assert info["reward_components"]["sync_engage"] == pytest.approx(0.0)

    # Releasing and re-engaging must not farm a second bonus.
    env.step(_action(0, sync=3))
    _, _, _, _, info = env.step(_action(0, sync=1))
    assert info["sync"] is True
    assert info["reward_components"]["sync_engage"] == pytest.approx(0.0)


def test_binary_sync_latches_once_and_holds():
    """With the collapsed head there is no release, only engage/no-op."""
    env = MoodMixEnv(max_steps=8, reward_profile="expressive", binary_sync=True)
    assert env.action_space.nvec[7] == 2
    env.reset(seed=5, options=PAIR)

    _, _, _, _, info = env.step(_action(0, sync=1))
    assert info["sync"] is True
    assert info["reward_components"]["sync_engage"] == pytest.approx(
        SYNC_ENGAGE_BONUS)

    # A no-op beat must not drop the latch.
    _, _, _, _, info = env.step(_action(0, sync=0))
    assert info["sync"] is True


def test_sync_engage_bonus_is_ablatable():
    env = MoodMixEnv(max_steps=8, reward_profile="expressive", sync_bonus=0.0)
    env.reset(seed=5, options=PAIR)
    _, _, _, _, info = env.step(_action(0, sync=1))
    assert info["sync"] is True
    assert info["reward_components"]["sync_engage"] == pytest.approx(0.0)

    with pytest.raises(ValueError):
        MoodMixEnv(sync_bonus=-0.1)


def test_term_scales_ablate_one_component():
    env = MoodMixEnv(max_steps=64, reward_profile="expressive",
                     term_scales={"bass_swap": 0.0})
    obs, info = env.reset(seed=5, options=PAIR)
    while True:
        obs, _, terminated, truncated, info = env.step(_follower(info))
        if terminated or truncated:
            break
    assert env.episode_components["bass_swap"] == pytest.approx(0.0, abs=1e-9)
    assert env.episode_components["mix_quality"] > 0.0


def test_reward_profiles_are_selectable_and_scale_differently():
    with pytest.raises(ValueError):
        MoodMixEnv(reward_profile="nope")
    curve, _, _ = _play("curve", _follower)
    expressive, _, _ = _play("expressive", _follower)
    assert np.isfinite([curve, expressive]).all()
    assert curve > 30.0
    assert abs(curve - expressive) > 5.0
