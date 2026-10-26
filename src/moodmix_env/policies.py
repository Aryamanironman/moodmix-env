"""Shared DJ policies used by the demo, evaluation and tests."""

from __future__ import annotations


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def heuristic_action(obs, info: dict) -> list[int]:
    """A hand-written DJ performance: the human baseline for the agent."""
    step = info["step"]
    max_steps = info["max_steps"]
    target = smoothstep((step / max_steps - 0.12) / 0.68)
    fader = int(round(target * 20))

    eq_a = [1, 1, 1]
    eq_b = [1, 1, 1]
    mid = max_steps // 2
    if step < 8:                # tease deck B with no bass or sparkle
        eq_b[0] = 0
        eq_b[2] = 0
    elif step < 16:             # let the bass in, keep it dark
        eq_b[2] = 0
    # The bass swap: only one kick sounds at a time.  Deck A owns the low end
    # until the midpoint, then the low end moves to deck B.  Both halves happen
    # while both decks are audible, which is the whole point of the move.
    if 8 <= step < mid:
        eq_b[0] = 0
    elif step >= mid:
        eq_a[0] = 0
        eq_b[0] = 1
    if step >= max_steps - 6:   # thin deck A out as it leaves
        eq_a[1] = 0

    sync = 1 if step == 1 else 0
    filt = 2 if step < 8 else 0     # high-pass sweep while B is in the intro
    if step % 4 == 0:
        drum = 1                    # kick on the downbeat
    elif step % 2 == 1:
        drum = 3                    # hat on the offbeat
    else:
        drum = 0
    if step == max_steps // 2:
        drum = 4                    # clap right at the midpoint
    return [fader, *eq_a, *eq_b, sync, drum, filt]
