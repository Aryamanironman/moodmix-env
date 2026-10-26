"""A hand-written, phrase-aware DJ for DJSet-v1.

This is the behaviour the policy is cloned from and the reference a rendered
transition is judged against: hold deck B's bass out, let the fader follow one
slow arc, swap the low end once around the middle, and release the high-pass
before the drop.
"""

from __future__ import annotations

import numpy as np


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def expert_action(env) -> np.ndarray:
    step = int(env.step_count)
    total = int(env.max_steps)
    mid = total // 2

    # Fader: hold for four beats, then one slow equal-power arc.
    fader = _smoothstep((step - 4.0) / max(1.0, total - 8.0))

    eq_a = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    eq_b = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # The bass swap: A owns the low end until the midpoint, then B does.
    # A four-beat ramp keeps the hand-over from clicking, and B lands slightly
    # hot (+2.4 dB) so cutting A's low does not take the low end out of the
    # mix for a beat - the audible dip the first version had.
    swap = float(np.clip((step - (mid - 3.0)) / 4.0, 0.0, 1.0))
    eq_a[0] = 0.50 - 0.48 * swap
    eq_b[0] = 0.04 + 0.66 * swap
    if step > mid + 4:
        settle = float(np.clip((step - (mid + 4)) / 6.0, 0.0, 1.0))
        eq_b[0] = 0.70 - 0.15 * settle

    # B's top end stays slightly dark while it is being introduced, then opens.
    open_highs = float(np.clip((step - 4.0) / max(1.0, mid - 4.0), 0.0, 1.0))
    eq_b[2] = 0.30 + 0.20 * open_highs

    # A high-pass on the incoming deck for the first phrase, released by beat
    # 12.  This is the "thin the intro" move that keeps the blend from mudding.
    if step < 8:
        filt = 0.65
    elif step < 12:
        filt = 0.65 * (1.0 - (step - 8) / 4.0)
    else:
        filt = 0.0

    return np.array([fader, *eq_a, *eq_b, filt], dtype=np.float32)
