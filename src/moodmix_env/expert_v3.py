"""Phrase-aware DJ expert for DJSet-v3.

The v1 fader, EQ and filter moves are unchanged.  The expert adds a drum
hand-off at the midpoint and two short echo throws: one on the outgoing deck
around the swap, and one on the incoming deck at the end of the blend.
"""

from __future__ import annotations

import numpy as np

from .expert_v1 import expert_action


def expert_v3_action(env) -> np.ndarray:
    base = expert_action(env)
    step = int(env.step_count)
    total = int(env.max_steps)
    mid = total // 2

    drum_a = 1.0
    if step >= mid - 2:
        drum_a = float(np.clip(1.0 - (step - (mid - 2)) / 4.0, 0.0, 1.0))
    drum_b = 1.0

    echo_a = 0.0
    if mid - 3 <= step <= mid + 1:
        echo_a = 0.6 * float(np.clip(1.0 - abs(step - mid) / 3.5, 0.0, 1.0))
    echo_b = 0.0
    if step >= total - 4:
        echo_b = 0.5 * float(np.clip((step - (total - 5)) / 4.0, 0.0, 1.0))

    return np.array([*base, drum_a, drum_b, echo_a, echo_b],
                    dtype=np.float32)
