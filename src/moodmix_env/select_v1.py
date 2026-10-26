"""Choose the best phrase pair for a demo render.

Training samples cues for variety.  A showcase should not: it should try the
strongest phrases and keep the one whose 32-beat blend is steadiest.  The
score is the environment return minus the macro-level range and low-band
overlap, so a phrase that dips or muddies loses even if its endpoint is good.
"""

from __future__ import annotations

import numpy as np

from .env_v1 import MoodMixV1Env, key_distance


def rollout(env: MoodMixV1Env, policy, options: dict,
            seed: int = 0) -> tuple[float, dict]:
    obs, _ = env.reset(seed=seed, options=options)
    total = 0.0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    levels = np.asarray([frame["level_smooth"] for frame in env.trace])
    level_range = (float(np.percentile(levels, 95)
                         - np.percentile(levels, 5))
                   if len(levels) else 0.0)
    overlap = float(np.mean(env._low_overlaps)) if env._low_overlaps else 0.0
    penalty = 2.0 if env.fader < 0.95 else 0.0
    score = total - 2.0 * level_range - 0.8 * overlap - penalty
    return score, {
        "cue_a": int(env.cue_a), "cue_b": int(env.cue_b),
        "return": float(total), "level_range": level_range,
        "low_overlap": overlap,
    }


def best_cues(env: MoodMixV1Env, policy, track_a: str, track_b: str,
              limit: int = 5, seed: int = 0) -> dict:
    a = next(track for track in env.tracks if track["id"] == track_a)
    b = next(track for track in env.tracks if track["id"] == track_b)
    if key_distance(a, b) > 5:
        # Not a fatal error for a deliberate pairing; just a warning signal.
        pass
    pool_a = sorted(env._cue_pool(a, "a"),
                    key=lambda cue: -cue.get("_weight", 0.0))[:limit]
    pool_b = sorted(env._cue_pool(b, "b"),
                    key=lambda cue: -cue.get("_weight", 0.0))[:limit]
    best = None
    for cue_a in pool_a:
        for cue_b in pool_b:
            options = {"track_a": track_a, "track_b": track_b,
                       "cue_a": int(cue_a["beat"]),
                       "cue_b": int(cue_b["beat"])}
            score, result = rollout(env, policy, options, seed=seed)
            result["score"] = score
            if best is None or score > best["score"]:
                best = result
    return best
