"""Local web demo: run a transition, or a short DJ set, and serve trace + audio."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .env import MOODS, MoodMixEnv
from .evaluation import model_policy, random_policy
from .policies import heuristic_action, smoothstep
from .render_audio import render_mix, render_set, render_transition

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_DIR = ROOT / "web"
DEFAULT_RUNS_DIR = ROOT / "runs" / "web"
POLICIES = ("ppo", "human", "random")
MAX_SET_TRANSITIONS = 8


def create_app(
    model_path: str | Path | None = None,
    runs_dir: str | Path | None = None,
    web_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Flask:
    web_dir = Path(web_dir) if web_dir else DEFAULT_WEB_DIR
    runs_dir = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = None
    model_error = None
    if model_path:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(model_path))
        except Exception as exc:  # pragma: no cover - environment dependent
            model_error = str(exc)

    # Which moods this crate can actually play.  Offering all six when only a
    # few exist made the demo look broken: picking an empty mood silently
    # played a different one.
    available_moods: list[str] = list(MOODS)
    try:
        catalog = MoodMixEnv(data_dir=data_dir)
        present = {track["mood"] for track in catalog.tracks}
        available_moods = [mood for mood in MOODS if mood in present]
        catalog.close()
    except Exception:  # pragma: no cover - empty or missing crate
        pass

    app = Flask(__name__, static_folder=str(web_dir), static_url_path="/static")

    def _policy_for(policy_name: str, env: MoodMixEnv):
        if policy_name == "ppo":
            if model is not None:
                return model_policy(model), False
            return heuristic_action, True
        if policy_name == "random":
            return random_policy(env), False
        return heuristic_action, False

    def _validate(mood: str, policy_name: str):
        """Return an error response, or None when the request is playable."""
        if mood not in MOODS:
            return jsonify({"error": f"unknown mood {mood!r}"}), 400
        if available_moods and mood not in available_moods:
            return jsonify({
                "error": f"this crate has no {mood} tracks; pick one of "
                         f"{', '.join(available_moods)}",
                "available_moods": available_moods,
            }), 400
        if policy_name not in POLICIES:
            return jsonify({"error": f"unknown policy {policy_name!r}"}), 400
        return None

    def _record(info: dict, reward: float, clock: float) -> dict:
        progress = info["step"] / info["max_steps"]
        target = smoothstep((progress - 0.12) / 0.68)
        return {
            "step": info["step"],
            "bar": info["bar"],
            "t": round(clock, 4),
            "fader": round(float(info["crossfader"]), 4),
            "target": round(float(target), 4),
            "gain_a": round(float(info["gain_a"]), 4),
            "gain_b": round(float(info["gain_b"]), 4),
            "eq_a": list(info["eq_a"]),
            "eq_b": list(info["eq_b"]),
            "sync": bool(info["sync"]),
            "filter": int(info["filter"]),
            "drum": info["drum"],
            "deck_a": info["deck_a"],
            "deck_b": info["deck_b"],
            "reward": round(float(reward), 4),
            "components": {k: round(float(v), 4)
                           for k, v in info["reward_components"].items()},
        }

    @app.get("/")
    def index():
        return send_from_directory(web_dir, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({
            "moods": list(MOODS),
            "available_moods": available_moods,
            "policies": list(POLICIES),
            "max_set_transitions": MAX_SET_TRANSITIONS,
            "model_loaded": model is not None,
            "model_error": model_error,
        })

    @app.get("/api/episode")
    def episode():
        mood = request.args.get("mood", "party")
        policy_name = request.args.get("policy", "human")
        try:
            seed = int(request.args.get("seed", "0"))
        except ValueError:
            return jsonify({"error": "seed must be an integer"}), 400
        error = _validate(mood, policy_name)
        if error is not None:
            return error

        env = MoodMixEnv(data_dir=data_dir, mood=mood, max_steps=64)
        env.action_space.seed(seed)
        policy, fallback = _policy_for(policy_name, env)

        obs, info = env.reset(seed=seed)
        beat = 60.0 / float(env.track_a["bpm"])
        steps: list[dict] = []
        while True:
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            steps.append(_record(info, reward, len(steps) * beat))
            if terminated or truncated:
                break

        audio_name = f"{mood}_{policy_name}_{seed}.wav"
        audio_path = runs_dir / audio_name
        if not audio_path.exists():
            render_transition(env, audio_path)

        summary = {
            "total": round(float(sum(env.episode_components.values())), 3),
            "mean_reward": round(float(sum(s["reward"] for s in steps) / len(steps)), 4),
            "completed": bool(steps[-1]["fader"] > 0.9),
            "final_fader": steps[-1]["fader"],
            "components": {k: round(float(v), 3)
                           for k, v in env.episode_components.items()},
            "transitions": 1,
        }
        return jsonify({
            "mood": mood,
            "policy": policy_name,
            "fallback": fallback,
            "seed": seed,
            "steps": steps,
            "summary": summary,
            "beat_seconds": round(beat, 5),
            "track_a": dict(env.track_a, band_curve=None),
            "track_b": dict(env.track_b, band_curve=None),
            "audio_url": f"/audio/{audio_name}",
        })

    @app.get("/api/set")
    def dj_set():
        mood = request.args.get("mood", "party")
        policy_name = request.args.get("policy", "ppo")
        try:
            seed = int(request.args.get("seed", "0"))
            count = int(request.args.get("transitions", "4"))
        except ValueError:
            return jsonify({"error": "seed and transitions must be integers"}), 400
        count = max(2, min(count, MAX_SET_TRANSITIONS))
        error = _validate(mood, policy_name)
        if error is not None:
            return error

        env = MoodMixEnv(data_dir=data_dir, mood=mood, max_steps=64)
        env.action_space.seed(seed)
        policy, fallback = _policy_for(policy_name, env)

        steps: list[dict] = []
        transitions: list[dict] = []
        mixes = []
        sample_rate = 44100
        clock = 0.0
        next_track: str | None = None
        played: list[str] = []
        first_a: dict = {}
        first_b: dict = {}

        for index in range(count):
            options: dict = {"mood": mood}
            if next_track is not None:
                # A set hands over: deck B of one transition opens the next.
                options["track_a"] = next_track
            if played:
                options["avoid_tracks"] = played[-2:]
            obs, info = env.reset(seed=seed + index, options=options)
            beat = 60.0 / float(env.track_a["bpm"])
            start = clock
            first_step = len(steps)
            while True:
                action = policy(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                clock_in_transition = (len(steps) - first_step) * beat
                record = _record(info, reward, start + clock_in_transition)
                record["transition"] = index
                steps.append(record)
                if terminated or truncated:
                    break

            if index == 0:
                first_a = {k: v for k, v in env.track_a.items() if k != "band_curve"}
                first_b = {k: v for k, v in env.track_b.items() if k != "band_curve"}

            mix, sample_rate = render_mix(env, sr=sample_rate)
            mixes.append(mix)
            duration = len(mix) / float(sample_rate)
            transitions.append({
                "index": index,
                "track_a": env.track_a["id"],
                "track_b": env.track_b["id"],
                "mood_a": env.track_a["mood"],
                "mood_b": env.track_b["mood"],
                "bpm_a": env.track_a["bpm"],
                "bpm_b": env.track_b["bpm"],
                "key_a": env.track_a["key"],
                "key_b": env.track_b["key"],
                "beat_seconds": round(beat, 5),
                "start_seconds": round(start, 3),
                "duration_seconds": round(duration, 3),
                "steps": len(env.trace),
                "total": round(float(sum(env.episode_components.values())), 3),
                "completed": bool(env.crossfader > 0.9),
                "components": {k: round(float(v), 3)
                               for k, v in env.episode_components.items()},
            })
            clock = start + duration
            played.extend([env.track_a["id"], env.track_b["id"]])
            next_track = env.track_b["id"]

        audio_name = f"set_{mood}_{policy_name}_{seed}.wav"
        audio_path = runs_dir / audio_name
        if not audio_path.exists():
            render_set(mixes, sample_rate, audio_path)

        totals = [t["total"] for t in transitions]
        components: dict[str, float] = {}
        for transition in transitions:
            for key, value in transition["components"].items():
                components[key] = components.get(key, 0.0) + value / len(transitions)
        return jsonify({
            "mood": mood,
            "policy": policy_name,
            "fallback": fallback,
            "seed": seed,
            "steps": steps,
            "set": {
                "count": len(transitions),
                "transitions": transitions,
                "duration_seconds": round(clock, 3),
            },
            "summary": {
                "total": round(float(sum(totals) / len(totals)), 3),
                "mean_reward": round(float(sum(s["reward"] for s in steps) / len(steps)), 4),
                "completed": all(t["completed"] for t in transitions),
                "final_fader": round(float(env.crossfader), 3),
                "components": {k: round(v, 3) for k, v in components.items()},
                "transitions": len(transitions),
            },
            "beat_seconds": transitions[0]["beat_seconds"],
            "track_a": first_a,
            "track_b": first_b,
            "audio_url": f"/audio/{audio_name}",
        })

    @app.get("/audio/<path:name>")
    def audio(name: str):
        return send_from_directory(runs_dir, name)

    return app
