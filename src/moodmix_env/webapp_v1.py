"""Web player for DJSet-v1: pick two tracks, render, watch the mix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .env_v1 import MoodMixV1Env, key_distance
from .expert_v1 import expert_action
from .render_v2 import render_transition
from .select_v1 import best_cues

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web_v1"
RUNS_DIR = ROOT / "runs" / "v1" / "web"
DEFAULT_MODEL = ROOT / "models" / "ppo_v1_final.zip"


def _track_summary(track: dict) -> dict:
    return {
        "id": track["id"],
        "file": track["file"],
        "bpm": track["bpm"],
        "key": track["key"],
        "mode": track["mode"],
        "mood": track["mood"],
        "duration": track["duration"],
        "cues": [
            {"beat": int(cue["beat"]), "bar": int(cue["bar"]),
             "section": cue["section"], "energy": float(cue["energy"])}
            for cue in track["cues"]
        ],
    }


def create_app(data_dir: str | Path | None = None,
               model_path: str | Path | None = DEFAULT_MODEL,
               runs_dir: str | Path | None = None,
               web_dir: str | Path | None = None) -> Flask:
    data_dir = Path(data_dir) if data_dir else ROOT / "crate"
    runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
    web_dir = Path(web_dir) if web_dir else WEB_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = None
    model_error = None
    model_path = Path(model_path) if model_path else None
    if model_path and model_path.exists():
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(model_path), device="cpu")
        except Exception as exc:  # pragma: no cover - input dependent
            model_error = str(exc)
    else:
        model_error = f"model not found: {model_path}"

    catalogue = MoodMixV1Env(data_dir=data_dir, max_steps=32)
    tracks = [_track_summary(track) for track in catalogue.tracks]
    track_ids = [track["id"] for track in tracks]
    pairs = []
    for a, b, weight in sorted(catalogue._pairs, key=lambda row: -row[2]):
        if a["id"] not in track_ids or b["id"] not in track_ids:
            continue
        pairs.append({
            "a": a["id"], "b": b["id"],
            "a_bpm": a["bpm"], "b_bpm": b["bpm"],
            "key_distance": key_distance(a, b),
            "weight": round(float(weight), 4),
        })
        if len(pairs) >= 120:
            break
    catalogue.close()

    app = Flask(__name__, static_folder=str(web_dir),
                static_url_path="/static")

    def _policy_for(kind: str, env: MoodMixV1Env):
        if kind == "model" and model is not None:
            def _model_policy(obs):
                action, _ = model.predict(obs, deterministic=True)
                return action
            return _model_policy
        return lambda obs: expert_action(env)

    @app.get("/")
    def index():
        return send_from_directory(web_dir, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({
            "model_loaded": model is not None,
            "model_error": model_error,
            "model_path": str(model_path) if model_path else None,
            "tracks": len(tracks),
            "max_steps": 32,
        })

    @app.get("/api/library")
    def library():
        return jsonify({
            "tracks": tracks,
            "pairs": pairs,
            "model_loaded": model is not None,
            "model_error": model_error,
            "model_path": str(model_path) if model_path else None,
            "max_steps": 32,
        })

    @app.post("/api/render")
    def render_endpoint():
        payload = request.get_json(silent=True) or {}
        track_a = payload.get("track_a")
        track_b = payload.get("track_b")
        if track_a not in track_ids or track_b not in track_ids:
            return jsonify({"error": "pick two known tracks"}), 400
        if track_a == track_b:
            return jsonify({"error": "pick two different tracks"}), 400
        policy_kind = payload.get("policy", "model")
        if policy_kind not in ("model", "expert"):
            return jsonify({"error": "unknown policy"}), 400
        if policy_kind == "model" and model is None:
            policy_kind = "expert"
        try:
            seed = int(payload.get("seed", 0))
        except (TypeError, ValueError):
            seed = 0
        use_best = bool(payload.get("best_cues", True))
        cue_a = payload.get("cue_a")
        cue_b = payload.get("cue_b")
        cue_a = int(cue_a) if cue_a is not None else None
        cue_b = int(cue_b) if cue_b is not None else None

        env = MoodMixV1Env(data_dir=data_dir, max_steps=32)
        options = {"track_a": track_a, "track_b": track_b}
        if cue_a is not None:
            options["cue_a"] = cue_a
        if cue_b is not None:
            options["cue_b"] = cue_b

        if use_best and cue_a is None and cue_b is None:
            picked = best_cues(env, _policy_for(policy_kind, env),
                               track_a, track_b, seed=seed)
            options["cue_a"] = picked["cue_a"]
            options["cue_b"] = picked["cue_b"]

        env = MoodMixV1Env(data_dir=data_dir, max_steps=32)
        policy = _policy_for(policy_kind, env)
        obs, info = env.reset(seed=seed, options=options)
        steps = []
        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.trace[-1]
            action = frame["action"]
            steps.append({
                "step": env.step_count,
                "beat_a": frame["beat_a"],
                "beat_b": frame["beat_b"],
                "fader": float(action[0]),
                "eq_a": [float(v) for v in action[1:4]],
                "eq_b": [float(v) for v in action[4:7]],
                "filter": float(action[7]),
                "gain_a": float(frame["gain_a"]),
                "gain_b": float(frame["gain_b"]),
                "bands_a": [float(v) for v in frame["bands_a"]],
                "bands_b": [float(v) for v in frame["bands_b"]],
                "mix_a": [float(v) for v in frame["mix_a"]],
                "mix_b": [float(v) for v in frame["mix_b"]],
                "level": float(frame["level"]),
                "level_smooth": float(frame["level_smooth"]),
                "low_overlap": float(frame["low_overlap"]),
                "blend": float(frame["blend"]),
                "target": float(frame["target"]),
                "shape": float(frame["shape"]),
                "jump": float(frame["jump"]),
                "section_a": env._section(env.track_a, int(frame["beat_a"])),
                "section_b": env._section(env.track_b, int(frame["beat_b"])),
            })
            if terminated or truncated:
                break

        beat_times = env.track_a["beat_times"]
        relative = (beat_times[env.cue_a:env.cue_a + env.step_count + 1]
                    - beat_times[env.cue_a]).tolist()
        name = (f"{track_a.replace(' ', '_')}__{track_b.replace(' ', '_')}"
                f"__{policy_kind}__{env.cue_a}_{env.cue_b}__{seed}.wav")
        audio_path = runs_dir / name
        if not audio_path.exists():
            render_transition(env, audio_path)
        total = float(sum(env.episode_components.values()))
        return jsonify({
            "track_a": _track_summary(env.track_a),
            "track_b": _track_summary(env.track_b),
            "cue_a": int(env.cue_a),
            "cue_b": int(env.cue_b),
            "policy": policy_kind,
            "seed": seed,
            "steps": steps,
            "beat_times": relative,
            "beat_seconds": 60.0 / float(env.track_a["bpm"]),
            "duration": float(relative[-1]) if relative else 0.0,
            "reward": total,
            "components": {key: float(value)
                           for key, value in env.episode_components.items()},
            "final_fader": float(env.fader),
            "mean_level_jump": float(np.mean(env._level_jumps))
            if env._level_jumps else 0.0,
            "mean_low_overlap": float(np.mean(env._low_overlaps))
            if env._low_overlaps else 0.0,
            "audio_url": f"/audio/{name}",
        })

    @app.get("/audio/<path:name>")
    def audio(name: str):
        return send_from_directory(runs_dir, name)

    return app
