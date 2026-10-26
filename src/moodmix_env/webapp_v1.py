"""Web player for DJSet-v1: pick any analysed folder, then mix it."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, redirect, request, send_from_directory

from .env_v1 import MoodMixV1Env, key_distance
from .expert_v1 import expert_action
from .library_v2 import AUDIO_EXTS, build_manifest
from .render_v2 import render_transition
from .select_v1 import best_cues

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web_v1"
RUNS_DIR = ROOT / "runs" / "v1" / "web"
CACHE_DIR = ROOT / "runs" / "v1" / "library_cache"
TRAINING_DIR = ROOT / "runs" / "training"
MLOPS_DIR = ROOT / "mlops"
EXPLAINER = ROOT / "docs" / "EXPLAINER.md"
DEFAULT_MODEL = ROOT / "models" / "ppo_v2_final.zip"

_STATE_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


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


def _audio_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )


def _signature(folder: Path, files: list[Path]) -> str:
    digest = hashlib.sha1(str(folder.resolve()).encode("utf-8"))
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
                      .encode("utf-8"))
    return digest.hexdigest()[:16]


def _has_vocal(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tracks = payload.get("tracks", [])
        return bool(tracks) and all(
            "vocal" in track.get("features", {}) for track in tracks)
    except Exception:
        return False


def _manifest_for(folder: Path, files: list[Path]) -> tuple[Path, bool]:
    """Return (manifest path, needs_analysis).

    A manifest inside the folder is used when it is newer than every audio
    file and contains the vocal features the current environment expects.
    Otherwise analysis goes to runs/v1/library_cache and the user's music
    folder is never written to.
    """
    newest = max(path.stat().st_mtime for path in files)
    inside = folder / "manifest_v2.json"
    if (inside.exists() and inside.stat().st_mtime >= newest
            and _has_vocal(inside)):
        return inside, False
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"manifest_{_signature(folder, files)}.json"
    if (cached.exists() and cached.stat().st_mtime >= newest
            and _has_vocal(cached)):
        return cached, False
    return cached, True


def _library_payload(data_dir: Path, manifest: Path) -> dict:
    env = MoodMixV1Env(data_dir=data_dir, manifest=manifest, max_steps=32)
    tracks = [_track_summary(track) for track in env.tracks]
    pairs = []
    for a, b, weight in sorted(env._pairs, key=lambda row: -row[2]):
        pairs.append({
            "a": a["id"], "b": b["id"],
            "a_bpm": a["bpm"], "b_bpm": b["bpm"],
            "key_distance": key_distance(a, b),
            "weight": round(float(weight), 4),
        })
        if len(pairs) >= 160:
            break
    env.close()
    return {"tracks": tracks, "pairs": pairs}


def create_app(data_dir: str | Path | None = None,
               model_path: str | Path | None = DEFAULT_MODEL,
               runs_dir: str | Path | None = None,
               web_dir: str | Path | None = None) -> Flask:
    runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    web_dir = Path(web_dir) if web_dir else WEB_DIR
    data_dir = Path(data_dir) if data_dir else ROOT / "crate"
    manifest = data_dir / "manifest_v2.json"

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

    state: dict[str, Any] = {
        "data_dir": data_dir,
        "manifest": manifest,
        "label": data_dir.name,
        "tracks": [],
        "pairs": [],
        "error": None,
    }
    if manifest.exists():
        try:
            state.update(_library_payload(data_dir, manifest))
        except Exception as exc:  # pragma: no cover - bad manifest
            state["error"] = str(exc)

    app = Flask(__name__, static_folder=str(web_dir),
                static_url_path="/static")

    def _snapshot() -> dict:
        with _STATE_LOCK:
            return {
                "data_dir": str(state["data_dir"]),
                "manifest": str(state["manifest"]),
                "label": state["label"],
                "tracks": list(state["tracks"]),
                "pairs": list(state["pairs"]),
                "error": state["error"],
            }

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

    @app.get("/training/")
    def training_index():
        if (MLOPS_DIR / "index.html").exists():
            return redirect("/mlops/")
        runs = []
        if TRAINING_DIR.exists():
            runs = sorted((path.name for path in TRAINING_DIR.iterdir()
                           if path.is_dir()), reverse=True)
        cards = "".join(
            f"<li><b>{html.escape(name)}</b>"
            f" <a href='/training/{html.escape(name)}/report.html'>report</a>"
            f" <a href='/training/{html.escape(name)}/curves.png'>curves</a>"
            f" <a href='/training/{html.escape(name)}/eval.csv'>eval.csv</a>"
            f" <a href='/training/{html.escape(name)}/train_returns.csv'>returns.csv</a>"
            f"</li>"
            for name in runs
        ) or "<li>No training runs recorded yet.</li>"
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>DJSet-v1 training records</title>
<style>
body{{background:#f5f6f8;color:#151a22;font-family:system-ui,sans-serif;
margin:0;padding:28px;line-height:1.55}} .wrap{{max-width:900px;margin:0 auto}}
h1{{font-size:1.4rem}} li{{margin:8px 0}} a{{color:#a84a06;margin-right:8px}}
.note{{color:#5b6472;font-size:.85rem}}
</style></head><body><div class="wrap">
<h1>DJSet-v1 training records</h1>
<p class="note">Runs under runs/training. Each report includes curves, eval
metrics and CSV downloads.</p><ul>{cards}</ul>
<p><a href="/">Back to the player</a></p></div></body></html>"""

    @app.get("/mlops/")
    def mlops_index():
        if (MLOPS_DIR / "index.html").exists():
            return send_from_directory(MLOPS_DIR, "index.html")
        return "<p>No registry yet. Run scripts/collect_runs.py.</p>"

    @app.get("/mlops/<path:name>")
    def mlops_file(name: str):
        return send_from_directory(MLOPS_DIR, name)

    @app.get("/training/<path:name>")
    def training_file(name: str):
        return send_from_directory(TRAINING_DIR, name)

    @app.get("/explainer")
    def explainer():
        page = web_dir / "explainer.html"
        if page.exists():
            return send_from_directory(web_dir, "explainer.html")
        text = ("# Explainer\n\nThe explainer has not been written yet."
                if not EXPLAINER.exists()
                else EXPLAINER.read_text(encoding="utf-8", errors="ignore"))
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>DJSet-v1 explainer</title>
<style>
body{{background:#f5f6f8;color:#151a22;font-family:system-ui,sans-serif;
margin:0;padding:28px;line-height:1.6}} .wrap{{max-width:980px;margin:0 auto}}
pre{{white-space:pre-wrap;background:#fff;border:1px solid #d7dde5;
padding:18px;font:0.86rem ui-monospace,Consolas,monospace}}
a{{color:#a84a06}}
</style></head><body><div class="wrap">
<p><a href="/">Back to the player</a></p>
<pre>{html.escape(text)}</pre></div></body></html>"""

    @app.get("/api/health")
    def health():
        return jsonify({
            "model_loaded": model is not None,
            "model_error": model_error,
            "model_path": str(model_path) if model_path else None,
            "library": _snapshot()["label"],
            "tracks": len(_snapshot()["tracks"]),
            "max_steps": 32,
        })

    @app.get("/api/library")
    def library():
        snapshot = _snapshot()
        return jsonify({
            **snapshot,
            "model_loaded": model is not None,
            "model_error": model_error,
            "model_path": str(model_path) if model_path else None,
            "max_steps": 32,
        })

    @app.post("/api/scan")
    def scan():
        payload = request.get_json(silent=True) or {}
        preset = payload.get("preset")
        if preset == "crate":
            folder = ROOT / "crate"
        elif preset == "demo":
            folder = ROOT / "data" / "demo"
        else:
            raw = str(payload.get("path", "")).strip().strip('"')
            if not raw:
                return jsonify({"error": "enter a folder path"}), 400
            folder = Path(raw).expanduser()
        try:
            folder = folder.resolve()
        except OSError as exc:
            return jsonify({"error": str(exc)}), 400
        if not folder.is_dir():
            return jsonify({"error": f"not a folder: {folder}"}), 400
        try:
            files = _audio_files(folder)
        except OSError as exc:
            return jsonify({"error": str(exc)}), 400
        if not files:
            return jsonify({
                "error": "no mp3/wav/flac/ogg/m4a files at the top level "
                         "of that folder"
            }), 400

        manifest_path, needs_analysis = _manifest_for(folder, files)
        job_id = uuid.uuid4().hex[:10]
        job: dict[str, Any] = {
            "status": "running",
            "done": 0,
            "total": len(files),
            "current": files[0].name,
            "label": folder.name,
            "data_dir": str(folder),
            "error": None,
        }
        with _JOBS_LOCK:
            _JOBS[job_id] = job
            for stale in list(_JOBS)[:-20]:
                _JOBS.pop(stale, None)

        def progress(done: int, total: int, current: str) -> None:
            job["done"] = max(0, done - 1)
            job["total"] = total
            job["current"] = current

        def worker() -> None:
            try:
                if needs_analysis:
                    print(f"[scan] analyzing {folder} ({len(files)} files)",
                          flush=True)
                    build_manifest(folder, manifest_path, verbose=False,
                                   progress=progress)
                else:
                    print(f"[scan] reusing {manifest_path}", flush=True)
                payload_lib = _library_payload(folder, manifest_path)
                with _STATE_LOCK:
                    state.update({
                        "data_dir": folder,
                        "manifest": manifest_path,
                        "label": folder.name,
                        "tracks": payload_lib["tracks"],
                        "pairs": payload_lib["pairs"],
                        "error": None,
                    })
                job["status"] = "done"
                job["done"] = len(files)
                job["library"] = payload_lib
                print(f"[scan] ready: {len(payload_lib['tracks'])} tracks",
                      flush=True)
            except Exception as exc:  # pragma: no cover - input dependent
                job["status"] = "error"
                job["error"] = str(exc)
                print(f"[scan] failed: {exc}", flush=True)

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"job": job_id, "total": len(files),
                        "analyzing": needs_analysis})

    @app.get("/api/scan/<job_id>")
    def scan_status(job_id: str):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return jsonify({"error": "unknown job"}), 404
            snapshot = dict(job)
        return jsonify(snapshot)

    @app.post("/api/browse")
    def browse():
        if sys.platform != "win32":
            return jsonify({
                "path": None,
                "error": "native picker is wired for Windows; type the path"
            })
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$d.Description = 'Choose a folder of audio files'; "
            "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ Write-Output $d.SelectedPath }"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", script],
                capture_output=True, text=True, timeout=600,
            )
        except Exception as exc:  # pragma: no cover - desktop dependent
            return jsonify({"path": None, "error": str(exc)})
        lines = [line.strip() for line in result.stdout.splitlines()
                 if line.strip()]
        return jsonify({"path": lines[-1] if lines else None,
                        "error": result.stderr.strip() or None})

    @app.post("/api/render")
    def render_endpoint():
        snapshot = _snapshot()
        track_ids = [track["id"] for track in snapshot["tracks"]]
        payload = request.get_json(silent=True) or {}
        track_a = payload.get("track_a")
        track_b = payload.get("track_b")
        if track_a not in track_ids or track_b not in track_ids:
            return jsonify({"error": "pick two tracks from the current library"}), 400
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

        data_dir = Path(snapshot["data_dir"])
        env = MoodMixV1Env(data_dir=data_dir,
                           manifest=Path(snapshot["manifest"]), max_steps=32)
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

        env = MoodMixV1Env(data_dir=data_dir,
                           manifest=Path(snapshot["manifest"]), max_steps=32)
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
