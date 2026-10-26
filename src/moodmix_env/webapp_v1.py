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
from urllib.parse import quote

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, redirect, request, send_from_directory

from .env_v1 import MoodMixV1Env, key_distance
from .env_v3 import MoodMixV3Env
from .expert_v1 import expert_action
from .expert_v3 import expert_v3_action
from .library_v2 import AUDIO_EXTS, build_manifest
from .render_v2 import prepare_decks, render_transition, write_wav
from .render_v3 import render_transition as render_transition_v3
from .select_v1 import best_cues

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web_v1"
RUNS_DIR = ROOT / "runs" / "v1" / "web"
CACHE_DIR = ROOT / "runs" / "v1" / "library_cache"
TRAINING_DIR = ROOT / "runs" / "training"
MLOPS_DIR = ROOT / "mlops"
EXPLAINER = ROOT / "docs" / "EXPLAINER.md"
DEFAULT_MODEL = ROOT / "models" / "ppo_v3_final.zip"

_STATE_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_FULL_LOCK = threading.Lock()
_STEM_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _waveform(path: Path, points: int = 320) -> list[float]:
    """Return a small RMS overview of a track without loading it all."""
    try:
        with sf.SoundFile(str(path)) as handle:
            total = len(handle)
            if total <= 0:
                return []
            block = max(1024, total // (points * 12))
            values: list[float] = []
            while True:
                chunk = handle.read(block, dtype="float32", always_2d=True)
                if chunk.size == 0:
                    break
                values.append(float(np.sqrt(np.mean(chunk ** 2))))
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float32)
        edges = np.linspace(0, len(arr), points + 1).astype(int)
        out = []
        for index in range(points):
            lo = edges[index]
            hi = max(lo + 1, edges[index + 1])
            out.append(float(arr[lo:hi].mean()))
        peak = max(out) or 1.0
        return [round(min(1.0, value / peak), 4) for value in out]
    except Exception:
        return []


def _stretch_track(source: Path, dest: Path, rate: float,
                   cue_seconds: float, delay_seconds: float,
                   duration: float) -> None:
    """Write a tempo-matched, silence-padded MP3 for a full-song session."""
    delay = max(0, int(round(delay_seconds * 1000)))
    chain = []
    if abs(rate - 1.0) > 1e-4:
        chain.append(f"rubberband=tempo={rate:.6f}")
    chain.extend([
        f"atrim=start={max(0.0, cue_seconds):.6f}",
        "asetpts=PTS-STARTPTS",
        f"adelay={delay}|{delay}",
        "asetpts=PTS-STARTPTS",
    ])
    spec = ",".join(chain)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(source), "-vn", "-ac", "2",
               "-filter:a", spec, "-t", f"{duration:.3f}",
               "-c:a", "libmp3lame", "-b:a", "320k", str(dest)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and "rubberband" in spec:
        fallback = spec.replace(f"rubberband=tempo={rate:.6f}",
                                f"atempo={rate:.6f}")
        command[command.index(spec)] = fallback
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def _write_mp3(audio: np.ndarray, sr: int, dest: Path) -> None:
    wav = dest.with_suffix(".wav")
    try:
        sf.write(str(wav), audio, int(sr), subtype="PCM_16")
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k",
             str(dest)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg stem encode failed")
    finally:
        if wav.exists():
            wav.unlink()


def _write_stems(source: Path, drums: Path, tonal: Path) -> None:
    """Split one track into percussive and tonal MP3 stems."""
    audio, sr = librosa.load(str(source), sr=44100, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    if audio.shape[0] > 2:
        audio = audio[:2]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    percussive = np.empty_like(audio)
    harmonic = np.empty_like(audio)
    for channel in range(audio.shape[0]):
        p, h = librosa.effects.hpss(
            np.ascontiguousarray(audio[channel]), margin=2.0)
        percussive[channel] = p
        harmonic[channel] = h
    _write_mp3(percussive.T, sr, drums)
    _write_mp3(harmonic.T, sr, tonal)


def _stem_paths(track: dict, data_dir: Path, runs_dir: Path
                ) -> tuple[Path, Path]:
    source = data_dir / track["file"]
    stat = source.stat()
    digest = hashlib.sha1(
        f"stems-v1|{track['file']}|{stat.st_size}|{stat.st_mtime}"
        .encode("utf-8")).hexdigest()[:16]
    stem_dir = runs_dir / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    drums = stem_dir / f"{digest}_drums.mp3"
    tonal = stem_dir / f"{digest}_tonal.mp3"
    if not drums.exists() or not tonal.exists():
        with _STEM_LOCK:
            if not drums.exists() or not tonal.exists():
                try:
                    _write_stems(source, drums, tonal)
                except Exception:
                    for path in (drums, tonal):
                        if path.exists():
                            path.unlink()
                    raise
    return drums, tonal


def _processed_stem(source: Path, rate: float, cue_seconds: float,
                    delay_seconds: float, duration: float,
                    stem_dir: Path) -> Path:
    stat = source.stat()
    digest = hashlib.sha1(
        (f"stem-session-v1|{source.name}|{stat.st_size}|{stat.st_mtime}|"
         f"{rate:.6f}|{cue_seconds:.3f}|{delay_seconds:.3f}|"
         f"{duration:.3f}").encode("utf-8")).hexdigest()[:16]
    dest = stem_dir / f"b_{digest}.mp3"
    if not dest.exists():
        with _STEM_LOCK:
            if not dest.exists():
                temp = dest.with_suffix(".part.mp3")
                try:
                    _stretch_track(source, temp, rate, cue_seconds,
                                   delay_seconds, duration)
                    temp.replace(dest)
                finally:
                    if temp.exists():
                        temp.unlink()
    return dest


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

    env_cls = MoodMixV1Env
    if model is not None:
        try:
            if int(model.action_space.shape[0]) == 12:
                env_cls = MoodMixV3Env
        except Exception:
            env_cls = MoodMixV1Env

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
        if env.action_space.shape[0] == 12:
            return lambda obs: expert_v3_action(env)
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
        live = bool(payload.get("live"))
        full = bool(payload.get("full"))
        cue_a = payload.get("cue_a")
        cue_b = payload.get("cue_b")
        cue_a = int(cue_a) if cue_a is not None else None
        cue_b = int(cue_b) if cue_b is not None else None

        data_dir = Path(snapshot["data_dir"])
        env = env_cls(data_dir=data_dir,
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

        env = env_cls(data_dir=data_dir,
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
                "drum_a": float(action[8]) if len(action) > 8 else 1.0,
                "drum_b": float(action[9]) if len(action) > 9 else 1.0,
                "echo_a": float(action[10]) if len(action) > 10 else 0.0,
                "echo_b": float(action[11]) if len(action) > 11 else 0.0,
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
        total = float(sum(env.episode_components.values()))
        result = {
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
            "audio_url": None,
            "deck_a_url": None,
            "deck_b_url": None,
            "live": live,
            "full": False,
        }

        if live and full:
            ta = np.asarray(env.track_a["beat_times"], dtype=np.float64)
            tb = np.asarray(env.track_b["beat_times"], dtype=np.float64)
            beat_a = float(np.median(np.diff(ta)))
            beat_b = float(np.median(np.diff(tb)))
            rate = beat_b / beat_a
            transition_start = float(ta[env.cue_a])
            transition_end = float(ta[env.cue_a + env.step_count])
            b_cue = float(tb[env.cue_b])
            b_cue_stretched = b_cue / rate
            b_duration = float(env.track_b.get("duration") or 0.0)
            b_remaining = max(0.0, b_duration / rate - b_cue_stretched)
            session_duration = max(transition_start + b_remaining,
                                   transition_end + 1.0)
            source_b = data_dir / env.track_b["file"]
            try:
                stat = source_b.stat()
                digest = hashlib.sha1(
                    (f"full-v2|{env.track_b['file']}|{stat.st_size}|"
                     f"{stat.st_mtime}|{b_cue:.3f}|{rate:.6f}|"
                     f"{transition_start:.3f}|{session_duration:.3f}")
                    .encode("utf-8")
                ).hexdigest()[:16]
                full_dir = runs_dir / "full"
                full_dir.mkdir(parents=True, exist_ok=True)
                b_name = f"a_to_b_{digest}.mp3"
                b_path = full_dir / b_name
                if not b_path.exists():
                    with _FULL_LOCK:
                        if not b_path.exists():
                            temp = b_path.with_suffix(".part.mp3")
                            try:
                                _stretch_track(source_b, temp, rate,
                                               b_cue_stretched,
                                               transition_start,
                                               session_duration)
                                temp.replace(b_path)
                            finally:
                                if temp.exists():
                                    temp.unlink()
            except Exception as exc:  # pragma: no cover - ffmpeg dependent
                return jsonify({
                    "error": f"full-song prep failed: {exc}"
                }), 500

            try:
                a_drums, a_tonal = _stem_paths(env.track_a, data_dir, runs_dir)
                b_drums_src, b_tonal_src = _stem_paths(env.track_b, data_dir,
                                                       runs_dir)
                stem_dir = runs_dir / "stems"
                b_drums = _processed_stem(b_drums_src, rate, b_cue_stretched,
                                          transition_start, session_duration,
                                          stem_dir)
                b_tonal = _processed_stem(b_tonal_src, rate, b_cue_stretched,
                                          transition_start, session_duration,
                                          stem_dir)
            except Exception as exc:  # pragma: no cover - librosa dependent
                return jsonify({
                    "error": f"stem prep failed: {exc}"
                }), 500

            rms_a = np.asarray(env.track_a["features"].get("rms", []),
                               dtype=np.float64)
            rms_b = np.asarray(env.track_b["features"].get("rms", []),
                               dtype=np.float64)
            level_a = float(np.median(rms_a)) if len(rms_a) else 0.5
            level_b = float(np.median(rms_b)) if len(rms_b) else 0.5
            b_gain = float(np.clip(level_a / max(level_b, 1e-6), 0.5, 2.0))
            result.update({
                "full": True,
                "duration": session_duration,
                "transition_start": transition_start,
                "transition_end": transition_end,
                "rate": rate,
                "deck_b_gain": b_gain,
                "deck_a_url": f"/track/{quote(env.track_a['file'])}",
                "deck_b_url": f"/audio/full/{b_name}",
                "wave_a": _waveform(data_dir / env.track_a["file"]),
                "wave_b": _waveform(source_b),
                "deck_a_drums_url": f"/audio/stems/{a_drums.name}",
                "deck_a_tonal_url": f"/audio/stems/{a_tonal.name}",
                "deck_b_drums_url": f"/audio/stems/{b_drums.name}",
                "deck_b_tonal_url": f"/audio/stems/{b_tonal.name}",
            })
            return jsonify(result)

        stem = (f"{track_a.replace(' ', '_')}__{track_b.replace(' ', '_')}"
                f"__{policy_kind}__{env.cue_a}_{env.cue_b}__{seed}")
        audio_name = f"{stem}.wav"
        deck_a_name = f"{stem}__deck_a.wav"
        deck_b_name = f"{stem}__deck_b.wav"
        audio_path = runs_dir / audio_name
        deck_a_path = runs_dir / deck_a_name
        deck_b_path = runs_dir / deck_b_name
        if live:
            if not deck_a_path.exists() or not deck_b_path.exists():
                deck_a, deck_b = prepare_decks(env)
                write_wav(deck_a, 44100, deck_a_path)
                write_wav(deck_b, 44100, deck_b_path)
            result.update({
                "deck_a_url": f"/audio/{deck_a_name}",
                "deck_b_url": f"/audio/{deck_b_name}",
            })
            return jsonify(result)

        if not audio_path.exists():
            if env.action_space.shape[0] == 12:
                render_transition_v3(env, audio_path)
            else:
                render_transition(env, audio_path)
        result["audio_url"] = f"/audio/{audio_name}"
        return jsonify(result)

    @app.get("/track/<path:name>")
    def track_file(name: str):
        data_dir = Path(_snapshot()["data_dir"])
        if Path(name).suffix.lower() not in AUDIO_EXTS:
            return jsonify({"error": "not an audio file"}), 404
        return send_from_directory(data_dir, name, conditional=True)

    @app.get("/audio/<path:name>")
    def audio(name: str):
        return send_from_directory(runs_dir, name)

    return app
