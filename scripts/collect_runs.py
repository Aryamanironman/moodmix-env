"""Build the MLOps registry: every model, checkpoint ladder and training record.

    python scripts/collect_runs.py

Writes mlops/index.json, runs.csv, README.md and index.html, copies the small
records (CSV, plots, reports, meta) into mlops/records, and records a SHA256
for every model/checkpoint zip.  The large zips stay where they are; the
registry points at them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLOPS = ROOT / "mlops"
RECORDS = MLOPS / "records"
META = MLOPS / "meta"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_info(path: Path) -> dict:
    info = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256_16": sha256(path),
    }
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(path), device="cpu")
        shape = getattr(model.observation_space, "shape", None)
        info.update({
            "obs_dim": int(shape[0]) if shape else None,
            "action_space": str(getattr(model, "action_space", "")),
            "num_timesteps": int(getattr(model, "num_timesteps", 0)),
            "n_steps": int(getattr(model, "n_steps", 0)),
            "batch_size": int(getattr(model, "batch_size", 0)),
            "learning_rate": float(getattr(model, "learning_rate", 0.0)),
            "ent_coef": float(getattr(model, "ent_coef", 0.0)),
        })
    except Exception as exc:
        info["load_error"] = str(exc)
    meta_path = path.with_name(f"{path.stem}_meta.json")
    if meta_path.exists():
        info["meta"] = read_json(meta_path)
    return info


def copy_records(source: Path, name: str) -> list[str]:
    target = RECORDS / name
    copied: list[str] = []
    if not source.exists():
        return copied
    target.mkdir(parents=True, exist_ok=True)
    allowed = {".csv", ".json", ".md", ".html", ".png", ".txt"}
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            shutil.copy2(path, target / path.name)
            copied.append(str((target / path.name).relative_to(MLOPS))
                          .replace("\\", "/"))
    return copied


def main() -> None:
    MLOPS.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    models: list[dict] = []
    for path in sorted((ROOT / "models").glob("*.zip")):
        info = model_info(path)
        model_name = path.stem
        records_dir = ROOT / "runs" / "training" / model_name
        info["records"] = copy_records(records_dir, model_name)
        if records_dir.exists():
            eval_path = records_dir / "eval.csv"
            if eval_path.exists():
                rows = list(csv.DictReader(eval_path.open(encoding="utf-8")))
                if rows:
                    info["last_eval"] = rows[-1]
        meta_path = path.with_name(f"{path.stem}_meta.json")
        if meta_path.exists():
            shutil.copy2(meta_path, META / meta_path.name)
        models.append(info)

    ladders: list[dict] = []
    for directory in sorted(ROOT.glob("checkpoints*")):
        if not directory.is_dir():
            continue
        entries = []
        for path in sorted(directory.glob("*.zip")):
            info = model_info(path)
            parts = path.stem.split("_")
            for index, part in enumerate(parts):
                if part == "steps" and index:
                    try:
                        info["timesteps"] = int(parts[index - 1])
                    except ValueError:
                        pass
            entries.append(info)
        if entries:
            ladders.append({
                "id": f"checkpoints:{directory.name}",
                "directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
                "entries": entries,
            })

    training: list[dict] = []
    runs_dir = ROOT / "runs" / "training"
    if runs_dir.exists():
        for directory in sorted(runs_dir.iterdir()):
            if not directory.is_dir():
                continue
            eval_rows = []
            eval_path = directory / "eval.csv"
            if eval_path.exists():
                eval_rows = list(csv.DictReader(eval_path.open(encoding="utf-8")))
            config = read_json(directory / "config.json")
            copied = copy_records(directory, directory.name)
            training.append({
                "id": f"training:{directory.name}",
                "name": directory.name,
                "config": config,
                "last_eval": eval_rows[-1] if eval_rows else None,
                "eval_points": len(eval_rows),
                "records": copied,
            })

    ablation: list[dict] = []
    ablation_dir = ROOT / "runs" / "ablation"
    if ablation_dir.exists():
        copied = copy_records(ablation_dir, "ablation")
        summary_path = ablation_dir / "summary.csv"
        rows = []
        if summary_path.exists():
            rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        ablation.append({"id": "ablation", "variants": rows,
                         "records": copied})

    registry = {
        "generated_utc": generated,
        "models": models,
        "checkpoint_ladders": ladders,
        "training_runs": training,
        "ablations": ablation,
    }
    (MLOPS / "index.json").write_text(json.dumps(registry, indent=2),
                                      encoding="utf-8")

    rows = []
    for model in models:
        meta = model.get("meta", {})
        last = model.get("last_eval", {})
        rows.append({
            "kind": "model",
            "name": Path(model["path"]).stem,
            "obs_dim": model.get("obs_dim", ""),
            "timesteps": model.get("num_timesteps", ""),
            "eval_return": meta.get("eval_mean_reward",
                                    last.get("mean_return", "")),
            "path": model["path"],
            "sha256_16": model["sha256_16"],
            "records": ";".join(model.get("records", [])),
        })
    for ladder in ladders:
        for entry in ladder["entries"]:
            rows.append({
                "kind": "checkpoint",
                "name": f"{ladder['directory']}/{Path(entry['path']).name}",
                "obs_dim": entry.get("obs_dim", ""),
                "timesteps": entry.get("timesteps", entry.get("num_timesteps", "")),
                "eval_return": "",
                "path": entry["path"],
                "sha256_16": entry["sha256_16"],
                "records": "",
            })
    for run in training:
        rows.append({
            "kind": "training",
            "name": run["name"],
            "obs_dim": "",
            "timesteps": run.get("last_eval", {}).get("timestep", "")
            if run.get("last_eval") else "",
            "eval_return": run.get("last_eval", {}).get("mean_return", "")
            if run.get("last_eval") else "",
            "path": f"runs/training/{run['name']}",
            "sha256_16": "",
            "records": ";".join(run.get("records", [])),
        })
    for run in ablation:
        for variant in run["variants"]:
            rows.append({
                "kind": "ablation",
                "name": variant.get("variant", ""),
                "obs_dim": "",
                "timesteps": variant.get("timesteps", ""),
                "eval_return": variant.get("final_return", ""),
                "path": "runs/ablation/summary.csv",
                "sha256_16": "",
                "records": ";".join(run.get("records", [])),
            })

    with (MLOPS / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())
                                if rows else ["kind", "name"])
        writer.writeheader()
        writer.writerows(rows)

    table = "\n".join(
        f"| {row['kind']} | {row['name']} | {row['obs_dim']} | "
        f"{row['timesteps']} | {row['eval_return']} | {row['sha256_16']} |"
        for row in rows
    )
    (MLOPS / "README.md").write_text(
        "# MLOps registry\n\n"
        f"Generated {generated}.\n\n"
        "Large model and checkpoint zips stay in `models/` and `checkpoints*/`; "
        "the hash identifies them. Small records are copied under `records/`.\n\n"
        "| kind | name | obs | steps | eval return | sha256 |\n"
        "|---|---|---|---|---|---|\n" + table + "\n",
        encoding="utf-8",
    )

    cards = []
    for model in models:
        last = model.get("last_eval") or {}
        meta = model.get("meta", {})
        ev = meta.get("eval_mean_reward", last.get("mean_return", "-"))
        records = "".join(
            f"<a href='{path.replace(chr(92), '/')}'>"
            f"{Path(path).name}</a> "
            for path in model.get("records", [])
        )
        cards.append(
            f"<tr><td>model</td><td>{Path(model['path']).name}</td>"
            f"<td>{model.get('obs_dim', '-')}</td>"
            f"<td>{model.get('num_timesteps', '-')}</td>"
            f"<td>{ev if ev == '-' else f'{float(ev):+.3f}'}</td>"
            f"<td>{model['sha256_16']}</td><td>{records}</td></tr>"
        )
    for ladder in ladders:
        for entry in ladder["entries"]:
            cards.append(
                f"<tr><td>checkpoint</td>"
                f"<td>{ladder['directory']}/{Path(entry['path']).name}</td>"
                f"<td>{entry.get('obs_dim', '-')}</td>"
                f"<td>{entry.get('timesteps', entry.get('num_timesteps', '-'))}</td>"
                f"<td>-</td><td>{entry['sha256_16']}</td><td></td></tr>"
            )
    for run in training:
        last = run.get("last_eval") or {}
        links = "".join(
            f"<a href='{path.replace(chr(92), '/')}'>"
            f"{Path(path).name}</a> " for path in run.get("records", []))
        cards.append(
            f"<tr><td>training</td><td>{run['name']}</td><td>-</td>"
            f"<td>{last.get('timestep', '-')}</td>"
            f"<td>{float(last.get('mean_return', 0)):+.3f}"
            f"</td><td>-</td><td>{links}</td></tr>"
        )
    for run in ablation:
        for variant in run["variants"]:
            links = "".join(
                f"<a href='{path.replace(chr(92), '/')}'>"
                f"{Path(path).name}</a> " for path in run.get("records", []))
            cards.append(
                f"<tr><td>ablation</td><td>{variant.get('variant', '')}</td>"
                f"<td>-</td><td>{variant.get('timesteps', '-')}</td>"
                f"<td>{variant.get('final_return', '-')}</td><td>-</td>"
                f"<td>{links}</td></tr>"
            )

    (MLOPS / "index.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>MoodMix MLOps registry</title>
<style>
body{{background:#0a0d12;color:#e8eef6;font-family:system-ui,sans-serif;
margin:0;padding:28px;line-height:1.55}}
.wrap{{max-width:1400px;margin:0 auto}} h1{{font-size:1.35rem;margin:0 0 4px}}
.sub{{color:#8a97a8;font-family:ui-monospace,monospace;font-size:.76rem;
margin-bottom:18px}} table{{border-collapse:collapse;width:100%;font-size:.76rem}}
th,td{{border-bottom:1px solid #222d3b;padding:7px 9px;text-align:left;
vertical-align:top}} th{{font-family:ui-monospace,monospace;font-size:.64rem;
text-transform:uppercase;color:#8a97a8}} a{{color:#f0a13a;margin-right:6px;
text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body><div class="wrap">
<h1>MoodMix MLOps registry</h1>
<div class="sub">generated {generated} · large zips stay in models/ and checkpoints*/ · sha256 first 16</div>
<table><tr><th>kind</th><th>name</th><th>obs</th><th>steps</th>
<th>eval return</th><th>sha256</th><th>records</th></tr>
{''.join(cards)}
</table></div></body></html>""",
        encoding="utf-8",
    )
    print(f"wrote {MLOPS / 'index.json'}")
    print(f"models {len(models)}  checkpoint ladders {len(ladders)}  "
          f"training {len(training)}  ablation {len(ablation)}")


if __name__ == "__main__":
    main()
