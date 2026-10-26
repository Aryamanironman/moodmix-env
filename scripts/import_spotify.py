"""Match a Spotify playlist to local audio files and build a manifest.

Spotify's API gives metadata, not full audio.  This script fetches the
playlist/liked-song list, matches each entry to a file you already own, and
analyzes the matched files.  It never downloads from Spotify.

    python scripts/import_spotify.py --playlist https://open.spotify.com/playlist/... --audio-dir ./crate
    python scripts/import_spotify.py --liked --audio-dir ./crate
    python scripts/import_spotify.py --playlist ... --audio-dir ./crate --dry-run

Credentials: create a Spotify app, then set SPOTIPY_CLIENT_ID and
SPOTIPY_CLIENT_SECRET (or pass --client-id / --client-secret).  The first run
opens a browser for user authorization and caches the token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.library import analyze_track
from moodmix_env.spotify import (
    build_local_index, fetch_spotify_tracks, match_tracks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--playlist", help="playlist URL, URI or ID")
    source.add_argument("--liked", action="store_true", help="your liked songs")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="folder of audio files you own")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <audio-dir>/manifest.json")
    parser.add_argument("--mood", default=None,
                        help="force one mood for all matched tracks")
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--client-secret", default=None)
    parser.add_argument("--redirect-uri", default="http://127.0.0.1:8888/callback")
    parser.add_argument("--cache", default=".spotify_token_cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="report matches without analyzing audio")
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()
    if not audio_dir.exists():
        raise SystemExit(f"audio dir not found: {audio_dir}")
    out = (args.out or audio_dir / "manifest.json").resolve()

    print("fetching Spotify metadata ...", flush=True)
    records = fetch_spotify_tracks(
        playlist=args.playlist, liked=args.liked,
        client_id=args.client_id, client_secret=args.client_secret,
        redirect_uri=args.redirect_uri, cache_path=args.cache,
    )
    print(f"spotify tracks: {len(records)}", flush=True)

    print("indexing local files ...", flush=True)
    index = build_local_index(audio_dir)
    matched, unmatched = match_tracks(records, index)
    print(f"matched: {len(matched)}   unmatched: {len(unmatched)}", flush=True)

    for record, path in matched:
        print(f"  {record['artist']} - {record['title']}  ->  {Path(path).name}")
    for record in unmatched:
        print(f"  UNMATCHED  {record['artist']} - {record['title']}")

    report = {
        "matched": [
            {"spotify": record, "file": str(Path(path).resolve())}
            for record, path in matched
        ],
        "unmatched": unmatched,
    }
    report_path = out.parent / "spotify_match_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"match report: {report_path}", flush=True)

    if args.dry_run:
        return
    if not matched:
        raise SystemExit("no local files matched; nothing to analyze")

    tracks = []
    for number, (record, path) in enumerate(matched, 1):
        print(f"analyzing {number}/{len(matched)}: {Path(path).name}", flush=True)
        track = analyze_track(path, mood=args.mood)
        track["file"] = str(Path(path).resolve())
        track["spotify_id"] = record["id"]
        track["spotify_title"] = record["title"]
        track["spotify_artist"] = record["artist"]
        tracks.append(track)

    manifest = {
        "sample_rate": 22050,
        "beats_per_bar": 4,
        "bars": max(len(t["sections"]) for t in tracks),
        "beats": max(len(t["band_curve"]) for t in tracks),
        "tracks": tracks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {out}")
    print(f"tracks:   {len(tracks)}")


if __name__ == "__main__":
    main()
