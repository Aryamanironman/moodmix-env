"""Download a Creative Commons starter crate from Jamendo.

Jamendo hosts independent music you can stream and download under CC licences,
which makes it the practical way to get real songs into MoodMix.  Get a free
client id at https://devportal.jamendo.com/signup and export it:

    export JAMENDO_CLIENT_ID=...

    python scripts/fetch_jamendo.py --mood party --limit 5 --out crate
    python scripts/fetch_jamendo.py --tags "electronic,house" --vocal --limit 20 --out crate
    python scripts/analyze_library.py --audio-dir crate
    python scripts/train.py --data-dir crate --timesteps 1000000

`--lyrics` also asks LRCLIB (free, no key) for timed lyrics and writes a .lrc
next to each track that has them.  Jamendo is mostly instrumental, so expect
few hits unless you filter with `--vocal`.

Every download is credited in <out>/attribution.json; keep it if you publish
anything built from the crate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.env import MOODS

API_DEFAULT = "https://api.jamendo.com"
API_VERSION = "v3.0"
AUDIO_FORMAT = "mp32"  # Jamendo encoding; the files it returns are MP3
LYRICS_API = "https://lrclib.net/api/get"
SIGNUP = "https://devportal.jamendo.com/signup"

# Jamendo tag filters that land roughly on each MoodMix mood.
MOOD_TAGS = {
    "party": "electronic,house,dance",
    "energetic": "electronic,rock,upbeat",
    "happy": "pop,acoustic,happy",
    "relaxed": "chillout,ambient,relax",
    "emotional": "piano,sad,acoustic",
    "focused": "lofi,study,instrumental",
}


def _slug(text: str, limit: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    return cleaned[:limit] or "track"


def _get_json(url: str, timeout: float, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise SystemExit(f"Jamendo API error {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise SystemExit(f"could not reach {url.split('?')[0]}: {last}")


def search_tracks(
    api_base: str,
    client_id: str,
    limit: int,
    offset: int,
    tags: str | None = None,
    vocal: str | None = None,
    order: str = "popularity_month",
    timeout: float = 30.0,
) -> list[dict]:
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "include": "musicinfo+licenses",
        "audioformat": AUDIO_FORMAT,
        "order": order,
    }
    if tags:
        params["tags"] = tags
    if vocal:
        params["vocalinstrumental"] = vocal
    url = f"{api_base}/{API_VERSION}/tracks/?{urllib.parse.urlencode(params)}"
    payload = _get_json(url, timeout=timeout)
    headers = payload.get("headers") or {}
    if headers.get("status") != "success":
        message = headers.get("error_message") or payload.get("error_message") or payload
        raise SystemExit(f"Jamendo rejected the request: {message}")
    return list(payload.get("results") or [])


def download(url: str, dest: Path, timeout: float = 120.0) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as source, open(partial, "wb") as out:
        shutil.copyfileobj(source, out, length=1 << 16)
    partial.replace(dest)
    return dest.stat().st_size


def fetch_lyrics(artist: str, title: str, album: str | None,
                 duration: float | None, timeout: float = 20.0) -> str | None:
    params = {"artist_name": artist, "track_name": title}
    if album:
        params["album_name"] = album
    if duration:
        params["duration"] = int(round(duration))
    request = urllib.request.Request(
        f"{LYRICS_API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "moodmix-env/0.1 (crate fetcher)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None
    return payload.get("syncedLyrics") or payload.get("plainLyrics")


def _credit(track: dict) -> dict:
    licenses = track.get("licenses") or []
    return {
        "id": track.get("id"),
        "title": track.get("name"),
        "artist": track.get("artist_name"),
        "album": track.get("album_name"),
        "duration": track.get("duration"),
        "license": ", ".join(item.get("url", "") for item in licenses if item.get("url")),
        "source": track.get("shareurl"),
    }


def _load_credits(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "crate",
                        help="folder to download into (default: ./crate)")
    parser.add_argument("--mood", choices=MOODS, default=None,
                        help="pick Jamendo tags that fit this mood")
    parser.add_argument("--tags", default=None,
                        help="comma-separated Jamendo tags; overrides --mood")
    parser.add_argument("--limit", type=int, default=10, help="tracks to fetch")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--order", default="popularity_month",
                        help="Jamendo order, e.g. popularity_month, releasedate_desc")
    parser.add_argument("--vocal", action="store_true",
                        help="only tracks with vocals (needed for lyrics)")
    parser.add_argument("--lyrics", action="store_true",
                        help="write .lrc sidecars from LRCLIB where available")
    parser.add_argument("--client-id", default=None,
                        help="Jamendo client id (default: $JAMENDO_CLIENT_ID)")
    parser.add_argument("--api-base", default=os.environ.get("JAMENDO_API_BASE", API_DEFAULT))
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="seconds between downloads, to stay polite")
    parser.add_argument("--dry-run", action="store_true",
                        help="search and list, but download nothing")
    args = parser.parse_args()

    client_id = args.client_id or os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        raise SystemExit(
            "no Jamendo client id. Create a free app at "
            f"{SIGNUP} then pass --client-id or set JAMENDO_CLIENT_ID."
        )

    tags = args.tags or (MOOD_TAGS[args.mood] if args.mood else None)
    mood = args.mood
    tracks = search_tracks(
        api_base=args.api_base.rstrip("/"),
        client_id=client_id,
        limit=args.limit,
        offset=args.offset,
        tags=tags,
        vocal="vocal" if args.vocal else None,
        order=args.order,
    )
    if not tracks:
        raise SystemExit(f"no tracks matched tags={tags!r}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    credits_path = out_dir / "attribution.json"
    credits = _load_credits(credits_path)

    downloaded = skipped = no_audio = with_lyrics = planned = 0
    for track in tracks:
        title = track.get("name") or f"track-{track.get('id')}"
        artist = track.get("artist_name") or "unknown"
        stem = f"{_slug(mood or 'mix')}_{track.get('id')}_{_slug(artist, 24)}-{_slug(title)}"
        dest = out_dir / f"{stem}.mp3"

        url = track.get("audiodownload") or ""
        allowed = track.get("audiodownload_allowed", True)
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            print(f"  skip     {dest.name} (already downloaded)")
            continue
        if not url or allowed is False:
            no_audio += 1
            print(f"  no audio {artist} - {title} (download not offered)")
            continue
        if args.dry_run:
            planned += 1
            print(f"  would get {artist} - {title}  [{track.get('audiodownload')}]")
            continue

        try:
            size = download(url, dest)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f"  failed   {artist} - {title}: {exc}")
            continue
        downloaded += 1
        print(f"  saved    {dest.name}  ({size / 1024:.0f} KB)")

        credits[stem] = _credit(track)
        credits_path.write_text(json.dumps(credits, indent=2), encoding="utf-8")

        if args.lyrics:
            lyrics = fetch_lyrics(artist, title, track.get("album_name"), track.get("duration"))
            if lyrics:
                dest.with_suffix(".lrc").write_text(lyrics, encoding="utf-8")
                with_lyrics += 1
                print("           + lyrics (.lrc)")
        time.sleep(max(0.0, args.sleep))

    count = planned if args.dry_run else downloaded
    verb = "would download" if args.dry_run else "downloaded"
    print(f"\n{verb} {count}, skipped {skipped}, unavailable {no_audio}")
    if with_lyrics:
        print(f"lyrics written: {with_lyrics}")
    if not args.dry_run and downloaded:
        print(f"crate: {out_dir}")
        print(f"next:  python scripts/analyze_library.py --audio-dir {out_dir}")
        print(f"then:  python scripts/train.py --data-dir {out_dir} --timesteps 1000000")


if __name__ == "__main__":
    main()
