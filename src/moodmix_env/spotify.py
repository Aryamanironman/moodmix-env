"""Spotify metadata → local-file matching.

Spotify's API does not provide full-track audio.  This module takes playlist
metadata, matches it to audio files the user already owns, and hands the
matched paths to the library analyzer.  It never downloads tracks.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = re.sub(r"\b(feat\.?|ft\.?|featuring)\b.*$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def playlist_id_from_url(value: str) -> str:
    value = value.strip()
    if value.startswith("spotify:playlist:"):
        return value.rsplit(":", 1)[-1]
    parsed = urlparse(value)
    if "spotify.com" in parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if "playlist" in parts:
            return parts[parts.index("playlist") + 1]
    return value.split("?")[0]


def track_record(item: dict) -> dict:
    track = item.get("track") or item
    artists = track.get("artists") or []
    album = track.get("album") or {}
    return {
        "id": track.get("id"),
        "title": track.get("name") or "",
        "artist": ", ".join(a.get("name", "") for a in artists if a.get("name")),
        "album": album.get("name", ""),
        "duration_seconds": round((track.get("duration_ms") or 0) / 1000),
        "isrc": (track.get("external_ids") or {}).get("isrc", ""),
        "preview_url": track.get("preview_url"),
        "uri": track.get("uri", ""),
    }


def read_tags(path: Path) -> dict:
    """Best-effort title/artist tags; falls back to the filename."""
    info = {"title": path.stem, "artist": ""}
    try:
        from mutagen import File
        audio = File(path)
        if audio is not None and audio.tags:
            tags = audio.tags
            title = tags.get("TIT2") or tags.get("title") or tags.get("\xa9nam")
            artist = tags.get("TPE1") or tags.get("artist") or tags.get("\xa9ART")
            if title:
                info["title"] = str(title)
            if artist:
                info["artist"] = str(artist)
    except Exception:
        pass
    return info


def build_local_index(audio_dir: str | Path) -> dict[str, Path]:
    audio_dir = Path(audio_dir)
    index: dict[str, Path] = {}
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        tags = read_tags(path)
        keys = {
            normalize(path.stem),
            normalize(f"{tags['artist']} {tags['title']}"),
            normalize(tags["title"]),
        }
        for key in keys:
            if key:
                index.setdefault(key, path)
    return index


def match_tracks(
    spotify_tracks: list[dict], local_index: dict[str, Path | str]
) -> tuple[list[tuple[dict, Path | str]], list[dict]]:
    """Match Spotify records to local files. Each file is used at most once."""
    used: set[str] = set()
    matched: list[tuple[dict, Path | str]] = []
    unmatched: list[dict] = []

    for track in spotify_tracks:
        title = normalize(track.get("title", ""))
        artist = normalize(track.get("artist", ""))
        target = f"{artist} {title}".strip()
        best_key = None
        best_score = 0.0
        for key in local_index:
            if key in used:
                continue
            if title and title in key and (not artist or artist in key):
                score = 1.0
            else:
                score = max(
                    SequenceMatcher(None, target, key).ratio(),
                    SequenceMatcher(None, title, key).ratio() if title else 0.0,
                )
            if score > best_score:
                best_key, best_score = key, score
        if best_key is not None and best_score >= 0.72:
            used.add(best_key)
            matched.append((track, local_index[best_key]))
        else:
            unmatched.append(track)
    return matched, unmatched


def fetch_spotify_tracks(
    playlist: str | None = None,
    liked: bool = False,
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str = "http://127.0.0.1:8888/callback",
    cache_path: str = ".spotify_token_cache",
) -> list[dict]:
    """Fetch metadata only. Requires `pip install spotipy` and user auth."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Spotify import needs spotipy: pip install 'moodmix-env[spotify]'"
        ) from exc

    scope = "playlist-read-private,playlist-read-collaborative,user-library-read"
    auth = SpotifyOAuth(
        client_id=client_id, client_secret=client_secret,
        redirect_uri=redirect_uri, scope=scope, cache_path=cache_path,
        open_browser=True,
    )
    sp = spotipy.Spotify(auth_manager=auth)

    records: list[dict] = []
    if liked:
        offset = 0
        while True:
            page = sp.current_user_saved_tracks(limit=50, offset=offset)
            records.extend(track_record(item) for item in page["items"])
            if not page.get("next"):
                break
            offset += 50
        return records

    if not playlist:
        raise ValueError("provide --playlist or --liked")
    playlist_id = playlist_id_from_url(playlist)
    offset = 0
    while True:
        page = sp.playlist_items(
            playlist_id, additional_types=("track",), limit=100, offset=offset
        )
        records.extend(track_record(item) for item in page["items"]
                       if (item.get("track") or {}).get("id"))
        if not page.get("next"):
            break
        offset += 100
    return records
