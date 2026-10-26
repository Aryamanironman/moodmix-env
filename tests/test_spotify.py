"""Tests for Spotify metadata matching (no network)."""

from __future__ import annotations

from moodmix_env.spotify import (
    match_tracks,
    normalize,
    playlist_id_from_url,
    track_record,
)


def test_normalize_strips_brackets_and_featuring():
    assert normalize("Song (feat. Someone) [Remastered]") == "song"
    assert normalize("Artist - Track!") == "artist track"


def test_playlist_id_from_url():
    assert playlist_id_from_url(
        "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=x"
    ) == "37i9dQZF1DXcBWIGoYBM5M"
    assert playlist_id_from_url("spotify:playlist:abc123") == "abc123"
    assert playlist_id_from_url("abc123") == "abc123"


def test_track_record_extracts_fields():
    item = {
        "track": {
            "id": "t1",
            "name": "Night Drive",
            "artists": [{"name": "Nova"}, {"name": "Guest"}],
            "album": {"name": "After Hours"},
            "duration_ms": 210000,
            "external_ids": {"isrc": "GB1234567890"},
            "preview_url": None,
        }
    }
    record = track_record(item)
    assert record["id"] == "t1"
    assert record["title"] == "Night Drive"
    assert record["artist"] == "Nova, Guest"
    assert record["isrc"] == "GB1234567890"


def test_match_tracks_by_title_and_filename():
    spotify = [
        {"id": "1", "title": "Night Drive", "artist": "Nova"},
        {"id": "2", "title": "Unknown Song", "artist": "Nobody"},
    ]
    local = {
        "nova night drive": "Nova - Night Drive.mp3",
        "other track": "Other - Track.mp3",
    }
    matched, unmatched = match_tracks(spotify, local)
    assert matched[0][0]["id"] == "1"
    assert matched[0][1] == "Nova - Night Drive.mp3"
    assert unmatched[0]["id"] == "2"
