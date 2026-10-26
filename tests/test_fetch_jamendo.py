"""Tests for the Jamendo crate fetcher. They never touch the network."""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_jamendo  # noqa: E402


def test_slug_is_filesystem_safe():
    assert fetch_jamendo._slug("Björk - Jóga / Live!") == "bjork-joga-live"
    assert fetch_jamendo._slug("!!!") == "track"


def test_search_tracks_sends_tags_and_vocal_filter(monkeypatch):
    captured: dict[str, str] = {}

    def fake_get_json(url, timeout, retries=3):
        captured["url"] = url
        return {"headers": {"status": "success"}, "results": [{"id": 1}]}

    monkeypatch.setattr(fetch_jamendo, "_get_json", fake_get_json)
    results = fetch_jamendo.search_tracks(
        api_base="https://api.example",
        client_id="abc",
        limit=5,
        offset=10,
        tags=fetch_jamendo.MOOD_TAGS["party"],
        vocal="vocal",
    )

    assert results == [{"id": 1}]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(captured["url"]).query)
    assert query["tags"] == [fetch_jamendo.MOOD_TAGS["party"]]
    assert query["vocalinstrumental"] == ["vocal"]
    assert query["limit"] == ["5"]
    assert query["offset"] == ["10"]
    assert "musicinfo" in query["include"][0]


def test_search_tracks_reports_api_errors(monkeypatch):
    monkeypatch.setattr(
        fetch_jamendo,
        "_get_json",
        lambda *args, **kwargs: {
            "headers": {"status": "failed", "error_message": "invalid client_id"},
            "results": [],
        },
    )
    with pytest.raises(SystemExit) as excinfo:
        fetch_jamendo.search_tracks(
            api_base="https://api.example", client_id="bad", limit=1, offset=0
        )
    assert "invalid client_id" in str(excinfo.value)


def test_main_requires_client_id(monkeypatch, tmp_path):
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["fetch_jamendo.py", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        fetch_jamendo.main()
    assert "devportal.jamendo.com" in str(excinfo.value)


def test_credit_keeps_licence_and_source():
    credit = fetch_jamendo._credit(
        {
            "id": 7,
            "name": "Night Drive",
            "artist_name": "Stub Artist",
            "album_name": "Stub EP",
            "duration": 24,
            "shareurl": "https://jamendo.example/7",
            "licenses": [{"url": "https://creativecommons.org/licenses/by/4.0/"}],
        }
    )
    assert credit["title"] == "Night Drive"
    assert credit["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert credit["source"] == "https://jamendo.example/7"
