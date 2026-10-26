# Spotify songs

## What Spotify actually gives you

The Spotify Web API gives **metadata**: playlist and liked-song lists, titles,
artists, albums, ISRCs, durations, and sometimes a 30-second preview URL. It
does **not** give full audio files, and its terms do not allow downloading or
scraping tracks. This project does not do that.

The practical workflow is:

1. Pull playlist metadata from Spotify.
2. Match each entry to an audio file you already own.
3. Analyze those local files into the manifest the environment reads.

That keeps the training data legal and gives you full-length songs with real
beats and structure.

## 1. Create a Spotify app

1. Open <https://developer.spotify.com/dashboard> and create an app.
2. Add the redirect URI `http://127.0.0.1:8888/callback`.
3. Copy the client ID and client secret.

```bash
export SPOTIPY_CLIENT_ID=...
export SPOTIPY_CLIENT_SECRET=...
```

The first run opens a browser for user authorization and caches the token in
`.spotify_token_cache`. On a headless AWS box, authorize once on your laptop and
copy the cache file, or use an SSH tunnel so the browser callback can reach the
instance.

## 2. Install the optional dependencies

```bash
pip install -e ".[spotify,library]"
```

## 3. Match a playlist to local files

Put the files you own in a folder, then dry-run the matcher:

```bash
python scripts/import_spotify.py \
  --playlist "https://open.spotify.com/playlist/..." \
  --audio-dir ./crate \
  --dry-run
```

This writes `crate/spotify_match_report.json` and prints matched/unmatched
tracks. Matching is by normalized title + artist against file tags or filenames
(a threshold of 0.72). If a file is unmatched, rename it `Artist - Title.mp3`
or add title/artist tags.

For your liked songs:

```bash
python scripts/import_spotify.py --liked --audio-dir ./crate --dry-run
```

## 4. Analyze and build the manifest

```bash
python scripts/import_spotify.py \
  --playlist "https://open.spotify.com/playlist/..." \
  --audio-dir ./crate
```

This writes `crate/manifest.json` with BPM, key, per-beat low/mid/high energy,
sections and an inferred mood. To inspect a local folder without Spotify:

```bash
python scripts/analyze_library.py --audio-dir ./crate
```

To force one mood for every track:

```bash
python scripts/import_spotify.py ... --audio-dir ./crate --mood party
```

Mood is inferred from tempo, major/minor mode and energy. Spotify's audio
features endpoint is not available to new apps, which is why the analysis runs
on your own audio.

## 5. Train and demo on the crate

```bash
python scripts/train.py --data-dir ./crate --timesteps 1000000 --n-envs 16 --vec-env subproc
python scripts/serve_demo.py --data-dir ./crate --model models/ppo_1m.zip
```

The environment loads `crate/manifest.json`; `file` entries point at the audio
next to it (absolute paths also work).

## Limitations

- 30-second Spotify previews are not used; they are too short for a real
  transition and are not full tracks.
- ISRC is fetched but matching is title/artist based. If your tags include ISRC,
  a future matcher can use it directly.
- Audio analysis is CPU-heavy (`librosa`); expect a few seconds per track.
- If the Spotify app is in development mode, only allow-listed users can
  authorize it. Add yourself in the dashboard.
