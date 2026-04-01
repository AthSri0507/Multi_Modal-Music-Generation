import json
import re
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


def norm_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def missing_rate(df, cols):
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = round(float(df[c].isna().mean() * 100), 3)
    return out


audit = {"errors": []}

try:
    spotify = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "devdope/900k-spotify",
    )
    audit["spotify_loaded"] = True
except Exception as e:
    audit["spotify_loaded"] = False
    audit["errors"].append(f"spotify_load_failed: {e}")
    spotify = None

try:
    genius = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "carlosgdcj/genius-song-lyrics-with-language-information",
    )
    audit["genius_loaded"] = True
except Exception as e:
    audit["genius_loaded"] = False
    audit["errors"].append(f"genius_load_failed: {e}")
    genius = None

if spotify is not None:
    audit["spotify_shape"] = list(spotify.shape)
    audit["spotify_columns_sample"] = spotify.columns[:60].tolist()

if genius is not None:
    audit["genius_shape"] = list(genius.shape)
    audit["genius_columns_sample"] = genius.columns[:60].tolist()

if spotify is not None:
    s_track = first_present(spotify, ["track_name", "name", "song_name", "title", "track", "song"])
    s_artist = first_present(spotify, ["artist_name", "artists", "artist", "artist_names"])
    s_album = first_present(spotify, ["album_name", "album", "release_name"])
    s_year = first_present(spotify, ["year", "release_year", "release_date", "date"])

    audit["spotify_join_keys"] = {
        "track": s_track,
        "artist": s_artist,
        "album": s_album,
        "year": s_year,
    }

    key_audio_candidates = [
        "danceability","energy","valence","tempo","loudness","speechiness",
        "acousticness","instrumentalness","liveness","mode","key","duration_ms"
    ]
    key_cols = [c for c in [s_track, s_artist, s_album, s_year] if c]
    audit["spotify_missing_rates_pct"] = missing_rate(spotify, key_cols + key_audio_candidates)

    if s_track and s_artist:
        tmp = spotify[[s_track, s_artist]].copy()
        tmp["_k"] = tmp[s_artist].map(norm_text) + "|" + tmp[s_track].map(norm_text)
        dup_count = int(tmp.duplicated("_k").sum())
        audit["spotify_dedup"] = {
            "rows": int(len(tmp)),
            "duplicate_rows_on_norm_artist_title": dup_count,
            "duplicate_rate_pct": round(dup_count / max(len(tmp), 1) * 100, 3),
            "unique_keys": int(tmp["_k"].nunique())
        }

if genius is not None:
    g_track = first_present(genius, ["title", "song", "song_title", "track_name", "name"])
    g_artist = first_present(genius, ["artist", "artist_name", "artists"])
    g_album = first_present(genius, ["album", "album_name"])
    g_year = first_present(genius, ["year", "release_year", "release_date", "date"])
    g_lyrics = first_present(genius, ["lyrics", "lyric", "text"])
    g_lang = first_present(genius, ["language", "lang"])

    audit["genius_join_keys"] = {
        "track": g_track,
        "artist": g_artist,
        "album": g_album,
        "year": g_year,
        "lyrics": g_lyrics,
        "language": g_lang,
    }

    key_cols = [c for c in [g_track, g_artist, g_album, g_year, g_lyrics, g_lang] if c]
    audit["genius_missing_rates_pct"] = missing_rate(genius, key_cols)

    if g_track and g_artist:
        tmp = genius[[g_track, g_artist]].copy()
        tmp["_k"] = tmp[g_artist].map(norm_text) + "|" + tmp[g_track].map(norm_text)
        dup_count = int(tmp.duplicated("_k").sum())
        audit["genius_dedup"] = {
            "rows": int(len(tmp)),
            "duplicate_rows_on_norm_artist_title": dup_count,
            "duplicate_rate_pct": round(dup_count / max(len(tmp), 1) * 100, 3),
            "unique_keys": int(tmp["_k"].nunique())
        }

if spotify is not None and genius is not None:
    s_track = audit.get("spotify_join_keys", {}).get("track")
    s_artist = audit.get("spotify_join_keys", {}).get("artist")
    g_track = audit.get("genius_join_keys", {}).get("track")
    g_artist = audit.get("genius_join_keys", {}).get("artist")
    g_lyrics = audit.get("genius_join_keys", {}).get("lyrics")

    if s_track and s_artist and g_track and g_artist:
        s = spotify[[s_artist, s_track]].copy()
        g_cols = [g_artist, g_track] + ([g_lyrics] if g_lyrics else [])
        g = genius[g_cols].copy()

        s["norm_key"] = s[s_artist].map(norm_text) + "|" + s[s_track].map(norm_text)
        g["norm_key"] = g[g_artist].map(norm_text) + "|" + g[g_track].map(norm_text)

        s = s[s["norm_key"] != "|"]
        g = g[g["norm_key"] != "|"]

        if g_lyrics:
            g = g[g[g_lyrics].notna()]

        g = g.drop_duplicates("norm_key")
        s = s.drop_duplicates("norm_key")

        merged = s.merge(g[["norm_key"] + ([g_lyrics] if g_lyrics else [])], on="norm_key", how="left")
        matched = int(merged[g_lyrics].notna().sum()) if g_lyrics else int(merged["norm_key"].isin(set(g["norm_key"])).sum())
        total = int(len(merged))

        audit["merge_quality"] = {
            "spotify_unique_keys": int(len(s)),
            "genius_unique_keys": int(len(g)),
            "matched_keys": matched,
            "match_rate_pct": round(matched / max(total,1) * 100, 3),
            "unmatched_keys": int(total - matched)
        }

print(json.dumps(audit, indent=2))
