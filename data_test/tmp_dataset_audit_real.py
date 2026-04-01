import json
import re
import pandas as pd

spotify_path = r"C:\Users\Atharva Srivastava\.cache\kagglehub\datasets\devdope\900k-spotify\versions\3\spotify_dataset.csv"
genius_path = r"C:\Users\Atharva Srivastava\.cache\kagglehub\datasets\carlosgdcj\genius-song-lyrics-with-language-information\versions\1\song_lyrics.csv"


def norm_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick(df, candidates):
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

# Read full (memory-conscious type inference)
spotify = pd.read_csv(spotify_path, low_memory=False)
genius = pd.read_csv(genius_path, low_memory=False)

report = {}
report["spotify_shape"] = [int(spotify.shape[0]), int(spotify.shape[1])]
report["genius_shape"] = [int(genius.shape[0]), int(genius.shape[1])]
report["spotify_columns"] = spotify.columns.tolist()
report["genius_columns"] = genius.columns.tolist()

s_track = pick(spotify, ["track_name","name","song_name","title","track","song"])
s_artist = pick(spotify, ["artist_name","artists","artist","artist_names"])
s_album = pick(spotify, ["album_name","album","release_name"])
s_year = pick(spotify, ["year","release_year","release_date","date"])


g_track = pick(genius, ["title","song","song_title","track_name","name"])
g_artist = pick(genius, ["artist","artist_name","artists"])
g_album = pick(genius, ["album","album_name"])
g_year = pick(genius, ["year","release_year","release_date","date"])
g_lyrics = pick(genius, ["lyrics","lyric","text"])
g_lang = pick(genius, ["language","lang"])

report["join_keys"] = {
    "spotify": {"track": s_track, "artist": s_artist, "album": s_album, "year": s_year},
    "genius": {"track": g_track, "artist": g_artist, "album": g_album, "year": g_year, "lyrics": g_lyrics, "language": g_lang}
}

key_audio = ["danceability","energy","valence","tempo","loudness","speechiness","acousticness","instrumentalness","liveness","mode","key","duration_ms"]
report["missing_rates_pct"] = {
    "spotify": missing_rate(spotify, [c for c in [s_track,s_artist,s_album,s_year] if c] + key_audio),
    "genius": missing_rate(genius, [c for c in [g_track,g_artist,g_album,g_year,g_lyrics,g_lang] if c])
}

if s_track and s_artist:
    s = spotify[[s_artist, s_track]].copy()
    s["norm_key"] = s[s_artist].map(norm_text) + "|" + s[s_track].map(norm_text)
    s = s[s["norm_key"] != "|"]
    dup = int(s.duplicated("norm_key").sum())
    report["spotify_dedup"] = {
        "rows_considered": int(len(s)),
        "duplicate_rows": dup,
        "duplicate_rate_pct": round(dup / max(len(s),1) * 100, 3),
        "unique_norm_keys": int(s["norm_key"].nunique())
    }

if g_track and g_artist:
    g = genius[[g_artist, g_track] + ([g_lyrics] if g_lyrics else [])].copy()
    g["norm_key"] = g[g_artist].map(norm_text) + "|" + g[g_track].map(norm_text)
    g = g[g["norm_key"] != "|"]
    dup = int(g.duplicated("norm_key").sum())
    report["genius_dedup"] = {
        "rows_considered": int(len(g)),
        "duplicate_rows": dup,
        "duplicate_rate_pct": round(dup / max(len(g),1) * 100, 3),
        "unique_norm_keys": int(g["norm_key"].nunique())
    }

# High-confidence merged subset (strict exact normalized key, lyrics non-null)
if s_track and s_artist and g_track and g_artist:
    s = spotify[[s_artist, s_track]].copy()
    s["norm_key"] = s[s_artist].map(norm_text) + "|" + s[s_track].map(norm_text)
    s = s[s["norm_key"] != "|"]
    s = s.drop_duplicates("norm_key")

    g_cols = [g_artist, g_track] + ([g_lyrics] if g_lyrics else []) + ([g_lang] if g_lang else [])
    g = genius[g_cols].copy()
    g["norm_key"] = g[g_artist].map(norm_text) + "|" + g[g_track].map(norm_text)
    g = g[g["norm_key"] != "|"]
    if g_lyrics:
        g = g[g[g_lyrics].notna()]
    g = g.drop_duplicates("norm_key")

    merged = s.merge(g[["norm_key"] + ([g_lyrics] if g_lyrics else []) + ([g_lang] if g_lang else [])], on="norm_key", how="left")

    if g_lyrics:
        matched_mask = merged[g_lyrics].notna()
    else:
        matched_mask = merged["norm_key"].isin(set(g["norm_key"]))

    matched = int(matched_mask.sum())
    total = int(len(merged))
    high_conf = merged[matched_mask].copy()

    report["merge_quality_strict_norm_artist_title"] = {
        "spotify_unique_norm_keys": int(len(s)),
        "genius_unique_norm_keys": int(len(g)),
        "matched_keys": matched,
        "match_rate_pct": round(matched / max(total,1) * 100, 3),
        "unmatched_keys": int(total - matched),
        "high_conf_subset_rows": int(len(high_conf))
    }

    if g_lang and g_lang in high_conf.columns:
        lang_dist = high_conf[g_lang].fillna("unknown").astype(str).str.lower().value_counts().head(10)
        report["high_conf_top_languages"] = {str(k): int(v) for k, v in lang_dist.items()}

print(json.dumps(report, indent=2))
