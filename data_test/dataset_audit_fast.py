import json
import re
import pandas as pd
import sys

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

report = {}

# Load only first 50K rows for speed
print("Loading Spotify (first 50K rows)...", file=sys.stderr, flush=True)
spotify = pd.read_csv(spotify_path, nrows=50000, low_memory=False)
print(f"✓ Loaded {len(spotify)} rows, {len(spotify.columns)} cols", file=sys.stderr, flush=True)

print("Loading Genius (first 50K rows)...", file=sys.stderr, flush=True)
genius = pd.read_csv(genius_path, nrows=50000, low_memory=False)
print(f"✓ Loaded {len(genius)} rows, {len(genius.columns)} cols", file=sys.stderr, flush=True)

report["note"] = "Audit based on first 50K rows of each dataset (sampling for speed)"
report["spotify_columns"] = spotify.columns.tolist()
report["genius_columns"] = genius.columns.tolist()

# Find join keys
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

report["join_keys_found"] = {
    "spotify": {"track": s_track, "artist": s_artist, "album": s_album, "year": s_year},
    "genius": {"track": g_track, "artist": g_artist, "album": g_album, "year": g_year, "lyrics": g_lyrics, "language": g_lang}
}

# Missing rates
key_audio = ["danceability","energy","valence","tempo","loudness","speechiness","acousticness","instrumentalness","liveness","mode","key","duration_ms"]
all_keys_s = [c for c in [s_track,s_artist,s_album,s_year] if c] + key_audio
all_keys_g = [c for c in [g_track,g_artist,g_album,g_year,g_lyrics,g_lang] if c]

report["missing_rates_pct_spotify"] = {}
for c in all_keys_s:
    if c in spotify.columns:
        report["missing_rates_pct_spotify"][c] = round(spotify[c].isna().mean() * 100, 2)

report["missing_rates_pct_genius"] = {}
for c in all_keys_g:
    if c in genius.columns:
        report["missing_rates_pct_genius"][c] = round(genius[c].isna().mean() * 100, 2)

# Dedup analysis
if s_track and s_artist:
    print("Computing Spotify dedup...", file=sys.stderr, flush=True)
    s = spotify[[s_artist, s_track]].copy()
    s["norm_key"] = s[s_artist].map(norm_text) + "|" + s[s_track].map(norm_text)
    s = s[s["norm_key"] != "|"]
    dup = int(s.duplicated("norm_key").sum())
    total = int(len(s))
    report["spotify_dedup_analysis"] = {
        "rows_valid": total,
        "duplicate_rows": dup,
        "duplicate_rate_pct": round(dup / max(total, 1) * 100, 2) if total > 0 else 0,
        "unique_keys": int(s["norm_key"].nunique())
    }

if g_track and g_artist:
    print("Computing Genius dedup...", file=sys.stderr, flush=True)
    g = genius[[g_artist, g_track] + ([g_lyrics] if g_lyrics else [])].copy()
    g["norm_key"] = g[g_artist].map(norm_text) + "|" + g[g_track].map(norm_text)
    g = g[g["norm_key"] != "|"]
    if g_lyrics:
        g_with_lyrics = g[g[g_lyrics].notna()]
    else:
        g_with_lyrics = g
    
    dup = int(g.duplicated("norm_key").sum())
    total = int(len(g))
    dup_lyrics = int(g_with_lyrics.duplicated("norm_key").sum())
    total_lyrics = int(len(g_with_lyrics))
    
    report["genius_dedup_analysis"] = {
        "rows_valid": total,
        "duplicate_rows": dup,
        "duplicate_rate_pct": round(dup / max(total, 1) * 100, 2) if total > 0 else 0,
        "unique_keys": int(g["norm_key"].nunique()),
        "rows_with_lyrics": total_lyrics,
        "duplicates_with_lyrics": dup_lyrics,
        "duplicate_rate_with_lyrics_pct": round(dup_lyrics / max(total_lyrics, 1) * 100, 2) if total_lyrics > 0 else 0
    }

# Merge quality
if s_track and s_artist and g_track and g_artist:
    print("Computing merge quality...", file=sys.stderr, flush=True)
    
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
    
    report["merge_quality"] = {
        "spotify_unique_keys": int(len(s)),
        "genius_unique_keys": int(len(g)),
        "matched_rows": matched,
        "match_rate_pct": round(matched / max(total, 1) * 100, 2) if total > 0 else 0,
        "unmatched_rows": int(total - matched),
        "high_confidence_rows": int(len(high_conf))
    }
    
    if g_lang and g_lang in high_conf.columns:
        lang_dist = high_conf[g_lang].fillna("unknown").astype(str).str.lower().value_counts().head(10)
        report["top_languages_in_matched_subset"] = {str(k): int(v) for k, v in lang_dist.items()}

print("\n=== DATASET AUDIT REPORT ===\n")
print(json.dumps(report, indent=2))
