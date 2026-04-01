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

print("Loading data...", file=sys.stderr, flush=True)
spotify = pd.read_csv(spotify_path, nrows=50000, low_memory=False)
genius = pd.read_csv(genius_path, nrows=50000, low_memory=False)
print(f"✓ Spotify: {len(spotify)} rows, {len(spotify.columns)} cols", file=sys.stderr, flush=True)
print(f"✓ Genius: {len(genius)} rows, {len(genius.columns)} cols", file=sys.stderr, flush=True)

report = {
    "note": "Audit on first 50K rows of each dataset",
    "spotify_columns": spotify.columns.tolist(),
    "genius_columns": genius.columns.tolist()
}

# Smart column mapping
s_track = "song" if "song" in spotify.columns else None
s_artist = "Artist(s)" if "Artist(s)" in spotify.columns else ("artist_name" if "artist_name" in spotify.columns else "artist" if "artist" in spotify.columns else None)
s_emotion = "emotion" if "emotion" in spotify.columns else None

g_track = "title" if "title" in genius.columns else "song_title" if "song_title" in genius.columns else None
g_artist = "artist" if "artist" in genius.columns else "artist_name" if "artist_name" in genius.columns else None
g_lyrics = "lyrics" if "lyrics" in genius.columns else "text" if "text" in genius.columns else None
g_lang = "language" if "language" in genius.columns else "language_cld3" if "language_cld3" in genius.columns else None

report["join_keys"] = {
    "spotify": {"track": s_track, "artist": s_artist, "emotion_label": s_emotion},
    "genius": {"track": g_track, "artist": g_artist, "lyrics": g_lyrics, "language": g_lang}
}

# Missing rates for key columns
report["missing_rates_pct"] = {
    "spotify": {},
    "genius": {}
}
for col in [s_track, s_artist, s_emotion]:
    if col and col in spotify.columns:
        report["missing_rates_pct"]["spotify"][col] = round(spotify[col].isna().mean() * 100, 2)

for col in [g_track, g_artist, g_lyrics, g_lang]:
    if col and col in genius.columns:
        report["missing_rates_pct"]["genius"][col] = round(genius[col].isna().mean() * 100, 2)

# Dedup analysis (Spotify)
if s_track and s_artist:
    print("Computing Spotify dedup...", file=sys.stderr, flush=True)
    s = spotify[[s_artist, s_track]].copy()
    s = s.dropna(subset=[s_artist, s_track])
    s["norm_key"] = s[s_artist].map(norm_text) + "|" + s[s_track].map(norm_text)
    s = s[s["norm_key"] != "|"]
    dup = int(s.duplicated("norm_key").sum())
    total = int(len(s))
    report["spotify_dedup"] = {
        "valid_rows": total,
        "duplicates": dup,
        "duplicate_rate_pct": round(dup / max(total, 1) * 100, 3),
        "unique_keys": int(s["norm_key"].nunique())
    }

# Dedup analysis (Genius)
if g_track and g_artist:
    print("Computing Genius dedup...", file=sys.stderr, flush=True)
    g = genius[[g_artist, g_track]].copy()
    g = g.dropna(subset=[g_artist, g_track])
    g["norm_key"] = g[g_artist].map(norm_text) + "|" + g[g_track].map(norm_text)
    g = g[g["norm_key"] != "|"]
    g_dup = int(g.duplicated("norm_key").sum())
    g_total = int(len(g))
    report["genius_dedup"] = {
        "valid_rows": g_total,
        "duplicates": g_dup,
        "duplicate_rate_pct": round(g_dup / max(g_total, 1) * 100, 3),
        "unique_keys": int(g["norm_key"].nunique())
    }

# Merge quality (strict normalized key, non-null lyrics in Genius)
if s_track and s_artist and g_track and g_artist:
    print("Computing merge quality...", file=sys.stderr, flush=True)
    
    s = spotify[[s_artist, s_track]].copy()
    s = s.dropna(subset=[s_artist, s_track])
    s["norm_key"] = s[s_artist].map(norm_text) + "|" + s[s_track].map(norm_text)
    s = s[s["norm_key"] != "|"]
    s = s.drop_duplicates("norm_key", keep="first")
    s_unique = int(len(s))
    
    g_cols = [g_artist, g_track]
    if g_lyrics:
        g_cols.append(g_lyrics)
    if g_lang:
        g_cols.append(g_lang)
    
    g = genius[g_cols].copy()
    g = g.dropna(subset=[g_artist, g_track])
    g["norm_key"] = g[g_artist].map(norm_text) + "|" + g[g_track].map(norm_text)
    g = g[g["norm_key"] != "|"]
    if g_lyrics:
        g = g[g[g_lyrics].notna()]
    g = g.drop_duplicates("norm_key", keep="first")
    g_unique = int(len(g))
    
    # Merge on normalized key
    merge_cols = ["norm_key"]
    if g_lyrics:
        merge_cols.append(g_lyrics)
    if g_lang:
        merge_cols.append(g_lang)
    
    merged = s.merge(g[merge_cols], on="norm_key", how="left")
    
    if g_lyrics:
        matched = int(merged[g_lyrics].notna().sum())
    else:
        matched = int(merged["norm_key"].isin(set(g["norm_key"])).sum())
    
    total = int(len(merged))
    
    report["merge_quality"] = {
        "spotify_unique_keys": s_unique,
        "genius_unique_keys": g_unique,
        "matched_rows": matched,
        "match_rate_pct": round(matched / max(total, 1) * 100, 2),
        "unmatched_rows": int(total - matched),
        "high_confidence_coverage_pct": round((matched / max(total, 1)) * 100, 2) if total > 0 else 0
    }

# Language distribution in matched high-confidence set
if matched > 0 and g_lang and g_lang in merged.columns:
    matched_rows = merged[merged[g_lyrics].notna()] if g_lyrics else merged
    if g_lang in matched_rows.columns:
        lang_dist = matched_rows[g_lang].fillna("unknown").astype(str).str.lower().value_counts().head(10)
        report["top_languages_matched"] = {str(k): int(v) for k, v in lang_dist.items()}

# Emotion label distribution in Spotify (if present)
if s_emotion and s_emotion in spotify.columns:
    emotion_dist = spotify[s_emotion].fillna("unknown").astype(str).str.lower().value_counts().head(10)
    report["emotion_distribution_spotify"] = {str(k): int(v) for k, v in emotion_dist.items()}

print("\n=== DATASET AUDIT REPORT ===\n")
print(json.dumps(report, indent=2))
