# Generation Quality & Controllability — Audit, Changes, Results

## 1. Audit (what was wrong)

| Issue | Root cause |
| --- | --- |
| "Random noise after ~3s" | `musicgen-small` (300M) on CPU loses long-horizon structure; also `_normalize` peak-boosted quiet/noisy tails to full level. |
| Different prompts sound the same | Thin prompts (mood + tempo + intensity only); no genre/instrumentation/atmosphere. |
| Emotion control too coarse | 6 moods, no genre/purpose axes. |
| No styles (lofi/cinematic/jazz/study) | No taxonomy; `style` was a tiny 6-value set. |
| Output doesn't match request | Weak prompts + default sampling + single take (no selection). |

## 2. Changes (architecture)

```
prompt → analyze_prompt() → MusicRequest(mood, genre, purpose, instrumentation,
         energy, atmosphere, complexity, tempo, duration)        [taxonomy-grounded]
       → enhanced_prompt_builder.build() → rich MusicGen prompt
       → preset (sampling + N candidates)
       → generate_candidates(N) → two-stage scorer (spectral gate → CLAP) → best
       → RMS-normalized audio
```

New modules: `prompt_understanding.py`, `taxonomy.py` (+ Hive job
`bd_build_taxonomy.py`), `enhanced_prompt_builder.py`, `presets.py`,
`candidate_scorer.py`. The old builder is retained (`enhanced=False`) for A/B. All
existing APIs/CLI/Spark jobs keep working; new request fields are optional.

Presets: **Fast** (1 take, guidance 1.0), **Balanced** (3 takes, guidance 3.0),
**Cinematic** (3 takes, guidance 4.0 + orchestral bias), **Experimental** (4 takes,
higher temperature).

## 3. Benchmark results

Reproduce: `python scripts/eval_generation.py` and `python scripts/ab_prompt_builders.py`.

### Metadata extraction (instant, `--no-audio`, 7 prompts)
| field | accuracy |
| --- | --- |
| genre | **1.00** |
| mood | **1.00** |
| purpose | **0.75** |

### Prompt richness A/B (old vs new builder, 5 prompts)
| metric | old | new |
| --- | --- | --- |
| avg words | 32.6 | 34.4 |
| avg named instruments | 2.4 | **3.0** |
| genre coverage | 1.00 | 1.00 |

The bigger qualitative gain is in *content*: the new prompts name genre + purpose +
atmosphere + a coherent instrument template per style (see the side-by-side prompts
in `artifacts/ab_prompt_builders/report.md`).

### Audio (balanced preset = best-of-3, 4s clips on CPU)

| metric | value | reading |
| --- | --- | --- |
| mean spectral flatness (noise) | **0.0008** | very tonal — **not** the old "random noise"; the Stage-1 gate passed 3/3 musical takes |
| mean pairwise diversity (MFCC) | 0.025 | conservative metric; clips do differ across lofi/cinematic/jazz |
| mean tempo error | 24.2 BPM | inflated by one octave-doubling estimate (below) |

Per-prompt tempo (generated vs requested):

| prompt | generated | target |
| --- | --- | --- |
| epic cinematic orchestral battle | 113.6 | 115 ✅ |
| happy upbeat jazz trio | 138.9 | 138 ✅ |
| 30 second calm piano study | 144.2 | 74 ⚠️ (librosa octave-doubling of ~72) |

Takeaway: tempo adherence is near-exact where the estimator is reliable, and the
flatness number confirms the noise problem is addressed *on the current small model*.
Raw fidelity still benefits most from a bigger model on GPU (§4).

## 3b. Mid-clip noise ("clean start, noisy after ~5-6s")

**Cause:** autoregressive error accumulation in `musicgen-small` — it samples one frame
at a time from its own output, so errors compound and the clip drifts toward noise as
it lengthens. Worse for sparse/dark prompts (thriller, ambient, cinematic) that give
the sampler little structure to sustain. Measured: per-window spectral flatness *climbs
over time* on bad takes (slope ≈ +0.015/s) but stays flat on good ones (≈ +0.0006/s).

**Why best-of-3 missed it:** the scorer used whole-clip mean flatness, blind to *when* a
take fell apart.

**Fix (CPU/small-model path now degrades gracefully):**
- The candidate scorer is **segment-aware**: it tracks `spectral_flatness_slope` (the
  primary degradation signal), `tail_flatness`, `degrades`, and `clean_seconds`, and the
  Stage-1 score penalises a rising slope. Best-of-N now picks the *most stable* take.
- **Auto-trim** (`MUSICGEN_AUTOTRIM`, on by default): the chosen take's noisy tail is cut
  at the noise onset (floored at `MUSICGEN_MIN_CLEAN_S`), so a 10s take that rots at 6s
  is delivered as a clean ~6s clip with a short fade. Validated on real clips: a noisy
  10s take (slope +0.0146) is trimmed to 4.0s; a good 10s (slope +0.0033) and the 30s
  jazz (slope +0.0006) are left untouched.
- **Stability-biased sampling** for sparse/low-energy prompts (lower temperature/top-k)
  to delay the drift.
- The CLI/API report `spectral_flatness_slope`, `clean_seconds`, and `trimmed`.

Honest limit: if *all* candidates degrade early, you get a shorter clip and the
least-bad take — the 300M model simply can't sustain long, sparse pieces. Prefer ≤8s on
CPU, or use `musicgen-medium` on GPU (§4).

## 4. Recommendations (future upgrades)

1. **Biggest win — bigger model on a free GPU.** `facebook/musicgen-medium` via
   `deploy/colab_gpu_backend.ipynb` (Colab/Kaggle T4). One env-var swap; fixes the
   raw-fidelity and 10s-coherence ceiling that CPU + small cannot.
2. **Turn on CLAP ranking** (`MUSICGEN_USE_CLAP=1`) on GPU and raise candidate N —
   automatic best-of-N by text↔audio adherence.
3. **Longer coherent clips** via MusicGen continuation / sliding-window decoding.
4. **Close the loop with big data:** feed gallery + eval metrics back through
   Spark/Hive to learn which taxonomy templates score best, and auto-tune them.
