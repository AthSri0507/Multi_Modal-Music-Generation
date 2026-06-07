import { useEffect, useRef, useState } from "react";
import { api, audioSrc, type Preset, type Track } from "../api";
import { moodStyle } from "../lib/moods";
import { AudioPlayer } from "./AudioPlayer";
import { MoodBadge } from "./MoodBadge";

const EXAMPLES = [
  "30 second calm piano study music",
  "epic cinematic orchestral battle theme",
  "happy upbeat jazz trio",
  "ambient meditation music with soft pads",
  "dark suspenseful thriller soundtrack",
];

const DURATIONS: { label: string; value?: number }[] = [
  { label: "Auto", value: undefined },
  { label: "5s", value: 5 },
  { label: "10s", value: 10 },
  { label: "20s", value: 20 },
  { label: "30s", value: 30 },
];

const FALLBACK_PRESETS: Preset[] = [
  { name: "fast", display: "Fast", description: "Quickest, single take, lower fidelity.", candidates: 1 },
  { name: "balanced", display: "Balanced", description: "Full guidance, 3 candidates, best of. Recommended.", candidates: 3 },
  { name: "cinematic", display: "Cinematic", description: "Strong guidance + orchestral bias.", candidates: 3 },
  { name: "experimental", display: "Experimental", description: "Higher temperature, more candidates.", candidates: 4 },
];

function EqLoader() {
  return (
    <div className="flex items-end gap-1" aria-hidden>
      {[0, 1, 2, 3, 4].map((i) => (
        <span key={i} className="eq-bar w-1.5 rounded-full bg-[var(--color-accent-fg)]" style={{ height: 18, animationDelay: `${i * 0.12}s` }} />
      ))}
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2.5 py-0.5 font-sans text-xs text-[var(--color-muted)]">
      {children}
    </span>
  );
}

export function Generator({ device, onGenerated }: { device: string; onGenerated: () => void }) {
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState<number | undefined>(undefined);
  const [presets, setPresets] = useState<Preset[]>(FALLBACK_PRESETS);
  const [preset, setPreset] = useState("balanced");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<Track | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    api.presets().then((p) => p.presets.length && setPresets(p.presets)).catch(() => {});
  }, []);

  useEffect(() => {
    if (loading) {
      const start = Date.now();
      timer.current = window.setInterval(() => setElapsed((Date.now() - start) / 1000), 200);
    } else if (timer.current) {
      window.clearInterval(timer.current);
    }
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [loading]);

  const activePreset = presets.find((p) => p.name === preset) ?? FALLBACK_PRESETS[1];

  const submit = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(0);
    try {
      const track = await api.generate(prompt.trim(), { duration, preset });
      setResult(track);
      onGenerated();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const slow = device !== "cuda";

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      {/* Composer */}
      <div className="rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5 sm:p-6">
        <label htmlFor="prompt" className="font-display text-sm font-medium text-[var(--color-muted)]">
          Describe the music you want
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submit();
          }}
          rows={3}
          placeholder="e.g. a warm lo-fi study beat with mellow piano and soft drums…"
          className="mt-2 w-full resize-none rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-2)] px-4 py-3 font-sans text-[var(--color-text)] placeholder:text-[var(--color-faint)] focus:border-[var(--color-accent)] focus:outline-none"
        />

        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setPrompt(ex)}
              className="rounded-full border border-[var(--color-border)] bg-[var(--color-bg-2)] px-3 py-1 font-sans text-xs text-[var(--color-muted)] transition-colors hover:border-[var(--color-accent)] hover:text-[var(--color-text)]"
            >
              {ex}
            </button>
          ))}
        </div>

        {/* Preset selector */}
        <div className="mt-5">
          <span className="font-sans text-xs text-[var(--color-faint)]">Style preset</span>
          <div className="mt-1.5 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {presets.map((p) => {
              const active = p.name === preset;
              return (
                <button
                  key={p.name}
                  onClick={() => setPreset(p.name)}
                  className={`rounded-xl border px-3 py-2 text-left transition-colors ${
                    active
                      ? "border-transparent bg-[var(--color-accent)] text-[var(--color-accent-fg)]"
                      : "border-[var(--color-border)] bg-[var(--color-bg-2)] text-[var(--color-muted)] hover:text-[var(--color-text)]"
                  }`}
                >
                  <span className="block font-display text-sm font-semibold">{p.display}</span>
                  <span className="block font-sans text-[10px] opacity-80">{p.candidates} take{p.candidates > 1 ? "s" : ""}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-2 font-sans text-xs text-[var(--color-faint)]">{activePreset.description}</p>
        </div>

        <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <span className="font-sans text-xs text-[var(--color-faint)]">Length</span>
            <div className="mt-1.5 inline-flex rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-2)] p-1">
              {DURATIONS.map((d) => {
                const active = d.value === duration;
                return (
                  <button
                    key={d.label}
                    onClick={() => setDuration(d.value)}
                    className={`rounded-lg px-3 py-1.5 font-sans text-sm transition-colors ${
                      active ? "bg-[var(--color-accent)] text-[var(--color-accent-fg)]" : "text-[var(--color-muted)] hover:text-[var(--color-text)]"
                    }`}
                  >
                    {d.label}
                  </button>
                );
              })}
            </div>
          </div>

          <button
            onClick={submit}
            disabled={loading || !prompt.trim()}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-[var(--color-accent)] px-6 font-display font-semibold text-[var(--color-accent-fg)] transition-all hover:bg-[var(--color-accent-strong)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? <EqLoader /> : "Generate"}
          </button>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="animate-rise rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)] p-6 text-center">
          <p className="font-display text-lg text-[var(--color-text)]">Composing {activePreset.candidates} take{activePreset.candidates > 1 ? "s" : ""}, keeping the best…</p>
          <p className="mt-1 font-sans text-sm text-[var(--color-muted)] tabular-nums">{elapsed.toFixed(1)}s elapsed</p>
          {slow && (
            <p className="mt-2 font-sans text-xs text-[var(--color-faint)]">
              On CPU each take is ~1 min; this preset renders {activePreset.candidates}. A GPU host makes it near-instant.
            </p>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div role="alert" className="rounded-[var(--radius-card)] border p-4 font-sans text-sm" style={{ color: "var(--color-mood-tense)", borderColor: "color-mix(in oklch, var(--color-mood-tense) 35%, transparent)", backgroundColor: "color-mix(in oklch, var(--color-mood-tense) 10%, transparent)" }}>
          {error}
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div className="animate-rise rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5 sm:p-6">
          <div className="flex items-center justify-between gap-3">
            <h3 className="font-display text-lg text-[var(--color-text)]">Your tune is ready</h3>
            <MoodBadge mood={result.mood} size="md" />
          </div>
          <p className="mt-1 font-sans text-sm text-[var(--color-muted)]">“{result.raw_prompt}”</p>

          <div className="mt-3 flex flex-wrap gap-2">
            {result.genre && <Chip>{result.genre}</Chip>}
            {result.purpose && <Chip>{result.purpose}</Chip>}
            {result.preset && <Chip>{result.preset}</Chip>}
            {result.num_candidates && result.num_candidates > 1 && <Chip>best of {result.num_candidates}</Chip>}
          </div>

          <div className="mt-5">
            <AudioPlayer src={audioSrc(result.audio_url)} accent={moodStyle(result.mood).color} />
          </div>

          <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 font-sans text-xs text-[var(--color-faint)]">
            {result.instrumentation && <span>instruments · <span className="text-[var(--color-muted)]">{result.instrumentation}</span></span>}
            {result.tempo_bpm != null && <span>tempo · <span className="text-[var(--color-muted)] tabular-nums">{Math.round(result.tempo_bpm)} BPM</span></span>}
            {result.duration_s != null && <span>length · <span className="text-[var(--color-muted)] tabular-nums">{Math.round(result.duration_s)}s</span></span>}
            {result.latency_s != null && <span>rendered in · <span className="text-[var(--color-muted)] tabular-nums">{result.latency_s}s</span></span>}
          </div>

          <details className="mt-4 font-sans text-xs text-[var(--color-faint)]">
            <summary className="cursor-pointer select-none hover:text-[var(--color-muted)]">Prompt sent to the model</summary>
            <p className="mt-2 leading-relaxed text-[var(--color-muted)]">{result.music_prompt}</p>
          </details>

          <p className="mt-4 font-sans text-xs text-[var(--color-faint)]">Saved to the gallery for everyone to hear.</p>
        </div>
      )}
    </div>
  );
}
