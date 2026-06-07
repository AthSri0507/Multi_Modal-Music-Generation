import { useEffect, useRef, useState } from "react";

function fmt(t: number): string {
  if (!isFinite(t)) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** Minimal, accessible audio player with a scrubbable progress bar. */
export function AudioPlayer({ src, accent = "var(--color-accent)" }: { src: string; accent?: string }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setCur(0);
  }, [src]);

  const toggle = () => {
    const a = ref.current;
    if (!a) return;
    if (a.paused) {
      a.play();
      setPlaying(true);
    } else {
      a.pause();
      setPlaying(false);
    }
  };

  const seek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const a = ref.current;
    if (!a) return;
    const t = Number(e.target.value);
    a.currentTime = t;
    setCur(t);
  };

  const pct = dur > 0 ? (cur / dur) * 100 : 0;

  return (
    <div className="flex items-center gap-3">
      <audio
        ref={ref}
        src={src}
        preload="metadata"
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration)}
        onTimeUpdate={(e) => setCur(e.currentTarget.currentTime)}
        onEnded={() => setPlaying(false)}
      />
      <button
        onClick={toggle}
        aria-label={playing ? "Pause" : "Play"}
        className="grid size-11 shrink-0 place-items-center rounded-full text-[var(--color-accent-fg)] transition-transform hover:scale-105 active:scale-95"
        style={{ backgroundColor: accent }}
      >
        {playing ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.54.84l10.79-6.86a1 1 0 0 0 0-1.68L9.54 4.3A1 1 0 0 0 8 5.14Z" /></svg>
        )}
      </button>

      <div className="min-w-0 flex-1">
        <input
          type="range"
          min={0}
          max={dur || 0}
          step={0.01}
          value={cur}
          onChange={seek}
          aria-label="Seek"
          className="h-1.5 w-full cursor-pointer appearance-none rounded-full"
          style={{
            background: `linear-gradient(to right, ${accent} ${pct}%, var(--color-surface-2) ${pct}%)`,
          }}
        />
        <div className="mt-1 flex justify-between font-sans text-xs tabular-nums text-[var(--color-faint)]">
          <span>{fmt(cur)}</span>
          <span>{fmt(dur)}</span>
        </div>
      </div>
    </div>
  );
}
