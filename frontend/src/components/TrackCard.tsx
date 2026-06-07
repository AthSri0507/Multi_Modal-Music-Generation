import { audioSrc, type Track } from "../api";
import { moodStyle } from "../lib/moods";
import { AudioPlayer } from "./AudioPlayer";
import { MoodBadge } from "./MoodBadge";

export function TrackCard({ track }: { track: Track }) {
  const accent = moodStyle(track.mood).color;
  return (
    <article className="flex flex-col gap-4 rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5 transition-colors hover:border-[color-mix(in_oklch,var(--color-accent)_45%,var(--color-border))]">
      <div className="flex items-start justify-between gap-3">
        <p className="font-display text-base leading-snug text-[var(--color-text)]">
          “{track.raw_prompt}”
        </p>
        <MoodBadge mood={track.mood} />
      </div>

      {(track.genre || track.purpose) && (
        <div className="flex flex-wrap gap-1.5">
          {track.genre && (
            <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-0.5 font-sans text-[11px] text-[var(--color-muted)]">{track.genre}</span>
          )}
          {track.purpose && (
            <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-0.5 font-sans text-[11px] text-[var(--color-muted)]">{track.purpose}</span>
          )}
        </div>
      )}

      <AudioPlayer src={audioSrc(track.audio_url)} accent={accent} />

      <dl className="flex flex-wrap gap-x-5 gap-y-1 font-sans text-xs text-[var(--color-faint)]">
        {track.instrumentation && (
          <div className="flex gap-1.5">
            <dt className="text-[var(--color-faint)]">instruments</dt>
            <dd className="text-[var(--color-muted)]">{track.instrumentation}</dd>
          </div>
        )}
        {track.tempo_bpm != null && (
          <div className="flex gap-1.5">
            <dt>tempo</dt>
            <dd className="text-[var(--color-muted)] tabular-nums">{Math.round(track.tempo_bpm)} BPM</dd>
          </div>
        )}
        {track.duration_s != null && (
          <div className="flex gap-1.5">
            <dt>length</dt>
            <dd className="text-[var(--color-muted)] tabular-nums">{Math.round(track.duration_s)}s</dd>
          </div>
        )}
      </dl>
    </article>
  );
}
