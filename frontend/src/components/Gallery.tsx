import { useEffect, useState } from "react";
import { api, type Track } from "../api";
import { moodStyle } from "../lib/moods";
import { TrackCard } from "./TrackCard";

export function Gallery({ refreshKey }: { refreshKey: number }) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [moods, setMoods] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([api.gallery({ limit: 60, mood: active ?? undefined }), api.moods()])
      .then(([page, m]) => {
        if (cancelled) return;
        setTracks(page.items);
        setMoods(m.moods);
      })
      .catch(() => {
        if (!cancelled) setTracks([]);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [refreshKey, active]);

  return (
    <div className="flex flex-col gap-6">
      {/* Mood filter */}
      {moods.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <FilterPill label="All" active={active === null} onClick={() => setActive(null)} />
          {moods.map((m) => (
            <FilterPill
              key={m}
              label={moodStyle(m).label}
              color={moodStyle(m).color}
              active={active === m}
              onClick={() => setActive(active === m ? null : m)}
            />
          ))}
        </div>
      )}

      {loading ? (
        <SkeletonGrid />
      ) : tracks.length === 0 ? (
        <EmptyState filtered={active !== null} />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {tracks.map((t) => (
            <TrackCard key={t.id} track={t} />
          ))}
        </div>
      )}
    </div>
  );
}

function FilterPill({
  label,
  color,
  active,
  onClick,
}: {
  label: string;
  color?: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="rounded-full border px-3.5 py-1.5 font-sans text-sm transition-colors"
      style={{
        color: active ? "var(--color-accent-fg)" : "var(--color-muted)",
        backgroundColor: active ? (color ?? "var(--color-accent)") : "var(--color-surface)",
        borderColor: active ? "transparent" : "var(--color-border)",
      }}
    >
      {label}
    </button>
  );
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          className="h-40 animate-pulse rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)]"
        />
      ))}
    </div>
  );
}

function EmptyState({ filtered }: { filtered: boolean }) {
  return (
    <div className="rounded-[var(--radius-card)] border border-dashed border-[var(--color-border)] bg-[var(--color-surface)]/40 p-12 text-center">
      <p className="font-display text-lg text-[var(--color-text)]">
        {filtered ? "No tunes in this mood yet" : "The gallery is quiet"}
      </p>
      <p className="mt-1 font-sans text-sm text-[var(--color-muted)]">
        {filtered ? "Try another mood, or create one." : "Be the first — head to Create and describe a feeling."}
      </p>
    </div>
  );
}
