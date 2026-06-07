import { moodStyle } from "../lib/moods";

export function MoodBadge({ mood, size = "sm" }: { mood: string | null; size?: "sm" | "md" }) {
  const s = moodStyle(mood);
  const pad = size === "md" ? "px-3 py-1 text-sm" : "px-2.5 py-0.5 text-xs";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${pad}`}
      style={{
        color: s.color,
        backgroundColor: "color-mix(in oklch, " + s.color + " 14%, transparent)",
        border: "1px solid color-mix(in oklch, " + s.color + " 30%, transparent)",
      }}
    >
      <span aria-hidden>{s.emoji}</span>
      {s.label}
    </span>
  );
}
