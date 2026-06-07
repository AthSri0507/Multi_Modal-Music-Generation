// Mood -> visual identity. Colors reference the OKLCH theme tokens in index.css.

export interface MoodStyle {
  label: string;
  color: string; // CSS var reference
  emoji: string;
}

const MOODS: Record<string, MoodStyle> = {
  happy: { label: "Happy", color: "var(--color-mood-happy)", emoji: "☀️" },
  calm: { label: "Calm", color: "var(--color-mood-calm)", emoji: "🌊" },
  sad: { label: "Sad", color: "var(--color-mood-sad)", emoji: "🌧️" },
  tense: { label: "Tense", color: "var(--color-mood-tense)", emoji: "⚡" },
  romantic: { label: "Romantic", color: "var(--color-mood-romantic)", emoji: "🌹" },
  neutral: { label: "Neutral", color: "var(--color-mood-neutral)", emoji: "◐" },
};

export function moodStyle(mood: string | null | undefined): MoodStyle {
  if (!mood) return MOODS.neutral;
  return MOODS[mood.toLowerCase()] ?? { label: mood, color: "var(--color-mood-neutral)", emoji: "♪" };
}

export const ALL_MOODS = Object.keys(MOODS);
