// Typed client for the FastAPI backend. Uses a configurable base URL so the same
// build works behind the Vite dev proxy, served by FastAPI, or pointed at a remote
// free-GPU backend via VITE_API_BASE.

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const url = (path: string) => `${BASE}${path}`;

export interface Track {
  id: string;
  created_at: number;
  raw_prompt: string;
  music_prompt: string;
  mood: string | null;
  style: string | null;
  genre: string | null;
  purpose: string | null;
  preset: string | null;
  instrumentation: string | null;
  tempo_bpm: number | null;
  intensity: number | null;
  duration_s: number | null;
  audio_url: string;
  latency_s?: number;
  num_candidates?: number;
  trimmed?: boolean;
  clean_seconds?: number | null;
  analysis?: Record<string, unknown> | null;
}

export interface GalleryPage {
  items: Track[];
  total: number;
  limit: number;
  offset: number;
}

export interface Health {
  status: string;
  model: string;
  device: string;
  default_fast: boolean;
}

export interface Preset {
  name: string;
  display: string;
  description: string;
  candidates: number;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// Resolve a possibly-relative audio URL against the configured API base.
export const audioSrc = (audioUrl: string) => (audioUrl.startsWith("http") ? audioUrl : `${BASE}${audioUrl}`);

export const api = {
  health: () => fetch(url("/api/v1/health")).then(json<Health>),

  presets: () => fetch(url("/api/v1/presets")).then(json<{ presets: Preset[] }>),

  generate: (prompt: string, opts: { duration?: number; seed?: number; preset?: string } = {}) =>
    fetch(url("/api/v1/music/generate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, duration: opts.duration, seed: opts.seed, preset: opts.preset }),
    }).then(json<Track>),

  gallery: (opts: { limit?: number; offset?: number; mood?: string } = {}) => {
    const p = new URLSearchParams();
    if (opts.limit) p.set("limit", String(opts.limit));
    if (opts.offset) p.set("offset", String(opts.offset));
    if (opts.mood) p.set("mood", opts.mood);
    return fetch(url(`/api/v1/gallery?${p.toString()}`)).then(json<GalleryPage>);
  },

  moods: () => fetch(url("/api/v1/moods")).then(json<{ moods: string[] }>),
};
