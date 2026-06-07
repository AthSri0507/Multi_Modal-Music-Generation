# Emotion Music Studio — Frontend

React + Vite + Tailwind v4 single-page app for the text-to-music backend. Built
following the [Impeccable](https://github.com/pbakaus/impeccable) UI principles
(OKLCH color, tinted neutrals, a distinctive type pairing, restrained motion,
real focus / empty / loading states).

## Develop

```bash
npm install
npm run dev      # http://localhost:5173  (proxies /api -> http://localhost:8000)
```

Run the backend separately: `uvicorn src.api.app:app --port 8000 --reload`.

## Build

```bash
npm run build    # -> dist/ ; FastAPI serves this at / in production
```

## Structure

- `src/api.ts` — typed client for the FastAPI endpoints (relative URLs).
- `src/App.tsx` — shell: hero, Create/Gallery tabs, device (CPU/GPU) badge.
- `src/components/Generator.tsx` — prompt composer, length presets, result player.
- `src/components/Gallery.tsx` — browse + mood filter + empty/skeleton states.
- `src/components/{AudioPlayer,TrackCard,MoodBadge}.tsx` — building blocks.
- `src/index.css` — design tokens (`@theme`, OKLCH) + base styles.
