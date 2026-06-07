import { useEffect, useState } from "react";
import { api } from "./api";
import { Generator } from "./components/Generator";
import { Gallery } from "./components/Gallery";

type Tab = "create" | "gallery";

export default function App() {
  const [tab, setTab] = useState<Tab>("create");
  const [refreshKey, setRefreshKey] = useState(0);
  const [device, setDevice] = useState("cpu");

  useEffect(() => {
    api.health().then((h) => setDevice(h.device)).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen">
      {/* Hero / header */}
      <header className="aurora border-b border-[var(--color-border)]">
        <div className="mx-auto max-w-5xl px-5 pt-10 pb-8 sm:pt-14">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <Logo />
              <span className="font-display text-lg font-semibold tracking-tight">Aurora</span>
            </div>
            <DeviceBadge device={device} />
          </div>

          <div className="mt-10 max-w-2xl">
            <h1 className="text-balance font-display text-4xl font-bold leading-[1.05] sm:text-5xl">
              Turn a feeling into music.
            </h1>
            <p className="mt-4 max-w-xl text-pretty font-sans text-lg leading-relaxed text-[var(--color-muted)]">
              Describe a mood in plain words. Aurora reads the emotion and composes an
              original instrumental tune — no lyrics, just the feeling.
            </p>
          </div>

          {/* Tabs */}
          <nav className="mt-8 inline-flex rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/70 p-1 backdrop-blur">
            <TabButton active={tab === "create"} onClick={() => setTab("create")}>
              Create
            </TabButton>
            <TabButton active={tab === "gallery"} onClick={() => setTab("gallery")}>
              Gallery
            </TabButton>
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-5xl px-5 py-10">
        {tab === "create" ? (
          <Generator device={device} onGenerated={() => setRefreshKey((k) => k + 1)} />
        ) : (
          <Gallery refreshKey={refreshKey} />
        )}
      </main>

      <footer className="mx-auto max-w-5xl px-5 pb-10 font-sans text-xs text-[var(--color-faint)]">
        Emotion → music · powered by MusicGen, grounded by a Spark/Hive analysis of the
        Spotify corpus.
      </footer>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg px-5 py-2 font-display text-sm font-medium transition-colors ${
        active
          ? "bg-[var(--color-accent)] text-[var(--color-accent-fg)]"
          : "text-[var(--color-muted)] hover:text-[var(--color-text)]"
      }`}
    >
      {children}
    </button>
  );
}

function DeviceBadge({ device }: { device: string }) {
  const gpu = device === "cuda";
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-sans text-xs"
      style={{
        color: gpu ? "var(--color-mood-calm)" : "var(--color-muted)",
        borderColor: "var(--color-border)",
        backgroundColor: "var(--color-surface)",
      }}
      title={gpu ? "Running on GPU" : "Running on CPU"}
    >
      <span
        className="size-1.5 rounded-full"
        style={{ backgroundColor: gpu ? "var(--color-mood-calm)" : "var(--color-faint)" }}
      />
      {gpu ? "GPU" : "CPU"}
    </span>
  );
}

function Logo() {
  return (
    <span className="grid size-8 place-items-center rounded-lg" style={{ background: "var(--color-accent)" }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-fg)" strokeWidth="2.2" strokeLinecap="round">
        <path d="M9 18V5l10-2v13" />
        <circle cx="6" cy="18" r="3" fill="var(--color-accent-fg)" stroke="none" />
        <circle cx="16" cy="16" r="3" fill="var(--color-accent-fg)" stroke="none" />
      </svg>
    </span>
  );
}
