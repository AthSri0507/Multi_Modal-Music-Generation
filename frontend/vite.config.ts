import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In dev, proxy API + audio calls to the FastAPI backend on :8000.
// In production the same FastAPI process serves this built app, so relative
// "/api/..." URLs work unchanged (that is what lets us deploy to a single Space).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
