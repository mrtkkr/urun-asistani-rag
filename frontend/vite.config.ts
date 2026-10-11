import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// /api requests go through the dev server to Django, so the browser sees one origin
// and no CORS setup is needed (K-12). Inside Docker the target is the `backend` service.
const backend = process.env.VITE_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Bind-mounted files on Docker Desktop for Windows do not emit file events.
    watch: { usePolling: true },
    // Keep the browser's Host header (localhost); Django only allows localhost/127.0.0.1.
    proxy: { "/api": { target: backend, changeOrigin: false } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});
