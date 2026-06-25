import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` so the built bundle works when FastAPI serves it from `/`.
// In dev, proxy the API to the uvicorn backend so the app is same-origin.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 2000,
  },
});
