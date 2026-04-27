import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const agentApiTarget = process.env.VITE_AGENT_API_PROXY_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: agentApiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "") || "/",
      },
    },
  },
});
