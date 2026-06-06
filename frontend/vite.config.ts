/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Two build targets via `--mode`: `demo` (static, reads public/demo-run.json) and `app`
// (reads VITE_API_BASE). Tests run under jsdom with the React Flow mocks in src/test/setup.ts.
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
