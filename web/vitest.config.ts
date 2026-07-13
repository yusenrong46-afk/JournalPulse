import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.ts"],
    setupFiles: ["./tests/setup.ts"],
    restoreMocks: true,
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, ".") },
  },
});
