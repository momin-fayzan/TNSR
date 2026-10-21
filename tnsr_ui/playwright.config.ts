import { existsSync } from "node:fs";

import { defineConfig } from "@playwright/test";

// Use a preinstalled Chromium when provided (e.g. CI images); otherwise Playwright's own
// browser (`npx playwright install chromium`).
const candidates = [process.env.TNSR_E2E_CHROMIUM, "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"];
const executablePath = candidates.find((p) => p && existsSync(p));

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.TNSR_E2E_URL ?? "http://127.0.0.1:8765",
    trace: "retain-on-failure",
    launchOptions: executablePath ? { executablePath } : {},
  },
});
