import type { GenerationSettings } from "./types";

// Mirrors tnsr_server/api/schemas.py::GenerationSettings bounds.
export const LIMITS = {
  max_new_tokens: { min: 1, max: 4096, step: 1, integer: true },
  temperature: { min: 0, max: 5, step: 0.05, integer: false },
  top_k: { min: 0, max: 65536, step: 1, integer: true },
  top_p: { min: 0.01, max: 1, step: 0.01, integer: false },
  repetition_penalty: { min: 1, max: 3, step: 0.05, integer: false },
} as const;

export type NumericKey = keyof typeof LIMITS;

export function validateField(key: NumericKey, raw: string): { value?: number; error?: string } {
  const lim = LIMITS[key];
  if (raw.trim() === "") return { error: "Required" };
  const n = Number(raw);
  if (!Number.isFinite(n)) return { error: "Must be a number" };
  if (lim.integer && !Number.isInteger(n)) return { error: "Must be a whole number" };
  if (n < lim.min || n > lim.max) return { error: `Must be between ${lim.min} and ${lim.max}` };
  if (key === "top_p" && n <= 0) return { error: "Must be greater than 0" };
  return { value: n };
}

export function validateSeed(raw: string): { value?: number | null; error?: string } {
  if (raw.trim() === "") return { value: null };
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0 || n > Number.MAX_SAFE_INTEGER) return { error: "Seed must be a non-negative integer" };
  return { value: n };
}

export function validateSettings(s: GenerationSettings): Partial<Record<keyof GenerationSettings, string>> {
  const errors: Partial<Record<keyof GenerationSettings, string>> = {};
  for (const key of Object.keys(LIMITS) as NumericKey[]) {
    const r = validateField(key, String(s[key]));
    if (r.error) errors[key] = r.error;
  }
  if (s.seed != null) {
    const r = validateSeed(String(s.seed));
    if (r.error) errors.seed = r.error;
  }
  return errors;
}
