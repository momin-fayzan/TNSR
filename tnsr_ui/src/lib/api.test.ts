import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonResponse, mockFetch } from "../test/utils";
import { api, ApiError } from "./api";
import { formatDuration, formatParams, modelLabel } from "./format";

afterEach(() => vi.unstubAllGlobals());


describe("request errors", () => {
  it("formats pydantic validation errors", async () => {
    mockFetch({
      "PUT /api/settings": () =>
        jsonResponse({ detail: [{ loc: ["body", "generation", "top_k"], msg: "Input should be greater than or equal to 0" }] }, 422),
    });
    await expect(api.putSettings({} as never)).rejects.toThrow("generation.top_k: Input should be greater than or equal to 0");
  });

  it("reports an unreachable backend clearly", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.message).toMatch(/Cannot reach the TNSR backend/);
  });
});

describe("formatting", () => {
  it("formats numbers and ids", () => {
    expect(formatParams(3_020_000)).toBe("3.02M");
    expect(formatDuration(3725)).toBe("1h 2m");
    expect(modelLabel("experiments/demo-1/checkpoints/step_0004000")).toBe("demo-1 · step 4000");
    expect(modelLabel("checkpoints/imported")).toBe("imported");
  });
});
