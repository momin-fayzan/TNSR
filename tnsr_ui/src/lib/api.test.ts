import { afterEach, describe, expect, it, vi } from "vitest";

import { controllableStream, jsonResponse, mockFetch, sse } from "../test/utils";
import { api, ApiError, parseSSE, streamRequest } from "./api";
import { formatDuration, formatParams, modelLabel } from "./format";
import { validateField, validateSeed, validateSettings } from "./settings";

afterEach(() => vi.unstubAllGlobals());

function streamOf(...chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(c) {
      chunks.forEach((ch) => c.enqueue(enc.encode(ch)));
      c.close();
    },
  });
}

describe("parseSSE", () => {
  it("parses events split across arbitrary chunk boundaries", async () => {
    const raw = sse("start", { generation_id: "g" }) + sse("token", { text: "Hel" }) + sse("token", { text: "lo\n" }) + sse("done", { stats: null });
    const chunks = [raw.slice(0, 7), raw.slice(7, 40), raw.slice(40, 41), raw.slice(41)];
    const events = [];
    for await (const e of parseSSE(streamOf(...chunks))) events.push(e);
    expect(events.map((e) => e.event)).toEqual(["start", "token", "token", "done"]);
    expect(events[2].data).toEqual({ text: "lo\n" });
  });

  it("handles multi-byte characters split between chunks", async () => {
    const bytes = new TextEncoder().encode(sse("token", { text: "日本🙂" }));
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, 25));
        c.enqueue(bytes.slice(25));
        c.close();
      },
    });
    const events = [];
    for await (const e of parseSSE(stream)) events.push(e);
    expect(events[0].data).toEqual({ text: "日本🙂" });
  });
});

describe("streamRequest", () => {
  it("dispatches start/token/done handlers in order", async () => {
    const s = controllableStream();
    mockFetch({ "POST /api/generate": () => s.response });
    const seen: string[] = [];
    const p = streamRequest("/generate", {}, {
      onStart: (d) => seen.push(`start:${d.generation_id}`),
      onToken: (t) => seen.push(`token:${t}`),
      onDone: () => seen.push("done"),
    });
    s.push(sse("start", { generation_id: "g1" }));
    s.push(sse("token", { text: "a" }) + sse("token", { text: "b" }));
    s.push(sse("done", { message: null, stats: null }));
    s.close();
    await p;
    expect(seen).toEqual(["start:g1", "token:a", "token:b", "done"]);
  });

  it("reports a stream that ends without done as an error", async () => {
    const s = controllableStream();
    mockFetch({ "POST /api/generate": () => s.response });
    const onError = vi.fn();
    const p = streamRequest("/generate", {}, { onError });
    s.push(sse("token", { text: "a" }));
    s.close();
    await p;
    expect(onError).toHaveBeenCalledWith("Connection closed before generation finished", null);
  });

  it("throws ApiError with the server's detail for rejected requests", async () => {
    mockFetch({ "POST /api/generate": () => jsonResponse({ detail: "No trained models found." }, 409) });
    await expect(streamRequest("/generate", {}, {})).rejects.toMatchObject({ status: 409, message: "No trained models found." });
  });
});

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

describe("settings validation", () => {
  it("enforces backend bounds", () => {
    expect(validateField("temperature", "0.7").value).toBe(0.7);
    expect(validateField("temperature", "6").error).toMatch(/between 0 and 5/);
    expect(validateField("top_k", "2.5").error).toMatch(/whole number/);
    expect(validateField("top_p", "0").error).toBeDefined();
    expect(validateField("max_new_tokens", "").error).toBe("Required");
    expect(validateField("max_new_tokens", "abc").error).toMatch(/number/);
    expect(validateSeed("").value).toBeNull();
    expect(validateSeed("-1").error).toBeDefined();
    expect(validateSettings({ max_new_tokens: 0, temperature: 1, top_k: 0, top_p: 1, repetition_penalty: 1, deterministic: false, seed: null, ignore_eos: false })).toHaveProperty("max_new_tokens");
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
