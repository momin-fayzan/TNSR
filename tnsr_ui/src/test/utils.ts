import { vi } from "vitest";

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** A streaming SSE response whose chunks the test pushes manually. */
export function controllableStream() {
  const encoder = new TextEncoder();
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
  });
  return {
    response: new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } }),
    push: (text: string) => controller.enqueue(encoder.encode(text)),
    close: () => controller.close(),
  };
}

type Handler = (url: string, init?: RequestInit) => Response | Promise<Response>;

/** Route fetch calls by "METHOD /path" prefix; unmatched calls fail loudly. */
export function mockFetch(routes: Record<string, Handler>) {
  const calls: { method: string; url: string; body: unknown }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, url, body });
    const key = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((k) => {
        const [m, p] = k.split(" ");
        return m === method && url.startsWith(p);
      });
    if (!key) throw new Error(`Unmocked fetch: ${method} ${url}`);
    return routes[key](url, init);
  });
  vi.stubGlobal("fetch", fn);
  return { fn, calls };
}

export const MODEL = {
  id: "experiments/demo-1/checkpoints/step_0000100",
  name: "step_0000100",
  run_id: "demo-1",
  run_name: "demo",
  step: 100,
  num_parameters: 3_020_000,
  model_config: {
    vocab_size: 1621, block_size: 256, n_layer: 6, n_head: 6, n_embd: 192, dropout: 0, bias: false,
    norm: "rmsnorm", pos_encoding: "learned", mlp_ratio: 4, tie_embeddings: true, attn_impl: "sdpa",
  },
  tokenizer: { kind: "bpe", vocab_size: 1621, identity: "bpe-1621-abc" },
  dataset: { name: "demo", identity: "demo-123" },
  metrics: { val_loss: 0.4, val_perplexity: 1.49 },
  saved_at: "2026-10-21T06:00:00+0000",
  is_best: true,
  is_latest: true,
  has_train_state: true,
  complete: true,
  size_bytes: 36_000_000,
  evaluation: null,
  benchmark: null,
  loaded: false,
};

export const SETTINGS = {
  theme: "dark",
  generation: { max_new_tokens: 160, temperature: 0.7, top_k: 40, top_p: 0.95, repetition_penalty: 1, deterministic: false, seed: null, ignore_eos: false },
  default_model_id: null,
  default_mode: "chat",
};

export const HEALTH = { status: "ok", version: "0.1.0", uptime_seconds: 1, model: { loaded: null, loading: null, error: null } };

export function conversation(overrides: Record<string, unknown> = {}) {
  return {
    id: "c1",
    title: "New conversation",
    model_id: MODEL.id,
    mode: "chat",
    settings: SETTINGS.generation,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_000,
    message_count: 0,
    messages: [],
    ...overrides,
  };
}

export function message(overrides: Record<string, unknown> = {}) {
  return {
    id: "m1",
    conversation_id: "c1",
    seq: 1,
    role: "user",
    content: "Hello",
    status: "complete",
    model_id: null,
    settings: null,
    stats: null,
    error: null,
    created_at: 1_700_000_000,
    ...overrides,
  };
}
