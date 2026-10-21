import type {
  AppSettings,
  Comparison,
  Conversation,
  DatasetMeta,
  ExperimentDetail,
  ExperimentSummary,
  GenerationSettings,
  GenerationStats,
  Health,
  Job,
  JobKind,
  Message,
  Mode,
  ModelInfo,
  ModelStatus,
  Preset,
  SystemInfo,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function detailMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => {
          const loc = Array.isArray(d?.loc) ? d.loc.filter((p: unknown) => p !== "body").join(".") : "";
          return loc ? `${loc}: ${d.msg}` : String(d?.msg ?? d);
        })
        .join("; ");
    }
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the TNSR backend. Is `tnsr serve` running?");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) throw new ApiError(res.status, detailMessage(body, `${res.status} ${res.statusText}`));
  return body as T;
}

const json = (body: unknown) => JSON.stringify(body);

// -- Server-Sent Events over POST -----------------------------------------------------------

export interface StreamHandlers {
  onStart?: (data: { generation_id: string; user_message?: Message; assistant_message?: Message }) => void;
  onToken?: (text: string) => void;
  onDone?: (data: { message: Message | null; stats: GenerationStats | null }) => void;
  onError?: (detail: string, message: Message | null) => void;
}

/** Parse an SSE byte stream into (event, data) pairs. Exported for tests. */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string; data: unknown }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      }
      if (dataLines.length) {
        let data: unknown = dataLines.join("\n");
        try {
          data = JSON.parse(data as string);
        } catch {
          /* keep raw text */
        }
        yield { event, data };
      }
    }
  }
}

export async function streamRequest(
  path: string,
  body: unknown,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: json(body ?? {}),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, "Cannot reach the TNSR backend. Is `tnsr serve` running?");
  }
  if (!res.ok || !res.body) {
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = JSON.parse(text);
    } catch {
      /* raw */
    }
    throw new ApiError(res.status, detailMessage(parsed, `${res.status} ${res.statusText}`));
  }
  let finished = false;
  for await (const { event, data } of parseSSE(res.body)) {
    const d = data as Record<string, unknown>;
    if (event === "start") handlers.onStart?.(d as never);
    else if (event === "token") handlers.onToken?.(String(d.text ?? ""));
    else if (event === "done") {
      finished = true;
      handlers.onDone?.(d as never);
    } else if (event === "error") {
      finished = true;
      handlers.onError?.(String(d.detail ?? "Generation failed"), (d.message as Message) ?? null);
    }
  }
  if (!finished) handlers.onError?.("Connection closed before generation finished", null);
}

// -- endpoints --------------------------------------------------------------------------------

export const api = {
  health: () => request<Health>("/health"),
  system: () => request<SystemInfo>("/system"),
  getSettings: () => request<AppSettings>("/settings"),
  putSettings: (s: AppSettings) => request<AppSettings>("/settings", { method: "PUT", body: json(s) }),

  models: () => request<{ models: ModelInfo[] } & ModelStatus>("/models"),
  model: (id: string) => request<ModelInfo>(`/models/${id}`),
  loadModel: (model_id: string) =>
    request<{ loaded: ModelStatus["loaded"] }>("/models/load", { method: "POST", body: json({ model_id }) }),
  unloadModel: () => request<ModelStatus>("/models/unload", { method: "POST" }),

  conversations: () => request<{ conversations: Conversation[] }>("/conversations"),
  conversation: (id: string) => request<Conversation>(`/conversations/${id}`),
  createConversation: (body: { title?: string; model_id?: string | null; mode?: Mode; settings?: GenerationSettings }) =>
    request<Conversation>("/conversations", { method: "POST", body: json(body) }),
  updateConversation: (
    id: string,
    body: { title?: string; model_id?: string; mode?: Mode; settings?: GenerationSettings },
  ) => request<Conversation>(`/conversations/${id}`, { method: "PATCH", body: json(body) }),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
  sendMessage: (
    id: string,
    body: { content: string; settings?: GenerationSettings; model_id?: string | null },
    handlers: StreamHandlers,
    signal?: AbortSignal,
  ) => streamRequest(`/conversations/${id}/messages`, body, handlers, signal),
  retryMessage: (
    id: string,
    messageId: string,
    body: { settings?: GenerationSettings },
    handlers: StreamHandlers,
    signal?: AbortSignal,
  ) => streamRequest(`/conversations/${id}/messages/${messageId}/retry`, body, handlers, signal),
  cancelGeneration: (generationId: string) =>
    request<{ cancelled: boolean }>(`/generations/${generationId}/cancel`, { method: "POST" }),

  experiments: () => request<{ experiments: ExperimentSummary[] }>("/experiments"),
  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`),
  stopRun: (id: string) => request<{ signalled: boolean }>(`/experiments/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  compare: (run_ids: string[]) => request<Comparison>("/experiments/compare", { method: "POST", body: json({ run_ids }) }),
  exportUrl: (id: string, format: "json" | "csv") => `${BASE}/experiments/${encodeURIComponent(id)}/export?format=${format}`,
  datasets: () => request<{ datasets: DatasetMeta[] }>("/datasets"),
  presets: () => request<{ presets: Preset[] }>("/presets"),

  jobs: () => request<{ jobs: Job[] }>("/jobs"),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  startJob: (kind: JobKind, params: Record<string, unknown>) =>
    request<Job>("/jobs", { method: "POST", body: json({ kind, params }) }),
  cancelJob: (id: string) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
};
