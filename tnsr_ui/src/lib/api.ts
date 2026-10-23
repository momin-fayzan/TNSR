import type {
  AppSettings,
  ChessAnalysis,
  ChessGame,
  ChessGameView,
  ChessState,
  Comparison,
  DatasetMeta,
  ExperimentDetail,
  ExperimentSummary,
  Health,
  Job,
  JobKind,
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

  experiments: () => request<{ experiments: ExperimentSummary[] }>("/experiments"),
  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`),
  stopRun: (id: string) => request<{ signalled: boolean }>(`/experiments/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  compare: (run_ids: string[]) => request<Comparison>("/experiments/compare", { method: "POST", body: json({ run_ids }) }),
  exportUrl: (id: string, format: "json" | "csv") => `${BASE}/experiments/${encodeURIComponent(id)}/export?format=${format}`,
  datasets: () => request<{ datasets: DatasetMeta[] }>("/datasets"),
  presets: () => request<{ presets: Preset[] }>("/presets"),

  chessModels: () => request<{ models: ModelInfo[] }>("/chess/models"),
  chessGames: () => request<{ games: ChessGame[] }>("/chess/games"),
  chessGame: (id: string) => request<ChessGameView>(`/chess/games/${id}`),
  newChessGame: (body: { human_color: "white" | "black"; model_id?: string | null; start_fen?: string }) =>
    request<ChessGameView>("/chess/games", { method: "POST", body: json(body) }),
  chessMove: (id: string, move: string) => request<ChessGameView>(`/chess/games/${id}/move`, { method: "POST", body: json({ move }) }),
  chessUndo: (id: string) => request<ChessGameView>(`/chess/games/${id}/undo`, { method: "POST" }),
  renameChessGame: (id: string, title: string) => request<ChessGame>(`/chess/games/${id}`, { method: "PATCH", body: json({ title }) }),
  deleteChessGame: (id: string) => request<void>(`/chess/games/${id}`, { method: "DELETE" }),
  chessAnalyse: (fen: string, model_id?: string | null) =>
    request<{ state: ChessState; analysis: ChessAnalysis }>("/chess/analyse", { method: "POST", body: json({ fen, model_id }) }),

  jobs: () => request<{ jobs: Job[] }>("/jobs"),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  startJob: (kind: JobKind, params: Record<string, unknown>) =>
    request<Job>("/jobs", { method: "POST", body: json({ kind, params }) }),
  cancelJob: (id: string) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
};
