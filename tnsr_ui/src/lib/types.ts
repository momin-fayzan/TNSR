// Shapes returned by the TNSR backend (tnsr_server/api/routes.py).

export interface GenerationSettings {
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  deterministic: boolean;
  seed: number | null;
  ignore_eos: boolean;
}

export type Mode = "chat" | "completion";
export type Theme = "dark" | "light" | "system";

export interface AppSettings {
  theme: Theme;
  generation: GenerationSettings;
  default_model_id: string | null;
  default_mode: Mode;
}

export interface GenerationStats {
  prompt_tokens: number;
  generated_tokens: number;
  prompt_truncated: boolean;
  time_to_first_token_ms: number | null;
  total_ms: number;
  decode_tokens_per_sec: number | null;
  tokens_per_sec: number | null;
  finish_reason: "eos" | "length" | "stop" | "cancelled" | string;
  cache_rebuilds: number;
}

export type MessageStatus = "complete" | "streaming" | "cancelled" | "error";

export interface Message {
  id: string;
  conversation_id: string;
  seq: number;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
  model_id: string | null;
  settings: GenerationSettings | null;
  stats: GenerationStats | null;
  error: string | null;
  created_at: number;
}

export interface Conversation {
  id: string;
  title: string;
  model_id: string | null;
  mode: Mode;
  settings: GenerationSettings;
  created_at: number;
  updated_at: number;
  message_count?: number;
  messages?: Message[];
}

export interface ModelConfig {
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
  dropout: number;
  bias: boolean;
  norm: string;
  pos_encoding: string;
  mlp_ratio: number;
  tie_embeddings: boolean;
  attn_impl: string;
}

export interface TokenizerInfo {
  kind: string;
  vocab_size: number;
  identity: string;
}

export interface DatasetRef {
  name: string;
  identity: string;
  val_text_sha256?: string;
  train_tokens?: number;
  val_tokens?: number;
  val_bytes?: number;
}

export interface EvaluationResult {
  kind: "evaluation";
  checkpoint_id: string;
  checkpoint_name: string;
  step: number;
  created_at: string;
  dataset: DatasetRef & { split: string };
  tokenizer: TokenizerInfo;
  metrics: {
    val_loss: number;
    perplexity: number;
    bits_per_byte: number;
    tokens_evaluated: number;
    covered_full_split: boolean;
  };
  seconds: number;
  device: string;
  samples: { prompt: string; completion: string; finish_reason: string }[];
  probes?: ProbeResult | null;
}

export interface ProbeResult {
  version: number;
  decoding: string;
  accuracy: number;
  total: number;
  by_category: Record<string, { correct: number; total: number; accuracy: number }>;
  failures: { category: string; question: string; expected: string; answer: string }[];
  seconds: number;
}

export interface BenchmarkRow {
  prompt_tokens: number;
  new_tokens: number;
  kv_cache: boolean;
  ttft_ms_median: number;
  decode_tokens_per_sec_median: number;
  total_ms_median: number;
}

export interface BenchmarkResult {
  kind: "benchmark";
  checkpoint_id: string;
  checkpoint_name: string;
  step: number;
  created_at: string;
  device: string;
  threads: number;
  repeats: number;
  results: BenchmarkRow[];
  memory: Record<string, number>;
}

export interface ModelInfo {
  id: string;
  name: string;
  run_id: string | null;
  run_name: string | null;
  step: number | null;
  num_parameters: number | null;
  model_config: ModelConfig | null;
  tokenizer: TokenizerInfo;
  dataset: DatasetRef | null;
  metrics: { train_loss?: number | null; val_loss?: number | null; val_perplexity?: number | null };
  saved_at: string | null;
  is_best: boolean;
  is_latest: boolean;
  has_train_state: boolean;
  complete: boolean;
  size_bytes: number;
  evaluation: EvaluationResult | null;
  benchmark: BenchmarkResult | null;
  loaded?: boolean;
  recommended?: boolean;
}

export interface LoadedModel {
  id: string;
  device: string;
  num_parameters: number;
  model_config: ModelConfig;
  tokenizer: TokenizerInfo;
  step: number | null;
  loaded_at: number | null;
}

export interface ModelStatus {
  loaded: LoadedModel | null;
  loading: string | null;
  error: string | null;
}

export interface Health {
  status: string;
  version: string;
  uptime_seconds: number;
  model: ModelStatus;
}

export interface DiagnosticCheck {
  name: string;
  status: "ok" | "warning" | "error";
  detail: string;
}

export interface SystemInfo {
  version: string;
  system: Record<string, string | number | boolean>;
  device: { selected: string; reason: string };
  diagnostics: { checks: DiagnosticCheck[]; ok: boolean; tnsr_home: string; cpu_count: number; torch_threads: number };
  storage: Record<string, string | number>;
  model: ModelStatus;
}

export interface RunStatus {
  state?: "running" | "completed" | "interrupted" | "failed" | "missing" | string;
  step?: number;
  max_steps?: number;
  progress?: number;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
  tokens_seen?: number;
  best_val_loss?: number | null;
  last_val_loss?: number | null;
  last_train_loss?: number | null;
  peak_rss_mb?: number;
  device?: string;
  updated_at?: string;
  error?: string;
  nonfinite_steps?: number;
}

export interface RunConfig {
  name: string;
  description: string;
  model: ModelConfig;
  data: { dataset: string };
  train: Record<string, number | string | boolean>;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  state: string | null;
  config: RunConfig;
  environment: Record<string, unknown> | null;
  dataset: DatasetRef | null;
  tokenizer: TokenizerInfo | null;
  num_parameters: number | null;
  best_val_loss: number | null;
  step: number | null;
  max_steps: number | null;
  created_at: string | null;
  missing: boolean;
  status: RunStatus;
}

export interface MetricRecord {
  type: "train" | "eval" | "nonfinite";
  step: number;
  loss?: number;
  lr?: number;
  grad_norm?: number;
  tokens_per_sec?: number;
  elapsed?: number;
  rss_mb?: number;
  train_loss?: number;
  val_loss?: number;
  val_perplexity?: number;
}

export interface ExperimentDetail {
  id: string;
  name: string;
  description: string;
  created_at: string;
  config: RunConfig;
  num_parameters: number;
  num_parameters_non_embedding: number;
  dataset: DatasetRef;
  tokenizer: TokenizerInfo;
  environment: Record<string, string | number | boolean | null>;
  status: RunStatus;
  checkpoints: ModelInfo[];
  best_checkpoint: string | null;
  latest_checkpoint: string | null;
  metrics: MetricRecord[];
  evaluations: EvaluationResult[];
  benchmarks: BenchmarkResult[];
}

export interface Comparison {
  runs: {
    id: string;
    name: string;
    num_parameters: number;
    model_config: ModelConfig;
    train_config: Record<string, number | string | boolean>;
    dataset: DatasetRef;
    tokenizer: TokenizerInfo;
    status: RunStatus;
    best_val_loss: number | null;
    evaluation: EvaluationResult["metrics"] | null;
    probe_accuracy: number | null;
    benchmark: { results: BenchmarkRow[]; memory: Record<string, number>; device: string } | null;
  }[];
  comparable: { loss_and_perplexity: boolean; bits_per_byte: boolean; benchmarks: boolean };
  warnings: string[];
}

export type JobKind = "train" | "resume" | "evaluate" | "benchmark" | "prepare";

export interface Job {
  id: string;
  kind: JobKind;
  params: Record<string, unknown>;
  status: "pending" | "running" | "stopping" | "completed" | "stopped" | "failed" | "lost";
  pid: number | null;
  run_id: string | null;
  exit_code: number | null;
  error: string | null;
  result: unknown;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  log?: string;
}

export interface DatasetMeta {
  name: string;
  identity: string;
  created_at: string;
  provenance: { source?: string; license?: string; description?: string };
  split: { method: string; seed: number; val_fraction: number };
  tokenizer: TokenizerInfo;
  stats: {
    documents: number;
    train_documents: number;
    val_documents: number;
    train_tokens: number;
    val_tokens: number;
    train_bytes: number;
    val_bytes: number;
    bytes_per_token: number;
  };
  size_bytes: number;
}

export interface Preset {
  name: string;
  config: RunConfig;
  toml: string;
}
