// Shared Lab building blocks: run state badges, job list, training form, dataset cards.
import clsx from "clsx";
import { ChevronDown, ChevronRight, Database, Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../lib/api";
import { formatBytes, formatDuration, formatInt, formatRelative } from "../lib/format";
import type { DatasetMeta, Job, Preset } from "../lib/types";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorBanner, Field, TextInput } from "./ui";

export function RunStateBadge({ state }: { state?: string | null }) {
  const tone =
    state === "running"
      ? "accent"
      : state === "completed"
        ? "ok"
        : state === "interrupted"
          ? "warn"
          : state === "failed" || state === "missing"
            ? "danger"
            : "neutral";
  return (
    <Badge tone={tone}>
      {state === "running" && <span className="size-1.5 animate-pulse rounded-full bg-current" aria-hidden />}
      {state ?? "unknown"}
    </Badge>
  );
}

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className={clsx("h-1.5 w-full overflow-hidden rounded-full bg-surface-3", className)}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
    </div>
  );
}

const JOB_TONE: Record<Job["status"], "accent" | "ok" | "warn" | "danger" | "neutral"> = {
  pending: "neutral",
  running: "accent",
  stopping: "warn",
  completed: "ok",
  stopped: "warn",
  failed: "danger",
  lost: "danger",
};

function jobTitle(job: Job): string {
  const p = job.params as Record<string, string>;
  switch (job.kind) {
    case "train":
      return `Train ${p.name || p.config}`;
    case "resume":
      return `Resume ${p.run_id}`;
    case "evaluate":
      return `Evaluate ${p.model_id}`;
    case "benchmark":
      return `Benchmark ${p.model_id}`;
    case "prepare":
      return `Prepare dataset ${p.name}`;
    case "chess_prepare":
      return `Label chess dataset ${p.name} (Lichess ${p.month})`;
  }
}

export function JobRow({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState<string | null>(null);
  const active = job.status === "running" || job.status === "pending" || job.status === "stopping";

  useEffect(() => {
    if (!open) return;
    let alive = true;
    const load = () =>
      api
        .job(job.id)
        .then((j) => alive && setLog(j.log ?? ""))
        .catch(() => undefined);
    load();
    const t = active ? setInterval(load, 2000) : undefined;
    return () => {
      alive = false;
      if (t) clearInterval(t);
    };
  }, [open, job.id, active]);

  const duration = job.started_at ? (job.finished_at ?? Date.now() / 1000) - job.started_at : null;
  return (
    <li className="border-b border-border last:border-b-0">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <button
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={open ? "Hide job log" : "Show job log"}
          className="text-faint hover:text-fg"
        >
          {open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm text-fg">{jobTitle(job)}</div>
          <div className="text-xs text-faint">
            {formatRelative(job.created_at)} · {formatDuration(duration)}
            {job.run_id && (
              <>
                {" · "}
                <Link to={`/lab/runs/${job.run_id}`} className="text-accent hover:underline">
                  {job.run_id}
                </Link>
              </>
            )}
            {job.error && <span className="text-danger"> · {job.error}</span>}
          </div>
        </div>
        <Badge tone={JOB_TONE[job.status]}>{job.status}</Badge>
        {active && job.status !== "stopping" && (
          <Button size="sm" variant="danger" onClick={() => onCancel(job.id)}>
            <Square className="size-3" aria-hidden /> Stop
          </Button>
        )}
      </div>
      {open && (
        <pre className="mx-4 mb-3 max-h-72 overflow-auto rounded-md border border-border bg-bg p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap text-muted">
          {log ?? "Loading…"}
        </pre>
      )}
    </li>
  );
}

export function NewRunForm({ presets, datasets, onStarted }: { presets: Preset[]; datasets: DatasetMeta[]; onStarted: (job: Job) => void }) {
  const [preset, setPreset] = useState(presets.find((p) => p.name === "chess-mini")?.name ?? presets[0]?.name ?? "");
  const cfg = presets.find((p) => p.name === preset)?.config;
  const [name, setName] = useState("");
  const [dataset, setDataset] = useState("");
  const [maxSteps, setMaxSteps] = useState("");
  const [lr, setLr] = useState("");
  const [batch, setBatch] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!cfg) return;
    setDataset(cfg.data.dataset);
    setMaxSteps(String(cfg.train.max_steps));
    setLr(String(cfg.train.learning_rate));
    setBatch(String(cfg.train.batch_size));
  }, [cfg]);

  const nameError = name && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name) ? "Letters, digits, '.', '_' or '-' only" : null;
  const stepsError = maxSteps && !(Number.isInteger(Number(maxSteps)) && Number(maxSteps) >= 1) ? "Positive whole number" : null;
  const lrError = lr && !(Number(lr) > 0 && Number(lr) < 1) ? "Between 0 and 1" : null;
  const batchError = batch && !(Number.isInteger(Number(batch)) && Number(batch) >= 1 && Number(batch) <= 4096) ? "1–4096" : null;
  const invalid = Boolean(nameError || stepsError || lrError || batchError || !preset);
  const datasetMissing = dataset !== "" && !datasets.some((d) => d.name === dataset);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const overrides = [`train.max_steps=${maxSteps}`, `train.learning_rate=${lr}`, `train.batch_size=${batch}`, `data.dataset="${dataset}"`];
    if (cfg && Number(lr) < Number(cfg.train.min_lr)) overrides.push(`train.min_lr=${Number(lr) / 10}`);
    try {
      const job = await api.startJob("train", { config: preset, name: name || undefined, overrides });
      onStarted(job);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!presets.length) return <p className="p-4 text-sm text-muted">Loading presets…</p>;
  const m = cfg?.model;
  return (
    <div className="space-y-4 p-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Preset" htmlFor="run-preset" hint={cfg?.description}>
          <select
            id="run-preset"
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
            className="h-8 w-full rounded-md border border-border bg-surface-2 px-2 text-sm focus:border-accent focus:outline-none"
          >
            {presets.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Run name" htmlFor="run-name" error={nameError} hint="Defaults to the preset name; a timestamp is appended.">
          <TextInput id="run-name" value={name} onChange={(e) => setName(e.target.value)} placeholder={preset} invalid={!!nameError} />
        </Field>
        <Field label="Dataset" htmlFor="run-dataset" error={datasetMissing ? "Dataset not prepared yet" : null}>
          <select
            id="run-dataset"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="h-8 w-full rounded-md border border-border bg-surface-2 px-2 text-sm focus:border-accent focus:outline-none"
          >
            {datasetMissing && <option value={dataset}>{dataset} (missing)</option>}
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} · {d.tokenizer.kind}/{d.tokenizer.vocab_size}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Max steps" htmlFor="run-steps" error={stepsError}>
          <TextInput id="run-steps" inputMode="numeric" value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} invalid={!!stepsError} className="font-mono" />
        </Field>
        <Field label="Learning rate" htmlFor="run-lr" error={lrError}>
          <TextInput id="run-lr" inputMode="decimal" value={lr} onChange={(e) => setLr(e.target.value)} invalid={!!lrError} className="font-mono" />
        </Field>
        <Field label="Batch size" htmlFor="run-batch" error={batchError}>
          <TextInput id="run-batch" inputMode="numeric" value={batch} onChange={(e) => setBatch(e.target.value)} invalid={!!batchError} className="font-mono" />
        </Field>
      </div>
      {m && (
        <p className="text-xs text-muted">
          Architecture: {m.n_layer} layers · {m.n_head} heads · d_model {m.n_embd} · context {m.block_size} · {m.norm} · {m.pos_encoding} positions
        </p>
      )}
      {error && <ErrorBanner>{error}</ErrorBanner>}
      <div className="flex justify-end">
        <Button variant="primary" onClick={submit} loading={busy} disabled={invalid || datasetMissing}>
          <Play className="size-3.5" aria-hidden /> Start training
        </Button>
      </div>
    </div>
  );
}

export function DatasetsPanel({ datasets, onPrepared }: { datasets: DatasetMeta[]; onPrepared: (job: Job) => void }) {
  const [name, setName] = useState("chess-2013-02");
  const [month, setMonth] = useState("2013-02");
  const [positions, setPositions] = useState("20000");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nameError = name && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name) ? "Letters, digits, '.', '_' or '-' only" : null;
  const monthError = /^20\d\d-(0[1-9]|1[0-2])$/.test(month) ? null : "Format: 2013-02";
  const posError = Number.isInteger(Number(positions)) && Number(positions) >= 100 ? null : "At least 100";
  const exists = datasets.some((d) => d.name === name);

  const prepare = async () => {
    setBusy(true);
    setError(null);
    try {
      onPrepared(await api.startJob("chess_prepare", { name, month, max_positions: Number(positions) }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader title="Datasets" subtitle="Positions from the Lichess open database, every legal move scored by Stockfish" />
      {datasets.length === 0 ? (
        <EmptyState icon={<Database className="size-6" />} title="No datasets yet">
          Label positions below, or run <code className="font-mono text-xs">uv run tnsr chess prepare --month 2013-01 --name chess-demo</code>.
        </EmptyState>
      ) : (
        <ul className="divide-y divide-border">
          {datasets.map((d) => (
            <li key={d.name} className="px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-sm text-fg">{d.name}</span>
                {d.kind === "chess_action_value" ? <Badge tone="violet">chess</Badge> : <Badge>{d.tokenizer.kind} · vocab {d.tokenizer.vocab_size}</Badge>}
                {d.kind === "chess_action_value" && d.complete === false && <Badge tone="warn">labeling incomplete</Badge>}
                <span className="text-xs text-faint">{d.identity}</span>
              </div>
              {d.kind === "chess_action_value" ? (
                <>
                  <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted sm:grid-cols-4">
                    <span>{formatInt(d.stats?.train_positions)} train positions</span>
                    <span>{formatInt(d.stats?.val_positions)} val positions</span>
                    <span>{formatInt((d.stats?.train_records ?? 0) + (d.stats?.val_records ?? 0))} scored moves</span>
                    <span>{formatBytes(d.size_bytes)} on disk</span>
                  </div>
                  <p className="mt-1 text-xs text-faint">
                    {d.source?.file} · {d.labeling?.engine}
                    {d.labeling?.nodes_per_move ? ` at ${formatInt(d.labeling.nodes_per_move)} nodes/move` : ""} · {d.source?.license}
                  </p>
                </>
              ) : (
                <>
                  <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted sm:grid-cols-4">
                    <span>{formatInt(d.stats?.documents)} docs</span>
                    <span>{formatInt(d.stats?.train_tokens)} train tok</span>
                    <span>{formatInt(d.stats?.val_tokens)} val tok</span>
                    <span>{formatBytes(d.size_bytes)} on disk</span>
                  </div>
                  <p className="mt-1 text-xs text-faint">
                    {d.provenance?.source} — {d.provenance?.license}
                  </p>
                </>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="border-t border-border p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-44">
            <Field label="New dataset" htmlFor="ds-name" error={nameError ?? (exists ? "Already exists" : null)}>
              <TextInput id="ds-name" value={name} onChange={(e) => setName(e.target.value)} invalid={!!nameError || exists} />
            </Field>
          </div>
          <div className="w-28">
            <Field label="Lichess month" htmlFor="ds-month" error={monthError}>
              <TextInput id="ds-month" value={month} onChange={(e) => setMonth(e.target.value)} invalid={!!monthError} className="font-mono" />
            </Field>
          </div>
          <div className="w-28">
            <Field label="Positions" htmlFor="ds-positions" error={posError}>
              <TextInput id="ds-positions" inputMode="numeric" value={positions} onChange={(e) => setPositions(e.target.value)} invalid={!!posError} className="font-mono" />
            </Field>
          </div>
          <Button onClick={prepare} loading={busy} disabled={!name || !!nameError || !!monthError || !!posError || exists}>
            Download &amp; label
          </Button>
        </div>
        {error && (
          <div className="mt-3">
            <ErrorBanner>{error}</ErrorBanner>
          </div>
        )}
        <p className="mt-2 text-xs text-faint">
          Runs Stockfish on every legal move (2,000 nodes each); roughly 20–30 positions/s on 4 cores. Larger months are bigger downloads
          (2013 months are ~20–40 MB).
        </p>
      </div>
    </Card>
  );
}
