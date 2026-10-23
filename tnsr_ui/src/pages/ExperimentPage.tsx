import { ArrowLeft, Download, Gauge, Play, Square, Target } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { LossChart, SeriesChart } from "../components/charts";
import { ProgressBar, RunStateBadge } from "../components/lab";
import { BenchmarkView, EvaluationView } from "../components/results";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorBanner, Segmented, Spinner, Stat } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import { formatBytes, formatDate, formatDuration, formatInt, formatNumber, formatParams } from "../lib/format";
import type { ExperimentDetail, Job } from "../lib/types";

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

export function ExperimentPage() {
  const { runId = "" } = useParams();
  const jobs = usePolling(() => api.jobs(), 3000);
  const runJobs = (jobs.data?.jobs ?? []).filter(
    (j) => j.run_id === runId || (j.params as Record<string, string>).model_id?.includes(`experiments/${runId}/`),
  );
  const exp = usePolling(() => api.experiment(runId), 3000, [runId]);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [logScale, setLogScale] = useState(true);

  const startJob = async (kind: Job["kind"], params: Record<string, unknown>, key: string) => {
    setBusy(key);
    setActionError(null);
    try {
      await api.startJob(kind, params);
      await jobs.refresh();
      await exp.refresh();
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  if (exp.loading && !exp.data) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner />
      </div>
    );
  }
  if (exp.error && !exp.data) {
    return (
      <EmptyState title="Experiment not available" action={<Link to="/lab" className="text-sm text-accent hover:underline">Back to Lab</Link>}>
        {exp.error}
      </EmptyState>
    );
  }
  const d = exp.data as ExperimentDetail;
  const st = d.status;
  const train = d.metrics.filter((m) => m.type === "train");
  const tps = median(train.map((m) => m.tokens_per_sec ?? 0).filter(Boolean));
  const resumable = (st.state === "interrupted" || st.state === "failed") && d.checkpoints.some((c) => c.has_train_state);
  const latestEval = d.evaluations.at(-1);
  const latestBench = d.benchmarks.at(-1);
  const trainingJob = runJobs.find((j) => (j.kind === "train" || j.kind === "resume") && ["running", "pending", "stopping"].includes(j.status));
  const mc = d.config.model;
  const isChess = d.task === "chess" || d.tokenizer.kind === "chess";
  const tc = d.config.train;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="border-b border-border px-6 py-5">
        <Link to="/lab" className="mb-2 inline-flex items-center gap-1 text-xs text-muted hover:text-fg">
          <ArrowLeft className="size-3.5" aria-hidden /> Lab
        </Link>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate font-mono text-lg font-semibold tracking-tight">{d.id}</h1>
              <RunStateBadge state={st.state} />
            </div>
            <p className="mt-1 text-sm text-muted">{d.description || d.name}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {trainingJob && (
              <Button variant="danger" onClick={() => api.cancelJob(trainingJob.id).then(() => jobs.refresh())} disabled={trainingJob.status === "stopping"}>
                <Square className="size-3.5" aria-hidden /> {trainingJob.status === "stopping" ? "Stopping…" : "Stop & checkpoint"}
              </Button>
            )}
            {!trainingJob && st.state === "running" && (
              <Button
                variant="danger"
                loading={busy === "stop"}
                onClick={async () => {
                  setBusy("stop");
                  setActionError(null);
                  try {
                    await api.stopRun(d.id);
                    await exp.refresh();
                  } catch (err) {
                    setActionError((err as Error).message);
                  } finally {
                    setBusy(null);
                  }
                }}
              >
                <Square className="size-3.5" aria-hidden /> Stop & checkpoint
              </Button>
            )}
            {!trainingJob && resumable && (
              <Button onClick={() => startJob("resume", { run_id: d.id }, "resume")} loading={busy === "resume"}>
                <Play className="size-3.5" aria-hidden /> Resume from latest checkpoint
              </Button>
            )}
            <a href={api.exportUrl(d.id, "json")} download>
              <Button>
                <Download className="size-3.5" aria-hidden /> JSON
              </Button>
            </a>
            <a href={api.exportUrl(d.id, "csv")} download>
              <Button>
                <Download className="size-3.5" aria-hidden /> CSV
              </Button>
            </a>
          </div>
        </div>
        {st.state === "running" && st.max_steps != null && (
          <div className="mt-4 max-w-xl">
            <ProgressBar value={(st.step ?? 0) / st.max_steps} />
            <div className="mt-1 text-xs text-muted">
              step {st.step}/{st.max_steps} · elapsed {formatDuration(st.elapsed_seconds)}
              {st.eta_seconds != null && ` · ETA ${formatDuration(st.eta_seconds)}`}
            </div>
          </div>
        )}
        {st.error && (
          <div className="mt-3">
            <ErrorBanner>{st.error}</ErrorBanner>
          </div>
        )}
        {actionError && (
          <div className="mt-3">
            <ErrorBanner>{actionError}</ErrorBanner>
          </div>
        )}
      </div>

      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <div className="grid grid-cols-2 gap-4 rounded-lg border border-border bg-surface p-4 sm:grid-cols-4 lg:grid-cols-7">
          <Stat label="Step" value={`${st.step ?? 0} / ${st.max_steps ?? "?"}`} />
          <Stat label="Best val loss" value={formatNumber(st.best_val_loss)} hint={st.best_val_loss != null ? `ppl ${formatNumber(Math.exp(st.best_val_loss), 3)}` : undefined} />
          <Stat label="Last train loss" value={formatNumber(st.last_train_loss)} />
          <Stat label="Parameters" value={formatParams(d.num_parameters)} hint={`${formatParams(d.num_parameters_non_embedding)} non-emb.`} />
          <Stat label="Tokens seen" value={formatInt(st.tokens_seen)} />
          <Stat label="Throughput" value={tps ? `${formatInt(Math.round(tps))} tok/s` : "—"} hint="median training" />
          <Stat label="Peak RSS" value={st.peak_rss_mb ? formatBytes(st.peak_rss_mb * 1e6) : "—"} hint={st.device} />
        </div>

        <Card>
          <CardHeader
            title="Loss"
            subtitle="Per-batch training loss, plus train/val loss on fixed evaluation windows"
            actions={
              <Segmented<"log" | "linear">
                label="Loss axis scale"
                value={logScale ? "log" : "linear"}
                onChange={(v) => setLogScale(v === "log")}
                options={[
                  { value: "log", label: "Log" },
                  { value: "linear", label: "Linear" },
                ]}
              />
            }
          />
          <div className="p-4">{d.metrics.length ? <LossChart metrics={d.metrics} height={300} log={logScale} /> : <EmptyState title="No metrics logged yet" />}</div>
        </Card>

        {isChess && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader title="Top move = Stockfish" subtitle="Share of held-out positions where the model's best move is Stockfish's best" />
              <div className="p-4">
                <SeriesChart metrics={d.metrics} field="move_accuracy" name="move accuracy" color="#38bdf8" type="eval" />
              </div>
            </Card>
            <Card>
              <CardHeader title="Regret" subtitle="Mean Stockfish win probability lost per move vs the best move (lower is better)" />
              <div className="p-4">
                <SeriesChart metrics={d.metrics} field="regret" name="regret" color="#f472b6" type="eval" />
              </div>
            </Card>
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader title="Learning rate" />
            <div className="p-4">
              <SeriesChart metrics={d.metrics} field="lr" name="learning rate" color="#f59e0b" />
            </div>
          </Card>
          <Card>
            <CardHeader title="Training throughput" subtitle="tokens / second" />
            <div className="p-4">
              <SeriesChart metrics={d.metrics} field="tokens_per_sec" name="tokens/s" color="#34d399" />
            </div>
          </Card>
        </div>

        <Card>
          <CardHeader title="Checkpoints" subtitle="Evaluate on the full validation split or benchmark inference speed" />
          {d.checkpoints.length === 0 ? (
            <EmptyState title="No checkpoints yet">Checkpoints are written at each evaluation interval.</EmptyState>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-[11px] tracking-wide text-faint uppercase">
                    <th className="px-4 py-2 font-medium">Checkpoint</th>
                    <th className="py-2 pr-4 font-medium">Val loss</th>
                    <th className="py-2 pr-4 font-medium">Size</th>
                    <th className="py-2 pr-4 font-medium">Saved</th>
                    <th className="py-2 pr-4 text-right font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {d.checkpoints.map((c) => (
                    <tr key={c.name} className="border-b border-border last:border-b-0">
                      <td className="px-4 py-2 font-mono text-xs">
                        {c.name} {c.is_best && <Badge tone="ok">best</Badge>} {c.is_latest && <Badge tone="accent">latest</Badge>}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">{formatNumber(c.metrics?.val_loss)}</td>
                      <td className="py-2 pr-4 text-xs text-muted">{formatBytes(c.size_bytes)}</td>
                      <td className="py-2 pr-4 text-xs text-faint">{formatDate(c.saved_at)}</td>
                      <td className="py-2 pr-4">
                        <div className="flex justify-end gap-1.5">
                          <Button size="sm" onClick={() => startJob("evaluate", { model_id: c.id }, `eval-${c.name}`)} loading={busy === `eval-${c.name}`}>
                            <Target className="size-3" aria-hidden /> Evaluate
                          </Button>
                          {!isChess && (
                            <Button size="sm" onClick={() => startJob("benchmark", { model_id: c.id }, `bench-${c.name}`)} loading={busy === `bench-${c.name}`}>
                              <Gauge className="size-3" aria-hidden /> Benchmark
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {runJobs.some((j) => (j.kind === "evaluate" || j.kind === "benchmark") && j.status === "running") && (
            <div className="flex items-center gap-2 border-t border-border px-4 py-2 text-xs text-muted">
              <Spinner className="size-3.5" /> Running evaluation/benchmark job…
            </div>
          )}
        </Card>

        <div className={isChess ? "grid gap-6" : "grid gap-6 lg:grid-cols-2"}>
          <Card>
            <CardHeader title="Evaluation" subtitle={d.evaluations.length ? `${d.evaluations.length} result(s); latest shown` : undefined} />
            <div className="p-4">
              {latestEval ? <EvaluationView result={latestEval} /> : <p className="text-sm text-muted">No evaluation yet. Use “Evaluate” on a checkpoint.</p>}
            </div>
          </Card>
          {!isChess && (
            <Card>
              <CardHeader title="Inference benchmark" subtitle={d.benchmarks.length ? `${d.benchmarks.length} result(s); latest shown` : undefined} />
              <div className="p-4">
                {latestBench ? <BenchmarkView result={latestBench} /> : <p className="text-sm text-muted">No benchmark yet. Use “Benchmark” on a checkpoint.</p>}
              </div>
            </Card>
          )}
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <KeyValueCard
            title="Model"
            rows={[
              ["layers", mc.n_layer],
              ["heads", mc.n_head],
              ["d_model", mc.n_embd],
              ["context", mc.block_size],
              ["vocab", mc.vocab_size],
              ["norm", mc.norm],
              ["positions", mc.pos_encoding],
              ["attention", mc.attn_impl],
              ["dropout", mc.dropout],
              ["tied embeddings", String(mc.tie_embeddings)],
            ]}
          />
          <KeyValueCard title="Training" rows={Object.entries(tc).map(([k, v]) => [k, String(v)])} />
          <KeyValueCard
            title="Data & environment"
            rows={[
              ["dataset", d.dataset.identity],
              ["train tokens", formatInt(d.dataset.train_tokens)],
              ["val tokens", formatInt(d.dataset.val_tokens)],
              ["tokenizer", d.tokenizer.identity],
              ...Object.entries(d.environment).map(([k, v]) => [k, String(v)] as [string, string]),
              ["created", formatDate(d.created_at)],
            ]}
          />
        </div>
      </div>
    </div>
  );
}

function KeyValueCard({ title, rows }: { title: string; rows: [string, string | number][] }) {
  return (
    <Card>
      <CardHeader title={title} />
      <dl className="divide-y divide-border text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 px-4 py-1.5">
            <dt className="text-faint">{k}</dt>
            <dd className="min-w-0 truncate text-right font-mono text-muted" title={String(v)}>
              {String(v)}
            </dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}
