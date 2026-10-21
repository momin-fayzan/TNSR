import { Boxes, CheckCircle2, Gauge, Target } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { BenchmarkView, EvaluationView } from "../components/results";
import { Badge, Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner, Stat } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import { useApp } from "../lib/app-context";
import { formatBytes, formatDate, formatNumber, formatParams } from "../lib/format";
import type { ModelInfo } from "../lib/types";

export function ModelsPage() {
  const { refreshHealth } = useApp();
  const models = usePolling(() => api.models(), 10000);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const act = async (key: string, fn: () => Promise<unknown>, done?: string) => {
    setBusy(key);
    setError(null);
    setNotice(null);
    try {
      await fn();
      if (done) setNotice(done);
      await models.refresh();
      await refreshHealth();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const list = models.data?.models ?? [];
  const loading = models.data?.loading;
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <PageHeader
        title="Models"
        subtitle="Checkpoints discovered under experiments/ and checkpoints/. Metadata is read from each checkpoint's meta.json."
        actions={
          models.data?.loaded && (
            <Button onClick={() => act("unload", () => api.unloadModel())} loading={busy === "unload"}>
              Unload model
            </Button>
          )
        }
      />
      <div className="mx-auto max-w-6xl space-y-4 p-6">
        {error && <ErrorBanner>{error}</ErrorBanner>}
        {notice && (
          <div role="status" className="rounded-md border border-ok/30 bg-ok/8 px-3 py-2 text-sm text-ok">
            {notice}
          </div>
        )}
        {models.error && !models.data && <ErrorBanner onRetry={models.refresh}>{models.error}</ErrorBanner>}
        {models.loading && !models.data ? (
          <div className="flex justify-center p-10">
            <Spinner />
          </div>
        ) : list.length === 0 && !models.error ? (
          <Card>
            <EmptyState
              icon={<Boxes className="size-6" />}
              title="No checkpoints yet"
              action={
                <Link to="/lab" className="text-sm font-medium text-accent hover:underline">
                  Start a training run in the Lab
                </Link>
              }
            >
              Train the demonstration model with <code className="font-mono text-xs">uv run tnsr train --config demo</code>, or import an
              exported checkpoint with <code className="font-mono text-xs">uv run tnsr models import</code>.
            </EmptyState>
          </Card>
        ) : (
          list.map((m) => (
            <ModelCard
              key={m.id}
              model={m}
              loadingThis={loading === m.id || busy === `load-${m.id}`}
              busy={busy}
              onLoad={() => act(`load-${m.id}`, () => api.loadModel(m.id), `Loaded ${m.id}`)}
              onJob={(kind) =>
                act(`${kind}-${m.id}`, () => api.startJob(kind, { model_id: m.id }), `${kind === "evaluate" ? "Evaluation" : "Benchmark"} started — results appear here and in the Lab when it finishes.`)
              }
            />
          ))
        )}
      </div>
    </div>
  );
}

function ModelCard({
  model: m,
  loadingThis,
  busy,
  onLoad,
  onJob,
}: {
  model: ModelInfo;
  loadingThis: boolean;
  busy: string | null;
  onLoad: () => void;
  onJob: (kind: "evaluate" | "benchmark") => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const mc = m.model_config;
  return (
    <Card className={m.loaded ? "border-accent/50" : undefined}>
      <div className="flex flex-wrap items-start justify-between gap-3 px-4 pt-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="truncate font-mono text-sm font-semibold text-fg">{m.id}</h2>
            {m.loaded && (
              <Badge tone="accent">
                <CheckCircle2 className="size-3" aria-hidden /> loaded
              </Badge>
            )}
            {m.is_best && <Badge tone="ok">best</Badge>}
            {m.is_latest && <Badge>latest</Badge>}
            {!m.complete && <Badge tone="danger">incomplete</Badge>}
          </div>
          <p className="mt-1 text-xs text-faint">
            saved {formatDate(m.saved_at)} · {formatBytes(m.size_bytes)}
            {m.run_id && (
              <>
                {" · run "}
                <Link to={`/lab/runs/${encodeURIComponent(m.run_id)}`} className="text-accent hover:underline">
                  {m.run_id}
                </Link>
              </>
            )}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" onClick={() => onJob("evaluate")} loading={busy === `evaluate-${m.id}`}>
            <Target className="size-3" aria-hidden /> Evaluate
          </Button>
          <Button size="sm" onClick={() => onJob("benchmark")} loading={busy === `benchmark-${m.id}`}>
            <Gauge className="size-3" aria-hidden /> Benchmark
          </Button>
          <Button size="sm" variant={m.loaded ? "secondary" : "primary"} onClick={onLoad} loading={loadingThis} disabled={m.loaded || !m.complete}>
            {m.loaded ? "Loaded" : "Load"}
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 px-4 py-4 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Parameters" value={formatParams(m.num_parameters)} />
        <Stat label="Step" value={m.step ?? "—"} />
        <Stat label="Layers · heads · dim" value={mc ? `${mc.n_layer} · ${mc.n_head} · ${mc.n_embd}` : "—"} />
        <Stat label="Context" value={mc ? `${mc.block_size} tok` : "—"} />
        <Stat label="Tokenizer" value={`${m.tokenizer.kind} · ${m.tokenizer.vocab_size}`} hint={<span className="font-mono">{m.tokenizer.identity}</span>} />
        <Stat
          label="Val loss (training eval)"
          value={formatNumber(m.metrics?.val_loss)}
          hint={m.metrics?.val_perplexity != null ? `ppl ${formatNumber(m.metrics.val_perplexity, 3)}` : undefined}
        />
      </div>
      {(m.evaluation || m.benchmark) && (
        <div className="border-t border-border">
          <button onClick={() => setExpanded((v) => !v)} aria-expanded={expanded} className="w-full px-4 py-2 text-left text-xs font-medium text-muted hover:text-fg">
            {expanded ? "Hide" : "Show"} evaluation & benchmark results
            {m.evaluation && ` · ppl ${formatNumber(m.evaluation.metrics.perplexity, 3)} · ${formatNumber(m.evaluation.metrics.bits_per_byte)} bits/byte`}
          </button>
          {expanded && (
            <div className="grid gap-6 px-4 pb-4 lg:grid-cols-2">
              <div>{m.evaluation ? <EvaluationView result={m.evaluation} /> : <p className="text-sm text-muted">Not evaluated yet.</p>}</div>
              <div>{m.benchmark ? <BenchmarkView result={m.benchmark} /> : <p className="text-sm text-muted">Not benchmarked yet.</p>}</div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
