import { FlaskConical, GitCompare, Plus, X } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { DatasetsPanel, JobRow, NewRunForm, ProgressBar, RunStateBadge } from "../components/lab";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import { formatDuration, formatNumber, formatParams, formatRelative } from "../lib/format";

export function LabPage() {
  const navigate = useNavigate();
  const [showForm, setShowForm] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const jobs = usePolling(() => api.jobs(), 3000);
  const anyActive = jobs.data?.jobs.some((j) => ["running", "pending", "stopping"].includes(j.status)) ?? false;
  const experiments = usePolling(() => api.experiments(), anyActive ? 3000 : 15000);
  const datasets = usePolling(() => api.datasets(), anyActive ? 5000 : null);
  const presets = usePolling(() => api.presets(), null);

  const refreshAll = () => {
    jobs.refresh();
    experiments.refresh();
    datasets.refresh();
  };

  const cancelJob = async (id: string) => {
    await api.cancelJob(id).catch(() => undefined);
    jobs.refresh();
  };

  const toggle = (id: string) => setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-8)));
  const exps = experiments.data?.experiments ?? [];
  const jobList = jobs.data?.jobs ?? [];

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <PageHeader
        title="TNSR Lab"
        subtitle="Training runs, evaluations and benchmarks — every number here comes from files written by TNSR's own pipeline."
        actions={
          <>
            {selected.length >= 1 && (
              <Button onClick={() => navigate(`/lab/compare?ids=${selected.map(encodeURIComponent).join(",")}`)}>
                <GitCompare className="size-4" aria-hidden /> Compare ({selected.length})
              </Button>
            )}
            <Button variant="primary" onClick={() => setShowForm((v) => !v)}>
              {showForm ? <X className="size-4" aria-hidden /> : <Plus className="size-4" aria-hidden />}
              {showForm ? "Close" : "New training run"}
            </Button>
          </>
        }
      />
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        {showForm && (
          <Card>
            <CardHeader title="New training run" subtitle="Runs as a background process; the chat stays responsive. Stop saves a checkpoint you can resume." />
            <NewRunForm
              presets={presets.data?.presets ?? []}
              datasets={datasets.data?.datasets ?? []}
              onStarted={() => {
                setShowForm(false);
                refreshAll();
              }}
            />
          </Card>
        )}

        {jobList.length > 0 && (
          <Card>
            <CardHeader title="Jobs" subtitle="Training, evaluation, benchmark and dataset jobs" />
            <ul>
              {jobList.slice(0, 8).map((j) => (
                <JobRow key={j.id} job={j} onCancel={cancelJob} />
              ))}
            </ul>
          </Card>
        )}

        <Card>
          <CardHeader title="Experiments" subtitle="Select runs to compare. Click a run for curves, checkpoints and evaluations." />
          {experiments.error && (
            <div className="p-4">
              <ErrorBanner onRetry={experiments.refresh}>{experiments.error}</ErrorBanner>
            </div>
          )}
          {experiments.loading && !experiments.data ? (
            <div className="flex justify-center p-8">
              <Spinner />
            </div>
          ) : exps.length === 0 ? (
            <EmptyState
              icon={<FlaskConical className="size-6" />}
              title="No experiments yet"
              action={
                <Button variant="primary" onClick={() => setShowForm(true)}>
                  Train the demo model
                </Button>
              }
            >
              Training runs appear here with real loss curves once they start. You can also run{" "}
              <code className="font-mono text-xs">uv run tnsr train --config demo</code>.
            </EmptyState>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-[11px] tracking-wide text-faint uppercase">
                    <th className="w-8 py-2 pl-4" aria-label="Select" />
                    <th className="py-2 pr-4 font-medium">Run</th>
                    <th className="py-2 pr-4 font-medium">State</th>
                    <th className="py-2 pr-4 font-medium">Progress</th>
                    <th className="py-2 pr-4 font-medium">Params</th>
                    <th className="py-2 pr-4 font-medium">Best val loss</th>
                    <th className="py-2 pr-4 font-medium">Dataset / tokenizer</th>
                    <th className="py-2 pr-4 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {exps.map((e) => {
                    const st = e.status;
                    const progress = st.step != null && st.max_steps ? st.step / st.max_steps : 0;
                    return (
                      <tr key={e.id} className="border-b border-border last:border-b-0 hover:bg-surface-2/60">
                        <td className="py-2.5 pl-4">
                          <input
                            type="checkbox"
                            aria-label={`Select ${e.id} for comparison`}
                            checked={selected.includes(e.id)}
                            onChange={() => toggle(e.id)}
                            disabled={e.missing}
                            className="accent-[var(--accent)]"
                          />
                        </td>
                        <td className="py-2.5 pr-4">
                          {e.missing ? (
                            <span className="font-mono text-xs text-faint line-through">{e.id}</span>
                          ) : (
                            <Link to={`/lab/runs/${encodeURIComponent(e.id)}`} className="font-mono text-xs text-fg hover:text-accent hover:underline">
                              {e.id}
                            </Link>
                          )}
                          {e.config.description && <div className="mt-0.5 max-w-xs truncate text-xs text-faint">{e.config.description}</div>}
                        </td>
                        <td className="py-2.5 pr-4">
                          <RunStateBadge state={st.state} />
                        </td>
                        <td className="py-2.5 pr-4">
                          <div className="w-36">
                            <ProgressBar value={progress} />
                            <div className="mt-1 font-mono text-[11px] text-faint">
                              {st.step ?? 0}/{st.max_steps ?? "?"} · {formatDuration(st.elapsed_seconds)}
                              {st.state === "running" && st.eta_seconds != null && ` · ETA ${formatDuration(st.eta_seconds)}`}
                            </div>
                          </div>
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-xs">{formatParams(e.num_parameters)}</td>
                        <td className="py-2.5 pr-4 font-mono text-xs">{formatNumber(st.best_val_loss)}</td>
                        <td className="py-2.5 pr-4 text-xs text-muted">
                          {e.dataset?.name} <Badge>{e.tokenizer ? `${e.tokenizer.kind}/${e.tokenizer.vocab_size}` : "?"}</Badge>
                        </td>
                        <td className="py-2.5 pr-4 text-xs text-faint">{formatRelative(e.created_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <DatasetsPanel datasets={datasets.data?.datasets ?? []} onPrepared={refreshAll} />
      </div>
    </div>
  );
}
