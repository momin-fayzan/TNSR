import { AlertTriangle, ArrowLeft } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";

import { CompareValChart } from "../components/charts";
import { RunStateBadge } from "../components/lab";
import { Badge, Card, CardHeader, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import { formatInt, formatNumber, formatParams } from "../lib/format";

export function ComparePage() {
  const [params] = useSearchParams();
  const ids = (params.get("ids") ?? "").split(",").filter(Boolean);
  const cmp = usePolling(() => api.compare(ids), null, [ids.join(",")]);
  const details = usePolling(() => Promise.all(ids.map((id) => api.experiment(id))), null, [ids.join(",")]);

  if (!ids.length) {
    return (
      <EmptyState title="Nothing to compare" action={<Link to="/lab" className="text-sm text-accent hover:underline">Select runs in the Lab</Link>} />
    );
  }
  const data = cmp.data;
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <PageHeader
        title="Compare runs"
        subtitle={
          <Link to="/lab" className="inline-flex items-center gap-1 hover:text-fg">
            <ArrowLeft className="size-3.5" aria-hidden /> Lab
          </Link>
        }
      />
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        {cmp.error && <ErrorBanner>{cmp.error}</ErrorBanner>}
        {!data ? (
          <div className="flex justify-center p-10">
            <Spinner />
          </div>
        ) : (
          <>
            {data.warnings.length > 0 && (
              <div className="space-y-2 rounded-lg border border-warn/30 bg-warn/8 p-4" role="note">
                {data.warnings.map((w) => (
                  <p key={w} className="flex items-start gap-2 text-sm text-warn">
                    <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden /> {w}
                  </p>
                ))}
              </div>
            )}
            <Card>
              <CardHeader
                title="Validation loss"
                subtitle={data.comparable.loss_and_perplexity ? "Same tokenizer and dataset: losses are directly comparable" : "Different tokenizers or data: compare shapes, not values"}
              />
              <div className="p-4">{details.data ? <CompareValChart runs={details.data.map((d) => ({ id: d.id, metrics: d.metrics }))} /> : <Spinner />}</div>
            </Card>
            <Card>
              <CardHeader title="Side by side" />
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-left text-faint">
                      <th className="px-4 py-2 font-medium">Metric</th>
                      {data.runs.map((r) => (
                        <th key={r.id} className="py-2 pr-4 font-mono font-medium text-fg">
                          <Link to={`/lab/runs/${encodeURIComponent(r.id)}`} className="hover:underline">
                            {r.id}
                          </Link>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    <Row label="state" values={data.runs.map((r) => <RunStateBadge key={r.id} state={r.status.state} />)} />
                    <Row label="parameters" values={data.runs.map((r) => formatParams(r.num_parameters))} />
                    <Row label="layers / heads / d_model" values={data.runs.map((r) => `${r.model_config.n_layer} / ${r.model_config.n_head} / ${r.model_config.n_embd}`)} />
                    <Row label="context" values={data.runs.map((r) => r.model_config.block_size)} />
                    <Row label="tokenizer" values={data.runs.map((r) => `${r.tokenizer.kind}/${r.tokenizer.vocab_size}`)} />
                    <Row label="dataset" values={data.runs.map((r) => r.dataset.identity)} />
                    <Row label="steps" values={data.runs.map((r) => `${r.status.step ?? 0}/${r.status.max_steps ?? "?"}`)} />
                    <Row label="tokens seen" values={data.runs.map((r) => formatInt(r.status.tokens_seen))} />
                    <Row label="learning rate" values={data.runs.map((r) => String(r.train_config.learning_rate))} />
                    <Row
                      label="best val loss"
                      flag={!data.comparable.loss_and_perplexity}
                      values={data.runs.map((r) => formatNumber(r.best_val_loss))}
                    />
                    <Row
                      label="eval perplexity"
                      flag={!data.comparable.loss_and_perplexity}
                      values={data.runs.map((r) => (r.evaluation ? formatNumber(r.evaluation.perplexity, 3) : "not evaluated"))}
                    />
                    <Row
                      label="eval bits/byte"
                      flag={!data.comparable.bits_per_byte}
                      values={data.runs.map((r) => (r.evaluation ? formatNumber(r.evaluation.bits_per_byte) : "not evaluated"))}
                    />
                    <Row
                      label="decode tok/s (KV cache)"
                      flag={!data.comparable.benchmarks}
                      values={data.runs.map((r) => {
                        const row = r.benchmark?.results.find((x) => x.kv_cache);
                        return row ? `${row.decode_tokens_per_sec_median.toFixed(0)} (${r.benchmark?.device})` : "not benchmarked";
                      })}
                    />
                    <Row
                      label="TTFT (KV cache)"
                      flag={!data.comparable.benchmarks}
                      values={data.runs.map((r) => {
                        const row = r.benchmark?.results.find((x) => x.kv_cache);
                        return row ? `${row.ttft_ms_median.toFixed(1)} ms` : "—";
                      })}
                    />
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

function Row({ label, values, flag }: { label: string; values: React.ReactNode[]; flag?: boolean }) {
  return (
    <tr className="border-b border-border last:border-b-0">
      <td className="px-4 py-2 font-sans text-faint">
        {label} {flag && <Badge tone="warn">not comparable</Badge>}
      </td>
      {values.map((v, i) => (
        <td key={i} className="py-2 pr-4 text-muted">
          {v}
        </td>
      ))}
    </tr>
  );
}
