// Evaluation and benchmark result views shared by the Lab and Models pages.
import type { BenchmarkResult, EvaluationResult } from "../lib/types";
import { formatDate, formatInt, formatNumber } from "../lib/format";
import { Badge } from "./ui";

export function EvaluationView({ result }: { result: EvaluationResult }) {
  const m = result.metrics;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Val loss" value={formatNumber(m.val_loss)} />
        <Metric label="Perplexity" value={formatNumber(m.perplexity, 3)} />
        <Metric label="Bits / byte" value={formatNumber(m.bits_per_byte)} />
        <Metric label="Tokens scored" value={formatInt(m.tokens_evaluated)} />
      </div>
      <p className="text-xs text-faint">
        {result.checkpoint_name} · {result.dataset.identity} (val split{m.covered_full_split ? ", full" : ", capped"}) · {result.tokenizer.identity} ·{" "}
        {result.device} · {result.seconds}s · {formatDate(result.created_at)}
      </p>
      {result.samples?.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-medium tracking-wide text-faint uppercase">Samples (T=0.8, top-k 40, fixed seeds)</div>
          {result.samples.map((s, i) => (
            <div key={i} className="rounded-md border border-border bg-bg p-2.5 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              <span className="text-muted">{s.prompt}</span>
              <span className="text-fg">{s.completion}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function BenchmarkView({ result }: { result: BenchmarkResult }) {
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-faint">
              <th className="py-1.5 pr-3 font-medium">Prompt tok</th>
              <th className="py-1.5 pr-3 font-medium">New tok</th>
              <th className="py-1.5 pr-3 font-medium">KV cache</th>
              <th className="py-1.5 pr-3 font-medium">TTFT (median)</th>
              <th className="py-1.5 pr-3 font-medium">Decode tok/s (median)</th>
              <th className="py-1.5 pr-3 font-medium">Total</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {result.results.map((r, i) => (
              <tr key={i} className="border-b border-border/60 last:border-b-0">
                <td className="py-1.5 pr-3">{r.prompt_tokens}</td>
                <td className="py-1.5 pr-3">{r.new_tokens}</td>
                <td className="py-1.5 pr-3">{r.kv_cache ? <Badge tone="accent">on</Badge> : <Badge>off</Badge>}</td>
                <td className="py-1.5 pr-3">{r.ttft_ms_median.toFixed(1)} ms</td>
                <td className="py-1.5 pr-3">{r.decode_tokens_per_sec_median.toFixed(0)}</td>
                <td className="py-1.5 pr-3">{r.total_ms_median.toFixed(0)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-faint">
        {result.checkpoint_name} · {result.device} · {result.threads} threads · {result.repeats} repeats · peak memory{" "}
        {Object.entries(result.memory)
          .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
          .join(", ")}{" "}
        · {formatDate(result.created_at)}
      </p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-surface-2 px-3 py-2">
      <div className="text-[11px] text-faint">{label}</div>
      <div className="mt-0.5 font-mono text-sm text-fg tabular-nums">{value}</div>
    </div>
  );
}
