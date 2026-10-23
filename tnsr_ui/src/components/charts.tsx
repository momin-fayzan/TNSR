import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { MetricRecord } from "../lib/types";

// Series colors chosen to stay distinguishable in both themes.
export const SERIES = ["#38bdf8", "#a78bfa", "#f59e0b", "#34d399", "#f472b6", "#94a3b8", "#fb7185", "#22d3ee"];

const axis = { stroke: "var(--faint)", fontSize: 11, tickLine: false };

function ChartFrame({ children, height = 240 }: { children: React.ReactElement; height?: number }) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}

const tooltipStyle = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    fontSize: 12,
    color: "var(--fg)",
  },
  labelStyle: { color: "var(--muted)" },
};

const fmt = (v: unknown) => (typeof v === "number" ? Number(v.toPrecision(4)).toString() : String(v));

export function LossChart({ metrics, height, log = false }: { metrics: MetricRecord[]; height?: number; log?: boolean }) {
  // Merge train-step loss and periodic eval losses on a shared step axis.
  const byStep = new Map<number, Record<string, number>>();
  for (const m of metrics) {
    const row = byStep.get(m.step) ?? { step: m.step };
    if (m.type === "train" && m.loss != null) row.train = m.loss;
    if (m.type === "eval") {
      if (m.val_loss != null) row.val = m.val_loss;
      if (m.train_loss != null) row.train_eval = m.train_loss;
    }
    byStep.set(m.step, row);
  }
  const data = [...byStep.values()].sort((a, b) => a.step - b.step);
  return (
    <ChartFrame height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...axis} />
        <YAxis {...axis} width={48} tickFormatter={fmt} domain={["auto", "auto"]} scale={log ? "log" : "auto"} allowDataOverflow={log} />
        <Tooltip {...tooltipStyle} formatter={fmt} labelFormatter={(s) => `step ${s}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line name="train loss (batch)" dataKey="train" stroke={SERIES[0]} dot={false} strokeWidth={1.5} connectNulls isAnimationActive={false} strokeOpacity={0.7} />
        <Line name="train loss (eval)" dataKey="train_eval" stroke={SERIES[2]} dot={{ r: 2 }} strokeWidth={1.5} connectNulls isAnimationActive={false} />
        <Line name="val loss" dataKey="val" stroke={SERIES[1]} dot={{ r: 2.5 }} strokeWidth={2} connectNulls isAnimationActive={false} />
      </LineChart>
    </ChartFrame>
  );
}

export function SeriesChart({
  metrics,
  field,
  name,
  color = SERIES[3],
  height = 180,
  type = "train",
}: {
  metrics: MetricRecord[];
  field: keyof MetricRecord;
  name: string;
  color?: string;
  height?: number;
  type?: MetricRecord["type"];
}) {
  const data = metrics.filter((m) => m.type === type && m[field] != null).map((m) => ({ step: m.step, v: m[field] as number }));
  return (
    <ChartFrame height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...axis} />
        <YAxis {...axis} width={56} tickFormatter={fmt} />
        <Tooltip {...tooltipStyle} formatter={fmt} labelFormatter={(s) => `step ${s}`} />
        <Line name={name} dataKey="v" stroke={color} dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ChartFrame>
  );
}

export function CompareValChart({ runs }: { runs: { id: string; metrics: MetricRecord[] }[] }) {
  const byStep = new Map<number, Record<string, number>>();
  for (const run of runs) {
    for (const m of run.metrics) {
      if (m.type !== "eval" || m.val_loss == null) continue;
      const row = byStep.get(m.step) ?? { step: m.step };
      row[run.id] = m.val_loss;
      byStep.set(m.step, row);
    }
  }
  const data = [...byStep.values()].sort((a, b) => a.step - b.step);
  return (
    <ChartFrame height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...axis} />
        <YAxis {...axis} width={48} tickFormatter={fmt} domain={["auto", "auto"]} />
        <Tooltip {...tooltipStyle} formatter={fmt} labelFormatter={(s) => `step ${s}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {runs.map((r, i) => (
          <Line key={r.id} name={r.id} dataKey={r.id} stroke={SERIES[i % SERIES.length]} dot={{ r: 2 }} strokeWidth={2} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ChartFrame>
  );
}
