import { Cpu } from "lucide-react";

import { formatParams, modelLabel } from "../lib/format";
import type { ModelInfo } from "../lib/types";
import { Spinner } from "./ui";

export function ModelPicker({
  models,
  value,
  onChange,
  loading,
  disabled,
}: {
  models: ModelInfo[];
  value: string | null;
  onChange: (id: string) => void;
  loading?: boolean;
  disabled?: boolean;
}) {
  const known = value == null || models.some((m) => m.id === value);
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      {loading ? <Spinner className="size-3.5" /> : <Cpu className="size-3.5 shrink-0 text-faint" aria-hidden />}
      <select
        aria-label="Model"
        value={value ?? ""}
        disabled={disabled || models.length === 0}
        onChange={(e) => e.target.value && onChange(e.target.value)}
        className="h-7 max-w-[16rem] min-w-0 truncate rounded-md border border-border bg-surface-2 px-2 text-xs text-fg focus:border-accent focus:outline-none disabled:opacity-50"
      >
        {value == null && <option value="">{models.length ? "Select a model…" : "No models"}</option>}
        {!known && value && <option value={value}>{modelLabel(value)} (missing)</option>}
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {modelLabel(m.id)} · {formatParams(m.num_parameters)}
            {m.is_best ? " · best" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
