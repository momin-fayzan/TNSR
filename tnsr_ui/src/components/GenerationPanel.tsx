import { RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

import { DEFAULT_GENERATION } from "../lib/app-context";
import { LIMITS, validateField, validateSeed, type NumericKey } from "../lib/settings";
import type { GenerationSettings } from "../lib/types";
import { Button, Field, TextInput, Toggle } from "./ui";

const LABELS: Record<NumericKey, { label: string; hint: string }> = {
  temperature: { label: "Temperature", hint: "Higher is more random; 0 is greedy." },
  top_k: { label: "Top-k", hint: "Sample from the k most likely tokens (0 = off)." },
  top_p: { label: "Top-p", hint: "Nucleus sampling mass (1 = off)." },
  max_new_tokens: { label: "Max new tokens", hint: "Upper bound on generated tokens." },
  repetition_penalty: { label: "Repetition penalty", hint: "Penalize tokens already in context (1 = off)." },
};

function NumberSetting({
  k,
  value,
  onCommit,
  disabled,
  slider,
}: {
  k: NumericKey;
  value: number;
  onCommit: (v: number) => void;
  disabled?: boolean;
  slider?: boolean;
}) {
  const [draft, setDraft] = useState(String(value));
  const [error, setError] = useState<string | undefined>();
  useEffect(() => {
    setDraft(String(value));
    setError(undefined);
  }, [value]);
  const lim = LIMITS[k];
  const id = `gen-${k}`;
  const change = (raw: string) => {
    setDraft(raw);
    const r = validateField(k, raw);
    setError(r.error);
    if (r.value !== undefined) onCommit(r.value);
  };
  return (
    <Field label={LABELS[k].label} hint={LABELS[k].hint} error={error} htmlFor={id}>
      <div className="flex items-center gap-2">
        {slider && (
          <input
            type="range"
            min={lim.min}
            max={k === "max_new_tokens" ? 1024 : lim.max}
            step={lim.step}
            value={Number.isFinite(Number(draft)) ? Number(draft) : value}
            onChange={(e) => change(e.target.value)}
            disabled={disabled}
            aria-label={`${LABELS[k].label} slider`}
            className="h-1 flex-1 cursor-pointer accent-[var(--accent)] disabled:opacity-40"
          />
        )}
        <TextInput
          id={id}
          inputMode="decimal"
          value={draft}
          invalid={!!error}
          disabled={disabled}
          onChange={(e) => change(e.target.value)}
          className={slider ? "w-20 font-mono" : "font-mono"}
        />
      </div>
    </Field>
  );
}

export function GenerationPanel({
  settings,
  onChange,
  onSaveDefault,
  savingDefault,
}: {
  settings: GenerationSettings;
  onChange: (s: GenerationSettings) => void;
  onSaveDefault?: () => void;
  savingDefault?: boolean;
}) {
  const [seedDraft, setSeedDraft] = useState(settings.seed == null ? "" : String(settings.seed));
  const [seedError, setSeedError] = useState<string | undefined>();
  useEffect(() => setSeedDraft(settings.seed == null ? "" : String(settings.seed)), [settings.seed]);
  const set = <K extends keyof GenerationSettings>(key: K, value: GenerationSettings[K]) => onChange({ ...settings, [key]: value });
  const greedy = settings.deterministic;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-surface-2 px-3 py-2.5">
        <div>
          <label htmlFor="gen-deterministic" className="text-sm font-medium text-fg">
            Deterministic
          </label>
          <p className="text-xs text-faint">Greedy decoding: same prompt, same output.</p>
        </div>
        <Toggle id="gen-deterministic" label="Deterministic decoding" checked={greedy} onChange={(v) => set("deterministic", v)} />
      </div>
      <NumberSetting k="temperature" value={settings.temperature} onCommit={(v) => set("temperature", v)} disabled={greedy} slider />
      <NumberSetting k="top_k" value={settings.top_k} onCommit={(v) => set("top_k", v)} disabled={greedy} />
      <NumberSetting k="top_p" value={settings.top_p} onCommit={(v) => set("top_p", v)} disabled={greedy} slider />
      <NumberSetting k="max_new_tokens" value={settings.max_new_tokens} onCommit={(v) => set("max_new_tokens", v)} slider />
      <NumberSetting k="repetition_penalty" value={settings.repetition_penalty} onCommit={(v) => set("repetition_penalty", v)} />
      <Field label="Seed" hint="Fix the seed for reproducible sampling (blank = random)." error={seedError} htmlFor="gen-seed">
        <TextInput
          id="gen-seed"
          inputMode="numeric"
          value={seedDraft}
          placeholder="random"
          invalid={!!seedError}
          disabled={greedy}
          onChange={(e) => {
            setSeedDraft(e.target.value);
            const r = validateSeed(e.target.value);
            setSeedError(r.error);
            if (r.value !== undefined) set("seed", r.value);
          }}
          className="font-mono"
        />
      </Field>
      <div className="flex items-center justify-between gap-3">
        <div>
          <label htmlFor="gen-ignore-eos" className="text-xs font-medium text-muted">
            Ignore end-of-text
          </label>
          <p className="text-xs text-faint">Keep generating past the model&apos;s end-of-text token.</p>
        </div>
        <Toggle id="gen-ignore-eos" label="Ignore end-of-text" checked={settings.ignore_eos} onChange={(v) => set("ignore_eos", v)} />
      </div>
      <div className="flex flex-wrap gap-2 pt-1">
        <Button size="sm" variant="ghost" onClick={() => onChange(DEFAULT_GENERATION)}>
          <RotateCcw className="size-3.5" aria-hidden /> Reset
        </Button>
        {onSaveDefault && (
          <Button size="sm" onClick={onSaveDefault} loading={savingDefault}>
            Save as default
          </Button>
        )}
      </div>
    </div>
  );
}
