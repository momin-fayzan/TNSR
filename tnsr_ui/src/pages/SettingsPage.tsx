import { useEffect, useState } from "react";

import { Badge, Button, Card, CardHeader, ErrorBanner, PageHeader, Segmented, Spinner, StatusDot } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import { useApp } from "../lib/app-context";
import type { AppSettings, Theme } from "../lib/types";

export function SettingsPage() {
  const { settings, saveSettings, connection, health } = useApp();
  const [draft, setDraft] = useState<AppSettings>(settings);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const system = usePolling(() => api.system(), null, [connection]);
  useEffect(() => setDraft(settings), [settings]);

  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);

  const save = async (next = draft) => {
    setSaving(true);
    setError(null);
    try {
      await saveSettings(next);
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <PageHeader
        title="Settings"
        actions={
          <Button variant="primary" onClick={() => save()} disabled={!dirty || connection !== "online"} loading={saving}>
            {saved ? "Saved" : "Save changes"}
          </Button>
        }
      />
      <div className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-2">
        {error && (
          <div className="lg:col-span-2">
            <ErrorBanner>{error}</ErrorBanner>
          </div>
        )}
        <div className="space-y-6">
          <Card>
            <CardHeader title="Appearance" />
            <div className="flex items-center justify-between p-4">
              <span className="text-sm text-muted">Theme</span>
              <Segmented<Theme>
                label="Theme"
                value={draft.theme}
                onChange={(theme) => {
                  const next = { ...draft, theme };
                  setDraft(next);
                  if (connection === "online") save(next);
                }}
                options={[
                  { value: "dark", label: "Dark" },
                  { value: "light", label: "Light" },
                  { value: "system", label: "System" },
                ]}
              />
            </div>
          </Card>
        </div>
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="Backend"
              actions={
                <span className="flex items-center gap-2 text-xs text-muted">
                  <StatusDot tone={connection === "online" ? "ok" : connection === "offline" ? "danger" : "neutral"} /> {connection}
                </span>
              }
            />
            {system.loading && !system.data ? (
              <div className="flex justify-center p-6">
                <Spinner />
              </div>
            ) : system.data ? (
              <dl className="divide-y divide-border text-xs">
                <KV k="TNSR version" v={system.data.version} />
                <KV k="Uptime" v={health ? `${Math.round(health.uptime_seconds)}s` : "—"} />
                <KV k="Device" v={`${system.data.device.selected} — ${system.data.device.reason}`} />
                <KV k="PyTorch" v={String(system.data.system.torch)} />
                <KV k="Python" v={String(system.data.system.python)} />
                <KV k="Platform" v={String(system.data.system.platform)} />
                <KV k="CPU threads" v={String(system.data.system.cpu_threads)} />
                <KV k="Loaded model" v={system.data.model.loaded?.id ?? "none"} />
              </dl>
            ) : (
              <p className="p-4 text-sm text-muted">{system.error ?? "Unavailable"}</p>
            )}
          </Card>
          {system.data && (
            <>
              <Card>
                <CardHeader title="Storage" subtitle="Artifacts are plain files; back them up or export with `tnsr models export`." />
                <dl className="divide-y divide-border text-xs">
                  {Object.entries(system.data.storage).map(([k, v]) => (
                    <KV key={k} k={k.replace(/_/g, " ")} v={String(v)} />
                  ))}
                </dl>
              </Card>
              <Card>
                <CardHeader title="Diagnostics" subtitle="Same checks as `tnsr doctor`" />
                <ul className="divide-y divide-border">
                  {system.data.diagnostics.checks.map((c) => (
                    <li key={c.name} className="flex items-start gap-3 px-4 py-2 text-xs">
                      <Badge tone={c.status === "ok" ? "ok" : c.status === "warning" ? "warn" : "danger"} className="mt-0.5 w-14 justify-center">
                        {c.status}
                      </Badge>
                      <div className="min-w-0">
                        <div className="font-mono text-fg">{c.name}</div>
                        <div className="break-words text-muted">{c.detail}</div>
                      </div>
                    </li>
                  ))}
                </ul>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 px-4 py-2">
      <dt className="shrink-0 text-faint">{k}</dt>
      <dd className="min-w-0 truncate text-right font-mono text-muted" title={v}>
        {v}
      </dd>
    </div>
  );
}
