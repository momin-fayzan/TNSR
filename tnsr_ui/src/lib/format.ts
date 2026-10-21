export function formatParams(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function formatNumber(n: number | null | undefined, digits = 4): string {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n.toFixed(digits)).toString();
}

export function formatInt(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toLocaleString("en-US");
}

export function formatBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function formatRelative(ts: number | string | null | undefined): string {
  if (ts == null) return "—";
  const ms = typeof ts === "number" ? ts * 1000 : Date.parse(normalizeIso(ts));
  if (Number.isNaN(ms)) return String(ts);
  const diff = (Date.now() - ms) / 1000;
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.round(diff / 86400)}d ago`;
  return new Date(ms).toLocaleDateString();
}

export function formatDate(ts: string | number | null | undefined): string {
  if (ts == null) return "—";
  const ms = typeof ts === "number" ? ts * 1000 : Date.parse(normalizeIso(ts));
  if (Number.isNaN(ms)) return String(ts);
  return new Date(ms).toLocaleString();
}

/** Python's %z gives "+0000"; Date.parse wants "+00:00". */
function normalizeIso(s: string): string {
  return s.replace(/([+-]\d{2})(\d{2})$/, "$1:$2");
}

export function shortId(id: string, max = 28): string {
  return id.length > max ? `${id.slice(0, max - 1)}…` : id;
}

export function modelLabel(id: string | null | undefined): string {
  if (!id) return "No model";
  const parts = id.split("/");
  if (parts[0] === "experiments" && parts.length >= 4) return `${parts[1]} · ${parts[3].replace("step_", "step ").replace(/ 0+/, " ")}`;
  if (parts[0] === "checkpoints") return parts[1];
  return id;
}
