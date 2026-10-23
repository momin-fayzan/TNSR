import clsx from "clsx";
import { Boxes, Crown, FlaskConical, Settings } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { useApp } from "../lib/app-context";
import { modelLabel } from "../lib/format";
import { StatusDot } from "./ui";

const NAV = [
  { to: "/", label: "Play", icon: Crown, match: (p: string) => p === "/" || p.startsWith("/g/") },
  { to: "/lab", label: "Lab", icon: FlaskConical },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function ConnectionIndicator({ compact = false }: { compact?: boolean }) {
  const { connection, modelStatus } = useApp();
  const tone = connection === "online" ? "ok" : connection === "offline" ? "danger" : "neutral";
  const text = connection === "online" ? "Backend online" : connection === "offline" ? "Backend offline" : "Connecting…";
  return (
    <div className="flex items-center gap-2 text-xs text-muted" role="status" aria-live="polite">
      <StatusDot tone={tone} />
      {!compact && <span>{text}</span>}
      {compact && <span className="sr-only">{text}</span>}
      {!compact && connection === "online" && (
        <span className="truncate text-faint" title={modelStatus?.loaded?.id}>
          · {modelStatus?.loading ? "loading model…" : modelStatus?.loaded ? modelLabel(modelStatus.loaded.id) : "no model loaded"}
        </span>
      )}
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const { connection } = useApp();
  const location = useLocation();
  return (
    <div className="flex h-full flex-col md:flex-row">
      <nav
        aria-label="Primary"
        className="flex shrink-0 items-center gap-1 border-b border-border bg-surface px-2 py-1.5 md:w-16 md:flex-col md:border-r md:border-b-0 md:px-0 md:py-3"
      >
        <div className="mr-2 flex size-9 items-center justify-center md:mr-0 md:mb-4" aria-hidden>
          <img src="/favicon.svg" alt="" className="size-7" />
        </div>
        {NAV.map(({ to, label, icon: Icon, match }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => {
              const active = match ? match(location.pathname) : isActive;
              return clsx(
                "group flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors md:w-12 md:flex-col md:gap-1 md:px-0 md:py-2",
                active ? "bg-surface-3 text-fg" : "text-muted hover:bg-surface-2 hover:text-fg",
              );
            }}
          >
            <Icon className="size-4.5" aria-hidden />
            <span className="md:text-[10px]">{label}</span>
          </NavLink>
        ))}
        <div className="ml-auto md:mt-auto md:ml-0">
          <ConnectionIndicator compact />
        </div>
      </nav>
      <main className="relative flex min-h-0 min-w-0 flex-1 flex-col">
        {connection === "offline" && (
          <div role="alert" className="border-b border-danger/30 bg-danger/10 px-4 py-2 text-center text-xs text-danger">
            Cannot reach the TNSR backend at <code className="font-mono">/api</code>. Start it with{" "}
            <code className="font-mono">uv run tnsr serve</code> — retrying automatically.
          </div>
        )}
        {children}
      </main>
    </div>
  );
}
