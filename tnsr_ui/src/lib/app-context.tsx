import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api } from "./api";
import type { AppSettings, GenerationSettings, Health, ModelStatus, Theme } from "./types";

export const DEFAULT_GENERATION: GenerationSettings = {
  max_new_tokens: 160,
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  repetition_penalty: 1.0,
  deterministic: false,
  seed: null,
  ignore_eos: false,
};

export const DEFAULT_SETTINGS: AppSettings = {
  theme: "dark",
  generation: DEFAULT_GENERATION,
  default_model_id: null,
  default_mode: "chat",
};

export type Connection = "connecting" | "online" | "offline";

interface AppState {
  connection: Connection;
  health: Health | null;
  modelStatus: ModelStatus | null;
  settings: AppSettings;
  settingsLoaded: boolean;
  saveSettings: (s: AppSettings) => Promise<void>;
  refreshHealth: () => Promise<void>;
}

const Ctx = createContext<AppState | null>(null);

const THEME_KEY = "tnsr.theme";

export function applyTheme(theme: Theme) {
  const dark = theme === "dark" || (theme === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", Boolean(dark));
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* storage may be unavailable */
  }
}

export function storedTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    if (t === "dark" || t === "light" || t === "system") return t;
  } catch {
    /* ignore */
  }
  return "dark";
}

export function AppProvider({ children, pollMs = 5000 }: { children: ReactNode; pollMs?: number }) {
  const [connection, setConnection] = useState<Connection>("connecting");
  const [health, setHealth] = useState<Health | null>(null);
  const [settings, setSettings] = useState<AppSettings>({ ...DEFAULT_SETTINGS, theme: storedTheme() });
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const mounted = useRef(true);

  const refreshHealth = useCallback(async () => {
    try {
      const h = await api.health();
      if (!mounted.current) return;
      setHealth(h);
      setConnection("online");
    } catch {
      if (!mounted.current) return;
      setConnection("offline");
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    refreshHealth();
    const t = setInterval(refreshHealth, pollMs);
    return () => {
      mounted.current = false;
      clearInterval(t);
    };
  }, [refreshHealth, pollMs]);

  useEffect(() => {
    if (connection !== "online" || settingsLoaded) return;
    api
      .getSettings()
      .then((s) => {
        setSettings(s);
        setSettingsLoaded(true);
      })
      .catch(() => undefined);
  }, [connection, settingsLoaded]);

  useEffect(() => {
    applyTheme(settings.theme);
    if (settings.theme !== "system") return;
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const listener = () => applyTheme("system");
    mq?.addEventListener?.("change", listener);
    return () => mq?.removeEventListener?.("change", listener);
  }, [settings.theme]);

  const saveSettings = useCallback(async (s: AppSettings) => {
    const saved = await api.putSettings(s);
    setSettings(saved);
  }, []);

  const value = useMemo<AppState>(
    () => ({
      connection,
      health,
      modelStatus: health?.model ?? null,
      settings,
      settingsLoaded,
      saveSettings,
      refreshHealth,
    }),
    [connection, health, settings, settingsLoaded, saveSettings, refreshHealth],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
  return ctx;
}
