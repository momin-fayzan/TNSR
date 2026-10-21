import clsx from "clsx";
import { ArrowDown, FlaskConical, MessageSquare, PanelLeft, SlidersHorizontal, X } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { Composer } from "../components/Composer";
import { ConversationSidebar } from "../components/ConversationSidebar";
import { GenerationPanel } from "../components/GenerationPanel";
import { MessageItem } from "../components/MessageItem";
import { ModelPicker } from "../components/ModelPicker";
import { Button, EmptyState, ErrorBanner, IconButton, Segmented, Spinner } from "../components/ui";
import { useChat } from "../hooks/useChat";
import { api } from "../lib/api";
import { useApp } from "../lib/app-context";
import { validateSettings } from "../lib/settings";
import type { Conversation, GenerationSettings, Mode, ModelInfo } from "../lib/types";

const CHAT_EXAMPLES = ["Hello!", "What is 7 plus 5?", "What sound does a cow make?", "Tell me a story about a fox."];
const COMPLETION_EXAMPLES = ["Once upon a time", "The capital of Japan is", "One morning, a brave owl named"];

export function StudioPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { connection, settings: appSettings, saveSettings, refreshHealth, modelStatus } = useApp();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [savingDefault, setSavingDefault] = useState(false);

  const loadConversations = useCallback(async () => {
    try {
      const { conversations } = await api.conversations();
      setConversations(conversations);
      setListError(null);
    } catch (err) {
      setListError((err as Error).message);
    } finally {
      setListLoading(false);
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const res = await api.models();
      setModels(res.models);
      setModelsLoaded(true);
    } catch {
      /* offline banner covers this */
    }
  }, []);

  useEffect(() => {
    if (connection !== "online") return;
    loadConversations();
    loadModels();
  }, [connection, loadConversations, loadModels]);

  const chat = useChat(id ?? null, () => {
    loadConversations();
    refreshHealth();
  });
  const { conversation, setConversation } = chat;

  // -- scrolling: stick to bottom unless the user scrolled up ------------------------------------
  const scroller = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);
  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  };
  const scrollToBottom = (smooth = false) => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  };
  useLayoutEffect(() => {
    if (atBottom) scrollToBottom();
  }, [chat.messages, atBottom]);
  useEffect(() => {
    setAtBottom(true);
    setShowSidebar(false);
  }, [id]);

  // -- actions -----------------------------------------------------------------------------------
  const createConversation = async (mode: Mode = appSettings.default_mode) => {
    setCreating(true);
    try {
      const c = await api.createConversation({
        mode,
        model_id: modelStatus?.loaded?.id ?? appSettings.default_model_id ?? null,
        settings: appSettings.generation,
      });
      setConversations((prev) => [c, ...prev]);
      navigate(`/c/${c.id}`);
    } catch (err) {
      setListError((err as Error).message);
    } finally {
      setCreating(false);
    }
  };

  const renameConversation = async (cid: string, title: string) => {
    const updated = await api.updateConversation(cid, { title });
    setConversations((prev) => prev.map((c) => (c.id === cid ? { ...c, ...updated, message_count: c.message_count } : c)));
    if (conversation?.id === cid) setConversation({ ...conversation, title: updated.title });
  };

  const deleteConversation = async (cid: string) => {
    try {
      await api.deleteConversation(cid);
      setConversations((prev) => prev.filter((c) => c.id !== cid));
      if (cid === id) navigate("/");
    } catch (err) {
      setListError((err as Error).message);
    }
  };

  const patchConversation = useCallback(
    async (patch: { mode?: Mode; settings?: GenerationSettings; model_id?: string }) => {
      if (!conversation) return;
      setConversation({ ...conversation, ...patch });
      try {
        await api.updateConversation(conversation.id, patch);
      } catch (err) {
        setModelError((err as Error).message);
      }
    },
    [conversation, setConversation],
  );

  // Debounce settings persistence while sliders move.
  const settingsTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const changeSettings = (s: GenerationSettings) => {
    if (!conversation) return;
    setConversation({ ...conversation, settings: s });
    if (settingsTimer.current) clearTimeout(settingsTimer.current);
    if (Object.keys(validateSettings(s)).length) return;
    settingsTimer.current = setTimeout(() => api.updateConversation(conversation.id, { settings: s }).catch(() => undefined), 400);
  };

  const selectModel = async (modelId: string) => {
    setModelLoading(true);
    setModelError(null);
    try {
      await api.loadModel(modelId);
      await patchConversation({ model_id: modelId });
      await Promise.all([refreshHealth(), loadModels()]);
    } catch (err) {
      setModelError((err as Error).message);
    } finally {
      setModelLoading(false);
    }
  };

  const send = async (text: string) => {
    if (!conversation) return false;
    setAtBottom(true);
    const errors = validateSettings(conversation.settings);
    if (Object.keys(errors).length) {
      setShowSettings(true);
      return false;
    }
    return chat.send(text, conversation.settings, conversation.model_id);
  };

  const lastAssistant = [...chat.messages].reverse().find((m) => m.role === "assistant");
  const effectiveModel = conversation?.model_id ?? modelStatus?.loaded?.id ?? null;
  const noModels = modelsLoaded && models.length === 0;
  const offline = connection === "offline";

  const composerDisabledReason = offline
    ? "Backend offline"
    : noModels
      ? "Train a model first (see Lab)"
      : undefined;

  return (
    <div className="flex min-h-0 flex-1">
      {/* Sidebar: persistent on desktop, overlay on mobile */}
      <div
        className={clsx(
          "z-30 w-72 shrink-0 md:static md:block",
          showSidebar ? "fixed inset-y-0 left-0 block shadow-2xl" : "hidden",
        )}
      >
        <ConversationSidebar
          conversations={conversations}
          activeId={id ?? null}
          loading={listLoading && connection !== "offline"}
          error={listError}
          onSelect={(cid) => navigate(`/c/${cid}`)}
          onCreate={() => createConversation()}
          onRename={renameConversation}
          onDelete={deleteConversation}
          creating={creating}
        />
      </div>
      {showSidebar && <div className="fixed inset-0 z-20 bg-black/40 md:hidden" onClick={() => setShowSidebar(false)} aria-hidden />}

      <section className="flex min-w-0 flex-1 flex-col" aria-label="Chat">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-3">
          <IconButton label="Show conversations" className="md:hidden" onClick={() => setShowSidebar(true)}>
            <PanelLeft className="size-4" />
          </IconButton>
          <h2 className="min-w-0 flex-1 truncate text-sm font-medium text-fg">{conversation?.title ?? "TNSR Studio"}</h2>
          {conversation && (
            <>
              <Segmented<Mode>
                label="Generation mode"
                value={conversation.mode}
                onChange={(mode) => patchConversation({ mode })}
                options={[
                  { value: "chat", label: "Chat" },
                  { value: "completion", label: "Completion" },
                ]}
              />
              <div className="hidden sm:block">
                <ModelPicker models={models} value={effectiveModel} onChange={selectModel} loading={modelLoading} disabled={chat.streaming} />
              </div>
              <IconButton
                label={showSettings ? "Hide generation settings" : "Show generation settings"}
                aria-pressed={showSettings}
                onClick={() => setShowSettings((v) => !v)}
                className={showSettings ? "bg-surface-3 text-fg" : ""}
              >
                <SlidersHorizontal className="size-4" />
              </IconButton>
            </>
          )}
        </header>

        {(modelError || chat.sendError) && (
          <div className="px-4 pt-3">
            <ErrorBanner
              onRetry={
                chat.sendError
                  ? () => {
                      chat.clearSendError();
                      setModelError(null);
                    }
                  : undefined
              }
            >
              {chat.sendError ?? modelError}
            </ErrorBanner>
          </div>
        )}

        <div className="relative min-h-0 flex-1">
          <div ref={scroller} onScroll={onScroll} className="h-full overflow-y-auto" data-testid="message-scroller">
            {!id ? (
              <WelcomeState noModels={noModels} onCreate={createConversation} creating={creating} offline={offline} />
            ) : chat.loading && !conversation ? (
              <div className="flex justify-center py-16">
                <Spinner />
              </div>
            ) : chat.loadError ? (
              <EmptyState title="Could not open this conversation" action={<Button onClick={() => navigate("/")}>Back</Button>}>
                {chat.loadError}
              </EmptyState>
            ) : conversation && chat.messages.length === 0 ? (
              <ConversationEmpty
                mode={conversation.mode}
                noModels={noModels}
                onPick={(t) => send(t)}
                disabled={offline || noModels || chat.streaming}
              />
            ) : (
              <div className="mx-auto max-w-3xl space-y-6 px-4 py-6" aria-live="polite" aria-busy={chat.streaming}>
                {chat.messages.map((m) => (
                  <MessageItem
                    key={m.id}
                    message={m}
                    canRetry={!chat.streaming && m.id === lastAssistant?.id && !m.id.startsWith("temp-")}
                    onRetry={(mid) => chat.retry(mid, conversation?.settings)}
                  />
                ))}
              </div>
            )}
          </div>
          {!atBottom && chat.messages.length > 0 && (
            <button
              onClick={() => {
                setAtBottom(true);
                scrollToBottom(true);
              }}
              className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted shadow-lg hover:text-fg"
            >
              <ArrowDown className="size-3.5" aria-hidden /> Jump to latest
            </button>
          )}
        </div>

        {id && conversation && (
          <Composer
            onSend={send}
            onStop={chat.stop}
            streaming={chat.streaming}
            disabled={offline || noModels}
            disabledReason={composerDisabledReason}
            placeholder={conversation.mode === "chat" ? "Message TNSR…" : "Write the beginning of a text for TNSR to continue…"}
          />
        )}
      </section>

      {showSettings && conversation && (
        <aside
          aria-label="Generation settings"
          className="fixed inset-y-0 right-0 z-30 w-80 overflow-y-auto border-l border-border bg-surface p-4 shadow-2xl lg:static lg:shadow-none"
        >
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Generation</h2>
            <IconButton label="Close settings" onClick={() => setShowSettings(false)}>
              <X className="size-4" />
            </IconButton>
          </div>
          <div className="mb-4 sm:hidden">
            <ModelPicker models={models} value={effectiveModel} onChange={selectModel} loading={modelLoading} disabled={chat.streaming} />
          </div>
          <GenerationPanel
            settings={conversation.settings}
            onChange={changeSettings}
            savingDefault={savingDefault}
            onSaveDefault={async () => {
              setSavingDefault(true);
              try {
                await saveSettings({ ...appSettings, generation: conversation.settings });
              } finally {
                setSavingDefault(false);
              }
            }}
          />
          <p className="mt-6 text-xs leading-relaxed text-faint">
            {conversation.mode === "chat"
              ? "Chat mode formats the history as User:/TNSR: turns — the dialogue format in TNSR's pretraining corpus — and stops at the next User: turn. TNSR is a base model, not instruction-tuned."
              : "Completion mode feeds your text directly to the model and continues it."}
          </p>
        </aside>
      )}
    </div>
  );
}

function WelcomeState({ noModels, onCreate, creating, offline }: { noModels: boolean; onCreate: (m?: Mode) => void; creating: boolean; offline: boolean }) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-20 text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-xl border border-border bg-surface">
        <MessageSquare className="size-5 text-accent" aria-hidden />
      </div>
      <h1 className="text-xl font-semibold tracking-tight">TNSR Studio</h1>
      <p className="mt-2 text-sm text-muted">
        Chat with a Transformer trained from scratch by TNSR&apos;s own pipeline. No pretrained weights, no external APIs.
      </p>
      {noModels ? (
        <div className="mt-6 w-full rounded-lg border border-border bg-surface p-4 text-left text-sm">
          <p className="font-medium">No trained models yet</p>
          <p className="mt-1 text-muted">
            Start a training run in the Lab, or run <code className="font-mono text-xs">uv run tnsr train --config demo</code>.
          </p>
          <Link to="/lab" className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:underline">
            <FlaskConical className="size-4" aria-hidden /> Open TNSR Lab
          </Link>
        </div>
      ) : (
        <div className="mt-6 flex gap-2">
          <Button variant="primary" onClick={() => onCreate("chat")} loading={creating} disabled={offline}>
            New chat
          </Button>
          <Button onClick={() => onCreate("completion")} disabled={offline || creating}>
            New completion
          </Button>
        </div>
      )}
    </div>
  );
}

function ConversationEmpty({ mode, noModels, onPick, disabled }: { mode: Mode; noModels: boolean; onPick: (t: string) => void; disabled: boolean }) {
  const examples = mode === "chat" ? CHAT_EXAMPLES : COMPLETION_EXAMPLES;
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-16 text-center">
      <h2 className="text-base font-semibold">{mode === "chat" ? "Start the conversation" : "Start a completion"}</h2>
      <p className="mt-1.5 text-sm text-muted">
        {noModels
          ? "There is no trained model yet. Train one in the Lab first."
          : mode === "chat"
            ? "The demo model learned greetings, small arithmetic, animal facts, capitals and short stories."
            : "Type the beginning of a text and the model will continue it."}
      </p>
      {!noModels && (
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {examples.map((e) => (
            <button
              key={e}
              disabled={disabled}
              onClick={() => onPick(e)}
              className="rounded-full border border-border bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-fg disabled:opacity-40"
            >
              {e}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
