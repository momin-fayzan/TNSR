import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "../lib/api";
import type { Conversation, GenerationSettings, Message } from "../lib/types";

export interface ChatState {
  conversation: Conversation | null;
  messages: Message[];
  loading: boolean;
  loadError: string | null;
  streaming: boolean;
  sendError: string | null;
}

let tempCounter = 0;
const tempId = () => `temp-${++tempCounter}`;

function placeholder(role: Message["role"], conversationId: string, content: string, seq: number): Message {
  return {
    id: tempId(),
    conversation_id: conversationId,
    seq,
    role,
    content,
    status: role === "assistant" ? "streaming" : "complete",
    model_id: null,
    settings: null,
    stats: null,
    error: null,
    created_at: Date.now() / 1000,
  };
}

/** Conversation state + streaming send/stop/retry against the real backend. */
export function useChat(conversationId: string | null, onChanged?: () => void) {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const generationId = useRef<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const onChangedRef = useRef(onChanged);
  onChangedRef.current = onChanged;

  const load = useCallback(async () => {
    if (!conversationId) {
      setConversation(null);
      setMessages([]);
      return;
    }
    setLoading(true);
    setLoadError(null);
    try {
      const c = await api.conversation(conversationId);
      setConversation(c);
      setMessages(c.messages ?? []);
    } catch (err) {
      setLoadError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [conversationId]);

  useEffect(() => {
    load();
    return () => abort.current?.abort();
  }, [load]);

  const runStream = useCallback(
    async (
      start: (handlers: Parameters<typeof api.sendMessage>[2], signal: AbortSignal) => Promise<void>,
      tempUserId: string | null,
      tempAssistantId: string,
    ) => {
      const controller = new AbortController();
      abort.current = controller;
      setStreaming(true);
      setSendError(null);
      let assistantId = tempAssistantId;
      const patchAssistant = (fn: (m: Message) => Message) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));
      try {
        await start(
          {
            onStart: (d) => {
              generationId.current = d.generation_id;
              setMessages((prev) =>
                prev.map((m) => {
                  if (tempUserId && m.id === tempUserId && d.user_message) return d.user_message;
                  if (m.id === tempAssistantId && d.assistant_message) return { ...d.assistant_message, content: "" };
                  return m;
                }),
              );
              if (d.assistant_message) assistantId = d.assistant_message.id;
            },
            onToken: (text) => patchAssistant((m) => ({ ...m, content: m.content + text })),
            onDone: (d) => {
              if (d.message) patchAssistant(() => d.message as Message);
              else patchAssistant((m) => ({ ...m, status: "complete", stats: d.stats }));
            },
            onError: (detail, message) => {
              if (message) patchAssistant(() => message);
              else patchAssistant((m) => ({ ...m, status: "error", error: detail }));
            },
          },
          controller.signal,
        );
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          patchAssistant((m) => ({ ...m, status: "cancelled" }));
        } else {
          const msg = err instanceof ApiError ? err.message : (err as Error).message;
          setSendError(msg);
          // The request was rejected before streaming: drop the optimistic placeholders.
          setMessages((prev) => prev.filter((m) => m.id !== tempAssistantId && m.id !== tempUserId));
          if (tempUserId) throw err;
        }
      } finally {
        setStreaming(false);
        generationId.current = null;
        abort.current = null;
        onChangedRef.current?.();
      }
    },
    [],
  );

  const send = useCallback(
    async (content: string, settings?: GenerationSettings, modelId?: string | null) => {
      if (!conversationId || streaming) return false;
      const nextSeq = (messages.at(-1)?.seq ?? 0) + 1;
      const user = placeholder("user", conversationId, content, nextSeq);
      const assistant = placeholder("assistant", conversationId, "", nextSeq + 1);
      setMessages((prev) => [...prev, user, assistant]);
      try {
        await runStream(
          (handlers, signal) => api.sendMessage(conversationId, { content, settings, model_id: modelId }, handlers, signal),
          user.id,
          assistant.id,
        );
        return true;
      } catch {
        return false;
      }
    },
    [conversationId, messages, runStream, streaming],
  );

  const retry = useCallback(
    async (messageId: string, settings?: GenerationSettings) => {
      if (!conversationId || streaming) return;
      const target = messages.find((m) => m.id === messageId);
      if (!target) return;
      const assistant = placeholder("assistant", conversationId, "", target.seq);
      setMessages((prev) => [...prev.filter((m) => m.seq < target.seq), assistant]);
      await runStream(
        (handlers, signal) => api.retryMessage(conversationId, messageId, { settings }, handlers, signal),
        null,
        assistant.id,
      ).catch(() => undefined);
    },
    [conversationId, messages, runStream, streaming],
  );

  const stop = useCallback(async () => {
    const gid = generationId.current;
    const controller = abort.current;
    if (gid) {
      try {
        const res = await api.cancelGeneration(gid);
        if (res.cancelled) {
          // The server finishes the stream with a "done" event; abort only if it hangs.
          setTimeout(() => {
            if (abort.current === controller) controller?.abort();
          }, 3000);
          return;
        }
      } catch {
        /* fall through to abort */
      }
    }
    controller?.abort();
  }, []);

  return {
    conversation,
    setConversation,
    messages,
    loading,
    loadError,
    streaming,
    sendError,
    clearSendError: () => setSendError(null),
    send,
    retry,
    stop,
    reload: load,
  };
}
