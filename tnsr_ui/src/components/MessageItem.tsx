import clsx from "clsx";
import { AlertTriangle, Check, Copy, RotateCcw } from "lucide-react";
import { useState } from "react";

import type { Message } from "../lib/types";
import { Badge, IconButton } from "./ui";

const FINISH_LABEL: Record<string, string> = {
  eos: "end of text",
  stop: "turn ended",
  length: "token limit",
  cancelled: "stopped",
};

export function MessageItem({
  message,
  canRetry,
  onRetry,
}: {
  message: Message;
  canRetry: boolean;
  onRetry?: (id: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const streaming = message.status === "streaming";
  const stats = message.stats;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (e.g. insecure context) */
    }
  };

  return (
    <article
      aria-label={isUser ? "Your message" : "TNSR response"}
      data-role={message.role}
      data-status={message.status}
      className={clsx("group flex w-full", isUser ? "justify-end" : "justify-start")}
    >
      <div className={clsx("min-w-0", isUser ? "max-w-[85%]" : "w-full max-w-full")}>
        {!isUser && (
          <div className="mb-1 flex items-center gap-2 text-[11px] font-medium text-faint">
            <span className="font-mono tracking-wide text-accent">TNSR</span>
            {message.status === "cancelled" && <Badge tone="warn">stopped</Badge>}
            {message.status === "error" && <Badge tone="danger">error</Badge>}
          </div>
        )}
        <div
          className={clsx(
            "text-[14.5px] leading-relaxed break-words whitespace-pre-wrap",
            isUser ? "rounded-2xl rounded-br-md bg-surface-3 px-3.5 py-2 text-fg" : "text-fg",
          )}
        >
          {message.content}
          {streaming && <span className="caret" aria-hidden />}
          {streaming && !message.content && <span className="sr-only">Generating…</span>}
          {!isUser && !streaming && !message.content && message.status !== "error" && (
            <span className="text-sm text-faint italic">(empty response)</span>
          )}
        </div>
        {message.status === "error" && message.error && (
          <div className="mt-2 flex items-start gap-1.5 text-xs text-danger" role="alert">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            <span>{message.error}</span>
          </div>
        )}
        {!streaming && (
          <div
            className={clsx(
              "mt-1.5 flex items-center gap-1 text-[11px] text-faint",
              isUser ? "justify-end" : "justify-start",
            )}
          >
            <div className="flex opacity-60 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
              <IconButton label={copied ? "Copied" : "Copy message"} onClick={copy} disabled={!message.content}>
                {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
              </IconButton>
              {!isUser && canRetry && onRetry && (
                <IconButton label="Retry generation" onClick={() => onRetry(message.id)}>
                  <RotateCcw className="size-3.5" />
                </IconButton>
              )}
            </div>
            {!isUser && stats && (
              <span className="font-mono tabular-nums" title="Generation statistics">
                {stats.generated_tokens} tok
                {stats.time_to_first_token_ms != null && ` · TTFT ${Math.round(stats.time_to_first_token_ms)} ms`}
                {stats.decode_tokens_per_sec != null && ` · ${Math.round(stats.decode_tokens_per_sec)} tok/s`}
                {stats.finish_reason && ` · ${FINISH_LABEL[stats.finish_reason] ?? stats.finish_reason}`}
                {stats.prompt_truncated && " · context truncated"}
              </span>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
