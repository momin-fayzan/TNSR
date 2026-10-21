import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";

export function Composer({
  onSend,
  onStop,
  streaming,
  disabled,
  disabledReason,
  placeholder = "Message TNSR…",
}: {
  onSend: (text: string) => Promise<boolean> | boolean | void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
  disabledReason?: string;
  placeholder?: string;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Auto-size up to ~10 lines.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const submit = async () => {
    const value = text;
    if (!value.trim() || streaming || disabled) return;
    setText("");
    const ok = await onSend(value);
    if (ok === false) setText(value); // restore on rejection so nothing typed is lost
    ref.current?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border bg-bg px-4 pt-3 pb-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-xl border border-border bg-surface px-3 py-2 shadow-sm focus-within:border-accent/60">
          <textarea
            ref={ref}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={disabled && disabledReason ? disabledReason : placeholder}
            aria-label="Message"
            maxLength={8000}
            disabled={disabled}
            className="max-h-60 min-h-[28px] flex-1 resize-none bg-transparent py-1 text-[14.5px] leading-relaxed text-fg placeholder:text-faint focus:outline-none disabled:cursor-not-allowed"
          />
          {streaming ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generation"
              title="Stop generation"
              className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-fg text-bg transition-opacity hover:opacity-85"
            >
              <Square className="size-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={disabled || !text.trim()}
              aria-label="Send message"
              title="Send (Enter)"
              className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-fg transition-opacity hover:brightness-110 disabled:opacity-35"
            >
              <ArrowUp className="size-4" />
            </button>
          )}
        </div>
        <p className="mt-1.5 text-center text-[11px] text-faint">
          Enter to send · Shift+Enter for a new line · TNSR is a small base model trained from scratch; expect simple, often wrong output.
        </p>
      </div>
    </div>
  );
}
