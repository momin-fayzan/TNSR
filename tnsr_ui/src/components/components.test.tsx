import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DEFAULT_GENERATION } from "../lib/app-context";
import type { Message } from "../lib/types";
import { conversation, message } from "../test/utils";
import { Composer } from "./Composer";
import { ConversationSidebar } from "./ConversationSidebar";
import { GenerationPanel } from "./GenerationPanel";
import { MessageItem } from "./MessageItem";

describe("Composer", () => {
  it("sends on Enter and inserts a newline on Shift+Enter", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    render(<Composer onSend={onSend} onStop={vi.fn()} streaming={false} />);
    const box = screen.getByLabelText("Message");
    await userEvent.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(box).toHaveValue("line one\nline two");
    expect(onSend).not.toHaveBeenCalled();
    await userEvent.keyboard("{Enter}");
    expect(onSend).toHaveBeenCalledWith("line one\nline two");
    expect(box).toHaveValue("");
  });

  it("does not send blank messages and restores text when sending fails", async () => {
    const onSend = vi.fn().mockResolvedValue(false);
    render(<Composer onSend={onSend} onStop={vi.fn()} streaming={false} />);
    const box = screen.getByLabelText("Message");
    await userEvent.type(box, "   {Enter}");
    expect(onSend).not.toHaveBeenCalled();
    await userEvent.clear(box);
    await userEvent.type(box, "keep me{Enter}");
    expect(onSend).toHaveBeenCalledWith("keep me");
    expect(box).toHaveValue("keep me");
  });

  it("shows a stop button while streaming", async () => {
    const onStop = vi.fn();
    render(<Composer onSend={vi.fn()} onStop={onStop} streaming />);
    expect(screen.queryByLabelText("Send message")).toBeNull();
    await userEvent.click(screen.getByLabelText("Stop generation"));
    expect(onStop).toHaveBeenCalled();
  });

  it("is disabled with a reason when offline", () => {
    render(<Composer onSend={vi.fn()} onStop={vi.fn()} streaming={false} disabled disabledReason="Backend offline" />);
    expect(screen.getByLabelText("Message")).toBeDisabled();
    expect(screen.getByPlaceholderText("Backend offline")).toBeInTheDocument();
  });
});

describe("MessageItem", () => {
  const assistant = (o: Partial<Message> = {}) => message({ id: "a1", role: "assistant", content: "Paris.", ...o }) as Message;

  it("copies message content", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
    render(<MessageItem message={assistant()} canRetry={false} />);
    await user.click(screen.getByLabelText("Copy message"));
    expect(writeText).toHaveBeenCalledWith("Paris.");
    expect(await screen.findByLabelText("Copied")).toBeInTheDocument();
  });

  it("offers retry only when allowed", async () => {
    const onRetry = vi.fn();
    const { rerender } = render(<MessageItem message={assistant()} canRetry={false} onRetry={onRetry} />);
    expect(screen.queryByLabelText("Retry generation")).toBeNull();
    rerender(<MessageItem message={assistant()} canRetry onRetry={onRetry} />);
    await userEvent.click(screen.getByLabelText("Retry generation"));
    expect(onRetry).toHaveBeenCalledWith("a1");
  });

  it("renders streaming, cancelled, error and stats states", () => {
    const { rerender, container } = render(<MessageItem message={assistant({ status: "streaming", content: "Par" })} canRetry={false} />);
    expect(container.querySelector(".caret")).not.toBeNull();
    expect(screen.queryByLabelText("Copy message")).toBeNull();
    rerender(<MessageItem message={assistant({ status: "cancelled" })} canRetry={false} />);
    expect(screen.getByText("stopped")).toBeInTheDocument();
    rerender(<MessageItem message={assistant({ status: "error", error: "Model unavailable" })} canRetry={false} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Model unavailable");
    rerender(
      <MessageItem
        message={assistant({
          stats: { prompt_tokens: 5, generated_tokens: 7, prompt_truncated: false, time_to_first_token_ms: 12.4, total_ms: 50, decode_tokens_per_sec: 140.2, tokens_per_sec: 120, finish_reason: "stop", cache_rebuilds: 0 },
        })}
        canRetry={false}
      />,
    );
    expect(screen.getByTitle("Generation statistics")).toHaveTextContent("7 tok · TTFT 12 ms · 140 tok/s · turn ended");
  });
});

describe("GenerationPanel", () => {
  it("rejects invalid values without propagating them", async () => {
    const onChange = vi.fn();
    render(<GenerationPanel settings={DEFAULT_GENERATION} onChange={onChange} />);
    const temp = screen.getByLabelText("Temperature");
    await userEvent.clear(temp);
    await userEvent.type(temp, "9");
    expect(screen.getByText("Must be between 0 and 5")).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalledWith(expect.objectContaining({ temperature: 9 }));
  });

  it("propagates valid values and toggles deterministic mode", async () => {
    const onChange = vi.fn();
    render(<GenerationPanel settings={DEFAULT_GENERATION} onChange={onChange} />);
    const topk = screen.getByLabelText("Top-k");
    await userEvent.clear(topk);
    await userEvent.type(topk, "5");
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ top_k: 5 }));
    await userEvent.click(screen.getByRole("switch", { name: "Deterministic decoding" }));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ deterministic: true }));
  });

  it("disables sampling controls in deterministic mode", () => {
    render(<GenerationPanel settings={{ ...DEFAULT_GENERATION, deterministic: true }} onChange={vi.fn()} />);
    expect(screen.getByLabelText("Temperature")).toBeDisabled();
    expect(screen.getByLabelText("Top-p")).toBeDisabled();
    expect(screen.getByLabelText("Max new tokens")).not.toBeDisabled();
  });
});

describe("ConversationSidebar", () => {
  const convs = [conversation({ id: "a", title: "Math questions" }), conversation({ id: "b", title: "Fox stories" })];
  const props = {
    conversations: convs as never,
    activeId: "a",
    loading: false,
    error: null,
    onSelect: vi.fn(),
    onCreate: vi.fn(),
    onRename: vi.fn().mockResolvedValue(undefined),
    onDelete: vi.fn().mockResolvedValue(undefined),
  };

  it("filters, selects and marks the active conversation", async () => {
    render(<ConversationSidebar {...props} />);
    expect(screen.getByRole("button", { name: /Math questions/ })).toHaveAttribute("aria-current", "page");
    await userEvent.type(screen.getByLabelText("Search conversations"), "fox");
    expect(screen.queryByText("Math questions")).toBeNull();
    await userEvent.click(screen.getByText("Fox stories"));
    expect(props.onSelect).toHaveBeenCalledWith("b");
  });

  it("renames with Enter and cancels with Escape", async () => {
    render(<ConversationSidebar {...props} />);
    await userEvent.click(screen.getAllByLabelText("Rename conversation")[0]);
    const input = screen.getByLabelText("Conversation title");
    await userEvent.clear(input);
    await userEvent.type(input, "Arithmetic{Enter}");
    expect(props.onRename).toHaveBeenCalledWith("a", "Arithmetic");
    await userEvent.click(screen.getAllByLabelText("Rename conversation")[1]);
    await userEvent.type(screen.getByLabelText("Conversation title"), "zzz{Escape}");
    expect(props.onRename).toHaveBeenCalledTimes(1);
  });

  it("asks for confirmation before deleting", async () => {
    render(<ConversationSidebar {...props} />);
    await userEvent.click(screen.getAllByLabelText("Delete conversation")[1]);
    expect(screen.getByText("Delete this conversation?")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Cancel delete"));
    expect(props.onDelete).not.toHaveBeenCalled();
    await userEvent.click(screen.getAllByLabelText("Delete conversation")[1]);
    await userEvent.click(screen.getByLabelText("Confirm delete"));
    expect(props.onDelete).toHaveBeenCalledWith("b");
  });

  it("shows empty and error states", () => {
    const { rerender } = render(<ConversationSidebar {...props} conversations={[]} />);
    expect(screen.getByText("No conversations yet.")).toBeInTheDocument();
    rerender(<ConversationSidebar {...props} conversations={[]} error="Cannot reach the TNSR backend" />);
    expect(screen.getByText("Cannot reach the TNSR backend")).toBeInTheDocument();
  });
});
