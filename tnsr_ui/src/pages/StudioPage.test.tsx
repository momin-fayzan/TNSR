import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { controllableStream, conversation, HEALTH, jsonResponse, message, MODEL, mockFetch, SETTINGS, sse } from "../test/utils";

afterEach(() => vi.unstubAllGlobals());

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

function baseRoutes(extra: Record<string, Parameters<typeof mockFetch>[0][string]> = {}) {
  return {
    "GET /api/health": () => jsonResponse(HEALTH),
    "GET /api/settings": () => jsonResponse(SETTINGS),
    "GET /api/conversations": () => jsonResponse({ conversations: [conversation()] }),
    "GET /api/conversations/c1": () => jsonResponse(conversation()),
    "GET /api/models": () => jsonResponse({ models: [MODEL], loaded: null, loading: null, error: null }),
    ...extra,
  };
}

const assistantMsg = (o: Record<string, unknown> = {}) => message({ id: "a1", seq: 2, role: "assistant", content: "", status: "streaming", ...o });

describe("Studio chat", () => {
  it("streams tokens into the transcript as they arrive", async () => {
    const stream = controllableStream();
    const { calls } = mockFetch(baseRoutes({ "POST /api/conversations/c1/messages": () => stream.response }));
    renderAt("/c/c1");
    const box = await screen.findByLabelText("Message");
    await waitFor(() => expect(box).not.toBeDisabled());
    await userEvent.type(box, "What is 2 plus 2?{Enter}");

    expect(await screen.findByText("What is 2 plus 2?")).toBeInTheDocument();
    stream.push(sse("start", { generation_id: "g1", user_message: message({ content: "What is 2 plus 2?" }), assistant_message: assistantMsg() }));
    stream.push(sse("token", { text: "2 plus 2" }));
    const reply = () => screen.getByRole("article", { name: "TNSR response" });
    await waitFor(() => expect(reply()).toHaveTextContent("2 plus 2"));
    expect(reply()).toHaveAttribute("data-status", "streaming");
    expect(screen.getByLabelText("Stop generation")).toBeInTheDocument();

    stream.push(sse("token", { text: " is 4." }));
    await waitFor(() => expect(reply()).toHaveTextContent("2 plus 2 is 4."));
    const stats = { prompt_tokens: 9, generated_tokens: 6, prompt_truncated: false, time_to_first_token_ms: 20, total_ms: 80, decode_tokens_per_sec: 90, tokens_per_sec: 75, finish_reason: "stop", cache_rebuilds: 0 };
    stream.push(sse("done", { message: assistantMsg({ content: "2 plus 2 is 4.", status: "complete", stats }), stats }));
    stream.close();
    await waitFor(() => expect(screen.getByRole("article", { name: "TNSR response" })).toHaveAttribute("data-status", "complete"));
    expect(screen.getByTitle("Generation statistics")).toHaveTextContent("6 tok");
    expect(screen.getByLabelText("Send message")).toBeInTheDocument();

    const sent = calls.find((c) => c.method === "POST" && c.url.endsWith("/messages"));
    expect(sent?.body).toMatchObject({ content: "What is 2 plus 2?", settings: SETTINGS.generation, model_id: MODEL.id });
  });

  it("stops a generation through the cancel endpoint", async () => {
    const stream = controllableStream();
    const { calls } = mockFetch(
      baseRoutes({
        "POST /api/conversations/c1/messages": () => stream.response,
        "POST /api/generations/g1/cancel": () => {
          queueMicrotask(() => {
            stream.push(sse("done", { message: assistantMsg({ content: "Once", status: "cancelled" }), stats: { finish_reason: "cancelled" } }));
            stream.close();
          });
          return jsonResponse({ cancelled: true });
        },
      }),
    );
    renderAt("/c/c1");
    const box = await screen.findByLabelText("Message");
    await waitFor(() => expect(box).not.toBeDisabled());
    await userEvent.type(box, "Tell me a story{Enter}");
    stream.push(sse("start", { generation_id: "g1", user_message: message({ content: "Tell me a story" }), assistant_message: assistantMsg() }));
    stream.push(sse("token", { text: "Once" }));
    await screen.findByText("Once");
    await userEvent.click(screen.getByLabelText("Stop generation"));
    await waitFor(() => expect(screen.getByRole("article", { name: "TNSR response" })).toHaveAttribute("data-status", "cancelled"));
    expect(calls.some((c) => c.url === "/api/generations/g1/cancel")).toBe(true);
    expect(within(screen.getByRole("article", { name: "TNSR response" })).getByText("stopped")).toBeInTheDocument();
  });

  it("shows server errors and keeps the typed text", async () => {
    mockFetch(baseRoutes({ "POST /api/conversations/c1/messages": () => jsonResponse({ detail: "Model unavailable: checkpoint deleted" }, 409) }));
    renderAt("/c/c1");
    const box = await screen.findByLabelText("Message");
    await waitFor(() => expect(box).not.toBeDisabled());
    await userEvent.type(box, "hi{Enter}");
    expect(await screen.findByText("Model unavailable: checkpoint deleted")).toBeInTheDocument();
    expect(box).toHaveValue("hi");
    expect(screen.queryByRole("article")).toBeNull();
  });

  it("renders persisted history and retries the last response", async () => {
    const stream = controllableStream();
    const history = [message({ content: "Hello!" }), message({ id: "a1", seq: 2, role: "assistant", content: "Hi there." })];
    const { calls } = mockFetch(
      baseRoutes({
        "GET /api/conversations/c1": () => jsonResponse(conversation({ messages: history, title: "Greeting" })),
        "POST /api/conversations/c1/messages/a1/retry": () => stream.response,
      }),
    );
    renderAt("/c/c1");
    expect(await screen.findByText("Hi there.")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Retry generation"));
    stream.push(sse("start", { generation_id: "g2", assistant_message: assistantMsg({ id: "a2" }) }));
    stream.push(sse("token", { text: "Hello again!" }));
    stream.push(sse("done", { message: assistantMsg({ id: "a2", content: "Hello again!", status: "complete" }), stats: null }));
    stream.close();
    expect(await screen.findByText("Hello again!")).toBeInTheDocument();
    expect(screen.queryByText("Hi there.")).toBeNull();
    expect(calls.some((c) => c.url.endsWith("/a1/retry"))).toBe(true);
  });

  it("creates a conversation from the welcome screen", async () => {
    const { calls } = mockFetch(
      baseRoutes({
        "GET /api/conversations": () => jsonResponse({ conversations: [] }),
        "POST /api/conversations": () => jsonResponse(conversation({ id: "new1" }), 201),
        "GET /api/conversations/new1": () => jsonResponse(conversation({ id: "new1" })),
      }),
    );
    renderAt("/");
    await userEvent.click(await screen.findByRole("button", { name: "New chat" }));
    expect(await screen.findByText("Start the conversation")).toBeInTheDocument();
    expect(calls.find((c) => c.method === "POST" && c.url === "/api/conversations")?.body).toMatchObject({ mode: "chat" });
  });

  it("explains how to get a model when none exist", async () => {
    mockFetch(baseRoutes({ "GET /api/models": () => jsonResponse({ models: [], loaded: null, loading: null, error: null }) }));
    renderAt("/");
    expect(await screen.findByText("No trained models yet")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Open TNSR Lab/ })).toHaveAttribute("href", "/lab");
  });

  it("shows an offline banner when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    renderAt("/");
    expect(await screen.findByText(/Cannot reach the TNSR backend/)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Backend offline");
  });
});

describe("Lab", () => {
  it("shows an empty state before experiments exist", async () => {
    mockFetch(
      baseRoutes({
        "GET /api/jobs": () => jsonResponse({ jobs: [] }),
        "GET /api/experiments": () => jsonResponse({ experiments: [] }),
        "GET /api/datasets": () => jsonResponse({ datasets: [] }),
        "GET /api/presets": () => jsonResponse({ presets: [] }),
      }),
    );
    renderAt("/lab");
    expect(await screen.findByText("No experiments yet")).toBeInTheDocument();
    expect(screen.getByText("No datasets prepared")).toBeInTheDocument();
  });
});
