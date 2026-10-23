import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { HEALTH, jsonResponse, MODEL, mockFetch, SETTINGS } from "../test/utils";

afterEach(() => vi.unstubAllGlobals());

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const AFTER = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
const CHESS_MODEL = { ...MODEL, id: "experiments/chess-1/checkpoints/step_0003000", tokenizer: { kind: "chess", vocab_size: 2001, identity: "chess-2001-x" }, recommended: true };

function view(over: Record<string, unknown> = {}) {
  return {
    id: "g1",
    title: "You (white) vs TNSR",
    model_id: CHESS_MODEL.id,
    human_color: "white",
    start_fen: START,
    moves: [],
    result: null,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_000,
    state: { fen: START, turn: "white", moves: [], san: [], legal: ["e2e4", "d2d4", "g1f3"], check: false, last_move: null, outcome: null },
    engine_move: null,
    analysis: { model_id: CHESS_MODEL.id, legal_count: 20, moves: [{ uci: "e2e4", san: "e4", win_prob: 0.54 }, { uci: "d2d4", san: "d4", win_prob: 0.53 }] },
    ...over,
  };
}

function routes(extra: Record<string, Parameters<typeof mockFetch>[0][string]> = {}) {
  return {
    "GET /api/health": () => jsonResponse(HEALTH),
    "GET /api/settings": () => jsonResponse(SETTINGS),
    "GET /api/chess/models": () => jsonResponse({ models: [CHESS_MODEL] }),
    "GET /api/chess/games": () => jsonResponse({ games: [] }),
    "GET /api/chess/games/g1": () => jsonResponse(view()),
    ...extra,
  };
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

const square = (sq: string) => document.querySelector(`[data-square="${sq}"]`) as HTMLElement;

describe("Play", () => {
  it("starts a game against the recommended model", async () => {
    const { calls } = mockFetch(routes({ "POST /api/chess/games": () => jsonResponse(view(), 201) }));
    renderAt("/");
    await userEvent.click(await screen.findByRole("button", { name: "Play as white" }));
    expect(await screen.findByTestId("chess-board")).toBeInTheDocument();
    expect(calls.find((c) => c.method === "POST" && c.url === "/api/chess/games")?.body).toEqual({ human_color: "white", model_id: CHESS_MODEL.id });
    expect(screen.getByText("Your move")).toBeInTheDocument();
  });

  it("sends the human move and shows TNSR's reply with its candidate scores", async () => {
    const replied = view({
      moves: ["e2e4", "e7e5"],
      state: { fen: AFTER, turn: "white", moves: ["e2e4", "e7e5"], san: ["e4", "e5"], legal: ["g1f3"], check: false, last_move: "e7e5", outcome: null },
      engine_move: { uci: "e7e5", san: "e5", win_prob: 0.47, alternatives: [{ uci: "e7e5", san: "e5", win_prob: 0.47 }, { uci: "c7c5", san: "c5", win_prob: 0.46 }] },
    });
    const { calls } = mockFetch(routes({ "POST /api/chess/games/g1/move": () => jsonResponse(replied) }));
    renderAt("/g/g1");
    await screen.findByTestId("chess-board");
    await waitFor(() => expect(square("e2")).toHaveAttribute("tabindex", "0"));
    await userEvent.click(square("e2"));
    await userEvent.click(square("e4"));
    expect(await screen.findByText("TNSR played e5")).toBeInTheDocument();
    expect(calls.find((c) => c.url === "/api/chess/games/g1/move")?.body).toEqual({ move: "e2e4" });
    expect(square("e5")).toHaveAttribute("aria-label", "e5, black pawn");
    expect(screen.getByText("c5")).toBeInTheDocument(); // alternative considered
  });

  it("shows server errors (e.g. illegal move) and the game result", async () => {
    const over = view({
      result: "0-1",
      state: { fen: START, turn: "white", moves: [], san: [], legal: [], check: true, last_move: null, outcome: { result: "0-1", termination: "checkmate" } },
      analysis: null,
    });
    mockFetch(routes({ "GET /api/chess/games/g1": () => jsonResponse(over) }));
    renderAt("/g/g1");
    expect(await screen.findByText(/TNSR wins by checkmate/)).toBeInTheDocument();
    expect(square("e2")).toHaveAttribute("tabindex", "-1"); // board is locked
  });

  it("explains how to train a model when none exist", async () => {
    mockFetch(routes({ "GET /api/chess/models": () => jsonResponse({ models: [] }) }));
    renderAt("/");
    expect(await screen.findByText("No chess models yet")).toBeInTheDocument();
  });

  it("shows an offline banner when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    renderAt("/");
    expect(await screen.findByText(/Cannot reach the TNSR backend/)).toBeInTheDocument();
  });

  it("hints are hidden until requested", async () => {
    mockFetch(routes());
    renderAt("/g/g1");
    await screen.findByTestId("chess-board");
    expect(await screen.findByText(/Hidden\. Turn on/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("switch", { name: "Show TNSR's suggestions" }));
    expect(screen.getByText("54.0%")).toBeInTheDocument();
  });
});

describe("Lab", () => {
  it("shows empty states before experiments exist", async () => {
    mockFetch(
      routes({
        "GET /api/jobs": () => jsonResponse({ jobs: [] }),
        "GET /api/experiments": () => jsonResponse({ experiments: [] }),
        "GET /api/datasets": () => jsonResponse({ datasets: [] }),
        "GET /api/presets": () => jsonResponse({ presets: [] }),
      }),
    );
    renderAt("/lab");
    expect(await screen.findByText("No experiments yet")).toBeInTheDocument();
    expect(screen.getByText("No datasets yet")).toBeInTheDocument();
  });
});
