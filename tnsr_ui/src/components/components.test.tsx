import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChessBoard, parsePlacement } from "./ChessBoard";
import { ItemSidebar } from "./ItemSidebar";

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const START_LEGAL = ["g1h3", "g1f3", "b1c3", "b1a3", "h2h3", "g2g3", "f2f3", "e2e3", "d2d3", "c2c3", "b2b3", "a2a3", "h2h4", "g2g4", "f2f4", "e2e4", "d2d4", "c2c4", "b2b4", "a2a4"];

const square = (sq: string) => document.querySelector(`[data-square="${sq}"]`) as HTMLElement;

describe("ChessBoard", () => {
  it("parses FEN placement", () => {
    const p = parsePlacement(START);
    expect(p.e1).toBe("K");
    expect(p.d8).toBe("q");
    expect(p.e4).toBeUndefined();
    expect(Object.keys(p)).toHaveLength(32);
  });

  it("renders all squares with accessible labels in both orientations", () => {
    const { rerender } = render(<ChessBoard fen={START} />);
    expect(screen.getAllByRole("gridcell")).toHaveLength(64);
    expect(square("e1")).toHaveAttribute("aria-label", "e1, white king");
    expect(screen.getAllByRole("gridcell")[0]).toHaveAttribute("data-square", "a8");
    rerender(<ChessBoard fen={START} orientation="black" />);
    expect(screen.getAllByRole("gridcell")[0]).toHaveAttribute("data-square", "h1");
  });

  it("moves by clicking a piece then a legal target", async () => {
    const onMove = vi.fn();
    render(<ChessBoard fen={START} legal={START_LEGAL} onMove={onMove} />);
    await userEvent.click(square("e2"));
    expect(square("e2")).toHaveAttribute("aria-selected", "true");
    expect(square("e4").getAttribute("aria-label")).toContain("legal target");
    expect(square("e5").getAttribute("aria-label")).not.toContain("legal target");
    await userEvent.click(square("e5")); // not legal: just deselects
    expect(onMove).not.toHaveBeenCalled();
    await userEvent.click(square("e2"));
    await userEvent.click(square("e4"));
    expect(onMove).toHaveBeenCalledWith("e2e4");
  });

  it("ignores clicks when not interactive or on opponent pieces", async () => {
    const onMove = vi.fn();
    const { rerender } = render(<ChessBoard fen={START} legal={START_LEGAL} onMove={onMove} interactive={false} />);
    await userEvent.click(square("e2"));
    expect(square("e2")).toHaveAttribute("aria-selected", "false");
    rerender(<ChessBoard fen={START} legal={START_LEGAL} onMove={onMove} />);
    await userEvent.click(square("e7"));
    expect(square("e7")).toHaveAttribute("aria-selected", "false");
  });

  it("asks which piece to promote to", async () => {
    const onMove = vi.fn();
    const fen = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1";
    render(<ChessBoard fen={fen} legal={["e7e8q", "e7e8r", "e7e8b", "e7e8n", "e1d1"]} onMove={onMove} />);
    await userEvent.click(square("e7"));
    await userEvent.click(square("e8"));
    const dialog = screen.getByRole("dialog", { name: "Choose promotion piece" });
    await userEvent.click(within(dialog).getByLabelText("Promote to knight"));
    expect(onMove).toHaveBeenCalledWith("e7e8n");
  });
});

describe("ItemSidebar", () => {
  const items = [
    { id: "a", title: "Morning blitz", meta: "2 moves" },
    { id: "b", title: "Endgame practice" },
  ];
  const props = () => ({
    items,
    activeId: "a",
    loading: false,
    error: null,
    onSelect: vi.fn(),
    onCreate: vi.fn(),
    onRename: vi.fn().mockResolvedValue(undefined),
    onDelete: vi.fn().mockResolvedValue(undefined),
  });

  it("filters, selects and marks the active item", async () => {
    const p = props();
    render(<ItemSidebar {...p} />);
    expect(screen.getByRole("button", { name: /Morning blitz/ })).toHaveAttribute("aria-current", "page");
    await userEvent.type(screen.getByLabelText("Search games"), "endgame");
    expect(screen.queryByText("Morning blitz")).toBeNull();
    await userEvent.click(screen.getByText("Endgame practice"));
    expect(p.onSelect).toHaveBeenCalledWith("b");
  });

  it("renames with Enter and asks before deleting", async () => {
    const p = props();
    render(<ItemSidebar {...p} />);
    await userEvent.click(screen.getAllByLabelText("Rename game")[0]);
    const input = screen.getByLabelText("Game title");
    await userEvent.clear(input);
    await userEvent.type(input, "Sicilian{Enter}");
    expect(p.onRename).toHaveBeenCalledWith("a", "Sicilian");
    await userEvent.click(screen.getAllByLabelText("Delete game")[1]);
    expect(screen.getByText("Delete this game?")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Confirm delete"));
    expect(p.onDelete).toHaveBeenCalledWith("b");
  });

  it("shows an empty state", () => {
    render(<ItemSidebar {...props()} items={[]} />);
    expect(screen.getByText("No games yet.")).toBeInTheDocument();
  });
});
