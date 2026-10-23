import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";

// Text-presentation selector keeps glyphs from rendering as colour emoji (notably the pawn).
const VS15 = "︎";
const GLYPH: Record<string, string> = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const NAME: Record<string, string> = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const FILES = "abcdefgh";

export type Orientation = "white" | "black";

/** Map of square name (e.g. "e4") to FEN piece letter (uppercase = white). */
export function parsePlacement(fen: string): Record<string, string> {
  const out: Record<string, string> = {};
  const rows = fen.split(" ")[0].split("/");
  rows.forEach((row, r) => {
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) file += Number(ch);
      else {
        out[`${FILES[file]}${8 - r}`] = ch;
        file += 1;
      }
    }
  });
  return out;
}

function pieceLabel(p: string): string {
  return `${p === p.toUpperCase() ? "white" : "black"} ${NAME[p.toLowerCase()]}`;
}

export function ChessBoard({
  fen,
  orientation = "white",
  legal = [],
  lastMove = null,
  check = false,
  interactive = true,
  hint = null,
  onMove,
}: {
  fen: string;
  orientation?: Orientation;
  legal?: string[];
  lastMove?: string | null;
  check?: boolean;
  interactive?: boolean;
  hint?: string | null;
  onMove?: (uci: string) => void;
}) {
  const pieces = useMemo(() => parsePlacement(fen), [fen]);
  const turn = fen.split(" ")[1] === "b" ? "black" : "white";
  const [selected, setSelected] = useState<string | null>(null);
  const [promotion, setPromotion] = useState<{ from: string; to: string } | null>(null);
  useEffect(() => {
    setSelected(null);
    setPromotion(null);
  }, [fen]);

  const fromSquares = useMemo(() => new Set(legal.map((m) => m.slice(0, 2))), [legal]);
  const targets = useMemo(
    () => new Set(selected ? legal.filter((m) => m.startsWith(selected)).map((m) => m.slice(2, 4)) : []),
    [legal, selected],
  );
  const kingInCheck = check
    ? Object.entries(pieces).find(([, p]) => p === (turn === "white" ? "K" : "k"))?.[0]
    : undefined;

  const ranks = orientation === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = orientation === "white" ? FILES.split("") : FILES.split("").reverse();

  const click = (sq: string) => {
    if (!interactive) return;
    if (selected && targets.has(sq)) {
      const candidates = legal.filter((m) => m.startsWith(selected + sq));
      if (candidates.some((m) => m.length === 5)) setPromotion({ from: selected, to: sq });
      else onMove?.(selected + sq);
      setSelected(null);
      return;
    }
    setSelected(fromSquares.has(sq) && sq !== selected ? sq : null);
  };

  return (
    <div className="relative aspect-square w-full select-none" data-testid="chess-board">
      <div role="grid" aria-label="Chess board" className="grid h-full w-full grid-cols-8 grid-rows-8 overflow-hidden rounded-md border border-border-strong shadow-lg">
        {ranks.map((rank, ri) =>
          files.map((file, fi) => {
            const sq = `${file}${rank}`;
            const piece = pieces[sq];
            const dark = (FILES.indexOf(file) + rank) % 2 === 1; // a1 is a dark square
            const isLast = lastMove && (lastMove.slice(0, 2) === sq || lastMove.slice(2, 4) === sq);
            const isHint = hint && (hint.slice(0, 2) === sq || hint.slice(2, 4) === sq);
            const isTarget = targets.has(sq);
            const movable = interactive && fromSquares.has(sq);
            return (
              <button
                key={sq}
                type="button"
                role="gridcell"
                data-square={sq}
                aria-label={`${sq}${piece ? `, ${pieceLabel(piece)}` : ""}${isTarget ? ", legal target" : ""}`}
                aria-selected={selected === sq}
                tabIndex={movable || isTarget ? 0 : -1}
                onClick={() => click(sq)}
                className={clsx(
                  "relative flex items-center justify-center outline-none focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-accent",
                  dark ? "bg-[#7e8fa6]" : "bg-[#dde3eb]",
                  (movable || isTarget) && "cursor-pointer",
                )}
              >
                {isLast && <span className="absolute inset-0 bg-amber-300/40" aria-hidden />}
                {isHint && !isLast && <span className="absolute inset-0 bg-sky-400/30" aria-hidden />}
                {selected === sq && <span className="absolute inset-0 bg-sky-500/45" aria-hidden />}
                {kingInCheck === sq && (
                  <span className="absolute inset-0 bg-[radial-gradient(circle,rgba(239,68,68,0.85)_0%,rgba(239,68,68,0)_70%)]" aria-hidden />
                )}
                {piece && (
                  <span
                    aria-hidden
                    className={clsx(
                      "relative z-[1] text-[clamp(1.4rem,8.5vmin,3.6rem)] leading-none",
                      piece === piece.toUpperCase()
                        ? "text-white [-webkit-text-stroke:1.2px_#1b2430] [paint-order:stroke]"
                        : "text-[#151b24] [-webkit-text-stroke:0.6px_#151b24]",
                    )}
                  >
                    {GLYPH[piece.toLowerCase()] + VS15}
                  </span>
                )}
                {isTarget && (
                  <span
                    aria-hidden
                    className={clsx(
                      "absolute z-[2] rounded-full",
                      piece ? "inset-[6%] border-[5px] border-sky-700/45" : "size-[28%] bg-sky-800/40",
                    )}
                  />
                )}
                {fi === 0 && (
                  <span aria-hidden className={clsx("absolute top-0.5 left-1 text-[10px] font-semibold", dark ? "text-[#dde3eb]" : "text-[#5f7088]")}>
                    {rank}
                  </span>
                )}
                {ri === 7 && (
                  <span aria-hidden className={clsx("absolute right-1 bottom-0.5 text-[10px] font-semibold", dark ? "text-[#dde3eb]" : "text-[#5f7088]")}>
                    {file}
                  </span>
                )}
              </button>
            );
          }),
        )}
      </div>
      {promotion && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/50" role="dialog" aria-label="Choose promotion piece">
          <div className="flex gap-2 rounded-lg border border-border bg-surface p-3 shadow-xl">
            {["q", "r", "b", "n"].map((p) => (
              <button
                key={p}
                type="button"
                aria-label={`Promote to ${NAME[p]}`}
                onClick={() => {
                  onMove?.(promotion.from + promotion.to + p);
                  setPromotion(null);
                }}
                className="flex size-14 items-center justify-center rounded-md bg-[#dde3eb] text-4xl hover:bg-white"
              >
                <span className={turn === "white" ? "text-white [-webkit-text-stroke:1.2px_#1b2430]" : "text-[#151b24]"}>
                  {GLYPH[p] + VS15}
                </span>
              </button>
            ))}
            <button type="button" onClick={() => setPromotion(null)} className="px-2 text-xs text-muted hover:text-fg">
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
