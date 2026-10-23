import clsx from "clsx";
import { Crown, FlipVertical2, PanelLeft, Undo2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ChessBoard, type Orientation } from "../components/ChessBoard";
import { ItemSidebar } from "../components/ItemSidebar";
import { ModelPicker } from "../components/ModelPicker";
import { Badge, Button, EmptyState, ErrorBanner, IconButton, Spinner, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { useApp } from "../lib/app-context";
import { formatRelative, modelLabel } from "../lib/format";
import type { ChessGame, ChessGameView, ModelInfo } from "../lib/types";

const TERMINATION: Record<string, string> = {
  checkmate: "checkmate",
  stalemate: "stalemate",
  insufficient_material: "insufficient material",
  seventyfive_moves: "75-move rule",
  fivefold_repetition: "fivefold repetition",
  fifty_moves: "50-move rule",
  threefold_repetition: "threefold repetition",
};

function resultText(game: ChessGameView): string | null {
  const o = game.state.outcome;
  if (!o) return null;
  const how = TERMINATION[o.termination] ?? o.termination;
  if (o.result === "1/2-1/2") return `Draw by ${how}`;
  const whiteWon = o.result === "1-0";
  const humanWon = whiteWon === (game.human_color === "white");
  return `${humanWon ? "You win" : "TNSR wins"} by ${how} (${o.result})`;
}

/** Win probability for White from an analysis of the side to move (best move's value). */
function whiteWinProb(game: ChessGameView): number | null {
  const best = game.analysis?.moves[0];
  if (!best) return null;
  return game.state.turn === "white" ? best.win_prob : 1 - best.win_prob;
}

export function PlayPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { connection } = useApp();
  const [games, setGames] = useState<ChessGame[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [game, setGame] = useState<ChessGameView | null>(null);
  const [loadingGame, setLoadingGame] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [orientation, setOrientation] = useState<Orientation>("white");
  const [showHints, setShowHints] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);

  const loadGames = useCallback(async () => {
    try {
      setGames((await api.chessGames()).games);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    if (connection !== "online") return;
    loadGames();
    api
      .chessModels()
      .then((r) => setModels(r.models))
      .catch(() => setModels([]));
  }, [connection, loadGames]);

  useEffect(() => {
    setShowSidebar(false);
    if (!id) {
      setGame(null);
      return;
    }
    setLoadingGame(true);
    api
      .chessGame(id)
      .then((g) => {
        setGame(g);
        setOrientation(g.human_color);
        setError(null);
      })
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoadingGame(false));
  }, [id]);

  const recommended = models?.find((m) => m.recommended)?.id ?? null;

  const newGame = async (color: "white" | "black", modelId: string | null = game?.model_id ?? recommended) => {
    setBusy(true);
    setError(null);
    try {
      const g = await api.newChessGame({ human_color: color, model_id: modelId });
      setGames((prev) => [g, ...prev]);
      setGame(g);
      setOrientation(color);
      navigate(`/g/${g.id}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const act = async (fn: () => Promise<ChessGameView>) => {
    setBusy(true);
    setError(null);
    try {
      const g = await fn();
      setGame(g);
      setGames((prev) => [g, ...prev.filter((x) => x.id !== g.id)]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const humanTurn = game ? game.state.turn === game.human_color && !game.state.outcome : false;
  const wp = game ? whiteWinProb(game) : null;
  const result = game ? resultText(game) : null;
  const noModels = models !== null && models.length === 0;

  return (
    <div className="flex min-h-0 flex-1">
      <div className={clsx("z-30 w-72 shrink-0 md:static md:block", showSidebar ? "fixed inset-y-0 left-0 block shadow-2xl" : "hidden")}>
        <ItemSidebar
          items={games.map((g) => ({
            id: g.id,
            title: g.title,
            meta: (
              <>
                <span>{formatRelative(g.updated_at)}</span>
                <span>· {Math.ceil(g.moves.length / 2)} moves</span>
                {g.result && <span>· {g.result}</span>}
              </>
            ),
          }))}
          activeId={id ?? null}
          loading={listLoading && connection !== "offline"}
          error={null}
          onSelect={(gid) => navigate(`/g/${gid}`)}
          onCreate={() => newGame("white")}
          onRename={async (gid, title) => {
            await api.renameChessGame(gid, title);
            setGames((prev) => prev.map((g) => (g.id === gid ? { ...g, title } : g)));
            if (game?.id === gid) setGame({ ...game, title });
          }}
          onDelete={async (gid) => {
            await api.deleteChessGame(gid);
            setGames((prev) => prev.filter((g) => g.id !== gid));
            if (gid === id) navigate("/");
          }}
          creating={busy && !game}
        />
      </div>
      {showSidebar && <div className="fixed inset-0 z-20 bg-black/40 md:hidden" onClick={() => setShowSidebar(false)} aria-hidden />}

      <section className="flex min-w-0 flex-1 flex-col overflow-y-auto" aria-label="Board">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-3">
          <IconButton label="Show games" className="md:hidden" onClick={() => setShowSidebar(true)}>
            <PanelLeft className="size-4" />
          </IconButton>
          <h2 className="min-w-0 flex-1 truncate text-sm font-medium">{game?.title ?? "TNSR Play"}</h2>
          {game && (
            <span className="hidden text-xs text-faint sm:inline" title={game.model_id ?? ""}>
              vs {modelLabel(game.model_id)}
            </span>
          )}
        </header>

        {error && (
          <div className="px-4 pt-3">
            <ErrorBanner onRetry={() => setError(null)}>{error}</ErrorBanner>
          </div>
        )}

        {!id ? (
          <Welcome noModels={noModels} busy={busy} offline={connection === "offline"} onNew={(c) => newGame(c)} models={models ?? []} />
        ) : loadingGame && !game ? (
          <div className="flex flex-1 items-center justify-center">
            <Spinner />
          </div>
        ) : game ? (
          <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 p-4 lg:flex-row lg:items-start">
            <div className="mx-auto flex w-full max-w-[min(78vh,640px)] gap-3">
              <EvalBar whiteProb={wp} orientation={orientation} />
              <div className="min-w-0 flex-1">
                <ChessBoard
                  fen={game.state.fen}
                  orientation={orientation}
                  legal={humanTurn && !busy ? game.state.legal : []}
                  lastMove={game.state.last_move}
                  check={game.state.check}
                  interactive={humanTurn && !busy}
                  hint={showHints && humanTurn ? (game.analysis?.moves[0]?.uci ?? null) : null}
                  onMove={(uci) => act(() => api.chessMove(game.id, uci))}
                />
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <StatusLine game={game} busy={busy} result={result} />
                  <div className="ml-auto flex gap-1.5">
                    <Button size="sm" variant="ghost" onClick={() => setOrientation((o) => (o === "white" ? "black" : "white"))}>
                      <FlipVertical2 className="size-3.5" aria-hidden /> Flip
                    </Button>
                    <Button size="sm" onClick={() => act(() => api.chessUndo(game.id))} disabled={busy || game.moves.length === 0}>
                      <Undo2 className="size-3.5" aria-hidden /> Take back
                    </Button>
                  </div>
                </div>
              </div>
            </div>

            <aside className="w-full space-y-4 lg:w-80 lg:shrink-0" aria-label="Game details">
              <div className="rounded-lg border border-border bg-surface p-3">
                <div className="mb-2 text-xs font-medium text-muted">New game</div>
                <div className="mb-3">
                  <ModelPicker models={models ?? []} value={game.model_id} onChange={(m) => newGame(game.human_color, m)} disabled={busy} />
                </div>
                <div className="flex gap-2">
                  <Button size="sm" className="flex-1" onClick={() => newGame("white")} disabled={busy}>
                    Play white
                  </Button>
                  <Button size="sm" className="flex-1" onClick={() => newGame("black")} disabled={busy}>
                    Play black
                  </Button>
                </div>
              </div>

              {game.engine_move && (
                <div className="rounded-lg border border-border bg-surface p-3">
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="font-medium text-muted">TNSR played {game.engine_move.san}</span>
                    <span className="font-mono text-faint">{(game.engine_move.win_prob * 100).toFixed(1)}%</span>
                  </div>
                  <MoveBars moves={game.engine_move.alternatives} chosen={game.engine_move.uci} />
                  <p className="mt-2 text-[11px] text-faint">Its predicted win probability for each candidate. It plays the highest; there is no search.</p>
                </div>
              )}

              {humanTurn && game.analysis && (
                <div className="rounded-lg border border-border bg-surface p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <label htmlFor="show-hints" className="text-xs font-medium text-muted">
                      TNSR's view of your options
                    </label>
                    <Toggle id="show-hints" label="Show TNSR's suggestions" checked={showHints} onChange={setShowHints} />
                  </div>
                  {showHints ? <MoveBars moves={game.analysis.moves} /> : <p className="text-[11px] text-faint">Hidden. Turn on to see how TNSR rates your legal moves.</p>}
                </div>
              )}

              <div className="rounded-lg border border-border bg-surface p-3">
                <div className="mb-2 text-xs font-medium text-muted">Moves</div>
                <MoveList san={game.state.san} startBlack={game.start_fen.split(" ")[1] === "b"} />
              </div>
            </aside>
          </div>
        ) : (
          <EmptyState title="Game not found" action={<Button onClick={() => navigate("/")}>Back</Button>} />
        )}
      </section>
    </div>
  );
}

function StatusLine({ game, busy, result }: { game: ChessGameView; busy: boolean; result: string | null }) {
  if (result)
    return (
      <Badge tone={result.startsWith("You win") ? "ok" : result.startsWith("Draw") ? "neutral" : "warn"} className="text-xs">
        <Crown className="size-3" aria-hidden /> {result}
      </Badge>
    );
  if (busy) return <span className="flex items-center gap-1.5 text-xs text-muted"><Spinner className="size-3" /> TNSR is thinking…</span>;
  const yours = game.state.turn === game.human_color;
  return (
    <span className="text-xs text-muted" role="status">
      {yours ? "Your move" : "TNSR to move"}
      {game.state.check && <span className="ml-1.5 font-medium text-danger">· check</span>}
    </span>
  );
}

function EvalBar({ whiteProb, orientation }: { whiteProb: number | null; orientation: Orientation }) {
  const pct = whiteProb == null ? 50 : Math.round(whiteProb * 100);
  return (
    <div
      className={clsx("relative w-3 shrink-0 overflow-hidden rounded-sm border border-border-strong bg-[#151b24]", orientation === "black" && "rotate-180")}
      role="meter"
      aria-label="TNSR's estimate of White's winning chances"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      title={whiteProb == null ? "No estimate" : `White ${pct}% (TNSR's estimate)`}
    >
      <div className="absolute inset-x-0 bottom-0 bg-white transition-[height] duration-300" style={{ height: `${pct}%` }} />
    </div>
  );
}

function MoveBars({ moves, chosen }: { moves: { uci: string; san: string; win_prob: number }[]; chosen?: string }) {
  return (
    <ul className="space-y-1">
      {moves.map((m) => (
        <li key={m.uci} className="flex items-center gap-2 text-xs">
          <span className={clsx("w-14 shrink-0 font-mono", m.uci === chosen ? "font-semibold text-accent" : "text-fg")}>{m.san}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-3">
            <div className={clsx("h-full", m.uci === chosen ? "bg-accent" : "bg-muted/60")} style={{ width: `${m.win_prob * 100}%` }} />
          </div>
          <span className="w-11 text-right font-mono text-faint">{(m.win_prob * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

function MoveList({ san, startBlack }: { san: string[]; startBlack: boolean }) {
  if (!san.length) return <p className="text-xs text-faint">No moves yet.</p>;
  const plies = startBlack ? ["…", ...san] : san;
  const rows = [];
  for (let i = 0; i < plies.length; i += 2) rows.push([plies[i], plies[i + 1]]);
  return (
    <ol className="grid max-h-64 grid-cols-[2rem_1fr_1fr] gap-x-2 gap-y-0.5 overflow-y-auto font-mono text-xs">
      {rows.map(([w, b], i) => (
        <li key={i} className="contents">
          <span className="text-faint">{i + 1}.</span>
          <span className="text-fg">{w}</span>
          <span className="text-fg">{b ?? ""}</span>
        </li>
      ))}
    </ol>
  );
}

function Welcome({ noModels, busy, offline, onNew, models }: { noModels: boolean; busy: boolean; offline: boolean; onNew: (c: "white" | "black") => void; models: ModelInfo[] }) {
  const rec = models.find((m) => m.recommended);
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-16 text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-xl border border-border bg-surface text-2xl">♞</div>
      <h1 className="text-xl font-semibold tracking-tight">Play TNSR</h1>
      <p className="mt-2 text-sm text-muted">
        A transformer trained from scratch to predict how good each move is. It picks its move from one forward pass over the legal moves:
        no search, no lookahead.
      </p>
      {noModels ? (
        <div className="mt-6 w-full rounded-lg border border-border bg-surface p-4 text-left text-sm">
          <p className="font-medium">No chess models yet</p>
          <p className="mt-1 text-muted">
            Prepare data and train one: <code className="font-mono text-xs">uv run tnsr train --config chess-mini</code>
          </p>
          <Link to="/lab" className="mt-3 inline-block text-sm font-medium text-accent hover:underline">
            Open TNSR Lab
          </Link>
        </div>
      ) : (
        <>
          <div className="mt-6 flex gap-2">
            <Button variant="primary" onClick={() => onNew("white")} loading={busy} disabled={offline}>
              Play as white
            </Button>
            <Button onClick={() => onNew("black")} disabled={offline || busy}>
              Play as black
            </Button>
          </div>
          {rec && <p className="mt-3 text-xs text-faint">Opponent: {modelLabel(rec.id)} (best puzzle accuracy)</p>}
        </>
      )}
    </div>
  );
}
