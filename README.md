# TNSR

**A transformer chess engine trained from scratch that plays without search.** For every
position, TNSR scores all legal moves in one batched forward pass and plays the move it
predicts is best: no tree search, no lookahead, no engine help at play time.

The goal is the strongest engine in that category. The approach follows DeepMind's
*Grandmaster-Level Chess Without Search* (Ruoss et al., 2024): learn Stockfish's win
probability for every move ("action values") from millions of labeled positions. Everything
around the model is built in this repo:

| | |
|---|---|
| **TNSR Core** | Transformer (bidirectional attention, classifier head), chess tokenizer (77-token positions with the candidate move marked on the board, 1,968-move vocabulary), Lichess data pipeline, parallel resumable Stockfish labeling with mate distances, generated endgame curriculum, trainer with bit-exact resume, engine + UCI, evaluation (puzzles, games vs Stockfish, held-out move quality) |
| **TNSR Play** | play TNSR in the browser: click-to-move board, TNSR's candidate scores and win-probability bar, optional hints, take-back, saved games |
| **TNSR Lab** | live training runs, loss / move-accuracy / regret curves, evaluations (puzzles by rating, Stockfish matches), run comparison, dataset labeling jobs |

The original text-LM pipeline (BPE tokenizer, text corpus, generation) remains in the core
as shared infrastructure (`tnsr data`, `tnsr generate`). The product is now chess.

## Results so far

Measured on the development VM (4 x86 cores, no GPU). All numbers come from evaluation files
written by `tnsr chess evaluate` (protocol and full experiment log in
[docs/chess.md](docs/chess.md)). Every model here is `chess-mini`, 1.1M parameters, trained for
about an hour.

| model | training data | top-1 = Stockfish¹ | regret¹ | Lichess puzzles² | vs Stockfish 1320 / 1600³ | Elo³ | vs random mover |
|---|---|---|---|---|---|---|---|
| random legal move | – | 6.3% | 0.184 | – | – | – | – |
| v0 | 38K Lichess positions | 10.5% | 0.138 | 0.4% | ½/12, 0/12 | – | +0 =5 −1 |
| + move marks | same | 20.3% | 0.107 | 7.0% | ½/12, 0/12 | – | +0 =6 −0 |
| v1 | 71K Lichess + 15K endgames, mate classes | 23.0% | 0.098 | 8.8% | 2½/50, 1½/50 | ~860 | +0 =6 −0 |
| **v2** | v1 + 22K positions from its own games | **26.5%** | **0.088** | **9.6%** | 4½/50, 2/50 | **~950** | **+3 =3 −0** |
| v2 with `--rules` | same | 26.5% | 0.088 | 25.2% | 2/50, 3½/50 | ~920 | +5 =1 −0 |

¹ 400 held-out positions from games never seen in training; regret is the mean Stockfish win
probability lost per move relative to the best move.
² 1,000 puzzles from the Lichess puzzle database (ratings 399–3099); a puzzle counts only if
every move of the solution is found.
³ Stockfish 16 with `UCI_LimitStrength` at 50 ms/move, both colours, fixed openings. All points
are draws (no wins in 400 games). Elo is a performance rating on Stockfish's scale, which is
not the Lichess or chess.com scale; at these scores it is uncertain by roughly ±150.

`--rules` (off by default) scores a move that ends the game by rule with its exact result:
checkmate = win, stalemate or repetition = draw. It looks at no replies.

**What these runs taught us.**
1. Marking the move's from/to squares on the board doubled move accuracy at no extra cost. A
   cheaper design that reads the board once and scores all moves together predicted values
   well but chose moves far worse (10–13% top-1), so marks stay.
2. A no-search model that is winning tends to shuffle into stalemate or repetition. Adding
   positions from **its own games**, labeled by Stockfish, fixed part of this (v2 beats the
   random mover); `--rules` fixes more.
3. On this CPU, strength is **compute-bound**: each run sees fewer examples than its dataset
   holds. DeepMind's ChessBench data (15B labeled moves, CC-BY 4.0, importable with
   `tnsr chess import-chessbench`) removes the data limit; the next gains need bigger models
   and more training time (Apple Silicon, then GPUs). See the
   [scaling plan](docs/chess.md#scaling-plan).

## Quickstart (macOS or Linux)

```bash
brew install stockfish uv node        # Linux: apt install stockfish; install uv and Node 20+
git clone -b claude/loving-gates-0xcngi https://github.com/momin-fayzan/TNSR.git && cd TNSR
uv sync --extra dev && npm install
uv run tnsr doctor

# Data: download a month of Lichess games (CC0) and label positions with Stockfish (resumable)
uv run tnsr chess prepare --month 2013-01 --name chess-v1 --max-positions 60000 --endgames 15000
uv run tnsr chess puzzles --count 5000          # evaluation puzzles

# Train and measure
uv run tnsr train --config chess-mini           # 1.1M params; chess-small (6.9M) for Apple Silicon/GPU
uv run tnsr chess evaluate --games 12           # puzzles + held-out move quality + games vs Stockfish

# Play
npm run build && uv run tnsr serve              # http://127.0.0.1:8000 → Play
uv run tnsr chess play                          # or in the terminal
uv run tnsr chess uci                           # or in any UCI chess GUI
```

On Apple Silicon, training uses the GPU (MPS) automatically and play uses the CPU, which is
faster for single positions at these model sizes (measured; see
[troubleshooting](docs/troubleshooting.md)).

## How it works

```
position (FEN) ──► 77 tokens: side · 64 squares · castling · en passant · clocks
candidate move ──► mark its from/to squares on the board + 1 move token
                   │
                   ▼
   Transformer encoder (bidirectional) ──► 128 win-probability bins + "mate in n" classes
                   │
legal moves ×N ──► one batched forward pass ──► play argmax (expected win, faster mates first)
```

- **Labels:** Stockfish MultiPV over every legal move (2,000 nodes per move by default),
  converted to win probability with Lichess's centipawn model; mates keep their distance.
- **Splits:** by game, with validation positions that also occur in training removed. Puzzles
  and Stockfish games are separate tests.
- **Honest numbers:** puzzle accuracy uses the same source DeepMind used. Elo against
  `UCI_Elo`-limited Stockfish is a different scale from Lichess ratings and is labeled as such.

Details: [docs/chess.md](docs/chess.md) · [architecture](docs/architecture.md) ·
[training](docs/training.md) · [ADR 0003](docs/decisions/0003-chess-engine-pivot.md)

## Commands

```bash
uv run tnsr chess download --month 2013-01       # Lichess games (CC0)
uv run tnsr chess prepare [PGN | --month M] --name N [--max-positions --per-game --nodes --endgames --min-elo]
uv run tnsr chess import-chessbench --files 0-3 --name chessbench   # DeepMind's labeled data (CC-BY 4.0)
uv run tnsr chess collect MODEL --name games1     # the model's own games → label with prepare
uv run tnsr chess puzzles --count 5000
uv run tnsr train --config chess-mini|chess-small [--set data.dataset=chess-v1+chessbench] [--resume RUN]
uv run tnsr chess evaluate [MODEL] [--rules] [--puzzles 1000 --games 50 --levels 1320,1600,1900]
uv run tnsr chess analyse "FEN" | play | uci [--rules]
uv run tnsr runs list · models list · models export/import · serve · doctor
```

## Tests

```bash
scripts/test.sh               # ruff, mypy, pytest, eslint, tsc, vitest
scripts/test.sh --slow --e2e  # plus long tests and the browser E2E (labels with Stockfish, trains, plays)
```

The chess tests cover the move vocabulary (all 1.78M legal moves of 300 random games), the
encoding and move marks, the value head, labeling with real Stockfish (skipped if absent),
training and resume, the engine and UCI protocol, puzzle scoring, Elo estimation, the chess
API (rules, engine replies, take-back, game over) and the Play UI. The Playwright E2E labels
positions with Stockfish, trains a model, and plays it in Chromium.

## Licensing

TNSR is MIT. Training data comes from the Lichess open database and puzzle database (CC0). Optional
ChessBench data (`tnsr chess import-chessbench`) is by Google DeepMind
([searchless_chess](https://github.com/google-deepmind/searchless_chess)), CC-BY 4.0 with CC0
Lichess portions.
Stockfish (GPL) is run as a separate program to label data and as a sparring partner; it is
not part of TNSR's code. python-chess (GPL-3) is a dependency (see ADR 0003).
