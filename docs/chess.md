# TNSR Chess: design and methodology

Goal: the strongest chess engine whose moves come from a transformer **without search**: no
tree search, no lookahead, no engine assistance at play time. For each position the model sees
every legal move once, in a single batched forward pass, and the engine plays the move with
the highest predicted win probability.

## Approach (action-value prediction)

This follows the recipe that produced grandmaster-level play without search in
*Ruoss et al., "Grandmaster-Level Chess Without Search" (DeepMind, 2024)*. As reported there
(quoted from memory; check the paper before relying on exact figures), a 270M-parameter
transformer trained on Stockfish action values for ~10M games (~15B state-action pairs) reached
roughly 2,900 Lichess blitz Elo. Their ablations found predicting **action values** (the win
probability after each move) outperformed predicting the best move directly (behavioral
cloning) or the position's state value.

| | TNSR |
|---|---|
| input | 77 position tokens (FEN fields, fixed length) + 1 move token = 78 tokens |
| move vocabulary | 1,968 UCI moves (every geometrically possible move incl. promotions) |
| target | Stockfish win probability for the side to move after the move, in 128 uniform bins |
| loss | cross-entropy over bins |
| architecture | TNSR Transformer encoder (bidirectional attention, RMSNorm, learned positions), classifier head on the move token |
| play | argmax over legal moves of the expected win probability (softmax · bin centers) |

### Position encoding

```
w rnbqkbnr pppppppp ........ ........ ........ ........ PPPPPPPP RNBQKBNR KQkq -. 0.. 1..
^ side     64 squares a8..h1, '.' = empty                           castling ep  half full
```

Every legal move in 300 random games (1.78M moves) is covered by the action vocabulary
(`tests/test_chess.py`).

### Labels

- Positions are sampled from Lichess games (`per_game` random plies per game, de-duplicated
  by board position). The open database is CC0.
- For each position, Stockfish runs MultiPV over **all** legal moves with a node budget of
  `nodes_per_move × number of legal moves` and reports a score for each move.
- Scores become win probabilities with Lichess's model `1 / (1 + exp(-0.00368208 · cp))`;
  mate scores map to 1 or 0.
- The train/validation split is **by game** (hash of the game URL). Validation positions whose
  board also occurs in training are removed. Puzzles are a separate test set.

Labeling cost is the main bottleneck. Measured on the development VM (Stockfish 16, 2,000
nodes/move): ~0.13 s per position per core, so ~27 positions/s with 4 workers, and ~30 scored
moves per position.

## Evaluation protocol

`tnsr chess evaluate` reports:

1. **Held-out move quality** on validation positions: the share of positions where the model's
   top move equals Stockfish's top move, and **regret** (the mean Stockfish win probability lost
   versus the best move). This is also logged at every training evaluation.
2. **Lichess puzzles** (CC0 puzzle database): a puzzle counts as solved only if the engine finds
   every one of its own moves in the solution line. Any mating move is accepted where the
   solution mates. Results are reported overall and by puzzle rating band.
3. **Games against Stockfish** limited with `UCI_LimitStrength` / `UCI_Elo` (min 1320 in
   Stockfish 16), from a fixed set of opening lines played with both colours, Stockfish at
   50 ms/move. The **performance rating** is the rating whose expected score against the
   opponents equals the actual score. Stockfish's UCI_Elo scale is calibrated by the Stockfish
   team to CCRL-style ratings, which are **not Lichess ratings**. If the score is 0% or 100%,
   the rating is unbounded and reported as such.
4. A **random-mover sanity check**: the engine should win almost every game.

Comparisons with published numbers must use the same yardstick. DeepMind reports Lichess blitz
Elo from games against humans and puzzle accuracy on Lichess puzzles. TNSR's puzzle numbers use
the same puzzle source; its Elo uses a different scale.

## Converting won positions

A no-search engine that is far ahead faces many moves that all keep a ~100% win probability,
so pure win-probability labels give it no reason to make progress, and it drifts into
stalemates and repetitions. DeepMind reported the same and broke ties with Stockfish at play
time; TNSR does not. It uses three training-side measures and one optional play-time rule:

1. **Mate-distance classes** (`data.mate_classes`): besides the 128 win-probability bins, the
   head predicts "mate in n" / "mated in n" for n up to 20, and ranking adds a small bonus for
   faster mates (`MATE_RANK_BONUS`).
2. **Generated endgames** (`tnsr chess prepare --endgames N`): random legal positions with
   lopsided material (KQ vs K, KR vs K, KQ vs KR, pawn endings, ...), labeled by Stockfish.
3. **On-policy positions** (`tnsr chess collect`): the current model plays a random mover and
   weak Stockfish; the positions *it* reaches are labeled by Stockfish and added to the
   training set (`data.dataset = "chess-v1+chess-onpolicy-v1"`). This is the DAgger idea:
   train on the states your own policy visits. In practice these positions contain about five
   times as many forced mates as positions from human games.
4. **Rules-aware scoring** (`--rules`, off by default): a move that ends the game by rule gets
   its exact value: checkmate = 1, stalemate / repetition or fifty-move claim / insufficient
   material = 1/2. No opponent reply is examined and the model is never called on a later
   position, but it does look at the position after each candidate move, so results with it
   are always reported separately.

## Known limitations

- **Draws by repetition:** the model sees only the FEN (no move history), so it cannot learn to
  avoid repetitions; only `--rules` can.
- **Label quality:** 2,000 nodes per move is far shallower than DeepMind's labels. Raising
  `--nodes` improves labels linearly in cost.

## Scaling plan

Strength in the paper grows with data and model size. The path, with the knobs that exist today:

| stage | data | model | where |
|---|---|---|---|
| v0 (done) | 40K positions, 1.2M scored moves | `chess-mini`, 1.1M params | dev VM CPU |
| v1 | 1–5M positions (several Lichess months, `--min-elo 1800`) | `chess-small`, 6.9M | Apple Silicon (labeling ~1–2 days on 10 cores) |
| v2 | 50M+ positions | 50–150M params | rented GPU for training; labeling on a many-core CPU box |
| v3 | ~500M+ positions | 270M+ params | multi-GPU |

Labeling is embarrassingly parallel and resumable (one shard per chunk). Labels made elsewhere
can be imported with `build_dataset_from_labels`.
