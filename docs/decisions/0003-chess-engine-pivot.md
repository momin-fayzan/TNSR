# ADR 0003: Pivot to a no-search transformer chess engine

**Status:** accepted · 2026-10-23

## Context

The owner's goal changed from a general text model to "the strongest LLM-based chess engine",
with three explicit choices: **no search**, **all in on chess**, and **GPU rental is acceptable
later** (but not by default).

## Decisions

1. **Objective: action values.** Predict Stockfish's win probability after each legal move
   (128 bins, cross-entropy, optional HL-Gauss smoothing) and play the argmax. This is the
   best-performing no-search objective reported by Ruoss et al. (2024). Behavioral cloning of
   human moves would cap strength at the level of the games' players.
2. **Reuse TNSR's core.** Same Transformer (now with bidirectional attention and a classifier
   head), same trainer (generalized over tasks), same checkpoint format, run folders, Lab and
   job system. A chess checkpoint is a normal TNSR checkpoint whose tokenizer is
   `ChessTokenizer` and whose metadata says `task = "chess"`.
3. **Data: Lichess open database (CC0) + Stockfish labels.** Stockfish is a separate GPL
   program that TNSR runs as a subprocess, so its license does not extend to TNSR's code. The
   puzzle benchmark is also from Lichess (CC0).
4. **python-chess (GPL-3) as a dependency** for move generation and PGN parsing. It is a
   dependency, not vendored code. If a purely permissive license becomes important,
   replacing it with in-house move generation is a contained change (`tnsr_core/chess`).
5. **Honest strength reporting.** Puzzle accuracy uses the same source as DeepMind's
   evaluation. Elo comes from games against `UCI_Elo`-limited Stockfish and is labeled as a
   different scale from Lichess ratings.
6. **Product: Studio becomes Play.** The text-chat UI was removed. The text language-model
   task remains in the core (CLI and training) as shared infrastructure.

## Consequences

The data pipeline (labeling throughput) is the bottleneck, not training. Scaling the engine
means scaling labeled positions first, then model size; see `docs/chess.md`.
