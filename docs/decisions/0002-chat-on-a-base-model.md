# ADR 0002: Chat on a base model

**Status:** accepted · 2026-10-21

## Context

TNSR Studio is a chat interface, but the model is a base (pretrained-only) language model. It is
not instruction-tuned, and the product must not claim that it is.

## Decision

- About 45% of the synthetic corpus consists of two-speaker documents in a fixed
  `User: … / TNSR: …` format (greetings, arithmetic, facts, spelling, short stories).
- **Chat mode** renders the conversation history in exactly that format, ends the prompt with
  `TNSR:`, and stops generation at the next `\nUser:`. Older turns are dropped whole when the
  history no longer fits the context window.
- **Completion mode** sends the user's text verbatim for continuation.
- The UI states that TNSR is a small base model trained from scratch and explains how chat mode
  works in the generation panel.

## Consequences

The chat works within the narrow distribution of the corpus (it answers "What is 7 plus 5?" or
"What sound does a cow make?") and degrades outside it. True instruction following needs a
separate supervised fine-tuning stage (see roadmap).
