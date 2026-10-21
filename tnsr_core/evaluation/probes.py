"""Task-level probes: exact-match accuracy on questions with known answers.

Questions and reference answers come from the same tables and arithmetic as the
synthetic corpus (``tnsr_core/data/synthetic.py``), asked in its ``User:``/``TNSR:``
format with greedy decoding. Most probe questions also occur somewhere in the
training data, so the score measures how well the model learned the corpus's
facts and skills, not generalization to unseen knowledge.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any

from tnsr_core.data.synthetic import (
    ANIMALS,
    CAPITALS,
    COLORS,
    DAYS,
    LEG_WORDS,
    MONTHS,
    OPPOSITES,
    SPELL_WORDS,
    _cap,
)
from tnsr_core.inference import GenerationConfig, InferenceEngine

PROBE_VERSION = 1


def build_probes(seed: int = 0) -> list[tuple[str, str, str]]:
    """Return (category, question, expected_answer) triples; deterministic for a seed."""
    rng = random.Random(seed)
    probes: list[tuple[str, str, str]] = []
    pairs = [(a, b) for a in range(21) for b in range(21)]
    for a, b in rng.sample(pairs, 50):
        probes.append(("addition", f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."))
    subs = [(a, b) for a in range(21) for b in range(a + 1)]
    for a, b in rng.sample(subs, 30):
        probes.append(("subtraction", f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."))
    for n in rng.sample(range(99), 20):
        probes.append(("counting", f"What comes after {n}?", f"After {n} comes {n + 1}."))
    for country, city in CAPITALS:
        probes.append(
            (
                "capitals",
                f"What is the capital of {country}?",
                f"The capital of {country} is {city}.",
            )
        )
    for name, verb, sound, legs, *_ in ANIMALS:
        probes.append(
            (
                "animal sounds",
                f"What sound does a {name} make?",
                f"A {name} {verb}. It says {sound}.",
            )
        )
        probes.append(
            (
                "animal legs",
                f"How many legs does a {name} have?",
                f"A {name} has {LEG_WORDS[legs]} legs.",
            )
        )
    for thing, color in COLORS:
        probes.append(("colors", f"What color is {thing}?", f"{_cap(thing)} is {color}."))
    for w1, w2 in OPPOSITES:
        probes.append(
            ("opposites", f"What is the opposite of {w1}?", f"The opposite of {w1} is {w2}.")
        )
    for i, day in enumerate(DAYS):
        probes.append(
            ("calendar", f"What day comes after {day}?", f"{DAYS[(i + 1) % 7]} comes after {day}.")
        )
    for i, month in enumerate(MONTHS):
        probes.append(
            (
                "calendar",
                f"What month comes after {month}?",
                f"{MONTHS[(i + 1) % 12]} comes after {month}.",
            )
        )
    for w in SPELL_WORDS:
        probes.append(
            ("spelling", f"How do you spell {w}?", f"{_cap(w)} is spelled {'-'.join(w.upper())}.")
        )
    return probes


def run_probes(engine: InferenceEngine, limit: int | None = None, seed: int = 0) -> dict[str, Any]:
    probes = build_probes(seed)
    if limit is not None:
        probes = probes[:limit]
    cfg = GenerationConfig(max_new_tokens=48, deterministic=True, stop=["\nUser:", "\n"])
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    failures: list[dict[str, str]] = []
    t0 = time.perf_counter()
    for category, question, expected in probes:
        answer = engine.generate(f"User: {question}\nTNSR:", cfg).text.strip()
        totals[category] += 1
        if answer == expected:
            correct[category] += 1
        elif len(failures) < 12:
            failures.append(
                {"category": category, "question": question, "expected": expected, "answer": answer}
            )
    n = sum(totals.values())
    return {
        "version": PROBE_VERSION,
        "decoding": "greedy, chat format, stop at newline",
        "accuracy": round(sum(correct.values()) / max(1, n), 4),
        "total": n,
        "by_category": {
            c: {
                "correct": correct[c],
                "total": totals[c],
                "accuracy": round(correct[c] / totals[c], 4),
            }
            for c in totals
        },
        "failures": failures,
        "seconds": round(time.perf_counter() - t0, 2),
    }
