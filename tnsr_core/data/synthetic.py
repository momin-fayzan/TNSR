"""Deterministic synthetic demonstration corpus ("TinyTales").

Every document is produced by the templates and word lists in this file, so
the corpus has no third-party provenance and is covered by this repository's
MIT license. It mixes three document types:

* short stories with a setup, problem and resolution;
* simple encyclopedia-style fact paragraphs;
* two-speaker dialogues ("User:" / "TNSR:") with greetings, small arithmetic,
  counting, spelling and factual questions.

The facts are real (animal sounds, capitals, colours, opposites) and the
arithmetic is computed, so a model trained on it learns checkable behaviour.
Training on dialogue-formatted text teaches the *format* of a conversation; it
is pretraining data, not instruction fine-tuning.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

NAMES = [
    "Pip",
    "Mia",
    "Leo",
    "Nora",
    "Finn",
    "Ivy",
    "Max",
    "Ada",
    "Theo",
    "Luna",
    "Sam",
    "Zoe",
    "Ben",
    "Lily",
    "Omar",
    "Rosa",
    "Kai",
    "Ella",
    "Jude",
    "Maya",
    "Tom",
    "Iris",
    "Ravi",
    "June",
]

# name, sound verb, sound word, legs, home, food
ANIMALS = [
    ("cat", "meows", "meow", 4, "house", "fish"),
    ("dog", "barks", "woof", 4, "house", "bones"),
    ("cow", "moos", "moo", 4, "farm", "grass"),
    ("duck", "quacks", "quack", 2, "pond", "seeds"),
    ("sheep", "bleats", "baa", 4, "farm", "grass"),
    ("horse", "neighs", "neigh", 4, "farm", "hay"),
    ("pig", "oinks", "oink", 4, "farm", "corn"),
    ("owl", "hoots", "hoot", 2, "forest", "mice"),
    ("frog", "croaks", "ribbit", 4, "pond", "flies"),
    ("lion", "roars", "roar", 4, "savanna", "meat"),
    ("bee", "buzzes", "buzz", 6, "hive", "nectar"),
    ("mouse", "squeaks", "squeak", 4, "field", "cheese"),
    ("snake", "hisses", "hiss", 0, "jungle", "eggs"),
    ("hen", "clucks", "cluck", 2, "farm", "grain"),
    ("fox", "yips", "yip", 4, "forest", "berries"),
    ("bird", "sings", "tweet", 2, "tree", "worms"),
    ("goat", "bleats", "maa", 4, "hill", "leaves"),
    ("wolf", "howls", "awoo", 4, "forest", "meat"),
]

LEG_WORDS = {0: "no", 2: "two", 4: "four", 6: "six", 8: "eight"}

COLORS = [
    ("the sky", "blue"),
    ("grass", "green"),
    ("snow", "white"),
    ("a banana", "yellow"),
    ("coal", "black"),
    ("a strawberry", "red"),
    ("an orange", "orange"),
    ("a lemon", "yellow"),
    ("milk", "white"),
    ("a tomato", "red"),
    ("chocolate", "brown"),
    ("the ocean", "blue"),
    ("a carrot", "orange"),
    ("a leaf", "green"),
    ("a cherry", "red"),
    ("a crow", "black"),
    ("a pumpkin", "orange"),
    ("a cloud", "white"),
    ("a frog", "green"),
    ("a plum", "purple"),
]

OPPOSITES = [
    ("hot", "cold"),
    ("big", "small"),
    ("up", "down"),
    ("day", "night"),
    ("fast", "slow"),
    ("happy", "sad"),
    ("open", "closed"),
    ("light", "dark"),
    ("old", "young"),
    ("full", "empty"),
    ("wet", "dry"),
    ("loud", "quiet"),
    ("early", "late"),
    ("hard", "soft"),
    ("high", "low"),
    ("left", "right"),
    ("in", "out"),
    ("yes", "no"),
    ("tall", "short"),
    ("near", "far"),
]

CAPITALS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Germany", "Berlin"),
    ("Egypt", "Cairo"),
    ("Canada", "Ottawa"),
    ("Kenya", "Nairobi"),
    ("Peru", "Lima"),
    ("India", "New Delhi"),
    ("China", "Beijing"),
    ("Australia", "Canberra"),
    ("Mexico", "Mexico City"),
    ("Greece", "Athens"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"),
    ("Argentina", "Buenos Aires"),
    ("Chile", "Santiago"),
    ("Turkey", "Ankara"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Poland", "Warsaw"),
    ("Austria", "Vienna"),
    ("Cuba", "Havana"),
    ("Nigeria", "Abuja"),
    ("Russia", "Moscow"),
    ("Brazil", "Brasilia"),
    ("South Korea", "Seoul"),
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
SPELL_WORDS = [
    "cat",
    "dog",
    "sun",
    "moon",
    "tree",
    "book",
    "fish",
    "bird",
    "star",
    "rain",
    "milk",
    "frog",
    "home",
    "ship",
    "cake",
    "kite",
    "lamp",
    "rose",
    "bell",
    "door",
    "hat",
    "cup",
    "map",
    "egg",
]

PLACES = [
    ("forest", "under the tall pine trees"),
    ("village", "near the old stone well"),
    ("garden", "between the rows of beans"),
    ("beach", "by the warm blue water"),
    ("mountain", "on a rocky path"),
    ("farm", "behind the red barn"),
    ("park", "next to the duck pond"),
    ("river", "beside the quiet water"),
    ("meadow", "among the yellow flowers"),
    ("town", "on a busy little street"),
]
OBJECTS = [
    "red ball",
    "kite",
    "small key",
    "shiny stone",
    "old map",
    "blue hat",
    "wooden boat",
    "silver bell",
    "green scarf",
    "basket of apples",
    "paper star",
    "tiny drum",
]
ADJECTIVES = ["little", "brave", "curious", "kind", "clever", "shy", "happy", "gentle", "quick"]
FEELINGS_BAD = ["sad", "worried", "upset", "lonely", "scared"]
FEELINGS_GOOD = ["happy", "proud", "glad", "excited", "calm"]
WEATHER = ["sunny", "rainy", "windy", "cold", "warm", "cloudy", "snowy", "bright"]
TIMES = ["One morning", "One day", "Late one afternoon", "One evening", "Early one spring day"]
SKILLS = [
    ("swim", "the water felt cold and deep"),
    ("fly a kite", "the wind kept pulling it down"),
    ("bake bread", "the dough would not rise"),
    ("climb a tree", "the branches were too high"),
    ("ride a bike", "the bike kept tipping over"),
    ("read a book", "the words looked hard"),
    ("sing a song", "the high notes were tricky"),
    ("build a sandcastle", "the waves washed it away"),
]
MORALS = [
    "It is good to ask for help.",
    "Friends make hard things easier.",
    "Practice helps us grow.",
    "Being kind is always worth it.",
    "Never give up on a good idea.",
    "Sharing makes everyone happier.",
    "Small steps lead to big changes.",
    "Patience is a quiet kind of strength.",
]


_PLURALS = {"sheep": "sheep", "mouse": "mice", "fox": "foxes", "wolf": "wolves"}


def _plural(word: str) -> str:
    return _PLURALS.get(word, word + "s")


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _character(rng: random.Random) -> tuple[str, str, str]:
    """(name, animal, intro phrase)."""
    name = rng.choice(NAMES)
    animal = rng.choice(ANIMALS)[0]
    adj = rng.choice(ADJECTIVES)
    return name, animal, f"{_a(adj)} {animal} named {name}"


def story(rng: random.Random, animal: str | None = None) -> str:
    name, a, intro = _character(rng)
    if animal is not None:
        intro = f"{_a(rng.choice(ADJECTIVES))} {animal} named {name}"
    friend_name, friend, friend_intro = _character(rng)
    while friend_name == name:
        friend_name, friend, friend_intro = _character(rng)
    place, where = rng.choice(PLACES)
    weather = rng.choice(WEATHER)
    kind = rng.randrange(4)
    s = [f"{rng.choice(TIMES)}, {intro} was in the {place} {where}."]
    s.append(
        rng.choice(
            [f"The day was {weather}.", f"It was a {weather} day.", f"The air was {weather}."]
        )
    )
    if kind == 0:
        obj = rng.choice(OBJECTS)
        s.append(f"{name} had lost a {obj} and felt {rng.choice(FEELINGS_BAD)}.")
        s.append(f"{name} looked everywhere, but the {obj} was nowhere to be found.")
        s.append(f"Then {friend_intro} came by and asked what was wrong.")
        s.append(
            f"Together they searched the {place}, and {friend_name} found the {obj} {rng.choice(['under a leaf', 'behind a rock', 'in the tall grass', 'near the path'])}."
        )
        s.append(f'{name} smiled and said, "Thank you, {friend_name}!"')
    elif kind == 1:
        skill, problem = rng.choice(SKILLS)
        s.append(f"{name} wanted to learn to {skill}, but {problem}.")
        s.append(f"{name} tried again and again, and felt {rng.choice(FEELINGS_BAD)}.")
        s.append(f"{_cap(friend_intro)} showed {name} a better way.")
        s.append(
            f"After a lot of practice, {name} could {skill} at last and felt {rng.choice(FEELINGS_GOOD)}."
        )
    elif kind == 2:
        s.append(f"{name} felt lonely because there was no one to play with.")
        s.append(f"Soon {name} met {friend_intro}, who was also alone.")
        game = rng.choice(["hide and seek", "tag", "a guessing game", "catch", "a counting game"])
        s.append(f"They played {game} until the sun went down.")
        s.append(f"From that day on, {name} and {friend_name} were the best of friends.")
    else:
        obj = rng.choice(OBJECTS)
        s.append(f"{name} found a {obj} and wanted to keep it.")
        s.append(f"But {friend_intro} looked at the {obj} with big eyes.")
        s.append(f"{name} thought for a moment and decided to share it.")
        s.append(
            f"They took turns with the {obj}, and both of them felt {rng.choice(FEELINGS_GOOD)}."
        )
    if rng.random() < 0.7:
        s.append(rng.choice(MORALS))
    s.append("The end.")
    return " ".join(s)


def fact_paragraph(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        name, verb, sound, legs, home, food = rng.choice(ANIMALS)
        parts = [
            f"The {name} is an animal.",
            f"A {name} {verb}; it says {sound}.",
            f"A {name} has {LEG_WORDS[legs]} legs.",
            f"Many {_plural(name)} live in the {home} or near it.",
            f"A {name} likes to eat {food}.",
        ]
        head, rest = parts[0], parts[1:]
        rng.shuffle(rest)
        return " ".join([head, *rest])
    if kind == 1:
        picks = rng.sample(COLORS, 4)
        return " ".join(f"{_cap(thing)} is {color}." for thing, color in picks)
    if kind == 2:
        picks = rng.sample(CAPITALS, 3)
        return " ".join(f"The capital of {c} is {city}." for c, city in picks)
    picks = rng.sample(OPPOSITES, 4)
    return " ".join(f"The opposite of {a} is {b}." for a, b in picks)


def _qa(rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(13)
    if kind == 0:
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        q = rng.choice(
            [f"What is {a} plus {b}?", f"What is {a} + {b}?", f"Can you add {a} and {b}?"]
        )
        return q, f"{a} plus {b} is {a + b}."
    if kind == 1:
        a = rng.randint(0, 20)
        b = rng.randint(0, a)
        q = rng.choice([f"What is {a} minus {b}?", f"What is {a} - {b}?"])
        return q, f"{a} minus {b} is {a - b}."
    if kind == 2:
        n = rng.randint(0, 98)
        return rng.choice(
            [f"What comes after {n}?", f"What number comes after {n}?"]
        ), f"After {n} comes {n + 1}."
    if kind == 3:
        name, verb, sound, *_ = rng.choice(ANIMALS)
        q = rng.choice([f"What sound does a {name} make?", f"What does a {name} say?"])
        return q, f"A {name} {verb}. It says {sound}."
    if kind == 4:
        name, _, _, legs, *_ = rng.choice(ANIMALS)
        return f"How many legs does a {name} have?", f"A {name} has {LEG_WORDS[legs]} legs."
    if kind == 5:
        thing, color = rng.choice(COLORS)
        return f"What color is {thing}?", f"{_cap(thing)} is {color}."
    if kind == 6:
        w1, w2 = rng.choice(OPPOSITES)
        if rng.random() < 0.5:
            w1, w2 = w2, w1
        return f"What is the opposite of {w1}?", f"The opposite of {w1} is {w2}."
    if kind == 7:
        c, city = rng.choice(CAPITALS)
        return rng.choice(
            [f"What is the capital of {c}?", f"Which city is the capital of {c}?"]
        ), f"The capital of {c} is {city}."
    if kind == 8:
        i = rng.randrange(7)
        return f"What day comes after {DAYS[i]}?", f"{DAYS[(i + 1) % 7]} comes after {DAYS[i]}."
    if kind == 9:
        i = rng.randrange(12)
        return (
            f"What month comes after {MONTHS[i]}?",
            f"{MONTHS[(i + 1) % 12]} comes after {MONTHS[i]}.",
        )
    if kind == 10:
        w = rng.choice(SPELL_WORDS)
        return f"How do you spell {w}?", f"{_cap(w)} is spelled {'-'.join(w.upper())}."
    if kind == 11:
        n = rng.randint(0, 20)
        return f"How do you write {n} in words?", f"{n} is written as {NUMBER_WORDS[n]}."
    animal = rng.choice(ANIMALS)[0]
    q = rng.choice(
        [f"Tell me a story about a {animal}.", f"Can you tell me a story about a {animal}?"]
    )
    return q, story(rng, animal=animal)


SMALL_TALK = [
    (
        ["Hello!", "Hi!", "Hey there!", "Hello", "Hi there."],
        [
            "Hello! How can I help you today?",
            "Hi! What would you like to talk about?",
            "Hello! It is nice to meet you.",
        ],
    ),
    (["Good morning!", "Good morning."], ["Good morning! I hope you have a lovely day."]),
    (
        ["Who are you?", "What are you?", "What is your name?"],
        [
            "I am TNSR, a small language model trained from scratch.",
            "My name is TNSR. I am a tiny language model.",
        ],
    ),
    (
        ["What can you do?", "How can you help me?"],
        [
            "I can tell short stories, do simple math, and answer simple questions.",
            "I can add and subtract small numbers, tell little stories, and share simple facts.",
        ],
    ),
    (
        ["How are you?", "How are you today?"],
        ["I am doing well, thank you! How are you?", "I feel great today. Thank you for asking!"],
    ),
    (
        ["Thank you!", "Thanks!", "Thank you very much."],
        ["You are welcome!", "You are welcome. I am happy to help."],
    ),
    (["Goodbye!", "Bye!", "See you later."], ["Goodbye! Have a nice day.", "Bye! Come back soon."]),
    (
        ["I am happy today.", "I feel great!"],
        ["That is wonderful to hear!", "I am glad you feel good today."],
    ),
    (
        ["I am sad.", "I feel sad today."],
        ["I am sorry you feel sad. Would you like to hear a happy story?"],
    ),
]


def _small_talk(rng: random.Random) -> tuple[str, str]:
    users, replies = rng.choice(SMALL_TALK)
    return rng.choice(users), rng.choice(replies)


def dialogue(rng: random.Random) -> str:
    turns: list[tuple[str, str]] = []
    if rng.random() < 0.4:
        turns.append(_small_talk(rng))
    for _ in range(rng.randint(1, 3)):
        turns.append(_qa(rng) if rng.random() < 0.85 else _small_talk(rng))
    return "\n".join(f"User: {u}\nTNSR: {a}" for u, a in turns)


def generate_documents(n_docs: int = 20000, seed: int = 0) -> Iterator[str]:
    """Yield ``n_docs`` documents; identical (n_docs, seed) always yields identical text."""
    rng = random.Random(seed)
    for _ in range(n_docs):
        r = rng.random()
        if r < 0.40:
            yield story(rng)
        elif r < 0.55:
            yield fact_paragraph(rng)
        else:
            yield dialogue(rng)


PROVENANCE = {
    "source": "synthetic:tinytales",
    "license": "MIT (generated by tnsr_core/data/synthetic.py; no third-party text)",
    "description": "Deterministic templated stories, fact paragraphs and User/TNSR dialogues.",
}
