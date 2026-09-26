"""RULER-style synthetic long-context tasks (NVIDIA RULER, Apache-2.0) -- the `long_ctx` domain.

RULER (Hsieh et al., 2024, github.com/NVIDIA/RULER) generates its contexts, so there is nothing a
model can have memorised and no item text to license. This is a faithful subset: three of its
task types with RULER's own prompt templates and its `string_match_all` metric, generated from a
seed per item. Context length is set in APPROXIMATE tokens (4 characters per token): every model
tokenises differently, and routing needs "a ~32k-token context", not one tokenizer's exact count.

Tasks: niah_multikey (a needle among decoy needles in a noise haystack), variable_tracking (chains
of `VAR X = Y` assignments), common_words (the 10 most frequent words in a long numbered list).
"""
from __future__ import annotations

import random
import string
from collections import Counter

from suites.base import Item, Suite

TASKS = ("niah_multikey", "variable_tracking", "common_words")
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
CHARS_PER_TOKEN = 4
WORDS = ("apple", "river", "stone", "cloud", "ember", "maple", "orbit", "prism", "quartz", "raven", "saddle", "tundra",
         "velvet", "willow", "zephyr", "anchor", "beacon", "canyon", "dagger", "falcon", "garnet", "harbor", "island",
         "jasper", "kettle", "lantern", "meadow", "nectar", "oyster", "pebble", "quiver", "ribbon", "summit", "timber",
         "umber", "valley", "walnut", "yonder", "zenith", "acorn", "bramble", "cobalt", "dune", "fern", "glacier", "hazel")


def string_match_all(prediction: str, references: list[str]) -> float:
    """RULER's metric: the fraction of references that appear in the prediction, case-insensitive."""
    if not references:
        return 0.0
    p = prediction.lower()
    return sum(1 for r in references if r.lower() in p) / len(references)


def _uuid(rng: random.Random) -> str:
    return "-".join("".join(rng.choice("0123456789abcdef") for _ in range(n)) for n in (8, 4, 4, 4, 12))


def _haystack(sentences: list[str], rng: random.Random, target_chars: int) -> str:
    """Noise up to the target length, with the given sentences inserted at random depths."""
    noise_len = max(0, target_chars - sum(len(s) + 1 for s in sentences) - 400)
    body = (NOISE * (noise_len // len(NOISE) + 1))[:noise_len]
    parts = body.split(". ")
    for s in sentences:
        parts.insert(rng.randrange(len(parts) + 1), s.rstrip("."))
    return ". ".join(parts)


def generate(task: str, seed: int, target_tokens: int = 32_000) -> Item:
    rng = random.Random(f"ruler:{task}:{seed}:{target_tokens}")
    target = target_tokens * CHARS_PER_TOKEN
    if task == "niah_multikey":
        keys = rng.sample(WORDS, 6)
        values = {k: str(rng.randrange(1_000_000, 9_999_999)) for k in keys}
        query = keys[0]
        needles = [f"One of the special magic numbers for {k} is: {values[k]}." for k in keys]
        context = _haystack(needles, rng, target)
        prompt = (f"Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n"
                  f"{context}\nWhat is the special magic number for {query} mentioned in the provided text?")
        gold = [values[query]]
    elif task == "variable_tracking":
        names = ["".join(rng.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(12)]
        value = str(rng.randrange(10_000, 99_999))
        chain, decoy = names[:5], names[5:10]
        needles = [f"VAR {chain[0]} = {value}."] + [f"VAR {chain[i]} = VAR {chain[i - 1]}." for i in range(1, len(chain))]
        other = str(rng.randrange(10_000, 99_999))
        needles += [f"VAR {decoy[0]} = {other}."] + [f"VAR {decoy[i]} = VAR {decoy[i - 1]}." for i in range(1, len(decoy))]
        context = _haystack(needles, rng, target)
        prompt = (f"Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
                  f"Question: Find all variables that are assigned the value {value} in the text above.")
        gold = chain
    elif task == "common_words":
        common = rng.sample(WORDS, 10)
        rare = [w for w in WORDS if w not in common]
        words: list[str] = []
        per = max(1, target // (8 * 40))
        for w in common:
            words += [w] * (per * 3)
        # Measured as the lines will be written ("12345. word\n"), not as bare words.
        size = sum(len(f"{i + 1}. {w}\n") for i, w in enumerate(words))
        while size < target - 400:
            w = rng.choice(rare)
            words.append(w)
            size += len(f"{len(words)}. {w}\n")
        rng.shuffle(words)
        context = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))
        prompt = (f"Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n"
                  f"{context}\nQuestion: What are the 10 most common words in the above list?")
        top = [w for w, _ in Counter(words).most_common(10)]
        gold = top
    else:
        raise ValueError(f"unknown RULER task {task!r}; known: {', '.join(TASKS)}")
    return Item(suite="ruler", native_id=f"{task}-{target_tokens}-{seed}", messages=[{"role": "user", "content": prompt}],
                gold=gold, meta={"task": task, "target_tokens": target_tokens})


class RULER(Suite):
    name, domain, licence = "ruler", "long_ctx", "Apache-2.0"
    source = "github.com/NVIDIA/RULER (task templates and metric; contexts generated locally)"
    revision = "ruler-subset-1"  # the generator's own version: bump it and every item changes
    notes = "approximate tokens (4 chars/token); 3 of RULER's 13 tasks"

    def __init__(self, target_tokens: int = 32_000, per_task: int = 12):
        self.target_tokens, self.per_task = target_tokens, per_task

    def load(self) -> list[Item]:
        return [generate(task, seed, self.target_tokens) for task in TASKS for seed in range(self.per_task)]

    def check(self, item: Item, answer: str) -> float:
        return string_match_all(answer, item.gold)

    def stratum(self, item: Item) -> str:
        return item.meta["task"]
