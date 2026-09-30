"""Question set v0 and its training labels (PREREG-g1 Amendment 7).

The brain answers `task`, `domain`, `level` and `continues_goal` over the session digest; the rest
(stage, tool loop, errors, tests, context) is computed by `router/digest.py`. Training labels are
deterministic maps from each source's own fields, stated here and committed before any run that
uses them. A dataset or benchmark with no entry is refused, never guessed.
"""
from __future__ import annotations

TASKS = ("code_fix", "code_write", "math", "knowledge_qa", "reasoning", "writing", "structured_extraction", "tool_agentic", "chat")
DOMAINS = ("software", "math", "science_medicine", "business_finance", "general", "language_creative", "other")

# LLMRouterBench / BFCL datasets → (task, domain).
DATASET_ANSWERS: dict[str, tuple[str, str]] = {
    "aime": ("math", "math"),
    "livemathbench": ("math", "math"),
    "math500": ("math", "math"),
    "mathbench": ("math", "math"),
    "arenahard_math": ("math", "math"),
    "humaneval": ("code_write", "software"),
    "mbpp": ("code_write", "software"),
    "livecodebench": ("code_write", "software"),
    "arenahard_coding": ("code_write", "software"),
    "swe-bench": ("code_fix", "software"),
    "mmlupro": ("knowledge_qa", "general"),
    "simpleqa": ("knowledge_qa", "general"),
    "medqa": ("knowledge_qa", "science_medicine"),
    "arcc": ("knowledge_qa", "science_medicine"),
    "hle": ("reasoning", "science_medicine"),
    "bbh": ("reasoning", "general"),
    "korbench": ("reasoning", "general"),
    "kandk": ("reasoning", "general"),
    "arc-agi": ("reasoning", "other"),
    "winogrande": ("reasoning", "language_creative"),
    "finqa": ("structured_extraction", "business_finance"),
    "meld": ("chat", "language_creative"),
    "emorynlp": ("chat", "language_creative"),
    "arenahard": ("chat", "general"),
    "arenahard_creative_writing": ("writing", "language_creative"),
    "tau2": ("tool_agentic", "business_finance"),
    "bfcl_multi_turn_base": ("tool_agentic", "software"),
}

# TwinRouterBench benchmarks → (task, domain). BFCL and PinchBench are both agent tool use, so
# they share an answer: the brain cannot be asked to name a benchmark.
BENCHMARK_ANSWERS: dict[str, tuple[str, str]] = {
    "swebench": ("code_fix", "software"),
    "bfcl": ("tool_agentic", "software"),
    "pinchbench": ("tool_agentic", "general"),
    "mtrag": ("knowledge_qa", "general"),
    "qmsum": ("structured_extraction", "business_finance"),
}


def _answer(table: dict[str, tuple[str, str]], name: str, what: str) -> tuple[str, str]:
    try:
        return table[name]
    except KeyError:
        raise KeyError(f"{what} {name!r} has no v0 answer in train/questions.py; add it before using it") from None


def dataset_answers(dataset: str) -> tuple[str, str]:
    return _answer(DATASET_ANSWERS, dataset, "dataset")


def benchmark_answers(benchmark: str) -> tuple[str, str]:
    return _answer(BENCHMARK_ANSWERS, benchmark, "benchmark")


assert all(t in TASKS and d in DOMAINS for t, d in {**DATASET_ANSWERS, **BENCHMARK_ANSWERS}.values())
