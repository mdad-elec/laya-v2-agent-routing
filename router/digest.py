"""The session digest: what Laya-g1 reads instead of the bare last message (PREREG-g1 Amendment 1).

A digest is built from an OpenAI-shape `messages` array (plus the `tools` offered, if any) and is
the same whatever the source: a chat turn, an agent step, a TwinRouterBench prefix. Keys are
fixed and ordered, every field is clipped by words, so the whole thing stays within Laya's
512-token window (`BUDGET_WORDS`), and it never carries a catalogue model name: names passed as
`redact` become `<model>` (seat lists in a system prompt collapsed v2 from 0.711 to 0.342).

- harness:  the system instruction's head: which agent/harness this session runs in.
- goal:     the FIRST user turn: the standing objective a short follow-up is read against.
- request:  the latest user turn, head + tail, when it is not the goal itself.
- prior_user: up to two user turns between the goal and the request.
- tools_offered: how many tools, by category.
- tool_rounds:  rounds so far; the last round's categories and call names; errors, a critical
            error and a test pass within the last WINDOW rounds; the last result's tail.
- progress: the latest assistant text (its stated next step).
- context:  a size bucket of the whole conversation; turn: the number of user turns.

The tool vocabulary and the error / test-pass evidence mirror dss-gateway's router.rs
(`builtin_semantic`, `error_evidence`, `is_test_pass`) so the Rust builder can match the golden
fixtures in router/fixtures/digest byte for byte.
"""
from __future__ import annotations

import re
from collections import Counter

WINDOW = 4
CAPS = {"harness": 16, "goal": 48, "request_head": 40, "request_tail": 16, "prior": 16, "result": 24, "progress": 30}
MAX_CALLS = 4
CHARS_PER_WORD = 6  # a field is also cut at 6 code points per allowed word: code and minified text pack many tokens into one "word"
BUDGET_WORDS = 300
BUDGET_TOKENS = 480  # p99 in Laya's own tokenizer, under its 512-token window (router/tests: measured on real sessions)

_TEST = ("test", "pytest", "jest", "vitest", "cargo_test", "spec")
_SHELL = ("bash", "shell", "exec", "run_command", "terminal", "powershell")
_PLAN = ("plan", "todo", "task", "think", "outline")
_MUTATE = ("write", "edit", "create", "update", "delete", "remove", "apply", "patch", "send", "post", "put", "move", "rename", "insert", "submit", "propose")
_OBSERVE = ("read", "grep", "glob", "ls", "list", "get", "search", "find", "cat", "view", "fetch", "query", "describe", "lookup", "stat", "head", "tail", "show")
_CRITICAL = ("traceback (most recent call last)", "panicked at", "segmentation fault", "fatal:", "fatal error", "out of memory", '"is_error":true', '"is_error": true')
_SOME = ("error", "exception", "failed", "failure", "permission denied", "not found", "enoent", "no such file", "command not found", "exit code 1", "exit code 2",
         "exit status 1", "http 4", "http 5", "status 4", "status 5", "timed out", "timeout")


def classify_tool(name: str) -> str:
    lower = name.lower()
    last = re.split(r"[.:/]", lower)[-1]
    tokens = [t for t in re.split(r"[^a-z]", last) if t]

    def has(needles):
        return any(t == n or (len(n) >= 4 and t.startswith(n)) for n in needles for t in tokens)

    for category, needles in (("test", _TEST), ("shell", _SHELL), ("plan", _PLAN), ("mutate", _MUTATE), ("observe", _OBSERVE)):
        if has(needles):
            return category
    return "unknown"


def error_evidence(result: str) -> str:
    lower = result.lower()
    if any(c in lower for c in _CRITICAL):
        return "critical"
    if any(c in lower for c in _SOME):
        return "some"
    return "none"


def is_test_pass(result: str) -> bool:
    lower = result.lower()
    return (("passed" in lower) or ("test result: ok" in lower) or ("all tests pass" in lower) or (" ok " in lower)) and error_evidence(result) == "none"


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def _words(text: str) -> list[str]:
    return text.split()


def clip(text: str, n: int) -> str:
    return " ".join(_words(text)[:n])[: n * CHARS_PER_WORD]


def clip_tail(text: str, n: int) -> str:
    return " ".join(_words(text)[-n:])[-n * CHARS_PER_WORD:] if n else ""


def head_tail(text: str, head: int, tail: int) -> str:
    ws = _words(text)
    if len(ws) <= head + tail:
        return " ".join(ws)[: (head + tail) * CHARS_PER_WORD]
    return clip(" ".join(ws[:head]), head) + " ... " + clip_tail(" ".join(ws[-tail:]), tail)


def _redactor(names: list[str]):
    names = sorted({n for n in names if n}, key=len, reverse=True)
    if not names:
        return lambda s: s
    pattern = re.compile("|".join(re.escape(n) for n in names), re.I)
    return lambda s: pattern.sub("<model>", s)


def _rounds(messages: list[dict]) -> list[dict]:
    """dss-gateway `tool_rounds`: an assistant message with tool_calls opens a round, the tool
    messages after it are its results; a result with no opening call still counts."""
    rounds, open_round = [], None
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            if open_round is not None:
                rounds.append(open_round)
                open_round = None
            names = [c.get("function", {}).get("name") for c in m.get("tool_calls") or [] if c.get("function", {}).get("name")]
            if names:
                open_round = {"calls": names, "results": []}
        elif role == "tool":
            text = text_of(m.get("content"))
            if open_round is None:
                open_round = {"calls": [], "results": []}
            open_round["results"].append(text)
    if open_round is not None:
        rounds.append(open_round)
    return rounds


def _bucket(chars: int) -> str:
    tokens = chars // 4
    return "<4k" if tokens < 4000 else "4-20k" if tokens < 20000 else "20-64k" if tokens < 64000 else ">64k"


def build(messages: list[dict], tools: list[dict] | None = None, redact: list[str] | None = None) -> dict:
    r = _redactor(redact or [])
    system = " ".join(text_of(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    users = [text_of(m.get("content")) for m in messages if m.get("role") == "user"]
    assistant_texts = [text_of(m.get("content")) for m in messages if m.get("role") == "assistant" and text_of(m.get("content")).strip()]
    rounds = _rounds(messages)
    recent = rounds[-WINDOW:]
    results = [t for rd in recent for t in rd["results"]]
    evidence = [error_evidence(t) for t in results]
    last = rounds[-1] if rounds else {"calls": [], "results": []}
    offered = [(t.get("function") or t).get("name", "") for t in tools or []]
    last_categories: list[str] = []
    for name in last["calls"]:
        if classify_tool(name) not in last_categories:
            last_categories.append(classify_tool(name))
    return {
        "harness": r(clip(system, CAPS["harness"])),
        "goal": r(clip(users[0], CAPS["goal"])) if users else "",
        "request": r(head_tail(users[-1], CAPS["request_head"], CAPS["request_tail"])) if len(users) > 1 else "",
        "prior_user": [r(clip(u, CAPS["prior"])) for u in users[1:-1][-2:]],
        "tools_offered": {"n": len(offered), "categories": dict(sorted(Counter(classify_tool(n) for n in offered).items()))},
        "tool_rounds": {"n": len(rounds), "last": last_categories, "last_calls": [r(clip(n, 4)) for n in last["calls"][:MAX_CALLS]],
                        "errors": sum(e != "none" for e in evidence), "critical": "critical" in evidence,
                        "tests_passed": any(is_test_pass(t) for t in results),
                        "result": r(clip_tail(last["results"][-1], CAPS["result"])) if last["results"] else ""},
        "progress": r(clip(assistant_texts[-1], CAPS["progress"])) if assistant_texts else "",
        "context": _bucket(sum(len(text_of(m.get("content"))) for m in messages)),
        "turn": len(users),
    }


def render(d: dict) -> str:
    """The text Laya reads. One line per key, always in this order, even when empty."""
    tr, to = d["tool_rounds"], d["tools_offered"]
    yn = {True: "yes", False: "no"}
    return "\n".join([
        f"harness: {d['harness']}",
        f"goal: {d['goal']}",
        f"request: {d['request']}",
        f"prior: {' | '.join(d['prior_user'])}",
        "tools: n=" + str(to["n"]) + "".join(f" {k}={v}" for k, v in to["categories"].items()),
        f"rounds: n={tr['n']} last={','.join(tr['last'])} calls={','.join(tr['last_calls'])} errors={tr['errors']} critical={yn[tr['critical']]} tests_passed={yn[tr['tests_passed']]}",
        f"result: {tr['result']}",
        f"progress: {d['progress']}",
        f"context: {d['context']} turn: {d['turn']}",
    ])
