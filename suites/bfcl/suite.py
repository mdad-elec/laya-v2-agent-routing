"""BFCL v4 multi-turn base (Berkeley, Apache-2.0): tool use across turns, state-checked -- `tools_multiturn`.

Entries, ground truth and function docs come from ShishirPatil/gorilla at a pinned commit; the
simulated APIs (file system, trading bot, travel, vehicle, ...), the executor and the checker are
BFCL's own code, vendored (vendor/NOTICE). An episode runs the way BFCL's prompting mode runs it:
BFCL's default system prompt, calls written as a Python-style list `[f(a=1), g()]`, each step's
calls executed on the simulated APIs, until the model answers without calls (that turn is done)
or 20 steps pass. BFCL's multi_turn_checker then compares the final API states and responses
against the ground truth: 1 when valid, else 0.

Text protocol for every seat, on purpose: a CLI seat cannot be handed native tools, and one
protocol keeps seats comparable. One disclosed deviation from BFCL: execution results go back as
a user message ("Execution results: ..."), not a `tool`-role message, which CLI seats do not take.
Replies are decoded with `ast`, never `eval`: anything but a list of plain calls is no calls.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.request
import uuid

from suites.base import Item, Suite

REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BASE = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{REVISION}/berkeley-function-call-leaderboard/bfcl_eval/data"
MAX_STEPS = 20  # BFCL's MAXIMUM_STEP_LIMIT

# BFCL's default prompting system prompt (constants/default_prompts.py, DEFAULT_SYSTEM_PROMPT_FORMAT:
# python calls, no tag, JSON function docs, plaintext, classic style), assembled verbatim.
_SYSTEM = (
    "You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, "
    "you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it "
    "out. If the given question lacks the parameters required by the function, also point it out.\n\n"
    "You should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it "
    "in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT "
    "include any other text in the response.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output "
    "functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to "
    "call, the system will consider the current turn complete and proceed to the next turn or task.\n\n"
    "Here is a list of functions in json format that you can invoke.\n{functions}\n"
)
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I)


def system_prompt(functions: list[dict]) -> str:
    return "You are an expert in composing functions." + _SYSTEM[len("You are an expert in composing functions."):].replace(
        "{functions}", json.dumps(functions, indent=4))


def decode_calls(text: str) -> list[str]:
    """`[f(a=1), g()]` -> ["f(a=1)", "g()"]; anything else -> []."""
    body = _FENCE.sub("", text.strip()).strip()
    if not body.startswith("["):
        return []
    try:
        tree = ast.parse(body, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(tree.body, ast.List):
        return []
    calls = []
    for node in tree.body.elts:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return []
        try:
            # Positional arguments are BFCL's own form too (its ground truth has sort('final_report.pdf')).
            for arg in node.args:
                if isinstance(arg, ast.Starred):  # *args
                    return []
                ast.literal_eval(arg)
            for kw in node.keywords:
                if kw.arg is None:  # **kwargs
                    return []
                ast.literal_eval(kw.value)  # literal values only
        except (ValueError, SyntaxError):
            return []
        calls.append(ast.unparse(node))
    return calls


def _fetch_jsonl(path, url) -> list[dict]:
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class BFCL(Suite):
    name, domain, licence = "bfcl_multi_turn", "tools_multiturn", "Apache-2.0"
    source = "github.com/ShishirPatil/gorilla berkeley-function-call-leaderboard (BFCL_v4_multi_turn_base)"
    revision = REVISION
    notes = "text protocol; execution results returned as a user message (BFCL uses the tool role)"

    def load(self) -> list[Item]:
        from suites.bfcl.vendor.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

        d = self.cache_dir()
        entries = _fetch_jsonl(d / "multi_turn_base.json", f"{BASE}/BFCL_v4_multi_turn_base.json")
        answers = {a["id"]: a["ground_truth"] for a in _fetch_jsonl(d / "possible_answer.json", f"{BASE}/possible_answer/BFCL_v4_multi_turn_base.json")}
        docs: dict[str, list] = {}
        items = []
        for e in entries:
            functions = []
            for cls in e["involved_classes"]:
                file = MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]
                if file not in docs:
                    docs[file] = _fetch_jsonl(d / f"doc_{file}", f"{BASE}/multi_turn_func_doc/{file}")
                functions += [f for f in docs[file] if f["name"] not in (e.get("excluded_function") or [])]
            items.append(Item(suite=self.name, native_id=e["id"], messages=[],
                              gold={"entry": e, "ground_truth": answers[e["id"]], "functions": functions},
                              meta={"classes": "+".join(sorted(e["involved_classes"])), "turns": len(e["question"])}))
        return items

    def check(self, item: Item, answer: str) -> float:
        raise TypeError("BFCL is scored by run(): an episode, not one answer")

    def run(self, item: Item, ask) -> tuple[float, list]:
        from suites.bfcl.vendor.multi_turn_checker import multi_turn_checker
        from suites.bfcl.vendor.multi_turn_utils import execute_multi_turn_func_call

        entry = item.gold["entry"]
        # BFCL caches API instances in globals keyed by model name + entry id: a fresh name per run
        # keeps one episode's state from leaking into the next.
        run_name = f"laya_g1_{uuid.uuid4().hex}"
        messages = [{"role": "system", "content": system_prompt(item.gold["functions"])}]
        decoded_turns: list[list[list[str]]] = []
        for turn in entry["question"]:
            messages += turn
            steps: list[list[str]] = []
            for _ in range(MAX_STEPS):
                reply = ask(messages)
                messages.append({"role": "assistant", "content": reply})
                calls = decode_calls(reply)
                if not calls:
                    break
                steps.append(calls)
                results, _ = execute_multi_turn_func_call(
                    func_call_list=calls, initial_config=entry["initial_config"], involved_classes=entry["involved_classes"],
                    model_name=run_name, test_entry_id=entry["id"], long_context=False, is_evaL_run=False)
                messages.append({"role": "user", "content": "Execution results:\n" + "\n".join(f"{c}: {r}" for c, r in zip(calls, results))})
            decoded_turns.append(steps)
        verdict = multi_turn_checker(decoded_turns, item.gold["ground_truth"], entry, "multi_turn_base", run_name)
        return (1.0 if verdict.get("valid") else 0.0), messages

    def stratum(self, item: Item) -> str:
        return item.meta["classes"]
