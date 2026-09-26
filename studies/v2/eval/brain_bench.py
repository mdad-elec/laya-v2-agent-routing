#!/usr/bin/env python3
"""Measure the brain before anything is built on it (rule 3).

Runs the labelled corpus straight at a System-One sidecar — no Rust, no gateway, no seats — and
reports what the router's decision would cost and how often it would be right. Ship 0 of the
router A/B exists so that if Laya cannot answer in time on this hardware we learn it here, for the
price of a script, instead of after a crate and an e2e lane have been written around it.

Four things are being separated on purpose:

  * **the checkpoint** — `laya` (512 ctx) against `typed-decisions` (1024 ctx);
  * **the wording** — the tier descriptions are three sentences someone wrote, and upstream
    measured a 21-point accuracy swing between two of them, so any single number here is partly a
    number about prose;
  * **the option order** — a `choice` question renders its options positionally (`[MASK] opt0
    [MASK] opt1 …`), so two orders of the same three tiers are two different prompts; measured here
    the spread within one wording reached 8.3 points, so it is ablated rather than left to whatever
    order someone typed;
  * **the corpus** — the borrowed 180 (whose labels a blind pass agrees with 0.817 of the time) and
    our own DSS rows (whose labels are a PROPOSAL until the owner audits them, and which are
    therefore reported separately and never folded into the headline).

Usage:
    scripts/laya/systemone_server.py &                       # or two, on 8099 and 8100
    python3 scripts/ab/brain_bench.py --base-url http://127.0.0.1:8099
    python3 scripts/ab/brain_bench.py --base-url http://127.0.0.1:8099 --json results/brain/ckpt.json

Stdlib only: it must run on a box where the venv holds torch and nothing else is installed.
"""
import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time
import copy
import subprocess
import urllib.error
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
REPO = os.path.dirname(os.path.dirname(HERE))
TIER_ORDER = ("small", "medium", "powerful")
WIRE_FIXTURE = os.path.join(CORPUS, "fixtures", "wire-state.json")
STATE_SHAPES = ("bare", "wrapped", "wrapped-noenv")
BAND_OF = {"trivial": "small", "easy": "small", "medium": "medium", "hard": "powerful"}   # = checks.BAND_OF
CALIBRATION_BINS = 10

# ---------------------------------------------------------------------------
# The wordings under test.
#
# `shipped` and the three variants are `glukicov/laya_router`'s (Apache-2.0, a2278d969067), kept
# verbatim so our numbers sit next to their published ones: shipped 0.600, names-only 0.428,
# example-led 0.639, cost-framed 0.439 on the 180. `dss` is ours — example-led (their winner) with
# the examples replaced by the work this estate actually asks for.
# ---------------------------------------------------------------------------
SHIPPED = {
    "small": "routine and predictable: a lookup, a greeting, a format change, a short rewrite, a one-line answer",
    "medium": "general analysis: several steps, ordinary code, a summary that needs judgement, a routine explanation",
    "powerful": ("complex or high-risk: long multi-step reasoning, specialist knowledge, system design, "
                 "or consequences in money, law, health or safety"),
}
NAMES_ONLY = {"small": "", "medium": "", "powerful": ""}
EXAMPLE_LED = {
    "small": "like: convert these units, fix this typo, what is the capital of Peru, reformat this list",
    "medium": "like: write this function, summarise this document, explain this error, draft this email",
    "powerful": "like: design this system, is this contract enforceable, prove this, plan this migration",
}
COST_FRAMED = {
    "small": "cheapest: use it whenever it would plausibly be enough",
    "medium": "roughly ten times the cost of small: use it when small would clearly fail",
    "powerful": "roughly a hundred times the cost of small: use it only when being wrong would be expensive",
}
DSS = {
    "small": "like: which port does the api listen on, rename this tile, restart the node, reformat this list",
    "medium": "like: write this query, explain this 409, summarise these minutes, add this column to the dashboard",
    "powerful": "like: design this rollout, plan this migration and its rollback, decide what we spend, prove this invariant holds",
}
WORDINGS = {"shipped": SHIPPED, "names-only": NAMES_ONLY, "example-led": EXAMPLE_LED,
            "cost-framed": COST_FRAMED, "dss": DSS}

BOOLEAN_QUESTIONS = {
    "needs_tools": {
        "type": "noul",
        "instructions": ("Does answering `request` need information the request does not already contain, such as a "
                         "live system, private records, or a current fact? Answer false when the material to work on "
                         "arrives with the request."),
    },
    "is_sensitive": {
        "type": "noul",
        "instructions": ("Does `request` carry real-world consequences in money, law, health or safety, such that a "
                         "wrong answer would cause harm?"),
    },
}


def questions_for(wording, with_booleans=True, order=None):
    """The one schema, with only the tier descriptions swapped.

    `order` re-orders the choice options. It is a real axis, not presentation: laya builds the
    sequence as `[MASK] opt0 [MASK] opt1 ...` and reads each option off its marker position, so the
    order IS part of the prompt. Measured on this box it moved accuracy further than either the
    wording or the checkpoint did, which is why it is ablated here rather than left to whatever
    order someone happened to type the dict in.
    """
    criteria = dict(wording)
    if order:
        missing = [t for t in criteria if t not in order]
        if missing or len(order) != len(criteria):
            raise ValueError("order %r does not cover exactly the tiers %r" % (order, sorted(criteria)))
        criteria = {tier: criteria[tier] for tier in order}
    questions = {"tier": {
        "type": "choice",
        "instructions": ("Which model tier should answer the user's `request`? Pick the cheapest tier that can do "
                         "it well."),
        "criteria": criteria,
    }}
    if with_booleans:
        questions.update({k: dict(v) for k, v in BOOLEAN_QUESTIONS.items()})
    return questions


# ---------------------------------------------------------------------------
# Scoring. Ported from laya_router/metrics.py (Apache-2.0) so the numbers are comparable, and kept
# dependency-free so the whole computation can be read end to end.
# ---------------------------------------------------------------------------
def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def macro_f1(predicted, gold):
    """Unweighted mean F1 over the classes present in the gold labels.

    Macro rather than micro: a router that never predicts `medium` at all should be visibly
    penalised, and upstream's `names-only` variant did exactly that (63.9% of its answers were
    `medium`, and accuracy alone barely noticed).
    """
    classes = sorted(set(gold))
    scores = []
    for label in classes:
        tp = sum(1 for p, g in zip(predicted, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def expected_calibration_error(confidences, correct, bins=CALIBRATION_BINS):
    """Average gap between stated confidence and observed accuracy, weighted by bin population.

    This is the number that decides whether `DSS_ROUTER_MIN_CONFIDENCE` can exist at all. A router
    can be accurate and badly calibrated at once, and only the second failure makes "below 0.3, do
    not downgrade" a rule that means anything.
    """
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, c in enumerate(confidences)
                   if (c > low or (index == 0 and c >= low)) and c <= high]
        if not members:
            continue
        accuracy = sum(correct[i] for i in members) / len(members)
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def spend_rates(predicted, gold):
    """Overspend and underspend, separately — they are different incidents.

    Overspend is a line on an invoice. Underspend is an agent that answered badly and a person who
    had to notice. One accuracy number hides which one a router is making.
    """
    over = sum(TIER_ORDER.index(p) > TIER_ORDER.index(g) for p, g in zip(predicted, gold))
    under = sum(TIER_ORDER.index(p) < TIER_ORDER.index(g) for p, g in zip(predicted, gold))
    return over / len(gold), under / len(gold)


# ---------------------------------------------------------------------------
# The wire.
# ---------------------------------------------------------------------------
def request_body(state, questions):
    """The bytes posted. json.dumps keeps dict insertion order, which is the option order the SDK
    renders positionally — never sort_keys here."""
    return json.dumps({"state": state, "questions": questions}).encode()


def load_schema(path):
    """A questions file, in its own key order (json.load keeps it)."""
    with open(path) as handle:
        return json.load(handle)


def render_state(message, shape, state_key="request"):
    """The state the brain is shown.

    bare           {state_key: message} — what Ship 0 measured
    wrapped        the router's own state as captured off the wire (`fixtures/wire-state.json`):
                   request + session + environment, in the order the router sent it
    wrapped-noenv  the same without `environment` (the servable rung list)
    """
    if shape == "bare":
        return {state_key: message}
    if shape not in STATE_SHAPES:
        raise ValueError("unknown state shape %r; known: %s" % (shape, ", ".join(STATE_SHAPES)))
    state = copy.deepcopy(_wire_state())
    state["request"] = message[:4000]          # BRAIN_REQUEST_CHARS in the platform's router
    if shape == "wrapped-noenv":
        state.pop("environment", None)
    return state


_WIRE = None


def _wire_state():
    global _WIRE
    if _WIRE is None:
        with open(WIRE_FIXTURE) as handle:
            _WIRE = json.load(handle)["body"]["state"]
    return _WIRE


def rows_for(corpus, split):
    """Rows as {id, message, tier, difficulty, corpus} from a corpus and a FROZEN split
    (scripts/ab/corpus/splits/). `cells-pilot` asks carry their band folded to a tier."""
    import split_tools
    if corpus == "tier-240":
        rows = [dict(r, corpus="requests") for r in load_corpus(os.path.join(CORPUS, "requests.jsonl"))] + \
               [dict(r, corpus="dss-requests") for r in load_corpus(os.path.join(CORPUS, "dss-requests.jsonl"))]
        frozen = split_tools.load("tier-240")
    elif corpus == "cells-pilot":
        rows = [{"id": a["id"], "message": a["prompt"], "tier": BAND_OF[a["band"]], "difficulty": a["band"],
                 "corpus": "cells-pilot"} for a in load_corpus(os.path.join(CORPUS, "cells-pilot.jsonl"))]
        frozen = split_tools.load("cells-pilot")
    else:
        raise ValueError("unknown corpus %r; known: tier-240, cells-pilot" % corpus)
    if split == "all":
        return rows
    keep = set(frozen[split])
    return [r for r in rows if r["id"] in keep]


def ask(base_url, state, questions, token=None, timeout=30.0):
    body = request_body(state, questions)
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = "Bearer %s" % token
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone", data=body, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    # Measured at the CALLER, which is where the router will measure it: the server's own
    # `latency_ms` excludes the socket, and the gateway pays for the socket too.
    payload["_round_trip_ms"] = (time.perf_counter() - started) * 1000
    return payload


def health(base_url, token=None):
    request = urllib.request.Request(base_url.rstrip("/") + "/health")
    if token:
        request.add_header("authorization", "Bearer %s" % token)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def chosen(answer):
    """The picked option and the probability OF THAT OPTION.

    Deliberately not the SDK's `confidence` field: for a `choice` question that is normalised
    entropy (how peaked the distribution is), which is not on the same scale as "how likely this
    answer is right". The policy in Ship 1 thresholds on P(chosen), so the benchmark must measure
    P(chosen) or it is calibrating a different number than the one the router will use.
    """
    if answer["type"] == "choice":
        probabilities = {k: float(v) for k, v in answer["probabilities"].items()}
        return str(answer["choice"]), max(probabilities.values())
    if answer["type"] == "noul":
        p_true = float(answer["noul"])
        return ("true" if p_true >= 0.5 else "false"), max(p_true, 1.0 - p_true)
    raise ValueError("unsupported answer type %r" % answer["type"])


CHEAP_FIRST = ("small", "medium", "powerful")
COSTLY_FIRST = ("powerful", "medium", "small")


def order_tag(order):
    if tuple(order) == CHEAP_FIRST:
        return "cheap-first"
    if tuple(order) == COSTLY_FIRST:
        return "costly-first"
    return "".join(t[0] for t in order)


def resolve_orders(spec):
    """Which option orders to ablate.

    A choice question's options are rendered positionally, so two orders of the same three tiers are
    two different prompts. Measured here they differed by more than the wording did, so 'whatever
    order the dict happened to be in' is not a defensible default.
    """
    if spec == "all":
        return list(itertools.permutations(CHEAP_FIRST))
    if spec == "cheap-first":
        return [CHEAP_FIRST]
    if spec == "costly-first":
        return [COSTLY_FIRST]
    orders = []
    for chunk in spec.split(";"):
        order = tuple(t.strip() for t in chunk.split(",") if t.strip())
        if sorted(order) != sorted(CHEAP_FIRST):
            raise SystemExit("order %r must be a permutation of %s" % (chunk, ", ".join(CHEAP_FIRST)))
        orders.append(order)
    return orders


def load_corpus(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_wording(base_url, rows, wording_name, token, with_booleans=True, state_key="request", order=None,
                state_shape="bare", schema=None, record_probs=False):
    questions = schema if schema is not None else questions_for(WORDINGS[wording_name], with_booleans, order)
    records = []
    for row in rows:
        # The state key is the name the question text points at. Upstream sends {"message": …} while
        # its questions say `request`; the pointer and the key disagreeing is exactly the class of
        # defect rule 8 is about, so ours match and the mismatch is measurable with --state-key.
        payload = ask(base_url, render_state(row["message"], state_shape, state_key), questions, token)
        answers = payload["answers"]
        tier, confidence = chosen(answers["tier"])
        record = {"id": row["id"], "gold": row["tier"], "predicted": tier, "confidence": confidence,
                  "difficulty": row.get("difficulty", "clear"), "corpus": row.get("corpus"),
                  "round_trip_ms": payload["_round_trip_ms"], "server_ms": payload.get("latency_ms"),
                  "input_tokens": payload.get("usage", {}).get("input_tokens")}
        if record_probs:
            record["probabilities"] = {qid: a.get("probabilities") or ({"true": a["noul"]} if "noul" in a else None)
                                       for qid, a in answers.items()}
        for qid in ("needs_tools", "is_sensitive"):
            if qid in answers:
                value, _ = chosen(answers[qid])
                record[qid] = {"predicted": value, "gold": row.get(qid)}
        records.append(record)
    return records


def score(records, label):
    predicted = [r["predicted"] for r in records]
    gold = [r["gold"] for r in records]
    correct = [p == g for p, g in zip(predicted, gold)]
    confidences = [r["confidence"] for r in records]
    latency = [r["round_trip_ms"] for r in records]
    over, under = spend_rates(predicted, gold)
    result = {
        "set": label,
        "n": len(records),
        "accuracy": sum(correct) / len(correct),
        "macro_f1": macro_f1(predicted, gold),
        "ece": expected_calibration_error(confidences, correct),
        "overspend_rate": over,
        "underspend_rate": under,
        "share": {tier: predicted.count(tier) / len(predicted) for tier in TIER_ORDER},
        "p50_ms": percentile(latency, 0.50),
        "p95_ms": percentile(latency, 0.95),
        "mean_ms": statistics.fmean(latency),
        "mean_input_tokens": statistics.fmean([r["input_tokens"] for r in records if r["input_tokens"]]) if any(
            r["input_tokens"] for r in records) else 0,
        "by_difficulty": {},
        # The scale the confidence floor lives on (typed.rs DEFAULT_MIN_CONFIDENCE): how much of the
        # set sits under each floor, and how far mean P(chosen) sits from accuracy (negative =
        # under-confident, the direction Ship 0 measured).
        "mean_confidence": statistics.fmean(confidences),
        "signed_gap": statistics.fmean(confidences) - sum(correct) / len(correct),
        "share_below_0.35": sum(1 for c in confidences if c < 0.35) / len(confidences),
        "share_below_0.45": sum(1 for c in confidences if c < 0.45) / len(confidences),
    }
    dists = [((r.get("probabilities") or {}).get("tier"), r["gold"]) for r in records]
    if dists and all(d for d, _ in dists):
        # Proper scores on the whole distribution, not only its argmax: what the fine-tune optimizes.
        result["brier"] = statistics.fmean(sum((d.get(t, 0.0) - (1.0 if t == g else 0.0)) ** 2 for t in TIER_ORDER) for d, g in dists)
        result["soft_accuracy"] = statistics.fmean(d.get(g, 0.0) for d, g in dists)
    for bucket in sorted({r["difficulty"] for r in records}):
        members = [r for r in records if r["difficulty"] == bucket]
        hits = sum(r["predicted"] == r["gold"] for r in members)
        result["by_difficulty"][bucket] = {"n": len(members), "accuracy": hits / len(members)}
    for qid in ("needs_tools", "is_sensitive"):
        graded = [r[qid] for r in records if qid in r and r[qid].get("gold") is not None]
        if graded:
            result[qid + "_accuracy"] = sum(g["predicted"] == g["gold"] for g in graded) / len(graded)
    return result


def table(rows, title):
    out = ["", title, "-" * len(title),
           "%-24s %5s %8s %8s %7s %7s %7s %8s %8s" % ("wording/order", "n", "accuracy", "macro-F1", "ECE",
                                                      "over", "under", "p50 ms", "p95 ms")]
    for row in rows:
        out.append("%-24s %5d %8.3f %8.3f %7.3f %7.3f %7.3f %8.0f %8.0f"
                   % (row["wording"], row["n"], row["accuracy"], row["macro_f1"], row["ece"],
                      row["overspend_rate"], row["underspend_rate"], row["p50_ms"], row["p95_ms"]))
    return "\n".join(out)


def provenance(args):
    """What a Laya-v2 result must carry to be quotable (PREREG.md): the commit, the pre-registration
    blob, and the split file it was cut from."""
    def git(*cmd):
        try:
            return subprocess.check_output(["git", "-C", REPO, *cmd], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    split = {"tier-240": "tier-240", "cells-pilot": "cells-pilot"}.get(args.corpus or "")
    return {"git_sha": git("rev-parse", "HEAD"),
            "prereg_sha": git("hash-object", os.path.join(REPO, "scripts", "laya", "PREREG.md")),
            "split_sha": git("hash-object", os.path.join(CORPUS, "splits", split + ".split.json")) if split else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("LAYA_BASE_URL", "http://127.0.0.1:8099"))
    parser.add_argument("--token", default=os.environ.get("LAYA_API_TOKEN"))
    parser.add_argument("--wordings", default="shipped,example-led,dss",
                        help="comma-separated; one of %s, or 'all'" % ",".join(WORDINGS))
    parser.add_argument("--state-key", default="request",
                        help="the key the state is sent under; the question text points at this name")
    parser.add_argument("--dss", action="store_true", help="also score the unaudited DSS rows")
    parser.add_argument("--tier-only", action="store_true",
                        help="ask ONLY the tier question. laya batches one sequence per question and "
                             "re-encodes the state in each, so wall time is linear in the question "
                             "count -- this is the cheapest schema that can still route.")
    parser.add_argument("--json", help="write the full per-request records here")
    parser.add_argument("--orders", default="cheap-first",
                        help="option orders to ablate: 'cheap-first' (small,medium,powerful), "
                             "'costly-first' (powerful,medium,small), 'all' (every permutation), or "
                             "explicit orders separated by ';' e.g. 'small,medium,powerful;powerful,medium,small'")
    parser.add_argument("--emit-order", default=None,
                        help="option order for --emit-questions (default: the ablated winner's order "
                             "is not guessed; pass it explicitly)")
    parser.add_argument("--emit-questions", metavar="WORDING",
                        help="write the schema for WORDING to --emit-path and exit. The shipped "
                             "schema is emitted from the same table the ablation measured, so the "
                             "wording that scored is byte-identical to the wording that serves "
                             "(rule 5: the winner is never retyped by hand).")
    parser.add_argument("--emit-path", default=os.path.join(REPO, "scripts", "laya", "questions.json"))
    parser.add_argument("--emit-crate-path",
                        default=os.path.join(REPO, "backend", "crates", "dss-providers", "content", "laya-questions.json"),
                        help="the gateway's embedded copy, written byte-identical (check_questions.py asserts it)")
    parser.add_argument("--state-shape", default="bare", choices=STATE_SHAPES,
                        help="bare: {request} as Ship 0 measured; wrapped: the router's own state as "
                             "captured off the wire; wrapped-noenv: that without the rung list")
    parser.add_argument("--corpus", choices=["tier-240", "cells-pilot"],
                        help="read rows from a corpus through its FROZEN split (default: the legacy "
                             "borrowed-180 run, with --dss for the DSS rows)")
    parser.add_argument("--split", default="all", choices=["tune", "held", "all"])
    parser.add_argument("--schema", help="ask this questions file (its own option order) instead of a wording")
    parser.add_argument("--record-probs", action="store_true", help="keep every question's probabilities per record")
    parser.add_argument("--gate-p50-ms", type=float, default=500.0)
    parser.add_argument("--gate-accuracy", type=float, default=0.60)
    args = parser.parse_args()

    if args.emit_questions:
        if args.emit_questions not in WORDINGS:
            sys.exit("unknown wording %r; known: %s" % (args.emit_questions, ", ".join(WORDINGS)))
        emit_order = tuple(t.strip() for t in args.emit_order.split(",")) if args.emit_order else None
        schema = questions_for(WORDINGS[args.emit_questions], not args.tier_only, emit_order)
        # NOT sort_keys. laya renders a choice question's options positionally
        # (`keys = list(q["crit"].keys())` in the SDK), so the key order IS part of the prompt.
        # Sorting it would ship a schema that is not the schema the ablation scored, which is
        # the exact failure this flag exists to prevent. Both copies get the same bytes.
        text = json.dumps(schema, indent=2) + "\n"
        for path in (args.emit_path, args.emit_crate_path):
            with open(path, "w") as handle:
                handle.write(text)
        print("wrote %s  (%s, options %s, %d question(s))"
              % (args.emit_path, args.emit_questions,
                 ",".join(schema["tier"]["criteria"]), len(schema)))
        return

    orders = resolve_orders(args.orders)
    wordings = list(WORDINGS) if args.wordings == "all" else [w.strip() for w in args.wordings.split(",")]
    unknown = [w for w in wordings if w not in WORDINGS]
    if unknown:
        sys.exit("unknown wording(s): %s" % ", ".join(unknown))

    try:
        info = health(args.base_url, args.token)
    except urllib.error.URLError as error:
        sys.exit("no System-One sidecar at %s: %s\n"
                 "start one with: ~/venvs/laya/bin/python scripts/laya/systemone_server.py"
                 % (args.base_url, error))
    # The thread counts are part of the measurement, not trivia: torch's own defaults measured
    # 10x slower than a pinned pool on this box, so a latency number without them is unreadable.
    print("brain: %s on %s, ctx %d, %s/%s threads, loaded in %.1fs  (%s)"
          % (info["model"], info["device"], info["max_len"], info.get("threads", "?"),
             info.get("interop_threads", "?"), info["load_seconds"], args.base_url))

    if args.corpus:
        rows = rows_for(args.corpus, args.split)
        # The unaudited DSS rows are never folded into the headline (ATTRIBUTION.md).
        borrowed = [r for r in rows if r.get("corpus") != "dss-requests"]
        dss_rows = [r for r in rows if r.get("corpus") == "dss-requests"]
        headline = "%s-%s (%d)" % (args.corpus, args.split, len(borrowed))
    else:
        borrowed = load_corpus(os.path.join(CORPUS, "requests.jsonl"))
        dss_rows = load_corpus(os.path.join(CORPUS, "dss-requests.jsonl")) if args.dss else []
        headline = "borrowed-180"
    schema = load_schema(args.schema) if args.schema else None
    if schema is not None:
        wordings, orders = ["schema:" + os.path.basename(args.schema)], [None]

    questions_asked = 1 if args.tier_only else 1 + len(BOOLEAN_QUESTIONS)
    print("schema: %d question(s) per call%s" % (questions_asked, "  (tier only)" if args.tier_only else ""))

    if schema is not None:
        questions_asked = len(schema)
    report = {"brain": info, "base_url": args.base_url, "state_key": args.state_key,
              "state_shape": args.state_shape, "corpus": args.corpus or "legacy", "split": args.split,
              "schema": args.schema, "questions_asked": questions_asked,
              "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **provenance(args),
              "borrowed": [], "dss": [], "records": {}}

    extra = dict(state_shape=args.state_shape, schema=schema, record_probs=args.record_probs)
    for wording in wordings:
        for order in orders:
            label = wording if len(orders) == 1 else "%s/%s" % (wording, order_tag(order))
            records = run_wording(args.base_url, borrowed, wording, args.token, not args.tier_only,
                                  args.state_key, order, **extra)
            scored = score(records, headline)
            scored["wording"], scored["order"] = label, list(order) if order else None
            report["borrowed"].append(scored)
            report["records"]["borrowed/" + label] = records
            print("  %-22s %s  acc %.3f  p50 %.0fms" % (label, headline, scored["accuracy"], scored["p50_ms"]))
            if dss_rows:
                records = run_wording(args.base_url, dss_rows, wording, args.token, not args.tier_only,
                                      args.state_key, order, **extra)
                scored = score(records, "dss-%d-UNAUDITED" % len(dss_rows))
                scored["wording"], scored["order"] = label, list(order) if order else None
                report["dss"].append(scored)
                report["records"]["dss/" + label] = records
                print("  %-22s dss-60 (unaudited)  acc %.3f  p50 %.0fms"
                      % (label, scored["accuracy"], scored["p50_ms"]))

    print(table(report["borrowed"], "Borrowed 180 (glukicov/laya_router, Apache-2.0) — comparable with their published run"))
    if report["dss"]:
        print(table(report["dss"], "DSS 60 — LABELS UNAUDITED: a proposal, not a policy. Not a headline number."))

    best = max(report["borrowed"], key=lambda r: r["accuracy"])
    print("\nbest wording on the borrowed set: %r  accuracy %.3f  macro-F1 %.3f  ECE %.3f"
          % (best["wording"], best["accuracy"], best["macro_f1"], best["ece"]))
    print("per-difficulty (%s): %s" % (best["wording"], ", ".join(
        "%s %.2f (n=%d)" % (k, v["accuracy"], v["n"]) for k, v in best["by_difficulty"].items())))
    for qid in ("needs_tools", "is_sensitive"):
        if qid + "_accuracy" in best:
            print("  %s accuracy %.3f" % (qid, best[qid + "_accuracy"]))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print("wrote %s" % args.json)

    # The Ship 0 gate, stated as a number and checked here rather than eyeballed. Failing it does
    # not mean the plan is wrong — it means the sidecar moves off this CPU before Ship 2.
    #
    # Both halves are read off the SAME wording. Taking the best accuracy from one run and the best
    # latency from another would describe a configuration that does not exist: on this box the
    # fastest wording is `names-only`, which is also the least accurate by 24 points. A gate that
    # mixes them passes a router nobody would ship (rule 11).
    gate_p50, gate_p95 = best["p50_ms"], best["p95_ms"]
    print("\nGATE on the shippable wording %r (not the fastest one)" % best["wording"])
    print("GATE  p50 %.0f ms (need <= %.0f)   p95 %.0f ms   accuracy %.3f (need >= %.2f)"
          % (gate_p50, args.gate_p50_ms, gate_p95, best["accuracy"], args.gate_accuracy))
    failures = []
    if gate_p50 > args.gate_p50_ms:
        failures.append("p50 %.0f ms > %.0f ms on wording %r: this CPU is not the sidecar's host"
                        % (gate_p50, args.gate_p50_ms, best["wording"]))
    if best["accuracy"] < args.gate_accuracy:
        failures.append("accuracy %.3f < %.2f: this checkpoint does not match the published result"
                        % (best["accuracy"], args.gate_accuracy))
    if failures:
        print("GATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
