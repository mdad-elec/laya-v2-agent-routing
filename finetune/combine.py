"""The combiner: a multinomial logistic regression over the brain's own probabilities, applied in
the sidecar so the gateway reads one calibrated answer per decision and stays policy-only.

A combiner file (written by scripts/ab/combiner.py, fitted on tune only):
    {"target": "tier", "features": [["tier", "small"], ["tier", "medium"], ...],
     "classes": ["small", "medium", "powerful"], "coef": [[...], ...], "intercept": [...]}
`apply` rewrites the target question's `probabilities` and `choice`; every other answer is untouched.
Stdlib only, so the serving side needs no sklearn.
"""
import hashlib
import json
import math


def load(path):
    raw = open(path, "rb").read()
    body = json.loads(raw)
    body["sha256"] = hashlib.sha256(raw).hexdigest()
    return body


def predict(probs_by_question, combiner):
    x = [float((probs_by_question.get(q) or {}).get(opt, 0.0)) for q, opt in combiner["features"]]
    logits = [b + sum(w * v for w, v in zip(row, x)) for row, b in zip(combiner["coef"], combiner["intercept"])]
    top = max(logits)
    exps = [math.exp(z - top) for z in logits]
    total = sum(exps)
    return {c: e / total for c, e in zip(combiner["classes"], exps)}


def apply(result, combiner):
    answers = result.get("answers") or {}
    probs = {qid: a.get("probabilities") for qid, a in answers.items() if isinstance(a, dict)}
    target = combiner["target"]
    if target not in answers:
        return result
    p = predict(probs, combiner)
    answers[target] = dict(answers[target], probabilities=p, choice=max(p, key=p.get),
                           combined_from=sorted({q for q, _ in combiner["features"]}))
    return result
