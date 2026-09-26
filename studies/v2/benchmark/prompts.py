"""Frozen v1 label policy, question schema and decoding rules. See README for provenance.

The label policy below is quoted from glukicov/laya_router `data/README.md` at commit
a2278d969067 (Apache-2.0). It governs the `expected` label of every case; it is never sent
to the model. What the model sees is the question schema in `data/questions.json`.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

POLICY = (
    'tier: the cheapest tier that can answer the request well. '
    'small: a lookup, a greeting, a format change, a short rewrite, a one-line answer. '
    'medium: several steps, ordinary code, a summary that needs judgement, a routine explanation. '
    'powerful: long multi-step reasoning, specialist knowledge, system design, or consequences in '
    'money, law, health or safety.'
)
POLICY_ATTRIBUTION = 'glukicov/laya_router data/README.md @ a2278d969067 (Apache-2.0)'

# Difficulty band of each case -> the tier it should be routed to.
BAND_OF = {'trivial': 'small', 'easy': 'small', 'medium': 'medium', 'hard': 'powerful'}
TIERS = ['small', 'medium', 'powerful']
COST = {'small': 0, 'medium': 1, 'powerful': 2}

# Option order is part of the prompt: a `choice` question is laid out as
# `[MASK] opt0 [MASK] opt1 ...`, so the dict order of `criteria` is load-bearing.
# json.load preserves it; the file is hashed as bytes, never re-serialised.
QUESTIONS_PATH = ROOT / 'data/questions.json'
QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))
QUESTION_ID = 'tier'
OPTION_ORDER = ['powerful', 'medium', 'small']
if list(QUESTIONS) != [QUESTION_ID] or list(QUESTIONS[QUESTION_ID]['criteria']) != OPTION_ORDER:
    raise ValueError('data/questions.json does not carry the frozen tier question in its frozen order')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    """sha256 of the canonical (key-sorted) form. Option order is guarded separately:
    by OPTION_ORDER above and by the byte hash of data/questions.json."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def requests_for(case):
    return {'choice': {'state': {'request': case['prompt']}, 'questions': QUESTIONS}}


def interpret(answers):
    """Decode a Laya `answers` object. p_chosen is the chosen option's probability, not the
    SDK's `confidence` field (which is an entropy-based score on a different scale)."""
    a = answers[QUESTION_ID]
    probs = a['probabilities']
    pred = a['choice']
    if pred not in TIERS or set(probs) != set(TIERS):
        raise ValueError('Invalid choice schema')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1
           for x in probs.values()) or abs(sum(probs.values()) - 1) > 0.01:
        raise ValueError('Invalid probabilities')
    return {'predicted': pred, 'p_chosen': probs[pred], 'probabilities': probs}


def interpret_judge(answers):
    """Decode a replayed judge decision: `{"band": <band or tier>}` folded to a tier."""
    band = answers['band']
    pred = BAND_OF.get(band, band if band in TIERS else None)
    if pred is None:
        raise ValueError('Invalid judge band')
    return {'predicted': pred, 'p_chosen': None}
