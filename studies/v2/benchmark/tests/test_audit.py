import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit  # noqa: E402
from audit import ROOT, audit_run, load_cases, verify_source  # noqa: E402
from metrics import quality  # noqa: E402
from prompts import OPTION_ORDER, QUESTIONS, TIERS, digest, interpret, requests_for  # noqa: E402

PROBS = {'small': [0.7, 0.2, 0.1], 'medium': [0.2, 0.6, 0.2], 'powerful': [0.1, 0.3, 0.6]}


def fake_answers(tier):
    probs = dict(zip(TIERS, PROBS[tier]))
    # `confidence` deliberately differs from P(chosen): interpret must ignore it.
    return {'tier': {'type': 'choice', 'choice': tier, 'probabilities': probs, 'confidence': 0.05}}


class AuditTests(unittest.TestCase):
    def test_source_and_manifest_hashes(self):
        verify_source()
        cases, manifest = load_cases()
        self.assertEqual(manifest['cases_sha256'], digest(cases))
        self.assertEqual(manifest['requests_sha256'], digest({c['id']: requests_for(c) for c in cases}))
        self.assertEqual(manifest['questions_sha256'],
                         hashlib.sha256((ROOT / 'data/questions.json').read_bytes()).hexdigest())
        self.assertEqual(list(QUESTIONS['tier']['criteria']), OPTION_ORDER)
        self.assertEqual(manifest['option_order'], ['powerful', 'medium', 'small'])

    def test_case_counts(self):
        cases, manifest = load_cases()
        self.assertEqual(len(cases), 117)
        bands = {b: sum(c['band'] == b for c in cases) for b in ['trivial', 'easy', 'medium', 'hard']}
        self.assertEqual(bands, {'trivial': 30, 'easy': 30, 'medium': 29, 'hard': 28})
        tiers = {t: sum(c['expected'] == t for c in cases) for t in TIERS}
        self.assertEqual(tiers, {'small': 60, 'medium': 29, 'powerful': 28})
        splits = {s: sum(c['split'] == s for c in cases) for s in ['tune', 'held']}
        self.assertEqual(splits, {'tune': 59, 'held': 58})
        self.assertEqual((manifest['bands'], manifest['splits']), (bands, splits))

    def test_default_audit_without_archive(self):
        with mock.patch.object(audit, 'RESULTS', Path(tempfile.gettempdir()) / 'no-such-results-dir'), \
                mock.patch.object(sys, 'argv', ['audit.py']), redirect_stdout(io.StringIO()) as out:
            audit.main()
        self.assertIn('no archived results', out.getvalue())

    def fixture(self, folder, repeats=1):
        cases, manifest = load_cases()
        metadata = {k: manifest[k] for k in ['cases_sha256', 'requests_sha256', 'questions_sha256']}
        metadata.update(backend='laya', repeats=repeats, evaluated_ids=[c['id'] for c in cases])
        (folder / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        rows = []
        for repeat in range(repeats):
            for c in cases:
                answers = fake_answers(c['expected'])
                decoded = interpret(answers)
                rows.append(dict(id=c['id'], band=c['band'], expected=c['expected'], split=c['split'],
                                 repeat=repeat, request_sha256=digest(requests_for(c)['choice']), status='ok',
                                 answers=answers, predicted=decoded['predicted'], p_chosen=decoded['p_chosen'],
                                 latency_ms=12.5))
        return rows

    def write(self, folder, rows):
        (folder / 'raw.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')

    def test_clean_fixture_is_perfect(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, self.fixture(folder, repeats=3))
            result = audit_run(folder)
            self.assertEqual((result['correct'], result['n'], result['macro_f1']), (117, 117, 1))
            self.assertEqual(result['repeat_consistency']['same_tier_all_repeats'], 117)
            self.assertEqual((result['by_split']['held']['n'], result['by_band']['hard']['n']), (58, 28))

    def test_reject_corrupted_records(self):
        for corruption in ['missing', 'duplicate', 'gold', 'band', 'request', 'prediction', 'p_chosen',
                           'bad_tier', 'negative_latency', 'status']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                rows = self.fixture(folder)
                r = rows[0]
                if corruption == 'missing': rows.pop()
                elif corruption == 'duplicate': rows.append(copy.deepcopy(r))
                elif corruption == 'gold': r['expected'] = 'powerful' if r['expected'] != 'powerful' else 'small'
                elif corruption == 'band': r['band'] = 'hard' if r['band'] != 'hard' else 'easy'
                elif corruption == 'request': r['request_sha256'] = 'wrong'
                elif corruption == 'prediction': r['predicted'] = 'powerful' if r['predicted'] != 'powerful' else 'small'
                elif corruption == 'p_chosen': r['p_chosen'] = r['answers']['tier']['confidence']
                elif corruption == 'bad_tier': r['predicted'] = 'enormous'
                elif corruption == 'negative_latency': r['latency_ms'] = -1
                else: r['status'] = 'maybe'
                self.write(folder, rows)
                with self.assertRaises(ValueError):
                    audit_run(folder)

    def test_failed_request_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = self.fixture(folder)
            rows[0].update(status='error', error_type='TimeoutError', answers=None, predicted=None,
                           p_chosen=None, latency_ms=None)
            self.write(folder, rows)
            result = audit_run(folder)
            self.assertEqual((result['n'], result['correct'], result['errors'], result['failed_requests']),
                             (117, 116, 1, 1))
            self.assertEqual((result['overspend'], result['underspend']), (0, 0))
            self.assertEqual(result['timing']['n'], 116)

    def test_macro_f1_perfect_and_failed(self):
        rows = [{'status': 'ok', 'expected': t, 'predicted': t, 'p_chosen': 0.9} for t in TIERS]
        self.assertEqual(quality(rows)['macro_f1'], 1)
        rows[0] = {'status': 'error', 'expected': 'small', 'predicted': None, 'p_chosen': None}
        result = quality(rows)
        self.assertEqual((result['n'], result['correct'], result['errors']), (3, 2, 1))
        self.assertLess(result['macro_f1'], 1)
        self.assertEqual(result['confusion_matrix']['small']['ERROR'], 1)

    def test_overspend_and_underspend_are_separate(self):
        rows = [{'status': 'ok', 'expected': 'small', 'predicted': 'powerful', 'p_chosen': 0.5},
                {'status': 'ok', 'expected': 'powerful', 'predicted': 'medium', 'p_chosen': 0.5},
                {'status': 'ok', 'expected': 'medium', 'predicted': 'medium', 'p_chosen': 0.5}]
        result = quality(rows)
        self.assertEqual((result['overspend'], result['underspend'], result['correct']), (1, 1, 1))

    def test_interpret_uses_p_chosen(self):
        decoded = interpret(fake_answers('medium'))
        self.assertEqual((decoded['predicted'], decoded['p_chosen']), ('medium', 0.6))
        self.assertNotIn('confidence', decoded)
        bad = fake_answers('small')
        bad['tier']['probabilities']['small'] = 2.0
        with self.assertRaises(ValueError):
            interpret(bad)


if __name__ == '__main__':
    unittest.main()
