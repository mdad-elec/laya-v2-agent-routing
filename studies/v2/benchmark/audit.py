"""Validate the frozen data/prompts and rescore saved runs offline. No model or API dependencies."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from metrics import summarize
from prompts import BAND_OF, OPTION_ORDER, QUESTIONS_PATH, TIERS, digest, interpret, interpret_judge, requests_for

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/v1'
BACKENDS = {'laya', 'judge-replay'}
RECORD_KEYS = {'id', 'band', 'expected', 'split', 'repeat', 'request_sha256', 'status', 'answers',
               'predicted', 'p_chosen', 'latency_ms'}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases():
    cases = [json.loads(line) for line in (ROOT / 'data/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    if digest(cases) != manifest['cases_sha256']:
        raise ValueError('Dataset hash mismatch')
    if file_sha(QUESTIONS_PATH) != manifest['questions_sha256']:
        raise ValueError('Question schema bytes changed')
    if manifest['option_order'] != OPTION_ORDER:
        raise ValueError('Option order differs from the manifest')
    if digest({c['id']: requests_for(c) for c in cases}) != manifest['requests_sha256']:
        raise ValueError('Prompt hash mismatch')
    ids = [c['id'] for c in cases]
    if len(cases) != manifest['cases'] or len(set(ids)) != len(ids):
        raise ValueError('Expected %d unique frozen cases' % manifest['cases'])
    for c in cases:
        if set(c) != {'id', 'band', 'expected', 'prompt', 'check_kind', 'split'}:
            raise ValueError('Unexpected case fields: ' + c['id'])
        if c['band'] not in BAND_OF or c['expected'] != BAND_OF[c['band']]:
            raise ValueError('Expected tier does not follow the band policy: ' + c['id'])
        if c['split'] not in ('tune', 'held') or not c['prompt'].strip():
            raise ValueError('Invalid split or empty prompt: ' + c['id'])
    for field, key in [('bands', 'band'), ('expected', 'expected'), ('splits', 'split')]:
        if {k: sum(c[key] == k for c in cases) for k in manifest[field]} != manifest[field] \
                or sum(manifest[field].values()) != len(cases):
            raise ValueError('Manifest %s counts do not match the cases' % field)
    return cases, manifest


def audit_run(folder):
    folder = Path(folder)
    cases, manifest = load_cases()
    by_id = {c['id']: c for c in cases}
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    for key in ['cases_sha256', 'requests_sha256', 'questions_sha256']:
        if metadata.get(key) != manifest[key]:
            raise ValueError('Run uses a different frozen protocol')
    backend = metadata.get('backend')
    ids = metadata.get('evaluated_ids')
    repeats = metadata.get('repeats')
    if (backend not in BACKENDS or not isinstance(repeats, int) or repeats < 1 or not ids
            or len(set(ids)) != len(ids) or not set(ids) <= set(by_id)):
        raise ValueError('Invalid run metadata')
    rows = [json.loads(line) for line in (folder / 'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(case_id, repeat) for case_id in ids for repeat in range(repeats)}
    seen = set()
    for row in rows:
        if not RECORD_KEYS <= set(row):
            raise ValueError('Record is missing fields')
        key = (row['id'], row['repeat'])
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected prediction')
        seen.add(key)
        case = by_id[row['id']]
        if (row['expected'], row['band'], row['split']) != (case['expected'], case['band'], case['split']):
            raise ValueError('Reference label/band/split was altered')
        if row['request_sha256'] != digest(requests_for(case)['choice']):
            raise ValueError('Request hash mismatch')
        if row['status'] == 'error':
            if row['predicted'] is not None or row['p_chosen'] is not None:
                raise ValueError('Failed requests must not have a scored prediction')
            continue
        if row['status'] != 'ok' or row['predicted'] not in TIERS:
            raise ValueError('Invalid status or tier')
        if backend == 'laya':
            latency = row['latency_ms']
            if isinstance(latency, bool) or not isinstance(latency, (int, float)) \
                    or not math.isfinite(latency) or latency < 0:
                raise ValueError('Invalid latency')
            decoded = interpret(row['answers'])
            decoded.pop('probabilities')
        else:
            if row['latency_ms'] is not None:
                raise ValueError('Replayed decisions carry no latency')
            decoded = interpret_judge(row['answers'])
        for name, value in decoded.items():
            if row[name] != value:
                raise ValueError('Saved prediction differs from raw answers')
    if seen != expected:
        raise ValueError('Incomplete run: missing predictions')
    return summarize(rows, expected_repeats=repeats)


def verify_source():
    source = json.loads((ROOT / 'SOURCE.json').read_text(encoding='utf-8'))
    commit = source.get('source_commit', '')
    if len(commit) != 40 or any(ch not in '0123456789abcdef' for ch in commit):
        raise ValueError('SOURCE.json must pin a full source commit')
    for item in source['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid source path')
        if file_sha(ROOT / relative) != item['sha256']:
            raise ValueError('Frozen source bytes changed: ' + item['path'])


def archived_runs():
    if not RESULTS.is_dir():
        return []
    return sorted(p for p in RESULTS.iterdir() if p.is_dir() and (p / 'metadata.json').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='One run containing metadata.json and raw.jsonl')
    parser.add_argument('--write-summary', nargs='?', const=RESULTS / 'summary.json', type=Path,
                        help='Write recomputed JSON (default results/v1/summary.json); never overwrites')
    args = parser.parse_args()
    verify_source()
    cases, manifest = load_cases()
    print(f"data: {len(cases)} cases, bands {manifest['bands']}, splits {manifest['splits']}; "
          'hashes, labels, requests and option order verified')
    folders = [args.run_dir] if args.run_dir else archived_runs()
    if not folders:
        print('results: no archived results under results/v1 (data and prompts audited only)')
    results = {p.name: audit_run(p) for p in folders}
    if args.write_summary:
        if not results:
            raise SystemExit('Nothing to summarise: no run audited')
        if args.write_summary.exists():
            raise SystemExit('Refusing to overwrite an existing summary')
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n',
                                      encoding='utf-8', newline='\n')
    for name, r in results.items():
        print(f"{name}: {r['correct']}/{r['n']} (first repeat); macro-F1={r['macro_f1']:.3f}; "
              f"over={r['overspend']} under={r['underspend']}; {r['failed_requests']} failed requests; "
              f"p50={r['timing']['p50_ms']} ms")


if __name__ == '__main__':
    main()
