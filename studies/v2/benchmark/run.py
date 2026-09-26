"""Run the frozen routing cases through Laya, or replay a judge's saved decisions in the same shape."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from audit import ROOT, load_cases
from prompts import digest, interpret, interpret_judge, requests_for

SEED = 20260923
ALLOW = ('rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def base_row(case, repeat):
    return dict(id=case['id'], band=case['band'], expected=case['expected'], split=case['split'],
                repeat=repeat, request_sha256=digest(requests_for(case)['choice']))


def resolve_checkpoint(args, parser):
    """Return a local checkpoint directory, downloading only the runtime files for --model."""
    if args.checkpoint:
        root = args.checkpoint.resolve()
        path = root / args.subfolder if args.subfolder else root
    else:
        from huggingface_hub import snapshot_download
        prefix = args.subfolder + '/' if args.subfolder else ''
        root = Path(snapshot_download(args.model, revision=args.revision,
                                      allow_patterns=[prefix + name for name in ALLOW]))
        path = root / args.subfolder if args.subfolder else root
    if not (path / 'model.safetensors').is_file() or not (path / 'rl_agent_config.json').is_file():
        parser.error('Not a Laya checkpoint directory: model.safetensors / rl_agent_config.json missing')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['laya', 'judge-replay'], required=True)
    parser.add_argument('--checkpoint', type=Path, help='Local checkpoint directory (no download)')
    parser.add_argument('--model', help='Hub id to download (laya), or a descriptive judge name (judge-replay)')
    parser.add_argument('--subfolder', help='Checkpoint subfolder inside --checkpoint or --model')
    parser.add_argument('--revision', help='Hub revision for --model; the weight hash is always recorded')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=None, help='torch intra-op threads (default: torch)')
    parser.add_argument('--decisions', type=Path, help='judge-replay: JSONL of {"id", "band"[, "repeat"]}')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Smoke test only: evaluate the first N selected cases')
    parser.add_argument('--split', choices=['all', 'tune', 'held'], default='all')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; results must not be overwritten')
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid repeat count or limit')
    cases, manifest = load_cases()
    cases = [c for c in cases if args.split == 'all' or c['split'] == args.split]
    cases = cases[:args.limit] if args.limit else cases
    source = json.loads((ROOT / 'SOURCE.json').read_text(encoding='utf-8'))
    metadata = dict(backend=args.backend, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    cases_sha256=manifest['cases_sha256'], requests_sha256=manifest['requests_sha256'],
                    questions_sha256=manifest['questions_sha256'], option_order=manifest['option_order'],
                    benchmark_source_commit=source['source_commit'], split=args.split,
                    evaluated_ids=[c['id'] for c in cases], seed=SEED,
                    python=platform.python_version(), os=platform.system(), architecture=platform.machine(),
                    runner_sha256=file_sha(__file__))

    if args.backend == 'judge-replay':
        if not args.decisions or not args.decisions.is_file():
            parser.error('--decisions FILE is required for judge-replay')
        decisions = {}
        for line in args.decisions.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            key = (d['id'], int(d.get('repeat', 0)))
            if key in decisions:
                parser.error('Duplicate decision for %s repeat %d' % key)
            decisions[key] = d['band']
        known = {c['id'] for c in load_cases()[0]}
        if any(i not in known for i, _ in decisions):
            parser.error('Decisions file names ids that are not in the frozen cases')
        repeats = max(r for _, r in decisions) + 1 if decisions else 1
        metadata.update(model=args.model or 'judge', repeats=repeats, decisions_sha256=file_sha(args.decisions),
                        smoke_test=len(cases) != manifest['cases'], warmups=[])
        args.output.mkdir(parents=True)
        with (args.output / 'raw.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for repeat in range(repeats):
                for case in cases:
                    row = base_row(case, repeat)
                    band = decisions.get((case['id'], repeat))
                    row.update(latency_ms=None)
                    try:
                        if band is None:
                            raise KeyError('missing decision')
                        answers = {'band': band}
                        row.update(status='ok', answers=answers, **interpret_judge(answers))
                    except (KeyError, ValueError) as e:
                        row.update(status='error', error_type=type(e).__name__, answers=None,
                                   predicted=None, p_chosen=None)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        metadata['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                   encoding='utf-8', newline='\n')
        return

    if not args.checkpoint and not args.model:
        parser.error('laya needs --checkpoint DIR or --model HUB_ID')
    path = resolve_checkpoint(args, parser)
    os.environ.update(HF_HUB_OFFLINE='1', USE_TF='0', USE_TORCH='1', TOKENIZERS_PARALLELISM='false')
    import torch
    import transformers
    import laya
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    start = time.perf_counter()
    agent = laya.load(str(path), device=args.device)
    load_seconds = time.perf_counter() - start
    run = getattr(agent, 'system_one', None) or agent.predict
    package = Path(laya.__file__).resolve().parent
    try:
        commit = subprocess.check_output(['git', '-C', str(package), 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata.update(model=args.model or 'laya-local', subfolder=args.subfolder, checkpoint_revision=args.revision,
                    checkpoint_model_sha256=file_sha(path / 'model.safetensors'), checkpoint_config=agent.cfg,
                    load_seconds=load_seconds, laya=getattr(laya, '__version__', None), laya_source_commit=commit,
                    torch=torch.__version__, transformers=transformers.__version__, device=str(agent.device),
                    cpu_threads=torch.get_num_threads(), repeats=args.repeats,
                    smoke_test=len(cases) != manifest['cases'] or args.repeats != manifest['repetitions'],
                    warmups=[])

    def sync():
        if agent.device.type == 'cuda':
            torch.cuda.synchronize()
        elif agent.device.type == 'mps':
            torch.mps.synchronize()

    def call(req):
        sync()
        t0 = time.perf_counter()
        response = run(req['state'], req['questions'])
        sync()
        return response, (time.perf_counter() - t0) * 1000

    args.output.mkdir(parents=True)
    try:
        _, elapsed = call(requests_for(cases[0])['choice'])
        metadata['warmups'].append(dict(id=cases[0]['id'], latency_ms=elapsed))
    except Exception as e:  # recorded, not fatal: the timed run still accounts for every request
        metadata['warmups'].append(dict(id=cases[0]['id'], error_type=type(e).__name__))
    (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                               encoding='utf-8', newline='\n')
    jobs = [(repeat, case) for repeat in range(args.repeats) for case in cases]
    random.Random(SEED).shuffle(jobs)
    with (args.output / 'raw.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for i, (repeat, case) in enumerate(jobs):
            row = base_row(case, repeat)
            try:
                response, elapsed = call(requests_for(case)['choice'])
                decoded = interpret(response['answers'])
                decoded.pop('probabilities')
                row.update(status='ok', answers=response['answers'], latency_ms=elapsed, **decoded)
            except Exception as e:
                row.update(status='error', error_type=type(e).__name__, answers=None, predicted=None,
                           p_chosen=None, latency_ms=None)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(jobs)} requests completed', flush=True)
    metadata['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                               encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
