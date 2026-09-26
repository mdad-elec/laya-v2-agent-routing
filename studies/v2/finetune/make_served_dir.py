"""A served checkpoint directory that says what it is: weights hard-linked from a base checkpoint,
`rl_agent_config.json` patched in exactly the named temperature buckets, and a MANIFEST.json naming
the base, the weights' sha256, the temperatures, the schema and combiner it was built for, and the
fit that produced them. The sidecar reports the MANIFEST on /health; check_questions.py --served
asserts the schema; the provider record pins the sha (rule 8).

    python3 scripts/laya/make_served_dir.py --from <checkpoint dir> --out ~/laya-ckpts/laya-root-cal \\
        --temperature-by-options '{"choice:3-5": 0.672}' --questions scripts/laya/questions.json [--combiner FILE] [--fit FIT.json]
    LAYA_MODEL=~/laya-ckpts/laya-root-cal LAYA_SERVED_ID=laya-root-cal …/systemone_server.py
"""
import argparse
import hashlib
import json
import os
import shutil
import sys


def _sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _link(src, dst):
    src = os.path.realpath(src)        # an HF snapshot holds symlinks into blobs/
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(src, out, temperature_by_options, questions=None, combiner=None, fit=None):
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists; a served directory is never overwritten — pick a new name")
    cfg = json.load(open(os.path.join(src, "rl_agent_config.json")))
    known = set(cfg.get("temperature_by_options", {}))
    unknown = sorted(set(temperature_by_options) - known)
    if unknown:
        raise ValueError(f"no temperature bucket {', '.join(unknown)} in this checkpoint (it has {', '.join(sorted(known))})")
    os.makedirs(out)
    cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}), **temperature_by_options}
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    _link(os.path.join(src, "model.safetensors"), os.path.join(out, "model.safetensors"))
    for sub in ("tokenizer", "encoder"):
        s = os.path.join(src, sub)
        if os.path.isdir(s):
            os.makedirs(os.path.join(out, sub))
            for name in os.listdir(s):
                _link(os.path.join(s, name), os.path.join(out, sub, name))
    manifest = {"base": os.path.realpath(src), "model_sha256": _sha(os.path.join(out, "model.safetensors")),
                "temperature_by_options": cfg["temperature_by_options"],
                "questions_sha256": _sha(questions) if questions else None,
                "combiner_sha256": _sha(combiner) if combiner else None, "fit": fit}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature-by-options", default="{}")
    ap.add_argument("--questions")
    ap.add_argument("--combiner")
    ap.add_argument("--fit")
    args = ap.parse_args(argv)
    m = build(args.src, os.path.expanduser(args.out), json.loads(args.temperature_by_options), args.questions, args.combiner,
              json.load(open(args.fit)) if args.fit else None)
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
