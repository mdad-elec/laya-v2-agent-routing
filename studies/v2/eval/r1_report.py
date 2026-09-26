"""Rung 1's table: checkpoint x state shape x wording, scored on the TUNE split, and the one
configuration it names for the held-out read.

    python3 scripts/ab/r1_report.py scripts/ab/results/laya-v2 > scripts/ab/results/laya-v2/r1-report.md

Each configuration is POOLED over the headline rows of both corpora (the borrowed tier rows of
tier-240 and the pilot asks); the unaudited DSS rows are scored in their own column and never enter
the pooled number or the selection (ATTRIBUTION.md, PREREG.md).
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain_bench  # noqa: E402
import stats  # noqa: E402


def _scored(recs):
    pred, gold = [r["predicted"] for r in recs], [r["gold"] for r in recs]
    conf = [r["confidence"] for r in recs]
    correct = [p == g for p, g in zip(pred, gold)]
    over, under = brain_bench.spend_rates(pred, gold)
    toks = [r["input_tokens"] for r in recs if r.get("input_tokens")]
    return {"n": len(recs), "accuracy": sum(correct) / len(recs), "macro_f1": brain_bench.macro_f1(pred, gold),
            "over": over, "under": under, "mean_p": statistics.fmean(conf),
            "signed_gap": statistics.fmean(conf) - sum(correct) / len(recs),
            "share_below_0.45": sum(c < 0.45 for c in conf) / len(conf),
            "share_below_0.35": sum(c < 0.35 for c in conf) / len(conf),
            "input_tokens": statistics.fmean(toks) if toks else None,
            "p50_ms": brain_bench.percentile([r["round_trip_ms"] for r in recs], 0.5)}


def rows(results_dir, split="tune"):
    """One row per (checkpoint, shape, wording[/tag]) with every corpus of that configuration pooled."""
    groups, splits = {}, {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"r1-*-{split}*.json"))):
        body = json.load(open(path))
        if body.get("split") != split:
            continue
        # Configurations are only comparable on the SAME rows: a result cut from another version of
        # the split file is refused by name, never silently pooled beside the others.
        splits.setdefault(body.get("corpus"), {}).setdefault(body.get("split_sha"), []).append(os.path.basename(path))
        tag = "-asshipped" if path.endswith("-asshipped.json") else ""
        for key, recs in body["records"].items():
            block, wording = key.split("/", 1)
            g = groups.setdefault((body["brain"]["model"], body["state_shape"], wording + tag), {"head": [], "dss": []})
            g["dss" if block == "dss" else "head"] += recs
    for corpus, by_sha in splits.items():
        if len(by_sha) > 1:
            raise ValueError(f"{corpus} results were cut from {len(by_sha)} different splits: " +
                             "; ".join(f"{sha}: {', '.join(files)}" for sha, files in by_sha.items()))
    out = []
    for (ckpt, shape, wording), g in sorted(groups.items()):
        row = {"ckpt": ckpt, "shape": shape, "wording": wording, **_scored(g["head"])}
        row["dss_n"] = len(g["dss"])
        row["dss_accuracy"] = _scored(g["dss"])["accuracy"] if g["dss"] else None
        row["correct"] = {r["id"]: r["predicted"] == r["gold"] for r in g["head"]}
        out.append(row)
    return out


def select(table):
    """The best pooled tune accuracy; ties go to lower underspend (PREREG, the adoption rule)."""
    return max((r for r in table if not r["wording"].endswith("-asshipped")),
               key=lambda r: (r["accuracy"], -r["under"]))


def markdown(table):
    q = lambda v, f="%.3f": "—" if v is None else f % v
    lines = ["| checkpoint | state | wording / order | n | acc | macro-F1 | under | over | mean p | p<0.45 | p<0.35 | signed gap | in tok | p50 ms | DSS acc (n) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(table, key=lambda r: -r["accuracy"]):
        lines.append(f"| {r['ckpt']} | {r['shape']} | {r['wording']} | {r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                     f"{r['under']:.3f} | {r['over']:.3f} | {r['mean_p']:.3f} | {r['share_below_0.45']:.2f} | {r['share_below_0.35']:.2f} | "
                     f"{r['signed_gap']:+.3f} | {q(r['input_tokens'], '%.0f')} | {r['p50_ms']:.0f} | {q(r['dss_accuracy'])} ({r['dss_n']}) |")
    return "\n".join(lines)


def main(argv):
    results_dir = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "laya-v2")
    table = rows(results_dir)
    if not table:
        print("no r1 tune results in", results_dir)
        return 1
    pick = select(table)
    print("# Rung 1 — checkpoint × state shape × wording (tune split)\n")
    print(f"Pooled over the headline tune rows of tier-240 and the pilot; DSS rows apart. Selection optimism for "
          f"{len(table)} configurations on {pick['n']} rows: ±{stats.selection_optimism(len(table), pick['n']):.3f}.\n")
    print(markdown(table))
    print(f"\n**Selected for the one held-out read:** `{pick['ckpt']} / {pick['shape']} / {pick['wording']}` "
          f"(tune accuracy {pick['accuracy']:.3f}, underspend {pick['under']:.3f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
