import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from labels.llmrouterbench import EXCLUDED_DATASETS, norm_prompt, prompt_hash, read_archive


def archive(files: dict[str, dict]) -> Path:
    path = Path(tempfile.mkdtemp()) / "bench.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in files.items():
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def result(model, dataset, scores):
    return {"model_name": model, "dataset_name": dataset, "split": "test", "counts": len(scores),
            "records": [{"index": str(i), "origin_query": f"Q{i} about {dataset}", "prompt": f"Q{i}", "score": str(s), "cost": "0.01",
                         "prompt_tokens": "10", "completion_tokens": "5"} for i, s in enumerate(scores)]}


class Reader(unittest.TestCase):
    def test_records_become_outcome_rows_and_text_stays_out_of_the_row(self):
        path = archive({"bench-release/mmlupro/m1/r.json": result("m1", "mmlupro", [1, 0, 1]),
                        "bench-release/mmlupro/m2/r.json": result("m2", "mmlupro", [0, 0, 1])})
        rows, prompts, report = read_archive(path)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted({r["model"] for r in rows}), ["m1", "m2"])
        r = next(r for r in rows if r["model"] == "m1" and r["native_id"] == "0")
        self.assertEqual((r["dataset"], r["score"], r["cost"]), ("mmlupro", 1.0, 0.01))
        self.assertNotIn("Q0", json.dumps(rows), "no prompt text in an outcome row")
        self.assertEqual(prompts[r["prompt_hash"]], "Q0 about mmlupro")

    def test_excluded_datasets_are_skipped_and_reported(self):
        self.assertIn("gpqa", EXCLUDED_DATASETS)
        path = archive({"bench-release/gpqa/m1/r.json": result("m1", "gpqa", [1])})
        rows, _, report = read_archive(path)
        self.assertEqual(rows, [])
        self.assertIn("gpqa", report["excluded"])

    def test_the_prompt_hash_survives_whitespace_and_case(self):
        self.assertEqual(prompt_hash("  Write ME a Zig\nprogram "), prompt_hash("write me a zig program"))
        self.assertEqual(norm_prompt("A  b\tC"), "a b c")

    def test_a_record_whose_score_cannot_be_read_is_reported_not_zeroed(self):
        bad = result("m1", "mmlupro", [1])
        bad["records"].append({"index": "9", "origin_query": "x", "score": "n/a", "cost": "0"})
        rows, _, report = read_archive(archive({"bench-release/mmlupro/m1/r.json": bad}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["unreadable_scores"], 1)


if __name__ == "__main__":
    unittest.main()
