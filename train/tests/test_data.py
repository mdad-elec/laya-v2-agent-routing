"""g1's training examples: tune only, never contaminated, openrouter never a cell, tiers become outcomes."""
import unittest

from labels.contamination import EvalIndex
from train import data


def lrb(ds, nid, h, model, score, cost=0.01):
    return {"source": "llmrouterbench@x", "dataset": ds, "native_id": nid, "prompt_hash": h, "model": model, "score": score, "cost": cost}


class Examples(unittest.TestCase):
    def test_single_turn_rows_keep_tune_drop_held_contaminated_and_the_reference_router(self):
        from suites.base import split_of

        hashes = [f"h{i}" for i in range(40)]
        tune = [h for h in hashes if split_of(h) == "tune"]
        held = [h for h in hashes if split_of(h) == "held"]
        prompts = {h: f"question text number {h} about widgets and gears and their many ratios" for h in hashes}
        prompts[tune[1]] = "A farmer has seventeen sheep and all but nine run away, how many are left over"
        index = EvalIndex({"routerarena/full/x": "A farmer has seventeen sheep and all but nine run away, how many are left"})
        rows = [lrb("d", h, h, "m1", 1.0) for h in hashes] + [lrb("d", tune[0], tune[0], "openrouter", 1.0)]
        out = data.single_turn_examples(rows, prompts, index)
        keys = {e["prompt_hash"] for e in out}
        self.assertIn(tune[0], keys)
        self.assertFalse(keys & set(held), "a held prompt trained")
        self.assertNotIn(tune[1], keys, "a contaminated prompt trained")
        self.assertNotIn("openrouter", {e["model"] for e in out})

    def test_a_tier_label_becomes_one_outcome_per_pool_model(self):
        row = {"id": "r", "benchmark": "swebench", "instance_id": "i", "messages": [{"role": "user", "content": "fix it"}], "target_tier_id": 2}
        out = data.tier_outcomes(row, {"low": "a", "mid": "b", "mid_high": "c", "high": "d"})
        self.assertEqual({e["model"]: e["target"] for e in out}, {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0})

    def test_the_probe_drops_held_models_and_the_scored_set_keeps_them(self):
        from labels.splits import model_split

        ex = [{"model": f"model-{i}", "target": 1.0} for i in range(30)]
        probe = data.for_checkpoint(ex, probe=True)
        self.assertTrue(all(model_split(e["model"]) == "tune" for e in probe))
        self.assertEqual(len(data.for_checkpoint(ex, probe=False)), 30)

    def test_the_measured_block_reads_tune_rows_only(self):
        rows = [lrb("mmlupro", "1", "a", "m", 1.0), lrb("mmlupro", "2", "b", "m", 0.0)]
        block = data.measured_blocks(rows, is_tune=lambda r: r["prompt_hash"] == "a")
        self.assertEqual(block["m"]["knowledge"], 1.0)


if __name__ == "__main__":
    unittest.main()
