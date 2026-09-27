"""The leakage rule: a label item whose prompt matches a HELD item of this repo's suites never trains."""
import unittest

from labels import twinrouterbench as trb
from labels.leakage import build_eval_index, held_hashes, is_trainable
from labels.llmrouterbench import prompt_hash
from suites.base import Item, split_of


class Leakage(unittest.TestCase):
    def test_a_label_matching_a_held_suite_item_is_not_trainable_and_a_tune_match_is(self):
        items = [Item(suite="s", native_id=str(i), messages=[], gold=None, meta={"source_text": f"Question number {i}?"}) for i in range(40)]
        held = [i for i in items if split_of(i.item_id) == "held"]
        tune = [i for i in items if split_of(i.item_id) == "tune"]
        hashes = held_hashes(items)
        self.assertEqual(len(hashes), len(held))
        # a public label row with the SAME raw text as a held suite item, whitespace and case aside
        leaked = prompt_hash("  QUESTION number " + held[0].native_id + "? ")
        self.assertFalse(is_trainable(leaked, hashes))
        self.assertTrue(is_trainable(prompt_hash(tune[0].meta["source_text"]), hashes), "a tune-split suite item may also be a label")

    def test_items_without_source_text_contribute_nothing(self):
        self.assertEqual(held_hashes([Item(suite="s", native_id="1", messages=[], gold=None)]), set())



def trb_row(instance: str, task: str) -> dict:
    return {"benchmark": "swebench", "instance_id": instance, "messages": [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": task}]}


class EvalIndexLeakage(unittest.TestCase):
    def test_every_eval_source_blocks_a_training_prompt_that_contains_it(self):
        rows = [trb_row(f"repo__issue-{i}", f"Fix the failing parser for nested brackets in module number {i} of the tokenizer package today") for i in range(40)]
        held = [r for r in rows if trb.split(r) == "held"]
        tune = [r for r in rows if trb.split(r) == "tune"]
        items = [Item(suite="s", native_id=str(i), messages=[], gold=None, meta={"source_text": f"Name every prime number between {i} and one thousand and explain the sieve used"}) for i in range(40)]
        held_item = next(i for i in items if split_of(i.item_id) == "held")
        index = build_eval_index(routerarena={"routerarena/full/X_1": "A farmer has seventeen sheep and all but nine run away, how many are left"},
                                 twinrouterbench_rows=rows, suite_items=items)
        wrap = "Please answer. Question: {} Answer:".format
        self.assertTrue(index.hits(wrap("a farmer has seventeen sheep and all but nine run away, how many are left?")))
        self.assertTrue(index.hits(wrap(held[0]["messages"][1]["content"])))
        self.assertFalse(index.hits(wrap(tune[0]["messages"][1]["content"])), "a tune trajectory may train")
        self.assertTrue(index.hits(wrap(held_item.meta["source_text"])))


if __name__ == "__main__":
    unittest.main()
