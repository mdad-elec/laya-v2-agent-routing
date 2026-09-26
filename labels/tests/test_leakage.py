"""The leakage rule: a label item whose prompt matches a HELD item of this repo's suites never trains."""
import unittest

from labels.leakage import held_hashes, is_trainable
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


if __name__ == "__main__":
    unittest.main()
