import unittest

from eval.ceiling_f3_mf import stratified_split


def items(ds, n):
    return [{"dataset": ds, "prompt_hash": f"{ds}-{i}"} for i in range(n)]


class StratifiedSplitTest(unittest.TestCase):
    def test_each_dataset_gets_max_of_15_percent_and_min_of_half_or_20(self):
        train, test = stratified_split(items("aime", 16) + items("livemathbench", 57) + items("simpleqa", 2056))
        count = lambda rows, ds: sum(r["dataset"] == ds for r in rows)  # noqa: E731
        self.assertEqual(count(test, "aime"), 8)
        self.assertEqual(count(test, "livemathbench"), 20)
        self.assertEqual(count(test, "simpleqa"), 309)
        self.assertEqual(len(train) + len(test), 16 + 57 + 2056)

    def test_the_split_is_a_partition_and_stable(self):
        rows = items("tau2", 118)
        a, b = stratified_split(rows)
        self.assertFalse({r["prompt_hash"] for r in a} & {r["prompt_hash"] for r in b})
        self.assertEqual([r["prompt_hash"] for r in b], [r["prompt_hash"] for r in stratified_split(rows)[1]])


if __name__ == "__main__":
    unittest.main()
