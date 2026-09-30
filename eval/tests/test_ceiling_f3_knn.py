import unittest

import numpy as np

from eval import llmrouterbench_f3 as f3
from eval.ceiling_f3_knn import KnnTable


def item(h, ds, best):
    return {"prompt_hash": h, "dataset": ds, "native_id": h,
            "scores": {m: 1.0 if m == best else 0.0 for m in f3.MODELS}, "costs": dict.fromkeys(f3.MODELS, 1.0)}


class KnnTableTest(unittest.TestCase):
    def setUp(self):
        a, b = f3.MODELS[0], f3.MODELS[1]
        self.a, self.b = a, b
        self.items = [item("x1", "aime", a), item("x2", "aime", b), item("y1", "hle", b)]
        self.vec = {"x1": np.array([1.0, 0.0]), "x2": np.array([0.0, 1.0]), "y1": np.array([1.0, 0.0]),
                    "q": np.array([0.9, 0.1])}

    def test_one_neighbour_and_little_shrinkage_follows_the_nearest_item_of_the_same_task(self):
        q = item("q", "aime", self.a)
        (p, _), = KnnTable(k=1, m=1e-9, by_task=True).fit(self.items, self.vec).predict([q], self.vec)
        self.assertAlmostEqual(p[self.a], 1.0)
        self.assertAlmostEqual(p[self.b], 0.0)

    def test_heavy_shrinkage_returns_the_task_row(self):
        q = item("q", "aime", self.a)
        (p, _), = KnnTable(k=1, m=1e9, by_task=True).fit(self.items, self.vec).predict([q], self.vec)
        self.assertAlmostEqual(p[self.a], 0.5, places=6)
        self.assertAlmostEqual(p[self.b], 0.5, places=6)

    def test_neighbours_never_cross_tasks_when_the_task_is_known(self):
        q = item("q", "hle", self.b)
        (p, _), = KnnTable(k=1, m=1e-9, by_task=True).fit(self.items, self.vec).predict([q], self.vec)
        self.assertAlmostEqual(p[self.b], 1.0)


if __name__ == "__main__":
    unittest.main()
