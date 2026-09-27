"""Avengers-Pro runs from LLMRouterBench's own code, offline: our vectors in its cache, a miss is loud."""
import unittest

import numpy as np

from eval.baselines import avengers_pro


def item(ds, i, q, scores, costs):
    return {"dataset": ds, "native_id": str(i), "query": q, "scores": scores, "costs": costs}


class AvengersPro(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train, self.emb = [], {}
        for i in range(60):
            topic = i % 2  # topic 0: only the expensive model solves; topic 1: both do
            q = f"t{topic} q{i}"
            self.emb[q] = (np.array([1.0, 0.0]) if topic == 0 else np.array([0.0, 1.0])) + rng.normal(0, 0.01, 2)
            self.train.append(item("d", i, q, {"big": 1.0, "small": 1.0 if topic else 0.0}, {"big": 1.0, "small": 0.01}))
        self.test = []
        for i in range(10):
            q = f"test t{i % 2} {i}"
            self.emb[q] = (np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0]))
            self.test.append(item("d", 100 + i, q, {"big": 1.0, "small": 0.0}, {"big": 1.0, "small": 0.01}))

    def test_cost_sensitive_routing_sends_the_easy_topic_to_the_cheap_model(self):
        choice = avengers_pro.route(self.train, self.test, self.emb, alpha=0.5, n_clusters=2)
        self.assertEqual({choice[("d", str(100 + i))] for i in range(0, 10, 2)}, {"big"})
        self.assertEqual({choice[("d", str(100 + i))] for i in range(1, 10, 2)}, {"small"})

    def test_a_query_without_our_embedding_fails_loudly(self):
        test = self.test + [item("d", 999, "never embedded", {"big": 1.0, "small": 0.0}, {"big": 1.0, "small": 0.0})]
        with self.assertRaises(RuntimeError):
            avengers_pro.route(self.train, test, self.emb, alpha=0.5, n_clusters=2)


if __name__ == "__main__":
    unittest.main()
