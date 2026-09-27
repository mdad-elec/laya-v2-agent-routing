"""EmbedLLM (F3 baseline): the benchmark's TextMF, fitted on tune only, routed by g1's policy."""
import unittest

import numpy as np

from eval.baselines import embedllm


def item(i, q, scores, costs):
    return {"dataset": "d", "native_id": str(i), "prompt_hash": f"h{i}", "query": q, "scores": scores, "costs": costs}


class EmbedLLM(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.emb, self.train = {}, []
        for i in range(400):
            topic = i % 2
            q = f"t{topic} {i}"
            self.emb[q] = np.concatenate([[1.0, 0.0] if topic == 0 else [0.0, 1.0], rng.normal(0, 0.01, 1022)])
            self.train.append(item(i, q, {"big": 1.0, "small": float(topic)}, {"big": 1.0, "small": 0.01}))

    def test_it_learns_who_solves_what_and_routes_cheapest_above_tau(self):
        fit = embedllm.fit(self.train, self.emb, epochs=800)  # one optimiser step per epoch here, at their lr 1e-4
        p0 = fit.predict([self.emb["t0 0"]])[0]
        p1 = fit.predict([self.emb["t1 1"]])[0]
        self.assertGreater(p0["big"], 0.8)
        self.assertLess(p0["small"], 0.2)
        self.assertGreater(p1["small"], 0.8)
        expected_cost = embedllm.expected_costs(self.train)
        self.assertEqual(embedllm.pick(p1, expected_cost, tau=0.5), "small")
        self.assertEqual(embedllm.pick(p0, expected_cost, tau=0.5), "big")
        self.assertEqual(embedllm.pick({"big": 0.1, "small": 0.3}, expected_cost, tau=0.9), "small", "none reaches tau: argmax")

    def test_the_epoch_is_chosen_on_tune_only(self):
        epochs, curve = embedllm.choose_epochs(self.train, self.emb, candidates=(1, 30))
        self.assertIn(epochs, (1, 30))
        self.assertEqual(set(curve), {1, 30})


if __name__ == "__main__":
    unittest.main()
