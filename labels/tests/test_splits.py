"""The two split rules beyond items: trajectories (TwinRouterBench) and held-out models (the probe)."""
import unittest

from labels.splits import model_split, probe_trainable


class Splits(unittest.TestCase):
    def test_model_split_is_stable_and_about_half(self):
        names = [f"model-{i}" for i in range(400)]
        held = [n for n in names if model_split(n) == "held"]
        self.assertTrue(150 < len(held) < 250)
        self.assertEqual([model_split(n) for n in names], [model_split(n) for n in names])

    def test_the_probe_never_trains_on_a_held_model(self):
        names = [f"model-{i}" for i in range(50)]
        for n in names:
            self.assertEqual(probe_trainable(n), model_split(n) == "tune")


if __name__ == "__main__":
    unittest.main()
