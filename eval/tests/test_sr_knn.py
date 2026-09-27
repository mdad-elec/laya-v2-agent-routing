"""SR-KNN, as TwinRouterBench runs it: Qwen3-Embedding features + a constant 'other' one-hot, cosine kNN."""
import unittest

import numpy as np

from eval.baselines.sr_knn import features, predict


class SRKNN(unittest.TestCase):
    def test_features_append_the_other_one_hot(self):
        f = features(np.ones((2, 1024), dtype=np.float32))
        self.assertEqual(f.shape, (2, 1038))
        self.assertEqual(f[0, 1024 + 10], 1.0)
        self.assertEqual(f[0, 1024:].sum(), 1.0)

    def test_one_nn_takes_the_nearest_label_and_leave_one_out_skips_self(self):
        X = features(np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32).repeat(512, axis=1))
        y = np.array([0, 1, 3])
        self.assertEqual(predict(X, y, X, k=1).tolist(), [0, 1, 3])
        self.assertEqual(predict(X, y, X, k=1, leave_one_out=True).tolist(), [1, 0, 1])

    def test_a_vote_tie_goes_to_the_nearer_neighbour(self):
        X = features(np.array([[1, 0], [0.8, 0.2], [0.7, 0.3], [0, 1]], dtype=np.float32).repeat(512, axis=1))
        q = features(np.array([[1, 0]], dtype=np.float32).repeat(512, axis=1))
        self.assertEqual(predict(X[:2], np.array([2, 0]), q, k=2).tolist(), [2])


if __name__ == "__main__":
    unittest.main()
