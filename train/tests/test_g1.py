"""Laya-g1's model: θ(digest)·a(cell) − b(cell); measured-block dropout; the loss sees only listed cells."""
import unittest

import torch

from train.g1 import G1, MEASURED_WIDTH, cell_vectors, drop_measured, loss_on


class StubEncoder(torch.nn.Module):
    """Stands in for ModernBERT: token embeddings as the last hidden state."""

    def __init__(self, vocab=50, hidden=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.config = type("C", (), {"hidden_size": hidden})()

    def forward(self, input_ids, attention_mask):
        return type("O", (), {"last_hidden_state": self.emb(input_ids)})()


class Model(unittest.TestCase):
    def test_logits_are_one_per_text_and_cell(self):
        net = G1(StubEncoder(), desc_width=10, d=8)
        ids = torch.randint(0, 50, (3, 7))
        logits = net(ids, torch.ones_like(ids), torch.randn(5, 10))
        self.assertEqual(tuple(logits.shape), (3, 5))

    def test_padding_does_not_change_the_task_vector(self):
        torch.manual_seed(0)
        net = G1(StubEncoder(), desc_width=10, d=8).eval()
        ids = torch.tensor([[5, 6, 7, 0, 0]])
        a = net.task(ids, torch.tensor([[1, 1, 1, 0, 0]]))
        b = net.task(ids[:, :3], torch.tensor([[1, 1, 1]]))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_measured_dropout_zeroes_value_and_mask_together(self):
        torch.manual_seed(0)
        v = torch.ones(1000, 5 + MEASURED_WIDTH)
        out = drop_measured(v, public_width=5, p=0.5)
        dropped = (out[:, 5:] == 0).all(dim=1)
        self.assertTrue(0.4 < dropped.float().mean() < 0.6)
        self.assertTrue((out[:, :5] == 1).all(), "the public block is never dropped")

    def test_the_loss_reads_only_the_cells_listed_for_each_text(self):
        logits = torch.tensor([[10.0, -10.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
        pairs = [(0, 0, 1.0, 1.0), (0, 1, 0.0, 1.0)]  # text 0: cell 0 solved, cell 1 not; text 1 unlisted
        loss = loss_on(logits, pairs)
        self.assertLess(loss.item(), 1e-3)
        loss.backward()
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.0)

    def test_cell_vectors_append_the_measured_block_with_masks(self):
        public = {"m": [0.5] * 3}
        measured = {"m": {"math": 0.9}}
        v = cell_vectors(["m", "unmeasured"], public={"m": public["m"], "unmeasured": [0.0] * 3}, measured=measured)
        self.assertEqual(v.shape[1], 3 + MEASURED_WIDTH)
        self.assertEqual(v[1, 3:].abs().sum().item(), 0.0)
        self.assertAlmostEqual(v[0, 3].item(), 0.9, places=6)
        self.assertEqual(v[0, 3 + MEASURED_WIDTH // 2].item(), 1.0)



class Predict(unittest.TestCase):
    def test_probabilities_are_one_row_per_text_one_column_per_cell(self):
        from train.predict import probabilities

        class Tok:
            def __call__(self, texts, **kw):
                ids = torch.tensor([[1 + (hash(t) % 40), 2, 3] for t in texts])
                return type("E", (dict,), {"to": lambda self, d: self})({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        net = G1(StubEncoder(), desc_width=4, d=8)
        p = probabilities(net, Tok(), ["a", "b", "c"], torch.randn(2, 4), torch.device("cpu"), batch=2)
        self.assertEqual((len(p), len(p[0])), (3, 2))
        self.assertTrue(all(0.0 <= x <= 1.0 for row in p for x in row))


if __name__ == "__main__":
    unittest.main()
