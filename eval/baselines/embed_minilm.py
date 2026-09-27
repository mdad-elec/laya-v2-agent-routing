"""The Amendment 4 sensitivity embedder: sentence-transformers/all-MiniLM-L6-v2 on CPU, one JSON line per id.

    python -m eval.baselines.embed_minilm texts.jsonl emb.jsonl
"""
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vecs = enc.encode([r["text"] for r in rows], batch_size=64, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
    with open(out, "w", encoding="utf-8") as f:
        for r, e in zip(rows, vecs, strict=True):
            f.write(json.dumps({"id": r["id"], "e": np.asarray(e, dtype=np.float32).tolist()}) + "\n")
    print("DONE", len(rows), vecs.shape[1])


if __name__ == "__main__":
    main()
