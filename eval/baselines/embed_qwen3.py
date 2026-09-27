"""Embed TwinRouterBench prefixes with Qwen/Qwen3-Embedding-0.6B exactly as the SR-KNN adapter does
(unnormalised, the model's default max length), one JSON line per row id; resumable.

    python embed_qwen3.py texts.jsonl emb.jsonl

texts.jsonl holds {"id", "text"} with the adapter's "role:\\ncontent" flattening. Run where a GPU is;
the 0.12 memory fraction keeps it beside the Spark's resident 27B.
"""
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    torch.cuda.set_per_process_memory_fraction(0.12)
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    print("max_seq_length", enc.max_seq_length, "todo", len(rows) - len(done), flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            e = enc.encode([r["text"]], normalize_embeddings=False, convert_to_numpy=True)[0].astype(np.float32)
            f.write(json.dumps({"id": r["id"], "e": e.tolist()}) + "\n")
            f.flush()
            if i % 100 == 0:
                print(i, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
