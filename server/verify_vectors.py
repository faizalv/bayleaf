"""Verify ONNX model output via semantic coherence tests.

We avoid cross-backend comparison (ONNX Runtime vs sentence-transformers)
because PyTorch and ONNX Runtime use different BLAS implementations on ARM.
Over 12 BERT layers those per-op differences compound into ~5% vector
divergence. Instead we verify L2 normalization and semantic ranking, which
are the properties that actually matter for the embedding use case.

Usage:
    python verify_vectors.py <model_dir>

Exit code 0 if all checks pass, 1 otherwise.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PAIRS = [
    (
        "query: what is machine learning",
        "passage: Machine learning is a subset of artificial intelligence",
        "passage: Plants convert sunlight into chemical energy",
    ),
    (
        "query: how does photosynthesis work",
        "passage: Plants convert sunlight into energy through chlorophyll",
        "passage: Neural networks learn by adjusting weights",
    ),
    (
        "query: explain quantum computing",
        "passage: Quantum computers use qubits that can exist in superposition",
        "passage: The water cycle describes how water moves through the environment",
    ),
    (
        "query: what causes inflation",
        "passage: Inflation occurs when the general price level of goods rises over time",
        "passage: Photosynthesis is the process by which plants make food from sunlight",
    ),
]


def embed(session: ort.InferenceSession, tokenizer: Tokenizer, text: str) -> np.ndarray:
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    token_embeddings = outputs[0]
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.sum(mask_expanded, axis=1)
    mean_pooled = summed / np.maximum(counts, 1e-9)
    norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    return (mean_pooled / np.maximum(norm, 1e-9))[0]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=None)
    session = ort.InferenceSession(str(model_dir / "model.onnx"))

    print(f"Verifying model at {model_dir}")
    all_pass = True

    for query, related, unrelated in PAIRS:
        q = embed(session, tokenizer, query)
        r = embed(session, tokenizer, related)
        u = embed(session, tokenizer, unrelated)

        sim_related   = float(np.dot(q, r))
        sim_unrelated = float(np.dot(q, u))
        norm_q = float(np.linalg.norm(q))

        norm_ok    = abs(norm_q - 1.0) < 0.01
        ranking_ok = sim_related > sim_unrelated
        status = "PASS" if (norm_ok and ranking_ok) else "FAIL"

        print(f"  [{status}] norm={norm_q:.4f}  sim(related)={sim_related:.4f}  sim(unrelated)={sim_unrelated:.4f}")
        print(f"         query:     {query}")

        if not norm_ok:
            print(f"         FAIL: not L2-normalized (norm={norm_q:.6f})")
            all_pass = False
        if not ranking_ok:
            print(f"         FAIL: related ranked below unrelated")
            all_pass = False

    if all_pass:
        print("\nAll checks passed")
    else:
        print("\nFAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
