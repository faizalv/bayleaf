"""Compare ONNX pipeline output against sentence-transformers on reference texts.

Usage:
    python verify_vectors.py <model_dir>

Requires both runtime and build dependencies installed.
Exit code 0 if all vectors match (cosine sim > 0.999), 1 otherwise.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

REFERENCE_TEXTS = [
    "query: what is machine learning",
    "passage: Machine learning is a subset of artificial intelligence",
    "query: how does photosynthesis work",
    "passage: Plants convert sunlight into chemical energy through chlorophyll",
    "query: explain quantum computing",
    "passage: Quantum computers use qubits that can exist in superposition",
    "query: what is the meaning of life",
    "passage: The meaning of life is a philosophical question about human existence",
    "query: how do neural networks learn",
    "passage: Neural networks learn by adjusting weights through backpropagation",
]

MODEL_NAME = "intfloat/e5-base"


def onnx_embed(tokenizer: Tokenizer, session: ort.InferenceSession, text: str) -> np.ndarray:
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

    print(f"Loading sentence-transformers model: {MODEL_NAME}")
    st_model = SentenceTransformer(MODEL_NAME)
    st_embeddings = st_model.encode(REFERENCE_TEXTS, normalize_embeddings=True)

    print(f"Loading ONNX model from: {model_dir}")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=None)
    session = ort.InferenceSession(str(model_dir / "model.onnx"))

    all_pass = True
    for i, text in enumerate(REFERENCE_TEXTS):
        onnx_vec = onnx_embed(tokenizer, session, text)
        cosine_sim = float(np.dot(onnx_vec, st_embeddings[i]) / (
            np.linalg.norm(onnx_vec) * np.linalg.norm(st_embeddings[i])
        ))
        status = "PASS" if cosine_sim > 0.999 else "FAIL"
        print(f"  [{status}] text {i}: cosine_sim={cosine_sim:.6f}  {text[:50]}")

        if cosine_sim <= 0.999:
            all_pass = False

    if all_pass:
        print("\nAll vectors match (cosine similarity > 0.999)")
    else:
        print("\nSome vectors did NOT match")
        sys.exit(1)


if __name__ == "__main__":
    main()
