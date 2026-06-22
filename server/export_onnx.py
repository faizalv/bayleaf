"""Export intfloat/e5-base to ONNX format.

Requires build-time dependencies: torch, optimum[exporters], sentence-transformers.
These are NOT shipped in the runtime tarball.

Usage:
    python export_onnx.py [output_dir]

Produces:
    <output_dir>/model.onnx
    <output_dir>/tokenizer.json
"""

import sys
from pathlib import Path

MODEL_NAME = "intfloat/e5-base"


def export(output_dir: Path):
    from optimum.exporters.onnx import main_export

    output_dir.mkdir(parents=True, exist_ok=True)
    main_export(MODEL_NAME, output=output_dir, task="feature-extraction")

    keep = {"model.onnx", "tokenizer.json"}
    for f in output_dir.iterdir():
        if f.name not in keep:
            f.unlink()

    print(f"Exported {MODEL_NAME} to {output_dir}")
    print(f"  model.onnx:     {(output_dir / 'model.onnx').stat().st_size / 1e6:.1f} MB")
    print(f"  tokenizer.json: {(output_dir / 'tokenizer.json').stat().st_size / 1e3:.1f} KB")


def verify(output_dir: Path):
    """Compare ONNX output against sentence-transformers on reference texts."""
    import numpy as np
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer

    reference_texts = [
        "query: what is machine learning",
        "passage: Machine learning is a subset of artificial intelligence",
        "query: how does photosynthesis work",
        "passage: Plants convert sunlight into chemical energy through chlorophyll",
        "query: explain quantum computing",
    ]

    print("\nVerifying ONNX output against sentence-transformers...")
    st_model = SentenceTransformer(MODEL_NAME)
    st_embeddings = st_model.encode(reference_texts, normalize_embeddings=True)

    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=None)
    session = ort.InferenceSession(str(output_dir / "model.onnx"))

    for i, text in enumerate(reference_texts):
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
        onnx_vec = (mean_pooled / np.maximum(norm, 1e-9))[0]

        cosine_sim = np.dot(onnx_vec, st_embeddings[i]) / (
            np.linalg.norm(onnx_vec) * np.linalg.norm(st_embeddings[i])
        )
        status = "PASS" if cosine_sim > 0.999 else "FAIL"
        print(f"  [{status}] text {i}: cosine_sim={cosine_sim:.6f}")

        if cosine_sim <= 0.999:
            print(f"    WARNING: cosine similarity below threshold for: {text[:50]}...")
            sys.exit(1)

    print("\nAll vectors match (cosine similarity > 0.999)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models")
    export(out)
    verify(out)
