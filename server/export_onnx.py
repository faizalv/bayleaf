"""Export intfloat/e5-base to ONNX format.

Requires build-time dependencies: torch, transformers, sentence-transformers.
These are NOT shipped in the runtime tarball.

Usage:
    python export_onnx.py [output_dir]

Produces:
    <output_dir>/model.onnx
    <output_dir>/tokenizer.json
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/e5-base"


def export(output_dir: Path):
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Wrapper so the ONNX graph has explicit named inputs/outputs
    # without the ModelOutput wrapper that BERT normally returns.
    class BertEmbedWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    wrapped = BertEmbedWrapper(model)

    dummy = tokenizer("warmup", return_tensors="pt", padding="max_length", max_length=16)

    print("Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(output_dir / "model.onnx"),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids":       {0: "batch", 1: "sequence"},
                "attention_mask":  {0: "batch", 1: "sequence"},
                "token_type_ids":  {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
        )

    # Save tokenizer.json then discard everything else save_pretrained writes.
    # Keep model.onnx, model.onnx.data (external weights, present when model > 2GB
    # protobuf limit), and tokenizer.json.
    tokenizer.save_pretrained(str(output_dir))
    for f in output_dir.iterdir():
        if f.is_file() and f.name != "tokenizer.json" and f.suffix != ".onnx" and not f.name.endswith(".onnx.data"):
            f.unlink()

    model_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.suffix == ".onnx" or f.name.endswith(".onnx.data"))
    print(f"Exported {MODEL_NAME} to {output_dir}")
    print(f"  model (total):  {model_size / 1e6:.1f} MB")
    print(f"  tokenizer.json: {(output_dir / 'tokenizer.json').stat().st_size / 1e3:.1f} KB")


def verify(output_dir: Path):
    """Compare ONNX pipeline output against sentence-transformers on reference texts."""
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

    all_pass = True
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

        cosine_sim = float(np.dot(onnx_vec, st_embeddings[i]) / (
            np.linalg.norm(onnx_vec) * np.linalg.norm(st_embeddings[i])
        ))
        status = "PASS" if cosine_sim > 0.999 else "FAIL"
        print(f"  [{status}] text {i}: cosine_sim={cosine_sim:.6f}  {text[:50]}")

        if cosine_sim <= 0.999:
            all_pass = False

    if not all_pass:
        print("\nFAIL: some vectors did not match")
        sys.exit(1)

    print("\nAll vectors match (cosine similarity > 0.999)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models")
    export(out)
    verify(out)
