import os
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from markitdown import MarkItDown
from pydantic import BaseModel

MODEL_DIR = os.environ.get("BAYLEAF_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_NAME = "intfloat/e5-base"

_session = None
_tokenizer = None
_lock = threading.Lock()


def _load_model():
    global _session, _tokenizer
    if _session is not None:
        return

    with _lock:
        if _session is not None:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = os.path.join(MODEL_DIR, "model.onnx")
            tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
            _tokenizer = Tokenizer.from_file(tokenizer_path)
            _tokenizer.enable_truncation(max_length=512)
            _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=None)
            _session = ort.InferenceSession(model_path)
        except Exception as e:
            _session = None
            _tokenizer = None
            raise RuntimeError(f"model_load_failed: {e}") from e


def _embed(text: str) -> list[float]:
    _load_model()
    encoded = _tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = _session.run(None, {
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
    normalized = mean_pooled / np.maximum(norm, 1e-9)

    return normalized[0].tolist()


app = FastAPI()


class EmbedRequest(BaseModel):
    text: str


class ConvertRequest(BaseModel):
    path: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model():
    return {"model": MODEL_NAME}


@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        embedding = _embed(req.text)
        return {"embedding": embedding}
    except RuntimeError as e:
        if "model_load_failed" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert")
def convert(req: ConvertRequest):
    if not os.path.isfile(req.path):
        raise HTTPException(status_code=400, detail=f"file not found: {req.path}")
    try:
        md = MarkItDown()
        result = md.convert(req.path)
        return {"markdown": result.text_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
