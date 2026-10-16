# CoDy — A Computational Framework for Representing
# the Temporal Organization of Conversation
#
# Copyright (C) 2026 Adolfo J. Céspedes
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License,
# version 3, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Affero General Public License for more details.


# ============================
# app.py - CoDy continuous multivariate analysis application
# ============================

from typing import List, Tuple, Optional

import numpy as np
import streamlit as st

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    TfidfVectorizer = None
    sk_normalize = None

def _load_native_transformer(model_name: str):
    """Load a Hugging Face encoder without importing sentence-transformers."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "The Transformers backend could not be imported. "
            f"{type(e).__name__}: {e}"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def _mean_pool(last_hidden_state, attention_mask):
    """Attention-mask-aware mean pooling used by all-MiniLM-L6-v2."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def _encode_native_sbert(
    texts: Tuple[str, ...],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode text with MiniLM/SBERT without importing sentence-transformers.

    all-MiniLM-L6-v2 uses a Transformer encoder followed by mean pooling and
    L2 normalization. This reproduces the canonical SentenceTransformer
    inference recipe without requiring the sentence-transformers package.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError(
            f"PyTorch could not be imported: {type(e).__name__}: {e}"
        ) from e

    tokenizer, model = _load_native_transformer(model_name)
    texts_list = [t if isinstance(t, str) else "" for t in texts]
    batches = []

    for start in range(0, len(texts_list), int(max(1, batch_size))):
        batch = texts_list[start:start + int(max(1, batch_size))]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)

        batches.append(pooled.detach().cpu().numpy())

    if not batches:
        # all-MiniLM-L6-v2 has 384 dimensions; preserve a valid empty matrix.
        hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 384))
        return np.zeros((0, hidden_size), dtype=float)

    return np.vstack(batches).astype(float, copy=False)

def embed_texts(
    texts: List[str],
    mode: str = "auto",
    sbert_model: Optional[str] = None,
) -> Tuple[np.ndarray, str, str]:
    """Return ``(E, used_mode, status_msg)``.

    ``auto`` and ``sbert`` use the canonical MiniLM/SBERT representation through
    the native Hugging Face Transformers inference path. TF-IDF is retained as
    an explicit lexical baseline and is never used as a silent fallback.

    Cross-embedding robustness tests are intentionally kept outside the public
    application.
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    mode = (mode or "auto").lower().strip()

    if mode == "auto":
        mode = "sbert"

    if mode == "sbert":
        model_name = (
            sbert_model.strip()
            if sbert_model and sbert_model.strip()
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            E = _encode_native_sbert(tuple(texts), model_name)
        except Exception as e:
            raise RuntimeError(
                f"SBERT model '{model_name}' failed to load or encode through the "
                f"native Transformers backend: {type(e).__name__}: {e}"
            ) from e

        return (
            np.asarray(E, dtype=float),
            f"sbert:{model_name}",
            f"✅ Using SBERT embeddings: {model_name}",
        )

    if mode != "tfidf":
        raise ValueError(
            f"Unknown embeddings mode '{mode}'. Expected one of: auto, sbert, tfidf."
        )

    if TfidfVectorizer is None:
        raise RuntimeError(
            "TF-IDF was explicitly requested, but scikit-learn is unavailable."
        )

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).astype(float)

    if sk_normalize is not None:
        X = sk_normalize(X, norm="l2", axis=1, copy=False)

    return X.toarray(), "tfidf", "ℹ️ Using TF-IDF embeddings (explicit baseline mode)."
