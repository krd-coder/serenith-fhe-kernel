"""Nomic document embedding and Matryoshka prefix extraction."""

import re
import unicodedata
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .quantization import validate_dimension

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


class DocumentEmbedder:
    """Load Nomic lazily; an injected model can be used for offline tests."""

    def __init__(self, model: Any = None) -> None:
        self._model = model

    def embed(self, text: str, dimension: int) -> NDArray[np.float64]:
        """Encode text, layer-normalize the full embedding, then select an MRL prefix."""
        validate_dimension(dimension)
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        text = unicodedata.normalize("NFC", text)
        text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\r\t")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            raise ValueError("text must not be empty")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        result = np.asarray(self._model.encode(f"search_document: {text}", convert_to_numpy=True))
        if result.ndim != 1 or result.size < dimension or result.dtype.kind not in "iuf":
            raise ValueError("model returned an invalid embedding")
        vector = result.astype(np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("model returned non-finite values")
        # Nomic layer normalization precedes MRL truncation.
        centered = vector - vector.mean()
        normalized = centered / np.sqrt(centered.var() + 1e-5)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("model embedding cannot be normalized")
        return normalized[:dimension]
