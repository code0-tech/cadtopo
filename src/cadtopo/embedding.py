"""Local sentence-embedding backend for the router's semantic matching."""

import numpy as np
from sentence_transformers import SentenceTransformer

from .logging_utils import get_logger

_log = get_logger("cadtopo.embedding")


class EmbeddingModel:
    """Wraps a local SentenceTransformer for asymmetric query/passage matching.

    :param model_name: A HuggingFace model name for sentence-transformers.
        Defaults to a model trained on 215M Q&A pairs for asymmetric
        query/passage matching (short round goal vs. skill definition) — an
        empirical sweep found this gives the best joint recall/false-activation
        tradeoff on real CADTopo goals over BGE/E5-style instruction models.
    :param query_instruction: Optional prefix prepended to query-side text
        (round_goal, dynamic "query"/demand). Passage-side text
        (skill_definition, dynamic "key"/offer) is never prefixed. Not needed
        for the default model; exists so a future BGE/E5-style swap can
        supply one without touching call sites.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        query_instruction: str = "",
    ):
        _log.debug(f"Loading SentenceTransformer model {model_name!r}")
        self.model = SentenceTransformer(model_name)
        self._query_instruction = query_instruction

    def _encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, convert_to_numpy=True)

    def get_embedding(self, text: str) -> np.ndarray:
        """Passage-side embedding: the static ``skill_definition`` (S_i) and
        the dynamic "key"/offer text — the documents being searched over."""
        return self._encode(text)

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Query-side embedding: the round goal and the dynamic "query"/demand
        text — what's searching for a matching passage."""
        if not text.strip():
            return self._encode(text)
        return self._encode(self._query_instruction + text)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors; ``0.0`` if either is zero."""
        norm_v1 = float(np.linalg.norm(v1))
        norm_v2 = float(np.linalg.norm(v2))
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2)) / (norm_v1 * norm_v2)
