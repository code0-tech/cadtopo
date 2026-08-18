import numpy as np
import pytest

from cadtopo.embedding import EmbeddingModel


class _FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer.

    Deterministic: encodes a string to a fixed-size vector derived from its
    length, so tests can assert on behaviour without a real model.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
        return np.array([float(len(text)), float(text.count("a")), 1.0])

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture
def embedder(monkeypatch) -> EmbeddingModel:
    monkeypatch.setattr("cadtopo.embedding.SentenceTransformer", _FakeSentenceTransformer)
    return EmbeddingModel()


class TestEncoding:
    def test_get_embedding_encodes_text(self, embedder):
        vec = embedder.get_embedding("aardvark")
        assert vec.tolist() == [8.0, 3.0, 1.0]

    def test_empty_text_returns_zero_vector_of_model_dimension(self, embedder):
        vec = embedder.get_embedding("   ")
        assert vec.tolist() == [0.0, 0.0, 0.0]

    def test_query_instruction_prefix_applied_to_query_side_only(self, monkeypatch):
        monkeypatch.setattr("cadtopo.embedding.SentenceTransformer", _FakeSentenceTransformer)
        embedder = EmbeddingModel(query_instruction="Q: ")

        passage_vec = embedder.get_embedding("hi")  # not prefixed
        query_vec = embedder.get_query_embedding("hi")  # prefixed with "Q: "

        assert passage_vec.tolist() == [2.0, 0.0, 1.0]
        assert query_vec.tolist() == [5.0, 0.0, 1.0]  # len("Q: " + "hi") == 5

    def test_empty_query_text_skips_instruction_prefix(self, embedder):
        vec = embedder.get_query_embedding("")
        assert vec.tolist() == [0.0, 0.0, 0.0]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert EmbeddingModel.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert EmbeddingModel.cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert EmbeddingModel.cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_without_raising(self):
        assert EmbeddingModel.cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0
        assert EmbeddingModel.cosine_similarity(np.array([1.0, 2.0, 3.0]), np.zeros(3)) == 0.0

    def test_scale_invariant(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([2.0, 4.0, 6.0])
        assert EmbeddingModel.cosine_similarity(v1, v2) == pytest.approx(1.0)
