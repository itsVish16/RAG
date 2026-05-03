from fastembed import SparseTextEmbedding, SparseEmbedding
from app.core.config import settings

_sparse_model: SparseTextEmbedding | None = None


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=settings.sparse_model_name)
    return _sparse_model


def embed_sparse(texts: list[str]) -> list[SparseEmbedding]:
    model = get_sparse_model()
    return list(model.embed(texts))


def embed_sparse_query(text: str) -> SparseEmbedding:
    model = get_sparse_model()
    return list(model.query_embed(text))[0]
