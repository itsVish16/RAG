from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)
from app.core.config import settings
from app.services.qdrant import get_qdrant_client
from app.services.mistral import get_embeddings
from app.services.sparse import embed_sparse_query


def hybrid_search(
    query: str,
    collection_id: str,
    top_k: int | None = None,
) -> list[dict]:
    top_k = top_k or settings.top_k_results
    prefetch_limit = top_k * 4

    embeddings_model = get_embeddings()
    dense_vector = embeddings_model.embed_query(query)

    sparse_result = embed_sparse_query(query)
    sparse_vector = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    collection_filter = Filter(
        must=[
            FieldCondition(
                key="collection_id",
                match=MatchValue(value=collection_id),
            )
        ]
    )

    client = get_qdrant_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                filter=collection_filter,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=sparse_vector,
                using="sparse",
                filter=collection_filter,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "text": point.payload.get("text", ""),
            "document_id": point.payload.get("document_id"),
            "filename": point.payload.get("filename"),
            "chunk_index": point.payload.get("chunk_index"),
            "score": point.score,
        }
        for point in results.points
    ]


def multi_query_hybrid_search(
    queries: list[str],
    collection_id: str,
    top_k: int | None = None,
) -> list[dict]:
    top_k = top_k or settings.top_k_results
    seen_texts = set()
    all_results = []
    score_map: dict[str, float] = {}

    for query in queries:
        results = hybrid_search(query, collection_id, top_k=top_k * 2)
        for rank, r in enumerate(results):
            text_key = r["text"][:200]
            rrf_score = 1.0 / (settings.rrf_k + rank + 1)

            if text_key in score_map:
                score_map[text_key] += rrf_score
            else:
                score_map[text_key] = rrf_score
                all_results.append(r)

    all_results.sort(key=lambda r: score_map.get(r["text"][:200], 0), reverse=True)
    return all_results[:top_k]
