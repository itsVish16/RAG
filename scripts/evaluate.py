import asyncio
import json
import os
import re
import sys
import uuid
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ingestion import chunk_text
from app.services.mistral import get_embeddings
from app.services.sparse import embed_sparse
from app.services.qdrant import get_qdrant_client, ensure_collection
from app.services.retriever import hybrid_search
from app.services.chain_rag import run_rag_chain
from app.core.config import settings
from qdrant_client.http.models import (
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)

EVAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")
EVAL_COLLECTION_ID = "eval-squad-benchmark"
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = Counter(tokenize(prediction))
    gt_tokens = Counter(tokenize(ground_truth))

    common = pred_tokens & gt_tokens
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / sum(pred_tokens.values())
    recall = num_same / sum(gt_tokens.values())
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize(prediction) == normalize(ground_truth)


def compute_answer_recall(prediction: str, ground_truth: str) -> float:
    gt_tokens = set(tokenize(ground_truth))
    pred_tokens = set(tokenize(prediction))
    if not gt_tokens:
        return 0.0
    return len(gt_tokens & pred_tokens) / len(gt_tokens)


async def ingest_eval_documents():
    manifest_path = os.path.join(EVAL_DATA_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: Run 'python scripts/download_dataset.py' first!")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Ingesting {len(manifest)} documents into Qdrant...")
    ensure_collection(dense_size=1024)

    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="collection_id",
                    match=MatchValue(value=EVAL_COLLECTION_ID),
                )
            ]
        ),
    )
    embeddings_model = get_embeddings()
    total_chunks = 0

    for doc_data in manifest:
        doc_path = os.path.join(EVAL_DATA_DIR, doc_data["filename"])
        with open(doc_path) as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        texts = [c.page_content for c in chunks]

        print(f"  {doc_data['filename']}: {len(texts)} chunks...", end=" ", flush=True)

        dense_vectors = embeddings_model.embed_documents(texts)
        sparse_vectors = embed_sparse(texts)

        points = []
        for i, (chunk_content, dense_vec, sparse_vec) in enumerate(
            zip(texts, dense_vectors, sparse_vectors)
        ):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec,
                        "sparse": SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "collection_id": EVAL_COLLECTION_ID,
                        "document_id": str(uuid.uuid4()),
                        "chunk_index": i,
                        "text": chunk_content,
                        "filename": doc_data["filename"],
                    },
                )
            )

        batch_size = 20
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[i : i + batch_size],
            )
        
        # Small delay to prevent embedding rate limits
        await asyncio.sleep(0.5) 

        total_chunks += len(texts)
        print("✓")

    print(f"\nIngestion complete: {total_chunks} total chunks\n")
    return manifest


async def run_evaluation():
    manifest = await ingest_eval_documents()

    total_questions = 0
    total_retrieval_hits = 0
    total_f1 = 0.0
    total_answer_recall = 0.0
    total_em = 0
    total_latency = 0.0
    all_results = []

    total_qa = sum(len(d["qa_pairs"]) for d in manifest)
    print("=" * 80)
    print(f"EVALUATING RAG PIPELINE — {total_qa} questions across {len(manifest)} docs")
    print("=" * 80)

    for doc_data in manifest:
        doc_title = doc_data["title"]
        print(f"\n📄 {doc_title}")
        print("-" * 60)

        for qa in doc_data["qa_pairs"]:
            total_questions += 1
            question = qa["question"]
            ground_truth = qa["answer"]
            expected_file = doc_data["filename"]

            start = time.time()
            try:
                result = await run_rag_chain(
                    query=question,
                    collection_id=EVAL_COLLECTION_ID,
                    chat_history=None,
                    top_k=5,
                )
                latency = time.time() - start
                # Respect 1 QPS limit (2 LLM calls per question = 2s total minimum)
                await asyncio.sleep(1.5) 
            except Exception as e:
                print(f"  ❌ ERROR on '{question[:50]}': {e}")
                all_results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "error": str(e),
                })
                await asyncio.sleep(2.0) # wait longer on error
                continue

            total_latency += latency

            retrieved_files = [s["filename"] for s in result["sources"]]
            retrieval_hit = expected_file in retrieved_files
            total_retrieval_hits += int(retrieval_hit)
            normalized_ground_truth = normalize(ground_truth)
            answer_in_context = any(
                normalized_ground_truth in normalize(source["text"])
                for source in result["sources"]
                if source.get("text")
            )

            answer = result["answer"]
            f1 = compute_f1(answer, ground_truth)
            em = compute_exact_match(answer, ground_truth)
            ar = compute_answer_recall(answer, ground_truth)
            total_f1 += f1
            total_em += int(em)
            total_answer_recall += ar

            status = "✅" if retrieval_hit and f1 > 0.3 else "⚠️" if retrieval_hit else "❌"
            print(f"  {status} Q: {question[:60]}")
            print(f"     GT: {ground_truth[:80]}")
            print(f"     Answer: {answer[:80]}")
            print(
                f"     Retrieval: {'HIT' if retrieval_hit else 'MISS'}"
                f" | Gold-in-context: {'YES' if answer_in_context else 'NO'}"
                f" | F1: {f1:.3f} | Recall: {ar:.3f} | {latency:.1f}s"
            )

            all_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "rewritten_query": result.get("rewritten_query"),
                "multi_queries": result.get("multi_queries"),
                "sources": [s["filename"] for s in result["sources"]],
                "retrieval_hit": retrieval_hit,
                "answer_in_context": answer_in_context,
                "f1": f1,
                "exact_match": em,
                "answer_recall": ar,
                "latency": latency,
            })

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    metrics = {
        "total_questions": total_questions,
        "retrieval_recall_at_5": total_retrieval_hits / max(total_questions, 1),
        "avg_f1": total_f1 / max(total_questions, 1),
        "avg_answer_recall": total_answer_recall / max(total_questions, 1),
        "exact_match_rate": total_em / max(total_questions, 1),
        "avg_latency_seconds": total_latency / max(total_questions, 1),
        "total_latency_seconds": total_latency,
    }

    print(f"  Total Questions:       {metrics['total_questions']}")
    print(f"  Retrieval Recall@5:    {metrics['retrieval_recall_at_5']:.2%}")
    print(f"  Average F1 Score:      {metrics['avg_f1']:.3f}")
    print(f"  Average Answer Recall: {metrics['avg_answer_recall']:.3f}")
    print(f"  Exact Match Rate:      {metrics['exact_match_rate']:.2%}")
    print(f"  Avg Latency:           {metrics['avg_latency_seconds']:.2f}s")
    print("=" * 80)

    report = {"metrics": metrics, "results": all_results}
    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
