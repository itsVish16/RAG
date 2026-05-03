import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from qdrant_client.http.models import PointStruct, SparseVector
from pypdf import PdfReader
import io

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.document import Document
from app.services.mistral import get_embeddings
from app.services.sparse import embed_sparse
from app.services.qdrant import get_qdrant_client, ensure_collection

IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/bmp"}


def extract_text(file_bytes: bytes, content_type: str, filename: str = "") -> str:
    if content_type in IMAGE_TYPES:
        from app.services.vision import extract_text_from_image
        return extract_text_from_image(file_bytes, filename=filename)

    if content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        has_images = False

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            if page.images:
                has_images = True

        text = "\n".join(text_parts)

        if has_images and len(text.strip()) < 100:
            from app.services.vision import extract_text_from_pdf_images
            return extract_text_from_pdf_images(file_bytes)

        return text

    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[LCDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.create_documents([text])


def ingest_document(
    document_id: uuid.UUID,
    collection_id: uuid.UUID,
    file_bytes: bytes,
    content_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return

        doc.status = "processing"
        db.commit()

        text = extract_text(file_bytes, content_type, filename=doc.filename)
        normalized_text = text.strip()
        if not normalized_text:
            doc.status = "failed"
            doc.error_message = "No text content extracted"
            db.commit()
            return

        chunks = chunk_text(normalized_text, chunk_size, chunk_overlap)
        texts = [c.page_content for c in chunks]

        embeddings_model = get_embeddings()
        dense_vectors = embeddings_model.embed_documents(texts)

        sparse_vectors = embed_sparse(texts)

        ensure_collection(dense_size=len(dense_vectors[0]))

        points = []
        for i, (chunk_text_content, dense_vec, sparse_vec) in enumerate(
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
                        "collection_id": str(collection_id),
                        "document_id": str(document_id),
                        "chunk_index": i,
                        "text": chunk_text_content,
                        "filename": doc.filename,
                    },
                )
            )

        client = get_qdrant_client()
        batch_size = 100
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[i : i + batch_size],
            )

        doc.status = "ready"
        doc.chunk_count = len(chunks)
        db.commit()

    except Exception as e:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_message = str(e)[:500]
            db.commit()
        raise
    finally:
        db.close()
