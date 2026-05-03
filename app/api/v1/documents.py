import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.document import Document
from app.models.collection import Collection
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.services.ingestion import ingest_document

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    collection_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    file_bytes = await file.read()

    doc = Document(
        collection_id=collection_id,
        filename=file.filename,
        content_type=file.content_type,
        file_size_bytes=len(file_bytes),
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(
        ingest_document,
        document_id=doc.id,
        collection_id=collection_id,
        file_bytes=file_bytes,
        content_type=file.content_type or "text/plain",
        chunk_size=collection.chunk_size,
        chunk_overlap=collection.chunk_overlap,
    )

    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        status="pending",
        message="Document queued for processing",
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    return (
        db.query(Document)
        .filter(Document.collection_id == collection_id)
        .order_by(Document.created_at.desc())
        .all()
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from app.services.qdrant import get_qdrant_client
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from app.core.config import settings

    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
            ),
        )
    except Exception:
        pass

    db.delete(doc)
    db.commit()
