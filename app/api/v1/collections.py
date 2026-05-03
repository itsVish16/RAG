import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.collection import Collection
from app.schemas.collection import CollectionCreate, CollectionUpdate, CollectionResponse

router = APIRouter()


@router.post("", response_model=CollectionResponse, status_code=201)
def create_collection(payload: CollectionCreate, db: Session = Depends(get_db)):
    existing = db.query(Collection).filter(Collection.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=409, detail="Collection with this name already exists")

    collection = Collection(**payload.model_dump())
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return collection


@router.get("", response_model=list[CollectionResponse])
def list_collections(db: Session = Depends(get_db)):
    return db.query(Collection).order_by(Collection.created_at.desc()).all()


@router.get("/{collection_id}", response_model=CollectionResponse)
def get_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


@router.patch("/{collection_id}", response_model=CollectionResponse)
def update_collection(collection_id: uuid.UUID, payload: CollectionUpdate, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(collection, key, value)

    db.commit()
    db.refresh(collection)
    return collection


@router.delete("/{collection_id}", status_code=204)
def delete_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    from app.services.qdrant import get_qdrant_client
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from app.core.config import settings

    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="collection_id", match=MatchValue(value=str(collection_id)))]
            ),
        )
    except Exception:
        pass

    db.delete(collection)
    db.commit()
