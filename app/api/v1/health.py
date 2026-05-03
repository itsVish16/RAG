from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.qdrant import get_qdrant_client
from app.schemas.common import HealthResponse

router = APIRouter()


@router.get("", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", service="rag-chatbot")


@router.get("/ready", response_model=HealthResponse)
def readiness(db: Session = Depends(get_db)):
    db.execute(__import__("sqlalchemy").text("SELECT 1"))

    client = get_qdrant_client()
    client.get_collections()

    return HealthResponse(status="ready", service="rag-chatbot")
