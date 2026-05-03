from fastapi import APIRouter
from app.api.v1 import health, collections, documents, chat
from app.core.config import settings

api_router = APIRouter(prefix=settings.api_v1_prefix)

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(collections.router, prefix="/collections", tags=["Collections"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
