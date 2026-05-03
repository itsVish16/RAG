import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    top_k: int = 5


class SourceResponse(BaseModel):
    document_id: uuid.UUID
    filename: str | None = None
    chunk_text: str
    relevance_score: float | None
    chunk_index: int | None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    sources: list[SourceResponse] = []


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: str
    sources: list[SourceResponse] = []

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID | None
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(ConversationResponse):
    messages: list[ChatMessageResponse] = []
