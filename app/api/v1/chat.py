import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.collection import Collection
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessageResponse,
    ConversationResponse,
    ConversationDetailResponse,
    SourceResponse,
)
from app.services.chain_rag import run_rag_chain
from app.services.chat import (
    delete_conversation,
    get_chat_history,
    get_conversation_with_messages,
    get_or_create_conversation,
    list_conversations,
    save_message,
)

router = APIRouter()


def _map_sources(sources: list) -> list[SourceResponse]:
    return [
        SourceResponse(
            document_id=source.document_id,
            filename=getattr(source.document, "filename", None),
            chunk_text=source.chunk_text,
            relevance_score=source.relevance_score,
            chunk_index=source.chunk_index,
        )
        for source in sources
    ]


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter(Collection.id == payload.collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    conversation = get_or_create_conversation(
        db=db,
        conversation_id=payload.conversation_id,
        collection_id=payload.collection_id,
        title=payload.query[:80] or "New Chat",
    )
    history = get_chat_history(db, conversation.id, limit=10)

    save_message(db, conversation.id, "user", payload.query)
    result = await run_rag_chain(
        query=payload.query,
        collection_id=str(payload.collection_id),
        chat_history=history,
        top_k=payload.top_k,
    )
    assistant_message = save_message(
        db,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        sources=result["sources"],
    )

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=assistant_message.id,
        answer=assistant_message.content,
        sources=_map_sources(assistant_message.sources),
    )


@router.get("/conversations", response_model=list[ConversationResponse])
def get_conversations(db: Session = Depends(get_db)):
    return list_conversations(db)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    conversation = get_conversation_with_messages(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        id=conversation.id,
        collection_id=conversation.collection_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            ChatMessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at.isoformat(),
                sources=_map_sources(message.sources),
            )
            for message in conversation.messages
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
def remove_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    deleted = delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
