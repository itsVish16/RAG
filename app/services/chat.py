import uuid
from sqlalchemy.orm import Session
from app.models.chat import Conversation, ChatMessage, ChatMessageSource


def get_or_create_conversation(
    db: Session,
    conversation_id: uuid.UUID | None,
    collection_id: uuid.UUID,
    title: str = "New Chat",
) -> Conversation:
    if conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            return conv

    conv = Conversation(collection_id=collection_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def save_message(
    db: Session,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> ChatMessage:
    msg = ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(msg)
    db.flush()

    if sources and role == "assistant":
        for src in sources:
            source = ChatMessageSource(
                message_id=msg.id,
                document_id=uuid.UUID(src["document_id"]),
                chunk_text=src["text"],
                relevance_score=src.get("score"),
                chunk_index=src.get("chunk_index"),
            )
            db.add(source)

    db.commit()
    db.refresh(msg)
    return msg


def get_chat_history(db: Session, conversation_id: uuid.UUID, limit: int = 10) -> list[dict]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


def list_conversations(db: Session) -> list[Conversation]:
    return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()


def get_conversation_with_messages(db: Session, conversation_id: uuid.UUID) -> Conversation | None:
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def delete_conversation(db: Session, conversation_id: uuid.UUID) -> bool:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True
