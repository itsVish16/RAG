from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from app.core.config import settings

async get_chat_llm() -> ChatMistralAI:
    return ChatMistralAI(
        api_key = settins.mistal_api_key,
        model=settings.mistral_chat_model,
        temperature=0.2,
    )

async def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(
        api_key = settins.mistal_api_key,
        model=settings.mistral_embedding_model
    )
