from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from app.core.config import settings

def is_rate_limit_error(exception):
    return "rate limit" in str(exception).lower() or "429" in str(exception).lower()

retry_decorator = retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(is_rate_limit_error),
)


@retry_decorator
def get_chat_llm() -> ChatMistralAI:
    return ChatMistralAI(
        api_key=settings.mistral_api_key,
        model=settings.mistral_chat_model,
        temperature=0.2,
    )


@retry_decorator
def get_small_llm() -> ChatMistralAI:
    return ChatMistralAI(
        api_key=settings.mistral_api_key,
        model=settings.mistral_small_model,
        temperature=0.0,
    )


@retry_decorator
def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(
        api_key=settings.mistral_api_key,
        model=settings.mistral_embedding_model,
    )
