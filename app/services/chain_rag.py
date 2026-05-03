from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from app.services.mistral import get_chat_llm
from app.services.retriever import hybrid_search, multi_query_hybrid_search
from app.services.query_rewriter import preprocess_search

SYSTEM_PROMPT = """You are a precise AI assistant. Answer the user's question based ONLY on the provided context.

Rules:
1. Use ONLY information from the context below. Do not use prior knowledge.
2. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
3. Cite your sources by referencing the source numbers [1], [2], etc.
4. Be concise and accurate.

Context:
{context}"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (Source: {chunk['filename']}, Score: {chunk['score']:.3f})\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def format_chat_history(messages: list[dict]) -> list:
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


async def run_rag_chain(
    query: str,
    collection_id: str,
    chat_history: list[dict] | None = None,
    top_k: int = 5,
) -> dict:
    preprocessed = await preprocess_search(query, chat_history)
    rewritten = preprocessed["standalone_query"]
    multi_queries = preprocessed["alternatives"]

    chunks = multi_query_hybrid_search(multi_queries, collection_id, top_k)

    if not chunks:
        chunks = hybrid_search(rewritten, collection_id, top_k)

    context = build_context(chunks) if chunks else "No relevant context found."

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    llm = get_chat_llm()
    chain = prompt | llm
    history = format_chat_history(chat_history or [])

    response = await chain.ainvoke({
        "context": context,
        "chat_history": history,
        "question": query,
    })

    return {
        "answer": response.content,
        "sources": chunks,
        "rewritten_query": rewritten,
        "multi_queries": multi_queries,
    }


async def stream_rag_chain(
    query: str,
    collection_id: str,
    chat_history: list[dict] | None = None,
    top_k: int = 5,
):
    preprocessed = await preprocess_search(query, chat_history)
    rewritten = preprocessed["standalone_query"]
    multi_queries = preprocessed["alternatives"]
    chunks = multi_query_hybrid_search(multi_queries, collection_id, top_k)

    if not chunks:
        chunks = hybrid_search(rewritten, collection_id, top_k)

    context = build_context(chunks) if chunks else "No relevant context found."

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    llm = get_chat_llm()
    chain = prompt | llm
    history = format_chat_history(chat_history or [])

    async for token in chain.astream({
        "context": context,
        "chat_history": history,
        "question": query,
    }):
        yield token.content, chunks
