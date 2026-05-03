import json
from app.services.mistral import get_small_llm

PREPROCESS_PROMPT = """Analyze the following user question and chat history.
Your goal is to prepare the queston for a vector search.

1.  **Standalone Query**: Rewrite the question into a clear, standalone, search-optimized query. Resolve any references (like "it", "that", "this") using the history.
2.  **Alternative Queries**: Generate 2-3 different versions of the standalone query to improve document retrieval from different angles.

Output MUST be a valid JSON object with the following structure:
{{
    "standalone_query": "the rewritten query",
    "alternatives": ["alt query 1", "alt query 2"]
}}

Chat History:
{history}

User Question: {question}

JSON Output:"""


async def preprocess_search(question: str, chat_history: list[dict] | None = None) -> dict:
    llm = get_small_llm()
    
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in (chat_history or [])[-6:]
    )

    response = await llm.ainvoke(
        PREPROCESS_PROMPT.format(history=history_str, question=question)
    )
    
    content = response.content.strip()
    # Clean up markdown code blocks if the LLM adds them
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        data = json.loads(content)
        return {
            "standalone_query": data.get("standalone_query", question),
            "alternatives": [question] + data.get("alternatives", [])
        }
    except Exception:
        # Fallback if JSON parsing fails
        return {
            "standalone_query": question,
            "alternatives": [question]
        }
