from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)


async def generate_title(message: str) -> str:
    prompt = (
        f"Generate a short chat title (max 50 characters, no quotes, no punctuation at end) "
        f"that is specific to this exact message. Use key terms or the actual subject from the message — "
        f"do NOT use generic labels like 'Math problem' or 'Question'. "
        f"Message: {message}"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()[:50]
