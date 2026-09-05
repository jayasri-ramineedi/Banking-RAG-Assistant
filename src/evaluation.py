from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

def evaluate_answer(question, context, answer):
    prompt = f"""
You are evaluating a RAG systm.

Question:
{question}

Restrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the answer using these metric:

1. Faithfulness: Is the answer supported by the retrived context?
2. Relevance: Does the answer directly answer the question?
3. Context Relevance: Is the retrieved context useful for answering the question?

Give a score from 0 to 10 for each metric.

Return ONLY in this format:

Faithfulness: X/10
Relevance: X/10
Context Relevance: X/10
"""

    response = llm.invoke(prompt)

    # Extract generated text
    try:
        return response.content[0]["text"]
    except (TypeError, IndexError, KeyError):
        return response.content