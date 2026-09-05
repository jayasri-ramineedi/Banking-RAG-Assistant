from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from embeddings import create_embeddings
from generator import prompt
from evaluation import evaluate_answer

import os
from dotenv import load_dotenv


load_dotenv()


def create_rag_chain():

    # Create embedding model
    embeddings = create_embeddings()

    # Load ChromaDB
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Create Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    return retriever, llm


if __name__ == "__main__":

    # Create RAG components
    retriever, llm = create_rag_chain()

    # Get question from user
    question = input("Enter your question: ")

    # Retrieve relevant documents
    retrieved_documents = retriever.invoke(question)

    print("\n===== RETRIEVED CONTEXT =====")

    for i, document in enumerate(retrieved_documents):

        print(f"\n--- Document {i + 1} ---")
        print(document.page_content)

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    # Create final prompt
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Generate answer using Gemini
    response = llm.invoke(final_prompt)

    # Extract answer
    try:
        answer = response.content[0]["text"]
    except (TypeError, IndexError, KeyError):
        answer = response.content

    print("\nANSWER:")
    print(answer)

    # Evaluate RAG response
    evaluation_result = evaluate_answer(
        question=question,
        context=context,
        answer=answer
    )

    print("\n===== RAG EVALUATION =====")
    print(evaluation_result)