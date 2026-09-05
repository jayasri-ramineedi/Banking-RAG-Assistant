from langchain_chroma import Chroma

from embeddings import create_embeddings


def create_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


if __name__ == "__main__":

    # Load embedding model
    embeddings = create_embeddings()

    # Load existing ChromaDB
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # User question
    question = "What is the latest OTS proposal for Apex Textile Mills Pvt. Ltd.?"

    # Retrieve top 3 documents
    retriever = create_retriever(vectorstore)

    results = retriever.invoke(question)

    print("Number of retrieved documents:", len(results))

    for i, document in enumerate(results):

        print(f"\n--- Retrieved Document {i + 1} ---")
        print(document.page_content)