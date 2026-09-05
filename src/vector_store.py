from langchain_chroma import Chroma

from embeddings import create_embeddings
from splitter import split_documents
from loader import load_pdfs

def create_vectorstore(chunks, embeddings):

    vectotstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = "chroma_db"
    )
    return create_vectorstore

if __name__ == "__main__":

    documents = load_pdfs("data/Banking")

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vectorestore = create_vectorstore(chunks, embeddings)

    print("ChromaDB vector store created successfully")
    print("Total chunks stored:", len(chunks))