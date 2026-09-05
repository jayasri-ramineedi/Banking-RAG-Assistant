from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

from splitter import split_documents
from loader import load_pdfs

load_dotenv()

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

if __name__ == "__main__":

    documents = load_pdfs("data/Banking")

    chunks = split_documents(documents)

    embeddings = create_embeddings()
    print("Local Embedding model loaded successfully")

    vector = embeddings.embed_query(
        chunks[0].page_content
    )
    print("Embedding dimension:", len(vector))
