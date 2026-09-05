from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_pdfs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":

    documents = load_pdfs("data/Banking")

    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))
    print("\nFirst chunk:")
    print(chunks[0].page_content)    