from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_pdfs(data_dir):

    loader = PyPDFDirectoryLoader(data_dir)

    documents = loader.load()

    for document in documents:

        print("Document loaded:", document.metadata.get("source"))

    return documents

if __name__ == "__main__":

    documents = load_pdfs("data/Banking")
    print("Total Pages:", len(documents))