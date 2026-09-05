# 🏦 Banking RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions from banking credit and recovery documents using LangChain, ChromaDB, HuggingFace embeddings, and Gemini.

# 🚀 Features

- 📄 Loads banking PDF documents
- ✂️ Splits documents into meaningful chunks
- 🔢 Generates semantic embeddings using HuggingFace
- 🗄️ Stores embeddings in ChromaDB
- 🔎 Retrieves relevant document chunks
- 🤖 Generates answers using Gemini
- 📊 Evaluates RAG responses using:
  - Faithfulness
  - Answer Relevance
  - Context Relevance
- 🖥️ Interactive Streamlit interface

# 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers
- Gemini API
- PyPDF
- Streamlit
- Python-dotenv

# 🔄 RAG Architecture

Banking PDF Documents
        ↓
PDF Loader
        ↓
Text Splitter
        ↓
HuggingFace Embeddings
        ↓
ChromaDB
        ↓
Retriever
        ↓
Relevant Context
        ↓
Prompt Template
        ↓
Gemini
        ↓
Generated Answer
        ↓
RAG Evaluation

# 📂 Project Structure

BankingRAG/
│
├── .venv/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   └── Banking/
│       ├── 01_WorkingCapital_CaseFile_DeccanAuto.pdf
│       ├── 02_TermLoan_CovenantBreach_VantageFoods.pdf
│       ├── 03_Restructuring_CaseFile_MetroHospitality.pdf
│       └── 04_NPA_Recovery_CaseFile_ApexTextile.pdf
│
├── chroma_db/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── loader.py
    ├── splitter.py
    ├── embeddings.py
    ├── vector_store.py
    ├── query.py
    ├── generator.py
    ├── evaluation.py
    ├── rag_chain.py
    └── app.py

# 📄 Banking Documents

The knowledge base contains four banking case files covering:

1. Working Capital Facility
2. Term Loan Covenant Breach
3. Loan Restructuring
4. NPA and Recovery

# 💬 Example Questions

1. When is the peak working capital requirement for Deccan Auto Components?
2. What is the DSCR of Vantage Foods Processing?
3. What is the latest OTS proposal for Apex Textile Mills?
4. What is the restructuring plan for Metro Hospitality Ventures?

# 📊 RAG Evaluation

The system evaluates generated answers using three metrics:

### Faithfulness

Measures whether the answer is supported by the retrieved context.

### Answer Relevance

Measures whether the answer directly addresses the user's question.

### Context Relevance

Measures how relevant the retrieved documents are to the question.

Each metric is scored from 0 to 10.

## ⚙️ Setup

### 1. Clone the repository

git clone <your-github-repository-url>
cd BankingRAG

### 2. Create virtual environment

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Add Gemini API Key

Create a .env file and add:

GOOGLE_API_KEY=your_api_key_here

### 5. Create ChromaDB

python src/vector_store.py

### 6. Run the application

streamlit run src/app.py

# 🔐 Security

The Gemini API key is stored in .env and should never be committed to GitHub.

The local ChromaDB database is also excluded using .gitignore.

# 🎯 Objective

The objective of this project is to demonstrate how Retrieval-Augmented Generation can be used to build a domain-specific banking assistant that answers questions using information retrieved from banking documents.

# 🚀 Future Improvements

- Hybrid search
- Conversation memory
- Source citations
- Metadata filtering
- Authentication
- Cloud deployment
- Advanced RAG evaluation