import streamlit as st

from rag_chain import create_rag_chain
from generator import prompt
from evaluation import evaluate_answer


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Banking RAG Assistant",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-top: 10px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏦 Banking LangChain RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered assistant for querying banking credit and recovery documents.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load RAG Components
# --------------------------------------------------

@st.cache_resource
def load_rag():

    retriever, llm = create_rag_chain()

    return retriever, llm


retriever, llm = load_rag()


# --------------------------------------------------
# Question Input
# --------------------------------------------------

question = st.text_input(
    "🔎 Enter your question",
    placeholder="Example: What is the latest OTS proposal for Apex Textile Mills?"
)


# --------------------------------------------------
# Ask Button
# --------------------------------------------------

if st.button("Ask Question", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching banking documents and generating answer..."):

            # Retrieve relevant documents
            retrieved_documents = retriever.invoke(question)

            # Create context
            context = "\n\n".join(
                document.page_content
                for document in retrieved_documents
            )

            # Create final prompt
            final_prompt = prompt.format(
                context=context,
                question=question
            )

            # Generate answer
            response = llm.invoke(final_prompt)

            # Extract answer
            try:
                answer = response.content[0]["text"]
            except (TypeError, IndexError, KeyError):
                answer = response.content


            # --------------------------------------------------
            # Answer Section
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">💡 Answer</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )


            # --------------------------------------------------
            # Evaluation
            # --------------------------------------------------

            with st.spinner("Evaluating response..."):

                evaluation_result = evaluate_answer(
                    question=question,
                    context=context,
                    answer=answer
                )


            st.markdown(
                '<div class="section-title">📊 RAG Evaluation</div>',
                unsafe_allow_html=True
            )

            # Parse evaluation scores
            faithfulness = "N/A"
            relevance = "N/A"
            context_relevance = "N/A"

            for line in evaluation_result.splitlines():

                line = line.strip()

                if line.startswith("Faithfulness:"):
                    faithfulness = line.split(":", 1)[1].strip()

                elif line.startswith("Context Relevance:"):
                    context_relevance = line.split(":", 1)[1].strip()

                elif line.startswith("Relevance:"):
                    relevance = line.split(":", 1)[1].strip()


            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Faithfulness",
                    faithfulness
                )

            with col2:
                st.metric(
                    "Answer Relevance",
                    relevance
                )

            with col3:
                st.metric(
                    "Context Relevance",
                    context_relevance
                )


            # --------------------------------------------------
            # Retrieved Context
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">📚 Retrieved Context</div>',
                unsafe_allow_html=True
            )

            with st.expander(
                "View retrieved documents"
            ):

                for i, document in enumerate(
                    retrieved_documents
                ):

                    st.markdown(
                        f"### Document {i + 1}"
                    )

                    st.write(
                        document.page_content
                    )

                    # Show source if available
                    source = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    st.caption(
                        f"Source: {source}"
                    )

                    st.divider()