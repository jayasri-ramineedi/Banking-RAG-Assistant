from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template = """
You are a helpful banking assistant.

Answer the question only using the context below.

If the answer is not available in the context,
say: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

Give a concise and accurate answer.""",
input_variables = ["context","question"]
)

if __name__ == "__main__":
    context = """
    The borrower's revenue is seasonal, with peak working capital
    requirement typically between October and February."""

    question = "When is the peak working capital requirement?"

    final_prompt = prompt.format(
        context = context,
        question = question
    )

    print("Generated Prompt:")
    print(final_prompt)