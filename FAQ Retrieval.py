import numpy as np
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dataset import load_faqs

# Load the dataset
faqs = load_faqs()

# Prepare the dataset for retrieval
faq_texts = [f"Question: {faq['question']} Answer: {faq['answer']}" for faq in faqs]

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(faq_texts, embeddings)

# Define the open-source LLM
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Define the RetrievalQA chain
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:
    """
)
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Main interaction loop
if __name__ == "__main__":
    print("Welcome to the FAQ Retrieval System! Type 'exit' to quit.")
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        result = retrieval_qa({"query": user_input})
        if not result['source_documents']:
            print("I'm sorry, I couldn't find an answer to your question in the FAQ dataset.")
        else:
            print(f"Answer: {result['result']}")
            print("Relevant Context:")
            for doc in result['source_documents']:
                print(doc.page_content)
