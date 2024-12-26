# FAQ-Retrieval


 **FAQ Retrieval System** using an open-source LLM  and a dataset of (FAQs). The system allows users to input questions and retrieves the most relevant answers from the dataset.

### Features

- **Open-Source LLM**: Utilizes `google/flan-t5-small` from Hugging Face for text generation.
- **Retrieval-Augmented Generation (RAG)**: Combines a vector database with an LLM to provide accurate and context-aware answers.

### How It Works


1. **Embeddings**: FAQ texts are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Vector Store**: The embeddings are stored in a FAISS vector database for efficient retrieval.
3. **LLM**: The `google/flan-t5-small` model processes the retrieved context and generates a relevant answer.
4. **User Interaction**: Users input questions, and the system retrieves and displays the best-matched answer.

---

### Installation

1. Install dependencies:
   ```bash
   pip install langchain faiss-cpu sentence-transformers transformers numpy
   ```



2. Run the system:
   ```bash
   python faq_retrieval.py
   ```

3. Ask a question in the terminal:
   ```bash
   Ask a question: What is quantum computing?
   ```

4. Exit the system by typing:
   ```bash
   exit
   ```

---

### Example Interaction

```plaintext
Welcome to the FAQ Retrieval System! Type 'exit' to quit.
Ask a question: What is quantum computing?
Answer: Quantum computing uses quantum-mechanical phenomena to perform computations.
Relevant Context:
Question: What is quantum computing? Answer: Quantum computing uses quantum-mechanical phenomena to perform computations.
Ask a question: exit
Goodbye!
```

---

### Dependencies

- `langchain`: Framework for building applications with LLMs.
- `faiss-cpu`: Vector search library for efficient retrieval.
- `sentence-transformers`: For embedding FAQ texts.
- `transformers`: Provides the Hugging Face LLM.
- `numpy`: Numerical operations.



Replace `your-username` in the clone URL with your GitHub username, and you’re all set!
