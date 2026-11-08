# Local PDF Chatbot Using RAG

An AI-powered chatbot that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## Features

- Upload and process PDF documents
- Extract text from PDF files
- Split text into smaller chunks
- Generate semantic embeddings
- Retrieve relevant information using FAISS
- Generate answers using a locally running LLM
- Display relevant PDF evidence and page numbers
- Interactive Streamlit interface

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Ollama
- Llama 3.2 (1B)

## Project Structure

    local-rag-chatbot/
    ├── app.py
    ├── pdf_processor.py
    ├── rag_engine.py
    ├── test_pdf.py
    ├── test_rag.py
    ├── test_answer.py
    ├── requirements.txt
    ├── README.md
    └── data/

## Installation and Setup

### 1. Clone the repository

    git clone https://github.com/Shanto-1712/local-pdf-chatbot-rag.git

    cd local-pdf-chatbot-rag

### 2. Create a virtual environment

    python3 -m venv .venv

### 3. Activate the environment

    source .venv/bin/activate

### 4. Install dependencies

    pip install -r requirements.txt

### 5. Install Ollama

Download and install Ollama from:

https://ollama.com/

### 6. Download the language model

    ollama pull llama3.2:1b

### 7. Run the application

    streamlit run app.py

Open the local URL displayed in your terminal.

## How It Works

1. Upload a PDF document.
2. Extract and split its text into chunks.
3. Convert chunks into vector embeddings.
4. Store embeddings in a FAISS index.
5. Retrieve relevant chunks for a user question.
6. Send retrieved context to the local LLM.
7. Display the generated answer with page references.

## Future Improvements

- Advanced retrieval and reranking
- Improved citation verification
- Automated RAG evaluation
- Support for multiple PDF documents

## Author

Md Shamim Akter Shanto

CSE Student | AI & Machine Learning

## License

MIT License