# RAG-PDF-Chatbot

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline using OpenAI GPT models to build a chatbot that can answer questions from PDF documents.

## Features
- **PDF Loader**: Extract text from PDFs with metadata (e.g., page numbers).
- **Vector Search**: Efficient retrieval using Chroma vector stores.
- **Token Tracking**: Monitor token usage and estimated costs for queries.
- **Streaming Responses**: Get incremental outputs from OpenAI GPT for a better user experience.
- **Persistence**: Save embeddings for reuse, avoiding reprocessing.

## Installation

### Prerequisites
- Python 3.12 or higher
- OpenAI API Key ([Get yours here](https://platform.openai.com/signup/))

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/JordiCorbilla/RAG-PDF-Chatbot.git
    cd RAG-PDF-Chatbot
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Add your OpenAI API key to the environment:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```

## Usage

1. Place your PDF files in the `example_pdfs/` directory.
2. Run the chatbot:
    ```bash
    python main.py
    ```
3. Ask questions about the document interactively.

### Example
```bash
> Ask a question (or type 'exit' to quit): What is the document about?
Answer:
The document explains the MiFID II application process.

Source Documents:
- Source: cp15-43.pdf, Page: 3

Token Usage:
- Prompt tokens: 250
- Completion tokens: 100
- Total tokens: 350
- Estimated cost: $0.00210
