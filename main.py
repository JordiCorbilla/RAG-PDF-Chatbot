# -*- coding: utf-8 -*-
# Copyright 2025 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "yourapikey"

cache = InMemoryCache()

VECTOR_STORE_DIR = "faiss_vector_store"

def load_and_preprocess_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks

def create_or_load_vector_store(chunks=None):
    embeddings = OpenAIEmbeddings()
    persist_directory = "chroma_vector_store"

    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Creating new vector store...")
        if chunks is None:
            raise ValueError("Chunks must be provided to create a new vector store.")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vectorstore

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0, streaming=True, verbose=True),
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

from langchain.callbacks import get_openai_callback

def query_rag_pipeline(rag_pipeline, query):
    with get_openai_callback() as callback:
        result = rag_pipeline({"query": query})
        print("\nAnswer:")
        print(result["result"])

        print("\nSource Documents:")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "Unknown page")
            print(f"- Source: {source}, Page: {page}")

        print("\nToken Usage:")
        print(f"- Prompt tokens: {callback.prompt_tokens}")
        print(f"- Completion tokens: {callback.completion_tokens}")
        print(f"- Total tokens: {callback.total_tokens}")
        print(f"- Estimated cost: ${callback.total_cost:.5f}")
    return result


def main():
    pdf_path = "yourdocument.pdf"

    print("Loading and processing PDF...")
    chunks = load_and_preprocess_pdf(pdf_path)

    print("Creating/loading vector store...")
    vectorstore = create_or_load_vector_store(chunks)

    print("Setting up RAG pipeline...")
    rag_pipeline = create_rag_pipeline(vectorstore)

    print("RAG pipeline is ready. You can now ask questions!")

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        try:
            query_rag_pipeline(rag_pipeline, query)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
