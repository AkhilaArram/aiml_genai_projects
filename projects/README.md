# AI / ML & Generative AI Projects

This repository contains **production-oriented Generative AI projects** focused on
Retrieval-Augmented Generation (RAG), document intelligence, and multimodal AI systems.

The projects demonstrate real-world use cases such as:
- Multi-pdf question answering
- OCR-based document understanding
- Audio-based conversational AI
- Graph and vector database integration

---

## Tech Stack

- Python
- Google Generative AI (Gemini)
- LangChain & LangGraph
- FAISS & Neo4j Vector
- Streamlit
- Whisper (Speech-to-Text)
- OCR (Tesseract)

---

## Featured Projects

### 1. multi_pdf_routerchain
**Location:** `streamlit_apps/multi_pdf_routerchain.py`

- Supports querying across multiple PDFs
- Uses LangChain RouterChain to route queries to:
  - PDF content QA
  - Document summaries
  - Chat history
- Built with Streamlit for interactive UI
- Designed with robust router output parsing

**Key Concepts:** RAG, RouterChain, FAISS, Gemini, Streamlit

---

### 2. ocr_langgraph_pipeline
**Location:** `rag_pipelines/ocr_langgraph_pipeline.py`

- Extracts text and images from PDFs
- Performs OCR on images
- Merges OCR text with document text
- Uses LangGraph state pipelines for structured processing
- Handles scanned and image-heavy PDFs effectively

**Key Concepts:** OCR, LangGraph, Document Intelligence, RAG

---

### 3. Neo4j Vector-Based Multi-User PDF Chatbot
**Location:** `streamlit_apps/neo4jrag_chatbot.py`

- Uses Neo4j as a vector database
- Supports multi-user and multi-PDF isolation
- Stores chunks with metadata and relationships
- Enables filtered and scalable retrieval

**Key Concepts:** Neo4j Vector, Metadata Filtering, RAG

---

### 4. Audio-Based RAG Chatbot
**Location:** `streamlit_apps/whisper_audio_chatbot.py`

- Upload or record audio input
- Transcription using Whisper
- RAG-based question answering
- Text-to-speech responses
- End-to-end multimodal AI workflow

**Key Concepts:** Whisper, Audio RAG, Multimodal AI

---

## How to Run

1. Clone the repository
2. Install dependencies  
   `pip install -r requirements.txt`
3. Add your Google API key as an environment variable
4. Run any Streamlit app  
   `streamlit run streamlit_apps/multi_pdf_rag_routerchain_app.py`

---

## Notes

These projects were built to explore **production-grade GenAI system design**
including routing, OCR pipelines, multimodal inputs, and scalable vector storage.
