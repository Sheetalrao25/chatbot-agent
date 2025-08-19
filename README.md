# chatbot-agent
Abdominal Pain Q/A Chatbot

A Retrieval-Augmented Generation (RAG) Chatbot built with FastAPI, FAISS, and Hugging Face Transformers.
The chatbot answers medical questions about Abdominal Pain in Adults using a curated Q/A dataset.

Features

✅ Upload knowledge (data.txt) with Q/A pairs

✅ Text chunking + embedding generation (SentenceTransformers)

✅ FAISS vector database for semantic search

✅ Hugging Face LLM for professional answers

✅ Handles small talk (hi/hello)

✅ Responds “Sorry, I’m not trained for this” if question is out of domain



pip install -r requirements.txt
uvicorn app:app --reload
