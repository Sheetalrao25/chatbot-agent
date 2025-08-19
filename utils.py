import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
CHUNKS_PATH = "chunks.pkl"

embedder = SentenceTransformer(EMBED_MODEL_NAME)
def chunk_texts_from_file(path="data.txt", chunk_size=400, chunk_overlap=50):
    with open(path, "r", encoding="utf-8") as f:
        full = f.read().strip()
    
    paras = [p.strip() for p in full.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
           
            start = 0
            while start < len(p):
                end = start + chunk_size
                chunk = p[start:end].strip()
                chunks.append(chunk)
                start += (chunk_size - chunk_overlap)
    return chunks
def build_faiss_index(chunks):
    embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    return index, embs

def load_faiss_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search_index(query, index, chunks, top_k=2):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append((chunks[int(idx)], float(score)))
    return results
