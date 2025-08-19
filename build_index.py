# build_index.py
from utils import chunk_texts_from_file, build_faiss_index

if __name__ == "__main__":
    print("Building FAISS index from data.txt ...")
    chunks = chunk_texts_from_file("data.txt")
    index, embs = build_faiss_index(chunks)
    print(f"Index built with {len(chunks)} chunks.")
