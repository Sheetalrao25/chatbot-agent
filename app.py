from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_faiss_index, search_index
from services import generate_answer

app = FastAPI(title="RAG Chatbot (Abdominal Pain Assistant)")
index, chunks = load_faiss_index()
if index is None:
    raise RuntimeError("FAISS index not found. Run `python build_index.py` first to create index from data.txt")
GREETINGS = {"hi", "hello", "hey", "hii", "good morning", "good evening", "hey there"}
SIM_THRESHOLD = 0.45
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    question = q.question.strip()
    if not question:
        return {"answer": "Please type a question."}

    qlow = question.lower().strip()
    if qlow in GREETINGS:
        return {"answer": "Hello! I’m a medical assistant trained on abdominal pain in adults. How can I help?"}

    results = search_index(question, index, chunks, top_k=2) 
    top_chunk, top_score = results[0]

    if top_score < SIM_THRESHOLD:
        return {"answer": "Sorry — I’m not trained for that."}

    context = " ".join([r[0] for r in results])
    answer = generate_answer(question, context)

    if "sorry" in answer.lower() and "not trained" in answer.lower():
        return {"answer": "Sorry — I’m not trained for that."}

    return {"answer": answer, "sources": [r[0] for r in results], "top_score": top_score}
