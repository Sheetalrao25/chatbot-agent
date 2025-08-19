from transformers import pipeline
GEN_MODEL = "google/flan-t5-small"
generator = pipeline("text2text-generation", model=GEN_MODEL)

def generate_answer(question: str, context: str) -> str:
    prompt = ("Use ONLY the following context to answer the question. "
        "If the answer is not contained in the context, say 'Sorry — I’m not trained for that.'\n\n"
        f"Context: {context}\n\nQuestion: {question}\nAnswer:")
    out = generator(prompt, max_length=256, do_sample=False, num_beams=2)
    text = out[0]["generated_text"].strip()
    return text
