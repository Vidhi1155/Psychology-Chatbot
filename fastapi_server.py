from fastapi import FastAPI, HTTPException
import faiss
import json
import os
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel
from contextlib import asynccontextmanager

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

# Global resources
index = None
embedding_model = None
data = None
chat_memory = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, data, embedding_model, chat_memory
    index = faiss.read_index("faiss_index.bin")
    with open("data_mapping.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    chat_memory = load_chat_memory()
    yield

app = FastAPI(lifespan=lifespan)

# Load chat memory
MEMORY_FILE = "chat_memory.json"
def load_chat_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_memory, f, indent=4)

# FAISS search with smaller top_k
def search_faiss(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    return [data[i] for i in indices[0]]

def psychology_chatbot(query):
    retrieved_docs = search_faiss(query)
    context = "\n".join([doc["Title"] + ": " + doc["Content"][:300] + "..." for doc in retrieved_docs])
    memory_context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in chat_memory[-3:]])

    prompt = f"""You are an AI psychology assistant. Answer the user's query using only the information from the context below.

Previous Conversation:
{memory_context}

Knowledge Context:
{context}

User Query: {query}

Answer:"""

    tt_client = ollama.Client(host='http://localhost:11435')
    response = tt_client.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.5, "top_p": 0.9}
    )
    answer = response["message"]["content"]

    chat_memory.append({"question": query, "answer": answer})
    save_chat_memory()

    return answer

@app.post("/chat")
async def chat_api(request_data: ChatRequest):
    query = request_data.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    response = psychology_chatbot(query)
    return {"response": response}

@app.get("/memory")
async def get_memory():
    return {"chat_memory": chat_memory}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8001)