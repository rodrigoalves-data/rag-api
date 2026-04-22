from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import faiss
import shutil
import anthropic
import os
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
chunks = []
index = None

def split_text(text, chunk_size=1000, overlap=200):
    result = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        next_start = end - overlap
        if next_start <= start:
            break
        start = next_start
    return result

def get_embedding(text):
    response = client.embeddings.create(
        model="voyage-3",
        input=text
    )
    return response.embeddings[0]

def search(query, k=3):
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "RAG API online"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks, index

    with open(file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file.filename)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    chunks = split_text(text)

    embeddings = np.array([get_embedding(c) for c in chunks]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return {"message": f"✅ PDF processado com {len(chunks)} chunks"}

@app.post("/ask")
async def ask(question: Question):
    context = "\n\n---\n\n".join(search(question.query))
    prompt = f"""Responde à pergunta usando APENAS o contexto fornecido.
Se a informação não estiver no contexto, diz "Não encontrei essa informação no documento."

=== CONTEXTO ===
{context}
================

Pergunta: {question.query}
Resposta:"""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"resposta": message.content[0].text}
