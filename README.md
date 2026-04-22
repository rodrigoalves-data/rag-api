# RAG API com FastAPI + Claude

API REST que permite fazer perguntas a documentos PDF usando RAG (Retrieval-Augmented Generation).

## Como funciona
1. Faz upload de um PDF via `/upload`
2. Faz perguntas via `/ask`
3. A API recupera os chunks mais relevantes e o Claude gera a resposta

## Stack
- **FastAPI** — API REST
- **FAISS** — indexação e pesquisa por similaridade
- **SentenceTransformers** — embeddings (`all-MiniLM-L6-v2`)
- **Claude** — geração de respostas
- **LangChain** — carregamento de PDFs

## Endpoints
- `GET /` — status da API
- `POST /upload` — upload de PDF
- `POST /ask` — fazer uma pergunta

## Variáveis de ambiente
- `ANTHROPIC_API_KEY` — a tua API key da Anthropic
