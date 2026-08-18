from __future__ import annotations

import time
from pathlib import Path

import chromadb
import requests

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "manuals"

EMBED_URL = "http://localhost:8081/v1/embeddings"
CHAT_URL = "http://localhost:8080/v1/chat/completions"
HEALTH_EMBED = "http://localhost:8081/health"
HEALTH_CHAT = "http://localhost:8080/health"

SYSTEM_PROMPT = (
    "Tu es l'assistant technique de la machine. "
    "Réponds à l'opérateur uniquement avec l'extrait suivant du manuel."
)


def wait_for(url: str, name: str, timeout_s: int = 180) -> None:
    last_error = None
    for _ in range(timeout_s):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code < 500:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"{name} n'est pas pret sur {url}: {last_error}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = requests.post(EMBED_URL, json={"input": texts}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    items = sorted(payload["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in items]


def embed_query(question: str) -> list[float]:
    return embed_texts([f"query: {question}"])[0]


def open_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search_chunks(question: str, n_results: int = 3) -> list[dict]:
    collection = open_collection()
    if collection.count() == 0:
        raise RuntimeError("La base est vide. Lance d'abord: python ingest.py")
    query_embedding = embed_query(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "distance": distance,
            }
        )
    return hits


def ask_chat(question: str, excerpts: list[dict]) -> str:
    excerpt_block = "\n\n".join(
        f"[{hit['source']} p.{hit['page']}]\n{hit['text']}" for hit in excerpts
    )
    user_prompt = (
        f"Extrait du manuel : {excerpt_block}\n\n"
        f"Question : {question}\n\n"
        "Reponds en francais, de facon courte et operationnelle. "
        "Si l'extrait ne contient pas la reponse, dis-le clairement."
    )
    response = requests.post(
        CHAT_URL,
        json={
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 256,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
