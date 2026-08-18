from __future__ import annotations

import time
import unicodedata
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
    "Tu es l'assistant technique de la machine OptiJet. "
    "Tu reponds UNIQUEMENT avec les etapes ecrites dans les extraits du manuel. "
    "Reproduis les actions concretes (boutons, durees, ordre). "
    "N'invente rien. N'utilise pas un extrait de depannage si la question "
    "porte sur une procedure normale. "
    "Si les extraits ne contiennent pas la procedure, dis exactement : "
    "Je n'ai pas trouve cette procedure dans le manuel."
)

SYNONYMS = {
    "demarrer": [
        "mise en marche",
        "mettre en marche",
        "allumer",
        "marche/arret",
        "appui long",
        "touche marche",
        "demarrage",
    ],
    "arreter": ["arret", "eteindre", "mise a l'arret", "marche/arret"],
    "batterie": ["batteries", "remplacement des batteries", "charger"],
    "erreur": ["depannage", "panne", "code", "defaut"],
}

PHRASE_BONUS = [
    "mise en marche",
    "mettre en marche",
    "touche marche",
    "appui long",
    "allumer",
]
PHRASE_PENALTY = [
    "ne demarre pas",
    "depannage",
    "panne",
    "bug logiciel",
    "si le probleme persiste",
]


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.lower()


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


def query_terms(question: str) -> list[str]:
    folded = fold(question)
    terms = [folded]
    for key, extras in SYNONYMS.items():
        if key in folded or any(fold(extra) in folded for extra in extras):
            terms.extend(extras)
            terms.append(key)
    if "optijet" not in folded:
        terms.append("optijet")
    return list(dict.fromkeys(terms))


def enrich_query(question: str) -> str:
    extras = query_terms(question)
    unique = [term for term in extras if term not in fold(question)]
    suffix = " ".join(unique[:6])
    body = question.strip()
    if suffix:
        body = f"{body} {suffix}"
    return f"query: {body}"


def embed_query(question: str) -> list[float]:
    return embed_texts([enrich_query(question)])[0]


def lexical_score(question: str, text: str, heading: str, doc_type: str) -> float:
    blob = fold(f"{heading} {text}")
    terms = query_terms(question)
    score = 0.0
    for term in terms:
        if term and term in blob:
            score += 1.5 if term in fold(heading) else 1.0
    for phrase in PHRASE_BONUS:
        if phrase in blob:
            score += 3.0
    for phrase in PHRASE_PENALTY:
        if phrase in blob:
            score -= 4.0
    if doc_type == "quick_guide" and "comment" in fold(question):
        score += 1.5
    return score


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


def search_chunks(
    question: str,
    n_results: int = 3,
    lang: str = "fr",
    max_distance: float = 0.75,
    fetch_k: int = 30,
) -> list[dict]:
    collection = open_collection()
    if collection.count() == 0:
        raise RuntimeError("La base est vide. Lance d'abord: python ingest.py")

    query_kwargs: dict = {
        "query_embeddings": [embed_query(question)],
        "n_results": min(fetch_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if lang and lang != "all":
        query_kwargs["where"] = {"lang": lang}

    try:
        result = collection.query(**query_kwargs)
    except Exception:
        result = collection.query(
            query_embeddings=query_kwargs["query_embeddings"],
            n_results=query_kwargs["n_results"],
            include=query_kwargs["include"],
        )

    hits = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        meta = meta or {}
        heading = meta.get("heading", "")
        doc_type = meta.get("doc_type", "")
        lex = lexical_score(question, doc, heading, doc_type)
        semantic = max(0.0, 1.0 - float(distance))
        combined = 0.35 * semantic + 0.65 * (lex / 12.0)
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "lang": meta.get("lang", ""),
                "heading": heading,
                "doc_type": doc_type,
                "distance": float(distance),
                "lexical": lex,
                "score": combined,
            }
        )

    hits.sort(key=lambda hit: hit["score"], reverse=True)
    filtered = [hit for hit in hits if hit["distance"] <= max_distance and hit["lexical"] > 0]
    selected = (filtered or hits[:1])[:n_results]
    return selected


def ask_chat(question: str, excerpts: list[dict]) -> str:
    excerpt_block = "\n\n".join(
        f"Extrait {index} [{hit['source']} p.{hit['page']}]\n{hit['text']}"
        for index, hit in enumerate(excerpts, start=1)
    )
    user_prompt = (
        f"{excerpt_block}\n\n"
        f"Question de l'operateur : {question}\n\n"
        "Donne une reponse courte, en francais, sous forme d'etapes numerotees "
        "tirees uniquement des extraits ci-dessus."
    )
    response = requests.post(
        CHAT_URL,
        json={
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 220,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
