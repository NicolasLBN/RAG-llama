from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Any

import chromadb
import requests

from settings import (
    CFG,
    CHAT_URL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_URL,
    HEALTH_CHAT,
    HEALTH_EMBED,
    load_synonyms,
    setup_logging,
)

LOGGER = logging.getLogger("rag")

FETCH_K = int(CFG.get("search", "fetch_k", default=30))
FINAL_K = int(CFG.get("search", "final_k", default=4))
EXACT_WEIGHT = float(CFG.get("search", "exact_weight", default=0.40))
LEXICAL_WEIGHT = float(CFG.get("search", "lexical_weight", default=0.25))
SEMANTIC_WEIGHT = float(CFG.get("search", "semantic_weight", default=0.35))
MIN_SEMANTIC_SCORE = float(CFG.get("search", "min_semantic_score", default=0.28))
MIN_FINAL_SCORE = float(CFG.get("search", "min_final_score", default=0.32))
ENRICH_QUERY = bool(CFG.get("search", "enrich_query", default=False))
QUERY_PREFIX = str(CFG.get("embedding", "query_prefix", default="query: "))
TEMPERATURE = float(CFG.get("generation", "temperature", default=0.0))
MAX_TOKENS = int(CFG.get("generation", "max_tokens", default=280))
CHAT_TIMEOUT = int(CFG.get("generation", "timeout_s", default=180))

NOT_FOUND = {
    "fr": "Je n'ai pas trouve cette information dans le manuel.",
    "en": "I did not find this information in the manual.",
}

TECHNICAL_PATTERNS = [
    re.compile(r"\b[A-Z]{1,4}[-_]\d{2,6}\b"),
    re.compile(r"\b[A-Z]\d{2,4}\b"),
    re.compile(r"\bP/?N\s*[\d.-]+\b", re.I),
    re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:VDC|VAC|V\b|bars?|psi|N\b|mm\b|m/min|%)", re.I),
    re.compile(r"\b\d+(?:\.\d+){1,3}\b"),
]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.I)

QUESTION_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("alarm", ("alarme", "alarm", "code erreur", "message d'erreur", "defaut", "fault")),
    (
        "diagnostic",
        (
            "ne demarre pas",
            "depannage",
            "panne",
            "probleme",
            "troubleshooting",
            "causes",
            "resolution des problemes",
        ),
    ),
    ("maintenance", ("entretien", "maintenance", "nettoyage", "stockage", "graissage")),
    (
        "procedure",
        (
            "comment",
            "how to",
            "etape",
            "procedure",
            "demarrer",
            "allumer",
            "arreter",
            "remplacer",
            "monter",
            "connecter",
        ),
    ),
    ("value", ("combien", "quelle pression", "quelle force", "fmax", "valeur", "specification")),
    ("definition", ("qu'est-ce", "what is", "c'est quoi", "signification", "definir", "definition")),
    ("explanation", ("pourquoi", "why", "a quoi sert", "explain", "role", "explication")),
]


class RagError(Exception):
    pass


class ServiceUnavailable(RagError):
    pass


class EmptyIndexError(RagError):
    pass


class EmbeddingError(RagError):
    pass


class ChatError(RagError):
    pass


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
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
    raise ServiceUnavailable(f"{name} n'est pas pret sur {url}: {last_error}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = requests.post(EMBED_URL, json={"input": texts}, timeout=120)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise EmbeddingError(f"llama-embed indisponible ({EMBED_URL}): {exc}") from exc
    except ValueError as exc:
        raise EmbeddingError(f"Reponse embedding JSON invalide: {exc}") from exc
    try:
        items = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]
    except (KeyError, TypeError) as exc:
        raise EmbeddingError(f"Reponse embedding inattendue: {exc}") from exc


def _synonym_data() -> dict[str, Any]:
    return load_synonyms()


def query_terms(question: str) -> list[str]:
    folded = fold(question)
    terms = [folded]
    synonyms = _synonym_data().get("synonyms") or {}
    for key, extras in synonyms.items():
        extra_list = [fold(str(item)) for item in extras]
        if fold(str(key)) in folded or any(extra in folded for extra in extra_list):
            terms.extend(extra_list)
            terms.append(fold(str(key)))
    return list(dict.fromkeys(term for term in terms if term))


def extract_technical_terms(question: str) -> list[str]:
    found: list[str] = []
    for pattern in TECHNICAL_PATTERNS:
        for match in pattern.findall(question):
            found.append(match.strip())
    return list(dict.fromkeys(found))


def has_term(blob: str, term: str) -> bool:
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(fold(term)) + r"(?![a-z0-9])"
    return re.search(pattern, blob) is not None


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(fold(text)) if len(token) >= 3]


def enrich_query(question: str) -> str:
    body = question.strip()
    if ENRICH_QUERY:
        extras = [term for term in query_terms(question) if term not in fold(question)]
        suffix = " ".join(extras[:6])
        if suffix:
            body = f"{body} {suffix}"
    return f"{QUERY_PREFIX}{body}".strip()


def embed_query(question: str) -> list[float]:
    vectors = embed_texts([enrich_query(question)])
    if not vectors:
        raise EmbeddingError("Embedding de la question vide")
    return vectors[0]


def classify_question(question: str) -> str:
    folded = fold(question)
    if extract_technical_terms(question) and any(
        token in folded for token in ("erreur", "alarme", "alarm", "code", "defaut")
    ):
        return "alarm"
    for qtype, needles in QUESTION_RULES:
        if any(needle in folded for needle in needles):
            return qtype
    if extract_technical_terms(question):
        return "technical"
    return "explanation"


def lexical_score(question: str, text: str, heading: str, doc_type: str) -> float:
    blob = fold(f"{heading} {text}")
    heading_blob = fold(heading)
    score = 0.0
    terms = list(dict.fromkeys(tokenize(question) + query_terms(question)))
    for term in terms:
        if not has_term(blob, term) and fold(term) not in blob:
            continue
        score += 1.5 if has_term(heading_blob, term) or fold(term) in heading_blob else 1.0
    data = _synonym_data()
    qfold = fold(question)
    for phrase in data.get("phrase_bonus") or []:
        if fold(phrase) in blob and fold(phrase) in qfold:
            score += 2.0
    diagnostic_question = classify_question(question) in {"diagnostic", "alarm"}
    if not diagnostic_question:
        for phrase in data.get("phrase_penalty") or []:
            if fold(phrase) in blob:
                score -= 3.0
    if doc_type == "quick_guide" and classify_question(question) == "procedure":
        score += 0.8
    return score


def exact_score(question: str, text: str, heading: str) -> float:
    terms = extract_technical_terms(question)
    if not terms:
        return 0.0
    blob = fold(f"{heading} {text}")
    hits = sum(1 for term in terms if has_term(blob, term) or fold(term) in blob)
    return hits / len(terms)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - low) / (high - low) for value in values]


def combine_scores(
    exact: float,
    lexical_norm: float,
    semantic: float,
    *,
    has_technical_terms: bool,
    exact_weight: float = EXACT_WEIGHT,
    lexical_weight: float = LEXICAL_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> float:
    """Fusionne trois signaux deja ramènes dans [0, 1].

    - exact: fraction des termes techniques (E102, 16 bar, 7.1...) trouves
      avec une frontiere de mot.
    - lexical_norm: score lexical min-max normalise sur le lot candidat.
    - semantic: 1 - distance cosinus renvoyee par Chroma.

    Les poids sont renormalises pour sommer a 1. S'il n'y a aucun terme
    technique dans la question, le poids exact est mis a 0 afin de ne pas
    fausser le classement.
    """
    if not has_technical_terms:
        exact_weight = 0.0
    weights = [max(0.0, exact_weight), max(0.0, lexical_weight), max(0.0, semantic_weight)]
    total = sum(weights) or 1.0
    weights = [weight / total for weight in weights]
    return (
        weights[0] * max(0.0, min(1.0, exact))
        + weights[1] * max(0.0, min(1.0, lexical_norm))
        + weights[2] * max(0.0, min(1.0, semantic))
    )


def open_collection(reset: bool = False):
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        if reset:
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception as exc:
                LOGGER.info("Pas de collection a supprimer: %s", exc)
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise RagError(f"ChromaDB inaccessible ({CHROMA_DIR}): {exc}") from exc


def _parse_image_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if not raw:
        return []
    return [item for item in str(raw).split(",") if item]


def _hit_from_row(doc: str, meta: dict[str, Any], distance: float) -> dict[str, Any]:
    meta = meta or {}
    start_page = int(meta.get("start_page") or meta.get("page") or 0)
    end_page = int(meta.get("end_page") or start_page)
    return {
        "id": meta.get("id", ""),
        "text": doc or "",
        "source": meta.get("source", ""),
        "page": start_page,
        "start_page": start_page,
        "end_page": end_page,
        "lang": meta.get("lang", ""),
        "heading": meta.get("heading", ""),
        "section_path": meta.get("section_path", ""),
        "doc_type": meta.get("doc_type", ""),
        "chunk_index": int(meta.get("chunk_index") or 0),
        "image_ids": _parse_image_ids(meta.get("image_ids")),
        "document_id": meta.get("document_id", ""),
        "document_version": meta.get("document_version", ""),
        "document_hash": meta.get("document_hash", ""),
        "distance": float(distance),
        "semantic": max(0.0, 1.0 - float(distance)),
    }


def retrieve(
    question: str,
    lang: str = "fr",
    fetch_k: int = FETCH_K,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Recherche vectorielle elargie + rappel exact des termes techniques."""
    collection = open_collection()
    if collection.count() == 0:
        raise EmptyIndexError("La base est vide. Lance d'abord: python ingest.py --update")

    where = {"lang": lang} if lang and lang != "all" else None
    candidate_ids: list[str] = []
    if where:
        try:
            peek = collection.get(where=where, include=["metadatas"])
            candidate_ids = peek.get("ids") or []
        except Exception as exc:
            LOGGER.error("Filtre langue Chroma impossible (%s): %s", lang, exc)
            raise RagError(f"Filtre de langue '{lang}' impossible: {exc}") from exc
        if not candidate_ids:
            LOGGER.warning("Aucun chunk pour lang=%s", lang)
            return [], {"embedding_time": 0.0, "retrieval_time": 0.0}

    embed_started = time.perf_counter()
    query_embedding = embed_query(question)
    embedding_time = time.perf_counter() - embed_started

    search_started = time.perf_counter()
    n_results = min(fetch_k, len(candidate_ids) if candidate_ids else collection.count())
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": max(1, n_results),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        result = collection.query(**query_kwargs)
    except Exception as exc:
        LOGGER.error("Requete Chroma echouee: %s", exc)
        raise RagError(f"Recherche Chroma impossible: {exc}") from exc

    hits_by_key: dict[str, dict[str, Any]] = {}

    def ingest_result(payload: dict[str, Any]) -> None:
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        ids = (payload.get("ids") or [[]])[0]
        for row_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            hit = _hit_from_row(doc, meta or {}, distance)
            hit["id"] = row_id
            hits_by_key[row_id] = hit

    ingest_result(result)

    for term in extract_technical_terms(question):
        extra_kwargs = dict(query_kwargs)
        extra_kwargs["n_results"] = min(10, extra_kwargs["n_results"])
        extra_kwargs["where_document"] = {"$contains": term}
        try:
            extra = collection.query(**extra_kwargs)
            ingest_result(extra)
        except Exception as exc:
            LOGGER.debug("Rappel exact ignore pour %s: %s", term, exc)

    retrieval_time = time.perf_counter() - search_started
    return list(hits_by_key.values()), {
        "embedding_time": embedding_time,
        "retrieval_time": retrieval_time,
    }


def rerank(hits: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """Point d'extension pour un futur cross-encoder. Identite pour l'instant."""
    return hits


def select_context(
    hits: list[dict[str, Any]],
    n_results: int = FINAL_K,
    min_semantic: float = MIN_SEMANTIC_SCORE,
    min_final: float = MIN_FINAL_SCORE,
    question: str = "",
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    has_terms = bool(extract_technical_terms(question))
    for hit in hits:
        semantic_ok = hit.get("semantic", 0.0) >= min_semantic
        final_ok = hit.get("score", 0.0) >= min_final
        exact_ok = has_terms and hit.get("exact", 0.0) >= 0.5
        if (semantic_ok and final_ok) or exact_ok:
            selected.append(hit)
        if len(selected) >= n_results:
            break
    LOGGER.debug(
        "select_context: %s/%s retenus (min_semantic=%.2f min_final=%.2f)",
        len(selected),
        len(hits),
        min_semantic,
        min_final,
    )
    return selected


def _score_hits(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_terms = bool(extract_technical_terms(question))
    for hit in hits:
        hit["lexical"] = lexical_score(
            question,
            hit.get("text", ""),
            hit.get("heading", ""),
            hit.get("doc_type", ""),
        )
        hit["exact"] = exact_score(question, hit.get("text", ""), hit.get("heading", ""))
    lexical_norms = _normalize([float(hit["lexical"]) for hit in hits])
    for hit, lexical_norm in zip(hits, lexical_norms):
        hit["lexical_norm"] = lexical_norm
        hit["score"] = combine_scores(
            float(hit["exact"]),
            lexical_norm,
            float(hit.get("semantic", 0.0)),
            has_technical_terms=has_terms,
        )
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits


def search_chunks(
    question: str,
    n_results: int | None = None,
    lang: str = "fr",
    max_distance: float | None = None,
    fetch_k: int | None = None,
    apply_threshold: bool = True,
) -> list[dict]:
    """API publique: retrieve -> score -> rerank -> select_context.

    Retourne une liste vide si aucun extrait ne depasse les seuils.
    Ne rabat plus sur hits[:1] (cause d'hallucinations).
    apply_threshold=False conserve le classement brut (evaluation Recall@k).
    """
    setup_logging()
    final_k = n_results if n_results is not None else FINAL_K
    fetch = fetch_k if fetch_k is not None else FETCH_K
    min_semantic = MIN_SEMANTIC_SCORE
    if max_distance is not None:
        min_semantic = max(0.0, 1.0 - float(max_distance))

    hits, search_timings = retrieve(question, lang=lang, fetch_k=fetch)
    scored = _score_hits(question, hits)
    ranked = rerank(scored, question)
    if apply_threshold:
        selected = select_context(
            ranked,
            n_results=final_k,
            min_semantic=min_semantic,
            min_final=MIN_FINAL_SCORE,
            question=question,
        )
    else:
        selected = ranked[:final_k]
    LOGGER.debug(
        "question=%r lang=%s type=%s enrich=%s fetch=%s selected=%s "
        "embedding=%.3fs retrieval=%.3fs",
        question,
        lang,
        classify_question(question),
        ENRICH_QUERY,
        len(scored),
        len(selected),
        search_timings.get("embedding_time", 0.0),
        search_timings.get("retrieval_time", 0.0),
    )
    for index, hit in enumerate(scored[:fetch]):
        LOGGER.debug(
            "hit[%s] score=%.3f sem=%.3f lex=%.2f exact=%.2f dist=%.3f %s p.%s %s",
            index,
            hit.get("score", 0),
            hit.get("semantic", 0),
            hit.get("lexical", 0),
            hit.get("exact", 0),
            hit.get("distance", 0),
            hit.get("source", ""),
            hit.get("page", 0),
            hit.get("heading", ""),
        )
    return selected


def _sources_from_excerpts(excerpts: list[dict]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for hit in excerpts:
        source = {
            "source": hit.get("source", ""),
            "page": hit.get("page") or hit.get("start_page") or 0,
            "start_page": hit.get("start_page") or hit.get("page") or 0,
            "end_page": hit.get("end_page") or hit.get("page") or 0,
            "heading": hit.get("heading", ""),
            "section_path": hit.get("section_path", ""),
            "image_ids": hit.get("image_ids") or [],
        }
        key = (source["source"], int(source["page"]), source["heading"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return sources


def _system_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "You are the technical assistant for the OptiJet machine. "
            "Use ONLY the provided manual excerpts. Do not invent anything. "
            "Never complete a procedure with general knowledge. "
            "Keep numeric values and units exactly as written. "
            "Do not mix two different procedures. "
            "Do not use a troubleshooting excerpt to answer a normal operating question. "
            "If the information is absent, answer exactly: "
            f"{NOT_FOUND['en']} "
            "Stay concise and suited to an operator."
        )
    return (
        "Tu es l'assistant technique de la machine OptiJet. "
        "Tu utilises UNIQUEMENT les extraits du manuel fournis. "
        "Tu n'inventes rien. Tu ne completes jamais une procedure avec tes "
        "connaissances generales. "
        "Tu respectes les valeurs numeriques et unites telles qu'ecrites. "
        "Tu ne melanges pas deux procedures differentes. "
        "Tu n'utilises pas une procedure de depannage pour une question de "
        "fonctionnement normal. "
        "Si l'information est absente, tu reponds exactement : "
        f"{NOT_FOUND['fr']} "
        "Tu restes concis et adapte a un operateur."
    )


def _format_instruction(question_type: str, lang: str) -> str:
    if lang == "en":
        mapping = {
            "procedure": "Answer with short numbered steps taken only from the excerpts.",
            "maintenance": "Answer with short numbered maintenance steps taken only from the excerpts.",
            "alarm": "Give the meaning, then cause/procedure only if present in the excerpts.",
            "diagnostic": "Give likely causes and remedies only if present in the excerpts.",
            "value": "Report the numeric value and unit exactly as written.",
            "definition": "Give a short definition using only the excerpts.",
            "technical": "Give the requested technical information using only the excerpts.",
            "explanation": "Give a short descriptive answer using only the excerpts.",
        }
    else:
        mapping = {
            "procedure": "Reponds par des etapes numerotees courtes tirees uniquement des extraits.",
            "maintenance": "Reponds par des etapes d'entretien numerotees, uniquement d'apres les extraits.",
            "alarm": "Donne la signification, puis cause/procedure uniquement si elles figurent dans les extraits.",
            "diagnostic": "Donne les causes et remedes uniquement s'ils figurent dans les extraits.",
            "value": "Donne la valeur numerique et l'unite exactement telles qu'ecrites.",
            "definition": "Donne une definition courte uniquement d'apres les extraits.",
            "technical": "Donne l'information technique demandee uniquement d'apres les extraits.",
            "explanation": "Donne une reponse descriptive courte uniquement d'apres les extraits.",
        }
    return mapping.get(question_type, mapping["explanation"])


def ask_chat(
    question: str,
    excerpts: list[dict],
    question_type: str | None = None,
    lang: str = "fr",
) -> dict[str, Any]:
    """Appelle Qwen et retourne {answer, sources, confidence}.

    Ancienne signature: ask_chat(question, excerpts) -> str
    Nouvelle valeur de retour: dict. Lire result['answer'] pour le texte.
    """
    setup_logging()
    qtype = question_type or classify_question(question)
    confidence = max((float(hit.get("score") or 0.0) for hit in excerpts), default=0.0)
    sources = _sources_from_excerpts(excerpts)
    if not excerpts:
        return {
            "answer": NOT_FOUND.get(lang, NOT_FOUND["fr"]),
            "confidence": 0.0,
            "sources": [],
            "question_type": qtype,
        }

    excerpt_block = "\n\n".join(
        (
            f"Extrait {index} [{hit.get('source', '')} "
            f"p.{hit.get('start_page') or hit.get('page')}"
            f"-{hit.get('end_page') or hit.get('page')} "
            f"| {hit.get('heading', '')}]\n{hit.get('text', '')}"
        )
        for index, hit in enumerate(excerpts, start=1)
    )
    language_line = (
        "Answer in English."
        if lang == "en"
        else "Reponds en francais."
        if lang == "fr"
        else "Answer in the same language as the operator question."
    )
    user_prompt = (
        f"{excerpt_block}\n\n"
        f"Type de question: {qtype}\n"
        f"Question de l'operateur: {question}\n\n"
        f"{language_line} {_format_instruction(qtype, lang)}"
    )
    started = time.perf_counter()
    try:
        response = requests.post(
            CHAT_URL,
            json={
                "model": "qwen",
                "messages": [
                    {"role": "system", "content": _system_prompt(lang if lang != "all" else "fr")},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            },
            timeout=CHAT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"].strip()
    except requests.RequestException as exc:
        raise ChatError(f"llama-chat indisponible ({CHAT_URL}): {exc}") from exc
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ChatError(f"Reponse LLM JSON invalide: {exc}") from exc
    generation_time = time.perf_counter() - started
    LOGGER.debug("generation_time=%.3fs type=%s confidence=%.3f", generation_time, qtype, confidence)
    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "sources": sources,
        "question_type": qtype,
        "generation_time": generation_time,
    }


def answer_question(
    question: str,
    lang: str = "fr",
    n_results: int | None = None,
    max_distance: float | None = None,
) -> dict[str, Any]:
    """Pipeline HMI: recherche, seuil de confiance, puis generation eventuelle."""
    setup_logging()
    started = time.perf_counter()
    qtype = classify_question(question)
    final_k = n_results if n_results is not None else FINAL_K
    min_semantic = MIN_SEMANTIC_SCORE
    if max_distance is not None:
        min_semantic = max(0.0, 1.0 - float(max_distance))

    hits, search_timings = retrieve(question, lang=lang, fetch_k=FETCH_K)
    scored = _score_hits(question, hits)
    ranked = rerank(scored, question)
    selected = select_context(
        ranked,
        n_results=final_k,
        min_semantic=min_semantic,
        min_final=MIN_FINAL_SCORE,
        question=question,
    )
    LOGGER.debug(
        "decision confiance: %s extraits retenus / %s candidats",
        len(selected),
        len(ranked),
    )
    if not selected:
        LOGGER.info("Aucun extrait au-dessus du seuil pour %r", question)
        return {
            "answer": NOT_FOUND.get(lang, NOT_FOUND["fr"]),
            "confidence": 0.0,
            "sources": [],
            "question_type": qtype,
            "hits": [],
            "timings": {
                "embedding_time": search_timings.get("embedding_time", 0.0),
                "retrieval_time": search_timings.get("retrieval_time", 0.0),
                "generation_time": 0.0,
                "total_time": time.perf_counter() - started,
            },
        }
    chat = ask_chat(question, selected, question_type=qtype, lang=lang)
    chat["hits"] = selected
    chat["timings"] = {
        "embedding_time": search_timings.get("embedding_time", 0.0),
        "retrieval_time": search_timings.get("retrieval_time", 0.0),
        "generation_time": chat.get("generation_time", 0.0),
        "total_time": time.perf_counter() - started,
    }
    return chat
