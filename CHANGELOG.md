# Changelog

Toutes les modifications notables de ce projet sont documentées ici.

Le format s'inspire de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Unreleased] — 2026-08-19

Refactorisation du RAG offline pour HMI industrielle Debian 8 Go RAM.
Priorité : fiabilité > qualité de recherche > traçabilité > performance > complexité.

### Added

- `config.json` : paramètres centralisés (chunking, poids de fusion, seuils, génération, logs).
- `synonyms.json` : synonymes et bonus/malus lexicaux éditables sans modifier le code.
- `settings.py` : chargement de la configuration et initialisation des logs.
- `evaluate.py` + `evaluation.json` : mesure Recall@1, Recall@3, Recall@5.
- Indexation incrémentale : `python ingest.py --update` (hash SHA-256 des PDF).
- Reconstruction complète : `python ingest.py --reset`.
- Rapport des pages scannées : `python ingest.py --report-ocr` (`ocr_needed.json`).
- OCR optionnel sur PC de préparation : `python ingest.py --ocr` (pytesseract/Pillow, non requis sur l'HMI).
- Métadonnées enrichies par chunk : `document_id`, `document_hash`, `source`, `page`, `start_page`, `end_page`, `lang`, `doc_type`, `heading`, `section_path`, `chunk_index`, `image_ids`, `document_version`.
- Liaison images ↔ chunks (identifiants page/image, légendes, `image_captions.json` hors ligne).
- Recherche hybride : exacte (codes, P/N, valeurs, sections) + lexicale + sémantique.
- Fusion de scores normalisés : `combine_scores()` avec `EXACT_WEIGHT`, `LEXICAL_WEIGHT`, `SEMANTIC_WEIGHT`.
- Pipeline extensible : `retrieve()` → `rerank()` (identité) → `select_context()`.
- Classification légère des questions (procédure, alarme, diagnostic, valeur, etc.).
- Sortie structurée pour l'HMI : `{ answer, confidence, sources }`.
- `python ask.py --json` et `python ask.py --debug`.
- `python ui.py` : fenêtre de chat locale (port 8090) branchée sur le RAG, pas sur l’UI llama.cpp.
- Logs fichier `logs/rag.log` : question, scores, décision de confiance, `embedding_time` / `retrieval_time` / `generation_time` / `total_time`.
- Filtre de langue strict : `fr`, `en`, `all` (plus de repli silencieux vers toutes les langues).

### Changed

- Le découpage n'est plus limité à une page : une section qui commence p.10 et continue p.11 reste une même section.
- Overlap de chunks configurable (`MIN_CHARS`, `MAX_CHARS`, `OVERLAP_CHARS`) ; plus de constantes magiques dispersées.
- `ingest.py` ne vide plus la base à chaque lancement (défaut : `--update`).
- `enrich_query()` est désactivé par défaut (`search.enrich_query` dans `config.json`).
- Prompt Qwen adapté au type de question : plus d'étapes numérotées forcées pour toute question.
- `temperature` reste à `0.0` ; `max_tokens` est configurable.
- Erreurs explicites : llama-server down, Chroma inaccessible, base vide, PDF corrompu, embedding/LLM timeout, JSON invalide.

### Fixed

- Suppression du repli `filtered or hits[:1]` qui envoyait un extrait non pertinent à Qwen et provoquait des hallucinations.
- Si aucun résultat ne dépasse `MIN_SEMANTIC_SCORE` / `MIN_FINAL_SCORE`, réponse « information non trouvée » **sans** appeler Qwen.
- Les extraits de dépannage ne sont plus mélangés à une question de fonctionnement normal.
- Titres coupés sur deux lignes (ex. `7.11.4`) correctement recombinés.
- Pertinence opérateur : `7.1 Mise en marche` n'était plus retrouvé (`enrich_query` off, stopwords absents, 4 procédures mélangées). `enrich_query` est réactivé par défaut ; une seule famille de section est envoyée à Qwen ; `etindre` est reconnu comme `eteindre`.

### Migration

- **Réindexer** après cette version (`python ingest.py --reset --lang fr`) : l'ancienne base n'a pas les nouvelles métadonnées.
- `ask_chat(question, excerpts)` retourne un **dict** au lieu d'une `str`. Utiliser `result["answer"]`.
- `search_chunks()` retourne une **liste vide** si rien ne dépasse le seuil, au lieu du meilleur hit par défaut.
