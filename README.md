# llama-embedded

Assistant technique local pour les manuels OPTIJET. Le système cherche le passage pertinent dans les PDF, puis rédige une réponse courte avec un modèle de chat.

Cible : machine HMI avec ~8 Go de RAM, CPU uniquement.

## Architecture

```
PDF manuels
│
▼
ingest.py  ──POST /v1/embeddings──►  llama-embed :8081  (Snowflake Arctic Embed)
│
▼
chroma_db  (texte + vecteurs + métadonnées)
│
ask.py  ──1. embed question──►  llama-embed :8081
│
├──2. extraits (hybride exact + lexical + sémantique)
│
└──3. POST /v1/chat/completions──►  llama-chat :8080  (Qwen 2.5 3B Instruct)
```

Deux serveurs distincts :

| Service | Port | Rôle | Modèle |
|---|---|---|---|
| `llama-embed` | 8081 | Transforme le texte en vecteurs | Snowflake Arctic Embed M v2.0 |
| `llama-chat` | 8080 | Rédige la réponse | Qwen 2.5 3B Instruct Q4_K_M |

Le GGUF Snowflake n’est pas compatible llama.cpp (`GteModel`). Il tourne via **embeddings.cpp**.

L’OCR et l’indexation se font sur un **PC de préparation**. L’HMI n’exécute que la recherche et Qwen, 100 % offline.

## Prérequis

- Docker Desktop
- Python 3.11+
- Fichiers modèles dans `models/` :
  - `snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`
  - `qwen2.5-3b-instruct-q4_k_m.gguf` ([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/qwen2.5-3b-instruct-q4_k_m.gguf))
- Manuels PDF dans `pdf/`

## Démarrage

```powershell
docker compose up --build -d
```

Le premier build de `llama-embed` compile embeddings.cpp (plusieurs minutes). Les suivants réutilisent l’image.

Santé des services :

```powershell
curl http://localhost:8081/health
curl http://localhost:8080/health
```

Attends que Qwen ait fini de charger (~2 Go en RAM) avant d’interroger le chat.

## Environnement Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Indexation des manuels (PC de préparation)

Mise à jour incrémentale (défaut) : n’indexe que les PDF nouveaux ou dont le hash a changé.

```powershell
python ingest.py --update --lang fr
```

Reconstruction complète :

```powershell
python ingest.py --reset --lang fr
```

Toutes les langues :

```powershell
python ingest.py --update --lang all
```

Pages scannées / OCR (jamais sur l’HMI) :

```powershell
python ingest.py --report-ocr --lang all
python ingest.py --ocr --update --lang fr
```

`--report-ocr` écrit `ocr_needed.json` sans indexer. `--ocr` tente Tesseract si `pytesseract` et `Pillow` sont installés.

Le script :

1. détecte texte natif vs page scannée
2. conserve titres, pages, images et légendes
3. découpe par section (un titre peut traverser plusieurs pages)
4. envoie chaque bloc à `http://localhost:8081/v1/embeddings`
5. stocke vecteur + texte + métadonnées dans `chroma_db/`

Paramètres de découpe : `config.json` (`MIN_CHARS`, `MAX_CHARS`, `OVERLAP_CHARS`).

Descriptions d’images hors ligne (optionnel) : `image_captions.json` au format `{"298.410-p16-img1": "Pupitre de commande"}`.

## Interface chat (RAG)

L’UI llama.cpp sur le port **8080** parle directement à Qwen, **sans** les manuels. Pour l’opérateur, lancer l’UI qui appelle le pipeline RAG :

```powershell
.\.venv\Scripts\Activate.ps1
python ui.py
```

Ouvrir http://127.0.0.1:8090 — les questions passent par `answer_question()` (même logique que `ask.py`), avec sources sous la réponse.

## Poser une question (ligne de commande)

```powershell
python ask.py "Comment démarrer la machine OPTIJET ?"
```

Sortie JSON pour l’HMI (`answer`, `confidence`, `sources`) :

```powershell
python ask.py --json "Code erreur E04"
```

Options :

```powershell
python ask.py --lang fr --top 4 "Comment remplacer les batteries ?"
python ask.py --debug "Quelle est la pression maximale ?"
```

`--debug` écrit le diagnostic dans `logs/rag.log` (pas à l’opérateur). Si aucun extrait ne dépasse le seuil de confiance, Qwen n’est pas appelé.

## Évaluation du moteur de recherche

```powershell
python evaluate.py
python evaluate.py --file evaluation.json --lang fr
```

Calcule Recall@1, Recall@3 et Recall@5. Ajouter des cas dans `evaluation.json` :

```json
{
  "question": "Comment démarrer la machine OPTIJET ?",
  "expected_source": "298.410-fr",
  "expected_page": 16,
  "expected_heading": "7.1 Mise en marche"
}
```

## Configuration

| Fichier | Rôle |
|---|---|
| `config.json` | Seuils, poids, tailles de chunks, `max_tokens`, logs |
| `synonyms.json` | Synonymes et bonus/malus lexicaux |

`search.enrich_query` est désactivé par défaut : l’ajout de synonymes dans l’embedding n’améliore pas toujours la recherche. Les synonymes restent utilisés pour le score lexical.

Poids de fusion (`combine_scores`) : `exact_weight`, `lexical_weight`, `semantic_weight`. Seuils : `min_semantic_score`, `min_final_score`.

## Tests manuels des API

Embeddings :

```powershell
curl -X POST http://localhost:8081/v1/embeddings -H "Content-Type: application/json" -d "{\"input\": [\"query: comment démarrer la machine\"]}"
```

Chat :

```powershell
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Explique OPTIJET en une phrase\"}]}"
```

## Mémoire et perf

Sur une machine 8 Go, viser ~2.5–3 Go au total :

- embeddings : ~200 Mo
- Qwen Q4_K_M, contexte 2048 : ~2 Go
- Python + Chroma : quelques centaines de Mo

Ajuster `-t` (threads CPU) dans `docker-compose.yaml` si les réponses dépassent 2–4 s.

```powershell
docker stats
```

`--debug` sépare `embedding_time`, `retrieval_time`, `generation_time`, `total_time`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `docker-compose.yaml` | Services embed + chat |
| `Dockerfile` | Image embeddings.cpp |
| `config.json` | Paramètres ajustables |
| `synonyms.json` | Synonymes |
| `ingest.py` | Indexation des PDF |
| `ask.py` | Question opérateur |
| `rag.py` | Embeddings / Chroma / chat |
| `evaluate.py` | Recall@k |
| `evaluation.json` | Jeu de test |
| `models/` | Fichiers GGUF |
| `pdf/` | Manuels OPTIJET |
| `chroma_db/` | Base vectorielle locale (générée) |
