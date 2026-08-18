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
chroma_db  (texte + vecteurs)
    │
ask.py  ──1. embed question──►  llama-embed :8081
    │
    ├──2. extraits les plus proches
    │
    └──3. POST /v1/chat/completions──►  llama-chat :8080  (Qwen 2.5 3B Instruct)
```

Deux serveurs distincts :

| Service | Port | Rôle | Modèle |
|---|---|---|---|
| `llama-embed` | 8081 | Transforme le texte en vecteurs | Snowflake Arctic Embed M v2.0 |
| `llama-chat` | 8080 | Rédige la réponse | Qwen 2.5 3B Instruct Q4_K_M |

Le GGUF Snowflake n’est pas compatible llama.cpp (`GteModel`). Il tourne via **embeddings.cpp**.

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

## Ingestion des manuels

Une fois les serveurs prêts :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ingest.py
```

Le script :

1. lit tous les `pdf/*.pdf`
2. découpe le texte en blocs de 300 à 500 caractères
3. envoie chaque bloc à `http://localhost:8081/v1/embeddings`
4. stocke `[vecteur, texte, source, page]` dans `chroma_db/`

Relancer `ingest.py` reconstruit la base.

## Poser une question

```powershell
python ask.py "Comment démarrer la machine OPTIJET ?"
```

Le script affiche les extraits trouvés, puis la réponse de Qwen.

Options :

```powershell
python ask.py --top 5 "Code erreur E04"
```

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

## Fichiers

| Fichier | Rôle |
|---|---|
| `docker-compose.yaml` | Services embed + chat |
| `Dockerfile` | Image embeddings.cpp |
| `ingest.py` | Indexation des PDF |
| `ask.py` | Question opérateur |
| `rag.py` | Client embeddings / Chroma / chat |
| `models/` | Fichiers GGUF |
| `pdf/` | Manuels OPTIJET |
| `chroma_db/` | Base vectorielle locale (générée) |
