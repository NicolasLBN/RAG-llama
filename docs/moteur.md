# ingest.py, rag.py et evaluate.py

English version: [engine.md](engine.md) (includes a glossary).

Ces trois fichiers forment le cœur documentaire du RAG OPTIJET. Ils ne chargent pas les modèles GGUF : ils parlent aux serveurs Docker (`llama-embed` :8081, `llama-chat` :8080) et à la base locale `chroma_db/`.

```
pdf/*.pdf
    │
    ▼
ingest.py  ── embeddings ──►  chroma_db  (texte + vecteurs + métadonnées)
    │
    │                         rag.py
    │                           │
    │              retrieve → scores → Qwen
    │                           │
    ├───────────────────────────┼── ask.py / ui.py  (opérateur)
    └───────────────────────────┴── evaluate.py     (qualité de recherche)
```

Les constantes (tailles de chunks, poids, seuils) viennent de `settings.py` et de `config.json`.

---

## Vecteur et embedding : c’est la même chose ?

**Presque.** Ici, les deux mots désignent le **même objet**.

Un **vecteur** est une liste de nombres, par exemple `[0.12, -0.45, 0.88, …]`.

Un **embedding** est un vecteur qui **représente le sens d’un texte**. Le modèle Snowflake (port 8081) **transforme une phrase en embedding**. On le stocke dans Chroma et on compare les embeddings : deux textes au même sens ont des vecteurs proches, même si les mots changent (« démarrer » / « mise en marche »).

Qwen, lui, lit le **texte brut** du manuel, pas les nombres. Le vecteur sert uniquement à **retrouver** le bon paragraphe.

Glossaire plus complet (anglais) : [engine.md](engine.md).

---

## ingest.py — préparer la mémoire documentaire

**Rôle.** Lire les manuels PDF, les découper en blocs (chunks), calculer un vecteur pour chaque bloc, les enregistrer dans ChromaDB. C’est un travail de **PC de préparation**, pas de l’HMI.

**Quand l’utiliser.** À chaque nouveau PDF, après une modification de manuel, ou après un changement de chunking. `llama-embed` (:8081) doit tourner. Qwen n’est pas nécessaire.

### Pipeline d’un PDF

1. **Identification** — à partir du nom de fichier (`298.410-fr-v1.5.0 …`) : `document_id`, langue (`fr` / `en`), version, type (`manual`, `quick_guide`, `profiling`).
2. **Hash SHA-256** du fichier — pour savoir s’il a changé (`--update`).
3. **Pages** — extraction du texte natif (PyMuPDF). Pages trop pauvres en texte + images : marquées « OCR » (`ocr_needed.json`). L’OCR réel (`--ocr`) n’est lancé que si Tesseract est installé.
4. **Images** — identifiants `document-p{page}-img{n}` dans les métadonnées (`image_ids`). Pas de modèle multimodal. Des légendes hors ligne peuvent être injectées via `image_captions.json`.
5. **Structure** — titres numérotés (1, 1.1, 7.1…), NOTE / DANGER / IMPORTANT. Une section peut **traverser plusieurs pages**.
6. **Chunking** — `MIN_CHARS` / `MAX_CHARS` / `OVERLAP_CHARS` (`config.json`). Chaque chunk répète le `section_path`.
7. **Embedding** — lots vers `POST http://localhost:8081/v1/embeddings`.
8. **Stockage** — Chroma (`chroma_db/`) + manifeste `chroma_db/manifest.json`.

Métadonnées typiques d’un chunk : `source`, `page`, `start_page`, `end_page`, `lang`, `heading`, `section_path`, `chunk_index`, `image_ids`, `document_hash`, `document_version`.

### Commandes

```powershell
python ingest.py --update --lang fr      # défaut : PDF nouveaux ou hash changé
python ingest.py --update --lang en
python ingest.py --update --lang all     # FR + EN dans la même base
python ingest.py --reset --lang all      # vide Chroma puis réindexe tout
python ingest.py --report-ocr --lang all # liste les pages scannées, n'indexe pas
python ingest.py --ocr --update --lang fr
```

`--reset` sans `--lang all` reconstruit la collection puis n’y remet que la langue demandée : les autres langues disparaissent.

`ingest.py` importe `embed_texts` et `open_collection` depuis `rag.py`.

---

## rag.py — chercher et faire rédiger

**Rôle.** Bibliothèque utilisée par `ask.py`, `ui.py`, `ingest.py` et `evaluate.py`. Elle ne s’exécute pas toute seule (`python rag.py` ne fait rien d’utile).

Deux phases à l’exploitation :

1. **Recherche** dans Chroma (rapide).
2. **Génération** avec Qwen si des extraits sont assez pertinents (lent, ~2 Go RAM).

### Fonctions publiques

| Fonction | Rôle |
|---|---|
| `embed_texts(texts)` | POST embeddings :8081 |
| `open_collection(reset=False)` | Ouvre / recrée la collection Chroma `manuals` |
| `search_chunks(question, …)` | Recherche hybride, liste d’extraits (ou `[]`) |
| `ask_chat(question, excerpts)` | Appelle Qwen, retourne `{answer, sources, confidence}` |
| `answer_question(question, lang=…)` | Pipeline HMI complet |

### Pipeline d’une question (`answer_question`)

1. **Filtre de langue** — `lang=fr` / `en` / `all`. Pas de repli silencieux vers l’autre langue.
2. **`enrich_query`** — ajoute des synonymes (`synonyms.json`) à l’embedding, préfixe `query: `.
3. **`retrieve`** — top `FETCH_K` voisins cosinus + rappel exact des codes (E102, 16 bar, 7.1…).
4. **Scores** — exact + lexical + sémantique, fusionnés par `combine_scores()` (poids dans `config.json`).
5. **`rerank`** — aujourd’hui identité (place pour un futur modèle).
6. **`select_context`** — seuils `MIN_SEMANTIC_SCORE` / `MIN_FINAL_SCORE`. Si rien ne passe : **Qwen n’est pas appelé**. Sinon, extraits de la **même famille de section** (évite de mélanger « mise en marche » et « pose de câble »).
7. **`ask_chat`** — prompt selon le type de question (procédure, alarme, valeur…). `temperature = 0.0`.

`ask_chat` retourne un **dict**, pas une chaîne : `result["answer"]`, `result["sources"]`, `result["confidence"]`.

Les logs détaillés vont dans `logs/rag.log` (`--debug` côté `ask.py` / `evaluate.py`), pas à l’opérateur.

---

## evaluate.py — mesurer la recherche

**Rôle.** Tester **uniquement le moteur de recherche** (pas Qwen). Pour chaque question de `evaluation.json`, il appelle `search_chunks(..., apply_threshold=False)` et vérifie si le bon passage est dans le top 1 / 3 / 5.

**Quand l’utiliser.** Après un changement de chunking, de synonymes ou de poids, pour voir si Recall@k monte ou descend. `llama-embed` doit tourner ; le chat Qwen n’est pas requis.

### Fichier `evaluation.json`

Chaque cas :

```json
{
  "question": "Comment demarrer la machine OPTIJET ?",
  "expected_source": "298.410-fr",
  "expected_page": 16,
  "expected_heading": "7.1 Mise en marche",
  "lang": "fr"
}
```

Un cas est réussi si un hit a :

- un `source` qui contient `expected_source` ;
- une page dans `[start_page, end_page]` ;
- un `heading` / `section_path` qui contient `expected_heading`.

### Métriques

- **Recall@1** — le 1er extrait est le bon.
- **Recall@3** / **Recall@5** — le bon est parmi les 3 / 5 premiers.

Affichage : `OK` (Recall@1), `~` (trouvé plus bas), `KO` (absent).

### Commandes

```powershell
python evaluate.py
python evaluate.py --file evaluation.json --lang fr
python evaluate.py --lang en --debug
```

Ajouter des questions réelles d’opérateur dans `evaluation.json` pour que le score reflète le terrain.

---

## Enchaînement type

| Étape | Script | Serveurs |
|---|---|---|
| 1. Indexer les PDF | `ingest.py` | embed :8081 |
| 2. Mesurer la recherche | `evaluate.py` | embed :8081 |
| 3. Question opérateur | `ask.py` ou `ui.py` | embed :8081 + chat :8080 |

`rag.py` est utilisé aux trois étapes.
