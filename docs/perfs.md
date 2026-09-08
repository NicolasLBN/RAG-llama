# Performances — pistes restantes

Mesurer avant d’optimiser :

```powershell
python ask.py --debug --lang fr "Comment demarrer la machine OPTIJET ?"
```

Les temps `embedding` / `retrieval` (~0,15 s) sont déjà négligeables. Le goulot est **Qwen** (`generation_time`) : préremplissage du prompt (~8–12 s sur CPU) puis génération des tokens.

Déjà en place : `max_tokens` et `final_k` selon le type de question, prompts courts, 6 threads Qwen / 2 threads embed, flash-attn, batch 512, `--parallel 1`.

---

## Impact élevé

**Streaming vers l’HMI**  
Le temps total ne baisse pas, mais la première phrase s’affiche en 2–3 s. llama-server gère `stream: true` ; il faudrait le brancher dans `ui.py` / `ask.py`.

**Cache du prompt système**  
Le préambule est identique à chaque question. Activer le cache de préfixe llama.cpp (souvent `cache_prompt` / reuse du slot) pour ne pas re-préremplir les ~200–400 tokens d’instructions.

**Raccourci extractif pour les valeurs**  
Si le type est `value` et que l’extrait contient clairement `2–3 bars` / `16 bar`, renvoyer le nombre **sans** appeler Qwen (~0,1 s). Garder Qwen pour les procédures et le dépannage.

**Modèle plus petit**  
Qwen2.5-**1.5B** Q4, ou 3B en **Q4_0**, est plus rapide sur CPU. Qualité un peu moindre, souvent suffisante si le RAG envoie le bon extrait.

**GPU**  
Si un GPU apparaît sur le PC de test : `-ngl 99` dans `docker-compose.yaml`. Ordre de grandeur : 20–50 tok/s au lieu de 4–10. L’HMI Debian 8 Go reste en CPU.

## Impact moyen

**llama-server natif** (sans Docker)  
Sur Debian, le binaire `llama-server` hôte gagne parfois 10–30 % (pas de couche VM/cgroup). Docker reste plus simple à déployer.

**Threads HMI**  
Dans `docker-compose.yaml`, `-t` Qwen = cœurs physiques − 1 (souvent 7 sur 8 cœurs). Ne pas dépasser le nombre de cœurs : trop de threads ralentit.

**Contexte plus petit**  
`-c 2048` est déjà bas. Si les extraits tiennent, `-c 1024` réduit la RAM et un peu le préremplissage.

**KV cache quantifié**  
`--cache-type-k q8_0` / `--cache-type-v q8_0` : moins de RAM, parfois plus rapide. À tester ; incompatibilité possible avec flash-attn selon le build.

## Impact faible / à éviter en premier

- Baisser `temperature` : déjà à `0.0`.
- `evaluate.py` : ne mesure que la recherche, pas Qwen.
- Augmenter `fetch_k` : n’accélère pas la génération.
- Un meilleur ranking : change la **page** trouvée, pas la vitesse de decode.

---

## Mémoire (cible 8 Go)

Viser ~2,5–3 Go au total :

- embeddings : ~200 Mo
- Qwen Q4_K_M, contexte 2048 : ~2 Go
- Python + Chroma : quelques centaines de Mo

```powershell
docker stats
```
