from __future__ import annotations

import argparse
import sys

from rag import HEALTH_CHAT, HEALTH_EMBED, ask_chat, search_chunks, wait_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose une question a l'assistant technique OPTIJET."
    )
    parser.add_argument("question", nargs="+", help="Question de l'operateur")
    parser.add_argument("--top", type=int, default=5, help="Nombre d'extraits a recuperer")
    parser.add_argument("--lang", default="fr", choices=["fr", "en", "all"])
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.55,
        help="Ignore les extraits au-dessus de ce seuil cosinus",
    )
    args = parser.parse_args()
    question = " ".join(args.question).strip()

    print("Attente des serveurs ...")
    wait_for(HEALTH_EMBED, "llama-embed")
    wait_for(HEALTH_CHAT, "llama-chat")

    hits = search_chunks(
        question,
        n_results=args.top,
        lang=args.lang,
        max_distance=args.max_distance,
    )
    print("\n--- Extraite(s) du manuel ---")
    if not hits:
        print("Aucun extrait pertinent.")
        return
    for index, hit in enumerate(hits, start=1):
        print(
            f"\n[{index}] {hit['source']} p.{hit['page']} "
            f"(distance={hit['distance']:.4f}, lang={hit.get('lang', '')})"
        )
        print(hit["text"])

    print("\n--- Reponse ---")
    answer = ask_chat(question, hits)
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
