from __future__ import annotations

import argparse
import sys

from rag import HEALTH_CHAT, HEALTH_EMBED, ask_chat, search_chunks, wait_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose une question a l'assistant technique OPTIJET."
    )
    parser.add_argument("question", nargs="+", help="Question de l'operateur")
    parser.add_argument("--top", type=int, default=3, help="Nombre d'extraits a recuperer")
    args = parser.parse_args()
    question = " ".join(args.question).strip()

    print("Attente des serveurs ...")
    wait_for(HEALTH_EMBED, "llama-embed")
    wait_for(HEALTH_CHAT, "llama-chat")

    hits = search_chunks(question, n_results=args.top)
    print("\n--- Extraite(s) du manuel ---")
    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}] {hit['source']} p.{hit['page']} (distance={hit['distance']:.4f})")
        print(hit["text"])

    print("\n--- Reponse ---")
    answer = ask_chat(question, hits)
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
