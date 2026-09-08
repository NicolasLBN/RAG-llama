from __future__ import annotations

import argparse
import json
import sys

from rag import RagError, answer_question, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose une question a l'assistant technique OPTIJET."
    )
    parser.add_argument("question", nargs="+", help="Question de l'operateur")
    parser.add_argument("--top", type=int, default=None, help="Nombre d'extraits finaux (FINAL_K)")
    parser.add_argument("--lang", default="fr", choices=["fr", "en", "all"])
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Seuil cosinus optionnel (semantic >= 1 - max_distance)",
    )
    parser.add_argument("--json", action="store_true", help="Sortie JSON pour l'HMI")
    parser.add_argument("--debug", action="store_true", help="Logs detailles dans logs/rag.log")
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    question = " ".join(args.question).strip()

    print("Attente des serveurs ...")
    from rag import HEALTH_CHAT, HEALTH_EMBED, wait_for

    wait_for(HEALTH_EMBED, "llama-embed")
    wait_for(HEALTH_CHAT, "llama-chat")

    result = answer_question(
        question,
        lang=args.lang,
        n_results=args.top,
        max_distance=args.max_distance,
    )
    if args.json:
        payload = {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("\n--- Reponse ---")
    print(result["answer"])
    if result["sources"]:
        print("\n--- Sources ---")
        for source in result["sources"]:
            heading = source.get("heading") or source.get("section_path") or ""
            pages = source.get("start_page") or source.get("page")
            end_page = source.get("end_page") or pages
            page_label = f"p.{pages}" if pages == end_page else f"p.{pages}-{end_page}"
            print(f"- {source.get('source', '')} {page_label} | {heading}")
    timings = result.get("timings") or {}
    if args.debug:
        print("\n--- Diagnostic ---")
        print(f"type={result.get('question_type')} confidence={result.get('confidence', 0):.3f}")
        print(
            f"embedding={timings.get('embedding_time', 0):.3f}s "
            f"retrieval={timings.get('retrieval_time', 0):.3f}s "
            f"generation={timings.get('generation_time', 0):.3f}s "
            f"total={timings.get('total_time', 0):.3f}s "
            f"max_tokens={result.get('max_tokens', '')} "
            f"hits={len(result.get('hits') or [])}"
        )
        for index, hit in enumerate(result.get("hits") or [], start=1):
            print(
                f"[{index}] score={hit.get('score', 0):.3f} sem={hit.get('semantic', 0):.3f} "
                f"lex={hit.get('lexical', 0):.1f} exact={hit.get('exact', 0):.2f} "
                f"{hit.get('source')} p.{hit.get('page')} {hit.get('heading')}"
            )


if __name__ == "__main__":
    try:
        main()
    except RagError as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
