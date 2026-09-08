from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag import RagError, fold, search_chunks, setup_logging, wait_for
from settings import HEALTH_EMBED, ROOT

DEFAULT_EVAL = ROOT / "evaluation.json"


def page_matches(hit: dict, expected_page: int | None) -> bool:
    if expected_page is None:
        return True
    start = int(hit.get("start_page") or hit.get("page") or 0)
    end = int(hit.get("end_page") or start)
    expected = int(expected_page)
    return (start - 1) <= expected <= (end + 1)


def heading_matches(hit: dict, expected_heading: str | None) -> bool:
    if not expected_heading:
        return True
    needle = fold(expected_heading)
    haystack = fold(f"{hit.get('heading', '')} {hit.get('section_path', '')}")
    return needle in haystack


def source_matches(hit: dict, expected_source: str | None) -> bool:
    if not expected_source:
        return True
    return fold(expected_source) in fold(hit.get("source", ""))


def is_hit(hit: dict, case: dict) -> bool:
    return (
        source_matches(hit, case.get("expected_source"))
        and page_matches(hit, case.get("expected_page"))
        and heading_matches(hit, case.get("expected_heading"))
    )


def recall_at_k(ranked: list[dict], case: dict, k: int) -> int:
    return int(any(is_hit(hit, case) for hit in ranked[:k]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mesure Recall@k du moteur RAG.")
    parser.add_argument("--file", default=str(DEFAULT_EVAL), help="Fichier evaluation.json")
    parser.add_argument("--lang", default="fr", choices=["fr", "en", "all"])
    parser.add_argument("--k", type=int, default=5, help="Nombre d'extraits a recuperer")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    setup_logging(debug=args.debug)

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"Fichier d'evaluation introuvable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise SystemExit("evaluation.json: liste 'cases' vide")

    print(f"Attente de llama-embed sur {HEALTH_EMBED} ...")
    wait_for(HEALTH_EMBED, "llama-embed")

    totals = {1: 0, 3: 0, 5: 0}
    for index, case in enumerate(cases, start=1):
        question = str(case.get("question", "")).strip()
        if not question:
            continue
        lang = str(case.get("lang") or args.lang)
        hits = search_chunks(
            question,
            n_results=max(args.k, 5),
            lang=lang,
            apply_threshold=False,
        )
        rec1 = recall_at_k(hits, case, 1)
        rec3 = recall_at_k(hits, case, 3)
        rec5 = recall_at_k(hits, case, 5)
        totals[1] += rec1
        totals[3] += rec3
        totals[5] += rec5
        mark = "OK" if rec1 else ("~" if rec3 or rec5 else "KO")
        top = hits[0] if hits else {}
        print(
            f"[{index:02d} {mark}] {question}\n"
            f"         attendu: {case.get('expected_source', '')} "
            f"p.{case.get('expected_page')} | {case.get('expected_heading', '')}\n"
            f"         obtenu:  {top.get('source', '-')} "
            f"p.{top.get('page', '-')} | {top.get('heading', '-')}"
        )

    count = len(cases)
    print("\n--- Recall ---")
    for k in (1, 3, 5):
        value = totals[k] / count if count else 0.0
        print(f"Recall@{k}: {value:.3f} ({totals[k]}/{count})")


if __name__ == "__main__":
    try:
        main()
    except RagError as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
