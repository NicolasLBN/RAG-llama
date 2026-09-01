from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from rag import HEALTH_CHAT, HEALTH_EMBED, RagError, answer_question, setup_logging, wait_for
from settings import ROOT

STATIC_DIR = ROOT / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def _json_bytes(payload: dict, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8")


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "optijet-ui/1.0"

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            if not INDEX_FILE.is_file():
                self._send(500, b"index.html manquant", "text/plain; charset=utf-8")
                return
            self._send(200, INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            status, body = self._health()
            self._send(status, body, "application/json; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/ask":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            status, body = _json_bytes({"error": "JSON invalide"}, 400)
            self._send(status, body, "application/json; charset=utf-8")
            return

        question = str(payload.get("question") or "").strip()
        lang = str(payload.get("lang") or "fr").strip().lower()
        if lang not in {"fr", "en", "all"}:
            lang = "fr"
        if not question:
            status, body = _json_bytes({"error": "Question vide"}, 400)
            self._send(status, body, "application/json; charset=utf-8")
            return

        try:
            result = answer_question(question, lang=lang)
            status, body = _json_bytes(
                {
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.0),
                    "sources": result.get("sources") or [],
                }
            )
        except RagError as exc:
            status, body = _json_bytes({"error": str(exc)}, 503)
        except Exception as exc:
            status, body = _json_bytes({"error": f"Erreur interne: {exc}"}, 500)
        self._send(status, body, "application/json; charset=utf-8")

    def _health(self) -> tuple[int, bytes]:
        embed_ok = chat_ok = False
        try:
            wait_for(HEALTH_EMBED, "llama-embed", timeout_s=2)
            embed_ok = True
        except Exception:
            pass
        try:
            wait_for(HEALTH_CHAT, "llama-chat", timeout_s=2)
            chat_ok = True
        except Exception:
            pass
        ready = embed_ok and chat_ok
        return _json_bytes(
            {"ok": ready, "embed": embed_ok, "chat": chat_ok},
            200 if ready else 503,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UI chat locale pour l'assistant OPTIJET (RAG)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    setup_logging()

    print("Attente de llama-embed et llama-chat ...", flush=True)
    wait_for(HEALTH_EMBED, "llama-embed")
    wait_for(HEALTH_CHAT, "llama-chat")

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"UI RAG: {url}", flush=True)
    print("Ctrl+C pour arreter.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret.")
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except RagError as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        sys.exit(1)
