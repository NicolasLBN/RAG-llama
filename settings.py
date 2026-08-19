from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
SYNONYMS_PATH = ROOT / "synonyms.json"
CHROMA_DIR = ROOT / "chroma_db"
PDF_DIR = ROOT / "pdf"
LOG_DIR = ROOT / "logs"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"
OCR_REPORT_PATH = ROOT / "ocr_needed.json"
IMAGE_CAPTIONS_PATH = ROOT / "image_captions.json"
COLLECTION_NAME = "manuals"

EMBED_URL = "http://localhost:8081/v1/embeddings"
CHAT_URL = "http://localhost:8080/v1/chat/completions"
HEALTH_EMBED = "http://localhost:8081/health"
HEALTH_CHAT = "http://localhost:8080/health"

DEFAULTS: dict[str, Any] = {
    "chunking": {
        "min_chars": 160,
        "max_chars": 900,
        "overlap_chars": 140,
        "min_native_chars": 80,
        "min_image_area_ratio": 0.04,
        "toc_dot_lines": 8,
    },
    "ingest": {"batch_size": 8, "ocr_dpi": 200},
    "embedding": {"query_prefix": "query: ", "document_prefix": ""},
    "search": {
        "fetch_k": 30,
        "final_k": 4,
        "exact_weight": 0.40,
        "lexical_weight": 0.25,
        "semantic_weight": 0.35,
        "min_semantic_score": 0.28,
        "min_final_score": 0.32,
        "enrich_query": False,
    },
    "generation": {"temperature": 0.0, "max_tokens": 280, "timeout_s": 180},
    "logging": {"level": "INFO", "file": "logs/rag.log"},
}


class Settings:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self._data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger("rag").error("Impossible de lire %s: %s", path, exc)
        return fallback


def load_settings() -> Settings:
    overlay = load_json(CONFIG_PATH, {})
    if not isinstance(overlay, dict):
        overlay = {}
    return Settings(_deep_merge(DEFAULTS, overlay))


def load_synonyms() -> dict[str, Any]:
    data = load_json(
        SYNONYMS_PATH,
        {"synonyms": {}, "phrase_bonus": [], "phrase_penalty": []},
    )
    if not isinstance(data, dict):
        return {"synonyms": {}, "phrase_bonus": [], "phrase_penalty": []}
    data.setdefault("synonyms", {})
    data.setdefault("phrase_bonus", [])
    data.setdefault("phrase_penalty", [])
    return data


def setup_logging(debug: bool = False) -> logging.Logger:
    settings = load_settings()
    level_name = "DEBUG" if debug else str(settings.get("logging", "level", default="INFO"))
    level = getattr(logging, level_name.upper(), logging.INFO)
    log_path = ROOT / str(settings.get("logging", "file", default="logs/rag.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag")
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if debug:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    else:
        logger.setLevel(level)
    logger.propagate = False
    return logger


CFG = load_settings()
