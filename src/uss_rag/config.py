"""Configuración central del Asistente de Reglamentos USS (lee .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FUENTES_JSON = ROOT / "data" / "fuentes.json"
RAG_STORAGE = Path(os.getenv("RAG_STORAGE", ROOT / "rag_storage"))

# --- Modelos locales vía Ollama (sin API key) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# --- PostgreSQL + pgvector ---
PG_DSN = os.getenv(
    "PG_DSN", "postgresql://uss:uss_rag@localhost:5433/rag_uss"
)
PG_TABLE = os.getenv("PG_TABLE", "vectores_rag_uss")

# --- Recuperación ---
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SIMILITUD = float(os.getenv("MIN_SIMILITUD", "0.35"))
