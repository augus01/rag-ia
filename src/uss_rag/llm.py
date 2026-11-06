"""Funciones de LLM y embeddings sobre Ollama local (compatibles con LightRAG/RAG-Anything)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import ollama

from . import config

_async_client: ollama.AsyncClient | None = None


def _client() -> ollama.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = ollama.AsyncClient(host=config.OLLAMA_HOST, timeout=600)
    return _async_client


async def embed(texts: List[str]) -> np.ndarray:
    resp = await _client().embed(model=config.EMBED_MODEL, input=texts)
    return np.array(resp.embeddings, dtype=np.float32)


async def chat(prompt: str, system_prompt: Optional[str] = None,
               history_messages: Optional[List[Dict]] = None, **kwargs) -> str:
    """Firma compatible con llm_model_func de LightRAG."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history_messages or [])
    msgs.append({"role": "user", "content": prompt})
    opts = {"temperature": kwargs.pop("temperature", config.LLM_TEMPERATURE),
            "num_ctx": config.LLM_NUM_CTX,
            # tope de salida: evita bucles de generación (vistos al extraer entidades de tablas)
            "num_predict": config.LLM_MAX_TOKENS}
    fmt = "json" if kwargs.get("response_format") or kwargs.get("keyword_extraction") else None
    resp = await _client().chat(model=config.LLM_MODEL, messages=msgs, options=opts, format=fmt)
    return resp.message.content
