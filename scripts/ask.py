"""Consulta al asistente USS.

    python scripts/ask.py "¿cuántas veces puedo reprobar un ramo antes de la eliminación?"
    python scripts/ask.py --motor grafo "..."          # RAG-Anything (grafo + vectores, modo mix)
    python scripts/ask.py --categoria CAT-03 "..."     # filtra por categoría (sólo motor pgvector)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uss_rag.taxonomy import CATEGORIAS  # noqa: E402


async def main(a) -> None:
    if a.motor == "pg":
        from uss_rag.answer import responder

        r = await responder(a.pregunta, categoria=a.categoria)
        print(r.texto)
        if r.fuentes:
            print("\nFuentes recuperadas:")
            for i, f in enumerate(r.fuentes, 1):
                print(f"  [{i}] {f['cita']}  ({f['categoria_id']}, sim={f['similitud']:.2f})")
    else:
        from uss_rag.rag_anything_engine import crear_rag, preguntar

        rag = crear_rag()
        print(await preguntar(rag, a.pregunta, modo=a.modo))
        await rag.finalize_storages()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pregunta")
    p.add_argument("--motor", choices=["pg", "grafo"], default="pg")
    p.add_argument("--modo", default="mix", help="modo RAG-Anything: mix|hybrid|local|global|naive")
    p.add_argument("--categoria", choices=list(CATEGORIAS))
    asyncio.run(main(p.parse_args()))
