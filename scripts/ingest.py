"""Pipeline de ingesta: PDF -> limpieza NLP -> chunks por artículo -> taxonomía ->
   (a) embeddings + PostgreSQL/pgvector (HNSW)   (b) grafo de conocimiento RAG-Anything.

Uso:
    python scripts/ingest.py                 # pgvector + RAG-Anything
    python scripts/ingest.py --solo-pg       # sólo pgvector (rápido, ~1 min)
    python scripts/ingest.py --solo-grafo    # sólo grafo RAG-Anything (lento: usa el LLM)
    python scripts/ingest.py --forzar        # re-ingesta aunque no haya cambios
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uss_rag import config, llm, pgstore  # noqa: E402
from uss_rag.chunking import chunkear_pdf  # noqa: E402


async def main(args) -> None:
    pdfs = sorted(config.DATA_RAW.glob("*.pdf"))
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    por_doc, advertencias = {}, []
    for pdf in pdfs:
        chunks, adv = chunkear_pdf(pdf)
        if adv:
            advertencias.append(adv)
            continue
        por_doc[pdf.name] = chunks
        (config.DATA_PROCESSED / f"{pdf.stem}.chunks.json").write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(map(len, por_doc.values()))
    print(f"Chunks generados: {total} en {len(por_doc)} documentos")
    for a in advertencias:
        print("  [!]", a)

    if not args.solo_grafo:
        conn = pgstore.conectar()
        pgstore.crear_esquema(conn)
        for nombre, chunks in por_doc.items():
            previos = pgstore.hashes_existentes(conn, nombre)
            nuevos = {c.chunk_id: c.metadata["hash"] for c in chunks}
            if previos == nuevos and not args.forzar:
                print(f"  = {nombre}: sin cambios, se omite")
                continue
            t0 = time.time()
            embs = []
            for i in range(0, len(chunks), 16):
                embs.extend(await llm.embed([c.page_content for c in chunks[i:i + 16]]))
            pgstore.reemplazar_documento(conn, nombre, chunks, embs)
            print(f"  + {nombre}: {len(chunks)} vectores ({time.time() - t0:.1f}s)")
        print("Resumen pgvector:")
        for fila in pgstore.resumen(conn):
            print("   ", *fila)
        conn.close()

    if not args.solo_pg:
        from uss_rag.rag_anything_engine import crear_rag, ingestar_documento

        rag = crear_rag()
        # reglamentos primero (lo más consultado), calendario al final (más largo y tabular)
        orden = sorted(por_doc, key=lambda n: ("reglamento" not in n, "calendario" in n, n))
        if args.doc:
            orden = [n for n in orden if args.doc in n]
        for nombre in orden:
            chunks = por_doc[nombre]
            t0 = time.time()
            print(f"  [grafo] {nombre} ({len(chunks)} chunks)...", flush=True)
            await ingestar_documento(rag, nombre, chunks)
            print(f"  [grafo] {nombre} listo ({time.time() - t0:.0f}s)", flush=True)
        await rag.finalize_storages()
    print("Ingesta completada.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solo-pg", action="store_true")
    p.add_argument("--solo-grafo", action="store_true")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--doc", help="sólo grafo: procesar documentos cuyo nombre contenga este texto")
    asyncio.run(main(p.parse_args()))
