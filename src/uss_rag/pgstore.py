"""Persistencia vectorial en PostgreSQL + pgvector con índice HNSW y búsqueda híbrida.

Búsqueda híbrida = similitud coseno (pgvector) + full-text en español (tsvector/BM25-like),
fusionadas con Reciprocal Rank Fusion (RRF). Permite filtrar por categoría (CAT-xx).
"""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from . import config

T = config.PG_TABLE

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {T} (
    id            SERIAL PRIMARY KEY,
    chunk_id      VARCHAR(120) NOT NULL UNIQUE,
    categoria_id  VARCHAR(10)  NOT NULL,
    seccion       VARCHAR(255),
    fuente        VARCHAR(255) NOT NULL,
    articulo      VARCHAR(40),
    paginas       INT[],
    page_content  TEXT NOT NULL,
    metadata_json JSONB,
    content_hash  VARCHAR(40),
    ingestado_en  TIMESTAMPTZ DEFAULT now(),
    embedding     VECTOR({config.EMBED_DIM}),
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('spanish', page_content)) STORED
);
CREATE INDEX IF NOT EXISTS {T}_hnsw ON {T} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {T}_tsv  ON {T} USING gin (tsv);
CREATE INDEX IF NOT EXISTS {T}_cat  ON {T} (categoria_id);
CREATE INDEX IF NOT EXISTS {T}_meta ON {T} USING gin (metadata_json);
"""


def conectar() -> psycopg.Connection:
    conn = psycopg.connect(config.PG_DSN, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def crear_esquema(conn) -> None:
    conn.execute(DDL)
    conn.commit()


def hashes_existentes(conn, fuente: str) -> dict[str, str]:
    rows = conn.execute(f"SELECT chunk_id, content_hash FROM {T} WHERE fuente=%s", (fuente,)).fetchall()
    return dict(rows)


def reemplazar_documento(conn, fuente: str, chunks, embeddings) -> None:
    """Invalidación por documento: borra los vectores antiguos de la fuente e inserta los nuevos
    en una sola transacción (nunca queda la fuente a medio actualizar)."""
    import numpy as np

    with conn.transaction():
        conn.execute(f"DELETE FROM {T} WHERE fuente=%s", (fuente,))
        for c, emb in zip(chunks, embeddings):
            conn.execute(
                f"""INSERT INTO {T} (chunk_id, categoria_id, seccion, fuente, articulo, paginas,
                        page_content, metadata_json, content_hash, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (c.chunk_id, c.categoria_id, c.seccion, c.documento, c.articulo, c.paginas,
                 c.page_content, Jsonb(c.metadata | {"categoria": c.categoria_id, "seccion": c.seccion}),
                 c.metadata["hash"], np.array(emb, dtype=np.float32)),
            )
    conn.commit()  # sin autocommit, conn.transaction() es sólo un savepoint


def busqueda_hibrida(conn, query_emb, query_text: str, k: int = config.TOP_K,
                     categoria: str | None = None, rrf_k: int = 60) -> list[dict]:
    import numpy as np

    filtro = "WHERE categoria_id = %(cat)s" if categoria else ""
    sql = f"""
    WITH sem AS (
        SELECT id, 1 - (embedding <=> %(emb)s) AS sim,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s) AS r
        FROM {T} {filtro}
        ORDER BY embedding <=> %(emb)s LIMIT 30
    ),
    kw AS (
        -- plainto_tsquery une los términos con AND: una pregunta completa casi nunca calza.
        -- Se convierte a OR (&  ->  |) y ts_rank_cd ordena por cuántos términos coinciden.
        SELECT id, ts_rank_cd(tsv, q) AS rank_kw,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
        FROM {T},
             to_tsquery('spanish', replace(plainto_tsquery('spanish', %(q)s)::text, '&', '|')) q
        WHERE tsv @@ q {"AND categoria_id = %(cat)s" if categoria else ""}
        ORDER BY rank_kw DESC LIMIT 30
    )
    SELECT v.chunk_id, v.categoria_id, v.seccion, v.fuente, v.articulo, v.paginas,
           v.page_content, v.metadata_json,
           COALESCE(sem.sim, 1 - (v.embedding <=> %(emb)s)) AS similitud,
           COALESCE(1.0/(%(rrf)s + sem.r), 0) + COALESCE(1.0/(%(rrf)s + kw.r), 0) AS score
    FROM {T} v
    LEFT JOIN sem ON sem.id = v.id
    LEFT JOIN kw  ON kw.id  = v.id
    WHERE sem.id IS NOT NULL OR kw.id IS NOT NULL
    ORDER BY score DESC
    LIMIT %(k)s
    """
    params = {"emb": np.array(query_emb, dtype=np.float32), "q": query_text,
              "k": k, "cat": categoria, "rrf": rrf_k}
    cur = conn.execute(sql, params)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def resumen(conn) -> list[tuple]:
    return conn.execute(
        f"SELECT fuente, categoria_id, COUNT(*) FROM {T} GROUP BY 1,2 ORDER BY 1,2").fetchall()
