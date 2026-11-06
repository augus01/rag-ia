-- Esquema pgvector del Asistente de Reglamentos USS (generado desde src/uss_rag/pgstore.py)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS vectores_rag_uss (
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
    embedding     VECTOR(1024),
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('spanish', page_content)) STORED
);
CREATE INDEX IF NOT EXISTS vectores_rag_uss_hnsw ON vectores_rag_uss USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS vectores_rag_uss_tsv  ON vectores_rag_uss USING gin (tsv);
CREATE INDEX IF NOT EXISTS vectores_rag_uss_cat  ON vectores_rag_uss (categoria_id);
CREATE INDEX IF NOT EXISTS vectores_rag_uss_meta ON vectores_rag_uss USING gin (metadata_json);
