# Asistente de Reglamentos USS — RAG con RAG-Anything + pgvector

Asistente que responde preguntas de estudiantes de la Universidad San Sebastián sobre el
**reglamento académico, titulación, calendario académico y mallas**, citando documento,
artículo y página. Ejemplo:

```
> ¿cuántas veces puedo reprobar un ramo antes de la eliminación?
Puedes reprobar un mismo curso tres veces antes de que se considere una causal de eliminación académica [1].
  [1] reglamento_docencia_pregrado_DR105bis_2024.pdf | Artículo 40 | pág. 12  (CAT-01)
```

Todo corre **local y sin API key**: LLM `qwen2.5:7b` y embeddings `bge-m3` vía Ollama.

## Arquitectura

```
PDFs oficiales uss.cl (data/raw, fuentes.json con URL)
        │  PyMuPDF  →  limpieza NLP (regex)  →  chunk por ARTÍCULO / página  →  taxonomía CAT-01..05
        ├──────────────► (A) bge-m3 → PostgreSQL + pgvector (HNSW + tsvector español + JSONB)
        └──────────────► (B) RAG-Anything / LightRAG: grafo de conocimiento (entidades + relaciones)
Pregunta ─► embedding ─► búsqueda híbrida (coseno + full-text, RRF) + ruteo por categoría
        ─► umbral de similitud (si no hay evidencia → "no encontrado", sin LLM)
        ─► prompt acotado + contexto numerado → qwen2.5:7b (temp 0) → respuesta con citas [n]
```

| Categoría | Contenido |
|---|---|
| CAT-01 | Reglamento académico (docencia de pregrado: inscripción, evaluación, eliminación) |
| CAT-02 | Egreso, graduación y titulación |
| CAT-03 | Calendario académico 2026 |
| CAT-04 | Mallas curriculares y planes de estudio |
| CAT-05 | Convivencia, becas y bienestar estudiantil |

Dos motores de consulta:
- `--motor pg` (por defecto): pgvector + búsqueda híbrida + filtros por categoría (pauta Solemne II).
- `--motor grafo`: RAG-Anything en modo `mix` (grafo de conocimiento + vectores), útil para preguntas
  que relacionan varios artículos.

## Puesta en marcha

Requisitos: Docker Desktop, [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/).

```powershell
# 1. Modelos locales
ollama pull qwen2.5:7b
ollama pull bge-m3

# 2. PostgreSQL + pgvector (puerto 15432; el 5433 lo ocupa otro Postgres en WSL)
docker run -d --name pgvector-uss -e POSTGRES_USER=uss -e POSTGRES_PASSWORD=uss_rag `
  -e POSTGRES_DB=rag_uss -p 127.0.0.1:15432:5432 -v pgvector_uss:/var/lib/postgresql/data `
  pgvector/pgvector:pg17

# 3. Entorno Python 3.12 (MinerU no soporta 3.14)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
copy .env.example .env

# 4. Ingesta y consultas
.venv\Scripts\python.exe scripts\ingest.py --solo-pg       # ~1 min
.venv\Scripts\python.exe scripts\ingest.py --solo-grafo    # grafo RAG-Anything (lento, usa el LLM)
.venv\Scripts\python.exe scripts\ask.py "¿cuándo empiezan las clases del segundo semestre?"
.venv\Scripts\python.exe scripts\ask.py --categoria CAT-02 "¿qué necesito para titularme?"
.venv\Scripts\python.exe scripts\ask.py --motor grafo "¿qué pasa si repruebo un ramo 3 veces?"
```

La re-ingesta es **incremental**: compara el hash de cada chunk; si un documento cambió, borra e
inserta sus vectores en una sola transacción (invalidación por documento).

## Estructura

```
data/raw/            PDFs oficiales descargados de uss.cl
data/fuentes.json    URL de origen y fecha de descarga de cada PDF
data/processed/      chunks generados (JSON, para inspección/auditoría)
src/uss_rag/
  config.py          configuración (.env)
  chunking.py        extracción PyMuPDF, limpieza NLP, chunking por artículo/página
  taxonomy.py        taxonomía CAT-01..05 (por chunk y ruteo de preguntas)
  pgstore.py         esquema pgvector (HNSW, GIN tsvector, JSONB) y búsqueda híbrida RRF
  llm.py             LLM y embeddings Ollama (firmas compatibles con LightRAG)
  answer.py          recuperación + prompt acotado + generación con citas
  rag_anything_engine.py   integración RAG-Anything (insert_content_list + aquery mix)
scripts/ingest.py, scripts/ask.py
sql/schema.sql       DDL equivalente para revisión
```

Pendientes: ver [TAREAS.md](TAREAS.md).
