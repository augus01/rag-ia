# Tareas pendientes — Asistente de Reglamentos USS

Estado al 2026-11-06. Lo hecho: pipeline de ingesta (PyMuPDF → limpieza → chunks por artículo →
taxonomía CAT-01..05), pgvector con HNSW + full-text + JSONB, búsqueda híbrida con ruteo por
categoría, generación con citas y umbral anti-alucinación, integración RAG-Anything (grafo).

## Entregables de la Actividad (Actividad.pdf)

- [ ] **Informe de arquitectura (Fases 0 a 4)**. Insumos ya disponibles en el código/README:
  - Fase 0: problema (reglamentos dispersos en PDFs, lenguaje jurídico; Ctrl+F no entiende
    "reprobar un ramo" ≈ "reprobar un mismo curso"), inventario (8 PDFs uss.cl, ~150 págs.).
  - Fase 1: chunk por artículo + encabezado contextual; metadatos documento/artículo/página/categoría/URL.
  - Fase 2: bge-m3 (multilingüe, 1024 dims, gratis, local) + pgvector HNSW (justificar vs Chroma/Pinecone).
  - Fase 3: híbrida coseno + tsvector español con RRF, ruteo por categoría, prompt acotado, temp 0.
  - Fase 4: métricas (ver tarea de evaluación), "no encontrado" bajo umbral, re-ingesta por hash.
- [ ] **Diagrama de flujo/arquitectura** (usar el esquema del README como base).
- [ ] **Fase 5: dos preguntas de auditoría** para el grupo asignado (depende de qué grupo toque).
- [ ] **Interfaz web de chat** (estilo `mvp_agente_rag_fae.html`), conectada a un endpoint
      FastAPI que llame a `uss_rag.answer.responder` (fastapi/uvicorn ya están instalados).

## Datos

- [ ] **OCR de PDFs escaneados**: `reglamento_becas_beneficios_DR106_2026.pdf` y
      `reglamento_convivencia_universitaria_DR61_2024.pdf` son imágenes sin texto y hoy NO se indexan
      (CAT-05 queda casi vacía). Opciones: `rag.process_document_complete(pdf, parse_method="ocr")`
      de RAG-Anything con MinerU (ya instalado; descarga modelos la primera vez), o `ocrmypdf`.
      **Ojo:** en este PC Windows Smart App Control bloquea `mineru.exe` (WinError 4551). Para usar
      MinerU hay que desactivar Smart App Control (Seguridad de Windows → Control de aplicaciones y
      navegador) o correr la ingesta en WSL/Docker. Luego quitar el bypass de
      `check_installation` en `src/uss_rag/rag_anything_engine.py`.
- [ ] **Reglamento específico de titulación** de la Facultad de Ingeniería y reglamento de la
      carrera (no están publicados en uss.cl; pedir a Dirección de Carrera).
- [ ] **Mallas**: la grilla de asignaturas (pág. 2) viene como texto desordenado; para preguntas
      del tipo "¿qué ramos tengo en 5º semestre?" conviene parsearla con MinerU (tablas) o a mano a CSV.
- [ ] Validar si el calendario semestral vs trimestral aplica según la carrera (hay ambos en el PDF).

## Calidad / evaluación

- [ ] **Set de evaluación** (20-30 preguntas con respuesta y artículo esperado) y medir con **Ragas**
      (faithfulness, answer relevancy, context precision/recall).
- [ ] **Precisión del LLM 7B**: a veces parafrasea mal la norma (ej. "puedes reprobar 3 veces
      antes de..." cuando la causal ES la 3ª reprobación) u omite el caso semestral (45 días) en
      suspensión. Probar `qwen2.5:14b` (no cabe completo en 8 GB VRAM, más lento) o un modelo vía API
      cambiando `LLM_MODEL`/`llm.py`.
- [ ] Re-ranking (p. ej. `bge-reranker-v2-m3` local) sobre los top-20 antes de pasar 6 al LLM.
- [ ] Calibrar `MIN_SIMILITUD` (hoy 0.35) con el set de evaluación.
- [ ] Comparar motor `pg` vs motor `grafo` (RAG-Anything modo mix) en el set de evaluación.

## Operación

- [ ] Script de actualización periódica: re-descargar PDFs desde las URLs de `data/fuentes.json`
      y re-ingestar (la ingesta ya es incremental por hash).
- [ ] Al re-ingestar un documento modificado en el grafo de RAG-Anything, borrar su doc_id
      (`lightrag.adelete_by_doc_id`) antes de insertarlo de nuevo.
