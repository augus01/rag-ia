"""Motor RAG-Anything (HKUDS) para el asistente USS.

RAG-Anything construye, sobre LightRAG, un grafo de conocimiento (entidades + relaciones:
p. ej. "Eliminación académica" --causada_por--> "Reprobar un curso tres veces") además
del índice vectorial de chunks. Se consulta en modo "mix" (grafo + vectores).

Ingesta: en lugar del parser MinerU (pesado; los PDFs con texto no lo requieren) usamos
`insert_content_list`, la API de RAG-Anything para contenido pre-parseado, alimentándola con
nuestros chunks semánticos (un chunk por artículo) separados por "\\n\\n".
Para PDFs escaneados se puede usar `process_document_complete` con MinerU (OCR): ver TAREAS.md.
"""
from __future__ import annotations

from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

from . import config, llm
from .chunking import Chunk

SEP = "\n\n"

SYSTEM_PROMPT = (
    "Eres el Asistente de Reglamentos para estudiantes de la Universidad San Sebastián (USS). "
    "Responde SIEMPRE en español y SOLO con información presente en el contexto recuperado. "
    "Cita el documento y el artículo o página de donde sale cada dato (ej: 'Reglamento de Docencia "
    "de Pregrado, Art. 40'). Si el contexto no contiene la respuesta, dilo explícitamente y sugiere "
    "consultar a Dirección de Carrera o Registro Académico. No inventes plazos, números ni artículos."
)


def crear_rag() -> RAGAnything:
    cfg = RAGAnythingConfig(
        working_dir=str(config.RAG_STORAGE),
        parser="mineru",
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
        display_content_stats=False,
    )
    rag = RAGAnything(
        config=cfg,
        llm_model_func=llm.chat,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.EMBED_DIM, max_token_size=8192, func=llm.embed
        ),
        lightrag_kwargs={
            "chunk_token_size": 1200,
            "chunk_overlap_token_size": 100,
            "llm_model_max_async": 1,      # 1 petición a la vez: 8 GB de VRAM
            "default_llm_timeout": 600,
            "embedding_func_max_async": 4,
            "addon_params": {
                "language": "Spanish",
                "entity_types": ["reglamento", "articulo", "proceso_academico", "requisito",
                                 "plazo", "fecha", "organo_universitario", "asignatura",
                                 "carrera", "grado_o_titulo", "sancion", "beneficio"],
            },
        },
    )
    # Al inicializar, RAG-Anything verifica el parser ejecutando `mineru --version`. En este equipo
    # Windows Smart App Control bloquea mineru.exe (WinError 4551), y nuestra ingesta usa
    # insert_content_list, que no necesita parser. Se omite la verificación. (OCR: ver TAREAS.md)
    rag.doc_parser.check_installation = lambda: True
    return rag


async def ingestar_documento(rag: RAGAnything, nombre: str, chunks: list[Chunk]) -> None:
    content_list = [
        {"type": "text", "text": c.page_content, "page_idx": c.paginas[0] - 1} for c in chunks
    ]
    await rag.insert_content_list(
        content_list=content_list,
        file_path=nombre,
        split_by_character=SEP,
        split_by_character_only=False,
        doc_id=f"doc-{nombre}",
    )


async def preguntar(rag: RAGAnything, pregunta: str, modo: str = "mix") -> str:
    await rag._ensure_lightrag_initialized()
    # Ojo: `system_prompt` de LightRAG REEMPLAZA la plantilla que inyecta el contexto;
    # las instrucciones de acotación se pasan como `user_prompt` (se agregan a la plantilla).
    return await rag.aquery(pregunta, mode=modo, user_prompt=SYSTEM_PROMPT)
