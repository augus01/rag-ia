"""Taxonomía RAG USS (5 categorías), análoga a la del ejemplo FAE de la Solemne II.

La categoría se asigna por chunk combinando:
  1. Un valor por defecto según el tipo de documento (fuente).
  2. Reglas de palabras clave sobre el texto del chunk, que pueden reasignarla
     (p. ej. un artículo del reglamento de docencia que trata de titulación -> CAT-02).
"""
from __future__ import annotations

import unicodedata

CATEGORIAS = {
    "CAT-01": "Reglamento Académico (docencia de pregrado)",
    "CAT-02": "Egreso, Graduación y Titulación",
    "CAT-03": "Calendario Académico",
    "CAT-04": "Mallas Curriculares y Planes de Estudio",
    "CAT-05": "Convivencia, Becas y Bienestar Estudiantil",
}

# Categoría por defecto según el tipo de documento.
DEFAULT_POR_TIPO = {
    "reglamento_docencia": "CAT-01",
    "calendario": "CAT-03",
    "malla": "CAT-04",
    "convivencia": "CAT-05",
    "becas": "CAT-05",
}

# Reglas de reasignación (solo se aplican a documentos normativos).
REGLAS = [
    ("CAT-02", ["titulacion", "titulo profesional", "egresado", "egreso",
                "graduacion", "grado academico", "licenciado", "examen de grado",
                "examen de titulo", "seminario de titulo", "tesina"]),
    ("CAT-05", ["beca", "beneficio", "convivencia", "acoso", "sancion disciplinaria",
                "falta grave", "discriminacion"]),
]


def _norm(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def asignar_categoria(texto: str, tipo_doc: str) -> str:
    """Devuelve el ID de categoría (CAT-01..CAT-05) para un chunk."""
    base = DEFAULT_POR_TIPO.get(tipo_doc, "CAT-01")
    if tipo_doc in ("calendario", "malla"):
        return base  # documentos con propósito único: no se reasignan
    t = _norm(texto)
    for cat, palabras in REGLAS:
        hits = sum(t.count(p) for p in palabras)
        if hits >= 2 and cat != base:
            return cat
    return base


# Raíces de palabras para enrutar la PREGUNTA del estudiante a una categoría probable.
RUTEO_CONSULTA = {
    "CAT-02": ["titul", "egres", "gradua", "licencia", "bachiller", "tesis", "tesina",
               "seminario de titulo", "examen de grado"],
    "CAT-03": ["calendario", "cuando", "fecha", "plazo", "inicio de clases", "vacaciones",
               "feriado", "receso", "semana", "inscripcion de asignaturas"],
    "CAT-04": ["malla", "plan de estudio", "semestres dura", "duracion", "ramos de",
               "asignaturas de", "sedes", "carrera de"],
    "CAT-05": ["beca", "beneficio", "convivencia", "acoso", "sancion", "disciplin"],
}


def detectar_categoria(pregunta: str) -> str | None:
    """Categoría más probable de una pregunta (None si no hay señal clara)."""
    p = _norm(pregunta)
    puntajes = {cat: sum(p.count(r) for r in raices) for cat, raices in RUTEO_CONSULTA.items()}
    cat, pts = max(puntajes.items(), key=lambda x: x[1])
    return cat if pts > 0 else None


def tipo_documento(nombre_archivo: str) -> str:
    n = nombre_archivo.lower()
    if "calendario" in n:
        return "calendario"
    if "malla" in n:
        return "malla"
    if "convivencia" in n:
        return "convivencia"
    if "beca" in n:
        return "becas"
    return "reglamento_docencia"
