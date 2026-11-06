"""Taxonomía RAG USS (5 categorías), análoga a la del ejemplo FAE de la Solemne II.

La categoría se asigna por chunk combinando:
  1. Un valor por defecto según el tipo de documento (fuente).
  2. Reglas de palabras clave sobre el texto del chunk, que pueden reasignarla
     (p. ej. un artículo del reglamento de docencia que trata de titulación -> CAT-02).
"""
from __future__ import annotations

import re
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


# Raíces (con peso) para enrutar la PREGUNTA del estudiante a una categoría probable.
# Se busca al INICIO de palabra (\braíz) para no coincidir dentro de otras palabras.
# Términos genéricos ("cuando", "plazo", "fecha", "semana") pesan poco: aparecen también en
# preguntas del reglamento (p. ej. "plazo para apelar") y no deben arrastrar al calendario.
RUTEO_CONSULTA = {
    "CAT-02": {"titul": 2, "egres": 2, "gradua": 2, "licencia": 1, "bachiller": 2, "tesis": 1,
               "tesina": 1, "seminario de titulo": 2, "examen de grado": 2},
    "CAT-03": {"calendario": 3, "inicio de clases": 3, "inicio del semestre": 3, "vacaciones": 2,
               "feriado": 3, "receso": 3, "inscripcion de asignaturas": 3,
               "cuando": 0.5, "fecha": 0.5, "plazo": 0.5, "semana": 0.5},
    "CAT-04": {"malla": 3, "plan de estudio": 3, "semestres dura": 2, "duracion de la carrera": 2,
               "ramos de": 1, "asignaturas de": 1, "sedes": 1},
    "CAT-05": {"beca": 2, "beneficio": 2, "convivencia": 2, "acoso": 2, "sancion": 1,
               "disciplin": 1},
}
UMBRAL_RUTEO = 1.5  # puntaje mínimo: una sola palabra genérica no basta para enrutar


def detectar_categoria(pregunta: str) -> str | None:
    """Categoría más probable de una pregunta (None si no hay señal clara o hay empate)."""
    p = _norm(pregunta)
    puntajes = {
        cat: sum(peso * len(re.findall(r"\b" + re.escape(raiz), p)) for raiz, peso in raices.items())
        for cat, raices in RUTEO_CONSULTA.items()
    }
    orden = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    (cat, pts), (_, segundo) = orden[0], orden[1]
    return cat if pts >= UMBRAL_RUTEO and pts > segundo else None


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
