"""Flujo de Recuperación + Generación sobre pgvector (ruta principal Solemne II).

1. Embedding de la pregunta (bge-m3).
2. Búsqueda híbrida (coseno + full-text) con filtro opcional por categoría.
3. Umbral de similitud: si nada supera MIN_SIMILITUD, se responde "no encontrado" SIN llamar al LLM
   (mitigación de alucinaciones).
4. Prompt acotado: instrucciones estrictas + contexto numerado + pregunta. Temperatura baja.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, llm, pgstore
from .taxonomy import detectar_categoria

INSTRUCCIONES = """Eres el Asistente de Reglamentos para estudiantes de la Universidad San Sebastián (USS).
Reglas OBLIGATORIAS:
1. Responde en español, de forma clara y breve, usando EXCLUSIVAMENTE la información del CONTEXTO.
2. Cada afirmación debe citar su fuente entre corchetes con el número del fragmento, ej: [1], [2].
3. Si el CONTEXTO no contiene la respuesta, responde exactamente: "No encontré esa información en los
   reglamentos cargados. Te recomiendo consultar con tu Dirección de Carrera o Registro Académico."
4. No inventes números, plazos, fechas ni artículos. No uses conocimiento externo.
5. Si la norma tiene excepciones o derechos asociados (p. ej. solicitudes de permanencia), menciónalos.
6. Cita la norma con precisión: si el reglamento dice "haber reprobado un curso tres veces" es causal,
   no lo reformules de otra forma que cambie el sentido."""

NO_ENCONTRADO = ("No encontré esa información en los reglamentos cargados. Te recomiendo consultar "
                 "con tu Dirección de Carrera o Registro Académico.")


@dataclass
class Respuesta:
    texto: str
    fuentes: list[dict]


def _cita(f: dict) -> str:
    partes = [f["fuente"]]
    if f.get("articulo"):
        partes.append(f["articulo"])
    if f.get("paginas"):
        partes.append("pág. " + ", ".join(map(str, f["paginas"])))
    return " | ".join(partes)


def construir_prompt(pregunta: str, fuentes: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] ({_cita(f)})\n{f['page_content']}" for i, f in enumerate(fuentes, 1))
    return f"CONTEXTO:\n{ctx}\n\nPREGUNTA DEL ESTUDIANTE: {pregunta}\n\nRESPUESTA (con citas [n]):"


async def responder(pregunta: str, categoria: str | None = None, k: int = config.TOP_K) -> Respuesta:
    q_emb = (await llm.embed([pregunta]))[0]
    with pgstore.conectar() as conn:
        fuentes = pgstore.busqueda_hibrida(conn, q_emb, pregunta, k=k, categoria=categoria)
        # Enrutamiento por taxonomía: si la pregunta apunta a una categoría, se agregan los
        # mejores chunks de esa categoría (intercalados) aunque no ganen en el ranking global.
        ruta = None if categoria else detectar_categoria(pregunta)
        if ruta:
            filtradas = pgstore.busqueda_hibrida(conn, q_emb, pregunta, k=k, categoria=ruta)
            fuentes = _intercalar(filtradas, fuentes, k)
    relevantes = [f for f in fuentes if f["similitud"] >= config.MIN_SIMILITUD]
    if not relevantes:
        return Respuesta(NO_ENCONTRADO, [])
    texto = (await llm.chat(construir_prompt(pregunta, relevantes), system_prompt=INSTRUCCIONES,
                            temperature=0.0)).strip()
    if texto.startswith("No encontré"):
        return Respuesta(NO_ENCONTRADO, [])
    for f in relevantes:
        f["cita"] = _cita(f)
    return Respuesta(texto, relevantes)


def _intercalar(a: list[dict], b: list[dict], k: int) -> list[dict]:
    vistos, out = set(), []
    for par in zip(a + [None] * len(b), b + [None] * len(a)):
        for f in par:
            if f and f["chunk_id"] not in vistos:
                vistos.add(f["chunk_id"])
                out.append(f)
    return out[:k]
