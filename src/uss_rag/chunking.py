"""Extracción, limpieza NLP y chunking semántico de los documentos USS.

Estrategia por tipo de documento:
  * Reglamentos  -> un chunk por ARTÍCULO, enriquecido con su TÍTULO (capítulo).
                    Artículos largos se subdividen por párrafos con solapamiento.
  * Calendario   -> un chunk por página (cada página corresponde a un tramo de un mes).
  * Mallas       -> un chunk por página (ficha de carrera / grilla de asignaturas).
Cada chunk lleva metadatos (documento, páginas, artículo, título, categoría, url).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from . import config
from .taxonomy import CATEGORIAS, asignar_categoria, tipo_documento

MAX_CHARS = 2200   # ~550 tokens: cabe holgado en bge-m3 (8k) y en el contexto del LLM
OVERLAP = 250

RE_TITULO = re.compile(r"^T[ÍI]TULO\s+([IVXLC]+)\s*$", re.M)
RE_ARTICULO = re.compile(r"(?m)^\s*Art[íi]culo\s+(\d+)\s*(?:°|º)?\s*(bis)?\s*\.?-", re.I)
RE_MESES = re.compile(
    r"\b(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)\b"
)


@dataclass
class Chunk:
    chunk_id: str
    documento: str
    tipo_doc: str
    categoria_id: str
    seccion: str
    paginas: list[int]
    texto: str
    url: str = ""
    articulo: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def page_content(self) -> str:
        """Texto enriquecido que se vectoriza (encabezado contextual + contenido)."""
        cab = [f"[Documento]: {self.documento}",
               f"[Categoría]: {self.categoria_id} - {CATEGORIAS[self.categoria_id]}",
               f"[Sección]: {self.seccion}"]
        if self.articulo:
            cab.append(f"[Artículo]: {self.articulo}")
        cab.append(f"[Página(s)]: {', '.join(str(p) for p in self.paginas)}")
        return "\n".join(cab) + "\n[Contenido]:\n" + self.texto

    def to_dict(self) -> dict:
        d = asdict(self)
        d["page_content"] = self.page_content
        return d


# ---------------------------------------------------------------- limpieza NLP
def limpiar_texto(texto: str) -> str:
    """Normaliza espacios, elimina números de página sueltos y ruido de PDF."""
    if not texto:
        return ""
    texto = texto.replace("­", "").replace("​", "").replace("\xa0", " ")
    texto = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", texto)          # nº de página aislado
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)              # palabras cortadas
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def a_una_linea(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


# ---------------------------------------------------------------- extracción
def extraer_paginas(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as doc:
        return [p.get_text() for p in doc]


def es_escaneado(paginas: list[str]) -> bool:
    return sum(len(p.strip()) for p in paginas) < 50 * max(1, len(paginas))


def _url_fuente(nombre: str) -> str:
    try:
        for f in json.loads(config.FUENTES_JSON.read_text(encoding="utf-8-sig")):
            if f["archivo"] == nombre:
                return f["url"]
    except FileNotFoundError:
        pass
    return ""


def _id(*partes) -> str:
    return hashlib.sha1("|".join(map(str, partes)).encode()).hexdigest()[:12]


def _subdividir(texto: str) -> list[str]:
    """Divide un texto largo por párrafos/oraciones con solapamiento."""
    if len(texto) <= MAX_CHARS:
        return [texto]
    piezas = re.split(r"(?<=[.;:])\s+(?=\(?\d+\)|[A-ZÁÉÍÓÚ])", texto)
    out, actual = [], ""
    for p in piezas:
        if len(actual) + len(p) > MAX_CHARS and actual:
            out.append(actual.strip())
            actual = actual[-OVERLAP:] + " "
        actual += p + " "
    if actual.strip():
        out.append(actual.strip())
    return out


# ---------------------------------------------------------------- por tipo
def _chunks_reglamento(nombre: str, paginas: list[str], tipo: str, url: str) -> list[Chunk]:
    # Texto completo con marcas de página para mapear offsets -> nº de página.
    textos, offsets, pos = [], [], 0
    for i, p in enumerate(paginas):
        t = limpiar_texto(p) + "\n"
        offsets.append((pos, i + 1))
        textos.append(t)
        pos += len(t)
    full = "".join(textos)

    def pagina_de(off: int) -> int:
        pag = 1
        for ini, n in offsets:
            if ini <= off:
                pag = n
        return pag

    # Títulos (capítulos): "TÍTULO IX" + línea siguiente con el nombre.
    titulos = []
    for m in RE_TITULO.finditer(full):
        resto = full[m.end():].lstrip("\n").split("\n", 1)[0]
        titulos.append((m.start(), f"Título {m.group(1)} - {a_una_linea(resto).title()}"))

    def titulo_de(off: int) -> str:
        t = "Preámbulo / Decreto"
        for ini, nombre_t in titulos:
            if ini <= off:
                t = nombre_t
        return t

    arts = list(RE_ARTICULO.finditer(full))
    chunks: list[Chunk] = []
    cortes = [(0, None)] + [(m.start(), f"Artículo {m.group(1)}{' bis' if m.group(2) else ''}") for m in arts]
    for idx, (ini, art) in enumerate(cortes):
        fin = cortes[idx + 1][0] if idx + 1 < len(cortes) else len(full)
        bloque = full[ini:fin]
        # quita encabezados de TÍTULO que quedaron al final del bloque (pertenecen al siguiente)
        bloque = RE_TITULO.split(bloque)[0] if art else bloque
        texto = a_una_linea(bloque)
        if len(texto) < 40:
            continue
        seccion = titulo_de(ini)
        paginas_bloque = sorted({pagina_de(ini), pagina_de(max(ini, fin - 1))})
        for j, parte in enumerate(_subdividir(texto)):
            cat = asignar_categoria(seccion + " " + parte, tipo)
            chunks.append(Chunk(
                chunk_id=f"{Path(nombre).stem[:30]}_{art or 'preambulo'}_{j}".replace(" ", ""),
                documento=nombre, tipo_doc=tipo, categoria_id=cat, seccion=seccion,
                paginas=paginas_bloque, texto=parte, url=url, articulo=art,
            ))
    return chunks


def _chunks_por_pagina(nombre: str, paginas: list[str], tipo: str, url: str) -> list[Chunk]:
    chunks, mes_actual = [], ""
    for i, p in enumerate(paginas, start=1):
        limpio = limpiar_texto(p)
        if len(limpio) < 40:
            continue
        if tipo == "calendario":
            meses = RE_MESES.findall(limpio[:200])
            mes_actual = meses[0].title() if meses else mes_actual
            seccion = f"Calendario Académico 2026 - {mes_actual or 'General'}"
        else:
            seccion = "Malla curricular - " + Path(nombre).stem.replace("malla_", "").replace("_", " ").title()
        texto = a_una_linea(limpio) if tipo == "malla" else limpio.replace("\n", " | ")
        for j, parte in enumerate(_subdividir(texto)):
            chunks.append(Chunk(
                chunk_id=f"{Path(nombre).stem[:30]}_p{i}_{j}",
                documento=nombre, tipo_doc=tipo,
                categoria_id=asignar_categoria(parte, tipo), seccion=seccion,
                paginas=[i], texto=parte, url=url,
            ))
    return chunks


def chunkear_pdf(pdf: Path) -> tuple[list[Chunk], str | None]:
    """Devuelve (chunks, advertencia). Advertencia != None si el PDF no tiene texto."""
    paginas = extraer_paginas(pdf)
    if es_escaneado(paginas):
        return [], f"{pdf.name}: PDF escaneado sin capa de texto (requiere OCR, ver TAREAS.md)"
    tipo = tipo_documento(pdf.name)
    url = _url_fuente(pdf.name)
    if tipo in ("calendario", "malla"):
        chunks = _chunks_por_pagina(pdf.name, paginas, tipo, url)
    else:
        chunks = _chunks_reglamento(pdf.name, paginas, tipo, url)
    for c in chunks:
        c.metadata = {"documento": c.documento, "tipo_doc": c.tipo_doc, "url": c.url,
                      "paginas": c.paginas, "articulo": c.articulo,
                      "hash": _id(c.texto)}
    return chunks, None
