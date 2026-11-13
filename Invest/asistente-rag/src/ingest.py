"""Fase 1 (indexación): lee las FAQs, las convierte en fragmentos y las guarda en Chroma.

Uso:  python -m src.ingest
"""
import re

import chromadb

from . import config, llm


def parsear_archivo(ruta) -> list[dict]:
    """Un fragmento por cada bloque '## Sección X | Pregunta' (chunking por estructura)."""
    texto = ruta.read_text(encoding="utf-8")
    m = re.search(r"^documento:\s*(.+)$", texto, re.MULTILINE)
    documento = m.group(1).strip() if m else ruta.stem
    cuerpo = re.split(r"^---\s*$", texto, maxsplit=2, flags=re.MULTILINE)[-1]

    fragmentos = []
    for bloque in re.split(r"^## ", cuerpo, flags=re.MULTILINE)[1:]:
        encabezado, _, respuesta = bloque.partition("\n")
        seccion, _, pregunta = encabezado.partition("|")
        seccion, pregunta, respuesta = seccion.strip(), pregunta.strip(), respuesta.strip()
        fragmentos.append({
            "id": f"{documento}#{seccion}",
            "documento": documento,
            "seccion": seccion,
            "pregunta": pregunta,
            # Se antepone el título del documento a cada fragmento (mejora la búsqueda)
            "texto": f"{documento} - {seccion}. {pregunta}\n{respuesta}",
        })
    return fragmentos


def cargar_fragmentos() -> list[dict]:
    fragmentos = []
    for ruta in sorted(config.DATA_DIR.glob("*.md")):
        fragmentos.extend(parsear_archivo(ruta))
    return fragmentos


def construir_indice() -> int:
    fragmentos = cargar_fragmentos()
    if not fragmentos:
        raise RuntimeError(f"No se encontraron FAQs en {config.DATA_DIR}")

    cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        cliente.delete_collection(config.COLLECTION)
    except Exception:  # noqa: BLE001  (si no existía, no pasa nada)
        pass
    col = cliente.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    embeddings = llm.embed([f["texto"] for f in fragmentos])
    col.add(
        ids=[f["id"] for f in fragmentos],
        documents=[f["texto"] for f in fragmentos],
        embeddings=embeddings,
        metadatas=[
            {"documento": f["documento"], "seccion": f["seccion"], "pregunta": f["pregunta"]}
            for f in fragmentos
        ],
    )
    return len(fragmentos)


if __name__ == "__main__":
    print(f"Modelo de embeddings: {config.EMBED_MODEL}")
    n = construir_indice()
    print(f"Listo: {n} fragmentos indexados en {config.CHROMA_DIR}")
