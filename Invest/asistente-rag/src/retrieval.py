"""Fase 2 (consulta), recuperación híbrida: búsqueda semántica (Chroma) + palabras clave (BM25),
combinadas con Reciprocal Rank Fusion (RRF)."""
import re
import unicodedata

import chromadb
from rank_bm25 import BM25Okapi

from . import config, llm

_STOP = {
    "de", "la", "el", "en", "y", "a", "que", "un", "una", "los", "las", "por", "para", "con", "mi",
    "me", "se", "es", "al", "del", "lo", "como", "o", "su", "si", "no", "hay", "puedo", "debo",
    "hacer", "tengo", "esta", "este", "mis", "te", "tu", "le", "ya", "muy",
}


def tokenizar(texto: str) -> list[str]:
    t = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    tokens = [w for w in re.findall(r"\w+", t) if w not in _STOP and len(w) > 1]
    return [w[:6] for w in tokens]  # "stemming" muy simple: restablecer/restablezco -> restab


def obtener_coleccion():
    cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return cliente.get_collection(config.COLLECTION)


class Recuperador:
    def __init__(self, coleccion=None):
        self.col = coleccion if coleccion is not None else obtener_coleccion()
        datos = self.col.get(include=["documents", "metadatas"])
        self.ids = datos["ids"]
        self.docs = datos["documents"]
        self.metas = datos["metadatas"]
        self.bm25 = BM25Okapi([tokenizar(d) for d in self.docs])

    def __len__(self) -> int:
        return len(self.ids)

    def buscar(self, pregunta: str, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K
        n = len(self.ids)

        # 1) semántica (con tan pocos fragmentos se pide el ranking completo)
        q = llm.embed([pregunta])[0]
        r = self.col.query(query_embeddings=[q], n_results=n, include=["distances"])
        ids_v, dist = r["ids"][0], r["distances"][0]
        sim = {i: 1 - d for i, d in zip(ids_v, dist)}
        rango_v = {i: pos + 1 for pos, i in enumerate(ids_v)}

        # 2) palabras clave
        puntajes = self.bm25.get_scores(tokenizar(pregunta))
        orden = sorted(range(n), key=lambda i: puntajes[i], reverse=True)
        rango_b = {self.ids[i]: pos + 1 for pos, i in enumerate(orden) if puntajes[i] > 0}
        bm = {self.ids[i]: float(puntajes[i]) for i in range(n)}

        # 3) fusión RRF
        rrf = {i: 1 / (60 + rango_v[i]) + (1 / (60 + rango_b[i]) if i in rango_b else 0) for i in self.ids}
        mejores = sorted(self.ids, key=lambda i: rrf[i], reverse=True)[:k]

        resultado = []
        for pos, i in enumerate(mejores, start=1):
            idx = self.ids.index(i)
            meta = self.metas[idx]
            resultado.append({
                "rank": pos,
                "id": i,
                "documento": meta["documento"],
                "seccion": meta["seccion"],
                "pregunta": meta["pregunta"],
                "texto": self.docs[idx],
                "sim": round(sim[i], 4),
                "bm25": round(bm[i], 3),
                "rrf": round(rrf[i], 5),
            })
        return resultado
