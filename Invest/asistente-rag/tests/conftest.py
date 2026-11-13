"""Pruebas SIN Ollama: se reemplazan los embeddings y el chat por versiones simuladas.
Sirven para comprobar la lógica (guardrails, recuperación, traza), no la calidad del modelo real."""
import math
import re
import zlib

import pytest

from src import config, ingest, llm
from src.retrieval import tokenizar

FAKE = {"modo": "ok"}


def fake_embed(textos):
    salida = []
    for t in textos:
        v = [0.0] * 512
        for tok in tokenizar(t):
            v[zlib.crc32(tok.encode()) % 512] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        salida.append([x / norma for x in v])
    return salida


def fake_chat(system, user):
    modo = FAKE["modo"]
    if modo == "sin_info":
        texto = "SIN_INFORMACION"
    elif modo == "sin_cita":
        texto = "Sigue los pasos del manual."
    elif modo == "inventa":
        texto = "Entra a portal.falso.com y llama al 99999. [1]"
    else:
        texto = "Ingresa a identidad.empresa.cr y regenera el QR. [1]"
    return {"texto": texto, "tokens_in": 100, "tokens_out": 20}


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "trazas.jsonl")
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(llm, "estado", lambda: {"ollama": True, "error": "", "chat": True, "embed": True, "modelos": []})
    FAKE["modo"] = "ok"
    ingest.construir_indice()
    return tmp_path
