"""Acceso a Ollama (embeddings y chat). Es el único módulo que habla con el modelo."""
import ollama

from . import config

_client = None


def client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def _get(obj, name, default=None):
    """Lee un campo tanto si la respuesta es un objeto como si es un diccionario."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def embed(textos: list[str]) -> list[list[float]]:
    r = client().embed(model=config.EMBED_MODEL, input=textos, keep_alive=config.KEEP_ALIVE)
    return [list(v) for v in _get(r, "embeddings")]


def chat(system: str, user: str) -> dict:
    r = client().chat(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": config.TEMPERATURE},
        keep_alive=config.KEEP_ALIVE,
    )
    msg = _get(r, "message")
    return {
        "texto": (_get(msg, "content", "") or "").strip(),
        "tokens_in": _get(r, "prompt_eval_count", 0) or 0,
        "tokens_out": _get(r, "eval_count", 0) or 0,
    }


def _nombres_modelos() -> list[str]:
    r = client().list()
    modelos = _get(r, "models", []) or []
    return [(_get(m, "model") or _get(m, "name") or "") for m in modelos]


def _esta(nombre: str, instalados: list[str]) -> bool:
    return any(i == nombre or i == f"{nombre}:latest" or i.split(":")[0] == nombre for i in instalados)


def estado() -> dict:
    """Diagnóstico para la barra lateral: ¿Ollama responde y están los modelos?"""
    try:
        instalados = _nombres_modelos()
    except Exception as e:  # noqa: BLE001
        return {"ollama": False, "error": str(e), "chat": False, "embed": False, "modelos": []}
    return {
        "ollama": True,
        "error": "",
        "chat": _esta(config.CHAT_MODEL, instalados),
        "embed": _esta(config.EMBED_MODEL, instalados),
        "modelos": instalados,
    }
