"""Trazabilidad: cada consulta se guarda como una línea JSON en logs/trazas.jsonl."""
import json

from . import config


def guardar(registro: dict) -> None:
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def leer_ultimas(n: int = 5) -> list[dict]:
    if not config.LOG_FILE.exists():
        return []
    with open(config.LOG_FILE, encoding="utf-8") as f:
        lineas = [ln for ln in f.read().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lineas[-n:]]
