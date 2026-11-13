"""Evalúa SOLO la recuperación (rápido, no usa el modelo de chat) y sugiere el umbral MIN_SIM.

Uso:  python -m src.evaluar
"""
import csv
from pathlib import Path

from . import config
from .retrieval import Recuperador

RUTA = Path(__file__).resolve().parent.parent / "eval" / "preguntas_prueba.csv"


def main() -> None:
    rec = Recuperador()
    filas = list(csv.DictReader(open(RUTA, encoding="utf-8")))

    sims_dentro, sims_fuera, aciertos, total_dentro = [], [], 0, 0
    print(f"{'tipo':7} {'sim':>5}  {'ok':3} pregunta")
    for f in filas:
        chunks = rec.buscar(f["pregunta"])
        mejor = max(c["sim"] for c in chunks)
        if f["tipo"] == "dentro":
            total_dentro += 1
            acierto = f["esperado"] in {c["documento"] for c in chunks}
            aciertos += acierto
            sims_dentro.append(mejor)
            marca = "si" if acierto else "NO"
        else:
            sims_fuera.append(mejor)
            marca = "-"
        print(f"{f['tipo']:7} {mejor:5.2f}  {marca:3} {f['pregunta'][:70]}")

    print(f"\nHit@{config.TOP_K} (el documento esperado está en el top-{config.TOP_K}): {aciertos}/{total_dentro}")
    min_dentro, max_fuera = min(sims_dentro), max(sims_fuera)
    print(f"Similitud mínima de preguntas DENTRO del dominio: {min_dentro:.2f}")
    print(f"Similitud máxima de preguntas FUERA del dominio:  {max_fuera:.2f}")
    if min_dentro > max_fuera:
        print(f"Umbral sugerido MIN_SIM = {(min_dentro + max_fuera) / 2:.2f}  (actual: {config.MIN_SIM})")
    else:
        print("Las similitudes se traslapan: no hay un umbral perfecto. Revisa las preguntas con 'NO' o agrega más FAQs.")


if __name__ == "__main__":
    main()
