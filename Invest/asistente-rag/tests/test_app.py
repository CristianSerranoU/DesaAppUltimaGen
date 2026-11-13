"""Prueba de humo de la interfaz Streamlit (con el modelo simulado)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_la_app_arranca_y_responde(entorno):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.sidebar.slider[0].set_value(0.15)  # umbral para embeddings simulados
    at.chat_input[0].set_value("¿Cómo restablezco mi token MFA para entrar a la VPN desde casa?").run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown)
    assert "identidad.empresa.cr" in textos


def test_boton_de_guion(entorno):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.slider[0].set_value(0.15)
    # el botón de prompt injection (4.º del guion)
    boton = [b for b in at.sidebar.button if "injection" in b.label][0]
    boton.click().run()
    assert not at.exception
    assert any("bloqueado" in m.value.lower() for m in at.markdown)
