"""Configuración del asistente. Todo se puede cambiar con variables de entorno."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama (modelo local) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")      # ~4.7 GB, cabe en 8 GB de VRAM
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")         # embeddings multilingües
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")              # mantiene el modelo cargado en memoria
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))       # 0 = respuestas estables

# --- Recuperación ---
TOP_K = int(os.getenv("TOP_K", "3"))                     # fragmentos que se pasan al modelo
MIN_SIM = float(os.getenv("MIN_SIM", "0.50"))            # umbral de similitud (calibrar con evaluar.py)

# --- Rutas ---
DATA_DIR = BASE_DIR / "data" / "faqs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION = "faqs_ti"
LOG_FILE = BASE_DIR / "logs" / "trazas.jsonl"

# --- Otros ---
PROMPT_VERSION = "v1.0"
EMPRESA = "Empresa Demo"
CONTACTO_HUMANO = "extensión 2200 o soporte@empresa.cr"
MAX_PREGUNTA = 500                                       # caracteres máximos por pregunta
TICKET_INICIAL = 4821                                    # solo para la numeración de la demo
