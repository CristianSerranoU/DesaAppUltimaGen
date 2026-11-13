# Asistente de soporte con LLM (demo RAG local)

Demo del Caso 1 (BIS-28): chatbot de Mesa de Ayuda TI que responde solo con las FAQs indexadas,
con guardrails básicos y trazabilidad de cada consulta. Todo corre local, sin pagar:
**Ollama + Chroma + Streamlit**.

> El contenido de `data/faqs/` es **ficticio** (Empresa Demo). Reemplácenlo por sus propias FAQs si quieren.

## Requisitos
- Windows con **Python 3.11 o 3.12** y **VS Code**
- **Ollama** instalado (https://ollama.com)
- Laptop de referencia: 16 GB RAM + RTX 4060 (el modelo de 7B cabe en la GPU)

## Puesta en marcha (PowerShell, dentro de la carpeta del proyecto)

```powershell
# 1) Modelos locales (una sola vez, ~6 GB de descarga)
ollama pull qwen2.5:7b
ollama pull bge-m3

# 2) Entorno de Python
python -m venv .venv
.venv\Scripts\Activate.ps1      # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt

# 3) Pruebas sin Ollama (opcional, 5 segundos)
python -m pytest

# 4) Indexar las FAQs en Chroma
python -m src.ingest

# 5) Calibrar el umbral de dominio (opcional pero recomendado)
python -m src.evaluar

# 6) Abrir la demo
streamlit run app.py
```

## Estructura
| Ruta | Qué hace |
|---|---|
| `data/faqs/*.md` | Base de conocimiento (un fragmento por pregunta/respuesta) |
| `src/ingest.py` | Fase 1: chunking por estructura + embeddings + Chroma |
| `src/retrieval.py` | Búsqueda híbrida (semántica + BM25) fusionada con RRF |
| `src/guardrails.py` | Enmascarar PII, detectar prompt injection, validar citas y datos |
| `src/pipeline.py` | Flujo completo de una consulta y prompt del sistema |
| `src/trace.py` | Escribe `logs/trazas.jsonl` (una línea por consulta) |
| `src/evaluar.py` | Mide la recuperación y sugiere el umbral `MIN_SIM` |
| `app.py` | Interfaz: chat + panel de trazabilidad |

## Guion de la demo (barra lateral de la app)
1. **Con respaldo:** responde con pasos y cita la fuente.
2. **Fuera de tema:** no llama al modelo y escala.
3. **Dato personal:** enmascara cédula y correo; el log no guarda los datos reales.
4. **Prompt injection:** bloquea la consulta.
5. **Sin respaldo:** no inventa; escala a un agente humano.

Al final, abrir `logs/trazas.jsonl` (o el expander del panel derecho) para mostrar la trazabilidad.

## Configuración (variables de entorno, opcional)
`CHAT_MODEL` (por defecto `qwen2.5:7b`), `EMBED_MODEL` (`bge-m3`), `MIN_SIM` (`0.50`), `TOP_K` (`3`).
Ejemplo: `$env:CHAT_MODEL="llama3.1:8b"; streamlit run app.py`

## Problemas frecuentes
- **"Ollama NO responde":** abre la aplicación Ollama o ejecuta `ollama serve`.
- **"Falta un modelo":** `ollama pull <nombre>` y revisa con `ollama list`.
- **"No hay índice":** ejecuta `python -m src.ingest`.
- **Respuestas lentas la primera vez:** el modelo se está cargando a la GPU; la app lo precalienta al abrir.
- **Todo sale "fuera de dominio" (o nada):** ajusta el slider de umbral o ejecuta `python -m src.evaluar`.

## Qué es real y qué no (para la presentación)
- La **latencia, los tokens y el costo ($0)** del panel se miden de verdad en cada consulta.
- No hay un *reranker* separado: la fusión híbrida (RRF) hace de ordenamiento final.
- No se calcula **RAGAS** en vivo. El panel muestra verificaciones simples (citas válidas y datos sin soporte en el contexto).
