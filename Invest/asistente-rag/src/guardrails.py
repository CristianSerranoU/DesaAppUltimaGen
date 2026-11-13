"""Guardrails básicos de entrada y de salida (reglas simples, fáciles de explicar)."""
import re
import unicodedata


def _sin_acentos(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# ----------------------------------------------------------------------------
# ENTRADA 1: enmascarar datos personales (PII) antes de buscar o registrar
# ----------------------------------------------------------------------------
_PII = [
    ("contraseña", re.compile(r"(?i)\b(contrase[ñn]a|clave|password|pin)\s*(es|:|=)?\s*\S+")),
    ("tarjeta", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("cédula", re.compile(r"\b[1-9]-?\d{4}-?\d{4}\b")),
    ("teléfono", re.compile(r"(?<!\d)(?:\+?506[ -]?)?[2-8]\d{3}[ -]?\d{4}(?!\d)")),
    ("correo", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
]


def enmascarar_pii(texto: str) -> tuple[str, list[str]]:
    """Devuelve (texto enmascarado, lista de tipos de dato encontrados)."""
    tipos = []
    for nombre, patron in _PII:
        if patron.search(texto):
            tipos.append(nombre)
            texto = patron.sub(f"[{nombre.upper()}]", texto)
    return texto, tipos


# ----------------------------------------------------------------------------
# ENTRADA 2: detectar intentos de prompt injection
# ----------------------------------------------------------------------------
_INYECCION = [
    r"ignora (todas |todo |las |tus )*(instrucciones|reglas)",
    r"olvida (todo|tus instrucciones|las instrucciones|las reglas)",
    r"ignore (all |any |the )*(previous|prior|above) (instructions|rules)",
    r"(muestra|revela|dime|imprime|repite).{0,40}(prompt|instrucciones) (del )?(sistema|interno|inicial)",
    r"system prompt",
    r"developer mode|modo (desarrollador|dios)|jailbreak",
    r"(actua|finge|haz de cuenta) (como|que eres) .{0,30}(administrador|root|sin restricciones)",
]
_INYECCION_RE = [re.compile(p) for p in _INYECCION]


def detectar_inyeccion(texto: str) -> bool:
    t = _sin_acentos(texto.lower())
    return any(p.search(t) for p in _INYECCION_RE)


# ----------------------------------------------------------------------------
# SALIDA 1: la respuesta debe citar fragmentos que existen
# ----------------------------------------------------------------------------
def validar_citas(respuesta: str, n_fragmentos: int) -> tuple[bool, list[int]]:
    citas = sorted({int(n) for n in re.findall(r"\[(\d+)\]", respuesta)})
    if not citas:
        return False, []
    return all(1 <= c <= n_fragmentos for c in citas), citas


# ----------------------------------------------------------------------------
# SALIDA 2: URLs, correos y números largos de la respuesta deben estar en el contexto
# ----------------------------------------------------------------------------
_DATOS = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|https?://[^\s)\]]+"
    r"|\b(?:[\w-]+\.)+(?:com|cr|org|net|edu|io)\b"
    r"|\b\d{3,}\b",
    re.IGNORECASE,
)


def verificar_fundamentacion(respuesta: str, contexto: str) -> list[str]:
    """Devuelve los datos de la respuesta que NO aparecen en el contexto (posibles inventos)."""
    ctx = contexto.lower()
    sin_soporte = []
    for m in _DATOS.finditer(respuesta):
        dato = m.group(0).rstrip(".,;:").lower()
        if dato and dato not in ctx:
            sin_soporte.append(dato)
    return sorted(set(sin_soporte))
