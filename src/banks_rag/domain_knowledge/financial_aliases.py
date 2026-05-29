"""Alias y ruteo del dominio financiero del BCCh.

Fuente de verdad compartida (paridad con el gemelo Mac) que conecta el
**vocabulario del usuario/LLM** con la estructura real del catálogo de parquets.
Resuelve tres problemas de los logs del frontend:

1. **Descubrimiento débil** — "DV01 fondos de pensiones" no encontraba
   ``dv01_spc_afp`` porque el dataset usa "afp", no "pensiones". Las
   ``expansions`` agregan sinónimos a la query antes del scoring léxico.
2. **Segmento inexistente** — ``discover_query(segment="AFP")`` reventaba; el
   segmento real es ``fondos_pension``. ``resolve_segment`` mapea el alias.
3. **Ruteo a especialistas** (Fase 1) — ``specialist_for`` enruta la pregunta
   al subagente de mercado correcto.

Hechos del catálogo (verificados): AFP y FFMM viven en ``fondos_pension``; los
no residentes (NR) son **transversales** (``posiciones_rfl`` + ``mercado_cambiario``
+ ``posiciones_cambiarias``), así que para NR/AFP/FFMM/seguros los
``dataset_hints`` (substrings de id) son señal más confiable que el segmento.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Segmentos canónicos del parquet_catalog (los 10 con datos + 'otros').
CANONICAL_SEGMENTS: frozenset[str] = frozenset({
    "renta_fija_chile",
    "mercado_cambiario",
    "fondos_pension",
    "spreads_credito",
    "liquidez_bancaria",
    "expectativas",
    "posiciones_rfl",
    "instrumentos_bcch",
    "posiciones_cambiarias",
    "balance_bancario",
    "otros",
})


@dataclass(frozen=True)
class Concept:
    """Un concepto del dominio y cómo se traduce al catálogo.

    Campos:
        name: identificador del concepto (para tests/logs).
        keywords: disparadores en la query (lowercase, se buscan como substring).
        expansions: tokens extra que se anexan a la query para el scoring léxico.
        segment: segmento canónico si el concepto mapea limpio (o None).
        dataset_hints: substrings de ``id`` de dataset a boostear en el ranking.
        specialist: key del subagente especialista (Fase 1) o None.
    """

    name: str
    keywords: tuple[str, ...]
    expansions: tuple[str, ...] = ()
    segment: str | None = None
    dataset_hints: tuple[str, ...] = ()
    specialist: str | None = None


# El orden importa solo para ``specialist_for`` (primer match gana); por eso los
# conceptos más específicos (NR, AFP, FFMM) van antes que los genéricos.
CONCEPTS: tuple[Concept, ...] = (
    Concept(
        name="no_residentes",
        # "extranjero" a secas es ambiguo (AFP invirtiendo afuera vs. inversor
        # extranjero en Chile); se exige el contexto "no residente"/"inversionista
        # extranjero" para no robarle queries de AFP.
        keywords=("no residente", "no residentes", "noresidente", " nr ", "nr,",
                  "inversionista extranjero", "inversionistas extranjeros",
                  "capitales extranjeros"),
        expansions=("nr", "no", "residentes", "extranjeros", "offshore"),
        segment=None,  # transversal
        dataset_hints=("_nr", "nr_", "posicion_rfl_nr", "flujo_spot_nr",
                       "posicion_nr_derivados", "posicion_nr_spc",
                       "variacion_rfl_dcv_nr"),
        specialist="no_residentes",
    ),
    Concept(
        name="fondos_mutuos",
        keywords=("fondo mutuo", "fondos mutuos", "ffmm", "fm "),
        expansions=("ffmm", "fondos", "mutuos"),
        segment="fondos_pension",
        dataset_hints=("ffmm", "flujos_ffmm", "flujos_acum_ffmm", "duracion_ffmm",
                       "dv01_ffmm", "dcv_composicion_ffmm", "flujos_spot_ffmm",
                       "dap_pdbc_ffmm"),
        specialist="fondos_mutuos",
    ),
    Concept(
        name="afp_pensiones",
        keywords=("afp", "fondo de pension", "fondos de pension", "fondo de pensiones",
                  "fondos de pensiones", "pension", "pensiones", "multifondo",
                  "multifondos", "cartera afp", "composicion de cartera"),
        expansions=("afp", "fondos", "pension", "pensiones", "multifondos",
                    "allocation", "cartera"),
        segment="fondos_pension",
        dataset_hints=("afp", "allocation", "stock_fondo_afp", "dv01_spc_afp",
                       "mtm_afp", "attribution", "cambiario_afp",
                       "posicion_rfl_afp", "spot_derivados_afp",
                       "allocation_int_nac", "variacion_dcv_afp"),
        specialist="afp",
    ),
    Concept(
        name="seguros",
        keywords=("seguro", "seguros", "aseguradora", "aseguradoras",
                  "compania de seguro", "compañia de seguro", "compañías de seguros"),
        expansions=("seguro", "seguros", "aseguradoras"),
        segment="posiciones_rfl",
        dataset_hints=("variacion_seguro", "seguro"),
        specialist="renta_fija",
    ),
    Concept(
        name="liquidez_bancaria",
        keywords=("liquidez", "lcr", "nsfr", "caja", "encaje", "reserva tecnica",
                  "rt exigible"),
        expansions=("liquidez", "lcr", "nsfr", "caja", "circulante", "rt"),
        segment="liquidez_bancaria",
        dataset_hints=("lcr", "nsfr", "caja", "rt_", "operaciones_liquidez",
                       "ratio_liquidez", "liquidez_mx_ihh"),
        specialist="liquidez",
    ),
    Concept(
        name="balance_bancario",
        # "dap" a secas es ambiguo (depósito a plazo aparece en balance Y en
        # spreads de crédito); se exige el contexto de balance.
        keywords=("balance bancario", "activos del banco", "activos bancarios",
                  "pasivos bancarios", "colocaciones", "depositos a plazo"),
        expansions=("activos", "pasivos", "balance", "banco", "colocaciones"),
        segment="balance_bancario",
        dataset_hints=("act_mn", "act_mx", "pas_mn", "pas_mx", "bonos_y_dap"),
        specialist="liquidez",
    ),
    Concept(
        name="spreads_credito",
        keywords=("spread de credito", "spreads de credito", "spread dap",
                  "prime", "spread prime", "tado", "sos"),
        expansions=("spread", "credito", "dap", "prime", "swap"),
        segment="spreads_credito",
        dataset_hints=("spreads_", "spread_dap", "spread_dap_prime"),
        specialist="renta_fija",
    ),
    Concept(
        name="expectativas",
        keywords=("expectativa", "expectativas", "tpm implicita", "tpm esperada",
                  "mipr", "encuesta", "eee", "eof"),
        expansions=("expectativas", "tpm", "implicita", "mipr"),
        segment="expectativas",
        dataset_hints=("expectativas", "mipr", "spread_tpm_fed_imp"),
        specialist="policy",
    ),
    Concept(
        name="renta_fija",
        keywords=("renta fija", "rfl", "bono", "bonos", "curva", "btp", "btu",
                  "bei", "swap", "spc", "ois", "pdbc", "tasa larga", "soberano"),
        expansions=("renta", "fija", "bonos", "curva", "tasa", "swap", "tenor"),
        segment="renta_fija_chile",
        dataset_hints=("btp", "btu", "bei", "spc", "ois", "spread_btp",
                       "spread_swap", "monto_btpbtu", "vol_btpbtu", "pendiente",
                       "pdbc"),
        specialist="renta_fija",
    ),
    Concept(
        name="cambiario_fx",
        keywords=("tipo de cambio", "dolar", "dólar", "usd/clp", "usdclp", "clp",
                  "cambiario", "spot", "forward", "fx", "paridad", "divisa",
                  "moneda"),
        expansions=("tipo", "cambio", "dolar", "clp", "usd", "cambiario", "spot",
                    "forward"),
        segment="mercado_cambiario",
        dataset_hints=("clp_monto", "forward_points", "flujo_cambiario", "bid_ask",
                       "posicion_spot_derivados", "fixing", "var_moneda",
                       "volatilidad_precio", "flujo_spot"),
        specialist="fx",
    ),
    Concept(
        name="commodities",
        keywords=("cobre", "petroleo", "petróleo", "wti", "commodity", "commodities",
                  "dxy"),
        expansions=("cobre", "petroleo", "commodities", "dxy"),
        segment="mercado_cambiario",
        dataset_hints=("cobre_dxy", "petroleo_tcn", "inflacion_wti"),
        specialist="fx",
    ),
    # Métricas transversales: no fijan segmento ni especialista, solo boostean
    # los datasets cuya id contiene la métrica y enriquecen la query.
    Concept(
        name="dv01",
        keywords=("dv01", "sensibilidad de tasa", "sensibilidad a la tasa"),
        expansions=("dv01", "sensibilidad", "duracion"),
        dataset_hints=("dv01",),
    ),
    Concept(
        name="mtm",
        keywords=("mtm", "mark to market", "mark-to-market", "valor de mercado",
                  "valorizacion"),
        expansions=("mtm", "valor", "mercado"),
        dataset_hints=("mtm",),
    ),
    Concept(
        name="duracion",
        keywords=("duracion", "duración", "duration"),
        expansions=("duracion",),
        dataset_hints=("duracion", "dv01"),
    ),
    Concept(
        name="atribucion",
        keywords=("atribucion", "atribución", "attribution"),
        expansions=("atribucion", "attribution"),
        dataset_hints=("attribution",),
    ),
)


# ── Lookups derivados ───────────────────────────────────────────────────────

# alias (lowercase) → segmento canónico, para resolver el parámetro `segment`.
_SEGMENT_ALIASES: dict[str, str] = {
    "afp": "fondos_pension",
    "afps": "fondos_pension",
    "pension": "fondos_pension",
    "pensiones": "fondos_pension",
    "fondos de pension": "fondos_pension",
    "fondos de pensiones": "fondos_pension",
    "ffmm": "fondos_pension",
    "fondos mutuos": "fondos_pension",
    "fx": "mercado_cambiario",
    "cambiario": "mercado_cambiario",
    "tipo de cambio": "mercado_cambiario",
    "divisas": "mercado_cambiario",
    "commodities": "mercado_cambiario",
    "renta fija": "renta_fija_chile",
    "rf": "renta_fija_chile",
    "bonos": "renta_fija_chile",
    "curvas": "renta_fija_chile",
    "liquidez": "liquidez_bancaria",
    "balance": "balance_bancario",
    "spreads": "spreads_credito",
    "credito": "spreads_credito",
    "expectativas": "expectativas",
    "rfl": "posiciones_rfl",
    "posiciones": "posiciones_rfl",
    "instrumentos bcch": "instrumentos_bcch",
    "pdbc": "instrumentos_bcch",
}


def _strip_accents(text: str) -> str:
    """Quita acentos para que 'crédito' matchee la keyword 'credito'."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _norm(text: str) -> str:
    return f" {_strip_accents(text.lower()).strip()} "


def matched_concepts(query: str) -> list[Concept]:
    """Conceptos cuyo keyword aparece en la query (substring, sin acentos/case)."""
    haystack = _norm(query)
    out: list[Concept] = []
    for c in CONCEPTS:
        if any(_strip_accents(kw) in haystack for kw in c.keywords):
            out.append(c)
    return out


def expand_query(query: str) -> str:
    """Anexa sinónimos del dominio a la query para mejorar el scoring léxico.

    No reemplaza la query: la enriquece. "DV01 de fondos de pensiones" →
    añade {dv01, sensibilidad, duracion, afp, fondos, pension, allocation…}
    de modo que datasets como ``dv01_spc_afp`` suban en el ranking.
    """
    extra: list[str] = []
    for c in matched_concepts(query):
        extra.extend(c.expansions)
    if not extra:
        return query
    # Dedup preservando orden, sin repetir tokens ya presentes en la query.
    present = set(re.findall(r"[a-z0-9]+", query.lower()))
    seen: set[str] = set()
    appended: list[str] = []
    for tok in extra:
        if tok in present or tok in seen:
            continue
        seen.add(tok)
        appended.append(tok)
    return f"{query} {' '.join(appended)}" if appended else query


def dataset_hints(query: str) -> set[str]:
    """Substrings de id de dataset a boostear, según los conceptos detectados."""
    hints: set[str] = set()
    for c in matched_concepts(query):
        hints.update(c.dataset_hints)
    return hints


def resolve_segment(value: str | None, valid_segments: set[str] | None = None) -> str | None:
    """Resuelve un ``segment`` (canónico o alias) al nombre canónico del catálogo.

    Args:
        value: el segmento pedido por el LLM (puede ser un alias como "AFP").
        valid_segments: si se pasa, restringe el resultado a segmentos que
            existen realmente en el catálogo cargado.

    Returns:
        El segmento canónico, o ``None`` si no se reconoce (el caller debe
        entonces buscar en todos los segmentos en vez de fallar).
    """
    if not value:
        return None
    v = value.strip()
    candidates = valid_segments if valid_segments is not None else CANONICAL_SEGMENTS
    if v in candidates:
        return v
    mapped = _SEGMENT_ALIASES.get(v.lower())
    if mapped and (valid_segments is None or mapped in valid_segments):
        return mapped
    return None


def specialist_for(query: str) -> str | None:
    """Enruta una pregunta al especialista de mercado (Fase 1). Primer match gana."""
    for c in matched_concepts(query):
        if c.specialist:
            return c.specialist
    return None
