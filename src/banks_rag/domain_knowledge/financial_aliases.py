"""Alias y ruteo del dominio financiero del BCCh.

Fuente de verdad compartida (paridad con el gemelo Mac) que conecta el
**vocabulario del usuario/LLM** con la estructura real del catálogo de parquets.
Resuelve tres problemas de los logs del frontend:

1. **Descubrimiento débil** — "DV01 fondos de pensiones" no encontraba
   ``dv01_spc_afp`` porque el dataset usa "afp", no "pensiones". Las
   ``expansions`` agregan sinónimos a la query antes del scoring léxico.
2. **Segmento inexistente** — ``discover_query(segment="AFP")`` reventaba; el
   segmento real es ``afp``. ``resolve_segment`` mapea el alias.
3. **Ruteo a especialistas** (Fase 1) — ``specialist_for`` enruta la pregunta
   al subagente de mercado correcto.

Segmentos canónicos del catálogo (``sql_catalog/parquet_catalog.yaml``):
``afp``, ``bancos``, ``color_mercados``, ``csv_seguros``, ``ffmm``, ``fisco``,
``fx``, ``fx_diferencial``, ``inflacion``, ``liquidez_mn``, ``liquidez_mx``,
``mercado_monetario``, ``no_residentes``, ``rf_tasas``, ``rf_volumenes``,
``spc_ois``, ``todos``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Segmentos canónicos del parquet_catalog (17 segmentos activos).
CANONICAL_SEGMENTS: frozenset[str] = frozenset({
    "afp",
    "bancos",
    "color_mercados",
    "csv_seguros",
    "ffmm",
    "fisco",
    "fx",
    "fx_diferencial",
    "inflacion",
    "liquidez_mn",
    "liquidez_mx",
    "mercado_monetario",
    "no_residentes",
    "rf_tasas",
    "rf_volumenes",
    "spc_ois",
    "todos",
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
        segment="no_residentes",
        dataset_hints=("flujo_spot_nr", "posicion_nr_derivados",
                       "posicion_nr_spc", "posicion_rfl_nr", "spot_acumulado_agente",
                       "spot_susc_vcto_agente", "flujos_acumulados_derivados",
                       "susc_vcto_agente_instrumento", "variacion_rfl_dcv_nr",
                       "spot_agente_afecto_derivado", "posicion_spot_derivados_nr"),
        specialist="no_residentes",
    ),
    Concept(
        name="fondos_mutuos",
        keywords=("fondo mutuo", "fondos mutuos", "ffmm", "fm "),
        expansions=("ffmm", "fondos", "mutuos"),
        segment="ffmm",
        dataset_hints=("ffmm", "flujos_ffmm", "flujos_acum_ffmm", "duracion_ffmm",
                       "dv01_ffmm", "dcv_composicion_ffmm", "flujos_spot_ffmm",
                       "dap_pdbc_ffmm", "allocation", "stock_bucket_ffmm",
                       "stock_nivel_ffmm", "variacion_stock_ffmm"),
        specialist="fondos_mutuos",
    ),
    Concept(
        name="afp_pensiones",
        keywords=("afp", "fondo de pension", "fondos de pension", "fondo de pensiones",
                  "fondos de pensiones", "pension", "pensiones", "multifondo",
                  "multifondos", "cartera afp", "composicion de cartera"),
        expansions=("afp", "fondos", "pension", "pensiones", "multifondos",
                    "allocation", "cartera"),
        segment="afp",
        # Sin patrones genéricos ("_afp", "afp_"): inflan todos los datasets del
        # segmento por igual y confunden el ranking para queries específicas.
        dataset_hints=("allocation_int_nac", "stock_fondo_afp",
                       "dv01_spc_afp", "mtm_afp", "cambiario_afp",
                       "spot_derivados_afp", "stock_bucket_afp",
                       "stock_nivel_afp", "variacion_stock_afp",
                       "movimientos_fondos", "attribution"),
        specialist="afp",
    ),
    Concept(
        name="seguros",
        keywords=("seguro", "seguros", "aseguradora", "aseguradoras",
                  "compania de seguro", "compañia de seguro", "compañías de seguros",
                  "csv"),
        expansions=("seguro", "seguros", "aseguradoras", "csv"),
        segment="csv_seguros",
        dataset_hints=("_csv", "csv_", "duracion_rfl_csv", "flujo_spot_csv",
                       "posicion_csv", "stock_bucket_csv", "stock_nivel_csv",
                       "variacion_stock_csv"),
        specialist="renta_fija",
    ),
    Concept(
        name="fisco",
        keywords=("tgr", "tesoro", "hacienda", "inversion fiscal", "tgr inversiones"),
        expansions=("tgr", "fisco", "tesoro", "hacienda"),
        segment="fisco",
        dataset_hints=("inversiones_tgr",),
        specialist="fx",
    ),
    Concept(
        name="liquidez_mn",
        keywords=("liquidez", "encaje", "reserva tecnica", "rt exigible",
                  "pdbc", "stock pdbc", "operaciones liquidez", "fpd", "fpl",
                  "interbancario"),
        expansions=("liquidez", "pdbc", "encaje", "circulante", "rt", "tib"),
        segment="liquidez_mn",
        dataset_hints=("pdbc", "rt_", "operaciones_liquidez", "fpd_",
                       "fpl_", "interbancario", "caja_y_circulante",
                       "duracion_pdbc"),
        specialist="liquidez",
    ),
    Concept(
        name="liquidez_mx",
        keywords=("lcr", "nsfr", "ratio de liquidez", "liquidez en dolares",
                  "liquidez moneda extranjera"),
        expansions=("lcr", "nsfr", "ratio", "liquidez", "mx"),
        segment="liquidez_mx",
        dataset_hints=("lcr", "nsfr", "liquidez_mx", "ratio_liquidez"),
        specialist="liquidez",
    ),
    Concept(
        name="balance_bancario",
        # "dap" a secas es ambiguo (depósito a plazo aparece en balance Y en
        # spreads de crédito); se exige el contexto de balance.
        keywords=("balance bancario", "activos del banco", "activos bancarios",
                  "pasivos bancarios", "colocaciones", "depositos a plazo",
                  "inversion bancaria", "caja banco", "bono bancario"),
        expansions=("activos", "pasivos", "balance", "banco", "colocaciones"),
        segment="bancos",
        dataset_hints=("act_mn", "act_mx", "pas_mn", "pas_mx", "bonos_y_dap",
                       "caja_bancos", "inv_por_banco", "inv_por_instrumento",
                       "rt_constitucion", "rt_exigible", "emision_bb",
                       "stock_bucket_bancos", "stock_nivel_bancos",
                       "variacion_stock_bancos", "stock_dap_bb",
                       "perfil_vencimiento"),
        specialist="liquidez",
    ),
    Concept(
        name="spreads_credito",
        keywords=("spread de credito", "spreads de credito", "spread dap",
                  "prime", "spread prime", "tado", "sos", "onshore",
                  "on shore", "spread tib", "spread mercado monetario"),
        expansions=("spread", "credito", "dap", "prime", "swap"),
        segment="liquidez_mx",
        dataset_hints=("spreads_", "spread_dap", "spread_prime", "spreads_12m",
                       "spreads_1m", "spreads_3m", "spreads_6m", "spreads_sos",
                       "spreads_tado"),
        specialist="renta_fija",
    ),
    Concept(
        name="mercado_monetario",
        keywords=("mercado monetario", "tib", "dap transado", "stock instrumento",
                  "stock por sector", "spread dap swap"),
        expansions=("mercado", "monetario", "tib", "dap"),
        segment="mercado_monetario",
        dataset_hints=("tib_monto_transado", "dap_flujos_plazo",
                       "spread_dap_swap", "spread_dap_prime_uf_swap",
                       "stock_por_instrumento", "stock_por_sector"),
        specialist="liquidez",
    ),
    Concept(
        name="expectativas",
        keywords=("expectativa", "expectativas", "tpm implicita", "tpm esperada",
                  "mipr", "encuesta", "eee", "eof", "expectativa inflacion"),
        expansions=("expectativas", "tpm", "implicita", "mipr"),
        segment="inflacion",
        dataset_hints=("expectativas_inflacion", "variacion_expectativas",
                       "mipr_cl", "mipr_us", "mipr_var",
                       "spread_tpm_fed_imp"),
        specialist="policy",
    ),
    Concept(
        name="inflacion",
        keywords=("inflacion", "inflación", "ipe", "ipc", "breakeven",
                  "bei", "ci ", "si ", "swap inflacion"),
        expansions=("inflacion", "ipc", "bei", "breakeven"),
        segment="inflacion",
        dataset_hints=("ci_spc", "si_por_afp", "si_por_agente",
                       "variacion_seguro", "inflacion_wti"),
        specialist="policy",
    ),
    Concept(
        name="renta_fija",
        keywords=("renta fija", "rfl", "bono", "bonos", "curva", "btp", "btu",
                  "swap", "spc", "ois", "pdbc", "tasa larga", "soberano",
                  "deuda soberana", "curva de tasas"),
        # "tenor" removido: al expandir "spread swap OIS" no debe añadir "tenor"
        # que infla datasets de curvas por tenor que no son el resultado buscado.
        expansions=("renta", "fija", "bonos", "curva", "tasa", "swap"),
        segment="rf_tasas",
        # IDs específicos; sin patrones "btp_"/"btu_" que inflan todos los datasets BTP.
        dataset_hints=("bei_btp_btu_plazo", "btp_curva", "btp_plazo",
                       "btu_curva", "btu_plazo", "evolucion_tasas_btp_btu",
                       "pendiente_btp_btu", "spread_btp_spc", "spread_btp_ust",
                       "monto_btpbtu", "vol_btpbtu", "monto_bb", "monto_bc",
                       "volatilidad_bb"),
        specialist="renta_fija",
    ),
    Concept(
        name="spc_ois",
        keywords=("spc", "ois", "swap promedio camara", "mipr curva",
                  "curva ois", "curva spc", "spread spc ois"),
        # "mipr" removido: al expandir "spread swap OIS" no debe añadir
        # "mipr" como token — infla mipr_spread_hist que no es el dataset buscado.
        expansions=("spc", "ois", "swap", "curva"),
        segment="spc_ois",
        # Sin prefijos genéricos ("spc_", "ois_"): inflan todos los datasets de
        # curvas por igual. Se usan solo IDs específicos para desambiguar.
        dataset_hints=("spread_swap_ois", "spread_cl_us_curve", "spc_ois_var",
                       "spc_curve", "spc_hist", "spc_plazo", "ois_hist", "ois_plazo",
                       "mipr_cl_curve", "mipr_cl_plazo", "mipr_us_plazo",
                       "mipr_var", "mipr_spread_hist"),
        specialist="renta_fija",
    ),
    Concept(
        name="cambiario_fx",
        keywords=("tipo de cambio", "dolar", "dólar", "usd/clp", "usdclp", "clp",
                  "cambiario", "spot", "forward", "fx", "paridad", "divisa",
                  "moneda"),
        expansions=("tipo", "cambio", "dolar", "clp", "usd", "cambiario", "spot",
                    "forward"),
        segment="fx",
        dataset_hints=("clp_monto", "forward_points", "bid_ask",
                       "var_moneda_1w", "var_moneda_ytd", "volatilidad_precio",
                       "flujo_cambiario", "fixing_banca",
                       "posicion_spot_derivados_agente"),
        specialist="fx",
    ),
    Concept(
        name="fx_diferencial",
        keywords=("fixing", "flujo cambiario", "sector cambiario",
                  "flujo neto cambiario"),
        expansions=("fixing", "flujo", "diferencial"),
        segment="fx_diferencial",
        dataset_hints=("fixing_banca_sector", "fixing_por_fecha",
                       "flujo_cambiario"),
        specialist="fx",
    ),
    Concept(
        name="commodities",
        keywords=("cobre", "petroleo", "petróleo", "wti", "commodity", "commodities",
                  "dxy", "precio petroleo", "precio cobre"),
        expansions=("cobre", "petroleo", "commodities", "dxy"),
        segment="inflacion",
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
    # Monitor de mercados globales (color de mercados): tablas cm_* con
    # niveles y retornos comparados de bolsas, FX, riesgo país y tasas 10Y.
    Concept(
        name="monitor_mercados",
        keywords=("color de mercados", "monitor de mercados", "bolsas",
                  "indices bursatiles", "índices bursátiles", "riesgo pais",
                  "riesgo país", "cds", "embi", "cembi", "tasas 10y",
                  "mercados globales"),
        expansions=("monitor", "bolsas", "riesgo", "cds", "embi", "tasas",
                    "globales", "retornos"),
        segment="color_mercados",
        dataset_hints=("cm_",),
        # El monitor cm_* (bolsas/FX/riesgo/tasas 10Y) lo lee el especialista FX,
        # que cubre el segmento color_mercados (ver subagents.SUBAGENTS["fx"]).
        specialist="fx",
    ),
    Concept(
        name="transversal",
        keywords=("stock total", "posicion total", "variacion total"),
        expansions=("stock", "total", "todos"),
        segment="todos",
        dataset_hints=("_todos", "stock_nivel_todos", "stock_bucket_todos",
                       "variacion_stock_todos"),
    ),
)


# ── Lookups derivados ───────────────────────────────────────────────────────

# alias (lowercase) → segmento canónico, para resolver el parámetro `segment`.
_SEGMENT_ALIASES: dict[str, str] = {
    # AFP
    "afp": "afp",
    "afps": "afp",
    "pension": "afp",
    "pensiones": "afp",
    "fondos de pension": "afp",
    "fondos de pensiones": "afp",
    "fondos_pension": "afp",
    # FFMM
    "ffmm": "ffmm",
    "fondos mutuos": "ffmm",
    "fondo mutuo": "ffmm",
    # Bancos
    "bancos": "bancos",
    "balance": "bancos",
    "balance bancario": "bancos",
    "banco": "bancos",
    # Seguros
    "seguros": "csv_seguros",
    "csv_seguros": "csv_seguros",
    "csv": "csv_seguros",
    # FX
    "fx": "fx",
    "cambiario": "fx",
    "tipo de cambio": "fx",
    "divisas": "fx",
    "usdclp": "fx",
    # FX diferencial
    "fx_diferencial": "fx_diferencial",
    "fixing": "fx_diferencial",
    "flujo cambiario": "fx_diferencial",
    # Commodities (están en inflacion)
    "commodities": "inflacion",
    "cobre": "inflacion",
    "petroleo": "inflacion",
    # Renta fija
    "renta fija": "rf_tasas",
    "rf": "rf_tasas",
    "bonos": "rf_tasas",
    "curvas": "rf_tasas",
    "rf_tasas": "rf_tasas",
    "rf_volumenes": "rf_volumenes",
    "volumenes renta fija": "rf_volumenes",
    # SPC/OIS
    "spc": "spc_ois",
    "ois": "spc_ois",
    "spc_ois": "spc_ois",
    "swap promedio camara": "spc_ois",
    # Liquidez MN
    "liquidez": "liquidez_mn",
    "liquidez mn": "liquidez_mn",
    "liquidez_mn": "liquidez_mn",
    "pdbc": "liquidez_mn",
    # Liquidez MX
    "lcr": "liquidez_mx",
    "nsfr": "liquidez_mx",
    "liquidez mx": "liquidez_mx",
    "liquidez_mx": "liquidez_mx",
    # Spreads → liquidez_mx
    "spreads": "liquidez_mx",
    "credito": "liquidez_mx",
    # Mercado monetario
    "mercado monetario": "mercado_monetario",
    "mercado_monetario": "mercado_monetario",
    "tib": "mercado_monetario",
    # Inflación / expectativas
    "inflacion": "inflacion",
    "inflación": "inflacion",
    "expectativas": "inflacion",
    "ipc": "inflacion",
    "bei": "inflacion",
    # No residentes
    "nr": "no_residentes",
    "no residentes": "no_residentes",
    "no_residentes": "no_residentes",
    "rfl": "no_residentes",
    "posiciones": "no_residentes",
    # Color de mercados
    "color de mercados": "color_mercados",
    "color_mercados": "color_mercados",
    "monitor": "color_mercados",
    "bolsas": "color_mercados",
    "mercados globales": "color_mercados",
    "riesgo pais": "color_mercados",
    # Fisco
    "fisco": "fisco",
    "tgr": "fisco",
    # Transversal
    "todos": "todos",
    # Taxonomía VIEJA del catálogo (pre-2026-06) → segmentos actuales. El
    # catálogo renombró los segmentos al regenerarse desde los parquets nuevos;
    # estos aliases evitan que prompts/frontends/modelos con los nombres
    # antiguos degraden a búsqueda global.
    "mercado_cambiario": "fx",
    "posiciones_cambiarias": "fx",
    "monitor_mercados": "color_mercados",
    "renta_fija_chile": "rf_tasas",
    "instrumentos_bcch": "liquidez_mn",
    "spreads_credito": "liquidez_mx",
    "liquidez_bancaria": "liquidez_mn",
    "balance_bancario": "bancos",
    "posiciones_rfl": "no_residentes",
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
