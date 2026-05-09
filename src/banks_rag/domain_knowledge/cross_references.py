"""Referencias cruzadas entre Monitor PM (daily) y PDFs (period).

Cuando un Comunicado/Minuta/IPOM cubre un mes (o trimestre), los chunks del
Monitor PM cuyas fechas caen dentro de ese período son ejemplos diarios de
las variables que el documento period describe.

Estas relaciones permiten al agente, dado un PDF, mostrar los días concretos
de mercado que contextualizan sus afirmaciones — y dado un día del Monitor PM,
encontrar el documento period que lo encuadra.

Funciones puras: reciben listas de dicts (no requieren Postgres ni embedder).
"""

from __future__ import annotations

# Tipos de doc cuyo período es trimestral. El resto se trata como mensual.
_QUARTERLY_DOC_TYPES = frozenset({"IPOM", "IEF"})


def period_contains_date(excel_date: str, pdf_date: str, pdf_doc_type: str) -> bool:
    """``True`` si el período de ``pdf_date`` contiene ``excel_date``.

    Para IPOM/IEF compara trimestres del mismo año. Para el resto compara mes
    del mismo año (``YYYY-MM``).
    """
    if not excel_date or not pdf_date or len(excel_date) < 7 or len(pdf_date) < 7:
        return False
    try:
        ex_ym = excel_date[:7]
        pdf_ym = pdf_date[:7]
        if pdf_doc_type in _QUARTERLY_DOC_TYPES:
            ex_q = (int(ex_ym[5:7]) - 1) // 3
            pdf_q = (int(pdf_ym[5:7]) - 1) // 3
            return ex_ym[:4] == pdf_ym[:4] and ex_q == pdf_q
        return ex_ym == pdf_ym
    except (ValueError, IndexError):
        return False


def build_cross_references(
    excel_records: list[dict],
    pdf_records: list[dict],
) -> dict:
    """Construye índice bidireccional entre records daily (Monitor PM) y period (PDFs).

    Conecta excel→pdf y pdf→excel solo si:
      1. Comparten al menos una variable económica detectada.
      2. La fecha del excel cae dentro del período del PDF (mes o trimestre).

    Returns:
        ``{"excel_to_period": {ex_id: [...]}, "period_to_daily": {pdf_id: [...]}}``
        donde cada entrada de la lista es un dict con
        ``{record_id, pointer_type, shared_vars}``.
    """
    excel_to_period: dict[str, list] = {}
    period_to_daily: dict[str, list] = {}

    for ex in excel_records:
        ex_id = ex.get("chunk_id", "")
        ex_date = ex.get("chunk_date") or ex.get("document_date") or ""
        ex_vars = set(ex.get("economic_variables", {}).keys())
        refs: list[dict] = []
        for pdf in pdf_records:
            pdf_id = pdf.get("chunk_id", "")
            pdf_date = pdf.get("document_date") or ""
            pdf_type = pdf.get("doc_type_category", "")
            pdf_vars = set(pdf.get("economic_variables", {}).keys())
            shared = sorted(ex_vars & pdf_vars)
            if shared and period_contains_date(ex_date, pdf_date, pdf_type):
                refs.append({
                    "record_id": pdf_id,
                    "pointer_type": "period_context",
                    "shared_vars": shared,
                })
                period_to_daily.setdefault(pdf_id, []).append({
                    "record_id": ex_id,
                    "pointer_type": "daily_example",
                    "shared_vars": shared,
                })
        if refs:
            excel_to_period[ex_id] = refs

    return {"excel_to_period": excel_to_period, "period_to_daily": period_to_daily}
