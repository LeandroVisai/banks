"""Funciones analíticas puras sobre series de tiempo.

Operan sobre listas de valores ya extraídas de los parquets — **sin I/O**.
Esto las hace triviales de testear y reutilizar: las analytics tools del
agente consultan DuckDB y luego delegan el cálculo aquí.

Convención: una "serie" es una lista de ``(fecha_iso, valor)`` ordenada
ascendente por fecha. Los valores ``None`` o no numéricos se descartan.
"""

from __future__ import annotations

import statistics
from typing import Any

Point = tuple[str, float]


def clean_series(rows: list[dict], date_col: str, value_col: str) -> list[Point]:
    """Extrae ``(fecha, valor)`` de filas crudas, descarta nulos y ordena por fecha.

    Filas con fecha o valor faltante / no numérico se omiten silenciosamente:
    los parquets del DW tienen huecos y un cálculo sobre ``None`` reventaría.
    """
    out: list[Point] = []
    for row in rows:
        fecha = row.get(date_col)
        raw = row.get(value_col)
        if fecha is None or raw is None:
            continue
        try:
            out.append((str(fecha), float(raw)))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda p: p[0])
    return out


def composition(
    rows: list[dict],
    date_col: str,
    category_col: str,
    value_col: str,
    *,
    fecha: str | None = None,
) -> dict[str, Any] | None:
    """Breakdown por categoría a una fecha (la última disponible si no se da ``fecha``).

    Para preguntas de COMPOSICIÓN/CARTERA: agrupa ``value_col`` por
    ``category_col`` en una fecha de corte y calcula el porcentaje de cada
    categoría sobre el total. Resuelve directamente "¿qué % está en X vs Y?"
    (ej. allocation AFP Chile vs. extranjero) — antes el LLM inventaba esas
    cifras.

    Args:
        rows: filas crudas con al menos ``date_col``, ``category_col``, ``value_col``.
        fecha: fecha de corte exacta (debe coincidir con un valor de ``date_col``);
            si es ``None`` se usa la última fecha presente en ``rows``.

    Returns:
        ``{fecha, total, n_categorias, breakdown:[{categoria, valor, share_pct}…]}``
        con ``breakdown`` ordenado desc por valor; o ``None`` si no hay datos
        válidos para la fecha de corte. ``share_pct`` es ``None`` si el total es 0.
    """
    # (fecha_str, categoria, valor) válidos.
    clean: list[tuple[str, str, float]] = []
    for row in rows:
        f = row.get(date_col)
        cat = row.get(category_col)
        raw = row.get(value_col)
        if f is None or cat is None or raw is None:
            continue
        try:
            clean.append((str(f), str(cat), float(raw)))
        except (TypeError, ValueError):
            continue
    if not clean:
        return None

    corte = fecha if fecha is not None else max(f for f, _, _ in clean)
    en_corte = [(c, v) for f, c, v in clean if f == corte]
    if not en_corte:
        return None

    sums: dict[str, float] = {}
    for cat, val in en_corte:
        sums[cat] = sums.get(cat, 0.0) + val

    total = sum(sums.values())
    breakdown = [
        {
            "categoria": cat,
            "valor": round(val, 6),
            "share_pct": round(100.0 * val / total, 2) if total else None,
        }
        for cat, val in sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "fecha": corte,
        "total": round(total, 6),
        "n_categorias": len(breakdown),
        "breakdown": breakdown,
    }


def composition_wide(
    rows: list[dict],
    date_col: str,
    value_cols: list[str],
    *,
    fecha: str | None = None,
) -> dict[str, Any] | None:
    """Composición a partir de columnas WIDE (una columna por categoría).

    Para datasets donde cada categoría es su propia columna numérica (ej.
    ``allocation_int_nac`` con ``Nacional`` y ``Extranjero``): a la fecha de
    corte, suma cada columna y calcula su share sobre el total. Resuelve "% en
    Chile vs. extranjero" sin que el LLM invente las cifras.

    Returns el mismo shape que ``composition`` (``breakdown`` con ``categoria``
    = nombre de columna), o ``None`` si no hay filas válidas en la fecha.
    """
    # Fecha de corte: la última con al menos una columna no nula.
    dated: list[tuple[str, dict]] = []
    for row in rows:
        f = row.get(date_col)
        if f is None:
            continue
        if any(row.get(c) is not None for c in value_cols):
            dated.append((str(f), row))
    if not dated:
        return None

    corte = fecha if fecha is not None else max(f for f, _ in dated)
    en_corte = [r for f, r in dated if f == corte]
    if not en_corte:
        return None

    sums: dict[str, float] = {c: 0.0 for c in value_cols}
    for row in en_corte:
        for c in value_cols:
            raw = row.get(c)
            if raw is None:
                continue
            try:
                sums[c] += float(raw)
            except (TypeError, ValueError):
                continue

    total = sum(sums.values())
    breakdown = [
        {
            "categoria": c,
            "valor": round(sums[c], 6),
            "share_pct": round(100.0 * sums[c] / total, 2) if total else None,
        }
        for c in sorted(value_cols, key=lambda k: sums[k], reverse=True)
    ]
    return {
        "fecha": corte,
        "total": round(total, 6),
        "n_categorias": len(breakdown),
        "breakdown": breakdown,
    }


def variation(series: list[Point]) -> dict[str, Any] | None:
    """Variación entre el primer y el último punto de la serie.

    Retorna cambio absoluto, variación porcentual y cambio en puntos base
    (válido solo si la unidad de la serie es una tasa/porcentaje), más el
    rango observado. ``None`` si hay menos de 2 puntos.
    """
    if len(series) < 2:
        return None
    (fecha_ini, v_ini), (fecha_fin, v_fin) = series[0], series[-1]
    valores = [v for _, v in series]
    cambio_abs = v_fin - v_ini
    return {
        "fecha_inicio_obs": fecha_ini,
        "valor_inicio": round(v_ini, 6),
        "fecha_fin_obs": fecha_fin,
        "valor_fin": round(v_fin, 6),
        "cambio_absoluto": round(cambio_abs, 6),
        "variacion_pct": _pct_change(v_ini, v_fin),
        "cambio_bps": round(cambio_abs * 100, 2),
        "minimo": round(min(valores), 6),
        "maximo": round(max(valores), 6),
        "n_observaciones": len(series),
    }


def descriptive_stats(series: list[Point]) -> dict[str, Any] | None:
    """Estadísticas descriptivas + percentil del último valor en la distribución.

    ``None`` si la serie está vacía.
    """
    if not series:
        return None
    valores = [v for _, v in series]
    ultimo = valores[-1]
    media = statistics.mean(valores)
    desv = statistics.pstdev(valores) if len(valores) > 1 else 0.0
    return {
        "n_observaciones": len(valores),
        "media": round(media, 6),
        "desviacion_estandar": round(desv, 6),
        "minimo": round(min(valores), 6),
        "maximo": round(max(valores), 6),
        "ultimo_valor": round(ultimo, 6),
        "ultima_fecha": series[-1][0],
        "percentil_ultimo_valor": _percentile_rank(ultimo, valores),
    }


def anomaly_check(
    series: list[Point],
    threshold_stds: float = 2.0,
) -> dict[str, Any] | None:
    """Evalúa si el último valor está fuera del rango histórico de la serie.

    Calcula media y desviación de la ventana (excluyendo el último dato), el
    z-score del último valor y lo marca como anomalía si ``|z| >= threshold_stds``.
    ``None`` si hay menos de 3 puntos (sin masa estadística).

    Caso límite: si la ventana histórica no tuvo variación alguna, el z-score
    es indefinido — cualquier valor distinto de ese nivel constante se reporta
    como anomalía (``z_score=None``).
    """
    if len(series) < 3:
        return None
    valores = [v for _, v in series]
    ultimo = valores[-1]
    historico = valores[:-1]  # el último no se incluye en su propia referencia
    media = statistics.mean(historico)
    desv = statistics.pstdev(historico) if len(historico) > 1 else 0.0

    if desv > 0:
        z = (ultimo - media) / desv
        es_anomalia = abs(z) >= threshold_stds
        z_score: float | None = round(z, 3)
        interpretacion = _anomaly_text(z, es_anomalia)
    else:
        # Ventana plana: el z-score no está definido (división por cero).
        es_anomalia = ultimo != media
        z_score = None
        nivel = round(media, 6)
        interpretacion = (
            f"El histórico de la ventana fue constante en {nivel}; el último "
            + ("valor se desvía de ese nivel — atípico." if es_anomalia
               else "valor se mantiene en ese nivel.")
        )

    return {
        "ultimo_valor": round(ultimo, 6),
        "ultima_fecha": series[-1][0],
        "media_ventana": round(media, 6),
        "desviacion_ventana": round(desv, 6),
        "z_score": z_score,
        "es_anomalia": es_anomalia,
        "umbral_desviaciones": threshold_stds,
        "n_observaciones": len(series),
        "interpretacion": interpretacion,
    }


def spread(series_a: list[Point], series_b: list[Point]) -> dict[str, Any] | None:
    """Spread ``A - B`` alineando ambas series por fecha común.

    Retorna el spread en la última fecha común, en la primera, su variación,
    y mín/máx/promedio del período. ``None`` si no hay fechas en común.
    """
    mapa_b = dict(series_b)
    pares: list[Point] = [
        (fecha, val_a - mapa_b[fecha]) for fecha, val_a in series_a if fecha in mapa_b
    ]
    if not pares:
        return None
    pares.sort(key=lambda p: p[0])
    spreads = [s for _, s in pares]
    return {
        "fecha_obs": pares[-1][0],
        "spread_fin": round(pares[-1][1], 6),
        "spread_inicio": round(pares[0][1], 6),
        "cambio_spread": round(pares[-1][1] - pares[0][1], 6),
        "spread_minimo": round(min(spreads), 6),
        "spread_maximo": round(max(spreads), 6),
        "spread_promedio": round(statistics.mean(spreads), 6),
        "n_observaciones_comunes": len(pares),
    }


# ── Internos ──────────────────────────────────────────────────────────────────

def _pct_change(inicio: float, fin: float) -> float | None:
    """Variación porcentual; ``None`` si el punto inicial es cero."""
    if inicio == 0:
        return None
    return round(100.0 * (fin - inicio) / abs(inicio), 4)


def _percentile_rank(valor: float, valores: list[float]) -> float:
    """Porcentaje de observaciones estrictamente menores que ``valor`` (0-100)."""
    if not valores:
        return 0.0
    menores = sum(1 for v in valores if v < valor)
    return round(100.0 * menores / len(valores), 1)


def _anomaly_text(z: float, es_anomalia: bool) -> str:
    direccion = "por encima" if z > 0 else "por debajo"
    magnitud = abs(z)
    if not es_anomalia:
        return f"Dentro del rango normal (z={z:+.2f})."
    return (
        f"Valor inusual: {magnitud:.1f} desviaciones estándar {direccion} "
        f"de la media de la ventana (z={z:+.2f})."
    )
