"""Spec declarativo de un informe CURADO por familia (orden + bloques).

El informe descriptivo genérico (``parquet_report``) emite un gráfico por dataset
en el orden del catálogo. Para REPLICAR un informe real del BCCh (p.ej. "Informe
Fondos Mutuos") hace falta una vista curada: gráficos específicos y derivados
(filtros por tipo de fondo, agregación mensual, ventanas, acumulados) en un orden
narrativo, agrupados en secciones con banner.

Este módulo define solo las ESTRUCTURAS. El orden y los títulos de una familia
viven en ``specs/<familia>_spec.py`` (la fuente de verdad de ESE informe); las
transformaciones parquet→serie en ``series_transforms.py``; el render en
``curated_report.py``. Todo Python puro, sin LLM: el texto se agrega después en
slots con id estable (``text_slot``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Familia de gráfico objetivo del bloque (canónica, no la del catálogo). El
# builder solo sabe dibujar las soportadas hoy; el resto caen a placeholder.
ChartKind = str  # "line" | "area" | "composition" | "grouped_bar" | "stacked_bar"
#                  | "stacked_area" | "pie" | "dual_line" | "bar_time" | "heatmap_table"
#                  | "hist_range" (caja mín/máx + promedio + "hoy" por categoría)
#                  | "point" (dispersión x/y etiquetada, con cruce en 0)

# Estado del bloque respecto del MVP.
STATUS_MVP = "mvp"    # el renderer actual lo dibuja ahora
STATUS_EXP = "exp"    # pendiente de renderer/transform de la 2ª iteración
STATUS_SKIP = "skip"  # sin parquet reproducible (placeholder "sin datos")


@dataclass(frozen=True)
class ReportBlock:
    """Un bloque del informe: un gráfico (o tabla) con su fuente y transformación.

    - ``section``: encabezado de sección (banner). Bloques consecutivos con el
      mismo ``section`` comparten banner.
    - ``source_id``: id de dataset del catálogo (``None`` si SKIP / sin fuente).
    - ``transform``: nombre de la transformación parquet→``PlotData``
      (resuelto en ``series_transforms``). ``params`` la parametriza
      (p.ej. ``{"funds": ["Tipo 1", "Tipo 2"]}``).
    - ``chart``: familia de gráfico objetivo (ChartKind).
    - ``text_slot``: id estable del slot de texto que acompaña al bloque.
    - ``status``: STATUS_MVP / STATUS_EXP / STATUS_SKIP.
    - ``scale``: factor que se aplica a los valores de la serie antes de
      graficar (1.0 = sin cambio). Sirve para corregir la unidad de un parquet
      cuyo dato está en otra escala que la declarada (p.ej. parquet en MILES de
      USD que el informe muestra en ``Mill US$`` → ``scale=0.001``; fracción que
      debe leerse como ``%`` → ``scale=100``). El renderer escala el valor una
      vez, así eje, tooltips y leyenda quedan consistentes.
    """

    section: str
    title: str
    chart: ChartKind
    status: str
    unit: str = ""
    source_id: str | None = None
    transform: str | None = None
    params: dict = field(default_factory=dict)
    text_slot: str = ""
    note: str = ""  # nota visible (p.ej. "*Datos hasta el 8/6")
    scale: float = 1.0
    # ``True`` → el bloque dibuja su GRÁFICO pero NO recibe párrafo de texto. Sirve
    # para vistas redundantes del mismo concepto (p.ej. dos gráficos de flujos): se
    # conservan ambos gráficos, pero el comentario se escribe UNA sola vez (en el
    # otro bloque), evitando prosa duplicada o contradictoria. ``_resolve_text_slots``
    # no le asigna ``text_slot`` y la síntesis excluye su párrafo.
    no_text: bool = False
    # Recorte de fechas del bloque, en ISO (``"2026-01-01"``). Vacío = sin límite
    # por ese lado; se puede fijar solo uno de los dos.
    #
    # El recorte se aplica al PARQUET, antes de que la transform lo lea: el bloque
    # ve un dataset que solo contiene esas fechas. Por eso funciona con CUALQUIER
    # transform sin tocarla, y todo lo que se deriva de la fecha queda coherente
    # con el recorte — la "última fecha" del gráfico, las ventanas 7d/30d, el
    # acumulado YtD y el corte que se cita en el pie. Cortar solo los puntos ya
    # calculados dejaría, por ejemplo, un "Δ T-7" medido contra días que el lector
    # no ve en el gráfico.
    #
    # El párrafo de texto del bloque hereda el MISMO recorte (ver
    # ``spec_date_filters``), así prosa y gráfico hablan siempre del mismo período.
    date_from: str = ""
    date_to: str = ""


@dataclass(frozen=True)
class FamilyReportSpec:
    """Informe curado completo de una familia: título + bloques ordenados."""

    family: str
    title: str
    blocks: tuple[ReportBlock, ...]
    # Secciones cuyas variaciones SEMANALES comparten una misma fecha de corte
    # (mín de las fechas máximas de SUS datasets de alta frecuencia). El resto de
    # secciones se ancla al máximo de su propio parquet. Vacío = todas las de alta
    # frecuencia comparten corte (comportamiento por defecto para otras familias).
    weekly_anchor_sections: tuple[str, ...] = ()
    # ``False`` → NO hay corte semanal común en este informe: cada gráfico/tabla se
    # ancla al máximo de SU propio parquet (no se arrastra todo a la fecha máxima del
    # parquet con menor fecha). Útil cuando los datasets de la familia tienen cortes
    # heterogéneos y forzar un corte común muestra datos viejos (caso afp).
    share_weekly_cutoff: bool = True

    def sections(self) -> list[str]:
        """Secciones en orden de aparición (sin repetir)."""
        out: list[str] = []
        for b in self.blocks:
            if b.section not in out:
                out.append(b.section)
        return out

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.status] = counts.get(b.status, 0) + 1
        return counts

    def no_text_source_ids(self) -> set[str]:
        """``source_id`` que SOLO alimentan bloques ``no_text`` (vistas redundantes sin
        comentario propio): la síntesis excluye su párrafo para no mezclar comentarios
        solapados del mismo concepto.

        Si el MISMO ``source_id`` alimenta también un bloque que SÍ comenta (p.ej.
        ``flujos_acum_ffmm``: barras t-7/t-30 comentadas + línea acumulada ``no_text``),
        NO es redundante: su único párrafo es el de ese bloque y debe entrar a la
        síntesis. Solo se excluyen las fuentes cuyo conjunto de bloques es 100%
        ``no_text``."""
        commented = {b.source_id for b in self.blocks if not b.no_text and b.source_id}
        return {
            b.source_id for b in self.blocks
            if b.no_text and b.source_id and b.source_id not in commented
        }
