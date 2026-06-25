"""Builder + render HTML de un informe CURADO por familia (Python puro, sin LLM).

Toma un ``FamilyReportSpec`` (orden + bloques de un informe real, p.ej. ffmm),
corre la transform de cada bloque sobre el parquet REAL y arma un HTML con la
estructura del informe: banners de sección, título por bloque, un slot de texto
vacío (id estable, se llena después con el LLM en el H100) y el gráfico SVG.

MVP: dibuja los bloques cuyo gráfico el renderer ya soporta (multi-línea /
composición); los demás quedan como tarjeta-placeholder marcada con su tipo
pendiente (EXP) o "sin datos reproducibles" (SKIP). Así la estructura y el ORDEN
del informe quedan completos desde la primera iteración.
"""

from __future__ import annotations

import dataclasses
import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ParquetDataset,
    get_dataset,
    get_parquet_dir,
    load_parquet_catalog,
)

from .parquet_facts import HtmlTable, PlotData, PlotSeries, compute_series, weekly_cutoff
from .report_spec import STATUS_SKIP, FamilyReportSpec, ReportBlock
from .series_transforms import get_transform
from .svg_chart import render_plot_svg, renders_natively

log = logging.getLogger(__name__)


def _scale_plot(plot: PlotData, factor: float) -> PlotData:
    """Devuelve una copia del ``PlotData`` con cada valor ``y`` multiplicado por
    ``factor`` (corrige la escala de un parquet cuya unidad real difiere de la
    declarada). El eje X (fecha o categoría) no se toca."""
    return PlotData(
        dataset_id=plot.dataset_id,
        family=plot.family,
        kind=plot.kind,
        unit=plot.unit,
        series=[
            PlotSeries(label=s.label, points=[(x, y * factor) for x, y in s.points])
            for s in plot.series
        ],
        overlay=plot.overlay,
        date_note=plot.date_note,
    )

# Familias objetivo que NO se pueden aproximar como serie (tabla): placeholder.
_TABLE_CHARTS = frozenset({"heatmap_table"})


@dataclass
class CuratedBlock:
    """Resultado de procesar un ``ReportBlock``: gráfico o placeholder + metadata."""

    block: ReportBlock
    render_kind: str       # "chart" | "placeholder" | "skip"
    body_html: str         # SVG inline o inner del placeholder
    reason: str = ""       # por qué placeholder (para la tarjeta)
    preliminary: bool = False  # gráfico dibujado como línea, tipo final pendiente
    date_note: str = ""    # fechas/ventanas que cubre el gráfico (eje X no temporal)


@dataclass
class CuratedReport:
    spec: FamilyReportSpec
    generated_at: str
    blocks: list[CuratedBlock] = field(default_factory=list)

    def render_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.render_kind] = counts.get(b.render_kind, 0) + 1
        return counts

    def summary(self) -> str:
        c = self.render_counts()
        prelim = sum(1 for b in self.blocks if b.preliminary)
        return (
            f"gráficos={c.get('chart', 0)} (de ellos preliminares={prelim}) "
            f"placeholders={c.get('placeholder', 0)} skip={c.get('skip', 0)} "
            f"/ {len(self.blocks)} bloques"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────


def _spec_source_ids(spec: FamilyReportSpec) -> list[str]:
    """``source_id`` distintos del spec, en orden de aparición."""
    seen: list[str] = []
    for b in spec.blocks:
        if b.source_id and b.source_id not in seen:
            seen.append(b.source_id)
    return seen


def weekly_anchor_source_ids(spec: FamilyReportSpec) -> set[str]:
    """``source_id`` cuyas variaciones semanales comparten el corte común: los de
    las secciones en ``spec.weekly_anchor_sections``. Si el spec no declara secciones
    de anclaje, devuelve TODOS los source_ids (corte global, comportamiento por
    defecto para otras familias)."""
    sections = set(spec.weekly_anchor_sections)
    if not sections:
        return set(_spec_source_ids(spec))
    return {b.source_id for b in spec.blocks if b.source_id and b.section in sections}


def compute_weekly_cutoff(
    spec: FamilyReportSpec, entries: list[ParquetDataset], parquet_dir: Path,
) -> str | None:
    """Corte semanal COMÚN del informe = ``weekly_cutoff`` sobre los datasets de las
    secciones de anclaje (``weekly_anchor_source_ids``; ``weekly_cutoff`` se queda
    solo con los de alta frecuencia).

    Es el conjunto CANÓNICO: el proceso de gráficos (``build_curated_report``) y el
    de texto (``generate_parquet_report``) lo calculan sobre los MISMOS source_ids
    → mismo corte en ambos, sin pasar datos de un proceso a otro. Para ffmm el
    anclaje es Flujos + Portafolio DCV: si flujos llega al 21 y DCV al 22, el corte
    es el 21 (mín de los máximos)."""
    anchor = weekly_anchor_source_ids(spec)
    datasets = [
        d for sid in _spec_source_ids(spec)
        if sid in anchor and (d := get_dataset(entries, sid)) is not None
    ]
    return weekly_cutoff(datasets, parquet_dir)


def monthly_section_source_ids(spec: FamilyReportSpec) -> set[str]:
    """``source_id`` que ABREN su sección: el PRIMER bloque con ``text_slot`` de cada
    sección del spec. Solo estos reciben el sub-párrafo MENSUAL (una vez por sección);
    el resto de la sección lleva solo el semanal. Reusa ``_resolve_text_slots`` (la
    misma asignación de slots que el HTML curado)."""
    seen_sections: set[str] = set()
    out: set[str] = set()
    for b in _resolve_text_slots(spec):
        if not b.text_slot or not b.source_id:
            continue
        if b.section not in seen_sections:
            seen_sections.add(b.section)
            out.add(b.source_id)
    return out


def _process_block(
    block: ReportBlock, *, entries: list[ParquetDataset], parquet_dir: Path,
    weekly_asof: str | None = None,
) -> CuratedBlock:
    """Dibuja el bloque siempre que el dato sea graficable como serie/composición.

    Criterio: SKIP (sin parquet) → tarjeta "sin datos". Tablas-heatmap SIN
    transform implementada → placeholder "pendiente". El resto se dibuja: si tiene
    transform propia (PlotData o HtmlTable) se usa; si no, se cae a la serie
    natural del parquet (VISTA PRELIMINAR). Best-effort: un bloque que falle no
    tumba el resto.
    """
    if block.status == STATUS_SKIP:
        return CuratedBlock(block, "skip", "", "sin parquet reproducible todavía")

    # Tabla cuya transform no está implementada aún: placeholder directo.
    transform = get_transform(block.transform)
    if block.chart in _TABLE_CHARTS and transform is None:
        return CuratedBlock(block, "placeholder", "", f"pendiente: {block.chart}")

    dataset = get_dataset(entries, block.source_id or "")
    if dataset is None:
        log.warning("[%s] source_id %r no está en el catálogo", block.title, block.source_id)
        return CuratedBlock(block, "placeholder", "", f"dataset {block.source_id} no encontrado")

    # Inyecta el corte semanal común en los params para las transforms semanales
    # (stacked_by_bucket, dcv_*): leen params["weekly_asof"] y anclan ahí la ventana.
    params = {**block.params, "weekly_asof": weekly_asof} if weekly_asof else block.params
    try:
        if transform is not None:
            result = transform(dataset, parquet_dir, params)
        else:
            # Vista preliminar: serie natural, filtrada por las categorías pedidas.
            cf = list(block.params.get("funds") or block.params.get("types") or []) or None
            result = compute_series(dataset, parquet_dir, category_filter=cf)
    except Exception:
        log.exception("[%s] el cálculo de la serie falló", block.title)
        return CuratedBlock(block, "placeholder", "", "error al calcular la serie")

    # Resultado HTML inline (tablas con color — heatmap DCV).
    if isinstance(result, HtmlTable):
        return CuratedBlock(block, "chart", result.html, preliminary=False)

    plot = result
    if plot is None or plot.is_empty():
        return CuratedBlock(block, "placeholder", "", "sin serie en el parquet")

    # Escala: corrección de unidad del dataset (catálogo) por override del bloque.
    # El gráfico queda en la MISMA escala que el texto (compute_facts también lee
    # dataset.value_scale), así prosa y curva coinciden.
    scale = dataset.value_scale * block.scale
    if scale != 1.0:
        plot = _scale_plot(plot, scale)

    svg = render_plot_svg(
        plot, chart=block.chart,
        right_axis=block.params.get("right_axis"),
        right_unit=block.params.get("right_unit", ""),
    )
    if svg is None:
        return CuratedBlock(block, "placeholder", "", "serie no graficable como línea")
    # Preliminar solo si la forma del dato NO permite el tipo objetivo (cae a
    # línea/composición). Con su transform propia, el bloque sale en su tipo final.
    preliminary = not renders_natively(plot.kind, block.chart)
    return CuratedBlock(block, "chart", svg, preliminary=preliminary, date_note=plot.date_note)


def _resolve_text_slots(spec: FamilyReportSpec) -> list[ReportBlock]:
    """Asigna ``text_slot`` automáticamente al PRIMER bloque que use cada
    ``source_id``, si el bloque no tiene ya un slot explícito en el spec.

    Esto garantiza que haya exactamente UN párrafo descriptivo por parquet, justo
    encima del primer gráfico que lo usa — independientemente del orden en que el
    spec declare los bloques. Los slots explícitos del spec tienen precedencia."""
    seen: set[str] = set()
    resolved: list[ReportBlock] = []
    for b in spec.blocks:
        slot = b.text_slot
        # Un bloque ``no_text`` dibuja su gráfico pero NO recibe párrafo: vista
        # redundante de un concepto cuyo comentario ya va en otro bloque.
        if (
            not slot and not b.no_text and b.source_id
            and b.source_id not in seen and b.status != STATUS_SKIP
        ):
            slot = f"{spec.family}:{b.source_id}"
        if b.source_id:
            seen.add(b.source_id)
        resolved.append(dataclasses.replace(b, text_slot=slot) if slot != b.text_slot else b)
    return resolved


def build_curated_report(
    spec: FamilyReportSpec,
    *,
    entries: list[ParquetDataset] | None = None,
    parquet_dir: Path | None = None,
    catalog_path: Path | str | None = None,
) -> CuratedReport:
    """Procesa todos los bloques del spec. ``entries``/``parquet_dir`` inyectables
    para tests; por defecto carga el catálogo real."""
    if entries is None:
        entries = load_parquet_catalog(catalog_path)
    if parquet_dir is None:
        parquet_dir = get_parquet_dir(catalog_path)

    resolved = _resolve_text_slots(spec)
    # Corte común (mín de máximos de Flujos+DCV) → ancla T, T-7 y T-30 de las tablas
    # DCV y la variación de flujos a la MISMA ventana temporal (los mismos días), en
    # gráficos Y texto. Solo las secciones de anclaje lo usan; el resto, su máximo.
    cutoff = compute_weekly_cutoff(spec, entries, parquet_dir)
    anchor_ids = weekly_anchor_source_ids(spec)
    if cutoff:
        log.info("Informe curado %s: corte común = %s (anclaje: %s)",
                 spec.family, cutoff, ", ".join(spec.weekly_anchor_sections) or "todas")
    blocks = [
        _process_block(
            b, entries=entries, parquet_dir=parquet_dir,
            weekly_asof=cutoff if b.source_id in anchor_ids else None,
        )
        for b in resolved
    ]
    report = CuratedReport(spec=spec, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"), blocks=blocks)
    log.info("Informe curado %s: %s", spec.family, report.summary())
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Render HTML
# ─────────────────────────────────────────────────────────────────────────────

_SHELL = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  :root { --blue:#0b3766; --banner:#4a5a72; --text:#1f1f1f; --muted:#777; }
  * { box-sizing: border-box; }
  body { margin:0; background:#fff; color:var(--text); font-family:Arial, Helvetica, sans-serif; line-height:1.45; }
  .page { width:min(1200px, calc(100% - 40px)); margin:18px auto 40px; }
  .report-title { background:var(--banner); color:#fff; text-align:center; font-size:24px; font-weight:800; padding:12px 16px; letter-spacing:.3px; }
  .subtitle { text-align:center; color:var(--muted); font-size:12px; margin:8px 0 4px; }
  .report-synthesis { border:1px solid #d8dee8; border-left:5px solid var(--blue); background:#f6f8fb; border-radius:6px; padding:12px 18px; margin:14px 0 8px; }
  .synthesis-title { color:var(--blue); font-size:16px; font-weight:800; margin:0 0 6px; }
  .synthesis-body { color:var(--text); font-size:13px; }
  .synthesis-body p { margin:4px 0; }
  .synthesis-body ul { margin:4px 0 8px; padding-left:20px; }
  .synthesis-body li { margin:2px 0; }
  .synthesis-body[data-synthesis-body]:empty::before { content:"—"; color:#cfcfcf; }
  .section-banner { background:var(--banner); color:#fff; text-align:center; font-size:18px; font-weight:800; padding:8px 14px; margin:30px 0 6px; }
  .block { margin:14px 0 8px; page-break-inside:avoid; }
  .block-title { color:var(--blue); font-size:16px; font-weight:700; margin:12px 0 2px; }
  .block-unit { color:var(--muted); font-size:12px; margin:0 0 6px; }
  .block-note { color:var(--muted); font-size:11px; font-style:italic; margin:0 0 6px; }
  .block-dates { color:var(--blue); font-size:11px; font-weight:600; background:#eef3f9; border:1px solid #d8dee8; border-radius:4px; padding:2px 8px; margin:0 0 6px; display:inline-block; }
  .prelim-note { color:#9a4b00; font-size:11px; margin:0 0 2px; }
  .section-text { min-height:18px; margin:4px 0 10px; color:var(--text); font-size:14px; }
  .section-text:empty::before { content:"—"; color:#cfcfcf; }
  .section-text p { margin:0 0 6px; }
  .section-text p:last-child { margin-bottom:0; }
  .report-chart { display:block; max-width:760px; margin:6px auto; }
  .placeholder-card { border:1px dashed #bcbcbc; background:#fafafa; color:var(--muted); border-radius:6px; padding:18px; text-align:center; font-size:13px; max-width:760px; margin:6px auto; }
  .placeholder-card .kind { font-weight:700; color:#9a4b00; }
  .placeholder-card.skip .kind { color:#8a8a8a; }
  /* Hover interactivo (estilo Plotly): el marcador aparece y el shape se resalta */
  .report-chart .tip-pt { cursor:pointer; transition:fill-opacity .08s; }
  .report-chart circle.tip-pt:hover { fill-opacity:1; stroke:#fff; stroke-width:1.4; }
  .report-chart rect.tip-pt:hover, .report-chart path.tip-pt:hover { stroke:#1f1f1f; stroke-width:1.4; }
  /* Crosshair overlay (series temporales) */
  .report-chart .hover-overlay { cursor:crosshair; }
  /* Leyenda clickeable para ocultar/mostrar series */
  .report-chart .lg-item { cursor:pointer; }
  .report-chart .lg-item:hover { opacity:0.75; }
  /* Tooltip flotante */
  #chart-tip { position:fixed; z-index:9999; display:none; pointer-events:none;
    background:#fff; border:1px solid #d8dee8; border-radius:6px; box-shadow:0 4px 14px rgba(0,0,0,.16);
    padding:7px 10px; font-size:12px; color:#1f1f1f; max-width:260px; line-height:1.35; }
  #chart-tip .ct-k { color:#777; font-size:11px; margin-bottom:3px; }
  #chart-tip .ct-r { display:flex; align-items:center; gap:6px; white-space:nowrap; margin-bottom:2px; }
  #chart-tip .ct-r:last-child { margin-bottom:0; }
  #chart-tip .ct-sw { width:10px; height:10px; border-radius:2px; flex:0 0 auto; }
  #chart-tip .ct-s { color:#444; }
  #chart-tip .ct-v { font-weight:700; }
  @media print { .section-banner, .report-chart, .placeholder-card, .block { break-inside:avoid; } #chart-tip { display:none !important; } }
</style>
</head>
<body>
  <div class="page">
    <div class="report-title">__TITLE__</div>
    <div class="subtitle">__SUBTITLE__</div>
    <div class="report-synthesis">
      <div class="synthesis-title">Síntesis — principales movimientos</div>
      <div class="synthesis-body" data-synthesis-body></div>
    </div>
__BODY__
  </div>
  <div id="chart-tip" role="tooltip"></div>
  <script>__TIP_JS__</script>
</body>
</html>
"""

# JS interactivo (vanilla, sin dependencias, self-contained, abre offline):
# 1. Leyenda clickeable: click en .lg-item oculta/muestra la serie [data-si="idx"].
# 2. Crosshair unificado (series temporales): al mover el mouse sobre .hover-overlay
#    muestra línea guía vertical + tooltip con TODAS las series en esa fecha.
# 3. Tooltip individual (barras, torta): mouseover en .tip-pt (comportamiento previo).
_TIP_JS = """
(function(){
  var tip=document.getElementById('chart-tip');
  if(!tip) return;
  function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
  function move(e){
    var pad=14,w=tip.offsetWidth,h=tip.offsetHeight,
        x=e.clientX+pad,y=e.clientY+pad;
    if(x+w>window.innerWidth)x=e.clientX-w-pad;
    if(y+h>window.innerHeight)y=e.clientY-h-pad;
    tip.style.left=x+'px';tip.style.top=y+'px';
  }
  function hide(){tip.style.display='none';}
  // Coordenadas pantalla → viewBox SVG.
  function svgCoords(svg,e){
    var pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  // Re-apila las series VISIBLES de un gráfico apilado (área/barra) recomputando
  // su geometría con el eje fijo embebido en data-stack: al ocultar una serie las
  // demás se completan hacia la base en vez de dejar un hueco. Replica exactamente
  // el apilado divergente de svg_chart.py (positivos sobre 0, negativos bajo 0).
  function restack(svg){
    var raw=svg.getAttribute('data-stack');
    if(!raw) return;
    var st; try{st=JSON.parse(raw);}catch(err){return;}
    var span=(st.ymax-st.ymin)||1;
    function py(v){return st.top+st.ph*(1-(v-st.ymin)/span);}
    function visible(i){
      var lg=svg.querySelector('.lg-item[data-idx="'+i+'"]');
      return !lg||lg.getAttribute('data-hidden')!=='1';
    }
    if(st.k==='area'){
      var X=st.x,n=X.length,pos=[],neg=[];
      for(var k=0;k<n;k++){pos.push(0);neg.push(0);}
      st.s.forEach(function(ser){
        if(!visible(ser.i)) return;  // oculta: no acumula ni se redibuja
        var poly=svg.querySelector('polygon[data-si="'+ser.i+'"]');
        if(!poly) return;
        var upper=[],lower=[];
        for(var j=0;j<n;j++){
          var v=ser.v[j],lo,hi;
          if(v>=0){lo=pos[j];hi=pos[j]+v;pos[j]=hi;}
          else{hi=neg[j];lo=neg[j]+v;neg[j]=lo;}
          upper.push(X[j].toFixed(1)+','+py(hi).toFixed(1));
          lower.push(X[j].toFixed(1)+','+py(lo).toFixed(1));
        }
        lower.reverse();
        poly.setAttribute('points',upper.join(' ')+' '+lower.join(' '));
      });
    } else if(st.k==='bar'){
      var ncat=st.s.length?st.s[0].v.length:0;
      for(var ci=0;ci<ncat;ci++){
        var posAcc=0,negAcc=0;
        st.s.forEach(function(ser){
          if(!visible(ser.i)) return;  // oculta: no acumula
          var v=ser.v[ci],yTop,yBot;
          if(v>=0){yTop=py(posAcc+v);yBot=py(posAcc);posAcc+=v;}
          else{yTop=py(negAcc);yBot=py(negAcc+v);negAcc+=v;}
          var rect=svg.querySelector('rect[data-si="'+ser.i+'"][data-ci="'+ci+'"]');
          if(rect){
            rect.setAttribute('y',Math.min(yTop,yBot).toFixed(1));
            rect.setAttribute('height',Math.abs(yBot-yTop).toFixed(1));
          }
        });
      }
    }
  }
  document.querySelectorAll('svg.report-chart').forEach(function(svg){
    // ── Leyenda clickeable ────────────────────────────────────────────────────
    svg.querySelectorAll('.lg-item').forEach(function(lg){
      lg.addEventListener('click',function(){
        var idx=lg.getAttribute('data-idx');
        var off=lg.getAttribute('data-hidden')==='1';
        if(off){
          lg.removeAttribute('data-hidden');lg.style.opacity='';
          svg.querySelectorAll('[data-si="'+idx+'"]').forEach(function(el){el.style.display='';});
        } else {
          lg.setAttribute('data-hidden','1');lg.style.opacity='0.25';
          svg.querySelectorAll('[data-si="'+idx+'"]').forEach(function(el){el.style.display='none';});
        }
        restack(svg);  // re-apila las visibles (área/barra apilada); no-op si no hay data-stack
      });
    });
    // ── Crosshair unificado + highlight de serie más cercana ─────────────────
    var ptsRaw=svg.getAttribute('data-pts');
    var pts=null;
    if(ptsRaw){try{pts=JSON.parse(ptsRaw);}catch(err){}}
    var guide=svg.querySelector('.x-guide');
    var overlay=svg.querySelector('.hover-overlay');
    if(pts&&overlay&&guide){
      var lastSI=-1;
      // Aplica highlight a la serie activa y atenúa las demás (solo cuando cambia).
      function setActive(si){
        if(si===lastSI) return;
        lastSI=si;
        svg.querySelectorAll('[data-si]').forEach(function(el){
          if(el.style.display==='none') return;
          var esi=parseInt(el.getAttribute('data-si'));
          if(si<0){
            el.style.opacity='';
            if(el.tagName==='polyline')el.style.strokeWidth='';
          } else if(esi===si){
            el.style.opacity='1';
            if(el.tagName==='polyline')el.style.strokeWidth='2.8';
          } else {
            el.style.opacity='0.2';
            if(el.tagName==='polyline')el.style.strokeWidth='1.2';
          }
        });
      }
      overlay.addEventListener('mousemove',function(e){
        var sc=svgCoords(svg,e),mx=sc.x,my=sc.y;
        // Fecha más cercana (eje X).
        var best=pts[0],bd=Infinity;
        for(var i=0;i<pts.length;i++){var d=Math.abs(pts[i].x-mx);if(d<bd){bd=d;best=pts[i];}}
        guide.setAttribute('x1',best.x);guide.setAttribute('x2',best.x);
        guide.removeAttribute('display');
        // Series visibles (no ocultas por leyenda).
        var vis=best.vals.filter(function(r){
          var lg=svg.querySelector('.lg-item[data-idx="'+r.i+'"]');
          return !lg||lg.getAttribute('data-hidden')!=='1';
        });
        if(!vis.length){setActive(-1);hide();return;}
        // Serie más cercana en Y (si los datos tienen coordenada y).
        var closestSI=-1;
        if(vis[0].y!==undefined){
          var minDY=Infinity;
          vis.forEach(function(r){var d=Math.abs(r.y-my);if(d<minDY){minDY=d;closestSI=r.i;}});
        }
        setActive(closestSI);
        // Tooltip: todas las series; la más cercana en negrita.
        var rows=vis.map(function(r){
          var hi=(r.i===closestSI);
          return '<div class="ct-r'+(hi?' ct-r-hi':'')+'">'+
            '<span class="ct-sw" style="background:'+esc(r.c)+'"></span>'+
            '<span class="ct-s"'+(hi?' style="color:#000;font-weight:700"':'')+'>'+esc(r.s)+'</span>'+
            '<span class="ct-v">'+esc(r.v)+'</span></div>';
        }).join('');
        tip.innerHTML='<div class="ct-k">'+esc(best.k)+'</div>'+rows;
        tip.style.display='block';move(e);
      });
      overlay.addEventListener('mouseleave',function(){
        guide.setAttribute('display','none');
        setActive(-1);
        hide();
      });
      return;
    }
    // ── Tooltip individual (barras, torta, composición) ──────────────────────
    svg.addEventListener('mouseover',function(e){
      var el=e.target.closest('.tip-pt');
      if(!el) return;
      var c=el.getAttribute('data-c')||'#0b3766',s=el.getAttribute('data-s')||'',
          k=el.getAttribute('data-k')||'',v=el.getAttribute('data-v')||'';
      tip.innerHTML=(k?'<div class="ct-k">'+esc(k)+'</div>':'')+
        '<div class="ct-r"><span class="ct-sw" style="background:'+esc(c)+'"></span>'+
        (s?'<span class="ct-s">'+esc(s)+'</span>':'')+(v?'<span class="ct-v">'+esc(v)+'</span>':'')+'</div>';
      tip.style.display='block';move(e);
    });
    svg.addEventListener('mousemove',function(e){if(tip.style.display==='block')move(e);});
    svg.addEventListener('mouseout',function(e){if(e.target.closest('.tip-pt'))hide();});
  });
})();
"""

_DISCLAIMER = (
    "Informe generado automáticamente desde bases DACE (Todos los textos fueron generados con IA) — "
    "uso interno; validar cifras antes de citar."
)


def _esc(text: str) -> str:
    return html.escape(text or "")


def _attr(text: str) -> str:
    return html.escape(text or "", quote=True)


def _block_html(cb: CuratedBlock) -> str:
    b = cb.block
    parts = [f'<div class="block-title">{_esc(b.title)}</div>']
    if b.unit:
        parts.append(f'<div class="block-unit">({_esc(b.unit)})</div>')
    if b.note:
        parts.append(f'<div class="block-note">{_esc(b.note)}</div>')
    # Fechas/ventanas que cubre el gráfico (para gráficos de ventana donde el eje X
    # no es la fecha: Mes/YtD/Δ7d/corte). Hace explícito qué datos toma cada serie.
    if cb.date_note:
        parts.append(f'<div class="block-dates">📅 {_esc(cb.date_note)}</div>')
    if b.text_slot:
        parts.append(f'<div class="section-text" data-text-slot="{_attr(b.text_slot)}"></div>')
    if cb.render_kind == "chart":
        if cb.preliminary:
            parts.append(
                '<div class="prelim-note">vista preliminar (línea) — el informe final '
                f"usa <code>{_esc(b.chart)}</code></div>"
            )
        parts.append(cb.body_html)
    elif cb.render_kind == "skip":
        parts.append(
            f'<div class="placeholder-card skip"><span class="kind">Sin datos</span><br>'
            f"{_esc(cb.reason)}</div>"
        )
    else:
        parts.append(
            f'<div class="placeholder-card"><span class="kind">{_esc(cb.reason)}</span><br>'
            f"gráfico de tipo <code>{_esc(b.chart)}</code> — se completa en la próxima iteración</div>"
        )
    return (
        f'<div class="block" data-block-status="{_attr(b.status)}" '
        f'data-chart="{_attr(b.chart)}" data-source="{_attr(b.source_id or "")}">'
        f"{''.join(parts)}</div>"
    )


def render_curated_html(report: CuratedReport, *, subtitle: str | None = None) -> str:
    """``CuratedReport`` → documento HTML con banners de sección + gráficos/slots."""
    body: list[str] = []
    current_section: str | None = None
    for cb in report.blocks:
        if cb.block.section != current_section:
            current_section = cb.block.section
            body.append(f'<div class="section-banner">{_esc(current_section)}</div>')
        body.append(_block_html(cb))

    return (
        _SHELL.replace("__TITLE__", _esc(report.spec.title))
        .replace("__SUBTITLE__", _esc(subtitle or _DISCLAIMER))
        .replace("__TIP_JS__", _TIP_JS)
        .replace("__BODY__", "\n".join(body))
    )


def _paragraphs_to_html(text: str) -> str:
    """Texto con párrafos separados por línea en blanco → ``<p>…</p>`` escapados.

    El redactor emite dos párrafos (mensual / semanal) separados por ``\\n\\n``;
    cada uno se envuelve en su propio ``<p>`` para que se vean como párrafos
    distintos en el informe."""
    parts = re.split(r"\n\s*\n", (text or "").strip())
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in parts if p.strip())


def _inline_md(text: str) -> str:
    """Escapa y convierte ``**negrita**`` a ``<strong>`` (markdown inline mínimo)."""
    s = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


def _synthesis_md_to_html(md: str) -> str:
    """Markdown simple de la síntesis → HTML (párrafos, listas ``- `` y negritas).

    Determinista, sin librería externa (mismo criterio que el resto del informe).
    Reconoce viñetas (``- ``/``* ``/``• ``) como ``<li>`` y el resto como ``<p>``;
    ignora los ``#`` de encabezado dejando el texto en un párrafo."""
    out: list[str] = []
    in_list = False
    for raw in (md or "").splitlines():
        s = raw.strip()
        if not s or re.fullmatch(r"[-*_]{3,}", s):
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if s[:2] in ("- ", "* ") or s.startswith("• "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_md(s[2:].strip())}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            text = re.sub(r"^#{1,6}\s+", "", s)
            out.append(f"<p>{_inline_md(text)}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def fill_synthesis_slot(html_content: str, synthesis_md: str) -> str:
    """Inyecta la síntesis en ``<div ... data-synthesis-body></div>`` (vacío) del
    HTML curado. ``synthesis_md`` admite párrafos, viñetas y ``**negrita**``."""
    if not synthesis_md or not synthesis_md.strip():
        return html_content
    inner = _synthesis_md_to_html(synthesis_md)
    pattern = re.compile(r'(<div\b[^>]*\bdata-synthesis-body[^>]*>)\s*(</div>)')
    return pattern.sub(lambda m: m.group(1) + inner + m.group(2), html_content, count=1)


def fill_text_slots(html_content: str, slots: dict[str, str]) -> str:
    """Inyecta párrafos en los ``<div data-text-slot="…">`` vacíos del HTML curado.

    ``slots`` es ``{slot_id: texto_plano}`` (cada valor puede traer dos párrafos
    separados por línea en blanco). Solo se tocan divs que aún estén vacíos
    (``></div>`` sin contenido); si ya tienen texto no se sobreescriben. El texto
    se escapa antes de insertar y cada párrafo va en su propio ``<p>``.
    """
    for slot_id, paragraph in slots.items():
        if not paragraph:
            continue
        inner = _paragraphs_to_html(paragraph)
        pattern = re.compile(
            r'(<div\b[^>]*\bdata-text-slot="' + re.escape(slot_id) + r'"[^>]*>)\s*(</div>)',
        )
        html_content = pattern.sub(lambda m, inner=inner: m.group(1) + inner + m.group(2), html_content)
    return html_content
