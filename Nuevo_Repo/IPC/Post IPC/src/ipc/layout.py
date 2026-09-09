"""Dónde va cada gráfico: secciones, orden y etiquetas del informe IPC.

Sigue el orden del "Post IPC" del informe del banco. Un gráfico que existe
(en el notebook o en figures.BUILDERS) pero no tiene slot acá no se dibuja
en ninguna parte — y tampoco pesa.

Para agregar un gráfico: una línea en el `slots` de la sección que
corresponda, con el `fig_id` que usaste en figures.BUILDERS o la clave que
usaste en el dict FIGURAS del notebook.

Además de los gráficos hay dos tipos de slot que no son figuras:

    {"tabla": "canasta"}   una tabla filtrable y ordenable (tables.py)
    {"nota": "..."}        un bloque en blanco para escribir y pegar

Los bloques `nota` son para lo que no sale del pipeline: análisis escrito a
mano y gráficos pegados desde Excel. Título y cuerpo se editan en el
navegador y viajan con "Guardar versión final".
"""
from __future__ import annotations


def build_nav_groups() -> list[dict]:
    return [
        {"label": "Resumen", "items": [
            {"id": "portada", "label": "Dato del mes", "cols": 2, "slots": [
                {"label": "IPC esperado vs efectivo",         "fig_id": "esperado_vs_efectivo"},
                {"label": "Variaciones del mes por agregado", "fig_id": "variaciones_mes"},
            ]},
        ]},
        {"label": "Expectativas", "items": [
            {"id": "expectativas", "label": "Esperado vs efectivo", "cols": 2, "hero": True, "slots": [
                {"label": "Incidencias: feedback de mercado vs efectivo", "fig_id": "incidencias_feedback"},
            ]},
        ]},
        {"label": "Difusión", "items": [
            {"id": "difusion", "label": "Difusión inflacionaria", "cols": 2, "hero": True, "slots": [
                {"label": "Difusión IPC general · banda histórica", "fig_id": "difusion_general"},
                {"label": "Bienes sin volátiles · banda",          "fig_id": "bienes_sv_banda"},
                {"label": "Servicios sin volátiles · banda",       "fig_id": "servicios_sv_banda"},
            ]},
        ]},
        # El treemap va solo, en una fila de una columna: es una jerarquía de
        # cuatro niveles y a media tarjeta las glosas no se leen. El resto de la
        # sección va en la grilla de dos columnas de siempre.
        {"label": "Canasta", "items": [
            {"id": "canasta", "label": "Canasta e incidencias", "cols": 2, "slots": [
                {"label": "Canasta por división, grupo y producto",  "fig_id": "treemap_canasta",
                 "wide": True, "card_height": 620},
                {"label": "Variación IPC por grupo",                 "fig_id": "variacion_ipc_grupo"},
                {"label": "Incidencia 12 meses por división",        "fig_id": "incidencia_12m_division"},
                {"label": "Variación histórica del mes",             "fig_id": "variacion_historica_mes"},
            ]},
        ]},
        # Para lo que no sale del pipeline: se escribe y se pega a mano en el
        # navegador. Cada bloque replica la estructura del informe Excel del
        # banco — franja de título, el texto, otra franja, y abajo los gráficos:
        #
        #     ┌──────────── Claves Flujos Día IPC ────────────┐   label
        #     │ Durante la jornada predominaron los flujos... │   texto
        #     ├──────────── Flujos (US$ MM) ──────────────────┤   label_graficos
        #     │   [gráfico]        [gráfico]                  │   se pegan con Ctrl+V
        #     │   [gráfico]        [gráfico]                  │
        #     └───────────────────────────────────────────────┘
        #
        # Los tres textos (los dos títulos y el cuerpo) son editables. Van a una
        # columna porque los gráficos pegados se acomodan de a dos adentro.
        {"label": "Anexo", "items": [
            {"id": "anexo", "label": "Análisis complementario", "cols": 1, "slots": [
                {"nota": "anexo-1", "label": "Claves del mes",
                 "label_graficos": "Gráficos"},
            ]},
        ]},
        {"label": "Variación de expectativas y Fundamentales", "items": [
            {"id": "variacion", "label": "Expectativas y Fundamentales", "cols": 2, "slots": [
                {"label": "Fundamentales: Petróleo y Tipo de Cambio",        "fig_id": "fundamentales_inf"},
                {"label": "Variación: Seguros de Inflación",             "fig_id": "variacion_si"},
                {"label": "Expectativas de Inflación",             "fig_id": "expectativas_si"},
                {"label": "Expectativas de Inflación a largo plazo",             "fig_id": "expectativas_ci"},
            ]},
        ]},
        # Al final, como en el dashboard anterior: todos los productos con su
        # variación e incidencia. Un slot con `tabla` en vez de `fig_id`; los
        # datos los arma tables.build_tablas().
        {"label": "Detalle", "items": [
            {"id": "detalle", "label": "Detalle por producto", "cols": 1, "slots": [
                {"label": "Canasta del último mes · variación e incidencia por producto", "tabla": "canasta"},
            ]},
        ]},
    ]


_ALTO_TARJETA = 440


def annotate_layout(groups: list, figs: dict) -> None:
    for group in groups:
        for item in group["items"]:
            vivos, pendientes, aparte = [], [], []
            for slot in item["slots"]:
                if slot.get("tabla"):
                    # Las tablas no son figuras: siempre anchas, nunca pendientes.
                    slot.update(pending=False, wide=True)
                    slot.setdefault("fig_id", None)
                    aparte.append(slot)
                    continue
                if slot.get("nota"):
                    # Los bloques para escribir/pegar tampoco: van en la grilla
                    # tal como se declararon, sin ensanchar ni reordenar.
                    slot.update(pending=False)
                    slot.setdefault("fig_id", None)
                    aparte.append(slot)
                    continue
                slot["pending"] = slot.get("fig_id") not in figs
                (pendientes if slot["pending"] else vivos).append(slot)
                if not slot["pending"] and not slot.get("card_height"):
                    # Un card_height puesto a mano en el slot manda; sólo se deduce
                    # de la figura cuando ella misma pidió más alto que la tarjeta.
                    alto = figs[slot["fig_id"]].get("layout", {}).get("height")
                    if alto and alto > _ALTO_TARJETA:
                        slot["card_height"] = alto
            # `hero` ensancha el primero de la sección. Los demás quedan en la
            # grilla de dos columnas tal cual: antes el último de un grupo impar
            # se ensanchaba solo para no dejar hueco, pero eso rompía la lectura
            # de dos columnas y ensanchaba gráficos que no lo necesitaban.
            if item.get("hero") and vivos:
                vivos[0]["wide"] = True
            item["slots"] = vivos + aparte + pendientes
