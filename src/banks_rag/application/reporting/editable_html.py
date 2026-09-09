"""
Versión EDITABLE de un informe HTML (puerto de "Guardar editable" del chartbuilder).

make_editable_html inyecta en un informe ya renderizado el mismo comportamiento
que tools/chart_builder.html ofrece con su botón "Guardar editable": marca la
Síntesis y los párrafos por dataset como contenteditable y añade un panel
flotante (💾 Guardar / 📄 Versión final) para escribir DENTRO del informe en el
navegador y guardarlo a mano, sin pasar por el chartbuilder.

El JS es autocontenido (sin red ni CDN): descarga el HTML actual vía Blob URL,
igual que el chartbuilder. Apunta a los MISMOS selectores que el chartbuilder
([data-synthesis-body], [data-text-slot], .section-text), de modo que el
informe descriptivo (html_render.py) queda editable tanto abriéndolo directo
como cargándolo en el chartbuilder.

strip_editable_chrome hace lo inverso (para el paso a correo / "versión final"):
quita el panel, el script y los contenteditable + su contorno de edición.
"""

from __future__ import annotations

import re


# JS autocontenido que corre al abrir el HTML: hace editable la Síntesis y los
# párrafos y monta el panel 💾/📄. ``\n`` queda literal (raw string): lo interpreta
# el motor JS del navegador, no Python.
_EDITABLE_JS = r"""(function(){
  function dl(h,n){
    var a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([h],{type:'text/html'}));
    a.download=n;
    a.click();
  }

  function base(){
    var file = '';

    try {
      file = decodeURIComponent((location.pathname || '').split('/').pop() || '');
    } catch(e) {
      file = '';
    }

    // Quita extensión .html / .htm
    file = file.replace(/\.html?$/i, '');

    // Quita solo el sufijo _editable si está al final
    // Ejemplo: ffmm_2026-07-23_editable -> ffmm_2026-07-23
    file = file.replace(/_editable$/i, '');

    // Fallback si no hay nombre de archivo, por ejemplo en HTML embebido
    if(!file) file = document.title || 'reporte';

    return file.replace(/[^\w.-]+/g,'_').slice(0,80);
  }

  function mk(){
    document.querySelectorAll('[data-synthesis-body],[data-text-slot],.section-text').forEach(function(el){
      el.setAttribute('contenteditable','true');
      el.style.outline='1px dashed #cbd5e1';
      el.style.outlineOffset='2px';
    });

    if(document.getElementById('cb-save-widget'))return;

    var bar=document.createElement('div');
    bar.id='cb-save-widget';
    bar.setAttribute('data-cb-chrome','');
    bar.style.cssText='position:fixed;bottom:16px;right:16px;z-index:99999;display:flex;gap:8px;';

    bar.innerHTML='<button type="button" data-cb="save" style="background:#0b3766;color:#fff;border:none;border-radius:6px;padding:10px 14px;font:700 13px Arial;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25);">💾 Guardar</button>'
      +'<button type="button" data-cb="final" style="background:#b08d57;color:#1f1f1f;border:none;border-radius:6px;padding:10px 14px;font:700 13px Arial;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25);">📄 Versión final</button>';

    document.body.appendChild(bar);

    function save(){
      dl('<!DOCTYPE html>\n'+document.documentElement.outerHTML,base()+'_editado.html');
    }

    function fin(){
      var c=document.documentElement.cloneNode(true);

      c.querySelectorAll('#cb-save-widget,#cb-editable-script,[data-cb-chrome]').forEach(function(el){
        el.remove();
      });

      c.querySelectorAll('[contenteditable]').forEach(function(el){
        el.removeAttribute('contenteditable');
        el.style.outline='';
        el.style.outlineOffset='';
      });

      dl('<!DOCTYPE html>\n'+c.outerHTML,base()+'.html');
    }

    bar.querySelector('[data-cb=save]').addEventListener('click',save);
    bar.querySelector('[data-cb=final]').addEventListener('click',fin);
  }

  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded',mk);
  }else{
    mk();
  }
})();"""


_SCRIPT_ID = "cb-editable-script"
_EDITABLE_SCRIPT = f'<script id="{_SCRIPT_ID}">{_EDITABLE_JS}</script>'


def make_editable_html(html: str) -> str:
    """Inyecta el panel editable (💾/📄) antes de </body>.

    Idempotente: si el HTML ya lo trae, lo devuelve sin cambios.
    No modifica el contenido del informe; la edición ocurre en el navegador.
    """
    if f'id="{_SCRIPT_ID}"' in html:
        return html

    if "</body>" in html:
        return html.replace("</body>", f"{_EDITABLE_SCRIPT}\n</body>", 1)

    return html + _EDITABLE_SCRIPT


# ── Inverso: quitar el chrome de edición (para "versión final" / paso a correo) ──

_RE_EDIT_SCRIPT = re.compile(
    r'<script id="(?:cb-editable-script|cb-save-script)"[^>]*>.*?</script>',
    re.S,
)

_RE_SAVE_WIDGET = re.compile(
    r'<div id="cb-save-widget"[^>]*>.*?</div>',
    re.S,
)

_RE_CHROME_NODE = re.compile(
    r'<[a-zA-Z]+[^>]*\sdata-cb-chrome[^>]*>.*?</[a-zA-Z]+>',
    re.S,
)

_RE_CONTENTEDITABLE = re.compile(
    r'\s+contenteditable="[^"]*"'
)

# El contorno de edición que el JS aplica en runtime (el.style.outline=...);
# el navegador puede serializar el color como rgb(...). Se quita de cualquier style.
_RE_OUTLINE = re.compile(
    r'\s*outline(?:-offset)?\s*:[^;"]*;?'
)

# style que quedó vacío tras quitar el contorno: se elimina para que un inliner
# posterior no lo confunda con "ya tiene estilo".
_RE_EMPTY_STYLE = re.compile(
    r'\s+style="\s*"'
)


def strip_editable_chrome(html: str) -> str:
    """Quita el panel flotante, el script editable y los contenteditable +
    contorno de edición. Deja el informe limpio, equivalente a "Versión final".
    """
    html = _RE_EDIT_SCRIPT.sub("", html)
    html = _RE_SAVE_WIDGET.sub("", html)
    html = _RE_CHROME_NODE.sub("", html)
    html = _RE_CONTENTEDITABLE.sub("", html)
    html = _RE_OUTLINE.sub("", html)
    html = _RE_EMPTY_STYLE.sub("", html)

    return html