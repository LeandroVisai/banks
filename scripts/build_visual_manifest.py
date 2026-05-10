"""Genera logs/visuals_manifest.json — un índice de los chunks visuales.

Permite revisar qué imágenes se vectorizaron y con qué metadata sin abrir
todo chunks_enriched.json. Útil para QA y para que el agente liste visuales
disponibles filtrando por institución/fecha/importancia/variable.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ENRICHED = Path("logs/chunks_enriched.json")
DOCS = Path("logs/documents.json")
OUT = Path("logs/visuals_manifest.json")


def main() -> int:
    chunks = json.loads(ENRICHED.read_text(encoding="utf-8"))
    docs = {d["document_id"]: d for d in json.loads(DOCS.read_text(encoding="utf-8"))}

    visuals = [c for c in chunks if c.get("image_path")]

    rows = []
    for c in visuals:
        doc = docs.get(c["document_id"], {})
        date = c.get("document_date") or doc.get("document_date")
        year = str(date)[:4] if date and len(str(date)) >= 4 else None
        month = str(date)[5:7] if date and len(str(date)) >= 7 else None

        top_var = None
        var_imp = None
        ev = c.get("economic_variables") or {}
        if ev:
            order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
            top = sorted(ev.items(), key=lambda kv: order.get(kv[1].get("importance", "MEDIUM"), 3))[0]
            top_var, var_imp = top[0], top[1].get("importance")

        rows.append({
            "chunk_id": c["chunk_id"],
            "image_path": c["image_path"],
            "document_id": c["document_id"],
            "page": c.get("page_start"),
            "institution": c.get("institution") or doc.get("institution"),
            "doc_type": c.get("doc_type_category") or doc.get("doc_type_category"),
            "date": date,
            "year": year,
            "month": month,
            "section_type": c.get("section_type"),
            "importance_score": round(c.get("importance_score", 0), 3),
            "is_policy_decision": c.get("is_policy_decision", False),
            "top_variable": top_var,
            "top_variable_importance": var_imp,
            "all_variables": list(ev.keys()),
            "caption_in_text": _extract_caption_line(c.get("text", "")),
        })

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ {OUT} ({len(rows)} visuales)")
    print()
    print("=== Cobertura ===")
    fields = ["institution", "doc_type", "year", "month", "top_variable"]
    for f in fields:
        cov = sum(1 for r in rows if r.get(f))
        print(f"  {f:20s} {cov}/{len(rows)}")

    print()
    print("=== Distribución por institución ===")
    for inst, n in Counter(r["institution"] for r in rows).most_common():
        print(f"  {inst:25s} {n}")

    print()
    print("=== Distribución por importance ===")
    for r in sorted(rows, key=lambda x: -x["importance_score"])[:10]:
        cap = (r["caption_in_text"] or "—")[:60]
        print(f"  imp={r['importance_score']:.2f}  {r['institution']:15s}  {r['date'] or '-':10s}  {cap}")
    return 0


def _extract_caption_line(text: str) -> str | None:
    """De `[VISUAL ...]\\n[CHART p.N]\\nFigure 3: ...\\n...` extrae 'Figure 3: ...'."""
    if not text:
        return None
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("["):
            continue
        return line[:200]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
