"""Sample 10% estratificado de chunks_enriched_full.json + todas las imágenes.

Uso:  py scripts/sample_chunks.py
Lee logs/chunks_enriched_full.json, escribe logs/chunks_enriched.json (sample).
"""
from __future__ import annotations
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("logs/chunks_enriched_full.json")
DST = Path("logs/chunks_enriched.json")
RATIO = 0.10
SEED = 42


def main() -> int:
    rng = random.Random(SEED)
    chunks = json.loads(SRC.read_text(encoding="utf-8"))

    images = [c for c in chunks if c.get("image_path")]
    text = [c for c in chunks if not c.get("image_path")]

    by_type: dict[str, list] = defaultdict(list)
    for c in text:
        by_type[c.get("doc_type_category", "UNKNOWN")].append(c)

    sampled_text: list = []
    for dt, group in by_type.items():
        k = max(1, round(len(group) * RATIO))
        sampled_text.extend(rng.sample(group, k))

    sample = images + sampled_text
    rng.shuffle(sample)

    DST.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total original:    {len(chunks)} (texto={len(text)}, imagen={len(images)})")
    print(f"Sampled:           {len(sample)} (texto={len(sampled_text)}, imagen={len(images)})")
    print("Por doc_type (texto):")
    orig_counts = Counter(c.get("doc_type_category", "UNKNOWN") for c in text)
    samp_counts = Counter(c.get("doc_type_category", "UNKNOWN") for c in sampled_text)
    for dt in sorted(orig_counts):
        print(f"  {dt:20s} {samp_counts[dt]:>5d} / {orig_counts[dt]:>5d}")
    print(f"Output: {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
