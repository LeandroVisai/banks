"""Tests del índice semántico del catálogo (con embedder fake, sin modelo real).

Verifica: scoring por coseno, degradación elegante (off por defecto / sin
modelo → {}), reuso por hash y robustez ante fallos del embedder.
"""

from __future__ import annotations

import numpy as np
import pytest

from banks_rag.infrastructure.sql import catalog_index as ci
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ColumnSpec,
    ParquetDataset,
)


def _ds(id_: str, name: str, desc: str, segment: str = "x") -> ParquetDataset:
    return ParquetDataset(
        id=id_, file=f"{id_}.parquet", name=name, description=desc,
        segment=segment, unit="u", date_range=None,
        columns=[ColumnSpec(name="Fecha", type="TIMESTAMP")],
    )


class _FakeEmbedder:
    """Embedder determinista: mapea palabras clave a ejes ortogonales."""

    name = "fake"
    dim = 4
    max_seq_length = 0
    is_multimodal = False
    # eje por keyword
    _AXES = {"dolar": 0, "cobre": 1, "pension": 2, "bono": 3}

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(4, dtype=np.float32)
        low = text.lower()
        for kw, ax in self._AXES.items():
            if kw in low:
                v[ax] += 1.0
        if not v.any():
            v[0] = 1e-3  # evita vector nulo
        return v

    def encode_text(self, texts, batch_size: int = 4):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def entries():
    return [
        _ds("usdclp", "Tipo de cambio dolar", "serie del dolar contra el peso"),
        _ds("cobre_dxy", "Precio del cobre", "cobre y dxy"),
        _ds("dv01_afp", "DV01 fondos de pension", "sensibilidad de pension"),
        _ds("btp", "Curva de bono soberano", "bono btp"),
    ]


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    # Aísla la persistencia a un tmp para no escribir en data/ del repo.
    monkeypatch.setattr(ci, "_CACHE_PATH", tmp_path / "catalog_index.npz")
    ci.reset_catalog_index()
    yield
    ci.reset_catalog_index()


@pytest.mark.unit
class TestCatalogSemanticScores:
    def test_disabled_without_embedder_returns_empty(self, entries, monkeypatch) -> None:
        monkeypatch.delenv("BANKS_CATALOG_SEMANTIC", raising=False)
        assert ci.catalog_semantic_scores("dolar", entries) == {}

    def test_scores_rank_relevant_dataset_top(self, entries) -> None:
        emb = _FakeEmbedder()
        scores = ci.catalog_semantic_scores("precio del dolar", entries, embedder=emb)
        assert scores  # no vacío
        top = max(scores, key=scores.get)
        assert top == "usdclp"

    def test_pension_query_matches_afp(self, entries) -> None:
        emb = _FakeEmbedder()
        scores = ci.catalog_semantic_scores("pension", entries, embedder=emb)
        assert max(scores, key=scores.get) == "dv01_afp"

    def test_reuses_index_when_catalog_unchanged(self, entries) -> None:
        calls = {"n": 0}

        class _CountingEmbedder(_FakeEmbedder):
            def encode_text(self, texts, batch_size: int = 4):
                calls["n"] += 1
                return super().encode_text(texts, batch_size)

        emb = _CountingEmbedder()
        ci.catalog_semantic_scores("dolar", entries, embedder=emb)
        first = calls["n"]
        ci.catalog_semantic_scores("cobre", entries, embedder=emb)
        # La segunda llamada NO reconstruye el índice (mismo hash): solo embebe
        # la query nueva, no los 4 datasets de nuevo.
        assert calls["n"] == first + 1

    def test_embedder_failure_degrades_to_empty(self, entries) -> None:
        class _BrokenEmbedder(_FakeEmbedder):
            def encode_text(self, texts, batch_size: int = 4):
                raise RuntimeError("modelo caído")

        assert ci.catalog_semantic_scores("dolar", entries, embedder=_BrokenEmbedder()) == {}


@pytest.mark.unit
class TestSearchDatasetsWithExtraScores:
    def test_extra_scores_boosts_ranking(self) -> None:
        from banks_rag.infrastructure.sql.parquet_catalog_loader import search_datasets
        ds = [
            _ds("a", "alfa", "uno"),
            _ds("b", "beta", "dos"),
        ]
        # Sin extra_scores 'a' (orden de catálogo) gana el empate; con un boost
        # semántico a 'b', 'b' pasa al frente.
        res = search_datasets(ds, "zzz sin overlap", top_k=2, extra_scores={"b": 9.0})
        assert res[0].id == "b"
