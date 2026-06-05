"""Tests de la subida de archivos como contexto efímero (upload_store)."""

from __future__ import annotations

import importlib

import pytest

import banks_rag.infrastructure.uploads.upload_store as store
from banks_rag.infrastructure.uploads import UploadError, extract_upload


@pytest.fixture
def uploads_dir(tmp_path, monkeypatch):
    """Aísla DATA_UPLOADS_DIR en un tmp por test."""
    monkeypatch.setattr(store, "DATA_UPLOADS_DIR", tmp_path / "uploads")
    return tmp_path / "uploads"


@pytest.mark.unit
class TestExtractUpload:
    def test_csv_becomes_table_with_schema_and_stats(self) -> None:
        csv = b"Fecha,Cobre\n2026-05-19,622.5\n2026-05-20,624.9\n"
        rec = extract_upload("cobre.csv", csv)
        assert rec["kind"] == "table"
        assert "Columnas: Fecha" in rec["text"]
        assert "624.9" in rec["text"]
        assert "Estadística" in rec["text"]

    def test_txt_becomes_document(self) -> None:
        rec = extract_upload("nota.txt", "La TPM es 4,5%.".encode())
        assert rec["kind"] == "document"
        assert "TPM" in rec["text"]

    def test_json_news_list_formatted(self) -> None:
        import json
        raw = json.dumps([
            {"titulo": "BCCh mantiene TPM", "fecha": "2026-05-20",
             "fuente": "DF", "contenido": "El Consejo decidió..."},
        ]).encode()
        rec = extract_upload("noticias.json", raw)
        assert rec["kind"] == "document"
        assert "BCCh mantiene TPM" in rec["text"]
        assert "DF" in rec["text"] and "El Consejo decidió" in rec["text"]
        assert "{" not in rec["text"]  # no vuelca el JSON crudo

    def test_json_nested_articles(self) -> None:
        import json
        raw = json.dumps({"articles": [{"title": "Fed holds", "content": "kept rates"}]}).encode()
        rec = extract_upload("scrape.json", raw)
        assert "Fed holds" in rec["text"] and "kept rates" in rec["text"]

    def test_json_real_scraper_format_emailtitle_text(self) -> None:
        """El formato real del scraper (emailTitle/text/source) se reconoce
        como noticia (no vuelca el JSON crudo) y repara el mojibake por campo."""
        import json

        def mangle(s: str) -> str:
            return s.encode("utf-8").decode("cp1252")

        raw = json.dumps([
            {"emailTitle": mangle("Sube la inflación"),
             "source": "DIARIO FINANCIERO - CHILE - ECONOMIA",
             "topic": "Banco Central",
             "text": mangle("El IPC sorprendió al alza en el país.")},
        ]).encode()
        rec = extract_upload("noticias_2026_06_04.json", raw)
        assert "Sube la inflación" in rec["text"]  # emailTitle reconocido como título
        assert "El IPC sorprendió" in rec["text"]  # text reconocido como cuerpo
        assert "país" in rec["text"]
        assert "{" not in rec["text"]  # formateado como noticia, no JSON crudo
        assert "Ã" not in rec["text"]  # mojibake reparado

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(UploadError, match="JSON inválido"):
            extract_upload("bad.json", b"{not valid json")

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(UploadError, match="no soportado"):
            extract_upload("virus.exe", b"MZ\x00")

    def test_empty_raises(self) -> None:
        with pytest.raises(UploadError, match="vac"):
            extract_upload("x.csv", b"")

    def test_oversize_raises(self) -> None:
        big = b"x" * (store.MAX_UPLOAD_BYTES + 1)
        with pytest.raises(UploadError, match="grande"):
            extract_upload("big.txt", big)

    def test_text_is_clipped(self, monkeypatch) -> None:
        monkeypatch.setattr(store, "MAX_TEXT_CHARS", 50)
        rec = extract_upload("long.txt", ("A" * 500).encode())
        assert len(rec["text"]) == 50
        assert rec["truncated"] is True


@pytest.mark.unit
class TestSaveLoadUpload:
    def test_save_then_load_roundtrip(self, uploads_dir) -> None:
        rec = store.save_upload("nota.txt", "contenido".encode())
        assert rec["upload_id"].startswith("up_")
        assert rec["kind"] == "document"
        assert "preview" in rec and "chars" in rec

        loaded = store.load_upload(rec["upload_id"])
        assert loaded is not None
        assert loaded["text"] == "contenido"

    def test_load_unknown_returns_none(self, uploads_dir) -> None:
        assert store.load_upload("up_doesnotexist") is None

    def test_load_rejects_traversal_ids(self, uploads_dir) -> None:
        assert store.load_upload("../etc/passwd") is None
        assert store.load_upload("up_../x") is None

    def test_expired_upload_is_purged(self, uploads_dir, monkeypatch) -> None:
        rec = store.save_upload("nota.txt", b"hola")
        monkeypatch.setattr(store, "TTL_SECONDS", -1)   # todo está "caducado"
        assert store.load_upload(rec["upload_id"]) is None


def test_module_imports_clean() -> None:
    # Smoke: el paquete re-exporta lo esperado.
    mod = importlib.import_module("banks_rag.infrastructure.uploads")
    assert hasattr(mod, "save_upload") and hasattr(mod, "load_upload")
