"""Tests de integración del API con TestClient y mocks."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from banks_rag.domain.agent import AgentResult
from banks_rag.domain.retrieval import SearchResult
from banks_rag.interface.api import AppState, create_app


class _MockLLM:
    name = "mock-llm"
    loaded = True

    async def generate(self, messages, **kwargs):
        from banks_rag.domain.agent import GenerationResult
        return GenerationResult(text="Respuesta mock", n_tokens=10)

    def count_text_tokens(self, text: str) -> int:
        return len(text) // 4


class _MockEmbedder:
    name = "mock-embedder"
    dim = 4
    max_seq_length = 256
    is_multimodal = False

    def encode_text(self, texts, batch_size=4):
        import numpy as np
        return np.zeros((len(texts), self.dim), dtype=float)

    def count_tokens(self, text: str) -> int:
        return 0


class _MockRepo:
    docs_table = "documents"
    chunks_table = "chunks"
    database = "test"

    def connect(self):
        from contextlib import contextmanager

        class _Conn:
            def cursor(self):
                class _Cur:
                    def execute(self, *a, **k): return None
                    def fetchone(self): return (1,)
                    def fetchall(self): return []
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                return _Cur()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        return _Conn()


def _build_app() -> tuple[TestClient, AppState]:
    deps = AppState()
    deps.llm = _MockLLM()
    deps.embedder = _MockEmbedder()
    deps.repo = _MockRepo()
    app = create_app(deps=deps)
    return TestClient(app), deps


@pytest.mark.integration
class TestHealth:
    def test_healthz(self) -> None:
        client, _ = _build_app()
        # Forzar al cliente a no triggerar el lifespan completo (mocks ya OK)
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["llm_loaded"] is True
        assert "version" in body

    def test_readyz_with_loaded_llm(self) -> None:
        client, _ = _build_app()
        response = client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ok", "degraded", "loading")

    def test_request_id_in_response_header(self) -> None:
        client, _ = _build_app()
        response = client.get("/healthz")
        assert "x-request-id" in {k.lower() for k in response.headers}

    def test_request_id_respects_client_header(self) -> None:
        client, _ = _build_app()
        custom_id = "test-id-12345"
        response = client.get("/healthz", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id


@pytest.mark.integration
class TestChat:
    def test_chat_with_mock_llm(self) -> None:
        client, _ = _build_app()
        response = client.post(
            "/v1/chat",
            json={"message": "¿Cuál fue la TPM en enero 2024?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "Respuesta mock"
        assert body["iterations"] >= 1
        assert body["model"] == "mock-llm"
        assert "prompt_version" in body

    def test_chat_accepts_long_history(self) -> None:
        """El chatbot acepta historial largo (memoria); el backend lo recorta a
        history_max_turns sin romper (antes el schema topaba en 40)."""
        client, _ = _build_app()
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(80)
        ]
        response = client.post(
            "/v1/chat",
            json={"message": "última pregunta", "history": history},
        )
        assert response.status_code == 200

    def test_chat_accepts_thinking_mode(self) -> None:
        """El frontend puede enviar thinking_mode por request (off/adaptive/on)."""
        client, _ = _build_app()
        response = client.post(
            "/v1/chat",
            json={"message": "precio del cobre", "thinking_mode": "off"},
        )
        assert response.status_code == 200

    def test_chat_rejects_invalid_thinking_mode(self) -> None:
        client, _ = _build_app()
        response = client.post(
            "/v1/chat",
            json={"message": "hola", "thinking_mode": "turbo"},
        )
        assert response.status_code == 422

    def test_upload_csv_then_chat_with_attachment(self, tmp_path, monkeypatch) -> None:
        """Sube un CSV (contexto efímero) y lo referencia en /v1/chat."""
        import base64

        import banks_rag.infrastructure.uploads.upload_store as store
        monkeypatch.setattr(store, "DATA_UPLOADS_DIR", tmp_path / "uploads")

        client, _ = _build_app()
        csv_b64 = base64.b64encode(b"Fecha,Cobre\n2026-05-20,624.9\n").decode()
        up = client.post("/v1/upload", json={"filename": "cobre.csv", "content_base64": csv_b64})
        assert up.status_code == 200
        body = up.json()
        assert body["kind"] == "table"
        assert body["upload_id"].startswith("up_")

        chat = client.post(
            "/v1/chat",
            json={"message": "analiza estos datos", "attachments": [body["upload_id"]]},
        )
        assert chat.status_code == 200

    def test_upload_rejects_bad_base64(self) -> None:
        client, _ = _build_app()
        r = client.post("/v1/upload", json={"filename": "x.csv", "content_base64": "!!!notb64!!!"})
        assert r.status_code == 400

    def test_upload_rejects_unsupported_type(self, tmp_path, monkeypatch) -> None:
        import base64

        import banks_rag.infrastructure.uploads.upload_store as store
        monkeypatch.setattr(store, "DATA_UPLOADS_DIR", tmp_path / "uploads")

        client, _ = _build_app()
        b64 = base64.b64encode(b"MZ\x00binary").decode()
        r = client.post("/v1/upload", json={"filename": "virus.exe", "content_base64": b64})
        assert r.status_code == 400

    def test_news_report_endpoint(self, tmp_path, monkeypatch) -> None:
        """/v1/news-report genera el reporte del JSON más reciente."""
        import json as _json

        import banks_rag.application.news.report as report
        monkeypatch.setattr(report, "NEWS_SCRAPING_DIR", tmp_path)
        (tmp_path / "n.json").write_text(_json.dumps([
            {"topic": "Banco Central", "audience": "100,00K", "title": "x", "text": "y"},
        ]), encoding="utf-8")

        client, _ = _build_app()
        r = client.post("/v1/news-report", json={"top_n": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["n_total"] == 1 and body["source_file"] == "n.json"
        assert isinstance(body["report"], str) and body["report"]

    def test_news_report_404_when_no_files(self, tmp_path, monkeypatch) -> None:
        import banks_rag.application.news.report as report
        monkeypatch.setattr(report, "NEWS_SCRAPING_DIR", tmp_path / "vacio")
        client, _ = _build_app()
        r = client.post("/v1/news-report", json={})
        assert r.status_code == 404

    def test_chat_returns_503_when_llm_unloaded(self) -> None:
        deps = AppState()
        deps.llm = None
        deps.embedder = _MockEmbedder()
        deps.repo = _MockRepo()
        app = create_app(deps=deps)
        client = TestClient(app)
        response = client.post("/v1/chat", json={"message": "hola"})
        assert response.status_code == 503

    def test_chat_validates_message_length(self) -> None:
        client, _ = _build_app()
        response = client.post("/v1/chat", json={"message": ""})
        assert response.status_code == 422  # Pydantic min_length=1

    def test_chat_rejects_extra_fields(self) -> None:
        client, _ = _build_app()
        response = client.post(
            "/v1/chat",
            json={"message": "hola", "rogue_field": "x"},
        )
        assert response.status_code == 422


@pytest.mark.integration
class TestSearch:
    def test_search_calls_hybrid_search(self) -> None:
        client, _ = _build_app()

        fake_result = SearchResult(
            query="x", clean_query="x",
            hits=[{
                "chunk_id": "c1", "document_id": "d1",
                "filename": "doc.pdf", "doc_type_category": "COMUNICADO",
                "document_date": "2024-03-15",
                "page_start": 1, "page_end": 1,
                "section_type": "DECISION",
                "importance_score": 0.85,
                "rrf_score": 0.5, "final_score": 0.6,
                "economic_variables": {},
                "tags": [], "image_path": None,
                "text": "ejemplo",
            }],
            parsed_filters={"year_from": 2024},
        )

        with patch(
            "banks_rag.interface.api.routes.search.hybrid_search",
            return_value=fake_result,
        ):
            response = client.post(
                "/v1/search",
                json={"query": "TPM 2024", "k": 5},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["n_results"] == 1
        assert body["results"][0]["chunk_id"] == "c1"
        assert body["filters"]["year_from"] == 2024

    def test_search_empty_results(self) -> None:
        client, _ = _build_app()
        empty = SearchResult(query="x", clean_query="x", hits=[], parsed_filters={})
        with patch(
            "banks_rag.interface.api.routes.search.hybrid_search",
            return_value=empty,
        ):
            response = client.post("/v1/search", json={"query": "nada"})
        assert response.status_code == 200
        assert response.json()["n_results"] == 0

    def test_search_validates_k_range(self) -> None:
        client, _ = _build_app()
        response = client.post("/v1/search", json={"query": "x", "k": 100})
        assert response.status_code == 422

    def test_search_with_filters(self) -> None:
        client, _ = _build_app()
        empty = SearchResult(query="x", clean_query="x", hits=[], parsed_filters={})
        with patch(
            "banks_rag.interface.api.routes.search.hybrid_search",
            return_value=empty,
        ) as mock:
            response = client.post(
                "/v1/search",
                json={
                    "query": "x",
                    "filters": {
                        "institutions": ["BANCO_CENTRAL_CHILE"],
                        "tags": ["DECISION_POLITICA"],
                        "min_importance": 0.5,
                    },
                },
            )
        assert response.status_code == 200
        # Verificar que filters se pasaron a hybrid_search
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["extra_filters"].institutions == ["BANCO_CENTRAL_CHILE"]
        assert call_kwargs["extra_filters"].tags == ["DECISION_POLITICA"]
        assert call_kwargs["extra_filters"].min_importance == 0.5


@pytest.mark.integration
class TestImagesEndpoint:
    def test_404_when_chunk_not_found(self) -> None:
        from banks_rag.interface.api import AppState

        deps = AppState()
        deps.repo = type("FakeRepo", (), {
            "get_chunk_image": staticmethod(lambda cid: None),
        })()
        from banks_rag.interface.api.main import create_app
        app = create_app(deps=deps)
        client = TestClient(app)
        response = client.get("/v1/images/nonexistent")
        assert response.status_code == 404

    def test_404_when_chunk_has_no_image(self) -> None:
        from banks_rag.interface.api import AppState
        deps = AppState()
        deps.repo = type("FakeRepo", (), {
            "get_chunk_image": staticmethod(
                lambda cid: {"chunk_id": cid, "image_path": None,
                             "kind": "TEXT", "visual_caption": None}
            ),
        })()
        from banks_rag.interface.api.main import create_app
        app = create_app(deps=deps)
        client = TestClient(app)
        response = client.get("/v1/images/text-chunk")
        assert response.status_code == 404
        assert "no tiene imagen" in response.json()["detail"]

    def test_403_when_path_outside_images_dir(self, tmp_path) -> None:
        from banks_rag.interface.api import AppState
        # image_path apuntando a /etc/passwd → fuera de IMAGES_DIR
        deps = AppState()
        deps.repo = type("FakeRepo", (), {
            "get_chunk_image": staticmethod(
                lambda cid: {"chunk_id": cid, "image_path": "/etc/passwd",
                             "kind": "VISUAL", "visual_caption": "evil"}
            ),
        })()
        from banks_rag.interface.api.main import create_app
        app = create_app(deps=deps)
        client = TestClient(app)
        response = client.get("/v1/images/evil-chunk")
        assert response.status_code == 403


@pytest.mark.integration
class TestAuth:
    def test_no_auth_when_no_keys_configured(self) -> None:
        client, _ = _build_app()
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_auth_blocks_chat_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from banks_rag.config import reset_settings

        monkeypatch.setenv("BANKS_API_KEYS", "secret123")
        reset_settings()
        client, _ = _build_app()
        try:
            response = client.post("/v1/chat", json={"message": "hi"})
            assert response.status_code == 401
        finally:
            reset_settings()

    def test_auth_passes_with_valid_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from banks_rag.config import reset_settings

        monkeypatch.setenv("BANKS_API_KEYS", "secret123")
        reset_settings()
        client, _ = _build_app()
        try:
            response = client.post(
                "/v1/chat",
                json={"message": "hi"},
                headers={"X-API-Key": "secret123"},
            )
            assert response.status_code == 200
        finally:
            reset_settings()

    def test_health_always_public(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from banks_rag.config import reset_settings

        monkeypatch.setenv("BANKS_API_KEYS", "secret123")
        reset_settings()
        client, _ = _build_app()
        try:
            response = client.get("/healthz")
            assert response.status_code == 200
        finally:
            reset_settings()
