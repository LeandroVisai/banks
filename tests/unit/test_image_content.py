"""Tests del builder de contenido multimodal (Fase 5 — fundación de visión)."""

from __future__ import annotations

from pathlib import Path

import pytest

from banks_rag.infrastructure.llm._image_content import build_image_content

# PNG 1x1 transparente (bytes mínimos válidos).
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"
)


@pytest.mark.unit
class TestBuildImageContent:
    def test_no_images_returns_plain_text(self) -> None:
        assert build_image_content("hola", None) == "hola"
        assert build_image_content("hola", []) == "hola"

    def test_nonexistent_paths_omitted_returns_text(self) -> None:
        out = build_image_content("texto", ["/no/existe/x.png", ""])
        assert out == "texto"  # ninguna válida → texto plano (seguro text-only)

    def test_valid_image_produces_blocks(self, tmp_path: Path) -> None:
        img = tmp_path / "grafico.png"
        img.write_bytes(_PNG_1x1)
        out = build_image_content("describe el gráfico", [str(img)])
        assert isinstance(out, list)
        assert out[0] == {"type": "text", "text": "describe el gráfico"}
        assert out[1]["type"] == "image_url"
        assert out[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_mixes_valid_and_invalid(self, tmp_path: Path) -> None:
        img = tmp_path / "g.png"
        img.write_bytes(_PNG_1x1)
        out = build_image_content("x", ["/no/existe.png", str(img)])
        assert isinstance(out, list)
        assert sum(1 for b in out if b["type"] == "image_url") == 1


@pytest.mark.unit
class TestVisionConfigured:
    def test_no_mmproj_is_false(self, tmp_path: Path) -> None:
        from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        eng = LlamaCppEngine(str(gguf))
        assert eng.vision_configured is False

    def test_mmproj_present_is_true(self, tmp_path: Path) -> None:
        from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"x")
        eng = LlamaCppEngine(str(gguf), mmproj_path=str(mmproj))
        assert eng.vision_configured is True
