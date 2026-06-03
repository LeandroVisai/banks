"""Tests del analizador de noticias (carga, priorización, map-reduce)."""

from __future__ import annotations

import json

import pytest

import jarvis_news.report as report
from banks_rag.domain.agent import GenerationResult
from jarvis_news.report import (
    _parse_metric,
    generate_news_report,
    load_news,
    prioritize,
)


@pytest.fixture
def news_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "NEWS_SCRAPING_DIR", tmp_path)
    return tmp_path


class _MockLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, messages, *, tools=None, **kwargs):
        self.calls.append(kwargs)
        return GenerationResult(text=f"out{len(self.calls)}", n_tokens=5)


@pytest.mark.unit
class TestMetrics:
    @pytest.mark.parametrize("raw,expected", [
        ("271,02K", 271_020),
        ("$ 18,90M", 18_900_000),
        ("9,83M €", 9_830_000),
        ("84,0", 84),
        ("$ 396,0", 396),
        (None, 0),
        ("", 0),
    ])
    def test_parse_metric(self, raw, expected) -> None:
        assert _parse_metric(raw) == pytest.approx(expected)


@pytest.mark.unit
class TestPrioritize:
    def test_domain_topic_outranks_higher_reach(self) -> None:
        news = [
            {"topic": "Otros", "audience": "500,00K", "title": "a"},
            {"topic": "Banco Central", "audience": "100,00K", "title": "b"},
            {"topic": "Banco Central", "audience": "50,00K", "title": "c"},
        ]
        top = prioritize(news, top_n=2)
        assert [n["title"] for n in top] == ["b", "c"]  # dominio primero

    def test_top_n_caps(self) -> None:
        news = [{"topic": "x", "audience": "1K"} for _ in range(10)]
        assert len(prioritize(news, top_n=3)) == 3


@pytest.mark.unit
class TestLoadAndReport:
    def _write(self, d, items) -> None:
        (d / "noticias.json").write_text(json.dumps(items), encoding="utf-8")

    def test_load_latest(self, news_dir) -> None:
        self._write(news_dir, [{"title": "x", "text": "y"}])
        path, news = load_news()
        assert len(news) == 1 and path.name == "noticias.json"

    def test_load_no_files_raises(self, news_dir) -> None:
        with pytest.raises(FileNotFoundError):
            load_news()

    def test_load_bad_json_raises(self, news_dir) -> None:
        (news_dir / "x.json").write_text('{"sin": "lista"}', encoding="utf-8")
        with pytest.raises(ValueError, match="lista de noticias"):
            load_news()

    @pytest.mark.asyncio
    async def test_report_is_mapreduce(self, news_dir) -> None:
        items = [
            {"topic": "Banco Central", "audience": f"{i + 1},00K",
             "title": f"n{i}", "text": "cuerpo"}
            for i in range(20)
        ]
        self._write(news_dir, items)
        llm = _MockLLM()
        r = await generate_news_report(llm, top_n=16, batch_size=8)
        assert r["n_total"] == 20 and r["n_used"] == 16
        assert len(llm.calls) == 3   # 2 lotes map + 1 reduce
        assert r["report"] == "out3"
        assert r["source_file"] == "noticias.json"
