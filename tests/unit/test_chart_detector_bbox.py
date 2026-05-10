"""Tests Fase 2.b — bbox detection + cropping en chart_detector.

Las funciones bajo test son puras (no dependen de PyMuPDF). Se construyen
los rectángulos como tuplas ``(x0, y0, x1, y1)`` y se valida la lógica
de clustering, filtrado por área y merge.
"""

from __future__ import annotations

import pytest

from banks_rag.domain.documents import BoundingBox
from banks_rag.infrastructure.extractors.chart_detector import (
    BBOX_PADDING_PT,
    CLUSTER_DISTANCE_PT,
    MAX_REGION_AREA_RATIO,
    MIN_CLUSTER_AREA_RATIO,
    MIN_IMAGE_AREA_RATIO,
    MIN_SHAPES_PER_CLUSTER,
    VISUAL_MIN_FILLED_SHAPES,
    _cluster_rects,
    _merge_overlapping_regions,
    _pad_bbox,
    _rects_close,
    _union_of_rects,
    detect_chart_regions,
    find_caption_block_for_region,
    find_text_blocks_around_region,
)


@pytest.mark.unit
class TestRectsClose:
    def test_overlapping(self) -> None:
        assert _rects_close((0, 0, 100, 50), (50, 25, 200, 100), distance=0)

    def test_touching(self) -> None:
        assert _rects_close((0, 0, 100, 50), (100, 0, 200, 50), distance=0)

    def test_close_within_distance(self) -> None:
        assert _rects_close((0, 0, 100, 50), (110, 0, 200, 50), distance=20)

    def test_too_far(self) -> None:
        assert not _rects_close((0, 0, 100, 50), (200, 0, 300, 50), distance=20)

    def test_close_diagonal(self) -> None:
        # Casi en esquinas opuestas pero la distancia X es 0 e Y es 5
        assert _rects_close((0, 0, 100, 100), (100, 105, 200, 200), distance=10)


@pytest.mark.unit
class TestUnionOfRects:
    def test_two_rects(self) -> None:
        u = _union_of_rects([(0, 0, 100, 50), (50, 25, 200, 100)])
        assert u == (0, 0, 200, 100)

    def test_single(self) -> None:
        u = _union_of_rects([(10, 20, 30, 40)])
        assert u == (10, 20, 30, 40)

    def test_three_disjoint(self) -> None:
        u = _union_of_rects([(0, 0, 10, 10), (50, 50, 60, 60), (100, 100, 110, 110)])
        assert u == (0, 0, 110, 110)


@pytest.mark.unit
class TestClusterRects:
    def test_empty(self) -> None:
        assert _cluster_rects([], distance=10) == []

    def test_two_distinct_clusters(self) -> None:
        rects = [
            (0, 0, 50, 50),       # cluster A
            (10, 10, 60, 60),     # solapa con A
            (200, 200, 250, 250), # cluster B
            (210, 210, 260, 260), # solapa con B
        ]
        clusters = _cluster_rects(rects, distance=10)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [2, 2]

    def test_chain_through_proximity(self) -> None:
        # Tres rects en cadena: A cerca de B, B cerca de C, A lejos de C.
        # Deben quedar todos en el mismo cluster (transitividad).
        rects = [
            (0, 0, 10, 10),
            (15, 0, 25, 10),
            (30, 0, 40, 10),
        ]
        clusters = _cluster_rects(rects, distance=10)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_far_apart_separate(self) -> None:
        rects = [(0, 0, 10, 10), (1000, 1000, 1010, 1010)]
        clusters = _cluster_rects(rects, distance=10)
        assert len(clusters) == 2


@pytest.mark.unit
class TestMergeOverlapping:
    def test_no_overlap_no_merge(self) -> None:
        rects = [(0, 0, 10, 10), (100, 100, 110, 110)]
        merged = _merge_overlapping_regions(rects, distance=0)
        assert len(merged) == 2

    def test_overlap_merges(self) -> None:
        rects = [(0, 0, 50, 50), (25, 25, 100, 100)]
        merged = _merge_overlapping_regions(rects, distance=0)
        assert len(merged) == 1
        assert merged[0] == (0, 0, 100, 100)


@pytest.mark.unit
class TestPadBbox:
    def test_pads_evenly(self) -> None:
        out = _pad_bbox((100, 100, 200, 200), padding=10, page_w=500, page_h=500)
        assert out == (90, 90, 210, 210)

    def test_clamps_to_page(self) -> None:
        out = _pad_bbox((5, 5, 495, 495), padding=10, page_w=500, page_h=500)
        assert out == (0, 0, 500, 500)


@pytest.mark.unit
class TestDetectChartRegions:
    PAGE_W, PAGE_H = 612.0, 792.0  # carta US ≈ 595×842 A4
    PAGE_AREA = PAGE_W * PAGE_H

    def _detect(self, **kw):
        return detect_chart_regions(
            embedded_image_rects=kw.get("imgs", []),
            filled_drawing_rects=kw.get("drawings", []),
            page_width=self.PAGE_W,
            page_height=self.PAGE_H,
        )

    def test_empty_page(self) -> None:
        assert self._detect() == []

    def test_invalid_page_area(self) -> None:
        result = detect_chart_regions(
            embedded_image_rects=[(0, 0, 100, 100)],
            filled_drawing_rects=[],
            page_width=0,
            page_height=100,
        )
        assert result == []

    def test_small_image_filtered_as_logo(self) -> None:
        # Imagen de 50×50 = 2500 pt², página 484k pt² → 0.5% < 5% threshold.
        regions = self._detect(imgs=[(10, 10, 60, 60)])
        assert regions == []

    def test_large_embedded_image_kept(self) -> None:
        # Imagen de 300×400 = 120k pt² ≈ 25% página → entra.
        regions = self._detect(imgs=[(50, 50, 350, 450)])
        assert len(regions) == 1
        bbox = regions[0]
        # Padding aplicado pero clamp a la página
        assert bbox.x0 == max(0.0, 50 - BBOX_PADDING_PT)
        assert bbox.x1 == min(self.PAGE_W, 350 + BBOX_PADDING_PT)

    def test_ignores_drawings_below_threshold(self) -> None:
        # Solo 7 formas — bajo VISUAL_MIN_FILLED_SHAPES (8) → no se evalúa
        drawings = [(100 + i * 5, 100, 105 + i * 5, 200) for i in range(7)]
        regions = self._detect(drawings=drawings)
        assert regions == []

    def test_clusters_drawings_into_chart(self) -> None:
        # 12 barras verticales, todas cerca → 1 cluster grande
        drawings = [(100 + i * 20, 200, 115 + i * 20, 400) for i in range(12)]
        regions = self._detect(drawings=drawings)
        assert len(regions) == 1
        bbox = regions[0]
        # Bbox cubre todas las barras + padding
        assert bbox.x0 < 100
        assert bbox.x1 > 100 + 11 * 20
        assert bbox.y0 < 200

    def test_two_separate_charts_two_regions(self) -> None:
        # Chart A: 8 formas en (50-200, 100-300)
        chart_a = [(50 + i * 18, 100, 65 + i * 18, 300) for i in range(8)]
        # Chart B: 8 formas en (350-500, 500-700) — bien lejos
        chart_b = [(350 + i * 18, 500, 365 + i * 18, 700) for i in range(8)]
        regions = self._detect(drawings=chart_a + chart_b)
        assert len(regions) == 2
        # Ordenadas por x0 para comparar
        regions_sorted = sorted(regions, key=lambda b: b.x0)
        assert regions_sorted[0].x1 < regions_sorted[1].x0

    def test_full_page_drawing_filtered(self) -> None:
        # 10 drawings que cubren toda la página → ratio > 95% → se descarta
        drawings = [(0, 0 + i * 10, self.PAGE_W, 10 + i * 10) for i in range(10)]
        # Forzamos cluster muy grande
        regions = self._detect(drawings=drawings)
        # Puede o no ser filtrado dependiendo del clustering, pero si pasa
        # debe tener área < MAX_REGION_AREA_RATIO * page_area
        for r in regions:
            assert r.area / self.PAGE_AREA <= MAX_REGION_AREA_RATIO

    def test_two_chart_backgrounds_split_correctly(self) -> None:
        # Caso real JPM Comodities: 2 charts side-by-side cuyos rects de fondo
        # están a ~30 pt de distancia. Sin background-anchor, se mergeaban en 1.
        # Cada background es bastante grande → debe ser tomado como anchor
        # y cada uno debe quedar como región separada.
        bg_left = (60, 170, 280, 325)    # 220×155 ≈ 7%
        bg_right = (310, 170, 530, 325)  # gap = 30 pt → claramente separados
        bars_left = [(70 + i * 15, 200, 80 + i * 15, 280) for i in range(8)]
        bars_right = [(320 + i * 15, 200, 330 + i * 15, 280) for i in range(8)]
        regions = self._detect(drawings=[bg_left, bg_right, *bars_left, *bars_right])
        assert len(regions) == 2, f"Esperaba 2 charts separados, got {len(regions)}"
        sorted_r = sorted(regions, key=lambda b: b.x0)
        # Después de padding de 6pt c/lado, todavía no se solapan
        assert sorted_r[0].x1 < sorted_r[1].x0, (
            f"Charts solapan tras padding: {sorted_r[0]} vs {sorted_r[1]}"
        )

    def test_overlapping_image_and_drawing_merge(self) -> None:
        # Imagen grande + cluster vectorial solapados → 1 sola región
        img = (100, 100, 400, 500)  # ~25% area
        drawings = [(150 + i * 10, 200, 160 + i * 10, 400) for i in range(10)]
        regions = self._detect(imgs=[img], drawings=drawings)
        assert len(regions) == 1


@pytest.mark.unit
class TestFindCaptionBlockForRegion:
    """find_caption_block_for_region: localiza la caption más cercana al chart."""

    def test_no_blocks(self) -> None:
        cap, bbox = find_caption_block_for_region([], (100, 200, 300, 400))
        assert cap is None and bbox is None

    def test_no_caption_match(self) -> None:
        blocks = [(100, 100, 300, 150, "Texto sin caption")]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert cap is None and bbox is None

    def test_caption_above_chart(self) -> None:
        # Chart en y=200..400, caption arriba en y=170..195
        blocks = [(100, 170, 300, 195, "Figure 3: Inflation expectations")]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert cap is not None and "Figure 3" in cap
        assert bbox == (100, 170, 300, 195)

    def test_caption_below_chart(self) -> None:
        # Chart en y=200..400, caption abajo en y=405..425
        blocks = [(100, 405, 300, 425, "Gráfico 5: TPM")]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert cap is not None and "Gráfico 5" in cap
        assert bbox == (100, 405, 300, 425)

    def test_picks_closest_when_two(self) -> None:
        blocks = [
            (100, 50, 300, 80, "Figure 1: Far above"),    # v_dist = 120
            (100, 170, 300, 195, "Figure 2: Closer"),     # v_dist = 5
        ]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert "Figure 2" in cap
        assert bbox == (100, 170, 300, 195)

    def test_skips_overlapping_block(self) -> None:
        # Bloque solapa con la región verticalmente → ignorar (no es caption)
        blocks = [(100, 250, 300, 300, "Figure 9: dentro del chart, ignorar")]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert cap is None and bbox is None

    def test_skips_no_horizontal_overlap(self) -> None:
        # Bloque encima pero a un lado (otra columna)
        blocks = [(500, 170, 700, 195, "Figure 7: en otra columna")]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert cap is None and bbox is None

    def test_partial_horizontal_overlap_accepted(self) -> None:
        # Bloque desplazado pero con overlap parcial → acepta
        blocks = [(250, 170, 400, 195, "Figure 11: half-overlap")]
        cap, bbox = find_caption_block_for_region(blocks, (100, 200, 300, 400))
        assert cap is not None and "Figure 11" in cap


@pytest.mark.unit
class TestFindTextBlocksAroundRegion:
    """find_text_blocks_around_region: bloques de título/subtítulo sin regex."""

    REGION = (100, 200, 400, 500)

    def test_empty(self) -> None:
        assert find_text_blocks_around_region([], self.REGION) == []

    def test_block_above_with_overlap(self) -> None:
        blocks = [(120, 170, 380, 195, "Cualquier título aquí")]
        result = find_text_blocks_around_region(blocks, self.REGION)
        assert len(result) == 1
        assert result[0][1] == "Cualquier título aquí"

    def test_returns_sorted_by_proximity(self) -> None:
        blocks = [
            (120, 100, 380, 130, "Far above"),    # v_dist = 70
            (120, 170, 380, 195, "Closer above"), # v_dist = 5
        ]
        result = find_text_blocks_around_region(blocks, self.REGION, max_distance=200)
        # Más cercano primero
        assert "Closer" in result[0][1]
        assert "Far" in result[1][1]

    def test_position_above_only_skips_below(self) -> None:
        blocks = [
            (120, 170, 380, 195, "Above"),
            (120, 510, 380, 540, "Below"),
        ]
        result = find_text_blocks_around_region(blocks, self.REGION, position="above")
        assert len(result) == 1 and "Above" in result[0][1]

    def test_position_below_only_skips_above(self) -> None:
        blocks = [
            (120, 170, 380, 195, "Above"),
            (120, 510, 380, 540, "Below"),
        ]
        result = find_text_blocks_around_region(blocks, self.REGION, position="below")
        assert len(result) == 1 and "Below" in result[0][1]

    def test_position_both(self) -> None:
        blocks = [
            (120, 170, 380, 195, "Above"),
            (120, 510, 380, 540, "Below"),
        ]
        result = find_text_blocks_around_region(blocks, self.REGION, position="both")
        assert len(result) == 2

    def test_too_far_skipped(self) -> None:
        blocks = [(120, 50, 380, 80, "Muy lejos arriba")]
        result = find_text_blocks_around_region(blocks, self.REGION, max_distance=50)
        assert result == []

    def test_no_horizontal_overlap_skipped(self) -> None:
        blocks = [(500, 170, 700, 195, "Lateral")]
        result = find_text_blocks_around_region(blocks, self.REGION)
        assert result == []

    def test_long_paragraph_skipped(self) -> None:
        blocks = [(120, 170, 380, 195, "x" * 500)]
        result = find_text_blocks_around_region(blocks, self.REGION, max_chars=400)
        assert result == []

    def test_empty_text_skipped(self) -> None:
        blocks = [(120, 170, 380, 195, "   ")]
        result = find_text_blocks_around_region(blocks, self.REGION)
        assert result == []


@pytest.mark.unit
class TestConstants:
    """Sanity-checks de las constantes del módulo."""

    def test_thresholds_are_reasonable(self) -> None:
        assert 0 < MIN_IMAGE_AREA_RATIO < MAX_REGION_AREA_RATIO <= 1.0
        assert 0 < MIN_CLUSTER_AREA_RATIO < MAX_REGION_AREA_RATIO
        assert MIN_SHAPES_PER_CLUSTER >= 1
        assert VISUAL_MIN_FILLED_SHAPES >= MIN_SHAPES_PER_CLUSTER
        assert CLUSTER_DISTANCE_PT > 0
        assert BBOX_PADDING_PT >= 0

    def test_bbox_returns_BoundingBox(self) -> None:
        regions = detect_chart_regions(
            embedded_image_rects=[(50, 50, 350, 450)],
            filled_drawing_rects=[],
            page_width=612,
            page_height=792,
        )
        assert all(isinstance(r, BoundingBox) for r in regions)
