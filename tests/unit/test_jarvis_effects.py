"""Tests del efecto DSP JARVIS y la selección de voz (sin SAPI ni modelos)."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from jarvis_news.effects import JarvisFx, apply_jarvis_effect
from jarvis_news.voices import VOICE_HINTS, pick_voice_id


def _make_wav(freqs_hz, *, fr=22050, secs=0.5) -> bytes:
    t = np.arange(int(fr * secs)) / fr
    sig = sum(0.4 * np.sin(2 * np.pi * f * t) for f in freqs_hz)
    pcm = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _band_energy(wav_bytes, lo, hi) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        fr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
    spec = np.abs(np.fft.rfft(a))
    fq = np.fft.rfftfreq(len(a), 1 / fr)
    return float(spec[(fq >= lo) & (fq <= hi)].sum())


@pytest.mark.unit
class TestJarvisEffect:
    def test_output_is_valid_mono_wav(self) -> None:
        out = apply_jarvis_effect(_make_wav([440]))
        with wave.open(io.BytesIO(out), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getnframes() > 0

    def test_bandpass_attenuates_highs(self) -> None:
        # 6 kHz está fuera del band-pass (high≈3600) → debe quedar muy atenuado.
        wav = _make_wav([440, 6000])
        out = apply_jarvis_effect(wav)
        assert _band_energy(out, 5900, 6100) < _band_energy(out, 400, 480) * 0.1

    def test_bandpass_attenuates_lows(self) -> None:
        # 80 Hz está por debajo del band-pass (low≈280) → atenuado.
        wav = _make_wav([80, 1000])
        out = apply_jarvis_effect(wav)
        assert _band_energy(out, 60, 100) < _band_energy(out, 900, 1100) * 0.2

    def test_custom_params(self) -> None:
        fx = JarvisFx(band_high_hz=2000)
        out = apply_jarvis_effect(_make_wav([1000, 3000]), fx)
        assert _band_energy(out, 2900, 3100) < _band_energy(out, 900, 1100) * 0.2

    def test_empty_wav_is_returned_as_is(self) -> None:
        empty = _make_wav([440], secs=0)
        assert apply_jarvis_effect(empty) == empty


@pytest.mark.unit
class TestVoiceSelection:
    class _V:
        def __init__(self, vid, name):
            self.id, self.name = vid, name

    def test_picks_en_gb_for_english(self) -> None:
        voices = [self._V("es", "Microsoft Helena"), self._V("TTS_MS_EN-GB_GEORGE", "Microsoft George")]
        assert pick_voice_id(voices, "en") == "TTS_MS_EN-GB_GEORGE"

    def test_picks_spanish_for_es(self) -> None:
        voices = [self._V("en", "Microsoft David"), self._V("ES-ES_HELENA", "Microsoft Helena - Spanish")]
        assert pick_voice_id(voices, "es") == "ES-ES_HELENA"

    def test_none_when_no_match(self) -> None:
        voices = [self._V("xx", "Klingon Voice")]
        assert pick_voice_id(voices, "es") is None

    def test_hints_cover_both_langs(self) -> None:
        assert "en" in VOICE_HINTS and "es" in VOICE_HINTS
