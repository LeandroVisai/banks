"""Tests del helper de audio de jarvis_news: codificación FLAC + fallback."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from jarvis_news import audio as audio_mod
from jarvis_news.audio import FlacUnavailableError, encode_audio, wav_to_flac


def _make_wav(*, fr: int = 22050, secs: float = 0.5) -> bytes:
    t = np.arange(int(fr * secs)) / fr
    sig = 0.3 * np.sin(2 * np.pi * 180 * t)
    pcm = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.mark.unit
class TestWavToFlac:
    def test_produces_valid_flac(self) -> None:
        pytest.importorskip("soundfile")
        wav = _make_wav()
        flac = wav_to_flac(wav)
        assert flac[:4] == b"fLaC"  # firma del contenedor FLAC

    def test_flac_is_smaller_than_wav(self) -> None:
        pytest.importorskip("soundfile")
        wav = _make_wav(secs=2.0)  # señal tonal → comprime bien
        assert len(wav_to_flac(wav)) < len(wav)


@pytest.mark.unit
class TestEncodeAudio:
    def test_flac_when_available(self) -> None:
        pytest.importorskip("soundfile")
        data, ext = encode_audio(_make_wav(), "flac")
        assert ext == "flac"
        assert data[:4] == b"fLaC"

    def test_wav_passthrough(self) -> None:
        wav = _make_wav()
        data, ext = encode_audio(wav, "wav")
        assert ext == "wav"
        assert data == wav

    def test_falls_back_to_wav_without_soundfile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simula que soundfile no está: encode_audio NO debe romper, cae a WAV.
        def _boom(_wav: bytes) -> bytes:
            raise FlacUnavailableError("sin soundfile")

        monkeypatch.setattr(audio_mod, "wav_to_flac", _boom)
        wav = _make_wav()
        data, ext = encode_audio(wav, "flac")
        assert ext == "wav"
        assert data == wav
