"""Efecto de voz "JARVIS / Stark Industries" por DSP — sin modelos.

Replica el tratamiento que recomiendan los tutoriales (ecualización + modulación
de tono/flanger + reverberación) sobre una voz base (SAPI del SO), usando solo
``numpy`` (ya es dependencia del proyecto). Cero descargas, peso ~0.

Cadena: band-pass (timbre de comunicador) → flanger (modulación metálica) →
reverb/eco corto (sensación de "IA en una sala") → normalización.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

_DTYPE_BY_WIDTH = {1: np.int8, 2: np.int16, 4: np.int32}


@dataclass(frozen=True)
class JarvisFx:
    """Parámetros del efecto (ajustables; viven en voices.py)."""

    band_low_hz: float = 280.0    # EQ: corta graves (timbre "comunicador")
    band_high_hz: float = 3600.0  # EQ: corta agudos extremos
    flanger_depth_ms: float = 2.2  # profundidad del delay modulado
    flanger_rate_hz: float = 0.35  # velocidad del LFO (lento = elegante)
    flanger_mix: float = 0.45      # mezcla seco/efecto
    reverb_ms: float = 60.0        # eco corto
    reverb_decay: float = 0.30
    output_peak: float = 0.95      # normalización (evita clipping)


def _bandpass(x: np.ndarray, fr: int, low: float, high: float) -> np.ndarray:
    """Band-pass por FFT (numpy puro, sin scipy)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fr)
    spec[(freqs < low) | (freqs > high)] = 0.0
    return np.fft.irfft(spec, n=len(x))


def _flanger(x: np.ndarray, fr: int, depth_ms: float, rate_hz: float, mix: float) -> np.ndarray:
    """Flanger: delay corto modulado por un LFO, mezclado con la señal seca."""
    n = len(x)
    t = np.arange(n)
    lfo = (1.0 - np.cos(2.0 * np.pi * rate_hz * t / fr)) / 2.0   # 0..1
    delay = lfo * (depth_ms / 1000.0 * fr)
    src = t - delay
    i0 = np.clip(np.floor(src).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    frac = src - np.floor(src)
    delayed = x[i0] * (1.0 - frac) + x[i1] * frac
    return (1.0 - mix) * x + mix * delayed


def _reverb(x: np.ndarray, fr: int, ms: float, decay: float) -> np.ndarray:
    """Eco corto con un par de taps (reverb barata)."""
    out = x.copy()
    for k in (1, 2):
        d = int(fr * ms * k / 1000.0)
        if 0 < d < len(x):
            out[d:] += x[:-d] * (decay ** k)
    return out


def apply_jarvis_effect(wav_bytes: bytes, fx: JarvisFx | None = None) -> bytes:
    """Aplica el efecto JARVIS a un WAV (bytes) y devuelve WAV mono (bytes)."""
    fx = fx or JarvisFx()
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_ch, width, fr, n_frames = (
            w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes(),
        )
        raw = w.readframes(n_frames)

    dtype = _DTYPE_BY_WIDTH.get(width, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)   # a mono
    if audio.size == 0:
        return wav_bytes
    maxv = float(np.iinfo(dtype).max)
    audio /= maxv

    audio = _bandpass(audio, fr, fx.band_low_hz, fx.band_high_hz)
    audio = _flanger(audio, fr, fx.flanger_depth_ms, fx.flanger_rate_hz, fx.flanger_mix)
    audio = _reverb(audio, fr, fx.reverb_ms, fx.reverb_decay)

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * fx.output_peak
    out = (audio * maxv).astype(dtype)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(fr)
        w.writeframes(out.tobytes())
    return buf.getvalue()
