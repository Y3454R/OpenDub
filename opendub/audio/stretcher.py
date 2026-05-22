import numpy as np
import librosa
import soundfile as sf

_RATIO_MIN = 0.5
_RATIO_MAX = 2.0

try:
    import pyrubberband as rb
    _USE_RUBBERBAND = True
except ImportError:
    _USE_RUBBERBAND = False


def stretch(audio_path: str, target_duration: float) -> None:
    """Time-stretch audio file in-place to match target_duration (seconds).

    Uses rubberband (pitch-preserving TSM) when available, falls back to
    librosa's phase vocoder. scipy.signal.resample is NOT used — it changes
    pitch by coupling it to duration (tape-speed effect).
    Skips segments whose ratio falls outside [0.5, 2.0].
    """
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) == 0 or target_duration <= 0:
        return

    src_dur = len(y) / sr
    ratio   = src_dur / target_duration  # >1 = BN longer than EN → speed up

    if not (_RATIO_MIN < ratio < _RATIO_MAX):
        return

    if _USE_RUBBERBAND:
        y_stretched = rb.time_stretch(y, sr, ratio).astype(np.float32)
    else:
        # librosa phase vocoder: preserves pitch, requires float32 input
        y_stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=ratio)

    sf.write(audio_path, y_stretched, sr)


def stretch_all(segments: list[dict]) -> None:
    """Stretch every segment's bn_audio in-place to match its original EN duration."""
    for seg in segments:
        if not seg.get("bn_audio") or not seg.get("bn_dur"):
            continue
        stretch(seg["bn_audio"], seg["dur"])
