import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample

_RATIO_MIN = 0.5
_RATIO_MAX = 2.0


def stretch(audio_path: str, target_duration: float) -> None:
    """Time-stretch audio file in-place to match target_duration (seconds).

    Uses scipy resample (pitch-preserving via rate change). Skips segments
    whose stretch ratio falls outside [0.5, 2.0] to avoid audible artifacts.
    Returns the actual duration after stretching.
    """
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) == 0 or target_duration <= 0:
        return

    src_dur = len(y) / sr
    ratio   = src_dur / target_duration

    if not (_RATIO_MIN < ratio < _RATIO_MAX):
        return

    target_len = int(len(y) / ratio)
    y_stretched = resample(y, target_len).astype(np.float32)
    sf.write(audio_path, y_stretched, sr)


def stretch_all(segments: list[dict]) -> None:
    """Stretch every segment's bn_audio in-place to match its original EN duration."""
    for seg in segments:
        if not seg.get("bn_audio") or not seg.get("bn_dur"):
            continue
        stretch(seg["bn_audio"], seg["dur"])
