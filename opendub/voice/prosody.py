"""Prosody transfer using WORLD vocoder (pyworld).

Copies F0 contour and energy envelope from the English source segment onto the
Bangla synthesized segment, improving naturalness and speaker similarity.

Reference: WORLD vocoder — https://github.com/mmorise/World
           pyworld       — https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
"""
import os
import numpy as np
import librosa
import soundfile as sf


class ProsodyTransfer:
    def __init__(self):
        try:
            import pyworld  # noqa: F401
            self._available = True
        except ImportError:
            print("pyworld not installed — prosody transfer disabled")
            self._available = False

    def transfer(self, source_audio_path: str, target_audio_path: str, output_path: str) -> str:
        """Transfer F0 + energy from source (EN) onto target (BN TTS/cloned).

        source_audio_path : English vocal segment (F0 donor)
        target_audio_path : Bangla audio to receive the prosody
        output_path       : where to write the result
        Returns output_path.
        """
        if not self._available:
            import shutil
            shutil.copy2(target_audio_path, output_path)
            return output_path

        import pyworld as pw

        src, src_sr = librosa.load(source_audio_path, sr=None, mono=True)
        tgt, tgt_sr = librosa.load(target_audio_path, sr=None, mono=True)

        src = src.astype(np.float64)
        tgt = tgt.astype(np.float64)

        # Extract source F0 and aperiodicity
        src_f0, src_t = pw.harvest(src, src_sr)
        src_f0 = pw.stonemask(src, src_f0, src_t, src_sr)

        # Extract target spectral envelope
        tgt_f0, tgt_t = pw.harvest(tgt, tgt_sr)
        tgt_f0 = pw.stonemask(tgt, tgt_f0, tgt_t, tgt_sr)
        tgt_sp = pw.cheaptrick(tgt, tgt_f0, tgt_t, tgt_sr)
        tgt_ap = pw.d4c(tgt, tgt_f0, tgt_t, tgt_sr)

        # Resample source F0 to target frame count
        if len(src_f0) != len(tgt_f0):
            src_f0_resampled = np.interp(
                np.linspace(0, 1, len(tgt_f0)),
                np.linspace(0, 1, len(src_f0)),
                src_f0,
            )
        else:
            src_f0_resampled = src_f0

        # Preserve voicing decision from target (0 = unvoiced)
        transferred_f0 = np.where(tgt_f0 > 0, src_f0_resampled, 0.0)

        # Synthesize with transferred F0, target timbre
        out = pw.synthesize(transferred_f0, tgt_sp, tgt_ap, tgt_sr)
        out = out.astype(np.float32)

        # Match RMS energy of source
        src_rms = np.sqrt(np.mean(src.astype(np.float32) ** 2)) + 1e-8
        tgt_rms = np.sqrt(np.mean(out ** 2)) + 1e-8
        out = out * (src_rms / tgt_rms)

        sf.write(output_path, out, tgt_sr)
        return output_path

    def transfer_all(self, segments: list[dict]) -> None:
        """Transfer prosody for every segment. Updates bn_audio in-place."""
        for seg in segments:
            if not seg.get("bn_audio") or not seg.get("en_audio"):
                continue
            out_path = seg["bn_audio"].replace(".wav", "_prosody.wav")
            self.transfer(seg["en_audio"], seg["bn_audio"], out_path)
            seg["bn_audio"] = out_path
