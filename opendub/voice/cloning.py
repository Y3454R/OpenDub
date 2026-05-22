"""kNN-VC voice cloning: convert generic MMS-TTS-ben output to the source speaker's voice.

This is the core research contribution of OpenDub. The interface is defined here;
implementation requires a trained kNN-VC model checkpoint.

Reference: kNN-VC (Baas et al., 2023) — https://github.com/bshall/knn-vc
"""
import os
import numpy as np
import soundfile as sf


class VoiceCloner:
    def __init__(self, checkpoint_path: str = None):
        """Load a kNN-VC model.

        checkpoint_path: path to a fine-tuned WavLM + HiFi-GAN checkpoint.
        If None, the module falls back to a no-op (returns audio unchanged)
        so the rest of the pipeline can run without a trained model.
        """
        self._ready = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load(checkpoint_path)

    def _load(self, checkpoint_path: str):
        # TODO: load knnvc model
        # from knnvc import KNN_VC
        # self.model = KNN_VC.from_pretrained(checkpoint_path)
        self._ready = True
        print(f"kNN-VC model loaded from {checkpoint_path}")

    def clone(self, source_audio_path: str, target_audio_path: str, output_path: str) -> str:
        """Convert source_audio to have the voice of target_audio.

        source_audio_path : synthesized Bangla TTS segment
        target_audio_path : English source vocal segment (reference speaker)
        output_path       : where to write the voice-converted output
        Returns output_path.
        """
        if not self._ready:
            # No-op fallback: just copy the TTS output unchanged
            import shutil
            shutil.copy2(source_audio_path, output_path)
            return output_path

        # TODO: implement kNN-VC inference
        # wavs, sr = self.model.convert(source_audio_path, target_audio_path)
        # sf.write(output_path, wavs, sr)
        raise NotImplementedError("kNN-VC inference not yet implemented")

    def clone_all(self, segments: list[dict]) -> None:
        """Clone every segment's bn_audio toward the speaker in en_audio.

        Updates bn_audio in-place (overwrites with cloned version).
        """
        for seg in segments:
            if not seg.get("bn_audio") or not seg.get("en_audio"):
                continue
            cloned_path = seg["bn_audio"].replace("_bn.wav", "_bn_cloned.wav")
            self.clone(seg["bn_audio"], seg["en_audio"], cloned_path)
            seg["bn_audio"] = cloned_path
