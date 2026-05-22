import os
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

_MODEL_ID = "facebook/mms-tts-ben"


class Synthesizer:
    def __init__(self):
        print(f"Loading Bangla TTS model ({_MODEL_ID})...")
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self.model     = VitsModel.from_pretrained(_MODEL_ID).to("cpu")
        self.sample_rate = self.model.config.sampling_rate
        print(f"TTS sample rate: {self.sample_rate}Hz")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (waveform_float32, sample_rate)."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        wav = output.squeeze().cpu().numpy().astype(np.float32)
        return wav, self.sample_rate

    def synthesize_all(self, segments: list[dict], output_dir: str) -> None:
        """Synthesize each segment and write WAV files.

        Adds 'bn_audio' and 'bn_dur' keys to each segment dict in-place.
        """
        os.makedirs(output_dir, exist_ok=True)

        for seg in segments:
            out_path = os.path.join(output_dir, f"seg_{seg['id']:04d}_bn.wav")

            if seg["dur"] < 0.3 or not seg.get("bn", "").strip():
                silence = np.zeros(int(seg["dur"] * self.sample_rate), dtype=np.float32)
                sf.write(out_path, silence, self.sample_rate)
                seg["bn_audio"] = out_path
                seg["bn_dur"]   = seg["dur"]
                continue

            try:
                wav, sr = self.synthesize(seg["bn"])
                sf.write(out_path, wav, sr)
                seg["bn_audio"] = out_path
                seg["bn_dur"]   = len(wav) / sr
                print(
                    f"  Seg {seg['id']:3d}: EN={seg['dur']:.2f}s → BN={seg['bn_dur']:.2f}s "
                    f"(diff={seg['bn_dur'] - seg['dur']:+.2f}s)"
                )
            except Exception as exc:
                print(f"  Seg {seg['id']:3d}: TTS failed ({exc}), using silence")
                silence = np.zeros(int(seg["dur"] * self.sample_rate), dtype=np.float32)
                sf.write(out_path, silence, self.sample_rate)
                seg["bn_audio"] = out_path
                seg["bn_dur"]   = seg["dur"]
