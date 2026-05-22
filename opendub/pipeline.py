import os
import gc
import torch

from .audio      import extractor, separator, stretcher, mixer
from .subtitles  import parser
from .translation import translator as translation_module
from .tts        import synthesizer as tts_module
from .voice      import cloning, prosody


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Pipeline:
    def __init__(
        self,
        voice_cloning: bool = True,
        prosody_transfer: bool = True,
        vc_checkpoint: str = None,
        output_dir: str = "outputs",
    ):
        self.voice_cloning    = voice_cloning
        self.prosody_transfer = prosody_transfer
        self.vc_checkpoint    = vc_checkpoint
        self.output_dir       = output_dir
        self.device           = _detect_device()
        print(f"Device: {self.device}")

    def run(
        self,
        input_path: str,
        output_path: str = "dubbed.mkv",
        start: str = None,
        end: str = None,
        srt_path: str = None,
    ) -> str:
        """Run the full dubbing pipeline.

        input_path  : source video (.mkv, .mp4, …)
        output_path : dubbed video destination
        start / end : optional clip window, e.g. "00:23:00"
        srt_path    : external .srt; if None, extracted from video
        Returns output_path.
        """
        work_dir   = os.path.join(self.output_dir, "segments")
        os.makedirs(work_dir, exist_ok=True)

        # ── 1. Extract audio (+ subtitles if not provided) ──────────────
        print("\n[1/8] Extracting audio...")
        audio_path, extracted_srt = extractor.extract(input_path, work_dir, start, end)
        if srt_path is None:
            srt_path = extracted_srt
        if srt_path is None:
            raise FileNotFoundError(
                "No subtitle stream found in video and no --srt provided."
            )

        # ── 2. Separate vocals / background ─────────────────────────────
        # Always separate so background music is preserved in the final output.
        # vocals_path is also used as the speaker reference for kNN-VC.
        print("\n[2/8] Separating vocals from background (Demucs)...")
        vocals_path, bg_path = separator.separate(audio_path, work_dir)
        speaker_ref = vocals_path if self.voice_cloning else audio_path

        # ── 3. Parse subtitles + slice audio ────────────────────────────
        print("\n[3/8] Parsing subtitles and slicing audio...")
        segments = parser.parse(srt_path, audio_path, work_dir)

        # ── 4. Translate EN → BN ────────────────────────────────────────
        print("\n[4/8] Translating (NLLB-200)...")
        trans = translation_module.Translator()
        trans.translate_all(segments)
        trans.unload()
        del trans
        gc.collect()

        # ── 5. Synthesize Bangla TTS ────────────────────────────────────
        print("\n[5/8] Synthesizing Bangla TTS (MMS-TTS-ben)...")
        synth = tts_module.Synthesizer()
        synth.synthesize_all(segments, work_dir)
        tts_sample_rate = synth.sample_rate
        del synth
        gc.collect()

        # ── 6. Voice cloning ────────────────────────────────────────────
        if self.voice_cloning:
            print("\n[6/8] Applying voice cloning (kNN-VC)...")
            cloner = cloning.VoiceCloner(self.vc_checkpoint)
            cloner.clone_all(segments)
        else:
            print("\n[6/8] Skipping voice cloning")

        # ── 7. Prosody transfer ─────────────────────────────────────────
        if self.prosody_transfer:
            print("\n[7/8] Transferring prosody (pyworld)...")
            pt = prosody.ProsodyTransfer()
            pt.transfer_all(segments)
        else:
            print("\n[7/8] Skipping prosody transfer")

        # ── 8. Time-stretch → assemble → mux ────────────────────────────
        print("\n[8/8] Stretching, assembling, and muxing...")
        stretcher.stretch_all(segments)

        import librosa
        audio, sr = librosa.load(audio_path, sr=tts_sample_rate, mono=True)
        total_duration = len(audio) / tts_sample_rate

        dubbed_wav = os.path.join(self.output_dir, "dubbed", "dubbed_bangla.wav")
        os.makedirs(os.path.dirname(dubbed_wav), exist_ok=True)
        mixer.assemble_audio(segments, total_duration, tts_sample_rate, dubbed_wav)

        mixer.mix_with_video(input_path, dubbed_wav, bg_path, output_path)

        mixer.isochrony_report(segments)
        print(f"\nDone — output: {output_path}")
        return output_path
