"""
Voice cloning module using TTS for OpenDub project.
Clones voice from source audio and generates new speech.
"""

import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.models.xtts import XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig


def voice_clone(source_wav, target_text, output_path):
    """
    Clone voice and generate new speech.

    Args:
        source_wav (str): Source audio file path.
        target_text (str): Text for new speech.
        output_path (str): Output audio file path.
    """
    # allow XttsConfig for torch.load unpickling
    torch.serialization.add_safe_globals(
        [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
    )
    # instantiate multilingual TTS model
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=False,
    )
    # synthesize waveform
    wav = tts.tts(text=target_text, speaker_wav=source_wav, language="en")
    audio_array = np.array(wav)
    if audio_array.ndim == 1:
        audio_array = audio_array[:, None]
    # write output file
    sf.write(output_path, audio_array, tts.synthesizer.output_sample_rate)
