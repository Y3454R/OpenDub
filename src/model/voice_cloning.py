"""
Voice cloning module using TTS for OpenDub project.
Clones voice from source audio and generates new speech.
"""

from TTS.api import TTS
import numpy as np
import soundfile as sf

def voice_clone(source_wav, target_text, output_path):
    """
    Clone voice and generate new speech.

    Args:
        source_wav (str): Source audio file path.
        target_text (str): Text for new speech.
        output_path (str): Output audio file path.
    """
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
    wav = tts.tts(text=target_text, speaker_wav=source_wav, language="en")

    audio_array = np.array(wav)
    if audio_array.ndim == 1:
        audio_array = np.array([audio_array, audio_array])
    
    audio_array = audio_array.T
    sf.write(output_path, audio_array, tts.synthesizer.output_sample_rate, 'PCM_16')

