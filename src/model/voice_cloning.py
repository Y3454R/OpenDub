# src/model/voice_cloning.py

import os
from TTS.api import TTS

# Define paths for input audio and output audio
input_audio_path = "src/audio/input_audio.wav"
output_audio_path = "src/output/output_audio.wav"

def voice_cloning(input_audio, text, output_audio):
    """
    Clones the voice from the input audio and synthesizes the translated text in that voice.
    
    Args:
        input_audio (str): Path to the input audio file for voice cloning.
        text (str): Translated text to synthesize.
        output_audio (str): Path where the generated audio will be saved.
    """
    try:
        # Initialize the TTS model for voice cloning (Coqui TTS)
        model_name = "tts_models/en/vctk/vits"  # You can explore other models from Coqui TTS
        tts = TTS(model_name)

        # Extract speaker embedding from input audio
        speaker_embedding = tts.get_speaker_embedding(input_audio)
        
        # Generate new audio with the translated text and cloned voice
        cloned_audio = tts.tts(text, speaker_embeddings=speaker_embedding)

        # Save the cloned voice audio output to the output_audio path
        tts.save_wav(cloned_audio, output_audio)
        print(f"Generated cloned audio saved at: {output_audio}")
        
    except Exception as e:
        print(f"Error during voice cloning: {str(e)}")

if __name__ == "__main__":
    # Check if input audio exists
    if os.path.exists(input_audio_path):
        print(f"Found input audio: {input_audio_path}")

        # Sample translated text from your translation system (this should be dynamic)
        translated_text = "Hello, how are you?"

        # Perform voice cloning using the input audio and translated text
        voice_cloning(input_audio_path, translated_text, output_audio_path)
    else:
        print(f"Error: Input audio file not found at {input_audio_path}")

