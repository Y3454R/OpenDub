"""
Text-to-Speech module using gTTS for OpenDub project.
Converts text to speech and saves as audio file.
"""

from gtts import gTTS

def text_to_speech(text, lang="en", output_file="output/output_audio.wav"):
    """
    Convert text to speech and save as audio.

    Args:
        text (str): Text to convert.
        lang (str): Language code. Default "en".
        output_file (str): Output path. Default "output/output_audio.wav".
    """
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_file)

