# text_to_speech.py
from gtts import gTTS

def text_to_speech(text, lang="en", output_file="output/output_audio.wav"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_file)

