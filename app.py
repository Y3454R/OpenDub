# app.py
from src.model.emotion_recognition import detect_emotion
from src.model.translation import translate_subtitles
from src.model.text_to_speech import text_to_speech
import pysrt

def main():
    # Paths to input files
    audio_file = 'src/audio/input_audio.wav'
    subtitle_file = 'src/subtitles/input_subtitles.srt'

    # Step 1: Detect emotions from the input audio
    emotion = detect_emotion(audio_file)
    print(f"Detected Emotion: {emotion}")

    # Step 2: Translate the subtitles from Spanish to English
    translated_subs = translate_subtitles(subtitle_file, source_lang='es', target_lang='en')
    for sub in translated_subs:
        print(sub.text)

    # Step 3: Convert the translated subtitles to speech
    full_text = " ".join([sub.text for sub in translated_subs])
    text_to_speech(full_text, lang="en", output_file="src/output/output_audio.wav")
    print("Dubbed audio generated in src/output/output_audio.wav")

if __name__ == '__main__':
    main()

