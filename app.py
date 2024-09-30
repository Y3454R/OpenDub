"""
OpenDub: Audio dubbing application.

This script performs:
1. Emotion detection from input audio
2. Subtitle translation
3. Voice cloning for translated text

Usage: Run directly with default inputs or modify file paths in main().

Dependencies: src.model.emotion_recognition, src.model.translation, src.model.voice_cloning, os
"""

import os
from src.model.emotion_recognition import detect_emotion
from src.model.translation import translate_subtitles
from src.model.voice_cloning import voice_clone

def main():
    """
    Run the OpenDub application.

    Steps:
    1. Set up file paths
    2. Detect emotion
    3. Translate subtitles
    4. Generate dubbed audio

    Raises:
        FileNotFoundError: If input files are missing.
        RuntimeError: If processing fails.
    """
    try:
        audio_file = 'src/audio/input_audio.wav'
        subtitle_file = 'src/subtitles/input_subtitles.srt'
        output_file = 'src/output/output_audio.wav'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print("Detecting emotion...")
        emotion = detect_emotion(audio_file)
        print(f"Detected Emotion: {emotion}")

        print("Translating subtitles...")
        translated_subs = translate_subtitles(subtitle_file, target_lang='en')
        for sub in translated_subs:
            print(sub.text)

        print("Generating dubbed audio...")
        full_text = " ".join([sub.text for sub in translated_subs])
        voice_clone(audio_file, full_text, output_file)
        print(f"Dubbed audio saved to {output_file}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
    except RuntimeError as e:
        print(f"Error: Processing failed. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
