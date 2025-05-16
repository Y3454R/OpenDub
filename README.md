# OpenDub 🥥

**OpenDub** is a powerful tool for generating translated dialogues from subtitles while preserving the original voice. It utilizes speech recognition, machine translation, and voice cloning technologies to ensure that the translated dialogue stays in sync with the original audio. Perfect for projects in film, media, or any context where translated content needs to maintain the integrity of the speaker's voice.

## Demo

This is a demo version for Spanish (audio + subtitles) to English dubbing generation. (Still not working though)

## Features

- **Automatic Speech Recognition (ASR):** Extracts text from original audio files.
- **Subtitle Translation:** Translates text from one language to another with high accuracy.
- **Voice Cloning:** Synthesizes the translated dialogue using the original speaker's voice.
- **Subtitle Synchronization:** Ensures the translated audio matches the original subtitle timings.

## How It Works

1. **Audio to Text Conversion**: The original audio is processed using Automatic Speech Recognition (ASR) to extract dialogue in the source language.
2. **Subtitle Translation**: The extracted text or subtitle file is translated into the target language using a powerful machine translation engine.
3. **Voice Cloning**: The system clones the speaker's voice using advanced TTS (Text-to-Speech) with voice cloning capabilities.

## How To Run

#### Setup Python environment with pyenv

```bash
pyenv install 3.11.5
pyenv virtualenv 3.11.5 opendub-env
cd /Users/adibasubah/OpenDub
pyenv local opendub-env
```

#### Install Dependencies: In your virtual environment, install the required libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Place Test Files:

Put a Spanish audio file in `src/audio/input_audio.wav`.
Place the matching Spanish subtitle file in `src/subtitles/input_subtitles.srt`.

#### Run the Application: Execute the main script to run the entire pipeline:

```bash
python app.py
```

#### Output: The dubbed audio will be saved in the `src/output/output_audio.wav` file.
