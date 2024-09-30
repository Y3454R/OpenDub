"""
Emotion recognition module using Wav2Vec2 for OpenDub project.
Detects emotion from audio files using a pre-trained model.
"""

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torchaudio

# Load the pre-trained model and feature extractor
model_name = "superb/wav2vec2-large-superb-er"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

def detect_emotion(audio_path):
    """
    Detect emotion in audio file.

    Args:
        audio_path (str): Path to audio file.

    Returns:
        str: Detected emotion label.
    """
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample the audio to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert stereo audio to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Process the audio for the model input
        inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        # Perform emotion classification
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get the predicted emotion
        predicted_ids = torch.argmax(logits, dim=-1).item()
        emotion = model.config.id2label[predicted_ids]
        
        return emotion

    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing audio or running the model: {str(e)}")

# Test the function if this script is run directly
if __name__ == "__main__":
    test_audio_path = "path/to/your/test/audio.wav"
    try:
        detected_emotion = detect_emotion(test_audio_path)
        print(f"Detected emotion: {detected_emotion}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")