from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import torchaudio

# Load the model and processor
model_name = "superb/wav2vec2-large-superb-er"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

def detect_emotion(audio_path):
    # Load the audio
    signal, fs = torchaudio.load(audio_path)
    signal = signal.squeeze()

    # Process the audio for the model
    inputs = processor(signal, sampling_rate=fs, return_tensors="pt", padding=True)

    # Perform emotion classification
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted emotion
    predicted_ids = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[predicted_ids]
    return emotion

