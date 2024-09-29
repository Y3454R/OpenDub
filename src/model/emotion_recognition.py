# emotion_recognition.py
from speechbrain.pretrained import EncoderClassifier
import torchaudio

def detect_emotion(audio_file):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
    signal, fs = torchaudio.load(audio_file)
    emotion = classifier.classify_batch(signal)
    return emotion

