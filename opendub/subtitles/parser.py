import re
import os
import pysrt
import librosa
import soundfile as sf


def parse(srt_path: str, audio_path: str, output_dir: str, sample_rate: int = 16000) -> list[dict]:
    """Parse an SRT file and extract per-segment audio slices.

    Returns a list of segment dicts with keys:
        id, start, end, dur, en, en_audio
    """
    os.makedirs(output_dir, exist_ok=True)

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    total_duration = len(audio) / sr
    print(f"Audio duration: {total_duration:.1f}s")

    subs = pysrt.open(srt_path)
    segments = []

    for sub in subs:
        start = sub.start.ordinal / 1000.0
        end   = sub.end.ordinal   / 1000.0
        text  = _clean(sub.text)
        if not text:
            continue

        s = int(start * sr)
        e = min(int(end * sr), len(audio))
        seg_audio = audio[s:e]
        seg_path  = os.path.join(output_dir, f"seg_{sub.index:04d}_en.wav")
        sf.write(seg_path, seg_audio, sr)

        segments.append({
            "id":       sub.index,
            "start":    start,
            "end":      end,
            "dur":      end - start,
            "en":       text,
            "en_audio": seg_path,
        })

    print(f"Parsed {len(segments)} subtitle segments")
    return segments


def _clean(text: str) -> str:
    text = re.sub(r'\{[^}]+\}', '', text)  # {an8} style tags
    text = re.sub(r'<[^>]+>', '', text)     # <i> style tags
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
