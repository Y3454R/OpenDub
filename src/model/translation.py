# translation.py
from transformers import MarianMTModel, MarianTokenizer
import pysrt

def translate_subtitles(subtitle_file, source_lang='es', target_lang='en'):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Read the subtitle file
    subs = pysrt.open(subtitle_file)

    translated_subs = []
    for sub in subs:
        translated = model.generate(**tokenizer(sub.text, return_tensors="pt", padding=True))
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        sub.text = translated_text
        translated_subs.append(sub)

    return translated_subs

