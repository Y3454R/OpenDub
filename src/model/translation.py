"""
Subtitle translation module using MarianMT for OpenDub project.
Detects source language and translates subtitles to target language.
"""

# translation.py
from transformers import MarianMTModel, MarianTokenizer
import pysrt
from langdetect import detect

def translate_subtitles(subtitle_file, target_lang='en'):
    """
    Translate subtitles to target language.

    Args:
        subtitle_file (str): Path to subtitle file.
        target_lang (str): Target language code. Default 'en'.

    Returns:
        list: Translated subtitle objects.
    """
    # Read the subtitle file
    subs = pysrt.open(subtitle_file)

    # Detect the source language from the first few subtitles
    sample_text = " ".join([sub.text for sub in subs[:10]])
    source_lang = detect(sample_text)

    # Map detected language to MarianMT model code if necessary
    lang_map = {
        # Chinese variations
        'zh-cn': 'zh', 'zh-tw': 'zh', 'zh-hk': 'zh', 'zh-sg': 'zh',
        'zh-hans': 'zh', 'zh-hant': 'zh', 'cmn': 'zh', 'yue': 'zh',
        
        # Norwegian variations
        'nb': 'no', 'nn': 'no', 'no-bok': 'no', 'no-nyn': 'no',
        
        # Serbo-Croatian variations
        'hr': 'sh', 'bs': 'sh', 'sr': 'sh', 'cnr': 'sh',  # Montenegrin
        
        # Malay variations
        'ms': 'msa', 'id': 'msa', 'zsm': 'msa',  # Standard Malay
        
        # Filipino/Tagalog
        'tl': 'fil',
        
        # Arabic variations
        'ar-001': 'ar', 'arz': 'ar',  # Egyptian Arabic
        'apc': 'ar',  # North Levantine Arabic
        'acm': 'ar',  # Mesopotamian Arabic
        'ajp': 'ar',  # South Levantine Arabic
        
        # Persian variations
        'fa-AF': 'fa', 'prs': 'fa',  # Dari
        'pes': 'fa',  # Iranian Persian
        
        # Azerbaijani
        'azj': 'az', 'azb': 'az',
        
        # Kurdish
        'ckb': 'ku', 'kmr': 'ku',
        
        # Pashto
        'pbt': 'ps', 'pbu': 'ps',
        
        # Quechua
        'quz': 'qu', 'qvc': 'qu',
        
        # Swahili
        'swh': 'sw', 'swc': 'sw',
        
        # Variations that might be detected differently
        'jap': 'ja', 'kor': 'ko', 'deu': 'de', 'fra': 'fr', 'spa': 'es',
        
        # Old or alternative codes
        'iw': 'he', 'in': 'id', 'ji': 'yi', 'jw': 'jv',
        'mo': 'ro',  # Moldavian to Romanian
        'scc': 'sr', 'scr': 'hr',  # Old codes for Serbian and Croatian
        
        # Macrolanguages and close varieties
        'zlm': 'ms',  # Malay macrolanguage
        'hbs': 'sh',  # Serbo-Croatian macrolanguage
        'nor': 'no',  # Norwegian macrolanguage
        
        # Indigenous languages of the Americas
        'nah': 'nhn',  # Nahuatl languages
        'grn': 'gn',   # Guarani
        'cre': 'cr',   # Cree
        
        # African languages
        'swa': 'sw',   # Swahili
        'nya': 'ny',   # Chichewa
        'lin': 'ln',   # Lingala
        'wol': 'wo',   # Wolof
        
        # South Asian languages
        'urd': 'ur',   # Urdu
        'hin': 'hi',   # Hindi
        'ben': 'bn',   # Bengali
        'tam': 'ta',   # Tamil
        
        # Southeast Asian languages
        'khm': 'km',   # Khmer
        'lao': 'lo',   # Lao
        'mya': 'my',   # Burmese
        
        # Variations in language names
        'bul': 'bg',   # Bulgarian
        'ces': 'cs',   # Czech
        'ell': 'el',   # Greek
        'eus': 'eu',   # Basque
        'gle': 'ga',   # Irish
        'hye': 'hy',   # Armenian
        'isl': 'is',   # Icelandic
        'kat': 'ka',   # Georgian
        'mlt': 'mt',   # Maltese
        'mri': 'mi',   # Maori
        'slk': 'sk',   # Slovak
        'sqi': 'sq',   # Albanian
    }
    source_lang = lang_map.get(source_lang, source_lang)

    print(f"Detected source language: {source_lang}")

    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated_subs = []
    for sub in subs:
        translated = model.generate(**tokenizer(sub.text, return_tensors="pt", padding=True))
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        sub.text = translated_text
        translated_subs.append(sub)

    return translated_subs

