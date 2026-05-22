import gc
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_MODEL_ID = "facebook/nllb-200-distilled-600M"
_TGT_LANG = "ben_Beng"


class Translator:
    def __init__(self):
        print(f"Loading translation model ({_MODEL_ID})...")
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16
        ).to("cpu")
        self._tgt_id = self.tokenizer.convert_tokens_to_ids(_TGT_LANG)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self._tgt_id,
                num_beams=5,
                max_length=256,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def translate_all(self, segments: list[dict]) -> None:
        """Add 'bn' key to each segment dict in-place."""
        total = len(segments)
        for i, seg in enumerate(segments):
            seg["bn"] = self.translate(seg["en"])
            print(f"  [{i+1}/{total}] {seg['en'][:45]}")
            print(f"           → {seg['bn'][:45]}")

    def unload(self):
        del self.model
        del self.tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
