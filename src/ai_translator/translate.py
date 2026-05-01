import torch
import yaml
from pathlib import Path
from .model import get_model, device
from .languages import get_flores_code


_config_path = Path(__file__).resolve().parent.parent.parent / "config.yml"
with open(_config_path) as _f:
    _cfg = yaml.safe_load(_f)["inference"]


MAX_LENGTH:     int = _cfg["max_length"]
NUM_BEAMS:      int = _cfg["num_beams"]
NO_REPEAT_NGRAM: int = _cfg["no_repeat_ngram_size"]
TEMPERATURE:   float = _cfg["temperature"]


def translate_text(
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = MAX_LENGTH,
        num_beams: int  = NUM_BEAMS,
) -> str:
    """Translate a single text from source_lang to target_lang."""

    if not text or not text.strip():
        return ""

    model, tokenizer = get_model()

    try:
        src_code = get_flores_code(source_lang, "eng_Latn")
        tgt_code = get_flores_code(target_lang, "fra_Latn")

        tokenizer.src_lang = src_code

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                temperature=TEMPERATURE,
                do_sample=True,
            )

        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    except Exception as exc:
        return f"Translation error: {str(exc)}"


def batch_translate(
        texts: list[str],
        source_lang: str,
        target_lang: str,
        separator: str = "\n",
) -> str:
    """Translate a list of texts, returning translations joined by separator."""
    if not texts:
        return ""

    # FIX — same lazy-load call as translate_text above
    model, tokenizer = get_model()

    try:
        sentences = [s.strip() for s in texts if s.strip()]
        if not sentences:
            return ""

        src_code = get_flores_code(source_lang, "eng_Latn")
        tgt_code = get_flores_code(target_lang, "fra_Latn")

        tokenizer.src_lang = src_code

        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                temperature=TEMPERATURE,
                do_sample=True,
                early_stopping=True,
            )

        translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return "\n".join(f"{i+1}. {t}" for i, t in enumerate(translations))

    except Exception as exc:
        return f"Batch translation error: {str(exc)}"


if __name__ == "__main__":
    result = translate_text(
        text="Hello, how are you?",
        source_lang="English",
        target_lang="French",
    )
    print(f"Single translation: {result}")

    batch_result = batch_translate(
        texts=["Good morning.", "See you later.", "Thank you!"],
        source_lang="English",
        target_lang="French",
    )
    print(f"Batch translation:\n{batch_result}")