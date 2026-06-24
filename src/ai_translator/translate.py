import time
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
# NOTE: temperature is intentionally unused now. It only has an effect when
# do_sample=True, and we generate with beam search (do_sample=False) below —
# deterministic, faster, and generally better BLEU for NMT than sampling.
# Kept in config.yml for anyone who wants to experiment with sampling-based
# generation, but it is not applied by default.


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
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                do_sample=False,
            )

        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    except Exception as exc:
        return f"Translation error: {str(exc)}"


def translate_text_timed(
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = MAX_LENGTH,
        num_beams: int  = NUM_BEAMS,
) -> tuple[str, float]:
    """
    Same as translate_text, but also returns wall-clock seconds spent.

    Used by the UI layer to show a meaningful "translated in N.Ns" instead
    of relying on Gradio's generic, stage-less progress timer.
    """
    start = time.perf_counter()
    result = translate_text(text, source_lang, target_lang, max_length, num_beams)
    elapsed = time.perf_counter() - start
    return result, elapsed


def batch_translate(
        texts: list[str],
        source_lang: str,
        target_lang: str,
        separator: str = "\n",
) -> str:
    """Translate a list of texts, returning translations joined by separator."""
    if not texts:
        return ""
    

    try:
        sentences = [s.strip() for s in texts if s.strip()]
        if not sentences:
            return ""
        model, tokenizer = get_model()

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
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                do_sample=False,
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