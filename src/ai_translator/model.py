import torch
import yaml
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
logging.disable(logging.WARNING)


_config_path = Path(__file__).resolve().parent.parent.parent / "config.yml" ## 1

with open(_config_path) as _f:
    config = yaml.safe_load(_f)

MODEL_NAME: str       = config["model"]["name"]
DEFAULT_SRC_LANG: str = config["model"]["src_lang"]
USE_FAST: bool        = config["model"]["use_fast_tokenizer"]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Pytorch version: {torch.__version__}")


_tokenizer = None
_model     = None


def get_model():
    """
    Return (model, tokenizer), downloading and loading on the first call only.

    Subsequent calls return the already-loaded objects instantly (singleton).
    Unit tests patch this function; integration tests call it for real.
    """
    global _model, _tokenizer

    if _model is None:
        print(f"\n{config['messages']['loading']}")
        print(f"\n{config['messages']['waiting']}\n")
        print(f"Using device:    {device}")
        print(f"PyTorch version: {torch.__version__}")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            src_lang=DEFAULT_SRC_LANG,
            use_fast=USE_FAST,
        )

        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        _model.eval()

        print("Model loaded successfully")
        print(f"Parameters : {sum(p.numel() for p in _model.parameters()) / 1e6:.1f}M")
        print(f"dtype      : {_model.dtype}")

    return _model, _tokenizer
