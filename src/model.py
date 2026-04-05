import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



_config_path = Path(__file__).resolve().parent.parent / "config.yml" ## 1

with open(_config_path) as _f:
    config = yaml.safe_load(_f)

MODEL_NAME: str       = config["model"]["name"]
DEFAULT_SRC_LANG: str = config["model"]["src_lang"]
USE_FAST: bool        = config["model"]["use_fast_tokenizer"]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Pytorch version: {torch.__version__}")


print(f"\n{config['messages']['loading']}")
print(f"\n{config['messages']['waiting']}\n")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang=DEFAULT_SRC_LANG,
    use_fast=USE_FAST,
)


model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)


model.eval()

print("Model loaded successfully")