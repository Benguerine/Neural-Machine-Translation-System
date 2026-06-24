"""
scripts/push_to_hub.py
======================
Upload model + tokenizer to Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py \
        --repo  your-username/nmt-nllb200 \
        --token hf_xxxxxxxxxxxxxxxxxxxx
"""

import argparse
from ai_translator.model import model, tokenizer


def push(repo_id: str, token: str, private: bool = False) -> None:
    print(f"Pushing tokenizer → {repo_id}")
    tokenizer.push_to_hub(repo_id, token=token, private=private)

    print(f"📤 Pushing model     → {repo_id}")
    model.push_to_hub(repo_id, token=token, private=private)

    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",    required=True, help="e.g. username/nmt-nllb200")
    parser.add_argument("--token",   required=True, help="HF write token (hf_...)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    push(args.repo, args.token, args.private)
