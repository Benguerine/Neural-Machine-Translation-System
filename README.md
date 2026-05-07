---
title: Neural Machine Translation AI Translator
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
python_version: "3.12"
pinned: false
---

<div align="center">

# Neural Machine Translation System

### Self-hosted AI translator across 100 languages — text, speech, batch, and BLEU evaluation

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.29-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## Overview

A production-grade Neural Machine Translation application powered by **Meta's NLLB-200-distilled-600M** model. Designed as a self-hosted alternative to proprietary translation APIs — no usage limits, no vendor lock-in, full control over the inference pipeline.

Supports text translation, speech-to-text translation, batch processing, and BLEU score evaluation through a clean Gradio web interface.

---

## Features

| Capability | Detail |
|---|---|
| 🌐 **100-Language Translation** | FLORES-200 language codes, Meta NLLB-200 model |
| 🎤 **Speech Translation** | Record or upload audio → transcription + translation |
| 📦 **Batch Processing** | Up to 50 sentences per request, numbered output |
| 📊 **BLEU Evaluation** | Score translation quality against a reference |
| 🏗️ **Modular Architecture** | Decoupled language, translation, speech, and eval layers |
| 🧪 **23-Test Suite** | Unit tests covering codes, translation logic, and edge cases |
| 🔊 **Speech Fallback** | Languages without Google SR support fall back gracefully |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Gradio Web UI (app.py)                   │
│      Text Tab │ Speech Tab │ Batch Tab │ BLEU Tab            │
└────────┬──────────────┬───────────────┬──────────────────────┘
         │              │               │
         ▼              ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│   translate.py │ speech.py │ batch_translate │ evaluate.py  │
└────────┬──────────────┬───────────────────────────────────────┘
         │              │
         ▼              ▼
┌─────────────────┐  ┌──────────────────┐
│ NLLB-200-600M   │  │ Google Speech    │
│ Local Inference │  │ Recognition API  │
└────────┬────────┘  └──────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                       languages.py                           │
│   LANGUAGE_CODES (FLORES-200) │ SPEECH_LANG_CODES (BCP-47)  │
└──────────────────────────────────────────────────────────────┘
```

**Translation pipeline:** Input validation → (optional) Google SR transcription → FLORES-200 code resolution → NLLB-200 tokenization → model inference → decoded output.

**Key design decisions:**
- `SPEECH_LANGUAGES` is scoped separately from `SUPPORTED_LANGUAGES` — the speech tab only exposes languages with confirmed Google SR support, preventing silent STT fallbacks in the UI
- All FLORES-200 codes are regex-validated at import time (`xxx_Xxxx` format) — malformed codes are caught before reaching the model
- The model is loaded once at startup, not per-request

### Model

| Property | Value |
|---|---|
| Model | `facebook/nllb-200-distilled-600M` |
| Architecture | Encoder-Decoder Transformer |
| Parameters | ~600M |
| Inference | Local — no external API |
| Framework | HuggingFace Transformers + PyTorch 2.x |

---

## Project Structure

```
ai-translator/
├── app.py                        # Gradio UI entry point
├── src/
│   └── ai_translator/
│       ├── languages.py          # FLORES-200 & BCP-47 maps, get_flores_code, get_speech_code
│       ├── translate.py          # Core NLLB-200 inference
│       ├── speech.py             # Google SR wrapper with language fallback
│       └── evaluate.py           # BLEU scoring
├── tests/
│   ├── test_languages.py         # 14 tests — codes, helpers, edge cases
│   └── test_translate.py         # 9 tests — translation logic, batch, empty inputs
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Tech Stack

**ML/AI:** PyTorch 2.x · HuggingFace Transformers · Meta NLLB-200-distilled-600M · NLTK (BLEU)  
**Backend:** Python 3.12 · SpeechRecognition · Google SR API  
**Frontend:** Gradio 5.29  
**DevOps:** Docker · Hugging Face Spaces · pytest

---

## Installation

**Requirements:** Python 3.12+, ~2 GB disk (model weights), GPU optional

```bash
git clone https://github.com/yourusername/ai-translator.git
cd ai-translator

python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env   # Set SERVER_HOST and SERVER_PORT if needed
python app.py          # Open http://localhost:7860
```

---

## Usage

```python
from src.ai_translator.translate import translate_text, batch_translate
from src.ai_translator.speech import speech_to_text
from src.ai_translator.evaluate import calculate_bleu
from src.ai_translator.languages import get_flores_code, get_speech_code

# Single translation
translate_text("Hello, how are you?", src_lang="English", tgt_lang="French")
# → "Bonjour, comment allez-vous ?"

# Batch translation
batch_translate("Hello\nGood morning\nHow are you?", "English", "Arabic")
# → "1. مرحبا\n2. صباح الخير\n3. كيف حالك"

# Speech → translation
transcription = speech_to_text("audio.wav", language="English")
translation   = translate_text(transcription, "English", "Spanish")

# BLEU evaluation
score, report = calculate_bleu(reference="Le chat...", hypothesis="Le chat...")

# Language code helpers
get_flores_code("Japanese")   # "jpn_Jpan"
get_flores_code("Klingon")    # "eng_Latn"  (fallback)
get_speech_code("Arabic")     # "ar-SA"
get_speech_code("Tibetan")    # "en-US"     (no SR support → fallback)
```

---

## Testing

```bash
pytest tests/ -v                                                    # Full suite
pytest tests/ --cov=src/ai_translator --cov-report=term-missing    # With coverage
pytest tests/test_languages.py::TestGetSpeechCode -v               # Single class
```

| Module | Tests | What's Covered |
|---|---|---|
| `languages.py` | 14 | FLORES format, speech codes, fallbacks, duplicate detection |
| `translate.py` | 9 | Output types, empty input, unknown languages, batch logic |

---

## Deployment

**Docker**

```bash
docker build -t ai-translator .
docker run -p 7860:7860 -e SERVER_HOST=0.0.0.0 -e SERVER_PORT=7860 ai-translator
```

**Hugging Face Spaces** — push to a Space repository; the YAML front matter in this file handles configuration automatically.

| Variable | Default | Description |
|---|---|---|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `7860` | Server port |

---

## Performance & Scalability

- **Distilled model** — NLLB-200-distilled-600M is ~2× faster than the 1.3B variant with comparable quality
- **GPU auto-detection** — PyTorch uses CUDA when available; no code changes required
- **Input guards** — 2,000-char limit and 50-line batch cap protect against OOM at inference time
- **Stateless design** — horizontally scalable; multiple instances run independently behind a load balancer

---

## Future Improvements

- [ ] FastAPI REST layer alongside the Gradio UI
- [ ] Streaming output for long translations via Gradio's streaming API
- [ ] Language auto-detection to remove the source language requirement
- [ ] Translation memory / caching for repeated phrases
- [ ] GitHub Actions CI/CD for automated test runs on push
- [ ] PDF and DOCX document upload support

---

## Contributing

Open an issue before submitting a PR. All contributions must pass the existing 23-test suite.

```bash
git checkout -b feature/your-feature
pytest tests/ -v
# then open a pull request
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with open-source AI · [GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile) · [Hugging Face](https://huggingface.co/yourusername)

</div>
