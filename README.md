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

# 🌍 Neural Machine Translation System

### Production-Grade AI Translator Powered by Meta's NLLB-200

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.29-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![BLEU](https://img.shields.io/badge/Evaluation-BLEU_Score-6366F1?style=for-the-badge)]()

<br/>

**Translate text and speech across 100 languages — with BLEU evaluation, batch processing, and a clean web UI.**

[🚀 Live Demo](#demo) · [📖 Documentation](#installation) · [🐛 Report Bug](issues) · [✨ Request Feature](issues)

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Why This Project Matters](#why-this-project-matters)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance & Scalability](#performance--scalability)
- [Challenges & Lessons Learned](#challenges--lessons-learned)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Neural Machine Translation System** is a full-stack AI application that delivers high-quality translation across **100 languages** using Meta's state-of-the-art **NLLB-200-distilled-600M** model. It supports text translation, speech-to-text translation via Google Speech Recognition, batch processing, and automated translation quality evaluation using BLEU scoring — all through an intuitive Gradio web interface.

### The Problem It Solves

Existing translation tools are either expensive APIs with usage limits, or research demos with no production-ready interface. This system bridges that gap: a self-hosted, open-source translation engine that handles real-world multilingual workflows — from single phrases to bulk content pipelines — with built-in quality measurement.

### Target Users

| User | Use Case |
|------|----------|
| **Developers** | Embed translation into apps via the modular Python API |
| **Researchers** | Evaluate NMT models using BLEU scoring |
| **Content Teams** | Batch-translate documents across dozens of languages |
| **Data Scientists** | Build multilingual NLP pipelines |
| **Enterprises** | Self-host translation without vendor lock-in |

---

## Why This Project Matters

> *"Translation is not just a technical problem — it's a bridge between cultures."*

This project demonstrates production-level ML engineering skills across the full stack:

- **Model integration**: Fine-tuned Facebook NLLB-200, one of the most capable open multilingual models available
- **Software engineering**: Clean architecture with separation of concerns, type hints, and comprehensive test coverage
- **MLOps thinking**: Evaluation metrics (BLEU), fallback handling, and graceful error degradation built in from day one
- **User-centered design**: A polished, accessible UI that non-technical users can operate immediately
- **Deployment-ready**: Containerized, environment-configurable, and Hugging Face Spaces compatible

---

## Features

### Core Capabilities
- 🌐 **100-Language Translation** — Powered by FLORES-200 language codes and Meta NLLB-200
- 🎤 **Speech-to-Text Translation** — Record or upload audio; transcribed and translated in one step
- 📦 **Batch Translation** — Translate up to 50 sentences simultaneously with numbered output
- 📊 **BLEU Score Evaluation** — Quantify translation quality against reference translations
- ⚡ **Input Validation** — Character limits, empty input guards, and line-count caps for robustness

### Engineering Highlights
- 🏗️ **Modular Architecture** — Languages, translation, speech, and evaluation are fully decoupled
- 🧪 **23-Test Suite** — Unit tests covering language codes, translation logic, and edge cases
- 🔡 **FLORES-200 Compliance** — All language codes validated against the standard format (`xxx_Xxxx`)
- 🔊 **Smart Speech Fallback** — Languages without Google SR support gracefully fall back to English STT
- 🐳 **Docker-Ready** — Environment variables for host/port; zero config changes needed for containerization

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Web UI (app.py)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Text Translate│ │   Speech     │ │  Batch   │ │   BLEU    │  │
│  │     Tab      │ │ Translate Tab│ │   Tab    │ │   Tab     │  │
│  └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └─────┬─────┘  │
└─────────┼────────────────┼──────────────┼─────────────┼─────────┘
          │                │              │             │
          ▼                ▼              ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Business Logic Layer                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐  │
│  │ translate.py │ │  speech.py   │ │translate │ │evaluate.py│  │
│  │translate_text│ │speech_to_text│ │  _batch  │ │   BLEU    │  │
│  └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └───────────┘  │
└─────────┼────────────────┼──────────────┼─────────────────────-─┘
          │                │              │
          ▼                ▼              │
┌──────────────────┐ ┌──────────────────┐ │
│   HuggingFace    │ │  Google Speech   │ │
│ NLLB-200-600M    │ │  Recognition API │ │
│ (Local Inference)│ │  (External API)  │ │
└──────────────────┘ └──────────────────┘ │
          │                               │
          ▼                               │
┌─────────────────────────────────────────┘
│              languages.py               │
│  LANGUAGE_CODES  │  SPEECH_LANG_CODES   │
│  (FLORES-200)    │  (BCP-47 / SR)       │
└─────────────────────────────────────────┘
```

### Translation Pipeline

```
User Input (text/audio)
        │
        ▼
  Input Validation
  (length, empty, line count)
        │
        ▼
  [If audio] Google SR → Transcription
        │
        ▼
  Language Code Resolution
  (language name → FLORES-200 code)
        │
        ▼
  NLLB-200 Tokenizer
  (source language tokenization)
        │
        ▼
  NLLB-200 Model Inference
  (forced BOS token = target language)
        │
        ▼
  Decoded Output Text
        │
        ▼
  Return to UI / Caller
```

### Model Details

| Property | Value |
|----------|-------|
| Model | `facebook/nllb-200-distilled-600M` |
| Architecture | Encoder-Decoder Transformer |
| Parameters | ~600M |
| Languages | 200 (FLORES-200 standard) |
| Inference | Local (no external API calls for translation) |
| Framework | HuggingFace Transformers + PyTorch 2.x |

---

## Project Structure

```
ai-translator/
│
├── app.py                          # Gradio entry point; UI layout & wrappers
│
├── src/
│   └── ai_translator/
│       ├── __init__.py
│       ├── languages.py            # FLORES-200 & BCP-47 code maps, helper functions
│       ├── translate.py            # Core translation logic (NLLB-200)
│       ├── speech.py               # Speech-to-text via Google SR
│       └── evaluate.py             # BLEU score calculation
│
├── tests/
│   ├── test_languages.py           # 14 unit tests: codes, helpers, edge cases
│   └── test_translate.py           # 9 unit tests: translation logic & batch
│
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
├── .env.example                    # Environment variable template
└── README.md
```

---

## Tech Stack

### Core ML / AI
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![NLLB-200](https://img.shields.io/badge/Meta_NLLB--200-0082FB?style=flat-square&logo=meta&logoColor=white)

### Backend & API
![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-Google_SR-4285F4?style=flat-square&logo=google&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-BLEU_Scoring-154F3C?style=flat-square)

### Frontend / UI
![Gradio](https://img.shields.io/badge/Gradio_5.29-FF7C00?style=flat-square&logo=gradio&logoColor=white)

### DevOps & Tooling
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace_Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## Installation

### Prerequisites

- Python 3.12+
- pip or uv
- ~2GB disk space (model weights)
- CUDA-capable GPU (optional, CPU inference supported)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-translator.git
cd ai-translator
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

```env
# .env
SERVER_HOST=0.0.0.0
SERVER_PORT=7860
```

### 5. Run the Application

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Usage

### Text Translation

```python
from src.ai_translator.translate import translate_text

result = translate_text("Hello, how are you?", src_lang="English", tgt_lang="French")
print(result)  # "Bonjour, comment allez-vous ?"
```

### Batch Translation

```python
from src.ai_translator.translate import batch_translate

texts = "Hello\nGood morning\nHow are you?"
result = batch_translate(texts, src_lang="English", tgt_lang="Arabic")
print(result)
# 1. مرحبا
# 2. صباح الخير
# 3. كيف حالك
```

### Speech-to-Text + Translation

```python
from src.ai_translator.speech import speech_to_text
from src.ai_translator.translate import translate_text

transcription = speech_to_text("audio.wav", language="English")
translation   = translate_text(transcription, "English", "Spanish")
```

### BLEU Evaluation

```python
from src.ai_translator.evaluate import calculate_bleu

score, report = calculate_bleu(
    reference="Le chat est sur le tapis",
    hypothesis="Le chat est sur le tapis"
)
print(report)
```

### Language Utilities

```python
from src.ai_translator.languages import get_flores_code, get_speech_code, SPEECH_LANGUAGES

# FLORES-200 code for NLLB model
get_flores_code("Japanese")   # "jpn_Jpan"
get_flores_code("Klingon")    # "eng_Latn"  ← fallback

# BCP-47 code for Google SR
get_speech_code("Arabic")     # "ar-SA"
get_speech_code("Tibetan")    # "en-US"  ← fallback (no SR support)

# Languages with confirmed speech recognition support
print(SPEECH_LANGUAGES)       # ['English', 'French', 'Spanish', ...]
```

---

## API Reference

### `translate_text(text, src_lang, tgt_lang) → str`

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text to translate |
| `src_lang` | `str` | Source language name (e.g. `"English"`) |
| `tgt_lang` | `str` | Target language name (e.g. `"French"`) |

### `batch_translate(texts, src_lang, tgt_lang) → str`

Translates newline-separated sentences. Returns numbered output lines.

### `speech_to_text(audio_file, language) → str`

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_file` | `str` | Path to audio file (WAV/MP3/etc.) |
| `language` | `str` | Source language name; falls back to English SR if unsupported |

### `calculate_bleu(reference, hypothesis) → tuple[float, str]`

Returns `(score, formatted_report)`.

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/ai_translator --cov-report=term-missing

# Run specific test class
pytest tests/test_languages.py::TestGetSpeechCode -v
```

### Test Coverage

| Module | Tests | Coverage Areas |
|--------|-------|----------------|
| `languages.py` | 14 | FLORES format, speech codes, fallbacks, duplicates |
| `translate.py` | 9 | String output, empty input, unknown languages, batch |
| **Total** | **23** | |

---

## Deployment

### Docker

```dockerfile
# Build
docker build -t ai-translator .

# Run
docker run -p 7860:7860 \
  -e SERVER_HOST=0.0.0.0 \
  -e SERVER_PORT=7860 \
  ai-translator
```

### Hugging Face Spaces

The `README.md` YAML front matter is pre-configured for Spaces deployment:

```yaml
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
python_version: "3.12"
```

Push to a Hugging Face Space repository — it deploys automatically.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `7860` | Port to listen on |

---

## Performance & Scalability

| Concern | Approach |
|---------|----------|
| **Inference speed** | NLLB-200-distilled-600M is the optimized distilled variant (~2× faster than the 1.3B model) |
| **GPU acceleration** | PyTorch auto-detects CUDA; no code changes needed |
| **Input limits** | Hard cap at 2,000 chars and 50 batch lines to protect model memory |
| **Batch efficiency** | Sentences processed in a single model call where possible |
| **Warning suppression** | Targeted `FutureWarning`/`UserWarning` suppression — never a blanket filter |
| **Horizontal scaling** | Stateless design; multiple instances can run behind a load balancer |

---

## Challenges & Lessons Learned

**Speech language coverage** — Google Speech Recognition supports a different (smaller) subset of languages than NLLB-200. Maintaining a separate `SPEECH_LANG_CODES` dict with a tested fallback path was essential to avoid silent failures in the UI.

**FLORES-200 code validation** — Language codes follow a strict `xxx_Xxxx` format. Enforcing this with regex in tests caught several typos early and made the codebase more trustworthy.

**Gradio dropdown scoping** — All translation dropdowns correctly use the full 100-language `SUPPORTED_LANGUAGES` list, while the speech input dropdown is scoped to `SPEECH_LANGUAGES` only — a subtle but important UX distinction that required understanding the data flow end-to-end.

**Model loading strategy** — Transformer models should be loaded once at startup, not per-request. Structuring the code to support this was key to acceptable response times.

---

## Future Improvements

- [ ] **Streaming translation** — Token-by-token output for long texts using Gradio's streaming API
- [ ] **Model selection** — Toggle between NLLB-200-600M and NLLB-200-1.3B in the UI
- [ ] **Document translation** — PDF and DOCX file upload support
- [ ] **Translation memory** — Cache repeated phrases to reduce model calls
- [ ] **REST API** — FastAPI wrapper alongside the Gradio UI for programmatic access
- [ ] **Confidence scores** — Surface model uncertainty alongside translations
- [ ] **CI/CD pipeline** — GitHub Actions for automated testing on every push
- [ ] **Language auto-detection** — Remove the requirement to specify source language manually

---

## Demo

> 📸 *Screenshots coming soon*

| Tab | Description |
|-----|-------------|
| 💬 Text Translation | Translate up to 2,000 characters between any 100 languages |
| 🎤 Speech Translation | Record or upload audio for transcription + translation |
| 📦 Batch Translation | Translate up to 50 sentences at once |
| 📊 BLEU Evaluation | Score translation quality with a reference comparison |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
# Fork → clone → branch
git checkout -b feature/your-feature-name

# Make changes, add tests, then:
pytest tests/ -v

# Submit a pull request
```

Please ensure all 23 existing tests pass and add tests for any new functionality.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Contact

**Built with ❤️ using open-source AI**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/yourusername)

---

<div align="center">

*If this project helped you, please consider giving it a ⭐ on GitHub*

</div>
