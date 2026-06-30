"""
app.py — Gradio entry point for the Neural Machine Translation System.

Launches a multi-tab web UI that exposes:
  - Single text translation
  - Speech-to-text + translation
  - Batch translation
  - BLEU score evaluation

Server settings are read from environment variables (SERVER_HOST, SERVER_PORT)
so the app can be configured without touching this file.
"""

import os
import warnings

import gradio as gr

from src.ai_translator.languages import SUPPORTED_LANGUAGES
from src.ai_translator.translate import translate_text, batch_translate
from src.ai_translator.speech import speech_to_text
from src.ai_translator.evaluate import calculate_bleu

# Targeted warning suppression (never suppress everything globally)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Constants
MAX_CHARS = 2_000          # hard cap on input length to protect the model
MAX_BATCH_LINES = 50       # max sentences in batch mode


# Gradio wrapper functions

def gradio_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Validate input then call translate_text; return a user-friendly error on failure."""
    if not text or not text.strip():
        return "Enter text to translate."
    if len(text) > MAX_CHARS:
        return f"Input too long ({len(text):,} characters). Keep it under {MAX_CHARS:,} characters."
    try:
        return translate_text(text, src_lang, tgt_lang)
    except Exception as exc:
        return f"Translation failed: {exc}"


def gradio_speech_translate(audio, src_lang: str, tgt_lang: str):
    """Transcribe audio then translate; returns (transcription, translation)."""
    if audio is None:
        return "No audio provided.", ""

    transcribed = speech_to_text(audio, src_lang)
    if transcribed.startswith(("Transcription failed", "No audio", "❌", "⚠️")):
        return transcribed, ""

    try:
        translation = translate_text(transcribed, src_lang, tgt_lang)
    except Exception as exc:
        return transcribed, f"Translation failed: {exc}"

    return transcribed, translation


def gradio_batch_translate(texts: str, src_lang: str, tgt_lang: str) -> str:
    """Validate batch input then translate; enforces line-count limit."""
    if not texts or not texts.strip():
        return "Enter at least one sentence."

    lines = [line for line in texts.splitlines() if line.strip()]
    if len(lines) > MAX_BATCH_LINES:
        return (
            f"Too many lines ({len(lines)}). "
            f"Submit at most {MAX_BATCH_LINES} sentences at a time."
        )
    try:
        return batch_translate(lines, src_lang, tgt_lang)
    except Exception as exc:
        return f"Batch translation failed: {exc}"


def gradio_bleu(reference: str, hypothesis: str) -> str:
    """Calculate BLEU score with basic validation."""
    if not reference or not reference.strip():
        return "Provide a reference translation."
    if not hypothesis or not hypothesis.strip():
        return "Provide a hypothesis translation."
    try:
        _, report = calculate_bleu(reference, hypothesis)
        return report
    except Exception as exc:
        return f"BLEU calculation failed: {exc}"


# Theme & styling
#
# Palette is drawn from editorial/linguistic reference tooling rather than
# generic SaaS blue: an ink/paper base with a single deep-teal accent for
# interactive elements and a restrained rust tone reserved for primary calls
# to action and score emphasis.

INK = "#13161A"
PAPER = "#FAFAF8"
TEAL = "#1F4E4E"
TEAL_DARK = "#163A3A"
RUST = "#B5491E"
SLATE = "#5B5F66"
HAIRLINE = "#E2E0DA"

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=PAPER,
    body_background_fill_dark=INK,
    background_fill_primary=PAPER,
    border_color_primary=HAIRLINE,
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color=HAIRLINE,
    block_radius="4px",
    block_shadow="none",
    block_title_text_weight="600",
    block_label_text_color=SLATE,
    block_label_text_size="13px",
    button_primary_background_fill=TEAL,
    button_primary_background_fill_hover=TEAL_DARK,
    button_primary_text_color="#FFFFFF",
    button_border_width="1px",
    input_border_color=HAIRLINE,
    input_radius="3px",
)

CUSTOM_CSS = f"""
    :root {{
        --ink: {INK};
        --paper: {PAPER};
        --teal: {TEAL};
        --rust: {RUST};
        --slate: {SLATE};
        --hairline: {HAIRLINE};
    }}

    .gradio-container {{
        max-width: 1180px !important;
        font-feature-settings: "tnum";
    }}

    /* Masthead */
    #masthead {{
        border-bottom: 1px solid var(--hairline);
        padding-bottom: 18px;
        margin-bottom: 4px;
    }}
    #masthead h1 {{
        font-size: 26px;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--ink);
        margin: 0 0 4px 0;
    }}
    #masthead p {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        letter-spacing: 0.02em;
        color: var(--slate);
        text-transform: uppercase;
        margin: 0;
    }}

    /* Tabs styled as a language toolbar, not rounded pills */
    .tab-nav {{
        border-bottom: 1px solid var(--hairline) !important;
        gap: 4px;
    }}
    .tab-nav button {{
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--slate) !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 4px !important;
        margin-right: 22px !important;
        transition: border-color 0.15s ease, color 0.15s ease;
    }}
    .tab-nav button.selected {{
        color: var(--ink) !important;
        border-bottom: 2px solid var(--teal) !important;
    }}

    /* Section helper text */
    .section-note {{
        font-size: 13px;
        color: var(--slate);
        line-height: 1.5;
        margin-bottom: 6px;
    }}

    /* Primary action buttons: quiet, not oversized */
    button.lg.primary {{
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }}

    /* Footer */
    #footer-rule {{
        border-top: 1px solid var(--hairline);
        margin-top: 28px;
        padding-top: 14px;
    }}
    #footer-rule p {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px;
        color: var(--slate);
        text-align: center;
        letter-spacing: 0.02em;
        margin: 0;
    }}

    /* BLEU guide rendered as a quiet legend, not a bulleted callout */
    .bleu-legend {{
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: var(--slate);
        border: 1px solid var(--hairline);
        border-radius: 3px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }}
    .bleu-legend span.tag {{
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 1px;
        margin-right: 6px;
    }}
"""


# Gradio UI

with gr.Blocks(
    title="Neural Machine Translation",
    theme=theme,
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(
        """
        <div id="masthead">
            <h1>Neural Machine Translation</h1>
            <p>NLLB-200 distilled-600M · 200+ languages · PyTorch 2.x</p>
        </div>
        """
    )

    with gr.Tabs():

        # Text Translation
        with gr.Tab("Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    src_lang_text = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="English",
                        label="Source language", interactive=True,
                    )
                    input_text = gr.Textbox(
                        lines=10,
                        placeholder=f"Enter text to translate (max {MAX_CHARS:,} characters)",
                        label="Input text",
                        show_copy_button=True,
                    )
                with gr.Column(scale=1):
                    tgt_lang_text = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="French",
                        label="Target language", interactive=True,
                    )
                    output_text = gr.Textbox(
                        lines=10, label="Translation", show_copy_button=True,
                    )

            translate_btn = gr.Button("Translate", variant="primary", size="lg")
            translate_btn.click(
                fn=gradio_translate,
                inputs=[input_text, src_lang_text, tgt_lang_text],
                outputs=output_text,
            )

            gr.Examples(
                label="Examples",
                examples=[
                    ["Hello, how are you today?", "English", "French"],
                    ["Machine learning is fascinating.", "English", "Spanish"],
                    ["I love traveling around the world.", "English", "Arabic"],
                    ["The weather is beautiful.", "English", "German"],
                ],
                inputs=[input_text, src_lang_text, tgt_lang_text],
            )

        # Speech Translation
        with gr.Tab("Speech"):
            gr.HTML(
                "<p class='section-note'>Record or upload audio. It will be "
                "transcribed, then translated into the target language.</p>"
            )
            with gr.Row():
                with gr.Column():
                    src_lang_speech = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="English",
                        label="Speech language",
                    )
                    tgt_lang_speech = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="French",
                        label="Target language",
                    )
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Audio input",
                    )

            with gr.Row():
                transcribed_output = gr.Textbox(label="Transcription", show_copy_button=True)
                speech_translation_output = gr.Textbox(label="Translation", show_copy_button=True)

            speech_translate_btn = gr.Button(
                "Transcribe & translate", variant="primary", size="lg"
            )
            speech_translate_btn.click(
                fn=gradio_speech_translate,
                inputs=[audio_input, src_lang_speech, tgt_lang_speech],
                outputs=[transcribed_output, speech_translation_output],
            )

        # Batch Translation
        with gr.Tab("Batch"):
            gr.HTML(
                f"<p class='section-note'>Enter one sentence per line — "
                f"up to {MAX_BATCH_LINES} lines per request.</p>"
            )
            with gr.Row():
                src_lang_batch = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES, value="English", label="Source language",
                )
                tgt_lang_batch = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES, value="Spanish", label="Target language",
                )

            batch_input = gr.Textbox(
                lines=10,
                placeholder="Sentence 1\nSentence 2\nSentence 3",
                label="Input sentences",
            )
            batch_output = gr.Textbox(
                lines=10, label="Batch translations", show_copy_button=True,
            )

            batch_btn = gr.Button("Translate batch", variant="primary", size="lg")
            batch_btn.click(
                fn=gradio_batch_translate,
                inputs=[batch_input, src_lang_batch, tgt_lang_batch],
                outputs=batch_output,
            )

            gr.Examples(
                label="Example",
                examples=[
                    ["Hello, how are you?\nWhat is your name?\nI love coding.", "English", "French"],
                ],
                inputs=[batch_input, src_lang_batch, tgt_lang_batch],
            )

        # BLEU Evaluation
        with gr.Tab("Evaluation"):
            gr.HTML(
                """
                <p class='section-note'>Compare a reference translation against
                model output using the BLEU metric.</p>
                <div class="bleu-legend">
                    <div><span class="tag" style="background:#1F4E4E"></span>60–100 Excellent</div>
                    <div><span class="tag" style="background:#5B8A72"></span>40–60 Good</div>
                    <div><span class="tag" style="background:#C99A3B"></span>20–40 Fair</div>
                    <div><span class="tag" style="background:#B5491E"></span>0–20 Poor</div>
                </div>
                """
            )

            with gr.Row():
                reference_text = gr.Textbox(
                    lines=5,
                    placeholder="Enter reference (ground truth) translation…",
                    label="Reference translation",
                )
                hypothesis_text = gr.Textbox(
                    lines=5,
                    placeholder="Enter model-generated translation…",
                    label="Model translation",
                )
            bleu_output = gr.Textbox(lines=15, label="BLEU score report", show_copy_button=True)

            bleu_btn = gr.Button("Calculate BLEU", variant="primary", size="lg")
            bleu_btn.click(
                fn=gradio_bleu,
                inputs=[reference_text, hypothesis_text],
                outputs=bleu_output,
            )

            gr.Examples(
                label="Examples",
                examples=[
                    ["Le chat est sur le tapis", "Le chat est sur le tapis"],
                    ["Bonjour, comment allez-vous?", "Bonjour, comment vas-tu?"],
                ],
                inputs=[reference_text, hypothesis_text],
            )

    gr.HTML(
        """
        <div id="footer-rule">
            <p>MODEL — FACEBOOK/NLLB-200-DISTILLED-600M &nbsp;·&nbsp; FRAMEWORK — PYTORCH 2.X + TRANSFORMERS &nbsp;·&nbsp; UI — GRADIO</p>
        </div>
        """
    )


# Launch
if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False)