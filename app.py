"""
Neural Machine Translation — AI Translator
Gradio app entry point.
"""

import warnings

import gradio as gr

from src.ai_translator.languages import SUPPORTED_LANGUAGES
from src.ai_translator.translate import translate_text, batch_translate
from src.ai_translator.speech    import speech_to_text
from src.ai_translator.evaluate  import calculate_bleu

warnings.filterwarnings("ignore")


# Gradio wrapper functions

def gradio_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Wrapper for single translation."""
    return translate_text(text, src_lang, tgt_lang)


def gradio_speech_translate(audio, src_lang: str, tgt_lang: str):
    """Wrapper for speech-to-text + translation."""
    if audio is None:
        return "⚠️ No audio provided", ""
    transcribed = speech_to_text(audio, src_lang)
    if transcribed.startswith(("❌", "⚠️")):
        return transcribed, ""
    return transcribed, translate_text(transcribed, src_lang, tgt_lang)


def gradio_batch_translate(texts: str, src_lang: str, tgt_lang: str) -> str:
    """Wrapper for batch translation."""
    return batch_translate(texts, src_lang, tgt_lang)


def gradio_bleu(reference: str, hypothesis: str) -> str:
    """Wrapper for BLEU evaluation."""
    if not reference or not hypothesis:
        return "Please provide both reference and hypothesis translations."
    _, report = calculate_bleu(reference, hypothesis)
    return report


# Gradio UI

with gr.Blocks(
    title="🌍 Neural Machine Translation",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 1200px !important; }
    .tab-nav button   { font-size: 16px !important; font-weight: 600 !important; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🌍 Neural Machine Translation System
        ### Powered by Facebook NLLB-200 | 200+ Languages | PyTorch 2.10
        """
    )

    with gr.Tabs():

        # Text Translation
        with gr.Tab("💬 Text Translation"):
            with gr.Row():
                with gr.Column(scale=1):
                    src_lang_text = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="English",
                        label="🌐 Source Language", interactive=True,
                    )
                    input_text = gr.Textbox(
                        lines=10, placeholder="Enter text to translate...",
                        label="📝 Input Text",
                    )
                with gr.Column(scale=1):
                    tgt_lang_text = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="French",
                        label="🌐 Target Language", interactive=True,
                    )
                    output_text = gr.Textbox(
                        lines=10, label="✨ Translation",
                    )

            translate_btn = gr.Button("🚀 Translate", variant="primary", size="lg")
            translate_btn.click(
                fn=gradio_translate,
                inputs=[input_text, src_lang_text, tgt_lang_text],
                outputs=output_text,
            )

            gr.Examples(
                examples=[
                    ["Hello, how are you today?",           "English", "French"],
                    ["Machine learning is fascinating.",    "English", "Spanish"],
                    ["I love traveling around the world.",  "English", "Arabic"],
                    ["The weather is beautiful.",           "English", "German"],
                ],
                inputs=[input_text, src_lang_text, tgt_lang_text],
            )

        # Speech Translation
        with gr.Tab("🎤 Speech Translation"):
            with gr.Row():
                with gr.Column():
                    src_lang_speech = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="English",
                        label="🌐 Speech Language",
                    )
                    tgt_lang_speech = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES, value="French",
                        label="🌐 Target Language",
                    )

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎙️ Record or Upload Audio",
            )
            transcribed_output        = gr.Textbox(label="📝 Transcribed Text")
            speech_translation_output = gr.Textbox(label="✨ Translation")

            speech_translate_btn = gr.Button("🚀 Transcribe & Translate", variant="primary", size="lg")
            speech_translate_btn.click(
                fn=gradio_speech_translate,
                inputs=[audio_input, src_lang_speech, tgt_lang_speech],
                outputs=[transcribed_output, speech_translation_output],
            )

        # Batch Translation
        with gr.Tab("📦 Batch Translation"):
            gr.Markdown(
                """
                ### Translate multiple sentences at once
                Enter one sentence per line for faster processing.
                """
            )
            with gr.Row():
                src_lang_batch = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES, value="English", label="🌐 Source Language",
                )
                tgt_lang_batch = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES, value="Spanish", label="🌐 Target Language",
                )

            batch_input  = gr.Textbox(
                lines=10,
                placeholder="Enter sentences (one per line):\n\nSentence 1\nSentence 2\nSentence 3",
                label="📝 Input Sentences",
            )
            batch_output = gr.Textbox(lines=10, label="✨ Batch Translations")

            batch_btn = gr.Button("🚀 Translate Batch", variant="primary", size="lg")
            batch_btn.click(
                fn=gradio_batch_translate,
                inputs=[batch_input, src_lang_batch, tgt_lang_batch],
                outputs=batch_output,
            )

            gr.Examples(
                examples=[
                    ["Hello, how are you?\nWhat is your name?\nI love coding.", "English", "French"],
                ],
                inputs=[batch_input, src_lang_batch, tgt_lang_batch],
            )

        # BLEU Evaluation
        with gr.Tab("📊 BLEU Evaluation"):
            gr.Markdown(
                """
                ### Evaluate Translation Quality
                Compare reference translation with model output using BLEU score.

                **BLEU Score Guide:**
                - 60-100: Excellent ✅
                - 40-60:  Good 👍
                - 20-40:  Fair ⚠️
                - 0-20:   Poor ❌
                """
            )

            reference_text  = gr.Textbox(
                lines=5, placeholder="Enter reference (ground truth) translation...",
                label="📚 Reference Translation",
            )
            hypothesis_text = gr.Textbox(
                lines=5, placeholder="Enter model-generated translation...",
                label="🤖 Model Translation",
            )
            bleu_output = gr.Textbox(lines=15, label="📊 BLEU Score Report")

            bleu_btn = gr.Button("📊 Calculate BLEU", variant="primary", size="lg")
            bleu_btn.click(
                fn=gradio_bleu,
                inputs=[reference_text, hypothesis_text],
                outputs=bleu_output,
            )

            gr.Examples(
                examples=[
                    ["Le chat est sur le tapis",     "Le chat est sur le tapis"],
                    ["Bonjour, comment allez-vous?", "Bonjour, comment vas-tu?"],
                ],
                inputs=[reference_text, hypothesis_text],
            )

    gr.Markdown(
        """
        ---
        **Model:** Facebook NLLB-200-distilled-600M | **Framework:** PyTorch 2.10 + Transformers

        Built with ❤️ using Gradio
        """
    )


# Launch the app

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)