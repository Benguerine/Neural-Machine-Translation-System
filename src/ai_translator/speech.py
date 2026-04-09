


import speech_recognition as sr
from typing import Optional
from .languages import get_speech_code



def speech_to_text(
    audio_file: Optional[str],
    language: str = "English",
) -> str:
    
    """Transcribe an audio file to text."""

    if not audio_file:
        return "No audio file provided"
    
    try:
        recognizer = sr.Recognizer()
        lang_code = get_speech_code(language, fallback="en-US")

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        return recognizer.recognize_google(audio_data, language=lang_code)
    
    except sr.UnknownValueError:
        return "couldn't understand audio"
    
    except sr.RequestError as exc:
        return f"Speech recognition service error: {exc}"
    except Exception as exc:
        return f"Error processing audio: {exc}"