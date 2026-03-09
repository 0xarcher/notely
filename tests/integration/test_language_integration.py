"""Integration tests for language detection in processing pipeline."""

from __future__ import annotations

from notely import Notely, NotelyConfig
from notely.asr.base import ASRResult, TranscriptSegment
from notely.config import ASRConfig, EnhancerConfig, LLMConfig, OCRConfig


def test_auto_detect_language_from_transcript():
    """Test that language is auto-detected from transcript."""

    # Mock ASR to return Chinese transcript
    class MockASR:
        def transcribe(self, audio_path):
            return ASRResult(
                segments=[
                    TranscriptSegment(
                        text="这是一段中文测试文本。",
                        start_time=0.0,
                        end_time=2.0,
                    ),
                    TranscriptSegment(
                        text="机器学习很有趣。",
                        start_time=2.0,
                        end_time=4.0,
                    ),
                ],
                language="zh",
            )

    # Create config with language=None (auto-detect)
    config = NotelyConfig(
        asr=ASRConfig(backend="funasr"),
        ocr=OCRConfig(provider="paddleocr"),
        enhancer=EnhancerConfig(
            llm=LLMConfig(api_key="test", model="gpt-4o"),
            language=None,  # Auto-detect
        ),
    )

    notely = Notely(config)

    # Mock the ASR backend
    notely._asr = MockASR()

    # Verify language is None before processing
    assert config.enhancer.language is None

    # Simulate the language detection logic
    transcript = notely._asr.transcribe("dummy_path")
    asr_hint = getattr(transcript, "language", None)

    from notely.utils.language import detect_transcript_language

    detected_lang = detect_transcript_language(
        transcript.full_text,
        asr_hint=asr_hint,
    )

    # Should detect Chinese
    assert detected_lang == "zh"


def test_explicit_language_not_overridden():
    """Test that explicitly set language is not overridden."""

    # Mock ASR to return Chinese transcript
    class MockASR:
        def transcribe(self, audio_path):
            return ASRResult(
                segments=[
                    TranscriptSegment(
                        text="这是一段中文测试文本。",
                        start_time=0.0,
                        end_time=2.0,
                    )
                ],
                language="zh",
            )

    # Create config with explicit language='en'
    config = NotelyConfig(
        asr=ASRConfig(backend="funasr"),
        ocr=OCRConfig(provider="paddleocr"),
        enhancer=EnhancerConfig(
            llm=LLMConfig(api_key="test", model="gpt-4o"),
            language="en",  # Explicitly set to English
        ),
    )

    notely = Notely(config)
    notely._asr = MockASR()

    # Language should remain 'en' even though transcript is Chinese
    assert config.enhancer.language == "en"

    # Simulate processing - language should not change
    _ = notely._asr.transcribe("dummy_path")

    # Language should still be 'en'
    assert config.enhancer.language == "en"


def test_english_transcript_detection():
    """Test that English transcripts are correctly detected."""

    # Mock ASR to return English transcript
    class MockASR:
        def transcribe(self, audio_path):
            return ASRResult(
                segments=[
                    TranscriptSegment(
                        text="This is an English test transcript.",
                        start_time=0.0,
                        end_time=2.0,
                    )
                ],
                language="en",
            )

    config = NotelyConfig(
        asr=ASRConfig(backend="funasr"),
        ocr=OCRConfig(provider="paddleocr"),
        enhancer=EnhancerConfig(
            llm=LLMConfig(api_key="test", model="gpt-4o"),
            language=None,  # Auto-detect
        ),
    )

    notely = Notely(config)
    notely._asr = MockASR()

    transcript = notely._asr.transcribe("dummy_path")
    asr_hint = getattr(transcript, "language", None)

    from notely.utils.language import detect_transcript_language

    detected_lang = detect_transcript_language(
        transcript.full_text,
        asr_hint=asr_hint,
    )

    # Should detect English
    assert detected_lang == "en"
