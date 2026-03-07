"""Language detection utilities for transcript analysis."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Threshold for CJK character ratio to determine language
# If >30% of characters are CJK, classify as that language
CJK_THRESHOLD = 0.3


def detect_transcript_language(
    text: str,
    asr_hint: str | None = None,
    confidence_threshold: float = 0.8,
) -> str:
    """Detect language from transcript with optional ASR hint.

    Args:
        text: Transcript text to analyze
        asr_hint: Optional language hint from ASR model (e.g., 'zh', 'en')
        confidence_threshold: Minimum confidence score to trust detection (0.0-1.0)

    Returns:
        Language code ('zh', 'en', 'ja', 'ko', etc.)

    Example:
        # With ASR hint
        lang = detect_transcript_language(text, asr_hint="zh")

        # Without hint (auto-detect)
        lang = detect_transcript_language(text)
    """
    # Priority 1: Trust ASR hint if provided
    if asr_hint:
        normalized = normalize_language_code(asr_hint)
        logger.info(f"Using ASR language hint: {asr_hint} -> {normalized}")
        return normalized

    # Priority 2: Use fastText detection
    try:
        from fast_langdetect import detect  # type: ignore[import-untyped]

        results = detect(text)
        if results and results[0]["score"] >= confidence_threshold:
            detected = normalize_language_code(results[0]["lang"])
            logger.info(f"Detected language: {detected} (confidence: {results[0]['score']:.2f})")
            return detected
    except ImportError:
        logger.warning("fast-langdetect not installed, falling back to simple detection")
    except Exception as e:
        logger.warning(f"fastText detection failed: {e}, falling back to simple detection")

    # Priority 3: Simple character-based fallback
    return _simple_detect(text)


def normalize_language_code(code: str) -> str:
    """Normalize language codes to standard 2-letter format.

    Args:
        code: Language code in various formats (e.g., 'zh-cn', 'zh_CN', 'chinese')

    Returns:
        Normalized 2-letter code ('zh', 'en', 'ja', 'ko')

    Example:
        normalize_language_code("zh-cn")  # Returns "zh"
        normalize_language_code("english")  # Returns "en"
    """
    # Extract base code (remove region/script)
    code = code.lower().split("-")[0].split("_")[0]

    # Map common variations to standard codes
    mapping = {
        "zh": "zh",
        "chinese": "zh",
        "cmn": "zh",  # Mandarin Chinese
        "en": "en",
        "english": "en",
        "ja": "ja",
        "japanese": "ja",
        "ko": "ko",
        "korean": "ko",
        "es": "es",
        "spanish": "es",
        "fr": "fr",
        "french": "fr",
        "de": "de",
        "german": "de",
        "ru": "ru",
        "russian": "ru",
    }

    normalized = mapping.get(code)
    if normalized is None:
        logger.warning(f"Unknown language code '{code}', defaulting to 'en'")
        return "en"
    return normalized


def _simple_detect(text: str) -> str:
    """Simple character-based language detection as fallback.

    Args:
        text: Text to analyze

    Returns:
        Language code ('zh', 'en', 'ja', 'ko')
    """
    if not text or not text.strip():
        return "en"

    # Count character types
    chinese_chars = len(re.findall(r"[\u4e00-\u9fa5]", text))
    japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    korean_chars = len(re.findall(r"[\uac00-\ud7af]", text))

    total_chars = len(text)

    # Guard against division by zero (shouldn't happen due to check above, but be safe)
    if total_chars == 0:
        return "en"

    # Threshold: >30% CJK characters indicates that language
    if chinese_chars / total_chars > CJK_THRESHOLD:
        logger.info(f"Simple detection: Chinese ({chinese_chars}/{total_chars} chars)")
        return "zh"
    elif japanese_chars / total_chars > CJK_THRESHOLD:
        logger.info(f"Simple detection: Japanese ({japanese_chars}/{total_chars} chars)")
        return "ja"
    elif korean_chars / total_chars > CJK_THRESHOLD:
        logger.info(f"Simple detection: Korean ({korean_chars}/{total_chars} chars)")
        return "ko"
    else:
        logger.info("Simple detection: English (default)")
        return "en"
