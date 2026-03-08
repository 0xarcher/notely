"""Tests for language detection utilities."""

from notely.utils.language import (
    _simple_detect,
    detect_transcript_language,
    normalize_language_code,
)


def test_detect_chinese():
    """Test detection of Chinese text."""
    text = "这是一段中文文本，用于测试语言检测功能。机器学习是人工智能的一个分支。"
    result = detect_transcript_language(text)
    assert result == "zh"


def test_detect_english():
    """Test detection of English text."""
    text = "This is an English text for testing language detection. Machine learning is awesome."
    result = detect_transcript_language(text)
    assert result == "en"


def test_asr_hint_priority():
    """Test that ASR hint takes priority over detection."""
    text = "Mixed 中文 and English content here"
    result = detect_transcript_language(text, asr_hint="zh")
    assert result == "zh"


def test_normalize_language_code():
    """Test language code normalization."""
    assert normalize_language_code("zh-cn") == "zh"
    assert normalize_language_code("zh_CN") == "zh"
    assert normalize_language_code("chinese") == "zh"
    assert normalize_language_code("en-us") == "en"
    assert normalize_language_code("english") == "en"


def test_normalize_unknown_language_code():
    """Test that unknown language codes default to English with warning."""
    result = normalize_language_code("xyz")
    assert result == "en"


def test_detect_empty_text():
    """Test detection with empty text."""
    assert detect_transcript_language("") == "en"


def test_detect_whitespace_only():
    """Test detection with whitespace-only text."""
    assert detect_transcript_language("   \n\t  ") == "en"


def test_simple_detect_japanese():
    """Test simple detection for Japanese text."""
    # Hiragana and Katakana
    text = "これはテストです。ひらがなとカタカナを使います。"
    result = _simple_detect(text)
    assert result == "ja"


def test_simple_detect_korean():
    """Test simple detection for Korean text."""
    text = "이것은 한국어 테스트입니다. 기계 학습은 인공 지능의 한 분야입니다."
    result = _simple_detect(text)
    assert result == "ko"


def test_simple_detect_chinese():
    """Test simple detection for Chinese text."""
    text = "这是中文测试。机器学习很有趣。"
    result = _simple_detect(text)
    assert result == "zh"


def test_simple_detect_english():
    """Test simple detection for English text."""
    text = "This is a test in English. Machine learning is fascinating."
    result = _simple_detect(text)
    assert result == "en"


def test_simple_detect_empty():
    """Test simple detection with empty text."""
    assert _simple_detect("") == "en"


def test_simple_detect_whitespace():
    """Test simple detection with whitespace-only text."""
    assert _simple_detect("   \n\t  ") == "en"


def test_detect_japanese_with_fasttext():
    """Test detection of Japanese text using fastText."""
    text = "これは日本語のテストです。機械学習は人工知能の一分野です。"
    result = detect_transcript_language(text)
    assert result == "ja"


def test_detect_korean_with_fasttext():
    """Test detection of Korean text using fastText."""
    text = "이것은 한국어 테스트입니다. 기계 학습은 인공 지능의 한 분야입니다."
    result = detect_transcript_language(text)
    assert result == "ko"
