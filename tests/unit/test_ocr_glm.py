"""Tests for GLM-OCR backend."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from notely.config import OCRConfig
from notely.ocr.base import OCRResult
from notely.ocr.glm import GLMOCRBackend


def test_glm_ocr_requires_api_key():
    """Test that GLM-OCR requires API key."""
    # OCRConfig validates API key in __post_init__, so this raises ValueError
    with pytest.raises(ValueError, match="api_key is required"):
        OCRConfig(provider="zhipu", api_key="")


def test_glm_ocr_initialization():
    """Test GLM-OCR backend initialization."""
    config = OCRConfig(provider="zhipu", api_key="test-key", model="glm-4v-flash")
    backend = GLMOCRBackend(config)

    assert backend.model == "glm-4v-flash"


@patch("zhipuai.ZhipuAI")
def test_glm_ocr_recognize(mock_zhipu_class, tmp_path):
    """Test image recognition with mocked API."""
    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "# Test\nExtracted text"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Create test image file
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake image data")

    # Test recognition
    config = OCRConfig(provider="zhipu", api_key="test-key", model="glm-4v-flash")
    backend = GLMOCRBackend(config)

    result = backend.recognize(test_image)

    assert result.full_text == "# Test\nExtracted text"
    assert len(result.text_blocks) == 1


@patch("zhipuai.ZhipuAI")
def test_glm_ocr_recognize_file_not_found(mock_zhipu_class):
    """Test that recognize raises FileNotFoundError for missing file."""
    mock_client = Mock()
    mock_zhipu_class.return_value = mock_client

    config = OCRConfig(provider="zhipu", api_key="test-key")
    backend = GLMOCRBackend(config)

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        backend.recognize("/nonexistent/image.jpg")


@patch("zhipuai.ZhipuAI")
@patch("pdfplumber.open")
def test_glm_ocr_recognize_pdf(mock_pdfplumber_open, mock_zhipu_class, tmp_path):
    """Test PDF recognition with mocked API."""
    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Page 1 content"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Mock PIL Image
    from io import BytesIO

    from PIL import Image

    # Create a simple 1x1 white image
    img = Image.new("RGB", (1, 1), color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Mock page image
    mock_page_image = Mock()
    mock_page_image.original = img

    # Mock page
    mock_page = Mock()
    mock_page.to_image.return_value = mock_page_image

    # Mock PDF context manager
    mock_pdf = Mock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = Mock(return_value=mock_pdf)
    mock_pdf.__exit__ = Mock(return_value=False)

    mock_pdfplumber_open.return_value = mock_pdf

    # Create test PDF file (content doesn't matter since we mock pdfplumber)
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_bytes(b"fake pdf data")

    # Test recognition
    config = OCRConfig(provider="zhipu", api_key="test-key", model="glm-4v-flash")
    backend = GLMOCRBackend(config)

    results = backend.recognize_pdf(test_pdf)

    assert len(results) == 1
    assert isinstance(results[0], OCRResult)
    assert results[0].full_text == "Page 1 content"
    assert results[0].page_number == 1


@patch("zhipuai.ZhipuAI")
def test_glm_ocr_is_available(mock_zhipu_class):
    """Test that is_available returns True when client is initialized."""
    mock_client = Mock()
    mock_zhipu_class.return_value = mock_client

    config = OCRConfig(provider="zhipu", api_key="test-key")
    backend = GLMOCRBackend(config)

    assert backend.is_available() is True
