"""Tests for image processing in core."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from notely import Notely, NotelyConfig
from notely.config import ASRConfig, EnhancerConfig, LLMConfig, OCRConfig
from notely.ocr.base import OCRResult, TextBlock


def make_test_config() -> NotelyConfig:
    """Create a test configuration with Zhipu provider."""
    return NotelyConfig(
        asr=ASRConfig(),
        ocr=OCRConfig(provider="zhipu", api_key="test-key", model="glm-4v-flash"),
        enhancer=EnhancerConfig(llm=LLMConfig(provider="zhipu", api_key="test-key", model="glm-4")),
    )


@pytest.mark.asyncio
async def test_process_image(tmp_path):
    """Test end-to-end image processing."""
    # Create mock OCR backend
    mock_ocr = Mock()
    mock_ocr.recognize.return_value = OCRResult(
        text_blocks=[TextBlock(text="# Image Text", confidence=1.0, bbox=(0, 0, 0, 0))],
        source_path="test.jpg",
    )

    # Create mock enhancer
    mock_enhancer = Mock()
    mock_enhancer.process = AsyncMock(return_value="# Enhanced")
    mock_enhancer.metrics = Mock()

    # Create test image
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake image data")

    config = make_test_config()
    notely = Notely(config)

    with patch.object(notely, "_ocr", mock_ocr), patch.object(notely, "_enhancer", mock_enhancer):
        result = await notely.process(test_image)

        assert result.markdown
        assert result.transcript is None
        assert len(result.ocr_results) > 0
        mock_ocr.recognize.assert_called_once()


@pytest.mark.asyncio
async def test_process_unsupported_format(tmp_path):
    """Test that unsupported formats raise ValueError."""
    config = make_test_config()
    notely = Notely(config)

    # Create unsupported file
    test_file = tmp_path / "test.xyz"
    test_file.write_text("test")

    with pytest.raises(ValueError, match="Unsupported file type"):
        await notely.process(test_file)


@pytest.mark.asyncio
async def test_process_png_image(tmp_path):
    """Test PNG image processing."""
    mock_ocr = Mock()
    mock_ocr.recognize.return_value = OCRResult(
        text_blocks=[TextBlock(text="PNG content", confidence=1.0, bbox=(0, 0, 0, 0))],
        source_path="test.png",
    )

    mock_enhancer = Mock()
    mock_enhancer.process = AsyncMock(return_value="# Notes")
    mock_enhancer.metrics = Mock()

    test_image = tmp_path / "test.png"
    test_image.write_bytes(b"fake png data")

    config = make_test_config()
    notely = Notely(config)

    with patch.object(notely, "_ocr", mock_ocr), patch.object(notely, "_enhancer", mock_enhancer):
        result = await notely.process(test_image)

        assert "# Notes" in result.markdown
        mock_ocr.recognize.assert_called_once()


@pytest.mark.asyncio
async def test_process_bmp_image(tmp_path):
    """Test BMP image processing."""
    mock_ocr = Mock()
    mock_ocr.recognize.return_value = OCRResult(
        text_blocks=[TextBlock(text="BMP content", confidence=1.0, bbox=(0, 0, 0, 0))],
        source_path="test.bmp",
    )

    mock_enhancer = Mock()
    mock_enhancer.process = AsyncMock(return_value="# Notes")
    mock_enhancer.metrics = Mock()

    test_image = tmp_path / "test.bmp"
    test_image.write_bytes(b"fake bmp data")

    config = make_test_config()
    notely = Notely(config)

    with patch.object(notely, "_ocr", mock_ocr), patch.object(notely, "_enhancer", mock_enhancer):
        result = await notely.process(test_image)

        assert "# Notes" in result.markdown
        mock_ocr.recognize.assert_called_once()
