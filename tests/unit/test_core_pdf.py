"""Tests for PDF processing in core."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from notely import Notely, NotelyConfig
from notely.config import ASRConfig, EnhancerConfig, LLMConfig, OCRConfig
from notely.ocr.base import OCRResult, TextBlock


@pytest.fixture
def mock_ocr_backend():
    """Create a mock OCR backend."""
    backend = Mock()
    text_block = TextBlock(
        text="# PDF Content\nTest text",
        confidence=1.0,
        bbox=(0, 0, 0, 0),
        block_type="text",
    )
    ocr_result = OCRResult(
        text_blocks=[text_block],
        source_path="test.pdf",
        metadata={},
    )
    backend.recognize_pdf.return_value = [ocr_result]
    return backend


@pytest.fixture
def mock_enhancer():
    """Create a mock enhancer that returns markdown."""
    enhancer = Mock()
    enhancer.process = AsyncMock(return_value="# Enhanced Notes\n\nThis is the content.")
    enhancer.metrics = Mock()
    return enhancer


def make_test_config() -> NotelyConfig:
    """Create a test configuration with Zhipu provider."""
    return NotelyConfig(
        asr=ASRConfig(),
        ocr=OCRConfig(provider="zhipu", api_key="test-key", model="glm-4v-flash"),
        enhancer=EnhancerConfig(llm=LLMConfig(provider="zhipu", api_key="test-key", model="glm-4")),
    )


@pytest.mark.asyncio
async def test_process_pdf(tmp_path, mock_ocr_backend, mock_enhancer):
    """Test end-to-end PDF processing."""
    # Create test PDF
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_bytes(b"fake pdf data")

    config = make_test_config()
    notely = Notely(config)

    # Mock the OCR backend and enhancer
    with (
        patch.object(notely, "_ocr", mock_ocr_backend),
        patch.object(notely, "_enhancer", mock_enhancer),
    ):
        result = await notely.process(test_pdf)

        assert result.markdown
        assert result.transcript is None
        assert len(result.ocr_results) > 0
        mock_ocr_backend.recognize_pdf.assert_called_once()


@pytest.mark.asyncio
async def test_process_pdf_calls_ocr_with_correct_path(tmp_path):
    """Test that PDF processing calls OCR with the correct file path."""
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_bytes(b"fake pdf data")

    config = make_test_config()
    notely = Notely(config)

    # Create mock OCR backend
    mock_ocr = Mock()
    mock_ocr.recognize_pdf.return_value = [
        OCRResult(
            text_blocks=[TextBlock(text="content", confidence=1.0, bbox=(0, 0, 0, 0))],
            source_path=str(test_pdf),
        )
    ]

    with patch.object(notely, "_ocr", mock_ocr), patch.object(notely, "_enhancer") as mock_enhancer:
        mock_enhancer.process = AsyncMock(return_value="# Notes")
        mock_enhancer.metrics = Mock()

        await notely.process(test_pdf)

        # Verify OCR was called with the PDF path
        mock_ocr.recognize_pdf.assert_called_once()
        called_path = mock_ocr.recognize_pdf.call_args[0][0]
        assert Path(called_path) == test_pdf
