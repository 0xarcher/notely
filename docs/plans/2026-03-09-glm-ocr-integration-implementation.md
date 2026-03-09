# GLM-OCR Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate Zhipu AI's GLM-OCR API to replace PaddleOCR and add PDF/image document support.

**Architecture:** Add GLMOCRBackend and ZhipuLLMBackend implementations, update configuration to support provider/model/api_key pattern, add PDF/image processing methods to core.py.

**Tech Stack:** Python 3.11+, zhipuai SDK, pytest, ruff, mypy

---

## Task 0: Add zhipuai Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (auto-generated)

**Step 1: Add dependency**

Edit `pyproject.toml` dependencies section:
```toml
dependencies = [
    "pyyaml>=6.0",
    "httpx>=0.25.0",
    "rich>=13.0.0",
    "zhipuai>=2.0.0",  # Zhipu AI official SDK
]
```

**Step 2: Install dependencies**

Run: `cd /Users/didi/coding/python/notely && uv sync`

Expected:
```
Resolved 200 packages in 1.2s
Installed 200 packages
```

**Step 3: Commit**

Run:
```bash
cd /Users/didi/coding/python/notely
git add pyproject.toml uv.lock
git commit -m "deps: add zhipuai for GLM-OCR integration"
```

Expected: Commit created successfully

---

## Task 1: Update Configuration Classes

**Files:**
- Modify: `src/notely/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write failing tests for OCRConfig**

Edit `tests/test_config.py`, add:
```python
def test_ocr_config_zhipu_provider():
    """Test OCRConfig with Zhipu provider."""
    config = OCRConfig(
        provider="zhipu",
        model="glm-4v-flash",
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    assert config.provider == "zhipu"
    assert config.model == "glm-4v-flash"
    assert config.api_key == "test-key"


def test_ocr_config_default_paddleocr():
    """Test OCRConfig defaults to PaddleOCR."""
    config = OCRConfig()
    assert config.provider == "paddleocr"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_config.py::test_ocr_config_zhipu_provider -v`

Expected: FAIL with "OCRConfig has no attribute 'provider'"

**Step 3: Update OCRConfig**

Edit `src/notely/config.py`, update `OCRConfig`:
```python
@dataclass
class OCRConfig:
    """
    OCR configuration supporting both local and cloud providers.

    Attributes:
        provider: OCR provider name ('paddleocr' or 'zhipu')
        model: Model identifier (for cloud providers, e.g., 'glm-4v-flash')
        api_key: API key for cloud providers
        base_url: Base URL for cloud provider API
        language: Language code for local OCR (e.g., 'ch', 'en')
        use_gpu: Whether to use GPU for local OCR
    """

    provider: str = "paddleocr"
    model: str = "glm-4v-flash"
    api_key: str = ""
    base_url: str = ""
    language: str = "ch"
    use_gpu: bool = True
```

**Step 4: Write failing tests for LLMConfig**

Edit `tests/test_config.py`, add:
```python
def test_llm_config_zhipu_provider():
    """Test LLMConfig with Zhipu provider."""
    config = LLMConfig(
        provider="zhipu",
        model="glm-4",
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    assert config.provider == "zhipu"
    assert config.model == "glm-4"
```

**Step 5: Update LLMConfig**

Edit `src/notely/config.py`, update `LLMConfig`:
```python
@dataclass
class LLMConfig:
    """
    LLM configuration supporting multiple providers.

    Attributes:
        provider: LLM provider name ('openai', 'zhipu', 'anthropic')
        model: Model identifier
        api_key: API key for authentication
        base_url: Base URL for API endpoint
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
    """

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
```

**Step 6: Write failing test for ASRConfig device removal**

Edit `tests/test_config.py`, add:
```python
def test_asr_config_no_device():
    """Test ASRConfig has no device field (auto-detected)."""
    config = ASRConfig()
    assert not hasattr(config, 'device')
```

**Step 7: Remove device from ASRConfig**

Edit `src/notely/config.py`, update `ASRConfig`:
```python
@dataclass
class ASRConfig:
    """
    ASR configuration.

    Attributes:
        backend: ASR backend name ('funasr')
        model: Model identifier
        language: Language code
    """

    backend: str = "funasr"
    model: str = "paraformer-zh"
    language: str = "zh"
```

**Step 8: Run all config tests**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_config.py -v`

Expected: All tests PASS

**Step 9: Commit**

Run:
```bash
cd /Users/didi/coding/python/notely
git add src/notely/config.py tests/test_config.py
git commit -m "feat(config): add provider/model/api_key config for OCR and LLM

- Add provider, model, api_key, base_url to OCRConfig
- Add provider, api_key, base_url to LLMConfig
- Remove device from ASRConfig (auto-detect GPU)
- Maintain backward compatibility with PaddleOCR"
```

---

## Task 2: Create GLMOCRBackend

**Files:**
- Create: `src/notely/ocr/glm.py`
- Modify: `src/notely/ocr/__init__.py`
- Create: `tests/test_ocr_glm.py`

**Step 1: Write failing test for GLMOCRBackend initialization**

Create `tests/test_ocr_glm.py`:
```python
"""Tests for GLM-OCR backend."""

from __future__ import annotations

import pytest

from notely.config import OCRConfig
from notely.ocr.glm import GLMOCRBackend


def test_glm_ocr_requires_api_key():
    """Test that GLM-OCR requires API key."""
    config = OCRConfig(provider="zhipu", api_key="")

    with pytest.raises(ValueError, match="API key required"):
        GLMOCRBackend(config)


def test_glm_ocr_initialization():
    """Test GLM-OCR backend initialization."""
    config = OCRConfig(
        provider="zhipu",
        api_key="test-key",
        model="glm-4v-flash"
    )
    backend = GLMOCRBackend(config)

    assert backend.model == "glm-4v-flash"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_ocr_glm.py::test_glm_ocr_requires_api_key -v`

Expected: FAIL with "cannot import name 'GLMOCRBackend'"

**Step 3: Create GLMOCRBackend**

Create `src/notely/ocr/glm.py`:
```python
"""GLM-OCR backend using Zhipu AI API."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from notely.config import OCRConfig
from notely.ocr.base import OCRBackend, OCRResult, TextBlock

logger = logging.getLogger(__name__)


class GLMOCRBackend(OCRBackend):
    """
    GLM-OCR backend using Zhipu AI's GLM-4V API.

    This backend sends images/PDFs to Zhipu AI's cloud API for OCR processing.
    Requires valid API key and internet connection.
    """

    def __init__(self, config: OCRConfig) -> None:
        """
        Initialize GLM-OCR backend.

        Args:
            config: OCR configuration with Zhipu AI credentials

        Raises:
            ValueError: If API key is missing
            ImportError: If zhipuai package is not installed
        """
        if not config.api_key:
            raise ValueError("API key required for Zhipu OCR provider")

        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError(
                "zhipuai package required for GLM-OCR. "
                "Install with: pip install zhipuai"
            ) from e

        self.client = ZhipuAI(
            api_key=config.api_key,
            base_url=config.base_url
            or "https://open.bigmodel.cn/api/paas/v4",
        )
        self.model = config.model

    def recognize(self, image_path: Path | str) -> OCRResult:
        """
        Recognize text in an image using GLM-4V API.

        Args:
            image_path: Path to image file

        Returns:
            OCRResult with detected text blocks

        Raises:
            RuntimeError: If API call fails
            FileNotFoundError: If image file doesn't exist
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Recognizing text in image: {image_path}")

        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine image MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        # Call GLM-4V API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract all text from this image. "
                                "Return the result in Markdown format.",
                            },
                        ],
                    }
                ],
            )

            text = response.choices[0].message.content or ""

            logger.info(f"OCR completed for {image_path.name}")

            # Parse response to OCRResult
            return OCRResult(
                text_blocks=[
                    TextBlock(
                        text=text,
                        confidence=1.0,
                        bbox=(0, 0, 0, 0),
                        block_type="text",
                    )
                ],
                source_path=str(image_path),
            )

        except Exception as e:
            logger.error(f"GLM-OCR API call failed: {e}")
            raise RuntimeError(f"OCR failed: {e}") from e

    def recognize_pdf(self, pdf_path: Path | str) -> list[OCRResult]:
        """
        Recognize text in a PDF using GLM-4V API.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of OCRResult, one per page

        Raises:
            RuntimeError: If API call fails
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Recognizing text in PDF: {pdf_path}")

        # Encode PDF to base64
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        # Call GLM-4V API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_data}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract all text from this PDF document. "
                                "Return the result in Markdown format.",
                            },
                        ],
                    }
                ],
            )

            text = response.choices[0].message.content or ""

            logger.info(f"OCR completed for {pdf_path.name}")

            # Return single OCRResult for entire PDF
            # (GLM-4V processes entire PDF at once)
            return [
                OCRResult(
                    text_blocks=[
                        TextBlock(
                            text=text,
                            confidence=1.0,
                            bbox=(0, 0, 0, 0),
                            block_type="text",
                        )
                    ],
                    source_path=str(pdf_path),
                )
            ]

        except Exception as e:
            logger.error(f"GLM-OCR API call failed: {e}")
            raise RuntimeError(f"OCR failed: {e}") from e

    def is_available(self) -> bool:
        """
        Check if GLM-OCR backend is available.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test call
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
```

**Step 4: Update __init__.py**

Edit `src/notely/ocr/__init__.py`:
```python
"""OCR backends for Notely."""

from notely.ocr.base import OCRBackend, OCRResult, TextBlock
from notely.ocr.glm import GLMOCRBackend
from notely.ocr.paddle import PaddleOCRBackend

__all__ = ["OCRBackend", "OCRResult", "TextBlock", "GLMOCRBackend", "PaddleOCRBackend"]
```

**Step 5: Write test for recognize() with mocked API**

Edit `tests/test_ocr_glm.py`, add:
```python
from pathlib import Path
from unittest.mock import Mock, patch, mock_open


@patch("notely.ocr.glm.ZhipuAI")
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
    assert result.text_blocks[0].text == "# Test\nExtracted text"


@patch("notely.ocr.glm.ZhipuAI")
def test_glm_ocr_recognize_file_not_found(mock_zhipu_class):
    """Test that recognize raises FileNotFoundError for missing file."""
    config = OCRConfig(provider="zhipu", api_key="test-key")
    backend = GLMOCRBackend(config)

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        backend.recognize("/nonexistent/image.jpg")
```

**Step 6: Run all tests**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_ocr_glm.py -v`

Expected: All tests PASS

**Step 7: Commit**

Run:
```bash
cd /Users/didi/coding/python/notely
git add src/notely/ocr/glm.py src/notely/ocr/__init__.py tests/test_ocr_glm.py
git commit -m "feat(ocr): add GLMOCRBackend using Zhipu AI API

- Implement GLMOCRBackend with recognize() and recognize_pdf()
- Support base64 image/PDF upload to GLM-4V API
- Add comprehensive unit tests with mocked API
- Export GLMOCRBackend from ocr module"
```

---

## Task 3: Create ZhipuLLMBackend

**Files:**
- Create: `src/notely/llm/zhipu.py`
- Modify: `src/notely/llm/__init__.py`
- Create: `tests/test_llm_zhipu.py`

**Step 1: Write failing test**

Create `tests/test_llm_zhipu.py`:
```python
"""Tests for Zhipu LLM backend."""

from __future__ import annotations

import pytest

from notely.config import LLMConfig
from notely.llm.zhipu import ZhipuLLMBackend


def test_zhipu_llm_requires_api_key():
    """Test that Zhipu LLM requires API key."""
    config = LLMConfig(provider="zhipu", api_key="")

    with pytest.raises(ValueError, match="API key required"):
        ZhipuLLMBackend(config)


def test_zhipu_llm_initialization():
    """Test Zhipu LLM backend initialization."""
    config = LLMConfig(
        provider="zhipu",
        api_key="test-key",
        model="glm-4"
    )
    backend = ZhipuLLMBackend(config)

    assert backend.model == "glm-4"
    assert backend.temperature == 0.7
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_llm_zhipu.py::test_zhipu_llm_requires_api_key -v`

Expected: FAIL with "cannot import name 'ZhipuLLMBackend'"

**Step 3: Read existing LLM base class**

Run: `cd /Users/didi/coding/python/notely && head -50 src/notely/llm/base.py`

**Step 4: Create ZhipuLLMBackend**

Create `src/notely/llm/zhipu.py`:
```python
"""Zhipu AI LLM backend."""

from __future__ import annotations

import logging
from typing import Any

from notely.config import LLMConfig
from notely.llm.base import LLMBackend

logger = logging.getLogger(__name__)


class ZhipuLLMBackend(LLMBackend):
    """
    Zhipu AI LLM backend using official SDK.

    Supports GLM-4 and other Zhipu AI models.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize Zhipu LLM backend.

        Args:
            config: LLM configuration with Zhipu AI credentials

        Raises:
            ValueError: If API key is missing
            ImportError: If zhipuai package is not installed
        """
        if not config.api_key:
            raise ValueError("API key required for Zhipu LLM provider")

        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError(
                "zhipuai package required. Install with: pip install zhipuai"
            ) from e

        self.client = ZhipuAI(
            api_key=config.api_key,
            base_url=config.base_url
            or "https://open.bigmodel.cn/api/paas/v4",
        )
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        logger.info(f"Initialized Zhipu LLM backend with model: {self.model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Zhipu AI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text

        Raises:
            RuntimeError: If API call fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Generating text with model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result = response.choices[0].message.content or ""

            logger.debug(f"Generated {len(result)} characters")

            return result

        except Exception as e:
            logger.error(f"Zhipu LLM API call failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        """
        Stream text generation using Zhipu AI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks

        Raises:
            RuntimeError: If API call fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Streaming text with model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Zhipu LLM streaming failed: {e}")
            raise RuntimeError(f"LLM streaming failed: {e}") from e
```

**Step 5: Update __init__.py**

Edit `src/notely/llm/__init__.py`:
```python
"""LLM backends for Notely."""

from notely.llm.base import LLMBackend
from notely.llm.openai import OpenAILLMBackend
from notely.llm.zhipu import ZhipuLLMBackend

__all__ = ["LLMBackend", "OpenAILLMBackend", "ZhipuLLMBackend"]
```

**Step 6: Write test for generate() with mocked API**

Edit `tests/test_llm_zhipu.py`, add:
```python
from unittest.mock import Mock, patch
import pytest


@pytest.mark.asyncio
@patch("notely.llm.zhipu.ZhipuAI")
async def test_zhipu_llm_generate(mock_zhipu_class):
    """Test text generation with mocked API."""
    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Generated text"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Test generation
    config = LLMConfig(
        provider="zhipu",
        api_key="test-key",
        model="glm-4"
    )
    backend = ZhipuLLMBackend(config)

    result = await backend.generate("Test prompt")

    assert result == "Generated text"
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch("notely.llm.zhipu.ZhipuAI")
async def test_zhipu_llm_generate_with_system_prompt(mock_zhipu_class):
    """Test generation with system prompt."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Response"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    config = LLMConfig(provider="zhipu", api_key="test-key")
    backend = ZhipuLLMBackend(config)

    result = await backend.generate(
        "User prompt",
        system_prompt="System instruction"
    )

    assert result == "Response"

    # Verify system prompt was included
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
```

**Step 7: Run all tests**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_llm_zhipu.py -v`

Expected: All tests PASS

**Step 8: Commit**

Run:
```bash
cd /Users/didi/coding/python/notely
git add src/notely/llm/zhipu.py src/notely/llm/__init__.py tests/test_llm_zhipu.py
git commit -m "feat(llm): add ZhipuLLMBackend using Zhipu AI SDK

- Implement ZhipuLLMBackend with generate() and stream()
- Support system prompts and streaming
- Add comprehensive async unit tests
- Export ZhipuLLMBackend from llm module"
```

---

## Task 4: Update ASR Auto-detect GPU

**Files:**
- Modify: `src/notely/asr/funasr.py`

**Step 1: Read current FunASRBackend implementation**

Run: `cd /Users/didi/coding/python/notely && head -80 src/notely/asr/funasr.py`

**Step 2: Update __init__ to auto-detect device**

Edit `src/notely/asr/funasr.py`, modify `__init__`:
```python
def __init__(self, config: ASRConfig) -> None:
    """
    Initialize FunASR backend with auto-detected device.

    Args:
        config: ASR configuration
    """
    # Auto-detect GPU availability
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    logger.info(f"FunASR using device: {device}")

    # Initialize model with detected device
    # ... rest of initialization
```

**Step 3: Run ASR tests**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_asr_funasr.py -v`

Expected: All tests PASS

**Step 4: Commit**

Run:
```bash
cd /Users/didi/coding/python/notely
git add src/notely/asr/funasr.py
git commit -m "feat(asr): auto-detect GPU for FunASR

- Remove manual device configuration
- Auto-detect CUDA availability using torch
- Default to CPU if CUDA unavailable"
```

---

## Task 5: Add PDF and Image Processing

**Files:**
- Modify: `src/notely/core.py`
- Create: `tests/test_core_pdf.py`
- Create: `tests/test_core_image.py`

**Step 1: Write failing test for PDF processing**

Create `tests/test_core_pdf.py`:
```python
"""Tests for PDF processing in core."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

from notely import Notely, NotelyConfig
from notely.config import OCRConfig, EnhancerConfig, LLMConfig


@pytest.mark.asyncio
@patch("notely.ocr.glm.ZhipuAI")
@patch("notely.llm.zhipu.ZhipuAI")
async def test_process_pdf(mock_llm_zhipu, mock_ocr_zhipu, tmp_path):
    """Test end-to-end PDF processing."""
    # Mock OCR response
    mock_ocr_response = Mock()
    mock_ocr_response.choices = [Mock()]
    mock_ocr_response.choices[0].message.content = "# PDF Content\nTest text"

    mock_ocr_client = Mock()
    mock_ocr_client.chat.completions.create.return_value = mock_ocr_response
    mock_ocr_zhipu.return_value = mock_ocr_client

    # Mock LLM response
    mock_llm_response = Mock()
    mock_llm_response.choices = [Mock()]
    mock_llm_response.choices[0].message.content = "# Enhanced Notes"

    mock_llm_client = Mock()
    mock_llm_client.chat.completions.create.return_value = mock_llm_response
    mock_llm_zhipu.return_value = mock_llm_client

    # Create test PDF
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_bytes(b"fake pdf data")

    # Configure Notely with Zhipu
    config = NotelyConfig(
        ocr=OCRConfig(
            provider="zhipu",
            api_key="test-key",
            model="glm-4v-flash"
        ),
        enhancer=EnhancerConfig(
            llm=LLMConfig(
                provider="zhipu",
                api_key="test-key",
                model="glm-4"
            )
        )
    )

    notely = Notely(config)
    result = await notely.process(test_pdf)

    assert result.markdown
    assert result.transcript is None
    assert len(result.ocr_results) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_core_pdf.py -v`

Expected: FAIL with appropriate error

**Step 3: Read current core.py implementation**

Run: `cd /Users/didi/coding/python/notely && head -100 src/notely/core.py`

**Step 4: Add PDF processing method**

Edit `src/notely/core.py`, add method:
```python
async def _process_pdf(
    self,
    pdf_path: Path,
    metadata: dict[str, Any] | None = None,
) -> NotelyResult:
    """
    Process PDF file using GLM-OCR.

    Args:
        pdf_path: Path to PDF file
        metadata: Optional metadata

    Returns:
        NotelyResult with generated notes
    """
    metadata = metadata or {}

    logger.info(f"Processing PDF: {pdf_path}")

    # OCR PDF
    ocr_results = self.ocr.recognize_pdf(pdf_path)

    # Generate notes with enhancer
    markdown = await self.enhancer.process(
        transcript=None,
        ocr_results=ocr_results,
        metadata=metadata,
    )

    # Format
    formatted = self._formatter.beautify(markdown)

    return NotelyResult(
        markdown=formatted,
        thinking_process=f"Processed PDF: {pdf_path.name}",
        transcript=None,
        ocr_results=ocr_results,
        metadata=metadata,
    )
```

**Step 5: Add image processing method**

Edit `src/notely/core.py`, add method:
```python
async def _process_image(
    self,
    image_path: Path,
    metadata: dict[str, Any] | None = None,
) -> NotelyResult:
    """
    Process image file using GLM-OCR.

    Args:
        image_path: Path to image file
        metadata: Optional metadata

    Returns:
        NotelyResult with generated notes
    """
    metadata = metadata or {}

    logger.info(f"Processing image: {image_path}")

    # OCR image
    ocr_result = self.ocr.recognize(image_path)

    # Generate notes
    markdown = await self.enhancer.process(
        transcript=None,
        ocr_results=[ocr_result],
        metadata=metadata,
    )

    # Format
    formatted = self._formatter.beautify(markdown)

    return NotelyResult(
        markdown=formatted,
        thinking_process=f"Processed image: {image_path.name}",
        transcript=None,
        ocr_results=[ocr_result],
        metadata=metadata,
    )
```

**Step 6: Update process() method to handle PDF and images**

Edit `src/notely/core.py`, update `process()`:
```python
async def process(
    self,
    input_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> NotelyResult:
    """
    Process input file and generate notes.

    Supports: audio, video, PDF, images

    Args:
        input_path: Path to input file
        metadata: Optional metadata

    Returns:
        NotelyResult with generated notes

    Raises:
        ValueError: If file type is not supported
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    # Audio files
    if suffix in [".wav", ".mp3", ".m4a"]:
        return await self._process_audio(input_path, metadata)

    # Video files
    elif suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        return await self._process_video(input_path, metadata)

    # PDF files
    elif suffix == ".pdf":
        return await self._process_pdf(input_path, metadata)

    # Image files
    elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        return await self._process_image(input_path, metadata)

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: audio (.wav, .mp3, .m4a), "
            f"video (.mp4, .avi, .mov, .mkv), "
            f"PDF (.pdf), images (.jpg, .jpeg, .png, .bmp)"
        )
```

**Step 7: Write test for image processing**

Create `tests/test_core_image.py`:
```python
"""Tests for image processing in core."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from notely import Notely, NotelyConfig
from notely.config import OCRConfig, EnhancerConfig, LLMConfig


@pytest.mark.asyncio
@patch("notely.ocr.glm.ZhipuAI")
@patch("notely.llm.zhipu.ZhipuAI")
async def test_process_image(mock_llm_zhipu, mock_ocr_zhipu, tmp_path):
    """Test end-to-end image processing."""
    # Mock OCR response
    mock_ocr_response = Mock()
    mock_ocr_response.choices = [Mock()]
    mock_ocr_response.choices[0].message.content = "# Image Text"

    mock_ocr_client = Mock()
    mock_ocr_client.chat.completions.create.return_value = mock_ocr_response
    mock_ocr_zhipu.return_value = mock_ocr_client

    # Mock LLM response
    mock_llm_response = Mock()
    mock_llm_response.choices = [Mock()]
    mock_llm_response.choices[0].message.content = "# Enhanced"

    mock_llm_client = Mock()
    mock_llm_client.chat.completions.create.return_value = mock_llm_response
    mock_llm_zhipu.return_value = mock_llm_client

    # Create test image
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake image data")

    # Configure Notely with Zhipu
    config = NotelyConfig(
        ocr=OCRConfig(
            provider="zhipu",
            api_key="test-key"
        ),
        enhancer=EnhancerConfig(
            llm=LLMConfig(
                provider="zhipu",
                api_key="test-key"
            )
        )
    )

    notely = Notely(config)
    result = await notely.process(test_image)

    assert result.markdown
    assert result.transcript is None
    assert len(result.ocr_results) > 0


@pytest.mark.asyncio
async def test_process_unsupported_format(tmp_path):
    """Test that unsupported formats raise ValueError."""
    config = NotelyConfig()
    notely = Notely(config)

    # Create unsupported file
    test_file = tmp_path / "test.xyz"
    test_file.write_text("test")

    with pytest.raises(ValueError, match="Unsupported file type"):
        await notely.process(test_file)
```

**Step 8: Run all new tests**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/test_core_pdf.py tests/test_core_image.py -v`

Expected: All tests PASS

**Step 9: Commit**

Run:
```bash
cd /Users/didi/coding/python/notely
git add src/notely/core.py tests/test_core_pdf.py tests/test_core_image.py
git commit -m "feat(core): add PDF and image processing support

- Add _process_pdf() method for PDF files
- Add _process_image() method for image files
- Update process() to route .pdf and image extensions
- Add comprehensive tests with mocked API
- Support .jpg, .jpeg, .png, .bmp formats"
```

---

## Task 6: Run All Tests

**Step 1: Run full test suite with coverage**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/ -v --cov=src/notely --cov-report=term-missing`

Expected: All tests PASS, coverage >80%

**Step 2: Fix any failing tests**

If tests fail, debug and fix before proceeding.

---

## Task 7: Code Quality Checks

**Step 1: Format code**

Run: `cd /Users/didi/coding/python/notely && uv run ruff format .`

Expected: Files formatted

**Step 2: Check linting**

Run: `cd /Users/didi/coding/python/notely && uv run ruff check .`

Expected: No errors

**Step 3: Type checking**

Run: `cd /Users/didi/coding/python/notely && uv run mypy src/notely --strict`

Expected: No errors

**Step 4: Fix any issues**

If any checks fail, fix issues and re-run.

**Step 5: Commit formatting changes**

Run:
```bash
cd /Users/didi/coding/python/notely
git add .
git commit -m "style: format code with ruff"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `README.md`

**Step 1: Update supported formats section**

Edit `README.md`, update supported formats:
```markdown
## Supported Input Formats

- **Audio**: `.wav`, `.mp3`, `.m4a` → Transcribed with FunASR
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv` → Audio + Frame OCR
- **PDF**: `.pdf` → OCR with GLM-4V
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp` → OCR with GLM-4V
```

**Step 2: Add configuration example**

Edit `README.md`, add example:
```markdown
## Configuration

### Using Zhipu AI (GLM-OCR)

```yaml
ocr:
  provider: zhipu
  model: glm-4v-flash
  api_key: your-zhipu-api-key
  base_url: https://open.bigmodel.cn/api/paas/v4

llm:
  provider: zhipu
  model: glm-4
  api_key: your-zhipu-api-key
  base_url: https://open.bigmodel.cn/api/paas/v4
  temperature: 0.7
  max_tokens: 4096
```

### Using PaddleOCR (Default)

```yaml
ocr:
  provider: paddleocr
  language: ch
  use_gpu: true
```
```

**Step 3: Commit documentation**

Run:
```bash
cd /Users/didi/coding/python/notely
git add README.md
git commit -m "docs: update README with GLM-OCR and PDF/image support

- Add GLM-OCR as OCR provider option
- Document PDF and image processing support
- Add Zhipu AI configuration examples
- Update supported formats list"
```

---

## Final Steps

**Step 1: Run full test suite one more time**

Run: `cd /Users/didi/coding/python/notely && uv run pytest tests/ -v`

Expected: All tests PASS

**Step 2: Check git status**

Run: `cd /Users/didi/coding/python/notely && git status`

Expected: Working tree clean

**Step 3: Push feature branch**

Run:
```bash
cd /Users/didi/coding/python/notely
git push -u origin feature/glm-ocr-integration
```

---

## Summary

This implementation adds:
1. ✅ GLM-OCR API integration via `zhipuai` SDK
2. ✅ Zhipu LLM backend for enhancement
3. ✅ PDF and image document processing
4. ✅ Provider/model/api_key configuration pattern
5. ✅ Auto-detect GPU for ASR
6. ✅ Comprehensive tests with >80% coverage
7. ✅ Updated documentation

All code follows strict open source standards:
- English comments and docstrings
- Google-style docstrings
- Strict type hints
- Ruff formatting
- mypy type checking
- TDD workflow
