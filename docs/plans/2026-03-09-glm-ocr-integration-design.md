# GLM-OCR Integration Design

**Date**: 2026-03-09
**Status**: Approved
**Author**: Archer

## Overview

Integrate Zhipu AI's GLM-OCR API to replace PaddleOCR and add PDF/image document support to Notely. This enables unified OCR processing through cloud API while maintaining backward compatibility.

## Goals

1. Replace PaddleOCR with GLM-OCR API for video frame OCR
2. Add PDF document processing support
3. Add image file processing support
4. Support all Zhipu AI services (OCR + LLM) through unified configuration
5. Maintain backward compatibility with existing PaddleOCR users
6. Auto-detect GPU availability for ASR (remove manual device configuration)

## Supported Input Types

### Current (Unchanged)
- **Audio**: `.wav`, `.mp3`, `.m4a` → FunASR

### Modified
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv` → FunASR + GLM-OCR (frames)

### New
- **PDF**: `.pdf` → GLM-OCR API
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp` → GLM-OCR API

## Architecture

### Processing Flow

```
Input File
  ↓
File Type Detection (by extension)
  ↓
┌─────────┬─────────┬─────────┬─────────┐
│  Audio  │  Video  │   PDF   │  Image  │
└─────────┴─────────┴─────────┴─────────┘
     ↓         ↓         ↓         ↓
  FunASR   FunASR+   GLM-OCR   GLM-OCR
           GLM-OCR     API       API
     ↓         ↓         ↓         ↓
 Enhancer  Enhancer  Enhancer  Enhancer
     ↓         ↓         ↓         ↓
  Output    Output    Output    Output
```

### Key Design Decisions

1. **OCR Provider Abstraction**: Keep `OCRBackend` interface, add `GLMOCRBackend` implementation
2. **Configuration by Function**: Configure by module (OCR/LLM/ASR), not by service provider
3. **Backward Compatibility**: PaddleOCR remains available, users choose via `provider` config
4. **Auto-detect GPU**: ASR automatically detects CUDA availability, no manual `device` config
5. **Cloud-First**: GLM-OCR uses API, supports future local deployment upgrade

## Implementation

### 1. Configuration Updates

**Updated `OCRConfig`** (`src/notely/config.py`):
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

**Updated `LLMConfig`** (`src/notely/config.py`):
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

**YAML Configuration Example** (All Zhipu AI):
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

asr:
  backend: funasr
  model: paraformer-zh
  # device removed - auto-detected
```

### 2. New OCR Backend

**GLMOCRBackend** (`src/notely/ocr/glm.py`):
```python
"""GLM-OCR backend using Zhipu AI API."""

from __future__ import annotations

import base64
from pathlib import Path

from notely.config import OCRConfig
from notely.ocr.base import OCRBackend, OCRResult, TextBlock


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
            base_url=config.base_url or "https://open.bigmodel.cn/api/paas/v4"
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
        """
        image_path = Path(image_path)

        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

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
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            },
                            {
                                "type": "text",
                                "text": "Extract all text from this image in Markdown format."
                            }
                        ]
                    }
                ]
            )

            text = response.choices[0].message.content

            # Parse response to OCRResult
            return OCRResult(
                text_blocks=[
                    TextBlock(
                        text=text,
                        confidence=1.0,
                        bbox=(0, 0, 0, 0),
                        block_type="text"
                    )
                ],
                source_path=str(image_path)
            )

        except Exception as e:
            raise RuntimeError(f"GLM-OCR API call failed: {e}") from e

    def recognize_pdf(self, pdf_path: Path | str) -> list[OCRResult]:
        """
        Recognize text in a PDF using GLM-4V API.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of OCRResult, one per page

        Raises:
            RuntimeError: If API call fails
        """
        # Similar implementation for PDF
        # GLM-4V can process PDF directly
        pass

    def is_available(self) -> bool:
        """Check if GLM-OCR backend is available."""
        try:
            # Test API connection
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
```

### 3. New LLM Backend

**ZhipuLLMBackend** (`src/notely/llm/zhipu.py`):
```python
"""Zhipu AI LLM backend."""

from __future__ import annotations

from notely.config import LLMConfig
from notely.llm.base import LLMBackend


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
            base_url=config.base_url or "https://open.bigmodel.cn/api/paas/v4"
        )
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def generate(self, prompt: str) -> str:
        """Generate text using Zhipu AI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
```

### 4. Core Processing Updates

**Updated `core.py`**:
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

async def _process_pdf(
    self,
    pdf_path: Path,
    metadata: dict[str, Any] | None = None,
) -> NotelyResult:
    """Process PDF file using GLM-OCR."""
    metadata = metadata or {}

    logger.info(f"Processing PDF: {pdf_path}")

    # OCR PDF pages
    ocr_results = self.ocr.recognize_pdf(pdf_path)

    # Generate notes with enhancer
    markdown = await self.enhancer.process(
        transcript=None,
        ocr_results=ocr_results,
        metadata=metadata
    )

    # Format
    formatted = self._formatter.beautify(markdown)

    return NotelyResult(
        markdown=formatted,
        thinking_process=f"Processed PDF: {pdf_path.name}",
        transcript=None,
        ocr_results=ocr_results,
        metadata=metadata
    )

async def _process_image(
    self,
    image_path: Path,
    metadata: dict[str, Any] | None = None,
) -> NotelyResult:
    """Process image file using GLM-OCR."""
    metadata = metadata or {}

    logger.info(f"Processing image: {image_path}")

    # OCR image
    ocr_result = self.ocr.recognize(image_path)

    # Generate notes
    markdown = await self.enhancer.process(
        transcript=None,
        ocr_results=[ocr_result],
        metadata=metadata
    )

    # Format
    formatted = self._formatter.beautify(markdown)

    return NotelyResult(
        markdown=formatted,
        thinking_process=f"Processed image: {image_path.name}",
        transcript=None,
        ocr_results=[ocr_result],
        metadata=metadata
    )
```

### 5. ASR Auto-detect GPU

**Updated `FunASRBackend`** (`src/notely/asr/funasr.py`):
```python
def __init__(self, config: ASRConfig) -> None:
    """
    Initialize FunASR backend with auto-detected device.

    Args:
        config: ASR configuration
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    logger.info(f"FunASR using device: {device}")

    # Initialize model with detected device
    # ...
```

**Remove `device` from `ASRConfig`**:
```python
@dataclass
class ASRConfig:
    """ASR configuration."""

    backend: str = "funasr"
    model: str = "paraformer-zh"
    language: str = "zh"
    # device removed - auto-detected
```

## Error Handling

### API Key Validation
```python
if config.provider == "zhipu" and not config.api_key:
    raise ValueError("API key required for Zhipu provider")
```

### API Call Failures
```python
try:
    response = self.client.chat.completions.create(...)
except Exception as e:
    logger.error(f"GLM-OCR API call failed: {e}")
    raise RuntimeError(f"OCR failed: {e}") from e
```

### Unsupported File Types
```python
if suffix not in SUPPORTED_FORMATS:
    raise ValueError(
        f"Unsupported file type: {suffix}. "
        f"Supported: {', '.join(SUPPORTED_FORMATS)}"
    )
```

## Testing Strategy

### Unit Tests

**`tests/test_ocr_glm.py`**:
```python
"""Tests for GLM-OCR backend."""

import pytest
from unittest.mock import Mock, patch

from notely.config import OCRConfig
from notely.ocr.glm import GLMOCRBackend


def test_glm_ocr_requires_api_key():
    """Test that GLM-OCR requires API key."""
    config = OCRConfig(provider="zhipu", api_key="")

    with pytest.raises(ValueError, match="API key required"):
        GLMOCRBackend(config)


@patch("notely.ocr.glm.ZhipuAI")
def test_glm_ocr_recognize(mock_zhipu):
    """Test image recognition."""
    # Mock API response
    mock_response = Mock()
    mock_response.choices[0].message.content = "# Test\nExtracted text"
    mock_zhipu.return_value.chat.completions.create.return_value = mock_response

    config = OCRConfig(provider="zhipu", api_key="test-key")
    backend = GLMOCRBackend(config)

    result = backend.recognize("test.jpg")

    assert result.full_text == "# Test\nExtracted text"
    assert len(result.text_blocks) == 1
```

### Integration Tests

**`tests/test_core_pdf.py`**:
```python
"""Integration tests for PDF processing."""

import pytest

from notely import Notely, NotelyConfig
from notely.config import OCRConfig, EnhancerConfig, LLMConfig


@pytest.mark.asyncio
async def test_process_pdf():
    """Test end-to-end PDF processing."""
    config = NotelyConfig(
        ocr=OCRConfig(provider="zhipu", api_key="test-key"),
        enhancer=EnhancerConfig(
            llm=LLMConfig(provider="zhipu", api_key="test-key")
        )
    )

    notely = Notely(config)
    result = await notely.process("test.pdf")

    assert result.markdown
    assert result.transcript is None
    assert len(result.ocr_results) > 0
```

### Code Quality

- **Ruff**: Format and lint all code
- **mypy**: Strict type checking
- **pytest**: >80% coverage
- **Docstrings**: Google style, English only
- **Comments**: English only

## Migration Guide

### From PaddleOCR to GLM-OCR

**Before**:
```yaml
ocr:
  backend: paddleocr
  language: ch
  use_gpu: true
```

**After**:
```yaml
ocr:
  provider: zhipu
  model: glm-4v-flash
  api_key: your-zhipu-api-key
  base_url: https://open.bigmodel.cn/api/paas/v4
```

### Backward Compatibility

- Existing PaddleOCR configurations continue to work
- No breaking changes to public APIs
- New features are opt-in via configuration

## Dependencies

**Add to `pyproject.toml`**:
```toml
dependencies = [
    # ... existing dependencies ...
    "zhipuai>=2.0.0",  # Zhipu AI official SDK
]
```

## Documentation Updates

1. **README.md**:
   - Add GLM-OCR as OCR provider option
   - Add PDF/image processing examples
   - Update supported formats table

2. **Configuration Guide**:
   - Document OCR providers (paddleocr, zhipu)
   - Document LLM providers (openai, zhipu, anthropic)
   - Add Zhipu AI setup instructions

3. **API Reference**:
   - Document `GLMOCRBackend`
   - Document `ZhipuLLMBackend`
   - Update `OCRConfig` and `LLMConfig` documentation

## Code Standards

All code must follow these standards:

- **Language**: All comments and docstrings in English
- **Style**: Google-style docstrings
- **Types**: Type hints for all public APIs
- **Format**: Ruff-compliant formatting
- **Lint**: Pass `ruff check` with no errors
- **Type Check**: Pass `mypy --strict`
- **Tests**: >80% coverage for new code
- **Commits**: Conventional commits format

## Future Enhancements

1. **Local GLM-OCR Deployment**: Support running GLM-OCR model locally
2. **Batch Processing**: Process multiple files in parallel
3. **Streaming**: Stream OCR results for large files
4. **Caching**: Cache OCR results to reduce API calls
5. **More Formats**: Add support for DOCX, PPTX, etc.

## References

- [Zhipu AI Official Website](https://open.bigmodel.cn)
- [GLM-OCR GitHub](https://github.com/zai-org/GLM-OCR)
- [Zhipu AI Python SDK](https://pypi.org/project/zhipuai/)
