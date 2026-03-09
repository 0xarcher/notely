# Document Support Design

**Date**: 2026-03-09
**Status**: Approved
**Author**: Archer

## Overview

Extend Notely from a "video/audio note-taking tool" to a "universal document note-taking tool" by adding support for multiple document formats using MinerU.

## Goals

1. Support PDF, Office documents (DOC/DOCX/PPT/PPTX), eBooks (EPUB/MOBI), and web pages (HTML)
2. Maintain existing video/audio processing pipeline
3. Minimize code changes and preserve architectural consistency
4. Follow project code standards (Ruff, mypy, pytest)

## Supported Formats

### Current (Media Files)
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`
- Audio: `.wav`, `.mp3`, `.m4a`

### New (Document Files)
- PDF: `.pdf`
- Office: `.doc`, `.docx`, `.ppt`, `.pptx`
- eBooks: `.epub`, `.mobi`
- Web: `.html`, `.htm`

## Architecture

### Processing Flow

```
Input File
  ↓
File Type Detection (by extension)
  ↓
┌─────────────┬──────────────┬─────────────┐
│ Video/Audio │     PDF      │   Other     │
│  (existing) │    (new)     │   (new)     │
└─────────────┴──────────────┴─────────────┘
      ↓              ↓              ↓
  ASR+OCR      MinerU Extract  MinerU Extract
      ↓              ↓              ↓
   Enhancer      Skip Enhance    Enhancer
      ↓              ↓              ↓
   Format         Format         Format
      ↓              ↓              ↓
    Output         Output         Output
```

### Key Decisions

1. **PDF Processing**: Extract with MinerU, skip enhancement (PDFs are already complete documents)
2. **Other Documents**: Extract with MinerU, apply enhancement (slides/web pages need context)
3. **Media Files**: Keep existing ASR → OCR → Enhancer pipeline unchanged

## Implementation

### 1. New Module: `utils/document.py`

Parallel to existing `utils/audio.py` and `utils/video.py`:

```python
"""Document extraction utilities using MinerU."""

from pathlib import Path
from dataclasses import dataclass

@dataclass
class DocumentContent:
    """Extracted document content."""
    markdown: str
    metadata: dict

def extract_document(file_path: Path) -> DocumentContent:
    """
    Extract content from any document using MinerU.

    Supports: PDF, DOC/DOCX, PPT/PPTX, EPUB, MOBI, HTML
    """
    from magic_pdf.pipe.UNIPipe import UNIPipe

    pipe = UNIPipe(str(file_path))
    pipe.pipe_classify()
    pipe.pipe_parse()

    return DocumentContent(
        markdown=pipe.get_markdown(),
        metadata=pipe.get_metadata()
    )
```

### 2. Core Logic Extension: `core.py`

Add document processing branch to `Notely.process()`:

```python
async def process(
    self,
    input_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> NotelyResult:
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    # Media files (existing flow)
    if suffix in [".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".m4a"]:
        return await self._process_media(input_path, metadata)

    # Document files (new flow)
    else:
        return await self._process_document(input_path, metadata)

async def _process_document(self, input_path: Path, metadata: dict) -> NotelyResult:
    """Process document files (PDF, Office, eBooks, HTML)."""
    from notely.utils.document import extract_document

    logger.info(f"Processing document: {input_path}")

    # Step 1: Extract document content
    doc_content = extract_document(input_path)
    logger.info(f"✓ Extracted {len(doc_content.markdown)} characters")

    # Step 2: Conditional enhancement
    is_pdf = input_path.suffix.lower() == ".pdf"

    if is_pdf:
        logger.info("PDF detected, skipping enhancement")
        markdown = doc_content.markdown
    else:
        logger.info("Enhancing document content...")
        markdown = await self.enhancer.process_text(doc_content.markdown)
        logger.info("✓ Enhancement completed")

    # Step 3: Format
    formatted = self._formatter.beautify(markdown)

    return NotelyResult(
        markdown=formatted,
        thinking_process=f"Document processed: {input_path.name}",
        transcript=None,
        ocr_results=[],
        metadata={**doc_content.metadata, **(metadata or {})}
    )
```

### 3. Model Updates: `models.py`

Make transcript and ocr_results optional for document processing:

```python
@dataclass
class NotelyResult:
    markdown: str
    thinking_process: str
    transcript: ASRResult | None = None  # Optional for documents
    ocr_results: list[OCRResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4. Dependencies: `pyproject.toml`

Add MinerU dependency:

```toml
dependencies = [
    # ... existing dependencies ...
    "magic-pdf>=0.7.0",  # MinerU PDF/document extraction
]
```

## Testing Strategy

### Unit Tests (`tests/test_document.py`)

```python
def test_extract_pdf():
    """Test PDF extraction."""
    result = extract_document(Path("tests/fixtures/sample.pdf"))
    assert isinstance(result, DocumentContent)
    assert len(result.markdown) > 0

def test_extract_docx():
    """Test DOCX extraction."""
    result = extract_document(Path("tests/fixtures/sample.docx"))
    assert isinstance(result, DocumentContent)
```

### Integration Tests (`tests/test_core_document.py`)

```python
@pytest.mark.asyncio
async def test_process_pdf(notely_instance):
    """Test PDF processing end-to-end."""
    result = await notely_instance.process("tests/fixtures/sample.pdf")
    assert result.markdown
    assert result.transcript is None

@pytest.mark.asyncio
async def test_process_pptx(notely_instance):
    """Test PPTX processing with enhancement."""
    result = await notely_instance.process("tests/fixtures/slides.pptx")
    assert result.markdown
```

### Test Fixtures

Required test files:
- `tests/fixtures/sample.pdf`
- `tests/fixtures/sample.docx`
- `tests/fixtures/slides.pptx`

### CI Checks

All changes must pass:
1. `ruff format --check .`
2. `ruff check .`
3. `mypy src/notely`
4. `pytest tests/ --cov=src/notely`

## Migration Guide

### For Users

No breaking changes. Existing code continues to work:

```python
# Existing usage (unchanged)
result = await notely.process("lecture.mp4")

# New usage (documents)
result = await notely.process("slides.pdf")
result = await notely.process("notes.docx")
```

### For Developers

- `NotelyResult.transcript` is now optional (`ASRResult | None`)
- `NotelyResult.ocr_results` defaults to empty list
- Check for `None` when accessing transcript in custom code

## Risks and Mitigations

### Risk 1: MinerU Dependency Size
- **Risk**: MinerU may have large dependencies
- **Mitigation**: Make it optional dependency, document installation

### Risk 2: PDF Enhancement Quality
- **Risk**: Users may want PDF enhancement
- **Mitigation**: Add configuration option in future if needed

### Risk 3: Unsupported Formats
- **Risk**: MinerU may not support all claimed formats
- **Mitigation**: Add format validation and clear error messages

## Future Enhancements

1. Add configuration option to enable/disable enhancement per format
2. Support image files (PNG/JPG) with OCR
3. Add progress callbacks for large documents
4. Support batch processing of multiple documents

## References

- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [magic-pdf PyPI](https://pypi.org/project/magic-pdf/)
- [MinerU Documentation](https://opendatalab.github.io/MinerU/)
