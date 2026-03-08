"""Unit tests for data models."""

import tempfile
from pathlib import Path

from notely.asr.base import ASRResult, TranscriptSegment
from notely.models import NotelyResult


def test_notely_result_creation():
    """Test creating NotelyResult."""
    transcript = ASRResult(segments=[TranscriptSegment(text="Test", start_time=0.0, end_time=1.0)])

    result = NotelyResult(
        markdown="# Test Note",
        thinking_process="Test process",
        transcript=transcript,
        ocr_results=[],
        metadata={"title": "Test"},
    )

    assert result.markdown == "# Test Note"
    assert result.metadata["title"] == "Test"
    assert len(result.transcript.segments) == 1


def test_notely_result_save():
    """Test saving NotelyResult to file."""
    transcript = ASRResult(segments=[TranscriptSegment(text="Test", start_time=0.0, end_time=1.0)])

    result = NotelyResult(
        markdown="# Test Note\n\nContent here",
        thinking_process="Process",
        transcript=transcript,
        ocr_results=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_note.md"
        result.save(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert content == "# Test Note\n\nContent here"
