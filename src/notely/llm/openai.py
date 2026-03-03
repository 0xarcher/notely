"""
OpenAI-compatible backend for LLM.
"""

from __future__ import annotations

from typing import Any, Union

from notely.asr.base import ASRResult
from notely.llm.base import LLMBackend, LLMResult
from notely.ocr.base import OCRResult
from notely.prompts import NoteTemplate


class OpenAIBackend(LLMBackend):
    """
    OpenAI-compatible LLM backend.

    Works with OpenAI, Azure OpenAI, and other compatible APIs.

    Args:
        base_url: API base URL.
        api_key: API key.
        model: Model name (e.g., "gpt-4o", "gpt-4-turbo").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 16384,  # Max for gpt-4o
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self) -> Any:
        """Get OpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Union[str, None] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate text from a prompt."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        return LLMResult(
            content=content,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            model=self.model,
        )

    def generate_notes(
        self,
        transcript: ASRResult,
        ocr_results: list[OCRResult],
        template: NoteTemplate,
        metadata: Union[dict[str, Any], None] = None,
    ) -> tuple[str, str]:
        """
        Generate structured notes from transcript and OCR results.

        Uses a two-pass approach for better content preservation:
        1. Extract topics from transcript
        2. Generate detailed notes for each topic

        Args:
            transcript: ASR transcript with timestamps.
            ocr_results: List of OCR results from slides/PDFs.
            template: Note template for formatting.
            metadata: Additional metadata (title, instructor, etc.)

        Returns:
            Tuple of (markdown_content, thinking_process).
        """
        metadata = metadata or {}

        # Build the context from transcript and OCR
        transcript_text = self._format_transcript(transcript)
        ocr_text = self._format_ocr_results(ocr_results)

        # Step 1: Extract topics from transcript
        topics = self._extract_topics(transcript_text, template)

        # Step 2: Generate detailed notes for each topic
        all_notes = []
        for topic in topics:
            note_section = self._generate_topic_notes(
                topic=topic,
                transcript_text=transcript_text,
                ocr_text=ocr_text,
                template=template,
            )
            all_notes.append(note_section)

        # Combine all notes
        markdown = self._combine_notes(all_notes, metadata, template)
        thinking = f"Extracted {len(topics)} topics: {', '.join(topics)}"

        return markdown, thinking

    def _extract_topics(self, transcript_text: str, template: NoteTemplate) -> list[str]:
        """Extract main topics from transcript."""
        extract_prompt = template.extract_topics_prompt.format(transcript=transcript_text[:15000])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content or ""
        # Parse topics - one per line
        topics = [line.strip() for line in content.strip().split("\n") if line.strip()]
        # Remove numbering if present
        topics = [t.lstrip("0123456789.-) ") for t in topics]
        return topics[:20]  # Limit to 20 topics

    def _generate_topic_notes(
        self,
        topic: str,
        transcript_text: str,
        ocr_text: str,
        template: NoteTemplate,
    ) -> str:
        """Generate detailed notes for a single topic."""
        topic_prompt = template.generate_topic_notes_prompt.format(
            topic=topic, transcript=transcript_text, ocr_text=ocr_text
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": topic_prompt}],
            temperature=self.temperature,
            max_tokens=4000,
        )

        return response.choices[0].message.content or ""

    def _combine_notes(
        self,
        all_notes: list[str],
        metadata: dict[str, Any],
        template: NoteTemplate,
    ) -> str:
        """Combine all topic notes into final markdown."""
        # Build header using template
        title = metadata.get("title", "课程笔记")

        # Extract key points from notes
        key_points = []
        for note in all_notes[:10]:
            lines = note.strip().split("\n")
            for line in lines:
                if line.startswith("##") and not line.startswith("###"):
                    topic = line.lstrip("# ").strip()
                    # Remove emoji for key points
                    import re

                    topic_clean = re.sub(r"[^\w\s\u4e00-\u9fff\-]", "", topic).strip()
                    if topic_clean:
                        key_points.append(f"- {topic_clean}")
                    break

        key_points_text = "\n".join(key_points[:8])

        # Use template header
        header = template.combine_notes_header.format(title=title, key_points=key_points_text)

        # Combine all notes
        body = "\n\n---\n\n".join(all_notes)

        # Fix LaTeX format: \(...\) -> $...$ and \[...\] -> $$...$$
        body = self._fix_latex_format(body)

        # Use template footer
        footer = template.combine_notes_footer

        return header + body + footer

    @staticmethod
    def _fix_latex_format(text: str) -> str:
        r"""Fix LaTeX format: \(...\) -> $...$ and \[...\] -> $$...$$"""
        import re

        # Fix \[...\] -> $$...$$ (display math)
        text = re.sub(r"\\\[([^]]*?)\\]", r"$$\1$$", text, flags=re.DOTALL)

        # Fix \(...\) -> $...$ (inline math)
        text = re.sub(r"\\\(([^)]*?)\\\)", r"$\1$", text)

        return text

    def _format_transcript(self, transcript: ASRResult) -> str:
        """Format transcript for the prompt."""
        # If we have detailed segments with timestamps, use them
        if len(transcript.segments) > 1:
            lines = []
            for seg in transcript.segments:
                if seg.start_time > 0:
                    time_marker = f"[{self._format_time(seg.start_time)}]"
                    lines.append(f"{time_marker} {seg.text}")
                else:
                    lines.append(seg.text)
            return "\n\n".join(lines)

        # Otherwise, use full_text directly (better for long transcripts)
        return transcript.full_text

    def _format_ocr_results(self, ocr_results: list[OCRResult]) -> str:
        """Format OCR results for the prompt."""
        if not ocr_results:
            return "（无幻灯片/PDF内容）"

        sections = []
        for result in ocr_results:
            if result.timestamp is not None:
                # Video frame
                time_marker = f"[{self._format_time(result.timestamp)}]"
                sections.append(f"\n### 幻灯片 {time_marker}\n{result.full_text}")
            elif result.page_number is not None:
                # PDF page
                sections.append(f"\n### PDF 第 {result.page_number} 页\n{result.full_text}")
            else:
                sections.append(f"\n### 图像内容\n{result.full_text}")

        return "\n".join(sections)

    @staticmethod
    def _build_user_prompt(
        transcript_text: str,
        ocr_text: str,
        template: NoteTemplate,
        metadata: dict[str, Any],
    ) -> str:
        """Build the user prompt with all context."""
        # Build metadata section
        meta_lines = []
        if title := metadata.get("title"):
            meta_lines.append(f"**课程标题**: {title}")
        if instructor := metadata.get("instructor"):
            meta_lines.append(f"**讲师**: {instructor}")
        if date := metadata.get("date"):
            meta_lines.append(f"**日期**: {date}")
        if duration := metadata.get("duration"):
            meta_lines.append(f"**时长**: {duration}")

        meta_section = "\n".join(meta_lines) if meta_lines else ""

        prompt = f"""
{meta_section}

---

## 📝 课程转录文本

{transcript_text}

---

## 📊 幻灯片/讲义内容

{ocr_text}

---

{template.get_instructions()}
"""
        return prompt.strip()

    def _parse_response(self, content: str) -> tuple[str, str]:
        """Parse thinking and markdown from response."""
        thinking = ""
        markdown = content

        # Check for thinking tag
        if "<thinking>" in content and "</thinking>" in content:
            start = content.find("<thinking>") + len("<thinking>")
            end = content.find("</thinking>")
            thinking = content[start:end].strip()
            markdown = content[end + len("</thinking>") :].strip()

        # Check for markdown tag
        if "<markdown>" in content and "</markdown>" in content:
            start = content.find("<markdown>") + len("<markdown>")
            end = content.find("</markdown>")
            markdown = content[start:end].strip()

        # Fix common formatting issues
        markdown = self._fix_formatting(markdown)

        return thinking, markdown

    @staticmethod
    def _fix_formatting(markdown: str) -> str:
        """Fix common formatting issues in LLM output."""
        import re

        # Fix "* *text**" -> "**text**" (common LLM formatting error)
        markdown = re.sub(r"\* \*([^*]+)\*\*", r"**\1**", markdown)

        # Fix "- --" -> "---" (separator)
        markdown = re.sub(r"^- --$", "---", markdown, flags=re.MULTILINE)

        # Fix "* *详细说明**" -> "**详细说明**"
        markdown = re.sub(r"\* \*([^*]+：)", r"**\1", markdown)

        return markdown

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return bool(self.api_key)
