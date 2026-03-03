"""
Note templates for generating beautiful, structured Markdown notes.

Templates are now stored as Markdown files with YAML frontmatter for better
maintainability and version control. This module provides the NoteTemplate
class and factory functions for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .loader import load_template as _load_template


@dataclass
class NoteTemplate:
    """
    Template for generating structured notes.

    This template controls how the LLM transforms raw transcript
    and OCR data into beautiful, readable Markdown.

    Templates are loaded from Markdown files in the templates/ directory.
    The LLM will automatically detect the input language and generate notes
    in the same language.
    """

    name: str = "default"
    style: str = "academic"  # "academic", "casual", "technical"

    # Whether to include timestamps
    include_timestamps: bool = False

    # Whether to include original transcript
    include_transcript: bool = False

    # Custom sections to include
    custom_sections: list[str] = field(default_factory=list)

    # Loaded template content (lazy-loaded)
    _system_prompt: str = ""
    _instructions: str = ""
    _example: str = ""
    _extract_topics_prompt: str = ""
    _generate_topic_notes_prompt: str = ""
    _combine_notes_header: str = ""
    _combine_notes_footer: str = ""
    _loaded: bool = False

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        if not self._loaded:
            self._load_from_file()
        return self._system_prompt

    @property
    def instructions(self) -> str:
        """Get the formatting instructions."""
        if not self._loaded:
            self._load_from_file()
        return self._instructions

    @property
    def extract_topics_prompt(self) -> str:
        """Get the prompt for extracting topics (two-pass strategy)."""
        if not self._loaded:
            self._load_from_file()
        return self._extract_topics_prompt

    @property
    def generate_topic_notes_prompt(self) -> str:
        """Get the prompt for generating notes for a single topic (two-pass strategy)."""
        if not self._loaded:
            self._load_from_file()
        return self._generate_topic_notes_prompt

    @property
    def combine_notes_header(self) -> str:
        """Get the header template for combining notes (two-pass strategy)."""
        if not self._loaded:
            self._load_from_file()
        return self._combine_notes_header

    @property
    def combine_notes_footer(self) -> str:
        """Get the footer template for combining notes (two-pass strategy)."""
        if not self._loaded:
            self._load_from_file()
        return self._combine_notes_footer

    def get_instructions(self) -> str:
        """Get specific formatting instructions (backward compatibility)."""
        return self.instructions

    def _load_from_file(self):
        """Load template content from Markdown file."""
        template_name = self.name

        try:
            data = _load_template(template_name)
        except FileNotFoundError:
            # Fallback to default template if specific template not found
            if template_name != "default":
                data = _load_template("default")
            else:
                raise

        # Load all fields from data
        self._system_prompt = data["system_prompt"]
        self._instructions = data["instructions"]
        self._example = data.get("example", "")
        self._extract_topics_prompt = data.get("extract_topics_prompt", "")
        self._generate_topic_notes_prompt = data.get("generate_topic_notes_prompt", "")
        self._combine_notes_header = data.get("combine_notes_header", "")
        self._combine_notes_footer = data.get("combine_notes_footer", "")
        self._loaded = True


def get_default_template() -> NoteTemplate:
    """Get the default note template."""
    return NoteTemplate(
        name="default",
        style="academic",
        include_timestamps=False,
        include_transcript=False,
    )


def get_academic_template() -> NoteTemplate:
    """Get academic-style template."""
    return NoteTemplate(
        name="academic",
        style="academic",
        include_timestamps=False,
        include_transcript=False,
        custom_sections=["References", "Further Reading"],
    )


def get_technical_template() -> NoteTemplate:
    """Get technical-style template for programming/tech content."""
    return NoteTemplate(
        name="technical",
        style="technical",
        include_timestamps=False,
        include_transcript=False,
        custom_sections=["Code Examples", "API Reference"],
    )


def get_meeting_template() -> NoteTemplate:
    """Get meeting notes template."""
    return NoteTemplate(
        name="meeting",
        style="casual",
        include_timestamps=True,
        include_transcript=False,
        custom_sections=["Action Items", "Decisions"],
    )


def get_template(name: str = "default", **kwargs) -> NoteTemplate:
    """
    Get a template by name.

    Args:
        name: Template name ("default", "academic", "technical", "meeting").
        **kwargs: Additional template parameters (for backward compatibility).

    Returns:
        NoteTemplate instance.
    """
    if name == "academic":
        return get_academic_template()
    elif name == "technical":
        return get_technical_template()
    elif name == "meeting":
        return get_meeting_template()
    return get_default_template()
