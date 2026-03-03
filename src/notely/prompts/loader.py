"""
Template loader for Markdown-based prompt templates.

This module provides functionality to load and parse prompt templates
from Markdown files with YAML frontmatter.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import yaml


class TemplateLoader:
    """
    Loader for Markdown-based prompt templates.

    Templates are stored as Markdown files with YAML frontmatter containing
    metadata. The loader parses these files and extracts sections for use
    in prompt generation.

    Example template format:
        ---
        name: default
        language: zh
        style: academic
        ---

        # System Prompt
        You are an expert...

        # Instructions
        Please follow these steps...
    """

    def __init__(self, template_dir: Union[Path, None] = None):
        """
        Initialize the template loader.

        Args:
            template_dir: Directory containing template files.
                         If None, uses the default templates directory.
        """
        if template_dir is None:
            # Default to package's templates directory
            template_dir = Path(__file__).parent / "templates"
        self.template_dir = Path(template_dir)

    @lru_cache(maxsize=32)
    def load(self, name: str) -> dict[str, Any]:
        """
        Load a template file and return parsed content.

        Args:
            name: Template name (e.g., "default", "academic").

        Returns:
            Dictionary with keys:
                - metadata: Dict from YAML frontmatter
                - system_prompt: System prompt text
                - instructions: Instructions text
                - example: Example output (if present)

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        path = self.template_dir / f"{name}.md"

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")

        content = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        metadata = {}
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()

        # Parse sections (split by # headings)
        sections = self._parse_sections(body)

        # Get system prompt (may be dict with subsections)
        system_prompt_section = sections.get("System Prompt", "")
        if isinstance(system_prompt_section, dict):
            # Combine all subsections into one string
            system_prompt = "You are an expert note-taking assistant. Transform lecture transcripts, videos, and educational materials into comprehensive, well-structured Markdown notes.\n\n"
            for subsection_name, subsection_content in system_prompt_section.items():
                system_prompt += f"## {subsection_name}\n\n{subsection_content}\n\n"
            system_prompt = system_prompt.strip()
        else:
            system_prompt = system_prompt_section

        # Extract two-pass strategy prompts
        two_pass = sections.get("Two-Pass Generation Strategy", {})
        if isinstance(two_pass, dict):
            extract_topics = two_pass.get("Extract Topics Prompt", "")
            generate_topic_notes = two_pass.get("Generate Topic Notes Prompt", "")
            combine_header = two_pass.get("Combine Notes Header Template", "")
            combine_footer = two_pass.get("Combine Notes Footer Template", "")

            # Extract content from code blocks if present
            combine_header = self._extract_from_code_block(combine_header)
            combine_footer = self._extract_from_code_block(combine_footer)
        else:
            extract_topics = ""
            generate_topic_notes = ""
            combine_header = ""
            combine_footer = ""

        return {
            "metadata": metadata,
            "system_prompt": system_prompt,
            "instructions": sections.get("Instructions", ""),
            "example": sections.get("Example Output", ""),
            "extract_topics_prompt": extract_topics,
            "generate_topic_notes_prompt": generate_topic_notes,
            "combine_notes_header": combine_header,
            "combine_notes_footer": combine_footer,
        }

    @staticmethod
    def _parse_sections(body: str) -> dict[str, str]:
        """
        Parse Markdown sections from template body.

        Sections are identified by level-1 headings (# Section Name).
        Subsections are identified by level-2 headings (## Subsection Name).

        Args:
            body: Template body text.

        Returns:
            Dictionary mapping section names to content.
            For sections with subsections, returns nested dict.
        """
        sections = {}
        current_section = None
        current_subsection = None
        current_content = []
        in_code_block = False

        for line in body.split("\n"):
            # Track code block state
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # Only process headings if not in code block
            if not in_code_block:
                if line.startswith("# ") and not line.startswith("## "):
                    # Save previous content
                    if current_section:
                        if current_subsection:
                            # Save subsection
                            if current_section not in sections:
                                sections[current_section] = {}
                            sections[current_section][current_subsection] = "\n".join(
                                current_content
                            ).strip()
                        else:
                            # Save section
                            sections[current_section] = "\n".join(current_content).strip()

                    # Start new section
                    current_section = line[2:].strip()
                    current_subsection = None
                    current_content = []
                    continue
                elif line.startswith("## "):
                    # Save previous subsection
                    if current_subsection and current_section:
                        if not isinstance(sections.get(current_section), dict):
                            sections[current_section] = {}
                        sections[current_section][current_subsection] = "\n".join(
                            current_content
                        ).strip()

                    # Start new subsection
                    current_subsection = line[3:].strip()
                    current_content = []
                    continue

            # Add line to current content
            current_content.append(line)

        # Save last content
        if current_section:
            if current_subsection:
                if not isinstance(sections.get(current_section), dict):
                    sections[current_section] = {}
                sections[current_section][current_subsection] = "\n".join(current_content).strip()
            else:
                sections[current_section] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def _extract_from_code_block(text: str) -> str:
        """
        Extract content from markdown code block if present.

        Args:
            text: Text that may contain a code block.

        Returns:
            Content inside code block, or original text if no code block.
        """
        if not text:
            return text

        # Match ```markdown ... ``` or ``` ... ```
        import re

        match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip()


# Global loader instance
_loader = TemplateLoader()


def load_template(name: str) -> dict[str, Any]:
    """
    Load a template using the global loader.

    This is a convenience function for the most common use case.

    Args:
        name: Template name.

    Returns:
        Parsed template dictionary.
    """
    return _loader.load(name)
