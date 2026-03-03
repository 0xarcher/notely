"""
Prompt templates for note generation.
"""

from notely.prompts.loader import load_template
from notely.prompts.templates import (
    NoteTemplate,
    get_academic_template,
    get_default_template,
    get_meeting_template,
    get_technical_template,
    get_template,
)

__all__ = [
    "NoteTemplate",
    "get_academic_template",
    "get_default_template",
    "get_meeting_template",
    "get_technical_template",
    "get_template",
    "load_template",
]
