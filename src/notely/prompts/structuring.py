"""Structuring prompts for organizing comprehension results into notes."""

from notely.prompts.registry import PromptRegistry

# Optimized structuring prompt with emphasis on detail preservation
STRUCTURING_PROMPT = """Organize comprehension results into structured notes with MAXIMUM detail preservation.

## Task
Integrate multiple comprehension results into a cohesive, well-structured note:
1. Generate executive summary (2-3 paragraphs, comprehensive)
2. Organize content by topic (not chronological order)
3. Merge related concepts across chunks WITHOUT losing details
4. Create clear hierarchical structure with detailed subsections
5. Add appropriate emoji icons for sections

## Critical Requirements
- PRESERVE ALL details from comprehension results - do NOT compress
- Each section should be comprehensive (minimum 200 words per major section)
- Include ALL examples, explanations, and technical details
- Expand and clarify rather than summarize
- If input has 10 concepts, output should have 10 concepts with full explanations

## Input

### Comprehension Results
{combined_summaries}

### Key Concepts ({concept_count} total)
{concepts_list}

### Examples ({example_count} total)
{examples_list}

### Cross-Chunk Patterns
{cross_chunk_hints}

## Output Format (JSON)
{{
    "title": "Descriptive note title",
    "summary": "Comprehensive executive summary (2-3 paragraphs, 150+ words)",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "sections": [
        {{
            "title": "Section Title",
            "emoji": "📚",
            "content": "Detailed section content in Markdown format. Minimum 200 words. Include ALL relevant details, examples, and explanations.",
            "subsections": [
                {{
                    "title": "Subsection Title",
                    "content": "Detailed subsection content with examples and explanations"
                }}
            ]
        }}
    ],
    "metadata": {{
        "source": "audio",
        "duration": "{duration}",
        "date": "{date}"
    }}
}}

## Constraints
- Output language: {language}
- Organize by topic, not time
- Merge cross-chunk concepts but PRESERVE all details
- Use clear heading hierarchy (H2 for sections, H3 for subsections)
- Include ALL technical details, formulas, numbers, and examples
- **Mathematical formulas**: Use LaTeX format in markdown content
  - Inline: `$formula$`, Display: `$$formula$$`
  - Example: `$f(x) = 1/(1 + e^{{-x}})$` or use simple notation if needed
- Valid JSON format only - ensure proper escaping of special characters
- Minimum 5 sections for comprehensive content
- Each major section should be detailed (200+ words)
- Do NOT use phrases like "as mentioned" or "briefly" - expand fully
"""

# Register the prompt
PromptRegistry.register("structuring", STRUCTURING_PROMPT)
