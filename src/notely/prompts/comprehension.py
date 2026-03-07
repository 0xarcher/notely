"""Comprehension prompts for extracting semantic information from transcripts."""

from notely.prompts.registry import PromptRegistry

# Optimized comprehension prompt with emphasis on detail preservation
COMPREHENSION_PROMPT = """Extract key information from the transcript chunk with MAXIMUM detail preservation.

## Task
Analyze the transcript and extract:
1. Detailed summary - MUST preserve ALL key points, explanations, and context
2. Core concepts with complete definitions and explanations
3. ALL examples, demonstrations, and case studies mentioned
4. Related questions for deeper understanding

## Critical Requirements
- MINIMUM summary length: 300 words (for typical chunks)
- Include ALL technical details, formulas, and specific numbers
- Preserve the logical flow and reasoning process
- Do NOT summarize or compress - expand and clarify instead
- If the speaker explains something step-by-step, preserve ALL steps

## Input
{transcript_text}

## Output Format (JSON)
{{
    "summary": "Comprehensive summary with ALL details, explanations, and context. Minimum 300 words. Include specific examples, numbers, and technical terms.",
    "key_concepts": ["concept1: complete definition with explanation", "concept2: complete definition with explanation"],
    "examples": ["example1 with full context and explanation", "example2 with full context and explanation"],
    "questions": ["question1", "question2"]
}}

## Constraints
- Output in the specified language: {language}
- Preserve all technical terms exactly as spoken
- NO information compression or loss - expand rather than compress
- Include specific details: numbers, formulas, examples
- **Mathematical formulas**: Use LaTeX format with proper JSON escaping
  - In JSON strings, use double braces: `$\\sigma(x) = \\frac{{{{1}}}}{{{{1 + e^{{{{-x}}}}}}}}$`
  - Or use simple text format if LaTeX causes issues: `sigma(x) = 1/(1 + e^(-x))`
- Valid JSON format only - ensure all special characters are properly escaped
"""

# Register the prompt
PromptRegistry.register("comprehension", COMPREHENSION_PROMPT)
