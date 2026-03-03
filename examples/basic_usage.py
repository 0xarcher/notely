"""
Basic usage example for Notely.

This example demonstrates how to use Notely to convert
a lecture video into beautiful Markdown notes.
"""

import os
from pathlib import Path

from notely import Notely


def main():
    """Basic usage example."""
    # 1. Initialize with API key
    notely = Notely(api_key=os.getenv("OPENAI_API_KEY"))

    # 2. Process a lecture video
    print("Processing lecture video...")
    result = notely.process(
        video_path="lecture.mp4",
        pdf_paths=["slides.pdf"],  # Optional: include PDF slides
        title="Introduction to Machine Learning",
        instructor="Prof. John Smith",
        date="2026-03-03",
    )

    # 3. Save the generated notes
    output_path = Path("output/notes.md")
    result.save(output_path)
    print(f"Notes saved to: {output_path}")

    # 4. Access the content programmatically
    print(f"\nGenerated {len(result.transcript.segments)} transcript segments")
    print(f"Total duration: {result.transcript.duration:.1f} seconds")
    print(f"OCR results: {len(result.ocr_results)} frames/pages")

    # Print first 500 characters of the generated notes
    print("\n--- Preview ---")
    print(result.markdown[:500] + "...")


def example_with_api_key():
    """Example: Specify API key explicitly."""
    notely = Notely(api_key="sk-xxx")

    result = notely.process(
        video_path="lecture.mp4",
        title="Machine Learning Basics",
    )

    result.save("notes.md")


def example_switch_provider():
    """Example: Switch to different LLM provider."""
    import os

    # Use Zhipu AI (智谱 AI)
    notely = Notely(
        api_key=os.getenv("ZHIPU_API_KEY"),
        provider="zhipu",
        model="glm-4",
    )

    result = notely.process(
        video_path="lecture.mp4",
        title="深度学习基础",
    )

    result.save("notes.md")


def example_custom_endpoint():
    """Example: Use custom OpenAI-compatible endpoint."""
    notely = Notely(
        api_key="sk-xxx",
        provider="custom",
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    result = notely.process(
        video_path="lecture.mp4",
        title="AI 应用开发",
    )

    result.save("notes.md")


def example_audio_only():
    """Example: Process audio file without video."""
    import os

    notely = Notely(api_key=os.getenv("OPENAI_API_KEY"))

    # Process audio file
    result = notely.process_audio(
        audio_path="podcast.mp3",
        title="Tech Podcast Episode 42",
    )

    result.save("podcast_notes.md")


def example_with_custom_template():
    """Example: Use a custom note template."""
    import os
    from notely.prompts import NoteTemplate

    # Create a custom template
    template = NoteTemplate(
        name="meeting",
        language="zh",
        style="casual",
        include_timestamps=True,
        include_transcript=False,
        custom_sections=["待办事项", "决策记录"],
    )

    notely = Notely(api_key=os.getenv("OPENAI_API_KEY"))

    result = notely.process(
        video_path="meeting_recording.mp4",
        template=template,
        title="Weekly Team Meeting",
    )

    result.save("meeting_notes.md")


def example_full_configuration():
    """Example: Full configuration with all options."""
    import os

    notely = Notely(
        # LLM configuration
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        # ASR configuration
        asr_backend="whisper",
        asr_device="cpu",
        # OCR configuration
        ocr_backend="paddleocr",
        ocr_lang="ch",
        # Processing settings
        key_frame_interval_seconds=5.0,
        min_frame_similarity=0.85,
        # Other settings
        template="academic",
        verbose=True,
    )

    result = notely.process(
        video_path="lecture.mp4",
        pdf_paths=["slides.pdf"],
        title="Advanced Topics in AI",
        instructor="Dr. Jane Doe",
        date="2026-03-03",
    )

    result.save("advanced_notes.md")


if __name__ == "__main__":
    main()
