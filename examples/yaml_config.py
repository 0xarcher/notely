#!/usr/bin/env python3
"""
Example: Using Notely with YAML configuration file.
"""

import asyncio
from pathlib import Path

from notely import Notely


async def main() -> None:
    """Main function."""
    # Load configuration from YAML file
    config_path = Path("config.yaml")

    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        print("  Please create config.yaml based on config.example.yaml")
        return

    # Create Notely from YAML
    notely = Notely.from_yaml(config_path)

    print("✓ Notely initialized from YAML configuration")
    print(f"  Provider: {notely.config.enhancer.llm.provider}")
    print(f"  Model: {notely.config.enhancer.llm.model}")
    print()

    # Process audio file
    audio_path = Path.home() / "Downloads" / "audio.wav"

    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        return

    print(f"Processing: {audio_path}")
    print()

    result = await notely.process(
        input_path=audio_path,
        metadata={"title": "课程笔记"},
    )

    # Save result
    output_path = Path("output/notes.md")
    result.save(output_path)

    print(f"✓ Notes saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
