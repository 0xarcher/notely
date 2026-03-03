"""
Command-line interface for Notely.
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from notely import Notely

console = Console()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transform multimodal lectures into beautiful Markdown notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input video or audio file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output Markdown file (default: input_name.md)",
    )

    parser.add_argument(
        "--pdf",
        type=str,
        nargs="+",
        default=None,
        help="PDF slides to include",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Lecture title",
    )

    parser.add_argument(
        "--instructor",
        type=str,
        default=None,
        help="Instructor name",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider (default: openai)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key (or set OPENAI_API_KEY, ZHIPU_API_KEY, etc.)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)",
    )

    parser.add_argument(
        "--asr",
        type=str,
        choices=["funasr", "whisper"],
        default="funasr",
        help="ASR backend to use (default: funasr)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for ASR/OCR (default: cuda)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {args.input}[/red]")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".md")

    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        # Try to get from environment based on provider
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        env_key = env_keys.get(args.provider, f"{args.provider.upper()}_API_KEY")
        api_key = os.getenv(env_key, "")

    if not api_key:
        console.print(
            f"[red]Error: LLM API key is required. "
            f"Set environment variable or use --api-key[/red]"
        )
        sys.exit(1)

    # Initialize Notely
    notely = Notely(
        api_key=api_key,
        provider=args.provider,
        model=args.model,
        asr_backend=args.asr,
        asr_device=args.device,
    )

    # Process
    console.print(f"[bold blue]Notely[/bold blue] - Processing {input_path.name}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Processing lecture...", total=None)

            result = notely.process(
                video_path=input_path,
                pdf_paths=args.pdf,
                title=args.title,
                instructor=args.instructor,
            )

        # Save result
        result.save(output_path)

        console.print(f"[green]✓[/green] Notes saved to: {output_path}")
        console.print(f"  - Segments: {len(result.transcript.segments)}")
        console.print(f"  - Duration: {result.transcript.duration:.1f}s")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
