# Contributing to Notely

Thank you for your interest in contributing to Notely! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/0xarcher/notely.git
   cd notely
   ```

2. **Install uv** (if not already installed)
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or via pip
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   # Recommended: Use uv sync (creates venv automatically)
   uv sync --all-extras

   # Or: Install in editable mode with all dependencies
   uv pip install -e ".[dev,all]"
   ```

4. **Install FFmpeg** (required for audio/video processing)
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Code Style

We use `ruff` for code formatting and linting:

```bash
# Format code
uv run ruff format .

# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .
```

**Style Guidelines**:
- Follow PEP 8
- Use type hints for all public APIs
- Use `Union[X, Y]` instead of `X | Y` for type annotations
- Write docstrings in Google style
- Keep functions focused and under 50 lines when possible

## Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=notely --cov-report=html
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, focused commits
   - Add tests for new features
   - Update documentation as needed

3. **Ensure code quality**
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```

4. **Submit PR**
   - Write a clear PR description
   - Reference any related issues
   - Ensure CI passes

## Commit Message Guidelines

Follow conventional commits:

```
feat: add PDF processing support
fix: resolve audio extraction issue on Windows
docs: update installation instructions
refactor: simplify template loading logic
test: add tests for OCR backend
```

## Adding New Features

### Adding a New ASR Backend

1. Create a new file in `src/notely/asr/`
2. Inherit from `ASRBackend`
3. Implement required methods
4. Add tests
5. Update documentation

### Adding a New Template

1. Create a new `.md` file in `src/notely/prompts/templates/zh/` or `templates/en/`
2. Follow the existing template format (YAML frontmatter + Markdown sections)
3. Test with sample content
4. Update documentation

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about contributing

Thank you for contributing! 🎉
