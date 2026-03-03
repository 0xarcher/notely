# Notely

English | [简体中文](README_zh.md)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/ASR-FunASR%20%7C%20Whisper-orange.svg" alt="ASR">
  <img src="https://img.shields.io/badge/OCR-PaddleOCR-red.svg" alt="OCR">
  <img src="https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic%20%7C%20Zhipu-purple.svg" alt="LLM">
</p>

<p align="center">
  <em>Automatically transform video/audio lectures into structured Markdown notes</em>
</p>

---

**Notely** is a Python SDK that uses ASR, OCR, and LLM technologies to automatically convert lecture videos, audio recordings, and presentations into high-quality Markdown notes.

## Core Features

- 🎯 **High-Quality Speech Recognition** - FunASR (Chinese CER < 3%), Whisper (multilingual)
- 📊 **Intelligent OCR** - PaddleOCR + key frame deduplication
- 🤖 **Multi-LLM Support** - OpenAI, Zhipu AI, Anthropic, Moonshot, DeepSeek
- ✨ **Beautiful Output** - Structured Markdown with automatic formatting
- 🔧 **Flexible Configuration** - Simple initialization with deep customization support

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/0xarcher/notely.git
cd notely

# Install dependencies (recommended: uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# Or use pip
pip install -e ".[all]"

# Install FFmpeg (required)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### 2. Basic Usage

```python
import os
from notely import Notely

# Explicitly pass API key
notely = Notely(api_key="sk-xxx")

# Or read from environment variable
notely = Notely(api_key=os.getenv("OPENAI_API_KEY"))

# Process lecture video
result = notely.process(
    video_path="lecture.mp4",
    title="Introduction to Machine Learning",
    instructor="Prof. Zhang",
)

# Save notes
result.save("notes.md")
```

### 3. Usage Flow

<p align="center">
  <img src="docs/images/usage-flow.png" alt="Usage Flow" width="600">
</p>

**Example Output:**

```markdown
# Introduction to Machine Learning

> 📌 Course Info: 45 minutes | Instructor: Prof. Zhang

## 📌 Course Overview

This lecture introduces the basic concepts of machine learning...

## 📚 Key Concepts

### What is Machine Learning

**Machine learning** is a technology that enables computers to learn from data...

### Types of Machine Learning

| Type | Characteristics | Use Cases |
|------|----------------|-----------|
| **Supervised Learning** | Labeled data | Classification, Regression |
| **Unsupervised Learning** | Unlabeled data | Clustering, Dimensionality Reduction |
| **Reinforcement Learning** | Environmental feedback | Games, Robotics |

## 💡 Key Takeaways

1. Machine learning is a core AI technology
2. Algorithm selection depends on data type and task
3. **Feature engineering** is crucial for model performance
```

---

## Detailed Usage Guide

### Initialization

#### Method 1: Basic Usage

```python
import os
from notely import Notely

# Explicitly pass API key
notely = Notely(api_key="sk-xxx")

# Or read from environment variable
notely = Notely(api_key=os.getenv("OPENAI_API_KEY"))
```

#### Method 2: Switch LLM Provider

```python
import os

# Use Zhipu AI
notely = Notely(
    api_key=os.getenv("ZHIPU_API_KEY"),
    provider="zhipu",
    model="glm-4",
)

# Use Anthropic
notely = Notely(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    provider="anthropic",
    model="claude-3-opus-20240229",
)

# Use Moonshot
notely = Notely(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    provider="moonshot",
    model="moonshot-v1-8k",
)
```

#### Method 3: Custom OpenAI-Compatible Endpoint

```python
notely = Notely(
    api_key="sk-xxx",
    provider="custom",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

#### Method 4: Full Configuration

```python
import os

notely = Notely(
    # LLM configuration
    api_key=os.getenv("OPENAI_API_KEY"),
    provider="openai",
    model="gpt-4o",
    base_url="https://api.openai.com/v1",  # Optional
    temperature=0.7,
    max_tokens=4096,

    # ASR configuration
    asr_backend="funasr",  # Recommended for Chinese: funasr, multilingual: whisper
    asr_device="cuda",     # Use cuda with GPU, otherwise cpu
    asr_model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",

    # OCR configuration
    ocr_backend="paddleocr",
    ocr_lang="ch",  # Chinese: ch, English: en

    # Processing settings
    key_frame_interval_seconds=5.0,  # Key frame extraction interval
    min_frame_similarity=0.85,       # Frame deduplication similarity threshold

    # Other settings
    template="academic",  # Note template: academic, technical, meeting
    verbose=True,         # Show detailed logs
)
```

### Supported LLM Providers

| Provider | Provider Value | Recommended Models |
|----------|---------------|-------------------|
| OpenAI | `openai` | gpt-4o, gpt-4-turbo |
| Zhipu AI | `zhipu` | glm-4, glm-4-plus |
| Anthropic | `anthropic` | claude-3-opus, claude-3-sonnet |
| Moonshot | `moonshot` | moonshot-v1-8k, moonshot-v1-32k |
| DeepSeek | `deepseek` | deepseek-chat |
| Custom | `custom` | Any OpenAI-compatible API |

### Processing Different Input Formats

#### Process Video

```python
# Basic usage
result = notely.process(
    video_path="lecture.mp4",
    title="Course Title",
)

# With PDF slides
result = notely.process(
    video_path="lecture.mp4",
    pdf_paths=["slides.pdf", "handout.pdf"],
    title="Deep Learning Fundamentals",
    instructor="Prof. Li",
    date="2026-03-03",
)
```

#### Process Audio

```python
# Method 1: Use process_audio
result = notely.process_audio(
    audio_path="podcast.mp3",
    title="Tech Podcast Episode 42",
)

# Method 2: Use process
result = notely.process(
    audio_path="recording.wav",
    title="Meeting Recording",
)
```

#### Process PDF

```python
result = notely.process_pdf(
    pdf_path="presentation.pdf",
    title="Product Launch",
)
```

### Custom Note Templates

```python
from notely.prompts import NoteTemplate

# Use built-in templates
notely = Notely(api_key="sk-xxx", template="academic")  # Academic style
notely = Notely(api_key="sk-xxx", template="technical") # Technical style
notely = Notely(api_key="sk-xxx", template="meeting")   # Meeting notes

# Custom template
template = NoteTemplate(
    name="meeting",
    language="en",
    style="casual",
    include_timestamps=True,
    include_transcript=False,
    custom_sections=["Action Items", "Decisions"],
)

result = notely.process(
    video_path="meeting.mp4",
    template=template,
)
```

### Access Processing Results

```python
result = notely.process("lecture.mp4")

# Get Markdown content
print(result.markdown)

# Get transcript
print(result.transcript.full_text)
print(f"Duration: {result.transcript.duration:.1f} seconds")
print(f"Segments: {len(result.transcript.segments)}")

# Get OCR results
for ocr_result in result.ocr_results:
    print(ocr_result.full_text)

# Get metadata
print(result.metadata)

# Save to file
result.save("output/notes.md")
```

---

## How It Works

### Processing Pipeline

<p align="center">
  <img src="docs/images/pipeline.png" alt="Processing Pipeline" width="700">
</p>

### Architecture Overview

<p align="center">
  <img src="docs/images/architecture.png" alt="Architecture Overview" width="800">
</p>

**Key Steps:**

1. **Input Processing** - Extract audio and key frames from video
2. **ASR Transcription** - Speech to text with timestamps (FunASR for Chinese, Whisper for multilingual)
3. **OCR Recognition** - Extract text from slides/frames using PaddleOCR
4. **LLM Generation** - Fuse multimodal information to generate structured notes
5. **Format Output** - Beautify Markdown for readability

---

## FAQ

### 1. How to choose ASR backend?

- **Chinese content**: Recommended `funasr` (higher accuracy, CER < 3%)
- **Multilingual content**: Use `whisper` (supports 99+ languages)
- **No GPU**: Use `whisper` + `asr_device="cpu"`

```python
# Chinese lectures
notely = Notely(api_key="sk-xxx", asr_backend="funasr", asr_device="cuda")

# English lectures
notely = Notely(api_key="sk-xxx", asr_backend="whisper", asr_device="cpu")
```

### 2. How to reduce costs?

- Use cheaper models: `gpt-4o-mini`, `glm-4-flash`
- Adjust `max_tokens` to limit output length
- Use domestic LLMs (Zhipu, Moonshot, DeepSeek)

```python
notely = Notely(
    api_key=os.getenv("ZHIPU_API_KEY"),
    provider="zhipu",
    model="glm-4-flash",  # Cheaper
    max_tokens=2048,      # Limit output
)
```

### 3. How to improve processing speed?

- Use GPU acceleration: `asr_device="cuda"`
- Increase key frame interval: `key_frame_interval_seconds=10.0`
- Increase frame similarity threshold: `min_frame_similarity=0.90`

```python
notely = Notely(
    api_key="sk-xxx",
    asr_device="cuda",
    key_frame_interval_seconds=10.0,
    min_frame_similarity=0.90,
)
```

### 4. How to handle long videos?

Notely automatically handles long videos, but it's recommended to:
- Ensure sufficient memory and disk space
- Use GPU acceleration
- Consider splitting videos manually for very long content

### 5. What video formats are supported?

All formats supported by FFmpeg:
- Video: mp4, avi, mov, mkv, flv, wmv, webm
- Audio: mp3, wav, m4a, flac, aac, ogg

---

## Project Structure

```
notely/
├── src/notely/
│   ├── __init__.py          # Main entry point
│   ├── core.py              # Core logic
│   ├── asr/                 # ASR backends
│   │   ├── funasr.py        # FunASR
│   │   └── whisper.py       # Whisper
│   ├── ocr/                 # OCR backends
│   │   └── paddle.py        # PaddleOCR
│   ├── llm/                 # LLM backends
│   │   └── openai.py        # OpenAI-compatible
│   ├── prompts/             # Note templates
│   │   ├── templates/       # Template files
│   │   └── loader.py        # Template loader
│   ├── formatter/           # Markdown formatter
│   └── utils/               # Utility functions
├── examples/                # Example code
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── pyproject.toml
```

---

## Development Guide

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/0xarcher/notely.git
cd notely

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --all-extras

# Install FFmpeg
brew install ffmpeg  # macOS
```

### Code Standards

```bash
# Format code
uv run ruff format .

# Check code
uv run ruff check .

# Auto-fix
uv run ruff check --fix .
```

### Run Tests

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_core.py

# Generate coverage report
uv run pytest --cov=notely --cov-report=html
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Quick Start:**

1. Fork this repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "feat: add your feature"`
4. Push branch: `git push origin feature/your-feature`
5. Submit Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Notely is built on these excellent open-source projects:

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - Alibaba ASR toolkit
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Baidu OCR toolkit
- [Whisper](https://github.com/openai/whisper) - OpenAI speech recognition model
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction

---

## Contact

- GitHub: [@0xarcher](https://github.com/0xarcher)
- Email: coder.archer@gmail.com
- Issues: [GitHub Issues](https://github.com/0xarcher/notely/issues)

---

<p align="center">
  <strong>Made with ❤️ by Archer</strong>
</p>
