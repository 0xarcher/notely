# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-03

### Added
- Multi-format input support (video, audio, PDF, images)
- ASR backends: FunASR (Chinese) and Whisper (multilingual)
- OCR backend: PaddleOCR with smart frame deduplication
- LLM integration with OpenAI-compatible APIs
- Template system with Markdown files and YAML frontmatter
- Beautiful Markdown output formatting with automatic beautification
- CLI interface for command-line usage
- Modular input processing architecture
- Comprehensive documentation (README, CONTRIBUTING)

### Technical
- Type-safe codebase with full type annotations
- Code quality tools: ruff for formatting and linting
- Unified `Union` type annotation syntax
- Eliminated code duplication in utility functions
- Fixed all static analysis warnings
