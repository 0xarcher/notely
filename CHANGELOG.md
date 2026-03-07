# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-03-07

### Improved
- Enhanced content generation quality with optimized prompts and chunking
- Better detail preservation in generated notes
- Added LaTeX formula support
- Added automatic language detection

### Fixed
- Fixed content loss in long audio processing (154% improvement)

## [0.2.0] - 2026-03-04

### Changed
- **BREAKING**: Refactored to configuration-driven architecture
- **BREAKING**: Changed API to `Notely(config=...)` or `Notely.from_dict({})`
- **BREAKING**: Changed to async processing with `asyncio.run(notely.process())`

### Added
- Configuration system with YAML and dictionary support
- Three-layer enhancement pipeline (Comprehension → Structuring → Formatting)

### Removed
- Old flat-parameter API
- Unused modules

## [0.1.0] - 2026-03-03

### Added
- Initial release
- Multi-format input support (video, audio)
- ASR backends: FunASR and Whisper
- OCR backend: PaddleOCR
- LLM integration with OpenAI-compatible APIs
- Markdown output formatting
