"""
Configuration classes for Notely SDK.

This module defines all configuration dataclasses used throughout the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """
    LLM configuration.

    Attributes:
        provider: LLM provider name (e.g., 'openai', 'anthropic')
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-opus')
        api_key: API key for authentication
        base_url: Base URL for API endpoint
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
    """

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        """Set default base_url based on provider if not specified."""
        if not self.base_url:
            if self.provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.provider == "anthropic":
                self.base_url = "https://api.anthropic.com/v1"
            elif self.provider == "zhipu":
                self.base_url = "https://open.bigmodel.cn/api/paas/v4"
            # Add more providers as needed


@dataclass
class EnhancerConfig:
    """
    Enhancer configuration.

    Attributes:
        llm: LLM configuration
        chunk_size: Maximum chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        language: Output language ('zh', 'en', 'ja', 'ko', or None for auto-detect)
        cache_dir: Directory for caching results
        max_concurrent: Maximum concurrent API calls
    """

    llm: LLMConfig
    chunk_size: int = 2000  # Reduced from 4000 to preserve more details
    chunk_overlap: int = 1000  # Increased from 800 for better context
    language: str | None = None  # None = auto-detect
    cache_dir: Path | None = None
    max_concurrent: int = 5

    def __post_init__(self) -> None:
        """Set default cache_dir if not specified."""
        if self.cache_dir is None:
            self.cache_dir = Path(".cache/enhancer")


@dataclass
class ASRConfig:
    """
    ASR (Automatic Speech Recognition) configuration.

    Attributes:
        backend: ASR backend name (e.g., 'funasr', 'whisper')
        model: Model identifier
        language: Language code (e.g., 'zh', 'en')
    """

    backend: str = "funasr"
    model: str = "paraformer-zh"
    language: str = "zh"


@dataclass
class OCRConfig:
    """
    OCR (Optical Character Recognition) configuration.

    Attributes:
        provider: OCR provider name (e.g., 'paddleocr', 'zhipu')
        model: Model identifier (e.g., 'glm-4v-flash' for Zhipu)
        api_key: API key for cloud providers
        base_url: Base URL for API endpoint
        language: Language code (e.g., 'ch', 'en')
        use_gpu: Whether to use GPU acceleration (for local providers)
    """

    provider: str = "paddleocr"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    language: str = "ch"
    use_gpu: bool = True

    def __post_init__(self) -> None:
        """Set provider-specific defaults and validate required fields."""
        if self.provider == "zhipu":
            if not self.model:
                self.model = "glm-4v-flash"
            if not self.base_url:
                self.base_url = "https://open.bigmodel.cn/api/paas/v4"
            if not self.api_key:
                raise ValueError("api_key is required for Zhipu OCR provider")
        elif self.provider == "paddleocr":
            if not self.model:
                self.model = "PP-OCRv4"


@dataclass
class NotelyConfig:
    """
    Notely SDK configuration container.

    This is the top-level configuration that holds all sub-configurations.

    Attributes:
        asr: ASR configuration
        ocr: OCR configuration
        enhancer: Enhancer configuration (includes LLM config)
    """

    asr: ASRConfig
    ocr: OCRConfig
    enhancer: EnhancerConfig

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> NotelyConfig:
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary with nested structure

        Returns:
            NotelyConfig instance

        Example:
            config = NotelyConfig.from_dict({
                "asr": {"backend": "funasr", "device": "cuda"},
                "ocr": {"backend": "paddleocr"},
                "llm": {"model": "gpt-4o", "api_key": "sk-xxx"},
                "enhancer": {"chunk_size": 4000},
            })
        """
        # Extract LLM config
        llm_dict = config_dict.get("llm", {})
        llm_config = LLMConfig(**llm_dict)

        # Extract enhancer config and merge with LLM
        enhancer_dict = config_dict.get("enhancer", {})
        enhancer_config = EnhancerConfig(llm=llm_config, **enhancer_dict)

        # Extract ASR and OCR configs
        asr_dict = config_dict.get("asr", {})
        asr_config = ASRConfig(**asr_dict)

        ocr_dict = config_dict.get("ocr", {})
        ocr_config = OCRConfig(**ocr_dict)

        return cls(
            asr=asr_config,
            ocr=ocr_config,
            enhancer=enhancer_config,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> NotelyConfig:
        """
        Create configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            NotelyConfig instance

        Raises:
            FileNotFoundError: If YAML file does not exist
            yaml.YAMLError: If YAML parsing fails

        Example:
            config = NotelyConfig.from_yaml("config.yaml")
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "asr": {
                "backend": self.asr.backend,
                "model": self.asr.model,
                "language": self.asr.language,
            },
            "ocr": {
                "provider": self.ocr.provider,
                "model": self.ocr.model,
                "api_key": self.ocr.api_key,
                "base_url": self.ocr.base_url,
                "language": self.ocr.language,
                "use_gpu": self.ocr.use_gpu,
            },
            "llm": {
                "provider": self.enhancer.llm.provider,
                "model": self.enhancer.llm.model,
                "api_key": self.enhancer.llm.api_key,
                "base_url": self.enhancer.llm.base_url,
                "temperature": self.enhancer.llm.temperature,
                "max_tokens": self.enhancer.llm.max_tokens,
            },
            "enhancer": {
                "chunk_size": self.enhancer.chunk_size,
                "chunk_overlap": self.enhancer.chunk_overlap,
                "language": self.enhancer.language,
                "cache_dir": str(self.enhancer.cache_dir) if self.enhancer.cache_dir else None,
                "max_concurrent": self.enhancer.max_concurrent,
            },
        }

    def to_yaml(self, yaml_path: Path | str) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to output YAML file
        """
        import yaml

        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
