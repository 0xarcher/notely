"""Unit tests for configuration classes."""

import tempfile
from pathlib import Path

import yaml

from notely.config import EnhancerConfig, LLMConfig, NotelyConfig


def test_llm_config_defaults():
    """Test LLMConfig default values."""
    config = LLMConfig(api_key="test-key")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.temperature == 0.7
    assert config.max_tokens == 4096


def test_llm_config_custom_provider():
    """Test LLMConfig with custom provider."""
    config = LLMConfig(provider="anthropic", api_key="test-key", model="claude-3-opus")
    assert config.base_url == "https://api.anthropic.com/v1"


def test_enhancer_config_defaults():
    """Test EnhancerConfig default values."""
    llm = LLMConfig(api_key="test")
    config = EnhancerConfig(llm=llm)

    assert config.chunk_size == 2000
    assert config.chunk_overlap == 1000
    assert config.language is None  # Auto-detect
    assert config.max_concurrent == 5
    assert config.cache_dir == Path(".cache/enhancer")


def test_notely_config_from_dict():
    """Test creating NotelyConfig from dictionary."""
    config_dict = {
        "asr": {"backend": "funasr", "device": "cuda"},
        "ocr": {"backend": "paddleocr", "use_gpu": True},
        "llm": {"model": "gpt-4o", "api_key": "sk-test"},
        "enhancer": {"chunk_size": 4000, "language": "zh"},
    }

    config = NotelyConfig.from_dict(config_dict)

    assert config.asr.backend == "funasr"
    assert config.asr.device == "cuda"
    assert config.ocr.backend == "paddleocr"
    assert config.enhancer.llm.model == "gpt-4o"
    assert config.enhancer.language == "zh"


def test_notely_config_from_yaml():
    """Test creating NotelyConfig from YAML file."""
    config_dict = {
        "asr": {"backend": "funasr"},
        "ocr": {"backend": "paddleocr"},
        "llm": {"api_key": "test", "model": "gpt-4o"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        yaml_path = f.name

    try:
        config = NotelyConfig.from_yaml(yaml_path)
        assert config.asr.backend == "funasr"
        assert config.enhancer.llm.model == "gpt-4o"
    finally:
        Path(yaml_path).unlink()


def test_notely_config_to_dict():
    """Test converting NotelyConfig to dictionary."""
    config = NotelyConfig.from_dict(
        {
            "asr": {"backend": "funasr"},
            "ocr": {"backend": "paddleocr"},
            "llm": {"api_key": "test", "model": "gpt-4o"},
        }
    )

    config_dict = config.to_dict()

    assert config_dict["asr"]["backend"] == "funasr"
    assert config_dict["llm"]["model"] == "gpt-4o"
    assert "api_key" in config_dict["llm"]
