"""Unit tests for configuration classes."""

import tempfile
from pathlib import Path

import yaml

from notely.config import ASRConfig, EnhancerConfig, LLMConfig, NotelyConfig, OCRConfig


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


def test_llm_config_zhipu_provider():
    """Test LLMConfig with Zhipu provider."""
    config = LLMConfig(
        provider="zhipu",
        model="glm-4",
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    assert config.provider == "zhipu"
    assert config.model == "glm-4"


def test_llm_config_zhipu_default_url():
    """Test LLMConfig with Zhipu provider gets default URL."""
    config = LLMConfig(provider="zhipu", api_key="test-key")
    assert config.base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_ocr_config_defaults():
    """Test OCRConfig default values for paddleocr provider."""
    config = OCRConfig()
    assert config.provider == "paddleocr"
    assert config.model == "PP-OCRv4"  # Auto-filled default
    assert config.language == "ch"
    assert config.use_gpu is True
    assert config.api_key == ""  # Not required for local provider
    assert config.base_url == ""  # Not required for local provider


def test_ocr_config_zhipu_provider():
    """Test OCRConfig with Zhipu provider."""
    config = OCRConfig(
        provider="zhipu",
        model="glm-4v-flash",
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    assert config.provider == "zhipu"
    assert config.model == "glm-4v-flash"
    assert config.api_key == "test-key"


def test_ocr_config_zhipu_auto_defaults():
    """Test OCRConfig with Zhipu provider gets auto-filled defaults."""
    config = OCRConfig(provider="zhipu", api_key="test-key")
    assert config.model == "glm-4v-flash"  # Auto-filled
    assert config.base_url == "https://open.bigmodel.cn/api/paas/v4"  # Auto-filled


def test_ocr_config_zhipu_requires_api_key():
    """Test OCRConfig with Zhipu provider requires api_key."""
    import pytest

    with pytest.raises(ValueError, match="api_key is required"):
        OCRConfig(provider="zhipu")


def test_asr_config_no_device():
    """Test ASRConfig has no device field (auto-detected)."""
    config = ASRConfig()
    assert not hasattr(config, "device")
    assert config.backend == "funasr"
    assert config.model == "paraformer-zh"


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
        "asr": {"backend": "funasr"},
        "ocr": {"provider": "paddleocr", "use_gpu": True},
        "llm": {"model": "gpt-4o", "api_key": "sk-test"},
        "enhancer": {"chunk_size": 4000, "language": "zh"},
    }

    config = NotelyConfig.from_dict(config_dict)

    assert config.asr.backend == "funasr"
    assert config.ocr.provider == "paddleocr"
    assert config.enhancer.llm.model == "gpt-4o"
    assert config.enhancer.language == "zh"


def test_notely_config_from_dict_with_zhipu():
    """Test creating NotelyConfig with Zhipu provider."""
    config_dict = {
        "asr": {"backend": "funasr"},
        "ocr": {"provider": "zhipu", "model": "glm-4v-flash", "api_key": "test"},
        "llm": {"provider": "zhipu", "model": "glm-4", "api_key": "test"},
    }

    config = NotelyConfig.from_dict(config_dict)

    assert config.ocr.provider == "zhipu"
    assert config.ocr.model == "glm-4v-flash"
    assert config.enhancer.llm.provider == "zhipu"
    assert config.enhancer.llm.model == "glm-4"


def test_notely_config_from_yaml():
    """Test creating NotelyConfig from YAML file."""
    config_dict = {
        "asr": {"backend": "funasr"},
        "ocr": {"provider": "paddleocr"},
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
            "ocr": {"provider": "paddleocr"},
            "llm": {"api_key": "test", "model": "gpt-4o"},
        }
    )

    config_dict = config.to_dict()

    assert config_dict["asr"]["backend"] == "funasr"
    assert config_dict["ocr"]["provider"] == "paddleocr"
    assert config_dict["llm"]["model"] == "gpt-4o"
    assert "api_key" in config_dict["llm"]
    assert "device" not in config_dict["asr"]  # device field removed
