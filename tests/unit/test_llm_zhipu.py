"""Tests for ZhipuLLMBackend."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from notely.config import LLMConfig


def test_zhipu_llm_requires_api_key():
    """Test that ZhipuLLMBackend requires API key."""
    from notely.llm.zhipu import ZhipuLLMBackend

    config = LLMConfig(provider="zhipu", api_key="", model="glm-4-flash")
    with pytest.raises(ValueError, match="API key required"):
        ZhipuLLMBackend(config)


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_initialization(mock_zhipu_class):
    """Test ZhipuLLMBackend initialization."""
    from notely.llm.zhipu import ZhipuLLMBackend

    config = LLMConfig(
        provider="zhipu",
        api_key="test-key",
        model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    backend = ZhipuLLMBackend(config)

    assert backend.model == "glm-4-flash"
    assert backend.temperature == 0.7
    assert backend.max_tokens == 4096
    mock_zhipu_class.assert_called_once_with(
        api_key="test-key", base_url="https://open.bigmodel.cn/api/paas/v4"
    )


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_generate(mock_zhipu_class):
    """Test generate method with mocked API."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Generated text response"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Test generate
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    result = backend.generate("Hello, world!")

    assert result == "Generated text response"
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "glm-4-flash"
    assert call_args.kwargs["messages"][0]["role"] == "user"
    assert call_args.kwargs["messages"][0]["content"] == "Hello, world!"


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_generate_with_system_prompt(mock_zhipu_class):
    """Test generate method with system prompt."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Response with system context"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Test generate with system prompt
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    result = backend.generate("Hello!", system_prompt="You are a helpful assistant.")

    assert result == "Response with system context"
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello!"


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_generate_with_custom_params(mock_zhipu_class):
    """Test generate method with custom temperature and max_tokens."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Custom response"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Test with custom params
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    result = backend.generate("Test", temperature=0.5, max_tokens=1000)

    assert result == "Custom response"
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.5
    assert call_args.kwargs["max_tokens"] == 1000


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_generate_api_error(mock_zhipu_class):
    """Test that generate raises RuntimeError on API error."""
    from notely.llm.zhipu import ZhipuLLMBackend

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_zhipu_class.return_value = mock_client

    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    with pytest.raises(RuntimeError, match="Zhipu LLM API call failed"):
        backend.generate("Hello")


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_stream(mock_zhipu_class):
    """Test stream method with mocked API."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
        Mock(choices=[Mock(delta=Mock(content="!"))]),
    ]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = iter(mock_chunks)
    mock_zhipu_class.return_value = mock_client

    # Test stream
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    chunks = list(backend.stream("Hello"))

    assert chunks == ["Hello", " world", "!"]
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["stream"] is True


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_stream_with_system_prompt(mock_zhipu_class):
    """Test stream method with system prompt."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hi"))]),
    ]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = iter(mock_chunks)
    mock_zhipu_class.return_value = mock_client

    # Test stream with system prompt
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    chunks = list(backend.stream("Hi", system_prompt="Be brief."))

    assert chunks == ["Hi"]
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be brief."


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_stream_api_error(mock_zhipu_class):
    """Test that stream raises RuntimeError on API error."""
    from notely.llm.zhipu import ZhipuLLMBackend

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("Stream Error")
    mock_zhipu_class.return_value = mock_client

    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    with pytest.raises(RuntimeError, match="Zhipu LLM API call failed"):
        list(backend.stream("Hello"))


@patch("zhipuai.ZhipuAI")
def test_zhipu_llm_stream_empty_content(mock_zhipu_class):
    """Test stream handles chunks with empty content."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock streaming response with some empty chunks
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
    ]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = iter(mock_chunks)
    mock_zhipu_class.return_value = mock_client

    # Test stream
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    chunks = list(backend.stream("Hello"))

    # Should filter out None content
    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
@patch("zhipuai.ZhipuAI")
async def test_zhipu_llm_agenerate(mock_zhipu_class):
    """Test async generate method."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Async response"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_zhipu_class.return_value = mock_client

    # Test agenerate
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    result = await backend.agenerate("Hello")

    assert result == "Async response"


@pytest.mark.asyncio
@patch("zhipuai.ZhipuAI")
async def test_zhipu_llm_astream(mock_zhipu_class):
    """Test async stream method."""
    from notely.llm.zhipu import ZhipuLLMBackend

    # Mock streaming response
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Async"))]),
        Mock(choices=[Mock(delta=Mock(content=" stream"))]),
    ]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = iter(mock_chunks)
    mock_zhipu_class.return_value = mock_client

    # Test astream
    config = LLMConfig(provider="zhipu", api_key="test-key", model="glm-4-flash")
    backend = ZhipuLLMBackend(config)

    chunks = []
    async for chunk in backend.astream("Hello"):
        chunks.append(chunk)

    assert chunks == ["Async", " stream"]
