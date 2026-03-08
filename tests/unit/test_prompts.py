"""Tests for prompt registry."""

import pytest

from notely.prompts.registry import PromptRegistry


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test to ensure clean state."""
    PromptRegistry.clear()
    yield
    PromptRegistry.clear()


def test_register_and_get_prompt():
    """Test registering and retrieving a prompt."""
    test_prompt = "This is a test prompt with {variable}"

    PromptRegistry.register("test_prompt", test_prompt)
    retrieved = PromptRegistry.get("test_prompt")

    assert retrieved == test_prompt


def test_get_nonexistent_prompt_raises_error():
    """Test that getting non-existent prompt raises KeyError."""
    with pytest.raises(KeyError, match="Prompt 'nonexistent' not found"):
        PromptRegistry.get("nonexistent")


def test_prompt_formatting():
    """Test that retrieved prompts can be formatted."""
    template = "Hello {name}, you are {age} years old"
    PromptRegistry.register("greeting", template)

    formatted = PromptRegistry.get("greeting").format(name="Alice", age=30)
    assert formatted == "Hello Alice, you are 30 years old"


def test_list_prompts():
    """Test listing all registered prompts."""
    PromptRegistry.register("prompt1", "Content 1")
    PromptRegistry.register("prompt2", "Content 2")

    prompts = PromptRegistry.list()
    assert "prompt1" in prompts
    assert "prompt2" in prompts


def test_list_empty_registry():
    """Test listing prompts from empty registry."""
    prompts = PromptRegistry.list()
    assert prompts == []


def test_overwrite_raises_error_by_default():
    """Test that overwriting a prompt raises ValueError by default."""
    PromptRegistry.register("test", "Original content")

    with pytest.raises(ValueError, match="Prompt 'test' already exists"):
        PromptRegistry.register("test", "New content")


def test_overwrite_with_allow_overwrite():
    """Test that overwriting works when allow_overwrite=True."""
    PromptRegistry.register("test", "Original content")
    PromptRegistry.register("test", "New content", allow_overwrite=True)

    retrieved = PromptRegistry.get("test")
    assert retrieved == "New content"


def test_thread_safety():
    """Test that registry is thread-safe for concurrent access."""
    import threading

    def register_prompts(start_idx: int):
        for i in range(start_idx, start_idx + 10):
            PromptRegistry.register(f"prompt_{i}", f"Content {i}")

    threads = [
        threading.Thread(target=register_prompts, args=(0,)),
        threading.Thread(target=register_prompts, args=(10,)),
        threading.Thread(target=register_prompts, args=(20,)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Should have exactly 30 prompts
    assert len(PromptRegistry.list()) == 30

    # All prompts should be retrievable
    for i in range(30):
        assert PromptRegistry.get(f"prompt_{i}") == f"Content {i}"
