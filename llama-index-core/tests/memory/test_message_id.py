"""Test message ID functionality in memory components."""

import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.memory import Memory


def test_chat_message_id():
    """Test that ChatMessage can have an ID set."""
    # Create message with ID
    message_with_id = ChatMessage(
        content="Test message", role=MessageRole.USER, id="test-id-123"
    )
    assert message_with_id.id == "test-id-123"

    # Create message without ID
    message_without_id = ChatMessage(content="Test message", role=MessageRole.USER)
    assert message_without_id.id is None


@pytest.mark.asyncio
async def test_memory_message_id():
    """Test that Memory handles message IDs properly."""
    memory = Memory.from_defaults()

    # Add message with ID
    message1 = ChatMessage(content="Message 1", role=MessageRole.USER, id="msg-1")
    await memory.aput(message1)

    # Add message without ID (should get auto-assigned)
    message2 = ChatMessage(content="Message 2", role=MessageRole.ASSISTANT)
    await memory.aput(message2)

    # Verify message2 got an ID
    messages = await memory.aget_all()
    assert len(messages) == 2
    assert messages[0].id == "msg-1"
    assert messages[1].id is not None
    assert messages[1].content == "Message 2"

    # Get message by ID
    retrieved_message = await memory.aget_by_id("msg-1")
    assert retrieved_message is not None
    assert retrieved_message.content == "Message 1"

    # Try getting message with non-existent ID
    non_existent = await memory.aget_by_id("non-existent-id")
    assert non_existent is None


@pytest.mark.asyncio
async def test_memory_multiple_messages():
    """Test that Memory.aput_messages assigns IDs to multiple messages."""
    memory = Memory.from_defaults()

    # Create messages
    messages = [
        ChatMessage(content="Message 1", role=MessageRole.USER, id="msg-1"),
        ChatMessage(content="Message 2", role=MessageRole.ASSISTANT),  # No ID
        ChatMessage(content="Message 3", role=MessageRole.USER),  # No ID
    ]

    # Add messages
    await memory.aput_messages(messages)

    # Verify messages
    retrieved = await memory.aget_all()
    assert len(retrieved) == 3
    assert retrieved[0].id == "msg-1"
    assert retrieved[1].id is not None  # Auto-assigned
    assert retrieved[2].id is not None  # Auto-assigned

    # Get message by ID
    msg1 = await memory.aget_by_id("msg-1")
    assert msg1 is not None
    assert msg1.content == "Message 1"
