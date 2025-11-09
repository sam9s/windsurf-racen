"""
OpenAI Response Converter

This module converts Anthropic Agents SDK message streams to OpenAI-compatible
streaming response format for compatibility with OpenAI API clients.
"""

import json
import time
import uuid
from typing import AsyncIterator, Any, Optional
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, ResultMessage
import tiktoken


def generate_completion_id() -> str:
    """Generate a unique completion ID in OpenAI format."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: The text to count tokens in
        model: The model name for encoding selection

    Returns:
        int: Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base encoding for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def convert_sdk_to_openai_stream(
    sdk_messages: AsyncIterator[Any],
    model_name: str = "claude-sonnet-4",
    completion_id: Optional[str] = None
) -> AsyncIterator[tuple[str, Optional[str]]]:
    """
    Convert Claude SDK message stream to OpenAI SSE format.

    This function takes the async iterator from Claude SDK's receive_messages()
    and converts it to OpenAI-compatible Server-Sent Events format.

    Args:
        sdk_messages: Async iterator of SDK messages
        model_name: Model name to include in responses
        completion_id: Completion ID (generated if not provided)

    Yields:
        tuple[str, Optional[str]]: SSE-formatted chunk and session_id (only in final chunk)

    Example:
        async with create_claude_agent() as agent:
            await agent.query("Hello")
            async for chunk, session_id in convert_sdk_to_openai_stream(agent.receive_messages()):
                yield chunk
                if session_id:
                    # Store session_id
    """
    completion_id = completion_id or generate_completion_id()
    created = int(time.time())
    full_response = ""
    session_id = None

    # Send initial chunk with role
    initial_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "system_fingerprint": f"fp_{model_name}",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "logprobs": None,
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n", None

    try:
        async for message in sdk_messages:
            # Handle AssistantMessage with content blocks
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    # Extract text content
                    if isinstance(block, TextBlock):
                        text_content = block.text
                        full_response += text_content + "\n"

                        # Send text chunk
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "system_fingerprint": f"fp_{model_name}",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text_content + "\n"},
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n", None

                    # Note: ToolUseBlock and ToolResultBlock are handled internally by SDK
                    # We only stream the text responses to the client

            # Check for completion and capture session ID
            if isinstance(message, ResultMessage):
                session_id = message.session_id
                print(f"📋 Captured session ID from ResultMessage: {session_id}")
                break

        # Send final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "system_fingerprint": f"fp_{model_name}",
            "choices": [{
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n", session_id

    except Exception as e:
        # Send error in OpenAI format
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n", None

    # Send [DONE] marker
    yield "data: [DONE]\n\n", None


async def convert_sdk_to_openai_completion_stream(
    sdk_messages: AsyncIterator[Any],
    model_name: str = "claude-sonnet-4",
    completion_id: Optional[str] = None
) -> AsyncIterator[str]:
    """
    Convert Claude SDK message stream to OpenAI legacy completion format.

    Similar to convert_sdk_to_openai_stream but uses text_completion object type.

    Args:
        sdk_messages: Async iterator of SDK messages
        model_name: Model name to include in responses
        completion_id: Completion ID (generated if not provided)

    Yields:
        str: SSE-formatted chunks in OpenAI legacy completion format
    """
    completion_id = completion_id or f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    full_response = ""

    try:
        async for message in sdk_messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_content = block.text
                        full_response += text_content + "\n"

                        # Send text chunk in legacy format
                        chunk = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "text": text_content + "\n",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            if isinstance(message, ResultMessage):
                break

        # Send final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"

    except Exception as e:
        error_chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"


def extract_full_response_text(messages: list[Any]) -> str:
    """
    Extract full text response from a list of SDK messages.

    Used for non-streaming responses to collect all text content.

    Args:
        messages: List of SDK messages

    Returns:
        str: Combined text content
    """
    full_text = ""

    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    full_text += block.text

    return full_text
