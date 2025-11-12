"""
OpenAI Response Converter
==========================

Converts Claude Agent SDK message streams to OpenAI-compatible streaming
response format (Server-Sent Events). This enables Claude SDK to work with
OpenAI-compatible clients like Obsidian Copilot.

Key Features:
- Streaming SSE format conversion
- Session ID capture for conversation continuity
- OpenAI-compatible message structure
"""

import json
import time
import uuid
from typing import AsyncIterator, Any, Optional, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage
)


def generate_completion_id() -> str:
    """Generate a unique completion ID in OpenAI format."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


async def convert_sdk_to_openai_stream(
    sdk_messages: AsyncIterator[Any],
    model_name: str = "claude-sonnet-4",
    completion_id: Optional[str] = None
) -> AsyncIterator[Tuple[str, Optional[str]]]:
    """
    Convert Claude SDK message stream to OpenAI SSE format.

    This function takes the async iterator from Claude SDK's receive_messages()
    and converts it to OpenAI-compatible Server-Sent Events format.

    Args:
        sdk_messages: Async iterator of SDK messages
        model_name: Model name to include in responses
        completion_id: Completion ID (generated if not provided)

    Yields:
        Tuple[str, Optional[str]]: (SSE-formatted chunk, session_id)
        The session_id is only returned in the final chunk, None otherwise.

    Example:
        async with create_claude_agent() as agent:
            await agent.query("Hello")
            async for chunk, session_id in convert_sdk_to_openai_stream(
                agent.receive_messages()
            ):
                yield chunk
                if session_id:
                    # Store session_id for conversation continuity
                    save_session(session_id)
    """
    completion_id = completion_id or generate_completion_id()
    created = int(time.time())
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

                    # ToolUseBlock and ToolResultBlock are handled internally by SDK
                    # We only stream the text responses to the client

            # Check for completion and capture session ID
            if isinstance(message, ResultMessage):
                session_id = message.session_id
                print(f"📋 Captured session ID: {session_id}")
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
        print(f"❌ Error in stream conversion: {str(e)}")
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


def extract_full_response_text(messages: list[Any]) -> str:
    """
    Extract full text response from a list of SDK messages.

    Used for non-streaming responses to collect all text content.

    Args:
        messages: List of SDK messages

    Returns:
        str: Combined text content

    Example:
        messages = await agent.receive_response()
        full_text = extract_full_response_text(messages)
    """
    full_text = ""

    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    full_text += block.text

    return full_text
