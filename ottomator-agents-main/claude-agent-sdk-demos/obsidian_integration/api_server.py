"""
OpenAI-Compatible API Server for Claude Agent SDK
==================================================

A simple FastAPI server that provides an OpenAI-compatible /v1/chat/completions
endpoint powered by the Claude Agent SDK. This enables integration with
OpenAI-compatible clients like Obsidian Copilot.

Key Features:
- OpenAI-compatible chat completions endpoint
- Streaming and non-streaming support
- Simple session management
- No complex authentication or dependencies

Usage:
    python api_server.py

    Or with uvicorn:
    uvicorn api_server:app --host 0.0.0.0 --port 8003
"""

import json
import time
import os
from typing import List, Optional, Union
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from openai_converter import convert_sdk_to_openai_stream, extract_full_response_text, generate_completion_id
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

# Load environment variables from .env file
load_dotenv()


# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: str
    content: Optional[Union[str, List[dict]]] = None  # Can be string or array (multimodal)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "claude-sonnet-4"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


class Choice(BaseModel):
    """OpenAI-compatible choice object."""
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    """OpenAI-compatible usage object."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


# ==================== Session Management ====================

SESSIONS_DIR = Path("api_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


def save_session(conversation_id: str, session_id: str):
    """Save session ID for a conversation."""
    session_file = SESSIONS_DIR / f"{conversation_id}.json"
    with open(session_file, "w") as f:
        json.dump({"session_id": session_id}, f)


def load_session(conversation_id: str) -> Optional[str]:
    """Load session ID for a conversation."""
    session_file = SESSIONS_DIR / f"{conversation_id}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            data = json.load(f)
            return data.get("session_id")
    return None


def is_continuing_conversation(messages: List[ChatMessage]) -> bool:
    """
    Determine if this is a continuing conversation or a new one.

    Args:
        messages: List of messages in the request

    Returns:
        True if continuing conversation (multiple messages), False if new (single message)
    """
    # Count user messages to determine if this is a continuation
    user_message_count = sum(1 for msg in messages if msg.role == "user")

    # More than one user message = continuing conversation
    return user_message_count > 1


# ==================== FastAPI Application ====================

app = FastAPI(
    title="Claude Agent SDK API",
    description="OpenAI-compatible API powered by Claude Agent SDK",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    This endpoint accepts OpenAI-format chat completion requests and processes
    them using the Claude Agent SDK.

    Args:
        request: OpenAI-format chat completion request

    Returns:
        ChatCompletionResponse in OpenAI-compatible format
    """
    try:
        # Extract the latest user message
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                # Handle both string and multimodal array format
                if isinstance(msg.content, str):
                    user_query = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from multimodal content
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_query += item.get("text", "")
                break

        if not user_query:
            raise HTTPException(status_code=400, detail="No user message found")

        # Determine if this is a continuing conversation
        is_continuation = is_continuing_conversation(request.messages)
        session_id = None

        if is_continuation:
            # Load the current session to continue the conversation
            session_id = load_session("current")
            if session_id:
                print(f"📂 Resuming session: {session_id}")
            else:
                print("⚠️  No session found, starting new conversation")
        else:
            print("🆕 New conversation")

        # Configure Claude Agent options
        working_dir = os.getenv("WORKING_DIRECTORY", os.getcwd())
        options_dict = {
            "cwd": working_dir,
            "system_prompt": "You are a helpful AI assistant. You are currently powered by Claude Sonnet 4.5, with the exact model ID claude-sonnet-4-5-20250929.",
            "allowed_tools": ["Read", "Write", "Bash", "Edit", "mcp__sequential-thinking"],
            "mcp_servers": {
                "sequential-thinking": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
                }
            }
        }
        if session_id:
            options_dict["resume"] = session_id

        options = ClaudeAgentOptions(**options_dict)

        # Handle streaming response
        if request.stream:
            async def stream_response():
                new_session_id = None

                async with ClaudeSDKClient(options=options) as client:
                    await client.query(user_query)

                    # Stream chunks and capture session ID
                    async for chunk, captured_session_id in convert_sdk_to_openai_stream(
                        client.receive_messages(),
                        model_name=request.model
                    ):
                        yield chunk
                        if captured_session_id:
                            new_session_id = captured_session_id

                # Save session ID for future requests
                if new_session_id:
                    save_session("current", new_session_id)
                    print(f"💾 Saved session: {new_session_id}")

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )

        # Handle non-streaming response
        else:
            completion_id = generate_completion_id()
            created = int(time.time())

            # Run agent and collect complete response
            new_session_id = None
            messages = []
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_query)

                async for message in client.receive_response():
                    messages.append(message)
                    if isinstance(message, ResultMessage):
                        new_session_id = message.session_id

            full_response_text = extract_full_response_text(messages)

            # Save session for future requests
            if new_session_id:
                save_session("current", new_session_id)
                print(f"💾 Saved session: {new_session_id}")

            # Build OpenAI-compatible response
            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(role="assistant", content=full_response_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=0,  # Simplified: not calculating tokens
                    completion_tokens=0,
                    total_tokens=0
                ),
                system_fingerprint=f"fp_{request.model}"
            )

            return JSONResponse(content=response.model_dump())

    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return error in OpenAI format
        if request.stream:
            async def error_stream():
                error_chunk = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": f"⚠️ Error: {str(e)}"
                        },
                        "finish_reason": "error"
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(e), "type": "internal_error"}}
            )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "claude-agent-sdk-api",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Claude Agent SDK API",
        "description": "OpenAI-compatible API powered by Claude Agent SDK",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", 8003))

    print("=" * 60)
    print("Claude Agent SDK API Server")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"Documentation: http://localhost:{port}/docs")
    print(f"Health check: http://localhost:{port}/health")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
