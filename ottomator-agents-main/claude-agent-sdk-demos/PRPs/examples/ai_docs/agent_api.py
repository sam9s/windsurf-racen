"""
Ultimate Productivity Agent API - Anthropic Agents SDK Version

This module provides an OpenAI-compatible API powered by the Anthropic Agents SDK,
designed for integration with Obsidian Copilot.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from mem0 import Memory
import asyncio
import time
import os
import uuid

# Import OpenAI compatibility models and authentication
from openai_api_compatibility import (
    ChatCompletionRequest,
    CompletionRequest,
    ChatMessage,
    verify_openai_api_key
)

# Import SDK wrapper and converter
from claude_sdk_wrapper import create_claude_agent
from openai_converter import count_tokens, convert_sdk_to_openai_stream

# Import Obsidian utilities
from obsidian_utils import (
    get_latest_conversation_data,
    update_conversation_session_id
)

# Import Mem0 client initialization
from clients import get_mem0_client_async

# Check if we're in production
is_production = os.getenv("ENVIRONMENT") == "production"

if not is_production:
    # Development: prioritize .env file
    project_root = Path(__file__).resolve().parent
    dotenv_path = project_root / '.env'
    load_dotenv(dotenv_path, override=True)
else:
    # Production: use cloud platform env vars only
    load_dotenv()

# Define clients as None initially
mem0_client = None

# Define the lifespan context manager for the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application.

    Handles initialization and cleanup of resources.
    """
    global mem0_client

    # Check authentication method
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        print("✓ Using Anthropic API key authentication")
    else:
        print("✓ Using Claude CLI OAuth authentication (assumed logged in via 'claude auth login')")

    # Initialize Mem0 client
    mem0_client = await get_mem0_client_async()
    print("✓ Mem0 client initialized")

    yield  # This is where the app runs

    # Shutdown: Clean up resources
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== OpenAI Compatible Endpoints ====================

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    authenticated: bool = Depends(verify_openai_api_key)
):
    """
    OpenAI-compatible chat completions endpoint.

    This endpoint accepts OpenAI-format chat completion requests and processes them
    using the Anthropic Agents SDK with full tool access.

    Args:
        request: OpenAI-format chat completion request
        authenticated: Bearer token authentication status

    Returns:
        ChatCompletionResponse in OpenAI-compatible format
    """
    try:
        # Extract user query from messages
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                content = msg.content
                if content is None:
                    continue

                # Handle multimodal content
                if isinstance(content, list):
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                    user_query = text_content
                else:
                    user_query = content
                break

        if not user_query:
            raise HTTPException(status_code=400, detail="No user message found in request")

        # Detect title generation requests (ephemeral, don't store session)
        is_title_generation = (
            "Generate a concise title" in user_query or
            "max 5 words" in user_query or
            (len(request.messages) == 1 and "title" in user_query.lower() and len(user_query) < 200)
        )

        # Detect first message (only one user message in history)
        is_first_message = len([m for m in request.messages if m.role == "user"]) == 1

        # Get session ID
        session_id = None
        if not is_title_generation:
            if is_first_message:
                # First message - no session yet
                session_id = None
            else:
                # Not first message - get stored session
                conversation_data = get_latest_conversation_data()
                if conversation_data:
                    session_id = conversation_data.get('claude_session_id')

        # Retrieve relevant memories with Mem0
        memories_str = ""
        # try:
        #     relevant_memories = await mem0_client.search(
        #         query=user_query,
        #         user_id="obsidian_user",
        #         limit=3
        #     )
        #     if relevant_memories and "results" in relevant_memories:
        #         memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
        # except Exception as e:
        #     print(f"⚠ Error retrieving memories: {str(e)}")

        # Fire-and-forget memory addition
        async def add_memory_async():
            try:
                memory_messages = [{"role": "user", "content": user_query}]
                await mem0_client.add(memory_messages, user_id="obsidian_user")
            except Exception as e:
                print(f"⚠ Error adding memory: {str(e)}")

        # asyncio.create_task(add_memory_async())

        # Handle streaming vs non-streaming
        if request.stream:
            # Streaming response
            async def stream_response():
                new_session_id = None

                async with create_claude_agent(
                    session_id=session_id,
                    memories=memories_str
                ) as agent:
                    await agent.query(user_query)

                    # Stream chunks and capture session ID
                    async for chunk, captured_session_id in convert_sdk_to_openai_stream(
                        agent.receive_messages(),
                        model_name=request.model
                    ):
                        yield chunk
                        if captured_session_id:
                            new_session_id = captured_session_id

                # Store session ID if this was first message
                if new_session_id and is_first_message:
                    update_conversation_session_id("", new_session_id)

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            # Non-streaming response
            from openai_api_compatibility import ChatCompletionResponse, Choice, Usage, generate_completion_id
            from openai_converter import extract_full_response_text
            from claude_agent_sdk import ResultMessage

            completion_id = generate_completion_id()
            created = int(time.time())

            # Run agent and collect complete response
            new_session_id = None
            async with create_claude_agent(
                session_id=session_id,
                memories=memories_str
            ) as agent:
                await agent.query(user_query)
                messages = await agent.receive_response()
                full_response_text = extract_full_response_text(messages)

                # Capture session ID from ResultMessage
                for msg in messages:
                    if isinstance(msg, ResultMessage):
                        new_session_id = msg.session_id
                        break

            # Calculate token counts
            prompt_tokens = sum(
                count_tokens(msg.content if isinstance(msg.content, str) else "")
                for msg in request.messages if msg.content
            )
            completion_tokens = count_tokens(full_response_text)

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
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                ),
                system_fingerprint=f"fp_{request.model}"
            )

            # Store session ID if this was first message
            if new_session_id and is_first_message:
                update_conversation_session_id("", new_session_id)

            # Return as JSONResponse to ensure proper content-type header
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        # Return OpenAI-format error
        print(f"❌ Error processing chat completion: {str(e)}")
        import traceback
        traceback.print_exc()

        # If streaming was requested, return error in SSE format
        if request.stream:
            async def error_stream():
                import json
                error_chunk = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": f"⚠️ Error: {str(e)}\n\nPlease try again or start a new conversation."
                        },
                        "finish_reason": "error"
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            # Non-streaming error
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "internal_error",
                        "code": None
                    }
                }
            )

@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration and monitoring.

    Returns:
        Dict with status and service health information
    """
    # Check if critical dependencies are initialized
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "mem0_client": mem0_client is not None,
            "anthropic_sdk": True  # SDK is stateless, always available
        }
    }

    # Check for authentication configuration
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    health_status["services"]["anthropic_auth"] = bool(anthropic_api_key)

    # If any critical service is not initialized, mark as unhealthy
    if not health_status["services"]["mem0_client"]:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
