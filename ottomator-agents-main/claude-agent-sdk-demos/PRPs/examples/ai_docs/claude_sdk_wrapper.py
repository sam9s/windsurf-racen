"""
Claude SDK Client Wrapper

This module provides a wrapper around the Anthropic Agents SDK (ClaudeSDKClient)
with session management and memory integration for Obsidian Copilot.
"""

import os
from typing import Optional, AsyncIterator, Any
from contextlib import asynccontextmanager
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock,
    ResultMessage
)
from obsidian_utils import get_vault_working_directory


class ClaudeAgentWrapper:
    """
    Wrapper for ClaudeSDKClient with enhanced session and memory management.

    This class handles:
    - SDK client initialization with proper options
    - Working directory configuration for Obsidian vault
    - System prompt construction with Mem0 memories
    - Session ID management for conversation continuity
    """

    def __init__(
        self,
        working_directory: Optional[str] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        memories: str = "",
        allowed_tools: Optional[list] = None
    ):
        """
        Initialize the Claude SDK wrapper.

        Args:
            working_directory: Working directory for file operations.
                             If None, uses path from environment variables (OBSIDIAN_VAULT_PATH and OBSIDIAN_VAULT_NAME)
            system_prompt: Base system prompt for the agent
            session_id: Session ID (epoch timestamp) for conversation continuity
            memories: Formatted memories string from Mem0
            allowed_tools: List of allowed tool names (default: all tools enabled)
        """
        self.working_directory = working_directory or get_vault_working_directory()
        self.session_id = session_id
        self.memories = memories

        # Verify working directory exists (debug log)
        import os
        if not os.path.exists(self.working_directory):
            print(f"⚠️  WARNING: Working directory does not exist: {self.working_directory}")
        else:
            print(f"✓ Working directory exists: {self.working_directory}")

        # Construct full system prompt with memories
        base_prompt = system_prompt or self._get_default_system_prompt()

        if memories:
            full_system_prompt = f"{base_prompt}\n\n## Relevant Memories\n{memories}"
        else:
            full_system_prompt = base_prompt

        # Configure MCP server connection
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/mcp")

        # Configure Claude Agent options
        # Use "*" for allowed_tools to enable ALL tools (Claude Code + MCP server tools)
        options_dict = {
            "cwd": self.working_directory,
            "system_prompt": full_system_prompt,
            "allowed_tools": ["mcp__agent-tools"],
            "mcp_servers": {
                "agent-tools": {
                    "type": "http",
                    "url": mcp_server_url
                }
            }
        }

        print(f"🗂️  Claude Code working directory: {self.working_directory}")
        print(f"✓ MCP server configured at: {mcp_server_url}")

        # Add session resume if session_id provided
        if session_id:
            options_dict["resume"] = session_id

        self.options = ClaudeAgentOptions(**options_dict)

        self.client: Optional[ClaudeSDKClient] = None

    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for Obsidian integration.

        Returns:
            str: Default system prompt
        """
        return """You are a helpful AI assistant working within an Obsidian vault.

You have access to file operations within the vault directory. Use the available tools to:
- Read and analyze files in the vault
- Create new notes and documents
- Edit existing content
- Execute shell commands when needed
- Search for files and content

Always maintain the integrity of the vault structure and help the user organize their knowledge effectively."""

    async def connect(self):
        """Initialize and connect the Claude SDK client."""
        self.client = ClaudeSDKClient(options=self.options)
        await self.client.connect()

    async def disconnect(self):
        """Disconnect the Claude SDK client."""
        if self.client:
            await self.client.disconnect()
            self.client = None

    async def query(self, user_prompt: str) -> None:
        """
        Send a query to the Claude SDK.

        Args:
            user_prompt: The user's question or request

        Note:
            This initiates the query but doesn't return the response.
            Use receive_messages() to get streaming responses.
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        await self.client.query(user_prompt)

    async def receive_messages(self) -> AsyncIterator[Any]:
        """
        Receive streaming messages from the Claude SDK.

        Yields:
            Messages from the SDK (AssistantMessage, ToolUseBlock, etc.)

        Example:
            async for message in wrapper.receive_messages():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        async for message in self.client.receive_messages():
            yield message

    async def receive_response(self) -> list[Any]:
        """
        Receive messages until and including ResultMessage.

        This method waits for the complete response and returns all messages.
        Use this for non-streaming contexts where you need the full response.

        Returns:
            List of messages including the final ResultMessage
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        messages = []
        async for message in self.client.receive_response():
            messages.append(message)
        return messages

    async def interrupt(self):
        """Interrupt the current Claude SDK operation."""
        if self.client:
            await self.client.interrupt()


@asynccontextmanager
async def create_claude_agent(
    working_directory: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    memories: str = "",
    allowed_tools: Optional[list] = None
) -> AsyncIterator[ClaudeAgentWrapper]:
    """
    Async context manager for creating and managing a Claude agent.

    Args:
        working_directory: Working directory for file operations.
                          If None, uses path from environment variables
        system_prompt: Base system prompt for the agent
        session_id: Session ID for conversation continuity
        memories: Formatted memories string from Mem0
        allowed_tools: List of allowed tool names

    Yields:
        ClaudeAgentWrapper: Connected Claude agent wrapper

    Example:
        async with create_claude_agent(memories=memories_str) as agent:
            await agent.query("Create a new note about Python")
            async for message in agent.receive_messages():
                # Process messages
                pass
    """
    wrapper = ClaudeAgentWrapper(
        working_directory=working_directory,
        system_prompt=system_prompt,
        session_id=session_id,
        memories=memories,
        allowed_tools=allowed_tools
    )

    try:
        await wrapper.connect()
        yield wrapper
    finally:
        await wrapper.disconnect()
