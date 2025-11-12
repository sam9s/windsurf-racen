"""
Simple Terminal CLI for Claude Agent SDK
=========================================

A basic command-line interface demonstrating the Claude Agent SDK with:
- Streaming message display
- Session persistence (save/resume conversations)
- Simple, educational code for workshop learning

Usage:
    python simple_cli.py              # Start new conversation
    python simple_cli.py --continue   # Continue last conversation
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from colorama import init, Fore, Style, Back
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage
)

# Initialize colorama for cross-platform color support
init(autoreset=True)


# Configuration
SESSIONS_DIR = Path("sessions")
CURRENT_SESSION_FILE = SESSIONS_DIR / "current_session.json"


def save_session(session_id: str):
    """Save the session ID to a file for resuming later."""
    SESSIONS_DIR.mkdir(exist_ok=True)
    with open(CURRENT_SESSION_FILE, "w") as f:
        json.dump({"session_id": session_id}, f)


def load_session() -> Optional[str]:
    """Load the last session ID from file."""
    if CURRENT_SESSION_FILE.exists():
        with open(CURRENT_SESSION_FILE, "r") as f:
            data = json.load(f)
            return data.get("session_id")
    return None


async def chat_loop(resume_session: bool = False):
    """
    Main chat loop for interacting with Claude.

    Args:
        resume_session: If True, attempts to resume the last conversation
    """
    # Determine session ID
    session_id = None
    if resume_session:
        session_id = load_session()
        if session_id:
            print(f"{Fore.CYAN}📂 Resuming previous conversation\n")
        else:
            print(f"{Fore.YELLOW}⚠️  No previous session found. Starting new conversation.\n")

    # Configure the Claude Agent
    options_dict = {
        "cwd": os.getcwd(),  # Working directory for file operations
        "system_prompt": "You are a helpful AI assistant.",
        # Uncomment to limit tools:
        "allowed_tools": ["Read", "Write", "Bash"],
    }

    # Add resume parameter if we have a session ID
    if session_id:
        options_dict["resume"] = session_id

    options = ClaudeAgentOptions(**options_dict)

    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print(f"{Fore.MAGENTA}  Claude Agent SDK - Terminal Chat")
    print(f"{Fore.MAGENTA}{'=' * 60}")
    print(f"{Style.DIM}Type 'exit' or 'quit' to end the conversation{Style.RESET_ALL}\n")

    # Create and connect the Claude SDK client
    async with ClaudeSDKClient(options=options) as client:
        while True:
            # Get user input
            try:
                user_input = input(f"{Fore.CYAN}{Style.BRIGHT}You: {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break

            # Send query to Claude
            await client.query(user_input)

            # Print "Claude: " prefix
            print(f"\n{Fore.GREEN}{Style.BRIGHT}Claude: {Style.RESET_ALL}", end="", flush=True)

            # Receive and display response
            # receive_response() automatically waits until ResultMessage
            current_session_id = None
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Print Claude's text response in green
                            print(f"{Fore.GREEN}{block.text}{Style.RESET_ALL}", end="", flush=True)
                        elif isinstance(block, ToolUseBlock):
                            # Show when Claude is using a tool with magenta highlight
                            print(f"\n{Back.MAGENTA}{Fore.WHITE} 🔧 {block.name.upper()} {Style.RESET_ALL}", flush=True)
                elif isinstance(message, ResultMessage):
                    # Capture session ID for persistence
                    current_session_id = message.session_id
                    print("\n")

            # Save session ID after interaction
            if current_session_id:
                save_session(current_session_id)


def main():
    """Entry point for the CLI application."""
    import sys

    # Check if user wants to resume
    resume = "--continue" in sys.argv or "-c" in sys.argv

    # Run the async chat loop
    try:
        asyncio.run(chat_loop(resume_session=resume))
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
