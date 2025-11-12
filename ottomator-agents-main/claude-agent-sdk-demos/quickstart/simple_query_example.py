"""
Simplest Possible Claude Agent SDK Example
===========================================

This demonstrates the absolute simplest way to use the Claude Agent SDK
using the stateless query() function. This version shows the RAW messages
as they stream in.

For multi-turn conversations with state, use ClaudeSDKClient (see simple_cli.py)
"""

import asyncio
from colorama import init, Fore, Style
from claude_agent_sdk import query, ClaudeAgentOptions

# Initialize colorama for cross-platform color support
init(autoreset=True)


async def main():
    """Simple one-shot query example showing raw messages."""

    # Configure options
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful Python expert.",
    )

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.CYAN}Claude Agent SDK - Simple Query Example")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"\n{Style.DIM}System: {options.system_prompt}")
    print(f"Query: Explain what async/await does in Python in one sentence.{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}{'─' * 60}{Style.RESET_ALL}\n")

    # Simple query - streams the raw messages
    async for message in query(
        prompt="Explain what async/await does in Python in one sentence.",
        options=options
    ):
        # Print raw message type and content
        print(f"{Fore.GREEN}[{type(message).__name__}]{Style.RESET_ALL}")
        print(message)
        print(f"\n{Fore.YELLOW}{'─' * 60}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}✓ Complete!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())
