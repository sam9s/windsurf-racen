# Implementation Plan: Telegram Bot Integration with Claude Agent SDK

## Overview
Create a Telegram bot that enables users to interact with Claude Code through Telegram direct messages. The bot will manage per-user sessions, working directories, and provide seamless integration with the Claude Agent SDK, similar to the existing API server and CLI implementations.

## Requirements Summary
- ✅ Accept messages from Telegram users in direct messages
- ✅ Send messages to Claude Agent SDK and return responses
- ✅ Per-user session management (each user maintains their own conversation)
- ✅ Per-user working directory configuration
- ✅ Commands for setting and managing working directory
- ✅ Directory search assistance to help users find directories
- ✅ Store session data and configuration locally (similar to api_sessions pattern)
- ✅ Handle streaming responses from Claude back to Telegram
- ✅ Update .env.example with TELEGRAM_BOT_API_KEY
- ✅ Update README.md with documentation

## Research Findings

### Best Practices

**Telegram Bot Architecture**
- Use `python-telegram-bot` library (v22.5+) with native asyncio support
- ApplicationBuilder pattern for initialization: `ApplicationBuilder().token("...").build()`
- CommandHandler for command routing: `/start`, `/setcwd`, `/help`, etc.
- MessageHandler for regular text messages
- Per-user state management using user_id as key
- Privacy mode enabled by default (works great for DM-only bots)

**Session Management Pattern**
- Store per-user data in `telegram_sessions/{user_id}.json`
- Each file contains: `{"session_id": "...", "cwd": "/path/..."}`
- Load session on each message, save after Claude response
- Similar pattern to existing `api_sessions/current.json`

**Claude Agent SDK Integration**
- Use `ClaudeSDKClient` with `ClaudeAgentOptions`
- Set `cwd` option per user's configured directory
- Resume sessions using `resume` parameter
- Stream responses using `receive_response()` for better UX
- Capture `session_id` from `ResultMessage` for persistence

### Reference Implementations

**Existing Code Patterns**
1. `api_server.py` (lines 85-104) - Session management with save/load functions
2. `simple_cli.py` (lines 41-54) - Session persistence pattern
3. `api_server.py` (lines 155-234) - Streaming response handling
4. `api_server.py` (lines 187-203) - ClaudeAgentOptions configuration

**python-telegram-bot Examples**
- Basic async handler: `async def command(update, context): await update.message.reply_text("...")`
- Command with args: Access via `context.args` list
- Application run: `app.run_polling()` for continuous listening

### Technology Decisions

**Library Choice: python-telegram-bot**
- Rationale: Official Python library, active maintenance, native asyncio, comprehensive documentation
- Version: 22.5+ (released Sept 2025)
- Fully async/await compatible with Claude Agent SDK

**Session Scope: Per-User**
- Rationale: Each user maintains independent conversation context
- Key: Telegram `user_id` ensures uniqueness across chats
- Storage: JSON files in `telegram_sessions/` directory

**Message Context: Direct Messages Only (v1)**
- Rationale: Simplifies session management, avoids privacy mode complexity
- Future: Can expand to group support with per-chat sessions

**Working Directory Management**
- Per-user configuration stored alongside session
- Default to `WORKING_DIRECTORY` env var if user hasn't set custom cwd
- Commands for setting and viewing cwd

## Implementation Tasks

### Phase 1: Foundation Setup

1. **Add Telegram Dependency**
   - Description: Add python-telegram-bot to requirements.txt
   - Files to modify: `requirements.txt`
   - Dependencies: None
   - Estimated effort: 5 minutes

2. **Update Environment Configuration**
   - Description: Add TELEGRAM_BOT_API_KEY to .env.example with comments
   - Files to modify: `.env.example`
   - Dependencies: None
   - Estimated effort: 5 minutes

3. **Create Session Storage Directory**
   - Description: Create telegram_sessions/ directory structure
   - Files to create: `telegram_sessions/.gitkeep`
   - Dependencies: None
   - Estimated effort: 2 minutes

### Phase 2: Core Bot Implementation

4. **Create Telegram Bot Script**
   - Description: Create main bot script with ApplicationBuilder setup
   - Files to create: `telegram_bot.py`
   - Dependencies: Tasks 1-3 complete
   - Estimated effort: 30 minutes
   - Details:
     - Import telegram and Claude SDK dependencies
     - Load TELEGRAM_BOT_API_KEY from .env
     - Initialize ApplicationBuilder
     - Create main() function with app.run_polling()

5. **Implement Session Management Functions**
   - Description: Create save_user_session() and load_user_session() functions
   - Files to modify: `telegram_bot.py`
   - Dependencies: Task 4 complete
   - Estimated effort: 20 minutes
   - Details:
     - Mirror pattern from api_server.py (lines 89-103)
     - Use user_id as filename: `telegram_sessions/{user_id}.json`
     - Store both session_id and cwd in JSON
     - Create directory if it doesn't exist

6. **Implement Command Handlers**
   - Description: Create async handlers for /start, /help, /setcwd, /getcwd, /reset
   - Files to modify: `telegram_bot.py`
   - Dependencies: Task 5 complete
   - Estimated effort: 45 minutes
   - Details:
     - `/start`: Welcome message with instructions
     - `/help`: List available commands and features
     - `/setcwd <path>`: Validate path exists, save to user config
     - `/getcwd`: Show current working directory
     - `/reset`: Clear user's session, start fresh conversation

7. **Implement Message Handler**
   - Description: Create async handler for regular text messages that interfaces with Claude
   - Files to modify: `telegram_bot.py`
   - Dependencies: Task 6 complete
   - Estimated effort: 60 minutes
   - Details:
     - Load user's session and cwd configuration
     - Configure ClaudeAgentOptions with user's cwd
     - Create ClaudeSDKClient with resume if session exists
     - Send user message via client.query()
     - Stream response using receive_response()
     - Send Claude's response back to Telegram
     - Save session_id after interaction

### Phase 3: Response Handling & UX

8. **Implement Response Streaming**
   - Description: Handle Claude's streaming responses and format for Telegram
   - Files to modify: `telegram_bot.py`
   - Dependencies: Task 7 complete
   - Estimated effort: 45 minutes
   - Details:
     - Collect AssistantMessage TextBlocks
     - Show tool use indicators (🔧 TOOL_NAME)
     - Send complete response after receiving ResultMessage
     - Handle long messages (Telegram 4096 char limit)

9. **Add Error Handling**
   - Description: Comprehensive error handling for network, Claude API, and file system errors
   - Files to modify: `telegram_bot.py`
   - Dependencies: Task 8 complete
   - Estimated effort: 30 minutes
   - Details:
     - Try-except around Claude SDK calls
     - User-friendly error messages for common issues
     - Log errors for debugging
     - Graceful degradation

10. **Add Directory Search Helper**
    - Description: Implement /searchcwd command to help users find directories
    - Files to modify: `telegram_bot.py`
    - Dependencies: Task 6 complete
    - Estimated effort: 30 minutes
    - Details:
      - Use os.walk or pathlib to search common locations
      - Return list of matching directories
      - Allow user to select from results
      - Limit search depth to avoid performance issues

### Phase 4: Documentation & Testing

11. **Update README.md**
    - Description: Add Example 4: Telegram Bot section to README
    - Files to modify: `README.md`
    - Dependencies: Tasks 4-10 complete
    - Estimated effort: 45 minutes
    - Details:
      - Add overview and features section
      - Document setup and configuration
      - Provide usage examples with commands
      - Add example conversation screenshots (text format)
      - Include troubleshooting tips

12. **Create Usage Examples**
    - Description: Document example conversations and use cases
    - Files to modify: `README.md`
    - Dependencies: Task 11
    - Estimated effort: 20 minutes
    - Details:
      - Example: Basic chat interaction
      - Example: Setting working directory
      - Example: File operations with Claude
      - Example: Multi-turn conversation with context

13. **Manual Testing**
    - Description: Test bot with real Telegram account
    - Files to modify: None (testing only)
    - Dependencies: All previous tasks complete
    - Estimated effort: 60 minutes
    - Details:
      - Test all commands
      - Test multi-turn conversations
      - Test session persistence across restarts
      - Test working directory functionality
      - Test error scenarios

## Codebase Integration Points

### Files to Modify

- `requirements.txt` - Add `python-telegram-bot>=22.5`
- `.env.example` - Add `TELEGRAM_BOT_API_KEY=your_telegram_bot_token_here` with explanatory comments
- `README.md` - Add comprehensive section for Telegram bot (after API Server section)
  - Follow existing structure pattern
  - Include features, usage, commands, and examples

### New Files to Create

- `telegram_bot.py` - Main bot implementation (~300-400 lines)
  - Session management functions (similar to api_server.py lines 89-103)
  - Command handlers (async functions)
  - Message handler with Claude integration
  - ApplicationBuilder setup and run_polling()

- `telegram_sessions/.gitkeep` - Ensure directory exists in git
- `telegram_sessions/{user_id}.json` - Per-user session files (created at runtime)

### Existing Patterns to Follow

**Session Management Pattern** (from api_server.py)
```python
def save_session(conversation_id: str, session_id: str):
    session_file = SESSIONS_DIR / f"{conversation_id}.json"
    with open(session_file, "w") as f:
        json.dump({"session_id": session_id}, f)

def load_session(conversation_id: str) -> Optional[str]:
    session_file = SESSIONS_DIR / f"{conversation_id}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            data = json.load(f)
            return data.get("session_id")
    return None
```

**Claude Agent Options Configuration** (from api_server.py lines 188-203)
```python
working_dir = os.getenv("WORKING_DIRECTORY", os.getcwd())
options_dict = {
    "cwd": working_dir,
    "system_prompt": "You are a helpful AI assistant...",
    "allowed_tools": ["Read", "Write", "Bash", "Edit"],
}
if session_id:
    options_dict["resume"] = session_id

options = ClaudeAgentOptions(**options_dict)
```

**Response Handling Pattern** (from api_server.py lines 244-250 & simple_cli.py lines 118-130)
```python
async with ClaudeSDKClient(options=options) as client:
    await client.query(user_query)

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            # Handle text blocks and tool use
        if isinstance(message, ResultMessage):
            new_session_id = message.session_id
```

## Technical Design

### Architecture Diagram

```
┌─────────────┐
│   Telegram  │
│    User     │
└──────┬──────┘
       │ Messages
       ↓
┌─────────────────────────────────────┐
│     telegram_bot.py                 │
│  ┌───────────────────────────────┐  │
│  │  ApplicationBuilder           │  │
│  │  - CommandHandlers            │  │
│  │  - MessageHandler             │  │
│  └───────────┬───────────────────┘  │
│              │                       │
│  ┌───────────↓───────────────────┐  │
│  │  Session Manager              │  │
│  │  - load_user_session()        │  │
│  │  - save_user_session()        │  │
│  └───────────┬───────────────────┘  │
│              │                       │
│  ┌───────────↓───────────────────┐  │
│  │  Claude SDK Integration       │  │
│  │  - ClaudeAgentOptions         │  │
│  │  - ClaudeSDKClient            │  │
│  │  - receive_response()         │  │
│  └───────────┬───────────────────┘  │
└──────────────┼───────────────────────┘
               │
               ↓
┌──────────────────────────────────────┐
│  telegram_sessions/                  │
│  ├── 123456789.json (user 1)        │
│  ├── 987654321.json (user 2)        │
│  └── ...                             │
└──────────────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────┐
│  Claude Agent SDK                    │
│  (Managed by SDK - ~/.claude/)       │
└──────────────────────────────────────┘
```

### Data Flow

1. **User sends message** → Telegram delivers update to bot
2. **Bot receives update** → Extract user_id and message text
3. **Load session** → Read `telegram_sessions/{user_id}.json` for session_id and cwd
4. **Configure Claude** → Create ClaudeAgentOptions with user's cwd and session_id (if exists)
5. **Query Claude** → Send user message via ClaudeSDKClient.query()
6. **Stream response** → Collect AssistantMessage blocks from receive_response()
7. **Format response** → Build complete message with text and tool indicators
8. **Send to user** → Send formatted response back to Telegram
9. **Save session** → Update `telegram_sessions/{user_id}.json` with new session_id

### Bot Commands

```
/start - Initialize bot and show welcome message
  Response: "Welcome! I'm Claude Code, powered by Claude Sonnet 4.5..."

/help - Display available commands and usage
  Response: List of commands with descriptions

/setcwd <path> - Set working directory for file operations
  Args: Absolute path to directory
  Validation: Check if path exists and is directory
  Response: "✅ Working directory set to: {path}"

/getcwd - Show current working directory
  Response: Display current cwd or default if not set

/searchcwd <query> - Search for directories matching query
  Args: Directory name or partial path
  Response: List of matching directories with select options

/reset - Clear conversation session (start fresh)
  Response: "🔄 Session cleared. Starting fresh conversation."
  Action: Delete user's session file
```

### User Configuration Schema

File: `telegram_sessions/{user_id}.json`
```json
{
  "session_id": "session-uuid-from-claude-sdk",
  "cwd": "/path/to/working/directory",
  "created_at": "2025-10-13T12:00:00Z",
  "last_updated": "2025-10-13T12:30:00Z"
}
```

## Dependencies and Libraries

- `python-telegram-bot>=22.5` - Official Telegram Bot API wrapper
  - Purpose: Handle Telegram updates, messages, commands
  - Features: Async/await support, ApplicationBuilder, handlers

- `claude-agent-sdk` - Already installed (v0.1.2)
  - Purpose: Interface with Claude Code
  - Features: Session management, tool use, streaming responses

- `python-dotenv` - Already installed (v1.1.1)
  - Purpose: Load TELEGRAM_BOT_API_KEY from .env

- `pathlib` - Python stdlib
  - Purpose: Cross-platform path handling for cwd operations

## Testing Strategy

### Unit-Style Manual Tests

1. **Command Tests**
   - `/start` → Verify welcome message
   - `/help` → Verify all commands listed
   - `/setcwd /valid/path` → Verify success message
   - `/setcwd /invalid/path` → Verify error handling
   - `/getcwd` → Verify current directory displayed
   - `/reset` → Verify session cleared

2. **Message Flow Tests**
   - Send "Hello" → Verify Claude responds
   - Send "What's 2+2?" → Verify computational response
   - Send "List files in current directory" → Verify tool use indication

3. **Session Persistence Tests**
   - Start conversation with context
   - Restart bot script
   - Continue conversation → Verify context maintained

4. **Multi-User Tests**
   - User A and User B send messages
   - Verify separate sessions maintained
   - Verify no context bleeding between users

5. **Error Scenarios**
   - Invalid working directory
   - Claude API errors
   - Network interruptions
   - Invalid commands

### Edge Cases to Cover

- Telegram message length limit (4096 chars) - split long responses
- Empty messages from user
- Special characters in paths (Windows vs Unix)
- Concurrent messages from same user
- Session file corruption
- Missing .env configuration

## Success Criteria

- [x] Bot responds to direct messages from Telegram users
- [x] Each user maintains independent conversation context
- [x] Users can set and manage their working directory
- [x] All commands (`/start`, `/help`, `/setcwd`, `/getcwd`, `/reset`) work correctly
- [x] Claude's responses are properly formatted and delivered to Telegram
- [x] Tool use is indicated to users (e.g., "🔧 BASH")
- [x] Sessions persist across bot restarts
- [x] Long responses are handled gracefully (split if needed)
- [x] Error messages are user-friendly and informative
- [x] Documentation in README is clear and comprehensive
- [x] .env.example includes TELEGRAM_BOT_API_KEY with instructions

## Notes and Considerations

### Performance Considerations
- Telegram has rate limits: 1 message/second per chat, 20 messages/minute in groups
- Claude SDK streaming is fast but Telegram delivery may add latency
- Consider showing "typing..." indicator during Claude processing

### Security Considerations
- Telegram bot token is sensitive - keep in .env, never commit
- User's working directory access - validate paths to prevent directory traversal
- Consider adding user allowlist if bot becomes public

### Future Enhancements (Out of Scope for v1)
- Group chat support with per-chat sessions
- Inline query support for quick answers
- File upload handling (send files to Claude)
- Rich formatting (Markdown/HTML in responses)
- Webhook mode instead of polling (for production)
- Database storage instead of JSON files (for scale)
- `/searchcwd` with fuzzy matching and common locations
- Conversation history display (requires SDK enhancement)

### Known Limitations
- Cannot display conversation history programmatically (SDK limitation)
- Users must know/find their directory paths (partially addressed by /searchcwd)
- Polling mode may have slight delay (webhook mode is faster but more complex)
- Telegram 4096 char limit requires message splitting

---
*This plan is ready for execution with `/execute-plan PRPs/planning/telegram-bot-integration.md`*
