# Claude Agent SDK Guide & Examples

Three comprehensive examples demonstrating the Claude Agent SDK for Python, organized from simple to advanced:

1. **Quickstart** - Simple examples to get started (stateless query & interactive CLI)
2. **Obsidian Integration** - OpenAI-compatible REST API for Obsidian Copilot
3. **Telegram Integration** - Full-featured Telegram bots with session management

## 📋 Prerequisites

- Python 3.10 or higher
- An Anthropic account (for authentication)
- Telegram or Obsidian if you want to use those integrations

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Authentication

You have two options for authentication:

**Option A: Claude CLI OAuth (Recommended)**
```bash
claude auth login
```

**Option B: API Key**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Get your API key from: https://console.anthropic.com/

---

## 📚 Quickstart Examples

Simple, educational examples to understand the Claude Agent SDK basics.

### Example 1: Simple Query (Stateless)

**Location**: `quickstart/simple_query_example.py`

The absolute simplest way to use the Claude Agent SDK for one-off queries.

**Usage:**
```bash
python quickstart/simple_query_example.py
```

This example uses the `query()` function which is perfect for:
- Single questions without conversation context
- Quick scripts and automations
- Learning the basics

For multi-turn conversations with state, see Example 2.

---

### Example 2: Terminal CLI (Interactive)

**Location**: `quickstart/simple_cli.py`

A simple interactive terminal chat interface with conversation persistence.

**Features:**
- Streaming responses in real-time
- Session management (save/resume conversations)
- Simple, educational code
- Full access to Claude's tools (file operations, bash commands, etc.)

**Usage:**

Start a new conversation:
```bash
python quickstart/simple_cli.py
```

Continue your last conversation:
```bash
python quickstart/simple_cli.py --continue
```

**How It Works:**

1. **Session Persistence**: Session IDs are saved to `quickstart/sessions/current_session.json`
2. **Streaming**: Messages are displayed in real-time as Claude generates them
3. **Message Types**: Handles text blocks, tool use, and results gracefully

**Example Session:**

```
============================================================
Claude Agent SDK - Simple Terminal Chat
============================================================
Type 'exit' or 'quit' to end the conversation

You: What files are in the current directory?

Claude: Let me check the current directory for you.

🔧 Bash

The current directory contains the following files:
- simple_cli.py
- simple_query_example.py
...
```

---

## 🔗 Obsidian Integration

OpenAI-compatible REST API powered by Claude Agent SDK, designed for seamless integration with Obsidian Copilot and other OpenAI-compatible clients.

**Location**: `obsidian_integration/`

### Features

- OpenAI `/v1/chat/completions` endpoint
- Streaming and non-streaming support
- Session management for conversation continuity
- CORS enabled for web clients
- Compatible with Obsidian Copilot and other OpenAI clients

### Usage

Start the server:
```bash
python obsidian_integration/api_server.py
```

The server will start on `http://localhost:8003` (configurable via `PORT` env var)

### Setting Up with Obsidian Copilot

#### Step 1: Install Obsidian

If you don't have Obsidian installed yet, download it from [obsidian.md](https://obsidian.md).

#### Step 2: Install Copilot Plugin

1. Open Obsidian and go to **Settings** (gear icon in the bottom left)
2. Navigate to **Community Plugins**
3. If this is your first time, click **Turn on community plugins**
4. Click **Browse** to open the Community Plugins marketplace
5. Search for **"Copilot"**
6. Find the official **Copilot** plugin by logancyang
7. Click **Install**
8. Once installed, toggle the plugin **ON** in your Community Plugins list

#### Step 3: Start the API Server

```bash
python obsidian_integration/api_server.py
```

The server will start on `http://localhost:8003`. Keep this running in the background.

#### Step 4: Configure Copilot to Use Claude

1. In Obsidian, go to **Settings → Copilot**
2. Under **API Provider**, select **Custom (OpenAI-compatible)**
3. Configure the following settings:
   - **API Base URL**: `http://localhost:8003/v1`
   - **API Key**: Enter any text (e.g., "dummy-key" - not used but required by the plugin)
   - **Model**: Doesn't matter, won't be used
4. Click **Save**

#### Step 5: Start Using Claude in Obsidian!

You can now use Claude directly in Obsidian through:
- The Copilot chat panel (open with the command palette: `Ctrl/Cmd + P` → "Copilot: Toggle Chat")
- Inline assistance in your notes
- Note analysis and summarization

**Note**: The API documentation is available at http://localhost:8003/docs if you need to integrate with other tools.

### How It Works

- **OpenAI Converter** (`openai_converter.py`): Converts Claude SDK messages to OpenAI-compatible format
- **Session Storage**: Sessions are stored in `obsidian_integration/api_sessions/`
- **Conversation Continuity**: Multi-message conversations are automatically detected and resumed

---

## 💬 Telegram Integration

Two powerful Telegram bot implementations that bring Claude Code directly to Telegram.

**Location**: `telegram_integration/`

### telegram_bot.py (Recommended)

**Location**: `telegram_integration/telegram_bot.py`

A clean, streamlined Telegram bot implementation without Sentry monitoring.

**Features:**
- **Per-user sessions**: Each user maintains their own conversation context
- **Per-user working directories**: Set custom working directories for file operations
- **Full tool access**: Claude can read, write, and execute commands in your configured directory
- **Session persistence**: Conversations persist across bot restarts
- **Directory search**: Built-in helper to find and set working directories
- **Private conversations**: All sessions stored locally, no cloud databases

---

### telegram_bot_sentry.py (With Sentry)

**Location**: `telegram_integration/telegram_bot_sentry.py`

Enhanced version with comprehensive Sentry monitoring for error tracking and performance observability.

**Additional Features:**
- Track token usage and costs per interaction
- Monitor tool execution (Read, Write, Bash, Edit)
- Capture errors with rich context
- Performance monitoring and distributed tracing
- View complete AI agent interaction flows

---

### Setup (Both Bots)

1. **Create a Telegram Bot**:
   - Message [@BotFather](https://t.me/BotFather) on Telegram
   - Use `/newbot` command and follow instructions
   - Copy the bot token you receive

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add:
   TELEGRAM_BOT_API_KEY=your_bot_token_here
   ```

3. **For telegram_bot_sentry.py with Sentry (Optional)**:
   ```bash
   # Add to .env:
   SENTRY_DSN=https://your-key@sentry.io/project-id
   SENTRY_ENVIRONMENT=production
   ```

4. **Start the Bot**:
   ```bash
   # Basic bot (recommended)
   python telegram_integration/telegram_bot.py

   # OR with Sentry monitoring
   python telegram_integration/telegram_bot_sentry.py
   ```

### Usage

Once the bot is running, find it on Telegram and start chatting!

**Available Commands**:
- `/start` - Welcome message and introduction
- `/help` - Show available commands and usage
- `/setcwd <path>` - Set your working directory for file operations
- `/getcwd` - Show your current working directory
- `/searchcwd <query>` - Search for directories matching a pattern
- `/reset` - Clear conversation history (keeps working directory)

**Example Conversations**:

```
You: /start
Bot: 👋 Welcome to Claude Code Bot!
     I'm powered by Claude Sonnet 4.5...

You: /setcwd C:\Users\YourName\Projects\MyProject
Bot: ✅ Working directory set to:
     C:\Users\YourName\Projects\MyProject

You: List all Python files in the current directory
Bot: Let me check that for you.
     🔧 BASH
     The current directory contains 12 Python files:
     - main.py
     - utils.py
     ...

You: Read the contents of main.py
Bot: 🔧 READ
     Here's the content of main.py:
     [file contents displayed]
```

### How It Works

1. **User Isolation**: Each Telegram user gets their own session file in `telegram_sessions/{user_id}.json`
2. **Session Storage**: Contains both Claude session ID and user's working directory preference
3. **Context Persistence**: Conversations maintain context across multiple messages
4. **Tool Integration**: Claude can use Read, Write, Bash, and Edit tools in the user's configured directory

### Session Storage

```
telegram_sessions/
├── 123456789.json    # User 1's session
├── 987654321.json    # User 2's session
└── .gitkeep
```

Each session file contains:
```json
{
  "session_id": "session-uuid-from-claude",
  "cwd": "/path/to/user/working/directory",
  "created_at": "2025-10-13T12:00:00Z",
  "last_updated": "2025-10-13T12:30:00Z"
}
```

### Security Notes

- **Bot Token**: Keep your `TELEGRAM_BOT_API_KEY` secret and never commit it to version control
- **Working Directory Access**: Users can only access directories they explicitly configure
- **Private Chats**: The bot works in direct messages (DMs) only
- **Local Storage**: All data is stored locally on your machine

---

## 📁 Project Structure

```
claude-agent-sdk-demos/
├── quickstart/                          # Basic examples
│   ├── simple_query_example.py         # Simplest stateless example
│   ├── simple_cli.py                   # Terminal chat interface
│   └── sessions/                       # CLI session storage
│       └── current_session.json
│
├── obsidian_integration/                # Obsidian Copilot integration
│   ├── api_server.py                   # FastAPI server
│   ├── openai_converter.py             # OpenAI format converter
│   └── api_sessions/                   # API session storage
│       └── current.json
│
├── telegram_integration/                # Telegram bot implementations
│   ├── telegram_bot.py                 # Basic bot (recommended)
│   ├── telegram_bot_sentry.py          # Bot with Sentry monitoring
│   └── tests/                          # Test files
│       ├── test_telegram_bot.py
│       └── test_sentry_monitoring.py
│
├── telegram_sessions/                   # Telegram bot session storage
│   ├── {user_id}.json
│   └── .gitkeep
│
├── requirements.txt                     # Python dependencies
├── .env.example                        # Environment variables template
├── README.md                           # This file
└── MONITORING.md                       # Sentry monitoring guide (optional)
```

---

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available options:

- `ANTHROPIC_API_KEY` - Your Anthropic API key (optional if using CLI OAuth)
- `PORT` - API server port (default: 8003)
- `WORKING_DIRECTORY` - Custom working directory for file operations
- `TELEGRAM_BOT_API_KEY` - Telegram bot token (required for telegram bots)
- `SENTRY_DSN` - Sentry DSN for monitoring (optional, for telegram_bot_sentry.py)
- `SENTRY_ENVIRONMENT` - Sentry environment tag (optional, default: production)

### Claude Agent Options

All examples support customization via `ClaudeAgentOptions`:

```python
options = ClaudeAgentOptions(
    cwd="/path/to/working/directory",
    system_prompt="You are a helpful assistant...",
    allowed_tools=["Read", "Write", "Bash"],  # Limit tools
    # resume="session_id_here"  # Resume a conversation
)
```

---

## 🎓 Learning Resources

### Key Concepts

1. **Two Interaction Methods**:
   - `query()`: Stateless, one-off queries (see `quickstart/simple_query_example.py`)
   - `ClaudeSDKClient`: Stateful conversations with context (see `quickstart/simple_cli.py`)

2. **Response Methods**:
   - `receive_response()`: Wait for complete response (recommended for turn-based chat)
   - `receive_messages()`: Stream messages in real-time (advanced use cases)

3. **Session Management**: Sessions maintain conversation context across multiple requests

4. **Streaming**: Responses are delivered incrementally as they're generated

5. **Tool Use**: Claude can use tools (file operations, bash commands, etc.)

6. **Message Types**:
   - `AssistantMessage`: Claude's responses
   - `TextBlock`: Text content
   - `ToolUseBlock`: Tool invocations
   - `ResultMessage`: Final message with session ID

### SDK Documentation

- Official docs: https://docs.claude.com/en/api/agent-sdk/python
- GitHub: https://github.com/anthropics/anthropic-sdk-python

---
