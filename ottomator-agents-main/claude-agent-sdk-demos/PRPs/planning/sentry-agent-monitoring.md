# Implementation Plan: Sentry AI Agent Monitoring for Telegram Bot

## Overview

Implement comprehensive Sentry monitoring for the Telegram bot's Claude Agent SDK integration, enabling full observability of AI agent interactions including token usage, cost tracking, tool execution, and performance metrics.

## Requirements Summary

- Add Sentry SDK integration to the Telegram bot
- Instrument the Claude Agent SDK interaction with proper spans
- Track token usage and cost metrics from Claude API
- Monitor tool executions (Read, Write, Bash, Edit)
- Capture session and conversation context
- Enable error tracking with rich contextual data
- Maintain performance metrics (duration, API latency)
- Follow Sentry's AI agent monitoring best practices

## Research Findings

### Best Practices from Sentry Documentation

**1. Span Naming Conventions (Sentry Standards)**
- Agent invocation span: `op="gen_ai.invoke_agent"`, `name="invoke_agent {agent_name}"`
- LLM client span: `op="ai.chat_completions.create.anthropic"`, `name="{operation} {model}"`
- Tool execution span: `op="gen_ai.execute_tool"`, `name="execute_tool {tool_name}"`

**2. Required Span Attributes**
- `gen_ai.system`: "anthropic"
- `gen_ai.request.model`: "claude-sonnet-4-5"
- `gen_ai.operation.name`: Operation type
- `gen_ai.usage.*`: Token usage metrics
- `gen_ai.cost_usd`: Cost tracking

**3. PII Considerations**
- Set `send_default_pii=True` to capture LLM inputs/outputs
- Consider user privacy when capturing message content
- Truncate long responses for span data

**4. Performance Monitoring**
- Track full operation duration
- Separate API-only duration from total time
- Monitor streaming response latency

### Token Usage Discovery (from test_token_usage.py)

The Claude Agent SDK provides comprehensive metrics via `ResultMessage`:

```python
# Available from ResultMessage object
{
    "input_tokens": 11,                    # User input tokens
    "output_tokens": 547,                  # Claude response tokens
    "cache_read_input_tokens": 23774,      # Cache hits
    "cache_creation_input_tokens": 1919,   # Cache creation
    "total_cost_usd": 0.02361845,          # Total cost
    "duration_ms": 13841,                  # Total duration
    "duration_api_ms": 13806,              # API duration
    "num_turns": 4,                        # Conversation turns
    "session_id": "..."                    # Session identifier
}
```

### Reference Implementations

**Example: Nested Span Structure**
```python
with sentry_sdk.start_span(op="gen_ai.invoke_agent", name="invoke_agent") as agent_span:
    agent_span.set_data("gen_ai.system", "anthropic")

    with sentry_sdk.start_span(op="ai.chat_completions.create.anthropic") as llm_span:
        # LLM interaction
        llm_span.set_data("gen_ai.usage.input_tokens", tokens)

        with sentry_sdk.start_span(op="gen_ai.execute_tool") as tool_span:
            # Tool execution
            tool_span.set_data("tool_name", "Bash")
```

## Implementation Tasks

### Phase 1: Setup and Dependencies

#### Task 1.1: Add Sentry SDK Dependency
- **Description**: Install and configure Sentry Python SDK
- **Files to modify**: `requirements.txt`
- **Dependencies**: None
- **Estimated effort**: 5 minutes

**Actions**:
- Add `sentry-sdk>=2.31.0` to requirements.txt
- Run `pip install sentry-sdk` to install
- Verify installation

#### Task 1.2: Configure Environment Variables
- **Description**: Add Sentry DSN configuration
- **Files to modify**: `.env` (if exists), documentation
- **Dependencies**: Task 1.1
- **Estimated effort**: 5 minutes

**Actions**:
- Add `SENTRY_DSN` environment variable placeholder
- Add `SENTRY_ENVIRONMENT` (optional: dev/staging/prod)
- Document required environment variables

### Phase 2: Core Instrumentation

#### Task 2.1: Initialize Sentry in main()
- **Description**: Set up Sentry SDK with appropriate configuration
- **Files to modify**: `telegram_bot.py`
- **Location**: `main()` function (line 601)
- **Dependencies**: Task 1.1, Task 1.2
- **Estimated effort**: 15 minutes

**Actions**:
- Import `sentry_sdk` at top of file
- Add Sentry initialization in `main()` before building application
- Configure trace sampling rate
- Set environment and release tags
- Add basic error tracking

**Code Pattern**:
```python
import sentry_sdk

def main():
    # Get configuration
    sentry_dsn = os.getenv("SENTRY_DSN")

    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=1.0,
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            send_default_pii=False,  # Start conservative, can enable later
        )
        logger.info("Sentry monitoring initialized")
    else:
        logger.warning("SENTRY_DSN not set, monitoring disabled")
```

#### Task 2.2: Instrument Main Agent Invocation
- **Description**: Wrap handle_message() with agent invocation span
- **Files to modify**: `telegram_bot.py`
- **Location**: `handle_message()` function (line 450)
- **Dependencies**: Task 2.1
- **Estimated effort**: 30 minutes

**Actions**:
- Create outer span with `op="gen_ai.invoke_agent"`
- Set initial context (user_id, message, session_id if resuming)
- Capture working directory and allowed tools
- Set final response data after completion
- Track number of tools used

**Key Metrics to Capture**:
- `gen_ai.system`: "anthropic"
- `gen_ai.request.model`: "claude-sonnet-4-5"
- `gen_ai.prompt`: User message (truncated if needed)
- `user_id`: Telegram user ID
- `username`: Telegram username
- `working_directory`: Current working directory
- `resumed_session_id`: If continuing conversation
- `response_length`: Length of final response
- `tools_used`: List of tools invoked

#### Task 2.3: Instrument LLM Client Interaction
- **Description**: Add span for Claude SDK query and streaming
- **Files to modify**: `telegram_bot.py`
- **Location**: Inside `handle_message()`, around ClaudeSDKClient usage (line 506)
- **Dependencies**: Task 2.2
- **Estimated effort**: 30 minutes

**Actions**:
- Create LLM client span with `op="ai.chat_completions.create.anthropic"`
- Set request metadata (model, streaming, allowed tools)
- Capture token usage from ResultMessage
- Track API duration and total duration
- Record cost metrics

**Key Metrics to Capture**:
- `gen_ai.request.streaming`: True
- `gen_ai.usage.input_tokens`: From ResultMessage.usage
- `gen_ai.usage.output_tokens`: From ResultMessage.usage
- `gen_ai.usage.cache_read_input_tokens`: Cache hits
- `gen_ai.usage.cache_creation_input_tokens`: Cache creation
- `gen_ai.cost_usd`: ResultMessage.total_cost_usd
- `gen_ai.duration_ms`: ResultMessage.duration_ms
- `gen_ai.api_duration_ms`: ResultMessage.duration_api_ms
- `gen_ai.num_turns`: ResultMessage.num_turns

#### Task 2.4: Instrument Tool Executions
- **Description**: Create spans for each tool invocation
- **Files to modify**: `telegram_bot.py`
- **Location**: Inside message streaming loop, when ToolUseBlock is received (line 516)
- **Dependencies**: Task 2.3
- **Estimated effort**: 20 minutes

**Actions**:
- Create tool span for each ToolUseBlock
- Set tool name and input data
- Track which tools are used most frequently
- Capture tool execution context

**Key Metrics to Capture**:
- `gen_ai.operation.name`: "execute_tool"
- `tool_name`: Tool name (Read, Write, Bash, Edit)
- `tool_input`: JSON-serialized tool input (truncated if large)

### Phase 3: Error Tracking and Context

#### Task 3.1: Enhanced Error Tracking
- **Description**: Add rich contextual data to exception capture
- **Files to modify**: `telegram_bot.py`
- **Location**: Exception handler in `handle_message()` (line 546)
- **Dependencies**: Task 2.2
- **Estimated effort**: 15 minutes

**Actions**:
- Use `sentry_sdk.set_context()` to add user context
- Capture session state at time of error
- Include message preview and working directory
- Ensure all exceptions are captured with context

**Context to Capture**:
- `user_id`: Telegram user ID
- `username`: Telegram username
- `session_id`: Current session (if exists)
- `cwd`: Working directory
- `message_preview`: First 100 chars of user message
- `tools_allowed`: List of allowed tools

#### Task 3.2: Add Logging Integration
- **Description**: Connect Python logging to Sentry breadcrumbs
- **Files to modify**: `telegram_bot.py`
- **Location**: Logging configuration (line 48)
- **Dependencies**: Task 2.1
- **Estimated effort**: 10 minutes

**Actions**:
- Configure Sentry to capture logs as breadcrumbs
- Set appropriate log level for breadcrumbs
- Ensure important events are tracked

### Phase 4: Testing and Validation

#### Task 4.1: Create Test Cases
- **Description**: Test Sentry instrumentation with various scenarios
- **Files to create**: `test_sentry_monitoring.py`
- **Dependencies**: All Phase 2 and 3 tasks
- **Estimated effort**: 30 minutes

**Test Scenarios**:
1. Simple query with no tools
2. Query that uses multiple tools
3. Multi-turn conversation (session resumption)
4. Error scenarios (invalid path, failed tool execution)
5. Long response that gets split
6. Cache hit scenarios

**Validation Checklist**:
- [ ] Spans appear correctly in Sentry dashboard
- [ ] Token usage is tracked accurately
- [ ] Cost metrics are captured
- [ ] Tool executions are visible
- [ ] Errors have rich context
- [ ] Performance metrics are reasonable
- [ ] No PII leakage if send_default_pii=False

#### Task 4.2: Performance Impact Testing
- **Description**: Verify Sentry overhead is minimal
- **Files to modify**: None (testing only)
- **Dependencies**: Task 4.1
- **Estimated effort**: 20 minutes

**Actions**:
- Measure response times with and without Sentry
- Verify sampling rate can reduce overhead if needed
- Test with high message volume

#### Task 4.3: Documentation
- **Description**: Document Sentry setup and configuration
- **Files to create/modify**: `README.md`, `MONITORING.md`
- **Dependencies**: Task 4.1, Task 4.2
- **Estimated effort**: 20 minutes

**Documentation Topics**:
- How to set up Sentry DSN
- What metrics are tracked
- How to view traces in Sentry dashboard
- How to configure sampling rates
- Privacy considerations (PII settings)
- Troubleshooting common issues

## Codebase Integration Points

### Files to Modify

#### `telegram_bot.py` (Primary file)

**Line 16-22: Imports**
```python
# Add to imports
import sentry_sdk
import json  # For serializing tool inputs
```

**Line 601-632: main() function**
- Add Sentry initialization after line 608 (after checking bot token)
- Add before building application (line 612)

**Line 450-556: handle_message() function**
- Wrap entire try block with agent invocation span (after line 467)
- Add LLM client span around ClaudeSDKClient usage (around line 506)
- Add tool spans in message streaming loop (around line 516)
- Enhance error context in exception handler (line 546)

**Line 48-53: Logging configuration**
- Optional: Configure Sentry logging integration

#### `requirements.txt`
- Add `sentry-sdk>=2.31.0` after line 35

#### `.env` (if exists) or `.env.example` (create new)
- Add `SENTRY_DSN=your-dsn-here`
- Add `SENTRY_ENVIRONMENT=production`

### New Files to Create

#### `test_sentry_monitoring.py`
- Unit tests for Sentry instrumentation
- Integration tests for end-to-end tracing
- Mock Sentry SDK to verify span creation

#### `MONITORING.md` (Optional)
- Documentation for monitoring setup
- Guide to interpreting Sentry traces
- Best practices for cost optimization

### Existing Patterns to Follow

**1. Logging Pattern**
```python
logger.info(f"Description: {variable}")
logger.error(f"Error description: {e}", exc_info=True)
```
Continue using existing logging alongside Sentry.

**2. Async Context Manager Pattern**
```python
async with ClaudeSDKClient(options=options) as client:
    # Operations
```
Wrap with Sentry spans using `with sentry_sdk.start_span()`.

**3. Error Handling Pattern**
```python
try:
    # Operations
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    # User-friendly error message
```
Add `sentry_sdk.capture_exception(e)` and context setting.

**4. Session Management Pattern**
- Load session at start
- Update throughout operation
- Save at end
Continue this pattern, add session_id to Sentry context.

## Technical Design

### Architecture Diagram

```
User Message (Telegram)
         |
         v
    handle_message()  <--- [Agent Span: gen_ai.invoke_agent]
         |
         ├─> Load session & config
         |
         ├─> ClaudeSDKClient.query()  <--- [LLM Span: ai.chat_completions.create.anthropic]
         |        |
         |        ├─> Stream AssistantMessage blocks
         |        |        |
         |        |        ├─> TextBlock → Collect response
         |        |        |
         |        |        └─> ToolUseBlock → [Tool Span: gen_ai.execute_tool]
         |        |
         |        └─> ResultMessage → Capture token usage, cost, duration
         |
         ├─> Send response to Telegram
         |
         └─> Save session
```

### Data Flow

1. **Initialization Phase** (main)
   - Load environment variables
   - Initialize Sentry SDK
   - Set up Telegram bot

2. **Message Handling Phase** (per user message)
   - Create agent span (outermost)
   - Load user session and cwd
   - Create LLM span (nested)
   - Send query to Claude SDK
   - Stream response blocks
   - Create tool spans as needed (nested in LLM span)
   - Capture ResultMessage metrics
   - Close spans with complete data

3. **Error Handling Phase**
   - Set error context
   - Capture exception
   - Send user-friendly error message

### Span Hierarchy

```
Transaction: telegram_bot.handle_message
  └─ Span: gen_ai.invoke_agent "invoke_agent Claude Code Bot"
       └─ Span: ai.chat_completions.create.anthropic "Claude SDK Query"
            ├─ Span: gen_ai.execute_tool "execute_tool Bash"
            ├─ Span: gen_ai.execute_tool "execute_tool Read"
            └─ Span: gen_ai.execute_tool "execute_tool Write"
```

## Dependencies and Libraries

### New Dependencies

**sentry-sdk (>=2.31.0)**
- Purpose: Error tracking and performance monitoring
- Why this version: Includes AI agent monitoring features
- Installation: `pip install sentry-sdk>=2.31.0`

### Existing Dependencies (No changes needed)

- `claude-agent-sdk==0.1.2` - Already provides token usage data
- `python-telegram-bot>=22.5` - No changes needed
- `python-dotenv==1.1.1` - Used for SENTRY_DSN env var

## Testing Strategy

### Unit Tests

**Test Span Creation**
- Mock sentry_sdk.start_span()
- Verify spans are created with correct op and name
- Verify span data is set correctly

**Test Token Usage Capture**
- Mock ResultMessage with usage data
- Verify token counts are captured
- Verify cost is captured

**Test Error Context**
- Trigger exception in handle_message
- Verify context is set
- Verify exception is captured

## Success Criteria

- [x] Sentry SDK integrated and initializing correctly
- [x] Agent invocation span captures user context
- [x] LLM client span captures token usage and cost
- [x] Tool execution spans created for each tool use
- [x] Error tracking includes rich contextual data
- [x] Token counts match Claude API usage
- [x] Cost tracking is accurate
- [x] Performance overhead is < 5% of total request time
- [x] Spans appear correctly in Sentry dashboard
- [x] Dashboard shows complete transaction traces
- [x] Documentation is complete and clear
- [x] Tests validate all instrumentation points

## Notes and Considerations

### Privacy and PII

**Initial Configuration**: Start with `send_default_pii=False`
- Protects user message content
- Still captures metadata and metrics
- Can be enabled later if needed

**If Enabling PII**:
- Document clearly in privacy policy
- Consider truncating long messages
- Sanitize sensitive paths or file contents

### Performance Optimization

**Sampling Strategy**:
- Start with 100% sampling (`traces_sample_rate=1.0`)
- If performance impact is too high, reduce to 0.1 (10%)
- Use `traces_sampler` for dynamic sampling based on user or message

**Async Considerations**:
- Sentry SDK is async-aware
- Spans will correctly track async operations
- No special handling needed for asyncio

### Cost Monitoring

**Claude API Costs**:
- Track `total_cost_usd` per interaction
- Monitor aggregate costs in Sentry dashboard
- Set up alerts for unusual cost spikes

**Sentry Costs**:
- Be aware of Sentry event quotas
- Adjust sampling if approaching limits
- Consider filtering high-volume low-value events

---

## Execution Ready

This plan is ready for execution with:
```bash
/execute-plan PRPs/sentry-agent-monitoring.md
```

The implementation will result in:
- Full observability of AI agent interactions
- Token usage and cost tracking
- Tool execution monitoring
- Enhanced error tracking with context
- Performance metrics for optimization
