# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Router is an intelligent proxy that classifies incoming requests by complexity and routes them to appropriate LLM models to optimize cost and performance. It implements a 5-tier complexity classification system (super_easy, easy, medium, hard, super_hard) and supports multiple providers: Anthropic, OpenAI, Google Gemini, Kimi/Moonshot, DeepSeek, and local Ollama models.

## Development Commands

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull classifier model (if using local classification)
ollama pull qwen2.5:3b

# Start the server
python server.py
python server.py --port 4001 --log  # with logging
python server.py --openclaw         # for OpenClaw integration
python server.py --openclaw --pinch  # with context pruning (experimental)

# Test classifier directly
python classifier.py "your message"
python classifier.py --test         # run classifier test suite

# Health check
curl http://localhost:4001/health
```

## Architecture

### Core Files

- **server.py** - Main HTTP server with routing logic, provider handlers, and message format conversions. Handles `/v1/chat/completions` (OpenAI-compatible), `/health`, and `/v1/models` endpoints.
- **classifier.py** - Message classification engine supporting multiple backends (local Ollama, Anthropic, OpenAI, Google, Kimi). CLI interface for testing.
- **config.yaml** - Runtime configuration (created from config.yaml.example)
- **ROUTES.md** - Classification rules table that drives classifier behavior

### Request Flow

1. Client sends OpenAI-compatible request to `/v1/chat/completions`
2. Server classifies message using configured classifier provider
3. Tool routing logic applies minimum complexity floor or model override if tools present
4. Complexity maps to `provider:model` via config.yaml MODEL_MAP
5. Message format converts from OpenAI → provider-specific format
6. Provider handler calls target API
7. Response converts back to OpenAI format and streams via SSE

### Provider Handlers

All handlers in server.py share signature: `model, messages, max_tokens, system, api_key, tools`
- `call_anthropic_model()` - Anthropic API with OAuth token support
- `call_openai_model()` - OpenAI GPT + reasoning models (o1/o3/o4 auto-detected)
- `call_google_model()` - Google Gemini API
- `call_kimi_model()` - Kimi/Moonshot with thinking mode
- `call_local_model()` - Ollama local models
- `call_deepseek_model()` - DeepSeek (OpenAI-compatible endpoint)

### Message Format Conversions

The router normalizes all messages to Anthropic content block format internally. Key conversion functions:
- `convert_openai_to_anthropic_messages()` - Input conversion
- `convert_tool_use_to_openai()` / `convert_tool_results_openai_to_anthropic()` - Tool handling
- Provider-specific: `convert_to_gemini_format()`, Kimi thinking mode handling

### OAuth Token Handling

OAuth tokens (`sk-ant-oat*`) receive special treatment:
- Auto-detected by prefix and use Claude Code identity headers
- Tool names remapped to official Claude Code names (Read, Write, Bash, etc.) via `CLAUDE_CODE_TOOLS` dict
- Automatic system prompt injection for Claude Code identity

## Configuration

Model format: `provider:model` (e.g., `anthropic:claude-sonnet-4-20250514`, `openai:gpt-4o`)

Tool routing in config.yaml:
- `tools.min_complexity` - Floor for requests with tools (bumps lower tiers up)
- `tools.model` - Override model for all tool-using requests

Auth priority: OAuth tokens from request header > config file API keys

## Key Defaults

- Port: 4001, Host: 127.0.0.1
- HTTP timeout: 120s (Anthropic), 300s (others)
- Max tokens: 8192 if not specified
- Classification fallback: "medium" on error
- Tool ID pattern: `^[a-zA-Z0-9_-]+$` (sanitized for Anthropic)

## Pinch: Intelligent Context Pruning (EXPERIMENTAL)

> **⚠️ Experimental & Optional** - This feature is under active development. Enable only if you want to test embedding-based context pruning. May have rough edges.

Enable with `--pinch` flag. Scores message relevance using embeddings to intelligently prune context.

### How it works
1. Identifies query window (recent messages to protect)
2. Scores older messages against current query via cosine similarity
3. Applies keep/summarize/drop decisions based on thresholds
4. Falls back to position-based pruning if embeddings unavailable

### Embedding Configuration (CRITICAL)

**Fallback chain (matches OpenClaw):** local (Ollama) → OpenAI → Gemini

Pinch follows the same fallback strategy as OpenClaw's `memorySearch`. Both systems should use the same primary model for cache compatibility.

**Dimension warning:** Different models produce incompatible vectors:
- `nomic-embed-text` (Ollama): 768 dimensions
- `text-embedding-3-small` (OpenAI): 1536 dimensions
- `text-embedding-004` (Gemini): 768 dimensions

```yaml
# config.yaml - llmrouter
pinch:
  embeddings:
    ollama_url: "http://127.0.0.1:11434"
    model: "nomic-embed-text"  # Should match OpenClaw
    # Falls back to OpenAI (OPENAI_API_KEY) then Gemini (GEMINI_API_KEY)
```

```json5
// ~/.openclaw/openclaw.json - OpenClaw
"memorySearch": {
  "provider": "openai",
  "model": "nomic-embed-text",  // Should match llmrouter
  "remote": {
    "baseUrl": "http://127.0.0.1:11434/v1/",
    "apiKey": "ollama"
  },
  "fallback": "openai"  // or "gemini" - should match llmrouter's fallback
}
```

If all embedding providers fail, Pinch falls back to position-based pruning.

### Setup
```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Start with pinch enabled (experimental)
python server.py --openclaw --pinch
```
