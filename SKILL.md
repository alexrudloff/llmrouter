---
name: llmrouter
description: Intelligent LLM proxy that routes requests to appropriate models based on complexity. Save money by using cheaper models for simple tasks. Currently tested with Anthropic only.
homepage: https://github.com/alexrudloff/llmrouter
metadata: {"openclaw":{"emoji":"🔀","homepage":"https://github.com/alexrudloff/llmrouter","os":["darwin","linux"],"requires":{"bins":["python3","ollama"],"anyBins":["pip","pip3"]},"primaryEnv":"ANTHROPIC_API_KEY"}}
---

# LLM Router

An intelligent proxy that classifies incoming requests by complexity and routes them to appropriate LLM models. Use cheaper/faster models for simple tasks and reserve expensive models for complex ones.

**Works with [OpenClaw](https://github.com/openclaw/openclaw)** to reduce token usage and API costs by routing simple requests to smaller models.

**Status:** Currently tested with Anthropic only. Other providers are implemented but untested.

## Quick Start

### Prerequisites

1. **Ollama** (optional - only if using local classification):
   ```bash
   ollama pull qwen2.5:3b
   ```

2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Anthropic API key** (or Claude Code OAuth token)

### Setup

```bash
# Clone if not already present
git clone https://github.com/alexrudloff/llmrouter.git
cd llmrouter

# Install dependencies
pip install -r requirements.txt

# Copy and customize config
cp config.yaml.example config.yaml
```

### Start the Server

```bash
python server.py
```

Options:
- `--port PORT` - Port to listen on (default: 4001)
- `--host HOST` - Host to bind (default: 127.0.0.1)
- `--config PATH` - Config file path (default: config.yaml)
- `--log` - Enable verbose logging
- `--openclaw` - Enable OpenClaw compatibility mode

## Configuration

Edit `config.yaml` to customize:

### Model Routing
```yaml
models:
  super_easy: "anthropic:claude-haiku-4-5-20251001"
  easy: "anthropic:claude-haiku-4-5-20251001"
  medium: "anthropic:claude-sonnet-4-20250514"
  hard: "anthropic:claude-opus-4-20250514"
  super_hard: "anthropic:claude-opus-4-20250514"
```

### Classifier

Two options for classifying request complexity:

**Local (default)** - Free, requires Ollama:
```yaml
classifier:
  provider: "local"
  model: "qwen2.5:3b"
```

**Remote** - No local hardware needed, uses your API token:
```yaml
classifier:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
```

Use remote if your machine can't run local models or you want simpler setup.

### Supported Providers

- `anthropic:claude-*` - Anthropic Claude models (tested)
- `openai:gpt-*` - OpenAI GPT models (untested)
- `google:gemini-*` - Google Gemini models (untested)
- `local:model-name` - Local Ollama models (untested)

## Complexity Levels

| Level | Use Case | Default Model |
|-------|----------|---------------|
| super_easy | Greetings, acknowledgments | Haiku |
| easy | Simple Q&A, reminders | Haiku |
| medium | Coding, emails, research | Sonnet |
| hard | Complex reasoning, debugging | Opus |
| super_hard | System architecture, proofs | Opus |

## Customizing Classification

Edit `ROUTES.md` to tune how messages are classified. The classifier reads the table in this file to determine complexity levels.

## API Usage

The router exposes an OpenAI-compatible API:

```bash
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm-router",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Testing Classification

```bash
python classifier.py "Write a Python sort function"
# Output: medium

python classifier.py --test
# Runs test suite
```

## Running as macOS Service

Create `~/Library/LaunchAgents/com.llmrouter.plist` and load with:
```bash
launchctl load ~/Library/LaunchAgents/com.llmrouter.plist
```

## Common Tasks

- **Check server status**: `curl http://localhost:4001/health`
- **View current config**: `cat config.yaml`
- **Test a classification**: `python classifier.py "your message"`
- **Restart server**: Stop and run `python server.py` again
