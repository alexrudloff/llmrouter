---
name: llmrouter
description: Intelligent LLM proxy that routes requests to appropriate models based on complexity. Save money by using cheaper models for simple tasks. Tested with Anthropic, OpenAI, Gemini, Kimi/Moonshot, and Ollama.
homepage: https://github.com/alexrudloff/llmrouter
metadata: {"openclaw":{"emoji":"🔀","homepage":"https://github.com/alexrudloff/llmrouter","os":["darwin","linux"],"requires":{"bins":["python3"],"anyBins":["pip","pip3"]},"primaryEnv":"ANTHROPIC_API_KEY"}}
---

# LLM Router

An intelligent proxy that classifies incoming requests by complexity and routes them to appropriate LLM models. Use cheaper/faster models for simple tasks and reserve expensive models for complex ones.

**Works with [OpenClaw](https://github.com/openclaw/openclaw)** to reduce token usage and API costs by routing simple requests to smaller models.

**Status:** Tested with Anthropic, OpenAI (including o-series), Google Gemini, Kimi/Moonshot, and Ollama.

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
- `--openclaw` - Enable OpenClaw compatibility (rewrites model name in system prompt)

## Configuration

Edit `config.yaml` to customize:

### Model Routing

```yaml
# Anthropic routing
models:
  super_easy: "anthropic:claude-haiku-4-5-20251001"
  easy: "anthropic:claude-haiku-4-5-20251001"
  medium: "anthropic:claude-sonnet-4-20250514"
  hard: "anthropic:claude-opus-4-20250514"
  super_hard: "anthropic:claude-opus-4-20250514"

# OpenAI routing (including o-series reasoning models)
models:
  super_easy: "openai:gpt-4o-mini"
  easy: "openai:gpt-4o-mini"
  medium: "openai:gpt-4o"
  hard: "openai:o3-mini"
  super_hard: "openai:o3"

# Google Gemini routing
models:
  super_easy: "google:gemini-2.0-flash"
  easy: "google:gemini-2.0-flash"
  medium: "google:gemini-2.0-flash"
  hard: "google:gemini-2.0-flash"
  super_hard: "google:gemini-2.0-flash"
```

**Note:** o-series models (o1, o3, o4-mini) are auto-detected and use correct API params.

### Classifier

Three options for classifying request complexity:

**Local (default)** - Free, requires Ollama:
```yaml
classifier:
  provider: "local"
  model: "qwen2.5:3b"
```

**Anthropic** - Uses Haiku, fast and cheap:
```yaml
classifier:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
```

**OpenAI** - Uses GPT-4o-mini:
```yaml
classifier:
  provider: "openai"
  model: "gpt-4o-mini"
```

**Google** - Uses Gemini:
```yaml
classifier:
  provider: "google"
  model: "gemini-2.0-flash"
```

**Kimi** - Uses Moonshot:
```yaml
classifier:
  provider: "kimi"
  model: "moonshot-v1-8k"
```

Use remote (anthropic/openai/google/kimi) if your machine can't run local models.

### Supported Providers

- `anthropic:claude-*` - Anthropic Claude models (tested)
- `openai:gpt-*` - OpenAI GPT models (tested)
- `openai:o1-*`, `openai:o3-*`, `openai:o4-*` - OpenAI reasoning models (tested)
- `google:gemini-*` - Google Gemini models (tested)
- `kimi:kimi-k2.5`, `kimi:moonshot-*` - Kimi/Moonshot models (tested)
- `local:model-name` - Local Ollama models (tested)

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

## OpenClaw Configuration

Add to `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "localrouter": {
        "baseUrl": "http://localhost:4001/v1",
        "apiKey": "via-router",
        "api": "openai-completions",
        "models": [{
          "id": "llm-router",
          "name": "LLM Router",
          "input": ["text", "image"],
          "contextWindow": 200000,
          "maxTokens": 8192
        }]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "localrouter/llm-router"
      }
    }
  }
}
```

Start with `python server.py --openclaw` to display actual model names in OpenClaw.

## Common Tasks

- **Check server status**: `curl http://localhost:4001/health`
- **View current config**: `cat config.yaml`
- **Test a classification**: `python classifier.py "your message"`
- **Restart server**: Stop and run `python server.py` again
