---
name: llmrouter
description: Intelligent LLM proxy that routes requests to appropriate models based on complexity. Save money by using cheaper models for simple tasks.
homepage: https://github.com/alexrudloff/llmrouter
metadata: {"openclaw":{"emoji":"🔀","homepage":"https://github.com/alexrudloff/llmrouter","os":["darwin","linux"],"requires":{"bins":["python3","ollama"],"anyBins":["pip","pip3"]},"primaryEnv":"ANTHROPIC_API_KEY"}}
---

# LLM Router

An intelligent proxy that classifies incoming requests by complexity and routes them to appropriate LLM models. Use cheaper/faster models for simple tasks and reserve expensive models for complex ones.

## Quick Start

### Prerequisites

1. **Ollama** running locally with the classifier model:
   ```bash
   ollama pull qwen2.5:3b
   ```

2. **Python dependencies**:
   ```bash
   pip install pyyaml requests
   ```

3. **API key** for your chosen provider (Anthropic, OpenAI, or Google)

### Setup

```bash
# Clone if not already present
git clone https://github.com/alexrudloff/llmrouter.git
cd llmrouter

# Copy and customize config
cp config.yaml.example config.yaml
```

### Start the Server

```bash
python server.py --port 4001
```

Options:
- `--port PORT` - Port to listen on (default: 4001)
- `--host HOST` - Host to bind (default: 127.0.0.1)
- `--config PATH` - Config file path (default: config.yaml)
- `--log` - Enable verbose logging
- `--openclaw` - Enable OpenClaw compatibility mode

## Configuration

Edit `config.yaml` to customize model routing:

```yaml
models:
  super_easy: "anthropic:claude-haiku-4-5-20251001"
  easy: "anthropic:claude-haiku-4-5-20251001"
  medium: "anthropic:claude-sonnet-4-20250514"
  hard: "anthropic:claude-opus-4-20250514"
  super_hard: "anthropic:claude-opus-4-20250514"
```

### Supported Providers

- `anthropic:claude-*` - Anthropic Claude models
- `openai:gpt-*` - OpenAI GPT models
- `google:gemini-*` - Google Gemini models
- `local:model-name` - Local Ollama models

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
