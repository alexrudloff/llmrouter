# LLM Router

An intelligent proxy that classifies incoming requests by complexity and routes them to appropriate LLM models. Save money by using cheaper/faster models for simple tasks and reserving expensive models for complex ones.

## Features

- **5-tier complexity routing**: super_easy, easy, medium, hard, super_hard
- **Local classification**: Uses Ollama + qwen2.5:3b to classify requests locally (no API costs)
- **Multi-provider support**: Anthropic, OpenAI, Google Gemini, Ollama
- **OAuth token support**: Works with Claude Code OAuth tokens (sk-ant-oat*)
- **OpenAI-compatible API**: Drop-in replacement for existing integrations
- **OpenClaw compatible**: Optional mode for OpenClaw integration

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with `qwen2.5:3b` model
- API key for your chosen provider(s)

## Installation

```bash
# Clone the repo
git clone https://github.com/alexrudloff/llmrouter.git
cd llmrouter

# Install dependencies
pip install pyyaml requests

# Pull the classifier model
ollama pull qwen2.5:3b

# Copy and customize config
cp config.yaml.example config.yaml
# Edit config.yaml with your preferred model mappings
```

## Configuration

Edit `config.yaml` to customize model routing:

```yaml
models:
  super_easy: "anthropic:claude-haiku-4-5-20251001"  # Fast, cheap
  easy: "anthropic:claude-haiku-4-5-20251001"        # Fast, cheap
  medium: "anthropic:claude-sonnet-4-20250514"       # Balanced
  hard: "anthropic:claude-opus-4-20250514"           # Powerful
  super_hard: "anthropic:claude-opus-4-20250514"     # Most capable
```

Supported provider formats:
- `anthropic:claude-*` - Anthropic Claude models
- `openai:gpt-*` - OpenAI GPT models
- `google:gemini-*` - Google Gemini models
- `local:model-name` - Local Ollama models

## Usage

### Start the server

```bash
python server.py
```

Options:
- `--port PORT` - Port to listen on (default: 4001)
- `--host HOST` - Host to bind to (default: 127.0.0.1)
- `--config PATH` - Path to config file (default: config.yaml)
- `--log` - Enable verbose request/response logging
- `--openclaw` - Enable OpenClaw compatibility mode

### Make requests

The router exposes an OpenAI-compatible API at `/v1/chat/completions`:

```bash
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm-router",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The router will:
1. Classify the message complexity using the local qwen model
2. Route to the appropriate provider/model based on `config.yaml`
3. Return the response in OpenAI-compatible format

### API Key Handling

Your API key is passed through to the target provider. The router supports:
- Standard Anthropic API keys (`sk-ant-api*`)
- Claude Code OAuth tokens (`sk-ant-oat*`) - requires Claude Code identity headers

## Running as a Service (macOS)

Create a LaunchAgent plist at `~/Library/LaunchAgents/com.llmrouter.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llmrouter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/llmrouter/server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/path/to/llmrouter</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/com.llmrouter.plist
```

## Classification Rules

Edit `ROUTES.md` to customize how messages are classified. The classifier reads the table in this file to determine complexity levels.

## License

MIT
