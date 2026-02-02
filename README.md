# LLM Router

An intelligent proxy that classifies incoming requests by complexity and routes them to appropriate LLM models. Save money by using cheaper/faster models for simple tasks and reserving expensive models for complex ones.

**Works with [OpenClaw](https://github.com/openclaw/openclaw)** to reduce token usage and API costs by routing simple requests to smaller models.

## Status

**Tested with Anthropic and OpenAI.** Google and local Ollama providers are implemented but untested.

## Features

- **5-tier complexity routing**: super_easy, easy, medium, hard, super_hard
- **Local classification**: Uses Ollama to classify requests locally (no API costs for classification)
- **Multi-provider support**: Anthropic (tested), OpenAI (tested), Google Gemini, Ollama (untested)
- **OpenAI o-series support**: Automatically handles o1, o3, o4-mini reasoning models with correct API parameters
- **OAuth token support**: Works with Claude Code OAuth tokens (sk-ant-oat*)
- **OpenAI-compatible API**: Drop-in replacement for existing integrations
- **Configurable classifier model**: Change the local model used for classification in config.yaml

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (optional if using Anthropic for classification)
- Anthropic API key (or Claude Code OAuth token)

## Installation

```bash
# Clone the repo
git clone https://github.com/alexrudloff/llmrouter.git
cd llmrouter

# Install dependencies
pip install -r requirements.txt

# Pull the classifier model (default: qwen2.5:3b, configurable in config.yaml)
ollama pull qwen2.5:3b

# Copy and customize config
cp config.yaml.example config.yaml
```

## Configuration

Edit `config.yaml` to customize:

### Model Routing

Configure which model handles each complexity level:

```yaml
# Anthropic routing
models:
  super_easy: "anthropic:claude-haiku-4-5-20251001"  # Fast, cheap
  easy: "anthropic:claude-sonnet-4-20250514"         # Balanced
  medium: "anthropic:claude-sonnet-4-20250514"       # Balanced
  hard: "anthropic:claude-opus-4-20250514"           # Powerful
  super_hard: "anthropic:claude-opus-4-20250514"     # Most capable
```

```yaml
# OpenAI routing (including o-series reasoning models)
models:
  super_easy: "openai:gpt-4o-mini"    # Fast, cheap
  easy: "openai:gpt-4o-mini"          # Fast, cheap
  medium: "openai:gpt-4o"             # Balanced
  hard: "openai:o3-mini"              # Reasoning model
  super_hard: "openai:o3"             # Most capable reasoning
```

**Note:** OpenAI o-series models (o1, o3, o4-mini) are automatically detected and use the correct API parameters (`max_completion_tokens` instead of `max_tokens`, `developer` role instead of `system`).

### Classifier

The classifier determines request complexity before routing. Three options:

#### Local Classification (default)

Uses Ollama running on your machine. Free, but requires local hardware.

```yaml
classifier:
  provider: "local"
  model: "qwen2.5:3b"  # Any Ollama model
  ollama_url: "http://localhost:11434/api/generate"
```

```bash
# Setup
ollama pull qwen2.5:3b
```

#### Anthropic Classification

Uses Anthropic Haiku for classification. Fast, low cost per classification.

```yaml
classifier:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
```

#### OpenAI Classification

Uses OpenAI for classification. Useful if you have OpenAI credits or prefer their models.

```yaml
classifier:
  provider: "openai"
  model: "gpt-4o-mini"
```

Choose remote (anthropic/openai) if:
- Your machine can't run local models
- You want simpler setup (no Ollama required)

### Provider API Keys

Configure API keys per provider in `config.yaml`. Keys in config take priority over the request Authorization header.

```yaml
providers:
  anthropic:
    url: "https://api.anthropic.com/v1/messages"
    api_key: "sk-ant-..."  # Your Anthropic key or OAuth token
  openai:
    url: "https://api.openai.com/v1/chat/completions"
    api_key: "sk-proj-..."
  deepseek:
    url: "https://api.deepseek.com/v1/chat/completions"
    api_key: "sk-..."
  kimi:
    url: "https://api.moonshot.cn/v1/chat/completions"
    api_key: "sk-..."
```

This allows routing to multiple providers without passing different keys per request.

### Provider Formats

- `anthropic:claude-*` - Anthropic Claude models (tested)
- `openai:gpt-*` - OpenAI GPT models (tested)
- `openai:o1-*`, `openai:o3-*`, `openai:o4-*` - OpenAI reasoning models (tested)
- `google:gemini-*` - Google Gemini models (untested)
- `deepseek:deepseek-*` - DeepSeek models (untested)
- `kimi:moonshot-*` - Kimi/Moonshot models (untested)
- `local:model-name` - Local Ollama models (untested)

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
- `--openclaw` - Enable OpenClaw compatibility (rewrites model name in system prompt)

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

## OpenClaw Integration

To use llmrouter with [OpenClaw](https://github.com/openclaw/openclaw), add a provider to your `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "localrouter": {
        "baseUrl": "http://localhost:4001/v1",
        "apiKey": "via-router",
        "api": "openai-completions",
        "models": [
          {
            "id": "llm-router",
            "name": "LLM Router (Auto-routes by complexity)",
            "reasoning": false,
            "input": ["text", "image"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 200000,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

Then set it as your default model in `agents.defaults.model.primary`:

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "localrouter/llm-router"
      }
    }
  }
}
```

Start the server with OpenClaw compatibility mode:

```bash
python server.py --openclaw
```

The `--openclaw` flag enables model name rewriting in system prompts so OpenClaw displays the actual model being used (rewrites `model=localrouter/...` to the actual provider/model).

Note: Tool name remapping for Claude Code OAuth tokens happens automatically when an OAuth token is detected.

## Classification Rules

Edit `ROUTES.md` to customize how messages are classified. The classifier reads the table in this file to determine complexity levels.

## License

MIT
