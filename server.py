#!/usr/bin/env python3
"""
Multi-Provider Intelligent Routing Proxy

Classifies incoming messages by complexity and routes to appropriate models.
Supports local models (Ollama), Anthropic, OpenAI, and Google providers.

Configuration:
  - config.yaml: Model mappings and settings
  - CLI arguments: --port, --config, --host

API Keys: Passed through from caller's Authorization header (Bearer token)

API Format: OpenAI-compatible (drop-in replacement)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import yaml

# Unbuffered output for launchd logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Global logging flag
VERBOSE_LOG = False

def log(msg):
    """Print only if verbose logging is enabled"""
    if VERBOSE_LOG:
        print(msg)

from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import requests

from classifier import classify

# Global configuration (loaded from config.yaml)
CONFIG = {}
MODEL_MAP = {}
PROVIDER_URLS = {}
OPENCLAW_MODE = False  # Set via --openclaw flag


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    global CONFIG, MODEL_MAP, PROVIDER_URLS

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_file) as f:
            CONFIG = yaml.safe_load(f)

        # Extract model mappings
        MODEL_MAP = CONFIG.get("models", {})
        if not MODEL_MAP:
            print("Error: No model mappings found in config", file=sys.stderr)
            sys.exit(1)

        # Extract provider URLs
        PROVIDER_URLS = {
            "anthropic": CONFIG.get("providers", {}).get("anthropic", {}).get("url", "https://api.anthropic.com/v1/messages"),
            "openai": CONFIG.get("providers", {}).get("openai", {}).get("url", "https://api.openai.com/v1/chat/completions"),
            "google": CONFIG.get("providers", {}).get("google", {}).get("url", "https://generativelanguage.googleapis.com/v1beta/models"),
            "ollama": CONFIG.get("providers", {}).get("ollama", {}).get("url", "http://localhost:11434/api/chat"),
        }

        return CONFIG
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)


def parse_provider_model(provider_model_str):
    """Parse 'provider:model' format. Returns (provider, model)."""
    if ":" in provider_model_str:
        parts = provider_model_str.split(":", 1)
        return parts[0], parts[1]
    # Default to anthropic if no provider specified
    return "anthropic", provider_model_str


def extract_text_content(content):
    """Extract text from OpenAI-style content (string or array of content blocks).

    Args:
        content: Either a string or a list of content blocks with type/text fields

    Returns:
        Extracted text as a string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def call_local_model(model, messages, max_tokens, system=None, api_key=None, tools=None):
    """Call local Ollama model (api_key and tools ignored)"""
    try:
        # Convert messages to simple text format for Ollama
        ollama_messages = []

        # Add system message if provided
        if system:
            ollama_messages.append({
                "role": "system",
                "content": system
            })

        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": extract_text_content(msg.get("content"))
            })

        response = requests.post(
            PROVIDER_URLS["ollama"],
            json={
                "model": model,
                "messages": ollama_messages,
                "stream": False,
                "options": {"num_predict": max_tokens}
            },
            timeout=300
        )
        response.raise_for_status()
        data = response.json()

        return {
            "id": f"local-{int(time.time())}",
            "content": [{"type": "text", "text": data["message"]["content"]}],
            "usage": {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0)
            },
            "stop_reason": "stop"
        }
    except Exception as e:
        raise Exception(f"Local model error: {e}")


# Claude Code official tool names - must use exact casing for OAuth tokens
CLAUDE_CODE_TOOLS = {
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "bash": "Bash",
    "grep": "Grep",
    "glob": "Glob",
    "askuserquestion": "AskUserQuestion",
    "enterplanmode": "EnterPlanMode",
    "exitplanmode": "ExitPlanMode",
    "notebookedit": "NotebookEdit",
    "skill": "Skill",
    "task": "Task",
    "taskoutput": "TaskOutput",
    "todowrite": "TodoWrite",
    "webfetch": "WebFetch",
    "websearch": "WebSearch",
}

def to_claude_code_name(name):
    """Convert tool name to Claude Code's official casing if it matches"""
    return CLAUDE_CODE_TOOLS.get(name.lower(), name)

# Reverse mapping for converting back
CLAUDE_CODE_TOOLS_REVERSE = {v: k for k, v in CLAUDE_CODE_TOOLS.items()}

def from_claude_code_name(name):
    """Convert Claude Code tool name back to original lowercase"""
    return CLAUDE_CODE_TOOLS_REVERSE.get(name, name)


def convert_openai_tools_to_anthropic(openai_tools, use_oauth=False, openclaw_mode=False):
    """Convert OpenAI tool format to Anthropic tool format"""
    if not openai_tools:
        return None

    # Only remap tool names for OAuth + openclaw mode
    should_remap = use_oauth and openclaw_mode

    anthropic_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            name = func.get("name", "")
            if should_remap:
                name = to_claude_code_name(name)
            anthropic_tools.append({
                "name": name,
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        else:
            # Already in Anthropic format or custom format
            name = tool.get("name", "")
            if should_remap:
                name = to_claude_code_name(name)
            anthropic_tools.append({
                **tool,
                "name": name
            })

    return anthropic_tools if anthropic_tools else None


def is_oauth_token(api_key):
    """Detect if the API key is an OAuth token based on prefix"""
    if not api_key:
        return False
    return "sk-ant-oat" in api_key


def call_anthropic_model(model, messages, max_tokens, system=None, api_key=None, tools=None):
    """Call Anthropic API - supports both OAuth tokens and API keys

    LITERALLY copies openclaw's approach from:
    /opt/homebrew/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/anthropic.js
    """
    if not api_key:
        raise Exception("No API key provided in Authorization header")

    # Detect auth type (line 323-325 in anthropic.js)
    use_oauth = is_oauth_token(api_key)

    # Build headers for raw HTTP request
    if use_oauth:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
            "user-agent": "claude-cli/2.1.2 (external, cli)",
            "x-app": "cli",
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "fine-grained-tool-streaming-2025-05-14",
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }

    # Build params (lines 363-428 in anthropic.js)
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    # For OAuth tokens, we MUST include Claude Code identity (lines 372-393)
    if use_oauth:
        params["system"] = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if system:
            params["system"].append({
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            })
    elif system:
        params["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if tools:
        anthropic_tools = convert_openai_tools_to_anthropic(tools, use_oauth=use_oauth, openclaw_mode=OPENCLAW_MODE)
        if anthropic_tools:
            params["tools"] = anthropic_tools

    # Make raw HTTP request
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=params,
            timeout=120
        )
        if response.status_code != 200:
            log(f"  ERROR {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log(f"  ERROR: {e}")
        raise


def call_openai_model(model, messages, max_tokens, system=None, api_key=None, tools=None):
    """Call OpenAI API (tools not yet implemented)"""
    if not api_key:
        raise Exception("No API key provided in Authorization header")

    # Prepend system message if provided
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    openai_messages.extend(messages)

    response = requests.post(
        PROVIDER_URLS["openai"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise Exception(f"OpenAI error: {response.text}")

    data = response.json()

    # Convert OpenAI format to Anthropic-like format
    return {
        "id": data["id"],
        "content": [{"type": "text", "text": data["choices"][0]["message"]["content"]}],
        "usage": {
            "input_tokens": data["usage"]["prompt_tokens"],
            "output_tokens": data["usage"]["completion_tokens"]
        },
        "stop_reason": data["choices"][0].get("finish_reason", "stop")
    }


def call_google_model(model, messages, max_tokens, system=None, api_key=None, tools=None):
    """Call Google Gemini API (tools not yet implemented)"""
    if not api_key:
        raise Exception("No API key provided in Authorization header")

    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": extract_text_content(msg["content"])}]
        })

    response = requests.post(
        f"{PROVIDER_URLS['google']}/{model}:generateContent?key={api_key}",
        json={
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens}
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise Exception(f"Google error: {response.text}")

    data = response.json()

    # Convert Gemini format to Anthropic-like format
    return {
        "id": f"gemini-{int(time.time())}",
        "content": [{"type": "text", "text": data["candidates"][0]["content"]["parts"][0]["text"]}],
        "usage": {
            "input_tokens": data.get("usageMetadata", {}).get("promptTokenCount", 0),
            "output_tokens": data.get("usageMetadata", {}).get("candidatesTokenCount", 0)
        },
        "stop_reason": "stop"
    }


# Provider registry - maps provider name to handler function
# All handlers have signature: (model, messages, max_tokens, system, api_key, tools)
PROVIDERS = {
    "local": call_local_model,
    "anthropic": call_anthropic_model,
    "openai": call_openai_model,
    "google": call_google_model,
}


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # --- Streaming helpers ---

    def _create_chunk(self, response_id, model_name, delta, finish_reason=None, usage=None):
        """Create an OpenAI-compatible SSE chunk."""
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    def _send_chunk(self, chunk):
        """Send an SSE chunk."""
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

    def _start_stream(self, response_id, model_name):
        """Start SSE stream: send headers and role chunk."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._send_chunk(self._create_chunk(response_id, model_name, {"role": "assistant"}))

    def _end_stream(self, response_id, model_name, finish_reason, usage):
        """End SSE stream: send finish chunk and done marker."""
        self._send_chunk(self._create_chunk(response_id, model_name, {}, finish_reason, usage))
        self.wfile.write(b"data: [DONE]\n\n")

    # --- Public streaming methods ---

    def send_streaming_response(self, response_id, model_name, content, usage):
        """Send OpenAI-compatible streaming response."""
        self._start_stream(response_id, model_name)
        self._send_chunk(self._create_chunk(response_id, model_name, {"content": content}))
        self._end_stream(response_id, model_name, "stop", usage)

    def send_streaming_tool_response(self, response_id, model_name, content_blocks, usage, stop_reason):
        """Send OpenAI-compatible streaming response with tool calls."""
        self._start_stream(response_id, model_name)

        tool_call_index = 0
        for block in content_blocks:
            if block.get("type") == "text":
                self._send_chunk(self._create_chunk(
                    response_id, model_name, {"content": block.get("text", "")}
                ))
            elif block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                if OPENCLAW_MODE:
                    tool_name = from_claude_code_name(tool_name)

                delta = {
                    "tool_calls": [{
                        "index": tool_call_index,
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    }]
                }
                self._send_chunk(self._create_chunk(response_id, model_name, delta))
                tool_call_index += 1
                log(f"  Tool call: {tool_name}")

        finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"
        self._end_stream(response_id, model_name, finish_reason, usage)

    def send_error_json(self, message, status=500):
        self.send_json({"error": {"message": message}}, status)
    
    def do_POST(self):
        path = urlparse(self.path).path
        
        if path == "/v1/chat/completions":
            self.handle_chat_completions()
        else:
            self.send_error_json(f"Unknown endpoint: {path}", 404)
    
    def handle_chat_completions(self):
        try:
            # Extract API key from Authorization header
            auth_header = self.headers.get("Authorization", "")
            api_key = None
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Strip "Bearer " prefix

            # Debug: log API key prefix and auth type
            if api_key:
                auth_type = "OAuth" if is_oauth_token(api_key) else "API Key"
                log(f"  Auth: {auth_type} ({api_key[:15]}...{api_key[-4:]})")
            else:
                log(f"  WARNING: No API key in Authorization header")

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request = json.loads(body)
            
            # Extract the last user message for classification
            messages = request.get("messages", [])
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = extract_text_content(msg.get("content", ""))
                    break
            
            if not user_message:
                self.send_error_json("No user message found", 400)
                return
            
            # Extract tools from request
            tools = request.get("tools")

            # Classify the message
            start = time.time()
            classifier_config = CONFIG.get("classifier", {})
            complexity = classify(
                user_message,
                model=classifier_config.get("model"),
                provider=classifier_config.get("provider"),
                ollama_url=classifier_config.get("ollama_url"),
                api_key=api_key,  # Pass through for remote classification
            )
            classify_time = (time.time() - start) * 1000

            # If tools are present, bump super_easy to easy (tool use requires more capable model)
            if tools and complexity == "super_easy":
                complexity = "easy"
                log(f"  Bumped super_easy -> easy (tools present)")

            log(f"  Classifying ({len(user_message)} chars): '{user_message[:100]}...'")
            log(f"  -> {complexity} in {classify_time:.0f}ms")

            # Map complexity to provider:model
            provider_model = MODEL_MAP.get(complexity, MODEL_MAP["medium"])
            provider, target_model = parse_provider_model(provider_model)

            log(f"  '{user_message[:50]}...' -> {complexity} -> {provider}:{target_model} ({classify_time:.0f}ms)")

            # Build provider-agnostic message format
            provider_messages = []
            system_content = None

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")

                # Handle system messages
                if role == "system":
                    system_content = extract_text_content(content)
                    continue

                # Convert OpenAI tool messages to Anthropic format
                # OpenAI: {"role": "tool", "tool_call_id": "xxx", "content": "result"}
                # Anthropic: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "xxx", "content": "result"}]}
                if role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    tool_content = content if isinstance(content, str) else json.dumps(content)
                    provider_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_content
                        }]
                    })
                    continue

                # Handle assistant messages with tool_calls (OpenAI format)
                # Convert to Anthropic's tool_use content blocks
                tool_calls = msg.get("tool_calls", [])

                anthropic_content = []

                # Add text content if present and not null
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                anthropic_content.append(item)
                            elif item.get("type") != "text":
                                anthropic_content.append(item)
                elif content:
                    anthropic_content.append({"type": "text", "text": content})

                # Convert OpenAI tool_calls to Anthropic tool_use blocks
                for tc in tool_calls:
                    tool_use = {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
                    }
                    anthropic_content.append(tool_use)

                if anthropic_content:
                    provider_messages.append({"role": role, "content": anthropic_content})

            # In openclaw mode, rewrite model= in system prompt Runtime line
            if OPENCLAW_MODE and system_content:
                actual_model = f"{provider}/{target_model}"
                system_content = re.sub(
                    r'\bmodel=localrouter/[^\s|]+',
                    f'model={actual_model}',
                    system_content
                )

            # Route to appropriate provider
            try:
                max_tokens = request.get("max_tokens", 8192)
                provider_fn = PROVIDERS.get(provider)
                if not provider_fn:
                    self.send_error_json(f"Unknown provider: {provider}", 400)
                    return
                provider_response = provider_fn(
                    target_model, provider_messages, max_tokens, system_content, api_key, tools
                )
            except Exception as e:
                self.send_error_json(str(e), 500)
                return

            # Extract response content
            anthropic_response = provider_response
            response_id = anthropic_response.get("id", f"chatcmpl-{int(time.time())}")
            model_name = f"local-router:{complexity}:{provider}:{target_model}"

            usage = {
                "prompt_tokens": anthropic_response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": anthropic_response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    anthropic_response.get("usage", {}).get("input_tokens", 0) +
                    anthropic_response.get("usage", {}).get("output_tokens", 0)
                ),
            }

            # Check for tool use in response
            content_blocks = anthropic_response.get("content", [])
            has_tool_use = any(block.get("type") == "tool_use" for block in content_blocks)

            if has_tool_use:
                # Send streaming response with tool calls
                self.send_streaming_tool_response(response_id, model_name, content_blocks, usage, anthropic_response.get("stop_reason", "tool_use"))
            else:
                # Extract text content
                response_content = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        response_content += block.get("text", "")

                # Log response
                log(f"  Response: '{response_content[:100]}...'")

                # Send streaming response (openclaw expects SSE format)
                self.send_streaming_response(response_id, model_name, response_content, usage)

        except json.JSONDecodeError as e:
            self.send_error_json(f"Invalid JSON: {e}", 400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error_json(str(e), 500)
    
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            classifier_config = CONFIG.get("classifier", {})
            self.send_json({
                "status": "ok",
                "mode": "proxy",
                "classifier_provider": classifier_config.get("provider", "local"),
                "classifier_model": classifier_config.get("model", "qwen2.5:3b"),
                "models": list(MODEL_MAP.keys()),
            })
        elif path == "/v1/models":
            self.send_json({
                "data": [
                    {"id": "local-router", "object": "model"},
                ]
            })
        else:
            self.send_error_json(f"Unknown endpoint: {path}", 404)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Multi-Provider Intelligent Routing Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --port 4001
  %(prog)s --config my-config.yaml --host 0.0.0.0 --port 8080

API Keys:
  Passed through from caller's Authorization header (Bearer token).
  Callers must provide their own API key for the target provider.
        """
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port to listen on (default: from config or 4001)"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--openclaw",
        action="store_true",
        help="Enable openclaw compatibility (rewrites model= in system prompt)"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable verbose request/response logging"
    )

    args = parser.parse_args()

    # Set global flags
    global OPENCLAW_MODE, VERBOSE_LOG
    OPENCLAW_MODE = args.openclaw
    VERBOSE_LOG = args.log

    # Load configuration
    print("Loading configuration...")
    load_config(args.config)

    # Get server settings
    host = args.host or CONFIG.get("server", {}).get("host", "127.0.0.1")
    port = args.port or CONFIG.get("server", {}).get("port", 4001)

    # Start server
    server = HTTPServer((host, port), RouterHandler)
    print(f"\n🚀 Multi-Provider Router (Proxy Mode)")
    print(f"   http://{host}:{port}")
    classifier_config = CONFIG.get('classifier', {})
    classifier_provider = classifier_config.get('provider', 'local')
    classifier_model = classifier_config.get('model', 'qwen2.5:3b')
    print(f"\n📊 Classifier: {classifier_provider}:{classifier_model}")
    print(f"\n🎯 Model Routing (5-tier):")
    for tier in ["super_easy", "easy", "medium", "hard", "super_hard"]:
        provider_model = MODEL_MAP.get(tier, "not configured")
        print(f"   {tier:12} -> {provider_model}")
    print(f"\n🔑 Auth: supports both OAuth tokens (sk-ant-oat*) and API keys (sk-ant-api*)")
    print(f"\n✓ Ready for requests")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        server.shutdown()


if __name__ == "__main__":
    main()
