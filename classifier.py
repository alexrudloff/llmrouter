#!/usr/bin/env python3
"""
Message classifier for LLM Router.
Reads ROUTES.md and classifies incoming messages into 5 complexity tiers.
Supports local (Ollama) or remote (Anthropic, OpenAI) classification.
"""

import re
import requests
import sys
from pathlib import Path

# Defaults (can be overridden via function parameters)
DEFAULT_PROVIDER = "local"  # "local", "anthropic", or "openai"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
ROUTES_PATH = Path(__file__).parent / "ROUTES.md"

# Valid complexity levels (single source of truth)
COMPLEXITY_LEVELS = ("super_easy", "easy", "medium", "hard", "super_hard")

# Fallback prompt if ROUTES.md can't be read
DEFAULT_PROMPT = """Classify by complexity. Reply with ONE word: super_easy, easy, medium, hard, super_hard

super_easy: greetings, acknowledgments, yes/no, farewells, single words
easy: casual Q&A, reminders, status checks, simple facts, format conversion
medium: email, any code, research, bug fixes, image description
hard: complex code, planning, hard debugging, image analysis, chained tools
super_hard: algorithms, proofs, system design, agentic tasks, long synthesis

Message: {MESSAGE}

Complexity:
"""

def _build_prompt(message: str) -> str:
    """Build the classification prompt from ROUTES.md template."""
    truncated = message[:500] + "..." if len(message) > 500 else message
    try:
        template = ROUTES_PATH.read_text()
    except Exception as e:
        print(f"Warning: Could not load ROUTES.md: {e}", file=sys.stderr)
        template = DEFAULT_PROMPT
    return template.replace("{MESSAGE}", truncated)


def _extract_complexity(result_text: str) -> str:
    """Extract complexity level from model response."""
    result_text = result_text.strip().lower()

    # Remove thinking tags if present
    result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
    result_text = re.sub(r'</?think>', '', result_text).strip()

    # Exact match first (for clean single-word responses)
    if result_text in COMPLEXITY_LEVELS:
        return result_text

    # Word boundary match (check super_ variants first to avoid partial matches)
    for level in ("super_hard", "super_easy", "hard", "medium", "easy"):
        if re.search(rf'\b{level}\b', result_text):
            return level

    return ""


def _classify_with_ollama(prompt: str, model: str, ollama_url: str) -> str:
    """Classify using local Ollama model."""
    response = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 50
            }
        },
        timeout=30
    )
    response.raise_for_status()
    resp_json = response.json()

    # Try response field first, then thinking field
    result_text = resp_json.get("response", "")
    if not result_text or result_text.strip().lower().startswith("<think"):
        thinking_text = resp_json.get("thinking", "")
        result_text = thinking_text if thinking_text else result_text

    return _extract_complexity(result_text)


def _is_oauth_token(api_key: str) -> bool:
    """Detect if the API key is an OAuth token based on prefix."""
    return api_key and "sk-ant-oat" in api_key


def _classify_with_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Classify using Anthropic API (e.g., Haiku). Supports both API keys and OAuth tokens."""
    use_oauth = _is_oauth_token(api_key)

    if use_oauth:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "user-agent": "claude-cli/2.1.2 (external, cli)",
            "x-app": "cli",
        }
        # OAuth requires Claude Code identity
        system = [{
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            "cache_control": {"type": "ephemeral"},
        }]
    else:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        system = None

    payload = {
        "model": model,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    resp_json = response.json()

    # Extract text from response
    content = resp_json.get("content", [])
    if content and content[0].get("type") == "text":
        return _extract_complexity(content[0].get("text", ""))
    return ""


def _is_openai_reasoning_model(model: str) -> bool:
    """Check if model is an OpenAI o-series reasoning model."""
    model_lower = model.lower()
    return (
        model_lower.startswith("o1") or
        model_lower.startswith("o3") or
        model_lower.startswith("o4")
    )


def _classify_with_openai(prompt: str, model: str, api_key: str) -> str:
    """Classify using OpenAI API."""
    # Build payload - o-series models use max_completion_tokens
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _is_openai_reasoning_model(model):
        payload["max_completion_tokens"] = 50
    else:
        payload["max_tokens"] = 50

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    resp_json = response.json()

    # Extract text from response
    choices = resp_json.get("choices", [])
    if choices and choices[0].get("message", {}).get("content"):
        return _extract_complexity(choices[0]["message"]["content"])
    return ""


DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"
DEFAULT_KIMI_MODEL = "moonshot-v1-8k"


def _classify_with_google(prompt: str, model: str, api_key: str) -> str:
    """Classify using Google Gemini API."""
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 50}
        },
        timeout=30
    )
    response.raise_for_status()
    resp_json = response.json()

    # Extract text from Gemini response
    candidates = resp_json.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts and parts[0].get("text"):
            return _extract_complexity(parts[0]["text"])
    return ""


def _classify_with_kimi(prompt: str, model: str, api_key: str) -> str:
    """Classify using Kimi/Moonshot API."""
    response = requests.post(
        "https://api.moonshot.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50
        },
        timeout=30
    )
    response.raise_for_status()
    resp_json = response.json()

    # Extract text from response
    choices = resp_json.get("choices", [])
    if choices and choices[0].get("message", {}).get("content"):
        return _extract_complexity(choices[0]["message"]["content"])
    return ""


def classify(message: str, model: str = None, provider: str = None,
             ollama_url: str = None, api_key: str = None) -> str:
    """
    Classify a message into super_easy/easy/medium/hard/super_hard.
    Returns the complexity level (lowercase with underscore).

    Args:
        message: The message to classify
        model: Model to use (defaults vary by provider)
        provider: "local" (Ollama), "anthropic", or "openai" (default: local)
        ollama_url: Ollama API URL (default: http://localhost:11434/api/generate)
        api_key: API key for remote providers
    """
    if provider is None:
        provider = DEFAULT_PROVIDER

    prompt = _build_prompt(message)

    try:
        if provider in ("api", "anthropic"):
            # "api" is legacy alias for "anthropic"
            if not api_key:
                raise ValueError("API key/token required for Anthropic classification")
            if model is None:
                model = DEFAULT_ANTHROPIC_MODEL
            result = _classify_with_anthropic(prompt, model, api_key)
        elif provider == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI classification")
            if model is None:
                model = DEFAULT_OPENAI_MODEL
            result = _classify_with_openai(prompt, model, api_key)
        elif provider == "google":
            if not api_key:
                raise ValueError("API key required for Google classification")
            if model is None:
                model = DEFAULT_GOOGLE_MODEL
            result = _classify_with_google(prompt, model, api_key)
        elif provider == "kimi":
            if not api_key:
                raise ValueError("API key required for Kimi classification")
            if model is None:
                model = DEFAULT_KIMI_MODEL
            result = _classify_with_kimi(prompt, model, api_key)
        else:
            # Default to local/Ollama
            if model is None:
                model = DEFAULT_MODEL
            if ollama_url is None:
                ollama_url = DEFAULT_OLLAMA_URL
            result = _classify_with_ollama(prompt, model, ollama_url)

        if result:
            return result

        print(f"Warning: Could not extract classification, defaulting to medium", file=sys.stderr)
        return "medium"

    except Exception as e:
        print(f"Error classifying message: {e}", file=sys.stderr)
        return "medium"  # Safe default

def main():
    """CLI interface for testing"""
    if len(sys.argv) < 2:
        print("Usage: classifier.py <message>")
        print("       classifier.py --test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        # Run test suite
        tests = [
            ("Hey", "super_easy"),
            ("What is 2+2?", "easy"),
            ("Write a Python sort function", "medium"),
            ("Refactor to microservices architecture", "hard"),
            ("Remind me at 9am", "easy"),
            ("Debug why this crashes with segfault", "hard"),
            ("Send email to John about meeting", "medium"),
            ("Design a distributed system architecture", "super_hard"),
        ]
        
        passed = 0
        for msg, expected in tests:
            result = classify(msg)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            print(f"{status} '{msg[:40]}...' -> {result} (expected {expected})")
        
        print(f"\n{passed}/{len(tests)} passed")
        sys.exit(0 if passed == len(tests) else 1)
    
    # Classify the provided message
    message = " ".join(sys.argv[1:])
    result = classify(message)
    print(result)

if __name__ == "__main__":
    main()
