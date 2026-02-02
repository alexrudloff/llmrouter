#!/usr/bin/env python3
"""
Message classifier for LLM Router.
Reads ROUTES.md and classifies incoming messages into 5 complexity tiers.
Supports local (Ollama) or remote (Anthropic) classification.
"""

import re
import requests
import sys
from pathlib import Path

# Defaults (can be overridden via function parameters)
DEFAULT_PROVIDER = "local"  # "local" (Ollama) or "anthropic"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ROUTES_PATH = Path(__file__).parent / "ROUTES.md"

# Valid complexity levels (single source of truth)
COMPLEXITY_LEVELS = ("super_easy", "easy", "medium", "hard", "super_hard")

# Default rules if ROUTES.md can't be parsed
DEFAULT_RULES = """
- super_easy: greetings, acknowledgments, very basic yes/no questions
- easy: casual chat, simple questions, reminders, quick facts
- medium: coding, writing, email, research, moderate tasks
- hard: complex reasoning, architecture, debugging, multi-step analysis
- super_hard: advanced algorithms, multi-step proofs, system design
"""

def load_classifier_rules():
    """Load classification rules from ROUTES.md"""
    try:
        content = ROUTES_PATH.read_text()

        # Extract the classification table section
        lines = content.split('\n')
        in_table = False
        rules = []

        for line in lines:
            # Start of classification table
            if '| Task Type | Complexity |' in line:
                in_table = True
                continue

            # Skip header separator
            if in_table and '|---' in line:
                continue

            # End of table (empty line or new section)
            if in_table and (not line.strip() or line.startswith('#')):
                break

            # Parse table rows
            if in_table and line.strip().startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 3:
                    task_type = parts[0]
                    complexity = parts[1].lower().replace(' ', '_')
                    notes = parts[2]

                    # Include all 5 complexity levels
                    if complexity in COMPLEXITY_LEVELS:
                        rules.append(f"- {complexity}: {task_type} — {notes}")

        if rules:
            return '\n'.join(rules)
        else:
            print("Warning: No rules found in ROUTES.md, using defaults", file=sys.stderr)
            return DEFAULT_RULES

    except Exception as e:
        print(f"Warning: Could not load ROUTES.md: {e}", file=sys.stderr)
        return DEFAULT_RULES

def _build_prompt(message: str, rules: str) -> str:
    """Build the classification prompt."""
    truncated = message[:500] + "..." if len(message) > 500 else message
    return f"""Classify this message by complexity: super_easy, easy, medium, hard, or super_hard.

Rules:
{rules}

Message: "{truncated}"

Complexity (answer with just one of: super_easy, easy, medium, hard, super_hard):"""


def _extract_complexity(result_text: str) -> str:
    """Extract complexity level from model response."""
    result_text = result_text.strip().lower()

    # Remove thinking tags if present
    result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
    result_text = re.sub(r'</?think>', '', result_text).strip()

    # Look for complexity levels in the text
    for level in COMPLEXITY_LEVELS:
        if level in result_text:
            return level

    # Try extracting first word as fallback
    if result_text:
        result = result_text.split()[0] if result_text.split() else ""
        if result in COMPLEXITY_LEVELS:
            return result

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


def classify(message: str, rules: str = None, model: str = None,
             provider: str = None, ollama_url: str = None, api_key: str = None) -> str:
    """
    Classify a message into super_easy/easy/medium/hard/super_hard.
    Returns the complexity level (lowercase with underscore).

    Args:
        message: The message to classify
        rules: Classification rules (loaded from ROUTES.md if not provided)
        model: Model to use (default: qwen2.5:3b for local, claude-haiku-4-5-20251001 for anthropic)
        provider: "local" (Ollama) or "anthropic" (default: local)
        ollama_url: Ollama API URL (default: http://localhost:11434/api/generate)
        api_key: Anthropic API key (required if provider is "anthropic", uses ANTHROPIC_API_KEY env var if not provided)
    """
    if rules is None:
        rules = load_classifier_rules()
    if provider is None:
        provider = DEFAULT_PROVIDER

    prompt = _build_prompt(message, rules)

    try:
        if provider == "anthropic":
            if not api_key:
                raise ValueError("API key/token required for remote classification (passed from request)")
            if model is None:
                model = DEFAULT_ANTHROPIC_MODEL
            result = _classify_with_anthropic(prompt, model, api_key)
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
