#!/usr/bin/env python3
"""
Local message classifier using Ollama + qwen2.5:3b
Reads ROUTES.md and classifies incoming messages into 5 complexity tiers.
"""

import json
import requests
import sys
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"  # Fast classification without thinking mode overhead
ROUTES_PATH = Path(__file__).parent / "ROUTES.md"

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
                    if complexity in ('super_easy', 'easy', 'medium', 'hard', 'super_hard'):
                        rules.append(f"- {complexity}: {task_type} — {notes}")

        if rules:
            return '\n'.join(rules)
        else:
            print("Warning: No rules found in ROUTES.md, using defaults", file=sys.stderr)
            return DEFAULT_RULES

    except Exception as e:
        print(f"Warning: Could not load ROUTES.md: {e}", file=sys.stderr)
        return DEFAULT_RULES

def classify(message: str, rules: str = None) -> str:
    """
    Classify a message into super_easy/easy/medium/hard/super_hard.
    Returns the complexity level (lowercase with underscore).
    """
    if rules is None:
        rules = load_classifier_rules()

    # Truncate long messages - we only need the beginning to classify intent
    truncated = message[:500] + "..." if len(message) > 500 else message

    prompt = f"""Classify this message by complexity: super_easy, easy, medium, hard, or super_hard.

Rules:
{rules}

Message: "{truncated}"

Complexity (answer with just one of: super_easy, easy, medium, hard, super_hard):"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 50  # Allow model to output thinking + answer
                }
            },
            timeout=30
        )
        response.raise_for_status()
        resp_json = response.json()

        # Try response field first, then thinking field
        result_text = resp_json.get("response", "").strip().lower()
        if not result_text or result_text.startswith("<think"):
            # Fall back to thinking field for models that use it
            thinking_text = resp_json.get("thinking", "").strip().lower()
            result_text = thinking_text if thinking_text else result_text

        # Remove thinking tags if present
        import re
        result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
        result_text = re.sub(r'</?think>', '', result_text).strip()

        # Valid complexity levels
        valid_levels = ["super_easy", "easy", "medium", "hard", "super_hard"]

        # Look for complexity levels in the text
        for level in valid_levels:
            if level in result_text:
                return level

        # Try extracting first word as fallback
        if result_text:
            result = result_text.split()[0] if result_text.split() else ""
            if result in valid_levels:
                return result

        # Default to medium if classification fails
        print(f"Warning: Could not extract classification from '{result_text[:50]}...', defaulting to medium", file=sys.stderr)
        return "medium"
            
    except Exception as e:
        print(f"Error classifying message: {e}", file=sys.stderr)
        return "sonnet"  # Safe default

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
