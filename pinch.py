"""
Pinch: Intelligent Context Pruning (EXPERIMENTAL)

Scores message relevance against the query window using embeddings,
then applies keep/summarize/drop decisions based on thresholds.

This module is experimental and optional. Enable with --pinch flag.
"""

import json
import os
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "enabled": True,
    "budget_tokens": 50000,

    # Query window: protect recent turns
    "query_window": {
        "turns": 4,                    # Protect last N user turns
        "time_window_ms": 180000,      # Group messages within 3 minutes
    },

    # Relevance thresholds (cosine similarity)
    "thresholds": {
        "keep": 0.55,                  # Above this: keep verbatim
        "summarize": 0.25,             # Above this: summarize (below = drop)
    },

    # Summarization
    "summarization": {
        "enabled": True,
        "max_tokens": 150,             # Max tokens per summary
    },

    # Tool handling
    "tools": {
        "always_keep": ["memory_recall", "memory_search"],
        "always_drop": [],
    },

    # Embeddings configuration
    # IMPORTANT: Must match OpenClaw's memorySearch.model for cache compatibility!
    "embeddings": {
        "ollama_url": "http://127.0.0.1:11434",
        "model": "nomic-embed-text",       # Must match OpenClaw's memorySearch.model
        "openclaw_state_dir": None,        # Auto-detect ~/.openclaw for cache
    },
}


# ============================================================================
# Embedding Provider (uses OpenClaw's SQLite cache)
# ============================================================================

class EmbeddingProvider:
    """
    Provides embeddings using OpenClaw's SQLite cache or Ollama API.

    IMPORTANT: The embedding model MUST match OpenClaw's memorySearch.model
    for cache lookups to work. Both should use the same model (e.g., nomic-embed-text).
    """

    def __init__(self, config: dict):
        self.config = config
        embed_cfg = config.get("embeddings", {})
        self.state_dir = embed_cfg.get("openclaw_state_dir")
        if not self.state_dir:
            self.state_dir = os.environ.get("OPENCLAW_STATE_DIR",
                                            os.path.expanduser("~/.openclaw"))
        self.ollama_url = embed_cfg.get("ollama_url", "http://127.0.0.1:11434")
        self.model = embed_cfg.get("model", "nomic-embed-text")
        self.cache_db = None
        self._init_cache()

    def _init_cache(self):
        """Initialize connection to OpenClaw's embedding cache."""
        # Try to find an existing memory database with embeddings
        memory_dir = Path(self.state_dir) / "memory"
        if memory_dir.exists():
            for db_file in memory_dir.glob("*.sqlite"):
                try:
                    conn = sqlite3.connect(str(db_file), timeout=1.0)
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
                    )
                    if cursor.fetchone():
                        self.cache_db = conn
                        return
                    conn.close()
                except:
                    pass

    def _hash_text(self, text: str) -> str:
        """Create hash for cache lookup."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _get_cached(self, text_hash: str) -> Optional[list]:
        """Look up embedding in cache."""
        if not self.cache_db:
            return None
        try:
            cursor = self.cache_db.execute(
                "SELECT embedding FROM embedding_cache WHERE hash = ? LIMIT 1",
                (text_hash,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        except:
            pass
        return None

    def _call_local(self, text: str) -> Optional[list]:
        """
        Call local Ollama embedding model.
        Uses the model configured in embeddings.model (must match OpenClaw's memorySearch.model).
        """
        import requests

        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:8000]},
                timeout=10,
            )
            if response.ok:
                data = response.json()
                return data.get("embedding")
        except:
            pass
        return None

    def _call_openai(self, text: str) -> Optional[list]:
        """Call OpenAI embedding API (text-embedding-3-small to match cache)."""
        import requests

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": text[:8000],
                },
                timeout=10,
            )
            if response.ok:
                data = response.json()
                return data["data"][0]["embedding"]
        except:
            pass
        return None

    def _call_gemini(self, text: str) -> Optional[list]:
        """Call Google Gemini embedding API (fallback when OpenAI unavailable)."""
        import requests

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None

        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text[:8000]}]},
                },
                timeout=10,
            )
            if response.ok:
                data = response.json()
                return data.get("embedding", {}).get("values")
        except:
            pass
        return None

    def _call_api(self, text: str) -> Optional[list]:
        """
        Call embedding API with OpenClaw-compatible fallback chain.
        Order: local (Ollama) → OpenAI → Gemini

        This matches OpenClaw's auto-selection fallback behavior.
        IMPORTANT: For cache compatibility, OpenClaw and Pinch should
        be configured to use the same primary model. Fallbacks are for when
        the primary is unavailable.
        """
        # Try local Ollama first (configured model)
        embedding = self._call_local(text)
        if embedding:
            return embedding

        # Fall back to OpenAI (text-embedding-3-small)
        embedding = self._call_openai(text)
        if embedding:
            return embedding

        # Last resort: Gemini
        return self._call_gemini(text)

    def embed(self, text: str) -> Optional[list]:
        """Get embedding for text, using cache if available."""
        if not text or len(text.strip()) < 10:
            return None

        text_hash = self._hash_text(text)

        # Try cache first
        cached = self._get_cached(text_hash)
        if cached:
            return cached

        # Fall back to API
        return self._call_api(text)

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]


# ============================================================================
# Relevance Scoring
# ============================================================================

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def extract_text(msg: dict) -> str:
    """Extract text content from a message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
        return " ".join(parts)
    return str(content) if content else ""


def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return len(text) // 4 if text else 0


# ============================================================================
# Query Window Detection
# ============================================================================

def identify_query_window(messages: list, config: dict) -> dict:
    """
    Identify the query window - recent messages to protect.

    Uses turn count and time grouping to handle fragmented user messages.
    """
    qw_config = config.get("query_window", {})
    max_turns = qw_config.get("turns", 4)
    time_window_ms = qw_config.get("time_window_ms", 180000)

    if not messages:
        return {"protected": [], "candidates": [], "boundary_idx": 0}

    # Walk backwards to find protected messages
    protected_indices = set()
    user_turn_count = 0
    last_user_time = None

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        role = msg.get("role", "")

        # Always protect recent assistant messages tied to protected user turns
        if role == "assistant" and any(j > i for j in protected_indices if messages[j].get("role") == "user"):
            protected_indices.add(i)
            continue

        # Tool results tied to protected assistant messages
        if role == "tool":
            # Find the assistant message this tool result belongs to
            for j in range(i - 1, -1, -1):
                if messages[j].get("role") == "assistant":
                    if j in protected_indices:
                        protected_indices.add(i)
                    break
            continue

        if role == "user":
            msg_time = msg.get("timestamp", 0)

            # Time grouping: if within time window of last user message, group together
            if last_user_time and msg_time:
                time_diff = last_user_time - msg_time
                if time_diff <= time_window_ms:
                    protected_indices.add(i)
                    continue

            user_turn_count += 1
            if user_turn_count <= max_turns:
                protected_indices.add(i)
                last_user_time = msg_time
            else:
                break

    # Build result
    boundary_idx = min(protected_indices) if protected_indices else len(messages)
    protected = [messages[i] for i in sorted(protected_indices)]
    candidates = [messages[i] for i in range(len(messages)) if i not in protected_indices]

    return {
        "protected": protected,
        "candidates": candidates,
        "boundary_idx": boundary_idx,
        "protected_indices": protected_indices,
    }


# ============================================================================
# Scoring and Classification
# ============================================================================

def score_candidates(candidates: list, query_embedding: list, config: dict, embedder: EmbeddingProvider) -> list:
    """
    Score each candidate message for relevance to the query.

    Returns list of (message, score, action, reason) tuples.
    """
    thresholds = config.get("thresholds", {})
    keep_threshold = thresholds.get("keep", 0.55)
    summarize_threshold = thresholds.get("summarize", 0.25)

    tools_config = config.get("tools", {})
    always_keep = set(tools_config.get("always_keep", []))
    always_drop = set(tools_config.get("always_drop", []))

    scored = []

    for msg in candidates:
        role = msg.get("role", "")
        tool_name = msg.get("name") or msg.get("tool_name")
        text = extract_text(msg)
        tokens = estimate_tokens(text)

        # User messages always kept
        if role == "user":
            scored.append({
                "message": msg,
                "score": 1.0,
                "action": "keep",
                "reason": "user message",
                "tokens": tokens,
            })
            continue

        # Assistant messages always kept
        if role == "assistant":
            scored.append({
                "message": msg,
                "score": 1.0,
                "action": "keep",
                "reason": "assistant message",
                "tokens": tokens,
            })
            continue

        # Tool overrides
        if tool_name:
            if tool_name in always_keep:
                scored.append({
                    "message": msg,
                    "score": 1.0,
                    "action": "keep",
                    "reason": f"protected tool: {tool_name}",
                    "tokens": tokens,
                })
                continue

            if tool_name in always_drop:
                scored.append({
                    "message": msg,
                    "score": 0.0,
                    "action": "drop",
                    "reason": f"always drop: {tool_name}",
                    "tokens": tokens,
                })
                continue

        # Compute relevance score via embeddings
        score = 0.5  # Default if no embedding

        if query_embedding and text:
            msg_embedding = embedder.embed(text[:2000])  # Truncate for embedding
            if msg_embedding:
                score = cosine_similarity(query_embedding, msg_embedding)

        # Classify based on thresholds
        if score >= keep_threshold:
            action = "keep"
            reason = f"high relevance ({score:.2f})"
        elif score >= summarize_threshold:
            action = "summarize"
            reason = f"medium relevance ({score:.2f})"
        else:
            action = "drop"
            reason = f"low relevance ({score:.2f})"

        scored.append({
            "message": msg,
            "score": score,
            "action": action,
            "reason": reason,
            "tokens": tokens,
        })

    return scored


# ============================================================================
# Summarization
# ============================================================================

def summarize_items(items: list, config: dict) -> Optional[str]:
    """
    Generate a summary of multiple items using LLM.
    Falls back to extractive summary if LLM unavailable.
    """
    if not items:
        return None

    sum_config = config.get("summarization", {})
    if not sum_config.get("enabled", True):
        return create_extractive_summary(items)

    max_tokens = sum_config.get("max_tokens", 150)

    # Build content to summarize
    content_parts = []
    for item in items:
        text = extract_text(item["message"])[:500]
        role = item["message"].get("role", "tool")
        content_parts.append(f"[{role}]: {text}")

    content = "\n".join(content_parts)

    # Try LLM summarization
    summary = call_llm_summarize(content, max_tokens)
    if summary:
        return summary

    # Fallback to extractive
    return create_extractive_summary(items)


def create_extractive_summary(items: list) -> str:
    """Create a simple extractive summary."""
    bullets = []
    for item in items[:10]:  # Limit to 10 items
        text = extract_text(item["message"])
        first_line = text.split('\n')[0][:100] if text else ""
        if first_line:
            bullets.append(f"- {first_line}")

    if not bullets:
        return "Previous context (details omitted)"

    return "Context summary:\n" + "\n".join(bullets)


def call_llm_summarize(content: str, max_tokens: int) -> Optional[str]:
    """Call LLM for summarization."""
    import requests

    # Try local router first (port 4001)
    prompt = f"""Summarize this conversation context concisely. Focus on key facts, decisions, and information that might be referenced later. Use bullet points.

Context:
{content[:4000]}

Summary (max {max_tokens} tokens):"""

    try:
        response = requests.post(
            "http://127.0.0.1:4001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "localrouter/llm-router",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=15,
        )
        if response.ok:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
    except:
        pass

    return None


# ============================================================================
# Main Pruning Function
# ============================================================================

def prune_context(messages: list, config: dict) -> tuple:
    """
    Intelligently prune context based on relevance scoring.

    Args:
        messages: List of messages in OpenAI format
        config: Pinch configuration

    Returns:
        (pruned_messages, stats_dict)
    """
    if not messages:
        return messages, {"pruned": False}

    # Merge with defaults
    cfg = {**DEFAULT_CONFIG, **config}

    # Estimate current tokens
    total_tokens = sum(estimate_tokens(extract_text(m)) for m in messages)
    budget = cfg.get("budget_tokens", 50000)

    # Check if pruning needed
    if total_tokens <= budget:
        return messages, {
            "pruned": False,
            "tokens": total_tokens,
            "reason": "within budget",
        }

    # Initialize embedder
    embedder = EmbeddingProvider(cfg)

    # Identify query window
    qw = identify_query_window(messages, cfg)
    protected = qw["protected"]
    candidates = qw["candidates"]

    if not candidates:
        return messages, {
            "pruned": False,
            "tokens": total_tokens,
            "reason": "no candidates",
        }

    # Get query embedding from protected window
    query_text = " ".join(extract_text(m) for m in protected[-3:])  # Last 3 protected
    query_embedding = embedder.embed(query_text[:2000]) if query_text else None

    # If no embeddings available, fall back to simple position-based pruning
    if query_embedding is None:
        return prune_context_simple(messages, cfg)

    # Score candidates
    scored = score_candidates(candidates, query_embedding, cfg, embedder)

    # Categorize by action
    keep_items = [s for s in scored if s["action"] == "keep"]
    summarize_items_list = [s for s in scored if s["action"] == "summarize"]
    drop_items = [s for s in scored if s["action"] == "drop"]

    # Build scored lookup by message id
    scored_lookup = {id(s["message"]): s for s in scored}

    # Generate summary of items to summarize
    summary = None
    if summarize_items_list:
        summary = summarize_items(summarize_items_list, cfg)

    # Build pruned message list - maintain original order and tool pairing
    pruned = []
    tokens_saved = 0

    # Add summary at start if we have one
    if summary:
        pruned.append({
            "role": "system",
            "content": f"[Context Summary]\n{summary}",
        })

    # Process ALL messages in original order
    protected_ids = qw.get("protected_indices", set())

    for i, msg in enumerate(messages):
        # Protected messages pass through unchanged
        if i in protected_ids:
            pruned.append(msg)
            continue

        # Check if this message was scored
        score_info = scored_lookup.get(id(msg))

        if not score_info:
            # Not scored (shouldn't happen), keep it
            pruned.append(msg)
            continue

        action = score_info["action"]
        role = msg.get("role", "")

        # User and assistant messages always kept verbatim
        if role in ("user", "assistant", "system"):
            pruned.append(msg)
            continue

        # Tool results: apply action but keep structure for API compatibility
        if role == "tool":
            if action == "keep":
                pruned.append(msg)
            elif action == "summarize":
                # Replace with truncated version
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    tokens_saved += estimate_tokens(content) - 30
                    pruned.append({
                        **msg,
                        "content": f"[Summarized: {content[:200]}...]"
                    })
                else:
                    pruned.append(msg)
            else:  # drop
                # Replace with minimal placeholder (can't fully drop due to API)
                content = msg.get("content", "")
                tokens_saved += estimate_tokens(content) - 5
                tool_name = msg.get("name", "tool")
                pruned.append({
                    **msg,
                    "content": f"[{tool_name}: old result cleared]"
                })
        else:
            # Other message types - keep as is
            pruned.append(msg)

    # Calculate final stats
    final_tokens = sum(estimate_tokens(extract_text(m)) for m in pruned)

    stats = {
        "pruned": True,
        "original_tokens": total_tokens,
        "final_tokens": final_tokens,
        "tokens_saved": tokens_saved,
        "items_kept": len(keep_items),
        "items_summarized": len(summarize_items_list),
        "items_dropped": len(drop_items),
        "has_summary": summary is not None,
        "used_embeddings": query_embedding is not None,
    }

    return pruned, stats


# ============================================================================
# Fallback: Simple Position-Based Pruning
# ============================================================================

def prune_context_simple(messages: list, config: dict) -> tuple:
    """
    Simple position-based pruning fallback.
    Used when embeddings are unavailable.
    """
    if not messages:
        return messages, {"pruned": False}

    budget = config.get("budget_tokens", 50000)
    keep_last = config.get("keep_last_assistants", 3)

    total_tokens = sum(estimate_tokens(extract_text(m)) for m in messages)

    if total_tokens <= budget:
        return messages, {"pruned": False, "tokens": total_tokens}

    # Find cutoff
    assistant_count = 0
    cutoff_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            assistant_count += 1
            if assistant_count >= keep_last:
                cutoff_idx = i
                break

    # Soft trim old tool results
    pruned = []
    tokens_saved = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if role in ("user", "assistant", "system") or i >= cutoff_idx:
            pruned.append(msg)
            continue

        # Trim tool results
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 4000:
            head = content[:1500]
            tail = content[-1500:]
            trimmed = len(content) - 3000
            new_content = f"{head}\n\n[...{trimmed:,} chars trimmed...]\n\n{tail}"
            tokens_saved += trimmed // 4
            msg = {**msg, "content": new_content}

        pruned.append(msg)

    final_tokens = sum(estimate_tokens(extract_text(m)) for m in pruned)

    return pruned, {
        "pruned": True,
        "original_tokens": total_tokens,
        "final_tokens": final_tokens,
        "tokens_saved": tokens_saved,
        "mode": "simple",
    }
