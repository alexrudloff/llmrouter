# ROUTES.md - Intelligent Message Classification

*How messages are classified by complexity to route to appropriate models.*

## Philosophy

- **Start cheap, escalate when needed**
- **Task complexity determines routing, not person**
- **Learn from feedback** — patterns improve over time
- **This file evolves** — edit this table to tune classification behavior

## Classification Table

| Task Type | Complexity | Notes |
|-----------|------------|-------|
| Greetings | Super Easy | Hey, hi, what's up, how are you |
| Simple acknowledgments | Super Easy | OK, thanks, got it, sure |
| Very basic Q&A | Super Easy | What time is it? Simple yes/no questions |
| Basic chat / Q&A | Easy | Quick answers, casual conversation (not emails!) |
| Reminders | Easy | Setting simple reminders |
| Status checks | Easy | Reading and summarizing state |
| Simple questions | Easy | Math calculations, facts, definitions (no code, no emails) |
| Email tasks | Medium | Writing, sending, drafting any emails |
| Writing any code | Medium | Functions, scripts, even simple ones |
| Email judgment | Medium | Triage, prioritization, tone decisions |
| Research | Medium | Gathering and summarizing information |
| Coding - bug fixes | Medium | Single file changes, clear scope |
| Coding - complex/refactor | Hard | Multi-file, architecture decisions |
| Deep thinking / planning | Hard | When reasoning depth matters |
| Debugging complex issues | Hard | Segfaults, race conditions, system-level bugs |
| Advanced algorithms | Super Hard | Complex optimization, novel approaches |
| Multi-step reasoning | Super Hard | Proofs, deep analysis, research-level questions |
| System architecture | Super Hard | Designing entire systems, critical decisions |

## Pipelines (multi-model workflows)

### Research Pipeline
1. **Generate questions** — Sonnet (understands what to explore)
2. **Fetch & summarize pages** — Haiku (cheap, just extraction)
3. **Synthesize findings** — Sonnet or Opus (depending on depth needed)

### Coding Pipeline (experimental)
1. **Understand & plan** — Opus (architecture decisions)
2. **Execute plan** — Sonnet (write the code)
3. **Review if stuck** — Opus (debug complex issues)

## Complexity Levels

- **Super Easy** — Trivial tasks, could be handled by smallest local models
- **Easy** — Simple tasks, fast responses needed
- **Medium** — Standard work, balanced capability/cost
- **Hard** — Complex reasoning, deep analysis required
- **Super Hard** — Most difficult tasks, maximum capability needed

## Model Assignment (Configurable)

Default Anthropic setup:
- **Super Easy** → Haiku (or local model for cost savings)
- **Easy** → Haiku
- **Medium** → Sonnet
- **Hard** → Opus
- **Super Hard** → Opus

*Models can be configured in server.py MODEL_MAP to use different providers (OpenAI, Google, local) based on strengths and cost preferences.*

## Changelog

- 2025-02-01: Initial classification table created
- 2025-02-01: Upgraded to 5-tier system (super_easy/easy/medium/hard/super_hard)
- 2025-02-01: Renamed from ROUTES.md to CLASSIFIER.md, then back to ROUTES.md

---

*Edit this file to tune classification behavior over time.*
