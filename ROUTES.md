# ROUTES.md - Intelligent Message Classification

*How messages are classified by complexity to route to appropriate models.*

## Philosophy

- **Start cheap, escalate when needed**
- **Task complexity determines routing**

## Classification Table

| Task Type | Complexity | Notes |
|-----------|------------|-------|
| Greetings | Super Easy | Hey, hi, what's up, how are you |
| Simple acknowledgments | Super Easy | OK, thanks, got it, sure |
| Very basic Q&A | Super Easy | What time is it? Simple yes/no questions |
| Farewells | Super Easy | Bye, see you, talk later |
| Single word responses | Super Easy | Yes, no, maybe, done |
| Basic chat / Q&A | Easy | Quick answers, casual conversation (not emails!) |
| Reminders | Easy | Setting simple reminders |
| Status checks | Easy | Reading and summarizing state |
| Simple questions | Easy | Math calculations, facts, definitions (no code, no emails) |
| Format conversion | Easy | Convert units, dates, simple data transforms |
| Email tasks | Medium | Writing, sending, drafting any emails |
| Writing any code | Medium | Functions, scripts, even simple ones |
| Research | Medium | Gathering and summarizing information |
| Coding - bug fixes | Medium | Single file changes, clear scope |
| Image description | Medium | Describe what's in an image, read text from screenshots |
| Coding - complex/refactor | Hard | Multi-file, architecture decisions |
| Deep thinking / planning | Hard | When reasoning depth matters |
| Debugging complex issues | Hard | Segfaults, race conditions, system-level bugs |
| Image analysis - detailed | Hard | Extract structured data, compare images, diagrams |
| Tool use - chained | Hard | Multiple sequential tool calls, conditional logic |
| Advanced algorithms | Super Hard | Complex optimization, novel approaches |
| Multi-step reasoning | Super Hard | Proofs, deep analysis, research-level questions |
| System architecture | Super Hard | Designing entire systems, critical decisions |
| Agentic workflows | Super Hard | Autonomous multi-step tasks, complex tool orchestration |
| Long context - synthesis | Super Hard | Synthesizing insights across very large documents |

## Complexity Levels

- **Super Easy** — Trivial tasks, could be handled by smallest local models
- **Easy** — Simple tasks, fast responses needed
- **Medium** — Standard work, balanced capability/cost
- **Hard** — Complex reasoning, deep analysis required
- **Super Hard** — Most difficult tasks, maximum capability needed

