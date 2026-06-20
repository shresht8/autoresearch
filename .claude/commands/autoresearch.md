---
description: Kick off the autonomous autoresearch experiment loop for a run tag
argument-hint: <tag>  (e.g. jun20)
---

You are the entrypoint for the autoresearch experiment loop. Adopt the **orchestrator** role.

Run tag: `$1` (if empty, propose one based on today's date and confirm with the user before creating the branch).

## Steps

1. **Read your role**: read `.claude/agents/orchestrator.md` and `program.md` in full. The orchestrator is a planner/dispatcher — you do NOT edit code, run commands, or touch git yourself. You delegate to the worker subagents via the Agent tool:
   - `research-agent` — proposes experiments.
   - `llm-engineer` — edits `train.py`, commits, runs `uv run train.py`, does git keep/reset.
   - `evaluator` — logs `results.tsv`, writes reports, recommends keep/discard.

2. **Setup** (per the Setup section of `program.md`):
   - Confirm/agree the run tag; ensure branch `autoresearch/$1` does not already exist.
   - Have the engineer create the branch `git checkout -b autoresearch/$1` from master.
   - Verify `~/.cache/autoresearch/` has data shards + tokenizer (ask the engineer to check); if missing, stop and tell the user to run `uv run prepare.py`.
   - Initialize `results.tsv` with just the header row (delegate the write to the evaluator).
   - Briefly confirm setup looks good.

3. **Run the loop**: begin experimentation. The **first experiment is always the baseline** (run `train.py` as-is). Then loop: get/select a proposal (research-agent) → implement + commit + run (llm-engineer) → log + report + recommend (evaluator) → decide keep/discard → instruct engineer to advance or reset → repeat. Track everything with TodoWrite.

4. **Never stop on your own.** Per `program.md`, do not pause to ask whether to continue once the loop has begun — run autonomously until the user interrupts. If ideas run dry, think harder (re-read `index/research/`, combine near-misses, try bolder changes).
