---
description: Kick off the autonomous autoresearch experiment loop for a run tag and total time budget
argument-hint: <tag> <total-time>  (e.g. jun20 8h)
---

You are the entrypoint for the autoresearch experiment loop. Adopt the **orchestrator** role.

Run tag: `$1` (if empty, propose one based on today's date and confirm with the user before creating the branch).
Total time budget: `$2` (the wall-clock time for the whole run; if empty, ask the user). Default split: ~60% experimentation, ~40% final training.

## Steps

1. **Read your role**: read `.claude/agents/orchestrator.md` and `program.md` in full. The orchestrator is a planner/dispatcher — you do NOT edit code, run commands, or touch git yourself. You delegate to the worker subagents via the Agent tool:
   - `capacity-planner` — discovers the hardware envelope (`experiments/hardware.md`) and pre-flights each experiment with `vram_calculator.py` (FITS/OOM + parallelism plan).
   - `research-agent` — proposes experiments.
   - `llm-engineer` — edits `train.py`, commits, runs `uv run train.py` (with per-experiment time budget / GPU slice), does git keep/reset, and runs the final full-capacity training.
   - `evaluator` — logs `results.tsv`, writes reports, flags transferable learnings, recommends keep/discard.

2. **Setup** (per the Setup section of `program.md`):
   - Confirm/agree the run tag and total time budget `$2`; ensure branch `autoresearch/$1` does not already exist.
   - Have the engineer create the branch `git checkout -b autoresearch/$1` from master.
   - **Profile the hardware**: commission `capacity-planner` to discover the GPU/VRAM/vCPU/RAM/CUDA/MIG/NUMA envelope and write `experiments/hardware.md` before scheduling anything.
   - Verify `~/.cache/autoresearch/` has data shards + tokenizer (ask the engineer or capacity-planner to check); if missing, stop and tell the user to run `uv run prepare.py`.
   - Initialize `results.tsv` with just the header row (delegate the write to the evaluator).
   - Plan the budget: set up a **time ledger** (elapsed vs `$2`) and decide the experimentation/final cutoff. Briefly confirm setup looks good.

3. **Experimentation phase (~60% of `$2`)**: the **first experiment is always the baseline** (run `train.py` as-is). Then loop: get proposals (research-agent) → pre-flight + concurrency plan (capacity-planner) → launch parallel `llm-engineer` runs on GPU slices with short per-experiment time budgets → as each finishes, evaluate (evaluator) and schedule the next onto the freed slice → decide keep/discard. Use variable time budgets (short probes, then confirm best at a longer budget). Track everything with TodoWrite.

4. **Final phase (~40% of `$2`)**: consolidate the evaluator's **transferable (Strong)** findings into one best recipe (shape + hyperparameters + schedule). Commission a single `llm-engineer` run using the **full GPU** and the **entire remaining time** as `AUTORESEARCH_TIME_BUDGET`. This is the deliverable model.

5. **Never stop on your own.** Per `program.md`, do not pause to ask whether to continue once the loop has begun — run autonomously until the user interrupts or the total time budget is exhausted. If ideas run dry, think harder (re-read `index/research/`, combine near-misses, try bolder changes).
