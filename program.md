# autoresearch

An experiment in having the LLM do its own research: autonomously modify `train.py`, run fixed-budget training experiments, and drive `val_bpb` down — keeping what works, discarding what doesn't.

The work is split across five role-specialized agents. This file holds the **shared ground truth** they all rely on (the facts their definitions point back to "from program.md") plus the setup procedure and how to start a run. Each agent's own responsibilities, tools, and artifact contract live in its definition.

## Agents

The **orchestrator runs in the main session** (Claude Code subagents can't reliably spawn nested subagents); it invokes the other four via the Agent tool.

- **[orchestrator](.claude/agents/orchestrator.md)** — plans experiments, dispatches the workers, tracks the time + experiment ledgers, schedules parallel experiments, decides keep-vs-discard, commissions the final run. Never edits code or runs commands.
- **[capacity-planner](.claude/agents/capacity-planner.md)** — discovers the hardware envelope (`experiments/hardware.md`) and runs `vram_calculator.py` to pre-flight each experiment (FITS/OOM) and plan conservative parallelism. Read-only.
- **[research-agent](.claude/agents/research-agent.md)** — mines `index/research/` and the code to write concrete experiment proposals (`experiments/proposals/`).
- **[llm-engineer](.claude/agents/llm-engineer.md)** — the only agent that edits `train.py`, commits, runs `uv run train.py`, debugs crashes, and executes git keep/reset. Produces `run.log`.
- **[evaluator](.claude/agents/evaluator.md)** — logs results to `results.tsv`, writes reports (`experiments/reports/`), recommends keep/discard. No code execution.

## Setup (once per run)

1. **Note the total time budget**: the user provides the total wall-clock time for the whole run (e.g. `8h`). With the hardware, this is the constraint the orchestrator optimizes under. Default split: **~60% experimentation, ~40% final training** (configurable).
2. **Agree on a run tag**: propose one based on today's date (e.g. `jun20`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Profile the hardware**: commission **capacity-planner** to discover the GPU/VRAM/vCPU/RAM/CUDA/MIG/NUMA envelope and write `experiments/hardware.md` *before* scheduling anything.
5. **Read the in-scope files** for full context: `README.md` (repo context), `prepare.py` (fixed constants, data, tokenizer, evaluation — read-only), `train.py` (the file you modify).
6. **Verify data exists**: check `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
7. **Initialize `results.tsv`** with just the header row (see schema below). The baseline is recorded after the first run.
8. **Confirm** setup looks good, then start the loop. The **first experiment is always the baseline** — run `train.py` as-is.

## Ground truth

**Goal**: get the lowest `val_bpb`. The metric is `evaluate_bpb` in `prepare.py` — it is the ground truth and must never be modified.

**Budget**: 5 minutes is the *default* per-experiment training budget, but the **orchestrator sets the budget per experiment** — short budgets (e.g. 60–120 s) for cheap parallel exploration, the full remaining time for the final run. The engineer overrides `TIME_BUDGET` via the `AUTORESEARCH_TIME_BUDGET` env var (no edit to read-only `prepare.py`); the script self-stops at that budget. Launch as `uv run train.py > run.log 2>&1`. Kill any run that overruns its assigned budget by ≈2×.

**Two-phase run + parallelism**: experiments run in parallel on GPU slices (MPS by default; MIG only if hard isolation is needed) to learn transferable patterns cheaply during the experimentation phase, then a **single final run uses the full GPU and the remaining time** to train the deliverable model with the learned recipe. Every experiment must get a capacity-planner FITS verdict before launch — be conservative, never oversubscribe the *free* VRAM/vCPU/RAM.

**VRAM** is a soft constraint: some increase is fine for meaningful `val_bpb` gains, but it shouldn't blow up dramatically.

**Simplicity criterion**: all else equal, simpler is better. A small gain that adds ugly complexity isn't worth it; removing code for equal-or-better results is a win. Weigh complexity cost against improvement magnitude (a 0.001 gain from 20 hacky lines? probably not. a 0.001 gain from deleting code? definitely keep. ~0 change but much simpler? keep).

**What can change**: only `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, model size; all fair game.

**What cannot change**: `prepare.py` (fixed constants, data, tokenizer, evaluation); the `evaluate_bpb` harness; the dependency set (no new packages beyond `pyproject.toml`).

**Autonomy — NEVER STOP**: once the loop begins, do not pause to ask the human whether to continue. They may be away and expect indefinite autonomous work (≈12 experiments/hour, ≈100 overnight). If ideas run dry, think harder: re-read `index/research/` and the in-scope files, combine near-misses, try more radical changes. The loop runs until the human interrupts.

## Output format

When the script finishes it prints a summary block:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key fields with: `grep "^val_bpb:\|^peak_vram_mb:" run.log`. Empty output means the run crashed (`tail -n 50 run.log` for the trace).

## results.tsv schema

Tab-separated (NOT comma — commas break the description column). Header row plus one row per experiment, **left untracked by git** (never commit it):

```
commit	val_bpb	memory_gb	status	description
```

1. `commit` — short 7-char git hash
2. `val_bpb` — e.g. `1.234567`; use `0.000000` for crashes
3. `memory_gb` — `peak_vram_mb / 1024`, rounded to .1f; use `0.0` for crashes
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short, comma-free summary of what the experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase matrix LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Starting a run

The orchestrator role lives in the main session. To kick off the loop, run the entrypoint slash command from the repo root:

```
/autoresearch <tag> <total-time>
```

e.g. `/autoresearch jun20 8h`. This tells the main session to adopt the orchestrator role, do setup for branch `autoresearch/<tag>`, profile the hardware, and run the two-phase loop within `<total-time>`. See [.claude/commands/autoresearch.md](.claude/commands/autoresearch.md).

For a headless / unattended run, launch Claude Code non-interactively with the same instruction:

```
claude "Adopt the orchestrator role defined in .claude/agents/orchestrator.md. Do the program.md setup for run tag jun20 with a total time budget of 8h, then run the two-phase experiment loop autonomously until interrupted or the budget is exhausted."
```
