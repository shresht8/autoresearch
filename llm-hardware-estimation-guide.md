# Understanding LLM Training Hardware

**VRAM, vCPU and throughput estimation for single-GPU experiments**

Companion to the autoresearch experiment loop. Target workloads: 25M–2B parameter PyTorch pretraining on H100 / GH200 / A100.

---

## 1. Introduction

### 1.1 What This Guide Covers

This guide explains how a PyTorch LLM training run consumes the hardware on a single GPU box — GPU memory (VRAM), GPU compute (SMs), host CPU cores (vCPU), and system RAM — and how to predict those numbers from your architecture and batching choices **before** you launch a run. It is designed to be used alongside the autoresearch experiment loop, where every experiment runs against a fixed 5-minute wall-clock budget on one GPU.

Use this guide when you are:

- Sizing a new model or a hyperparameter change and want to know if it will fit on the target GPU.
- Debugging an OOM crash and need to know which knob to lower first.
- Comparing two recipes (e.g. AdamW vs Muon, micro-batch 64 vs 128) on a memory and throughput basis.
- Planning two parallel experiments on a GH200 — combine with [gh200-parallel-training-guide.docx](gh200-parallel-training-guide.docx) to slice the GPU into MIG/MPS halves and estimate per-half budgets.

The guide ships with a Python calculator ([vram_calculator.py](vram_calculator.py)) that does the arithmetic. The math here is intentionally simple and conservative: it is meant to keep you out of the OOM ditch, not to predict peak VRAM to the megabyte.

### 1.2 Key Terms

Before diving in, here are the terms used throughout this guide:

| Term | What it means |
|---|---|
| **VRAM (Video RAM)** | The GPU's dedicated high-bandwidth memory. This is where model weights, gradients, optimizer state, and activations live during training. When this fills up, training crashes with an out-of-memory (OOM) error. |
| **SM (Streaming Multiprocessor)** | A compute unit on the GPU. An H100 has 132 SMs; the more SMs are busy doing useful matmuls, the higher your throughput. Reported indirectly via MFU. |
| **MFU (Model FLOPs Utilization)** | Fraction of the GPU's theoretical peak FLOPS your training actually achieves. 40–55% is normal for a well-tuned bf16 transformer; below 20% means a bottleneck (data pipeline, small matmuls, Python overhead). |
| **FLOPs / token** | Floating-point operations needed to push one token through forward + backward. Scales as ~`6N` (where `N` = non-embedding parameter count) plus an attention correction. |
| **Activations** | The intermediate tensors a forward pass produces (attention outputs, MLP hidden states, etc.) and stores so the backward pass can compute gradients. At large batch / long context, **activations are the single largest VRAM line item**. |
| **bf16 / fp16 / fp32** | Number formats. `bf16` and `fp16` use 2 bytes; `fp32` uses 4. Modern LLMs train in `bf16` with `fp32` accumulators — half the memory of pure `fp32` with negligible accuracy loss. |
| **Mixed precision** | A recipe where parameters are stored once in `fp32` (the *master copy*) and cast to `bf16` for forward/backward. Adds ~50% to parameter memory in exchange for numerical stability. |
| **Gradient accumulation** | Running several micro-batches and summing their gradients before stepping the optimizer. Lets you train at a large effective batch size on hardware that can only fit a small micro-batch. |
| **DataLoader worker** | A CPU process (spawned by PyTorch's `DataLoader`) that decodes / tokenizes / collates the next batch in parallel with GPU compute. More workers = more CPU and RAM, but also better GPU utilization. |
| **Pinned memory** | Host RAM marked non-swappable so the GPU can DMA from it without an intermediate copy. Required for the asynchronous CPU→GPU transfers PyTorch uses by default. |
| **NUMA** | Non-Uniform Memory Access. On a multi-socket or Grace+Hopper box, RAM attached to one CPU is slower for another CPU. Pin your training process to the cores closest to your GPU (`numactl --physcpubind`). |
| **FlashAttention** | An attention kernel that streams the QK<sup>T</sup> softmax and never materialises the full `s × s` attention matrix. Cuts attention activations from `O(s² × h)` to `O(s × h)` — the difference between "fits" and "OOM" at long context. |
| **Optimizer state** | Per-parameter buffers the optimizer keeps between steps — e.g. Adam's first/second moments (~8 bytes/param in fp32). Often as large as the model itself. |

### 1.3 The autoresearch Training Loop at a Glance

The autoresearch loop runs a single file ([train.py](train.py)) on a single GPU for a 5-minute budget. The relevant moving parts — the knobs that this guide and the calculator reason about — are:

| Knob | Default | What it controls | VRAM impact |
|---|---|---|---|
| `DEPTH` | 8 | Transformer layers | ~linear (params + activations) |
| `ASPECT_RATIO` | 64 | Model dim per layer: `n_embd = depth * 64` (rounded to head_dim) | quadratic on attention/MLP params, linear on activations |
| `HEAD_DIM` | 128 | Per-head dimension; `n_head = n_embd / head_dim` | neutral if you keep `n_embd` fixed |
| `DEVICE_BATCH_SIZE` | 128 | Micro-batch per forward pass | linear — the #1 knob to drop on OOM |
| `TOTAL_BATCH_SIZE` | 524,288 tokens | Effective batch; sets `grad_accum_steps = TOTAL / (DEVICE * SEQ)` | negligible (grad accum is sequential, not parallel) |
| `WINDOW_PATTERN` | `SSSL` | Sliding window per layer (S=half ctx, L=full) | minor on attention compute; FA cost dominates |
| Optimizer (`MuonAdamW`) | fixed | Muon for 2D matrices, AdamW for embeds/scalars | ~2 bytes/param × dtype state size |

**Fixed (read-only) constants from [prepare.py](prepare.py):**

- `MAX_SEQ_LEN = 2048` — context length.
- `TIME_BUDGET = 300` — seconds the run trains for after warmup.
- `VOCAB_SIZE = 8192` — BPE tokenizer vocabulary.

---

## 2. How VRAM is Used During Training

**Peak VRAM = the maximum of (model parameters + gradients + optimizer state + activations + CUDA workspaces) at any instant during a training step.** Knowing how big each piece is, and *when* in the step it peaks, tells you which knob to turn.

### 2.1 The Five Buckets

Every byte of VRAM lives in one of these buckets:

| Bucket | When it lives in VRAM | Typical share of peak | Scales with |
|---|---|---|---|
| Model parameters | Allocated at model build, freed only at shutdown | 1–5% (small), 30–50% (multi-B) | `N × param_dtype_bytes` |
| Gradients | Allocated on the first backward, kept until `optimizer.step()` | Same size as params | `N × grad_dtype_bytes` |
| Optimizer state | Allocated lazily on the first `optimizer.step()` | 1× – 2× parameter size | `N × state_dtype_bytes × state_multiplier` |
| **Activations** | Allocated on forward, freed gradually during backward | **DOMINANT** at large batch / long context | `batch × seq_len × n_embd × n_layer` |
| CUDA workspaces / kernels / NCCL buffers / fragmentation | Always live | 0.5–2 GB | roughly constant |

### 2.2 Why Activations Are Usually the Biggest

Each transformer block saves a handful of intermediate tensors per token so the backward pass can compute gradients. With bf16, the per-layer activation cost is roughly:

```
activations_per_layer (bytes) ≈ activation_factor × batch × seq_len × n_embd
                                 with activation_factor ~ 12–40 bytes per (b·s·h)
```

For the autoresearch baseline (depth=8, n_embd=512, seq=2048, micro-batch=128), this evaluates to:

```
activations ≈ 40 × 128 × 2048 × 512 × 8 layers
            ≈ 43,000 MB ≈ 42 GB
```

Compare that to the parameters themselves (50M × 2 bytes = 100 MB) and you see why dropping `DEVICE_BATCH_SIZE` is the first lever when a run OOMs: it is the only one that touches the biggest bucket linearly.

> **Tip:** FlashAttention is the reason activations are linear in `seq_len` rather than quadratic. If you propose an experiment that disables FA, predict roughly `batch × n_head × seq_len² × 4` bytes of extra attention memory per layer — this number explodes past 100 GB very quickly.

### 2.3 Bytes per Parameter for Common Recipes

Each precision / optimizer combination has a fixed "tax" per parameter beyond the model itself. Multiply this by the parameter count `N` to get the params+grads+optim contribution.

| Recipe | Params | Grads | Optim state | **Bytes / param** | Notes |
|---|---:|---:|---:|---:|---|
| bf16 + AdamW (everything bf16) | 2 | 2 | 4 | **8** | Used by this repo's `wte` / value embeddings / scalars. |
| fp32 + AdamW | 4 | 4 | 8 | **16** | Classic full-precision. Rare for LLMs > 100M. |
| bf16 mixed (fp32 master + AdamW state) | 2 + 4 | 2 | 8 | **16** | Stable but heavy. Mainstream HF default. |
| bf16 + Muon (matrix params) | 2 | 2 | ~2 | **~6** | Muon stores one momentum buffer + tiny second-moment scalar. |
| bf16 + Lion | 2 | 2 | 2 | **6** | Single momentum buffer. |
| bf16 + plain SGD | 2 | 2 | 0 | **4** | Almost never used for LLMs; a lower bound. |

### 2.4 The Step Timeline (When Peak Happens)

During one training step the VRAM occupancy moves through a predictable curve. Knowing the shape lets you predict whether you have headroom for a doubling, or whether you are already at the cliff.

```
t0: idle               → params + optim state held
t1: forward begins     → + activations growing per layer
t2: forward complete   → peak A (params + optim + full activations)
t3: backward begins    → + gradients allocated; activations freed layer-by-layer
t4: backward midway    → peak B (params + optim + grads + half activations)   ← usually true peak
t5: optimizer.step()   → + temporary buffers (Muon stacked grads, Adam denom) ← spike
t6: zero_grad          → grads freed → back to (params + optim state)
```

Practical implication: the calculator's "peak VRAM" estimate is the **t4–t5 maximum**. Reducing `DEVICE_BATCH_SIZE` primarily attacks t4; switching from Adam to Muon attacks t1 and t6.

---

## 3. The Estimation Formulas

This section gives you the closed-form math the calculator runs. Useful when you want a back-of-the-envelope number without running anything.

### 3.1 Parameter Count

For a decoder-only Transformer with embeddings (no tying), per-layer:

```
attention_per_layer = 4 × n_embd²                          (MHA: q, k, v, proj)
                    = n_embd × (q_dim + 2·kv_dim + n_embd) (GQA: smaller k, v)
mlp_per_layer       = 2 × mlp_mult × n_embd²               (c_fc + c_proj, mlp_mult=4)

params = 2 × vocab_size × n_embd                  (wte + lm_head)
       + n_layer × (attention_per_layer + mlp_per_layer)
```

For the autoresearch baseline (depth=8, n_embd=512, vocab=8192, mlp_mult=4, plus value embeddings on every other layer):

```
embeddings    = 2 × 8192 × 512                  =   8.4 M
attention     = 8 × 4 × 512²                    =   8.4 M
mlp           = 8 × 2 × 4 × 512²                =  16.8 M
value embeds  = 4 × 8192 × 512                  =  16.8 M
TOTAL                                           ≈  50.3 M   ← matches measured num_params_M
```

### 3.2 Peak VRAM

Add the buckets and multiply by a fragmentation factor:

```
params_bytes        = N × param_dtype_bytes
grad_bytes          = N × grad_dtype_bytes
optim_state_bytes   = N × optim_mult × state_dtype_bytes      (see table 2.3)
master_copy_bytes   = N × 4                                    (only if using fp32 master)
activation_bytes    = activation_factor × batch × seq_len × n_embd × n_layer
cuda_overhead_bytes = 1 GB                                     (cuBLAS workspaces, kernels, allocator caches)

peak_VRAM ≈ 1.10 × (params + grads + optim + master + activations + overhead)
```

> **Tip:** The 1.10 fragmentation factor is empirical: PyTorch's caching allocator holds blocks slightly larger than requested. The calculator uses this same factor.

### 3.3 FLOPs per Token and Throughput Ceiling

```
flops_per_token  ≈  6 × N_non_embedding   +   12 × n_head × head_dim × seq_len × n_layer
                                              ^ attention correction (full causal)
tokens_per_sec   ≈  (peak_TFLOPS × MFU) / flops_per_token
step_time        ≈  tokens_per_step / tokens_per_sec
```

For autoresearch's depth=8 baseline on an H100 SXM (989.5 TFLOPS BF16 peak), assuming 45% MFU, the calculator predicts ~1.77 M tokens/sec, or ~296 ms per 524K-token optimizer step. The observed run reports ~40% MFU and ~1.6 M tok/s — the prediction is within 10%.

### 3.4 Host CPU and RAM

Heuristics for the data pipeline side:

```
dataloader_workers ≈ min(16, max(2, device_batch_size / 16))
vCPU_recommended    = dataloader_workers × 2 + 4              (driver + main + bg threads)
RAM_recommended     = 1 GB python baseline
                    + 2 GB tokenizer + parquet row groups
                    + workers × 2 × (batch × seq × 8 bytes)   (pinned prefetch, int64 ids)
```

For the baseline (b=128, s=2048, 8 workers) that is ~3 GB total RAM and ~20 vCPU. A 16-core / 32 GB host is plenty; the autoresearch run is GPU-bound, not host-bound.

---

## 4. Using the Calculator

[vram_calculator.py](vram_calculator.py) is a pure-stdlib Python script. No dependencies, no install — run it from the repo root with the system Python.

### 4.1 Quick Reference

```bash
# 1. Inspect this repo's current train.py recipe
python vram_calculator.py --preset autoresearch

# 2. Same recipe but predict for a deeper model
python vram_calculator.py --preset autoresearch --depth 12

# 3. Will batch 256 fit on H100?
python vram_calculator.py --preset autoresearch --device-batch-size 256 --gpu h100-sxm

# 4. Bare custom architecture (e.g. 1B model on H200)
python vram_calculator.py \
    --depth 24 --head-dim 128 --n-embd 2048 --seq-len 4096 \
    --vocab-size 50272 --device-batch-size 8 --total-batch-size 524288 \
    --optimizer adamw --master-weights --gpu h200

# 5. JSON output for piping into other scripts
python vram_calculator.py --preset autoresearch --json
```

### 4.2 Reading the Output

A typical report has four sections:

1. **Parameter count** — broken down by embedding / attention / MLP / value-embed / misc. Cross-check against the `train.py num_params_M` log to confirm your spec is correct.
2. **Peak VRAM breakdown** — the five buckets from §2.1, plus the fragmentation-adjusted peak and a `FITS` / `WILL OOM` verdict against the chosen GPU.
3. **Compute / throughput** — FLOPs/token, predicted tokens/sec ceiling at the assumed MFU, and predicted step time. Compare against your actual run's `tok/sec` to back out real MFU.
4. **Host** — recommended DataLoader workers, vCPU, RAM headroom, pinned-memory buffer. These are conservative; you can usually go lower.

### 4.3 Calibrating the Activation Factor

The single fudge factor in the model is `activation_bytes_per_bsh` — how many bytes of saved activations each layer holds per `batch × seq × hidden` unit. The default is **40**, tuned against this repo's depth=8 baseline (measured peak ~45 GB). If you fork the training script and reduce its activation footprint (e.g. by checkpointing, dropping value embeddings, or removing `torch.compile`), re-calibrate:

```bash
# 1. Run your modified script once. Look at run.log:
grep '^peak_vram_mb:' run.log         # e.g. peak_vram_mb: 32100

# 2. Subtract the non-activation buckets the calculator already prints:
#    params + grads + optim + cuda_overhead (call this 'fixed')
#    measured_activations = peak_vram_mb / 1.10 - fixed_mb

# 3. Divide by b × s × h × L to get bytes per (b·s·h) per layer:
#    new_factor = measured_activations × 1024² / (batch × seq × n_embd × n_layer)

# 4. Pass it back:
python vram_calculator.py --preset autoresearch --activation-bytes-per-bsh 28
```

### 4.4 Worked Example: Will depth=12 fit on an H100?

Run:

```bash
python vram_calculator.py --preset autoresearch --depth 12 --gpu h100-sxm
```

The calculator widens `n_embd` to 768 (12 × 64), grows attention/MLP per layer accordingly, scales activations linearly with depth, and reports:

```
TOTAL                 :    135,267,480  (135.3 M)
Estimated peak        :    103,552 MB  (101.1 GB)
Headroom target       : <= 68.0 GB  (15% reserved)
Verdict               : WILL OOM
```

Mitigation candidates (try in order):

- Drop `DEVICE_BATCH_SIZE` 128 → 64 — activations halve, taking peak from 101 GB to ~52 GB. **FITS.**
- Enable activation checkpointing — trades ~30% slower steps for ~`sqrt(L)` less activation memory.
- Reduce `ASPECT_RATIO` — shrinks `n_embd`, which shrinks params quadratically and activations linearly.

---

## 5. Verification, Monitoring and Debugging

Before relying on a prediction, sanity-check what the box actually reports. These commands work on any NVIDIA Linux box where you have `nvidia-smi` available.

### 5.1 Pre-Flight Checks

```bash
# Confirm the GPU and driver are visible
nvidia-smi

# CUDA / PyTorch can see the GPU and report capability (9.0 = Hopper)
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"

# Free VRAM right now (before launching anything)
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits

# CPU topology and NUMA layout
nproc
lscpu | head -20
numactl -H
```

### 5.2 Watching a Run in Real Time

| Tool | What it shows | Install / usage |
|---|---|---|
| `nvidia-smi dmon` | Per-GPU SM and memory utilisation polled every N seconds. Works on all NVIDIA driver versions. | `nvidia-smi dmon -s u -d 2` |
| `nvitop` | Colour terminal UI. Lists processes, per-GPU memory, MIG instances. Interactive sort + kill. | `pip install nvitop && nvitop` |
| `dcgmi dmon` (DCGM) | The only way to get **per-MIG-instance** SM utilisation. Required if you partition the GPU. | `dcgmi dmon -e 1001,1002 -d 1000` |
| `htop` / `top` | Per-core CPU usage on the host. Useful to check that DataLoader workers are saturating CPU. | `htop` |
| `py-spy` | Live profiler that attaches to a running Python process. Use when a step is slow but the GPU is idle (Python bottleneck). | `pip install py-spy && py-spy top --pid <pid>` |

### 5.3 PyTorch Memory Inspection

From inside `train.py`, useful one-liners while debugging an OOM:

```python
import torch

# Current allocator state
torch.cuda.memory_allocated() / 1e9          # GB currently held by tensors
torch.cuda.max_memory_allocated() / 1e9      # peak since last reset
torch.cuda.memory_reserved() / 1e9           # GB held by the caching allocator
torch.cuda.reset_peak_memory_stats()         # reset the peak counter

# Full per-pool breakdown (very verbose, useful when chasing fragmentation)
print(torch.cuda.memory_summary(abbreviated=False))

# Snapshot the allocator history for offline analysis with PyTorch's memory viz
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... run a step ...
torch.cuda.memory._dump_snapshot('/tmp/mem.pkl')
# Open in https://pytorch.org/memory_viz/
```

### 5.4 Reading peak_vram_mb from a Finished Run

After every autoresearch run the script prints a summary block:

```bash
grep '^peak_vram_mb:' run.log
# → peak_vram_mb:     45060.2
```

Divide by 1024 for GB, then compare to the calculator's prediction for the same config. A persistent gap of more than ~20% means the calculator's activation factor needs re-calibrating for your fork — see §4.3.

### 5.5 Common OOM Signatures and First Response

| Symptom in run.log / stderr | What it means | First thing to try |
|---|---|---|
| `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GB` | A specific tensor alloc failed mid-step. | Halve `DEVICE_BATCH_SIZE`. If still OOM, halve again. |
| OOM only after several steps | Memory fragmentation. The allocator can't find a contiguous block even though sums fit. | Set `PYTORCH_ALLOC_CONF=expandable_segments:True` (already set in `train.py`). |
| OOM during `evaluate_bpb` (after training) | Eval batch size too large. | Lower the batch_size passed to `evaluate_bpb` (it follows `DEVICE_BATCH_SIZE`). |
| `nvidia-smi` shows 0 MiB free at idle | Another process leaked GPU memory (e.g. crashed prior run). | `fuser -v /dev/nvidia*` then kill the stuck PID. Or reboot. |
| Step time doubles partway through | Driver auto-throttle from thermal or power cap. | `nvidia-smi -q -d POWER,TEMPERATURE` — check if power limit was hit. |
| MFU < 20%, GPU at 100%, but slow | Step is compute-bound but the kernel mix is bad (e.g. tiny matmuls, no flash attn). | Verify FA3 is loaded; check `n_embd` is a multiple of 64; profile with `py-spy`. |
| MFU < 20%, GPU often at 0–30% | Data pipeline starved. CPU can't keep up. | Check `num_workers`; check tokenizer throughput; pin CPU with `numactl`. |

---

## 6. Designing Experiments for autoresearch

Every proposal the research-agent writes commits llm-engineer to a 5-minute training run. Use the calculator before each commit to avoid wasting a run on a config that will OOM.

### 6.1 Pre-Flight Checklist for a Proposal

1. Identify which knobs the proposal moves (depth, n_embd, batch size, optimizer, seq len).
2. Run the calculator with `--preset autoresearch` and the proposed knob values.
3. Confirm `Verdict: FITS` with at least 15% headroom on the target GPU.
4. If the verdict is `WILL OOM`, decide: shrink the change, add activation checkpointing, or split into two smaller experiments.
5. Note the predicted `step_time`. If a step takes > 1 s, the 5-minute budget will fit fewer than 300 optimizer steps — which may not be enough to see a signal.
6. Cross-check the FLOPs/token estimate against the current `train.py` log (`estimate_flops` in `GPT.estimate_flops`) for sanity.

### 6.2 Knob Cheat Sheet (Effect on VRAM and Throughput)

| Knob change | VRAM effect | Throughput effect | Watch out for |
|---|---|---|---|
| `DEVICE_BATCH_SIZE × 2` | +~ batch × activations (often +20–30 GB at b=128) | +30–50% tok/sec (better MFU) | Most common OOM cause. |
| `DEPTH +1` | +~12% (params + activations both linear) | −12% tok/sec | Also shifts learning dynamics; not pure infra. |
| `ASPECT_RATIO × 1.5` | +~50% on params/grads, +50% on activations | −30% tok/sec, higher MFU | Quadratic on matmul cost; big effect on flops/token. |
| `seq_len × 2` | +~ 2× activations | Roughly neutral tok/sec, lower MFU | **If FA disabled, +4× attention memory!** |
| AdamW → Muon (matrices) | −50% on optimizer state for those params | Neutral (small overhead) | Different convergence behaviour. |
| fp32 master copy enabled | +4 bytes/param (e.g. +200 MB at 50M) | Neutral | Often the missing 5% of measured peak. |
| `WINDOW_PATTERN` all-L (full attn) | Minor on activations (FA handles it) | −10–20% (more attn FLOPs per layer) | Check FA still handles `seq_len`. |
| Disable `torch.compile` | −5–10% (no compile workspace) | −15–30% (no fused kernels) | Useful for debugging only. |
| Activation checkpointing on | −~`sqrt(L)` on activations (often −50%) | −25–35% throughput | Trades compute for memory; revisit before scaling depth. |

### 6.3 Sane Default Envelopes per GPU

Rough largest "safe" single-GPU autoresearch-style config (5-min budget, 85% of VRAM as the headroom target):

| GPU | VRAM | Safe max `DEPTH` (n_embd auto) | Safe max `DEVICE_BATCH_SIZE` | Comment |
|---|---:|---|---:|---|
| RTX 4090 | 24 GB | 6 (n_embd 384) | 64 | Tight; need ckpt for depth=8. |
| L40S | 48 GB | 8 (n_embd 512) | 128 | Roughly matches the autoresearch baseline. |
| A100 40 GB | 40 GB | 8 (n_embd 512) | 96 | Need to drop batch below 128. |
| A100 80 GB / H100 SXM 80 GB | 80 GB | 12 (n_embd 768) | 128 | Comfortable; can push to depth=14 with batch=96. |
| H200 | 141 GB | 16 (n_embd 1024) | 128 | Activations no longer the bottleneck below 2B params. |
| GH200 96 GB | 96 GB | 12 (n_embd 768) | 128 | Same as H100 but with C2C for streaming bigger datasets. |
| GH200 144 GB | 144 GB | 18 (n_embd 1152) | 128 | Largest single-GPU envelope for this loop. |

> **Warning:** These are estimates from the calculator — always run `vram_calculator.py` with your exact config first. They assume the autoresearch recipe (bf16, FlashAttention, MuonAdamW, no checkpointing).

### 6.4 If You're Splitting a GH200 with MIG or MPS

When running two parallel autoresearch experiments on one GH200 (see the [companion gh200 guide](gh200-parallel-training-guide.docx)), estimate each half independently against a smaller GPU profile:

- **MPS 50/50 split:** estimate each job against `--gpu gh200-96` but with a 40 GB headroom target (`--headroom-pct ~58`).
- **MIG 3g.48gb × 2:** estimate each job against `--gpu a100-40` (closest VRAM size) and treat the 48 GB cap as the ceiling. Note that MIG disables NVLink-C2C on GH200 (see gh200 guide §4.3).
- Always halve your DataLoader workers (and your vCPU estimate) when running two jobs — the 72 Grace cores split into ~32 cores each after driver overhead.

---

## 7. Quick-Start Cheat Sheet

### 7.1 Estimate Before Every Experiment

```bash
# Baseline
python vram_calculator.py --preset autoresearch

# What you actually want to test
python vram_calculator.py --preset autoresearch --depth 10 --device-batch-size 96

# Sanity check on a non-autoresearch GPU (e.g. A100)
python vram_calculator.py --preset autoresearch --gpu a100-80
```

### 7.2 Watch the Run

```bash
# Terminal A: launch
uv run train.py > run.log 2>&1

# Terminal B: live GPU usage
nvidia-smi dmon -s u -d 2

# Terminal C: live process / CPU
nvitop      # or: htop
```

### 7.3 After the Run

```bash
grep '^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^num_params_M:' run.log

# Compare measured peak to the calculator's prediction:
python vram_calculator.py --preset autoresearch --json | python -c \
  "import json,sys; d=json.load(sys.stdin); print('predicted_gb:', round(d['memory']['peak_vram_gb'],1))"
```

### 7.4 OOM Triage (Top 3)

1. **Halve `DEVICE_BATCH_SIZE`** (keep `TOTAL_BATCH_SIZE` the same; `grad_accum` doubles automatically).
2. If still OOM, **enable activation checkpointing** (~30% slower steps, ~half activations).
3. If still OOM, **shrink `ASPECT_RATIO` by 25% or `DEPTH` by 2** (revisit whether the experiment is in scope for this GPU).

### 7.5 Throughput Triage

1. Confirm MFU > 35% first via `grep '^mfu_percent:' run.log`. If yes, you are compute-bound and only architecture changes will help.
2. If MFU < 25%, run `nvitop`. If GPU often idles, the pipeline is starved — raise DataLoader workers and check `num_workers` vs num CPU cores.
3. If MFU < 35% with GPU pegged, profile with `py-spy top --pid $(pgrep -f train.py)` to find the Python hot path.

---

## Appendix: Mapping Calculator Outputs to run.log Fields

Every line `train.py` prints at the end has a calculator counterpart you can verify against.

| `run.log` field | Calculator field | How they should compare |
|---|---|---|
| `num_params_M` | `params.total / 1e6` | Should match within < 0.1%. If not, your spec is wrong — check vocab, depth, head_dim. |
| `peak_vram_mb` | `memory.peak_vram_mb` | Should match within ~10%. Larger gap → recalibrate `activation_bytes_per_bsh`. |
| `mfu_percent` | `throughput.target_mfu × 100` | The calculator assumes; the script measures. Plug measured back in for re-estimating. |
| `total_tokens_M` | `throughput.tokens_per_sec × 300 / 1e6` | Within a few percent of measured at steady state. |
| `num_steps` | `(throughput.tokens_per_sec × 300) / throughput.tokens_per_step` | Same logic. |

---

*End of guide.* For parallelizing two autoresearch runs on a single GH200, see the companion [gh200-parallel-training-guide.docx](gh200-parallel-training-guide.docx) in this repo.
