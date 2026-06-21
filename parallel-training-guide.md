# User Guide: Running Parallel Training Jobs on a Single NVIDIA GPU

**MIG and MPS Methods Explained**

> Supported GPUs: A100, H100, H200, GH200, B200
> Target workload: N independent PyTorch training runs on a single GPU
> Also covers: Multi-GPU nodes and Kubernetes clusters

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [1.1 What This Guide Covers](#11-what-this-guide-covers)
   - [1.2 Key Terms You Need to Know](#12-key-terms-you-need-to-know)
   - [1.3 Supported GPUs at a Glance](#13-supported-gpus-at-a-glance)
2. [MIG vs MPS: Which Should You Use?](#2-mig-vs-mps-which-should-you-use)
   - [2.1 Recommendation](#21-recommendation)
3. [Deciding How Many Parallel Jobs to Run](#3-deciding-how-many-parallel-jobs-to-run)
   - [3.1 Resource Budget per Job](#31-resource-budget-per-job)
   - [3.2 Practical Limits](#32-practical-limits)
4. [Method 1: MPS (Multi-Process Service)](#4-method-1-mps-multi-process-service)
   - [4.1 How MPS Works](#41-how-mps-works)
   - [4.2 When to Use MPS](#42-when-to-use-mps)
   - [4.3 Prerequisites](#43-prerequisites)
   - [4.4 Step-by-Step Setup](#44-step-by-step-setup)
   - [4.5 Debugging Common Issues](#45-debugging-common-issues)
5. [Method 2: MIG (Multi-Instance GPU)](#5-method-2-mig-multi-instance-gpu)
   - [5.1 How MIG Works](#51-how-mig-works)
   - [5.2 Understanding MIG Profiles](#52-understanding-mig-profiles)
   - [5.3 Choosing the Right Profile for Your GPU and Split](#53-choosing-the-right-profile-for-your-gpu-and-split)
   - [5.4 When to Use MIG](#54-when-to-use-mig)
   - [5.5 Prerequisites](#55-prerequisites)
   - [5.6 Step-by-Step Setup](#56-step-by-step-setup)
   - [5.7 Using MIG with Docker Containers](#57-using-mig-with-docker-containers)
   - [5.8 Debugging Common Issues](#58-debugging-common-issues)
6. [Multi-GPU and Cluster Setups](#6-multi-gpu-and-cluster-setups)
   - [6.1 Multiple GPUs in a Single Node](#61-multiple-gpus-in-a-single-node)
   - [6.2 Kubernetes Clusters](#62-kubernetes-clusters)
7. [Best Practices (All Methods)](#7-best-practices-all-methods)
   - [7.1 CPU and Memory Pinning (NUMA)](#71-cpu-and-memory-pinning-numa)
   - [7.2 PyTorch Optimisations](#72-pytorch-optimisations)
   - [7.3 Isolate All I/O State](#73-isolate-all-io-state)
   - [7.4 Monitoring](#74-monitoring)
   - [7.5 Using tmux for Session Management](#75-using-tmux-for-session-management)
8. [Quick-Start Cheat Sheet](#8-quick-start-cheat-sheet)
9. [GPU-Specific Notes](#9-gpu-specific-notes)

---

## 1. Introduction

### 1.1 What This Guide Covers

This guide walks you through running **N independent machine learning training jobs at the same time on a single NVIDIA GPU** — where N can be 2, 3, 4, or up to 7 depending on your GPU and method. This is useful when you want to run experiments in parallel — for example, sweeping hyperparameters, testing model variants, or comparing training configurations — without needing additional GPUs.

There are two methods to achieve this, each with different trade-offs:

- **MIG (Multi-Instance GPU):** Physically splits the GPU into up to 7 smaller, completely independent GPUs. Each partition gets its own dedicated memory and compute resources. One job cannot see or affect the other. The number of partitions is constrained to specific hardware profiles.
- **MPS (Multi-Process Service):** Lets any number of jobs share the full GPU at the same time by running their work concurrently. You set soft limits on how much of the GPU each job can use. The number of concurrent jobs is limited only by available memory and practical throughput.

This guide covers single-GPU setups in detail and includes guidance for multi-GPU nodes and Kubernetes clusters.

---

### 1.2 Key Terms You Need to Know

Before diving in, here are the core terms used throughout this guide:

| Term | What It Means |
|---|---|
| **GPU** | Graphics Processing Unit — the chip that does the heavy math for training ML models. |
| **HBM / VRAM** | High Bandwidth Memory — the GPU's dedicated fast memory. This is where your model weights, gradients, and training data batches live during training. Different GPUs have different amounts (see Section 1.3). |
| **SMs (Streaming Multiprocessors)** | The compute engines inside a GPU. Think of them like CPU cores but specialised for parallel math. MIG divides them into fixed groups. |
| **CUDA** | NVIDIA's programming framework that lets software (like PyTorch) use the GPU. When you run training, PyTorch sends work to the GPU via CUDA. |
| **MIG (Multi-Instance GPU)** | A hardware feature that splits one physical GPU into multiple isolated mini-GPUs. Each has its own SMs, memory, and cache — completely independent. |
| **MPS (Multi-Process Service)** | A CUDA feature that lets multiple programs share one GPU at the same time. Without MPS, programs take turns; with MPS, their work runs concurrently. |
| **MIG Profile (e.g. `3g.40gb`)** | A specification for a MIG partition. The first number (e.g. "3g") means 3 GPU Instance slices — roughly 3/7 of the compute. The second (e.g. "40gb") is the dedicated VRAM. Profile names vary by GPU model. |
| **`CUDA_VISIBLE_DEVICES`** | An environment variable that controls which GPU (or MIG partition) a program can see. Setting it restricts a training job to a specific device. |
| **NUMA (Non-Uniform Memory Access)** | Describes how memory access speed varies depending on which CPU socket or chip is accessing which memory region. Relevant for multi-socket servers and superchips like the GH200. |
| **NVLink** | A high-speed interconnect between GPUs (and between CPU and GPU on the GH200/GB200). Much faster than PCIe. |
| **bf16 (bfloat16)** | A 16-bit number format used to speed up training and halve memory usage compared to standard 32-bit floats, with minimal accuracy loss. |

---

### 1.3 Supported GPUs at a Glance

| GPU | Architecture | VRAM | MIG Support | Max MIG Instances | CPU Connection |
|---|---|---|---|---|---|
| **A100 40GB** | Ampere | 40 GB HBM2e | Yes | 7 | PCIe Gen4 / NVLink 3.0 |
| **A100 80GB** | Ampere | 80 GB HBM2e | Yes | 7 | PCIe Gen4 / NVLink 3.0 |
| **H100 80GB** | Hopper | 80 GB HBM3 | Yes | 7 | PCIe Gen5 / NVLink 4.0 |
| **H200** | Hopper | 141 GB HBM3e | Yes | 7 | PCIe Gen5 / NVLink 4.0 |
| **GH200** | Hopper (Grace CPU) | 96 GB HBM3 | Yes | 7 | NVLink-C2C (900 GB/s) |
| **B200** | Blackwell | 180 GB HBM3e | Yes | Up to 7 | PCIe Gen6 / NVLink 5.0 |

> **Note:** All MIG-capable GPUs divide their compute into **7 slices**. The maximum number of MIG instances you can create depends on the profile size you choose — smaller profiles allow more instances.

---

## 2. MIG vs MPS: Which Should You Use?

| | **MIG** | **MPS** |
|---|---|---|
| **What it does** | Physically splits the GPU into N independent mini-GPUs (hardware partitioning) | Lets N processes share the full GPU with software-enforced limits (software partitioning) |
| **Memory isolation** | Hard — each partition has its own dedicated VRAM. A job cannot access or crash another's memory. | Soft — you set a memory cap per process, but it is enforced by the CUDA driver, not by hardware. |
| **Compute isolation** | Hard — each partition has fixed SMs (compute cores). One job cannot steal cycles from another. | Soft — you set a percentage cap on GPU threads per process, but the GPU scheduler is shared. |
| **Fault isolation** | Full — if one job crashes, the other partitions are unaffected. | None — if one job triggers a fatal GPU error, the MPS server can crash all other jobs too. |
| **Throughput** | Each job gets a fixed share. If one job stalls, its SMs sit idle; other jobs cannot borrow them. | Higher total throughput — if one job stalls, others can use idle GPU resources. Better for bursty workloads. |
| **Number of jobs** | Limited to specific MIG profile combinations (typically 2, 3, 4, or 7 equal partitions). | Any number of jobs, limited only by memory and practical throughput. |
| **Setup complexity** | Requires root, GPU reset, `nvidia-smi` MIG commands. Must be done during downtime. | Simple — start a daemon process, set two environment variables per job, launch. |
| **GPU support** | Only MIG-capable GPUs: A100, H100, H200, GH200, B200 | Any CUDA GPU with Volta architecture or newer (including non-MIG GPUs) |
| **Best for** | Multi-tenant / untrusted code, strict QoS guarantees, preventing OOM interference between jobs. | Experiment parallelism with trusted code, maximising total GPU utilisation. |

### 2.1 Recommendation

For running parallel experiments with your own trusted code, **MPS is the recommended default.** It is simpler to set up, gives higher aggregate throughput, scales to any number of concurrent jobs, and works on every modern NVIDIA GPU. Switch to MIG if you need guaranteed resource isolation, fault isolation, or reproducible contention-free benchmarking.

---

## 3. Deciding How Many Parallel Jobs to Run

Before setting up MIG or MPS, you need to decide how many concurrent jobs (N) your GPU can support. This depends on the memory each job needs and the minimum compute each job requires to train at a reasonable speed.

### 3.1 Resource Budget per Job

Use this formula to estimate the maximum number of jobs:

```
N = floor( (GPU_VRAM × 0.85) / VRAM_per_job )
```

The `0.85` factor reserves ~15% of VRAM for CUDA context overhead, kernel caches, and memory fragmentation. Each concurrent CUDA context (each job) consumes 200–500 MB of overhead.

**Estimating VRAM per job** (PyTorch, bf16 mixed precision, Adam optimiser):

| Model Size | Approximate VRAM per Job (bf16 + Adam) |
|---|---|
| 50M parameters | ~1.5 GB |
| 100M parameters | ~2.5 GB |
| 200M parameters | ~4 GB |
| 400M parameters | ~7 GB |
| 1B parameters | ~16 GB |
| 3B parameters | ~45 GB |

> **Example:** Training 4× 200M models on an H100 80GB: each needs ~4 GB, so total is ~16 GB. The H100 has 80 GB, leaving massive headroom. The bottleneck here is compute (SMs), not memory — each job only gets 1/4 of the GPU's compute, which may slow training but is fine for hyperparameter sweeps.

### 3.2 Practical Limits

| Method | Practical Max N | Why |
|---|---|---|
| **MPS** | ~8–10 | Each CUDA context adds overhead. Beyond ~8 concurrent processes, context-switching overhead and CPU contention usually outweigh the benefits. |
| **MIG** | 7 | Hard hardware limit: all MIG-capable GPUs have 7 compute slices. The smallest profile (`1g`) uses one slice, so 7 is the theoretical max. In practice, 2–4 equal partitions are most common. |

> **Tip:** Start with 2–3 parallel jobs and measure throughput. If each job's GPU utilisation is low (under 40%), you can likely add more. If jobs are fighting for memory or compute, reduce N.

---

## 4. Method 1: MPS (Multi-Process Service)

### 4.1 How MPS Works

Normally, when multiple programs try to use the same GPU, CUDA gives them turns — one runs a batch of work, then pauses while the next runs its batch (this is called time-slicing). This wastes time because the GPU sits partially idle during each switch.

MPS changes this by acting as a middleman (a background daemon). All N programs submit their GPU work to the MPS daemon, which funnels it all to the GPU at the same time. The GPU can then run computations from all programs concurrently on different SMs.

You control how the GPU is shared by setting two environment variables before launching **each** job:

- **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** — limits what fraction of the GPU's compute cores (SMs) each job can use. For N equal jobs, set this to `floor(100 / N)`. For example: 50 for 2 jobs, 33 for 3 jobs, 25 for 4 jobs.
- **`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`** — caps how much GPU memory (VRAM) each job can allocate. For N equal jobs, divide the usable VRAM (total × 0.85) by N.

**Recommended memory limits per job** (for N equal jobs):

| GPU | Total VRAM | N=2 | N=3 | N=4 | N=7 |
|---|---|---|---|---|---|
| A100 40GB | 40 GB | 16G | 11G | 8G | 4G |
| A100 80GB | 80 GB | 35G | 22G | 17G | 9G |
| H100 80GB | 80 GB | 35G | 22G | 17G | 9G |
| H200 | 141 GB | 60G | 40G | 30G | 17G |
| GH200 | 96 GB | 40G | 27G | 20G | 11G |
| B200 | 180 GB | 80G | 51G | 38G | 21G |

> **Tip:** These are conservative starting points. If your models are small (under 200M parameters), you can often fit more jobs than the memory limit suggests — compute becomes the bottleneck, not memory.

---

### 4.2 When to Use MPS

- You are running your own trusted experiment code (not untrusted third-party code).
- You want maximum total throughput — if one job stalls on data loading, the others automatically use more of the GPU.
- You want a setup that takes minutes, not a GPU reset.
- You want to run more concurrent jobs than MIG supports (e.g. 5 or 6 jobs, which MIG cannot split evenly).
- Your GPU does not support MIG, or you want to avoid MIG's setup overhead.

---

### 4.3 Prerequisites

- NVIDIA driver installed (R535+ recommended; R570+ ideal). Verify with: `nvidia-smi`
- CUDA toolkit installed (12.4+). Verify with: `nvcc --version`
- PyTorch with CUDA support. Install with:
  ```bash
  # For x86_64 systems (A100, H100, B200 in standard servers)
  pip install torch --index-url https://download.pytorch.org/whl/cu128

  # For ARM64 / aarch64 systems (GH200)
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```
- Confirm GPU is visible:
  ```bash
  python -c "import torch; print(torch.cuda.get_device_name(0))"
  ```

> **Tip:** If you are using an NGC PyTorch container (e.g. `nvcr.io/nvidia/pytorch:24.05-py3`), all prerequisites are already met.

---

### 4.4 Step-by-Step Setup

This example shows how to launch **N** parallel jobs. Replace `<N>` with your desired number of concurrent jobs throughout.

#### Step 1: Identify Your GPU and Calculate Limits

Run `nvidia-smi` to confirm which GPU you have and how much VRAM is available:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Example output: NVIDIA A100-SXM4-80GB, 81920 MiB
```

Then calculate your per-job settings:

```bash
# Thread percentage: divide 100 by N
THREAD_PCT=$((100 / N))   # e.g. 100/3 = 33 for 3 jobs

# Memory limit: (total_vram_GB × 0.85) / N, rounded down
MEM_LIMIT="<value>G"      # Use the table in Section 4.1, or calculate manually
```

#### Step 2: Verify MIG is Disabled

MPS and MIG cannot be active at the same time. Check that MIG is off:

```bash
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader
```

Expected output: `Disabled` (or `[N/A]` on GPUs that don't support MIG). If it shows `Enabled`, disable it first:

```bash
sudo nvidia-smi -mig 0
sudo nvidia-smi --gpu-reset   # or reboot
```

#### Step 3: Start the MPS Daemon

The MPS daemon is a background process that manages shared GPU access. Start it once; it stays running until you stop it.

```bash
# Create a directory for MPS communication files
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Start the daemon (runs in background)
nvidia-cuda-mps-control -d

# Verify it is running
ps aux | grep mps
```

You should see a `nvidia-cuda-mps-control` process and a `nvidia-cuda-mps-server` process.

#### Step 4: Launch Each Training Job

Open a separate terminal (or tmux pane) for each job. Each job gets the same environment variables but different config and output paths.

**Job 1:**
```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<THREAD_PCT>
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=<MEM_LIMIT>"

python train.py \
    --config experiment_1.yaml \
    --output-dir /data/runs/experiment_1 \
    --wandb-run-name exp_1
```

**Job 2:**
```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<THREAD_PCT>
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=<MEM_LIMIT>"

python train.py \
    --config experiment_2.yaml \
    --output-dir /data/runs/experiment_2 \
    --wandb-run-name exp_2
```

**Job 3, 4, ... N:** Repeat the same pattern with different config files and output directories.

> **Automating N jobs with a loop:**
> ```bash
> N=4           # number of parallel jobs
> THREAD_PCT=$((100 / N))
> MEM_LIMIT="17G"  # example for H100 with N=4
>
> for i in $(seq 1 $N); do
>   CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
>   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$THREAD_PCT \
>   CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=${MEM_LIMIT}" \
>   python train.py \
>       --config experiment_${i}.yaml \
>       --output-dir /data/runs/experiment_${i} \
>       --wandb-run-name exp_${i} &
> done
> wait  # Wait for all jobs to finish
> ```

> **Note on CPU pinning:** If your system has enough CPU cores, pin each job to a separate set of cores for best performance. See Section 7.1 for details.

#### Step 5: Validate That All Jobs Are Running

Run `nvidia-smi`. You should see N python processes listed under the GPU, each marked as type `M+C` (MPS client):

```
nvidia-smi

# Look for N processes:
# | Processes:                                                       |
# |  GPU   Type   PID   Process name                  GPU Memory    |
# |    0   M+C   1234   python train.py ...              XXXX MiB   |
# |    0   M+C   2345   python train.py ...              XXXX MiB   |
# |    0   M+C   3456   python train.py ...              XXXX MiB   |
# |   ...                                                            |
```

The `M+C` type confirms they are running through MPS (not time-slicing).

#### Step 6: Stop MPS When Finished

After all training runs complete:

```bash
echo "quit" | nvidia-cuda-mps-control
```

---

### 4.5 Debugging Common Issues

| Problem | Solution |
|---|---|
| `"CUDA error: out of memory"` on job N | Earlier jobs consumed too much VRAM. Ensure all jobs have `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` set before launch. Also add `torch.cuda.set_per_process_memory_fraction(0.85 / N)` in your training script as a guardrail. |
| MPS daemon won't start | Check that MIG is disabled (`nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv`). Check that no other MPS daemon is running. Check log files in `$CUDA_MPS_LOG_DIRECTORY`. |
| Processes show as type `C` (not `M+C`) | The process is not connecting to MPS. Ensure `CUDA_MPS_PIPE_DIRECTORY` is exported and points to the same directory the daemon is using. |
| DataLoader crashes with `"unable to open shared memory"` | Add `--ipc=host` if running in Docker, or increase shared memory: `--shm-size=16g`. Outside Docker, this is usually not an issue. |
| All jobs are slow / low GPU utilisation | Too many concurrent jobs. Reduce N. Also check CPU contention with `top` — with N jobs, each should have its own set of CPU cores (see Section 7.1). |
| One job crashes and others freeze | This is a known MPS limitation: no fault isolation. Restart the MPS daemon and relaunch all jobs. If this happens frequently, switch to MIG. |
| Jobs have unequal performance | If one job has higher GPU utilisation than others, it may be consuming more than its thread share. Verify that `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is set in every job's environment, not just some. |

---

## 5. Method 2: MIG (Multi-Instance GPU)

### 5.1 How MIG Works

MIG physically splits a single GPU into multiple smaller, fully independent GPUs. Each partition (called a GPU Instance) gets its own dedicated compute cores (SMs), its own memory, its own cache, and its own memory controllers. From the perspective of your training script, each MIG partition looks and behaves exactly like a smaller standalone GPU.

The key difference from MPS is that this is a hardware-level split. One partition literally cannot access another's memory, even if the software tries. If one partition's job crashes, the others continue running unaffected.

---

### 5.2 Understanding MIG Profiles

When you split the GPU, you choose a profile that defines how big each partition is. The profile name tells you its resources using the format `<compute_slices>g.<memory>gb`:

- The first number (e.g. `3g`) is the number of GPU Instance slices — each slice is roughly 1/7 of the total compute (SMs).
- The second number (e.g. `40gb`) is the dedicated VRAM for that partition.

All MIG-capable GPUs divide their compute into **7 slices**. The profiles you choose must fit within these 7 slices. For example, on an H100 80GB you could create:

- 2× `3g.40gb` = 6 slices used, 1 unused (two large partitions)
- 1× `3g.40gb` + 2× `2g.20gb` = 7 slices used (one large + two small)
- 3× `2g.20gb` = 6 slices used, 1 unused (three medium partitions)
- 7× `1g.10gb` = 7 slices used (seven small partitions)

You can discover the available profiles on your specific GPU by running:

```bash
nvidia-smi mig -lgip
```

---

### 5.3 Choosing the Right Profile for Your GPU and Split

#### Equal-Partition Profiles (N identical jobs)

The following table shows recommended profiles for splitting each GPU into **N equal partitions**:

| GPU | N=2 | N=3 | N=4 | N=7 |
|---|---|---|---|---|
| **A100 40GB** | `3g.20gb` × 2 (ID 9) | `2g.10gb` × 3 (ID 5) | `1g.5gb` × 4 (ID 14) | `1g.5gb` × 7 (ID 14) |
| **A100 80GB** | `3g.40gb` × 2 (ID 9) | `2g.20gb` × 3 (ID 5) | `1g.10gb` × 4 (ID 14) | `1g.10gb` × 7 (ID 14) |
| **H100 80GB** | `3g.40gb` × 2 (ID 9) | `2g.20gb` × 3 (ID 5) | `1g.10gb` × 4 (ID 14) | `1g.10gb` × 7 (ID 14) |
| **H200 141GB** | `3g.70gb` × 2 (ID 9) | `2g.35gb` × 3 (ID 5) | `1g.18gb` × 4 (ID 14) | `1g.18gb` × 7 (ID 14) |
| **GH200 96GB** | `3g.48gb` × 2 (ID 9) | `2g.24gb` × 3 (ID 5) | `1g.12gb` × 4 (ID 14) | `1g.12gb` × 7 (ID 14) |
| **B200 180GB** | `2g.45gb` × 2 (ID 5) | See note | `1g.23gb` × 4 (ID 14) | `1g.23gb` × 7 (ID 14) |

> **Notes:**
> - **N=2 uses `3g` profiles** on most GPUs (6 of 7 slices used, 1 unused). On the B200, the largest equal two-way split is `2g.45gb` × 2.
> - **N=3 uses `2g` profiles** (6 of 7 slices used, 1 unused). On the B200, `2g.45gb` × 3 may not fit; use `1g.23gb` × 3 instead.
> - **N=4 and N=7 use `1g` profiles**, giving each job the minimum 1/7 of compute and the smallest memory allocation.
> - **N=5 and N=6 are not possible** with equal-sized MIG partitions (5 and 6 don't divide evenly into the 7-slice grid using identical profiles). Use MPS instead, or create asymmetric MIG partitions.
> - **Profile IDs may vary** by GPU model and driver version. Always verify with `nvidia-smi mig -lgip`.

#### Per-Partition Resources Summary

| Profile Size | Compute (% of GPU SMs) | Example VRAM (H100 80GB) |
|---|---|---|
| `1g` | ~14% (1/7) | 10 GB |
| `2g` | ~29% (2/7) | 20 GB |
| `3g` | ~43% (3/7) | 40 GB |
| `4g` | ~57% (4/7) | 40 GB |
| `7g` | 100% (7/7) | 80 GB |

#### Asymmetric Splits

MIG also supports mixing different profile sizes. This is useful when you have one primary training run and one or more smaller experiments:

```bash
# Example: 1 large job + 2 small jobs on H100
sudo nvidia-smi mig -cgi 9,14,14 -C
# Creates: 1× 3g.40gb + 2× 1g.10gb (total 5 of 7 slices used)

# Example: 1 large + 1 medium + 1 small on A100 80GB
sudo nvidia-smi mig -cgi 9,5,14 -C
# Creates: 1× 3g.40gb + 1× 2g.20gb + 1× 1g.10gb (6 of 7 slices used)
```

> **Tip:** Not all profile combinations are valid — some combinations conflict due to the GPU's physical memory controller layout. If a combination fails, run `nvidia-smi mig -lgip` to see available profiles and their maximum instance counts given what is already created.

---

### 5.4 When to Use MIG

- You need guaranteed memory isolation — one job absolutely must not be able to OOM-kill another.
- You need fault isolation — if one experiment's code has a bug that crashes the GPU, the others must survive.
- You are running untrusted or third-party training code.
- You want reproducible, contention-free benchmarking (each job gets fixed resources, no variability from sharing).

---

### 5.5 Prerequisites

- NVIDIA driver R535+ (R570+ recommended for improved MIG enumeration).
- Root / sudo access (MIG commands require elevated privileges).
- No running GPU processes (MIG enable requires a GPU reset).
- CUDA 12.4+ and PyTorch with CUDA support (same as MPS prerequisites).
- Stop any GPU monitoring daemons before enabling MIG:
  ```bash
  sudo systemctl stop dcgm nvsm
  ```
- **B200 / GB200 only:** Ensure NVIDIA Fabric Manager is running (`sudo systemctl status nvidia-fabricmanager`). Fabric Manager manages NVLink/NVSwitch and must be active for MIG operations on Blackwell GPUs.

---

### 5.6 Step-by-Step Setup

This example shows how to create **N** MIG partitions and launch a job on each.

#### Step 1: Identify Your GPU and Choose Profiles

Confirm your GPU model:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Then look up the recommended profile for your GPU and desired N in the table in Section 5.3.

#### Step 2: Enable MIG Mode

This flips a hardware switch on the GPU. All running GPU work will be terminated.

```bash
# Enable MIG on GPU 0
sudo nvidia-smi -mig 1

# If output says "pending", reset the GPU:
sudo nvidia-smi --gpu-reset
# OR reboot the machine if the reset fails (common if a display is attached)

# Verify MIG is now enabled
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader
# Expected output: Enabled
```

#### Step 3: Verify Available Profiles

List the profiles your GPU supports:

```bash
nvidia-smi mig -lgip
```

Confirm that the profile you want to use is available and note its **Profile ID**.

#### Step 4: Create GPU Instances

Pass the profile ID once for each partition you want. The `-C` flag automatically creates a Compute Instance inside each GPU Instance (required for running CUDA programs).

```bash
# For N equal partitions, repeat the profile ID N times:

# Example: 2 partitions using profile ID 9
sudo nvidia-smi mig -cgi 9,9 -C

# Example: 3 partitions using profile ID 5
sudo nvidia-smi mig -cgi 5,5,5 -C

# Example: 4 partitions using profile ID 14
sudo nvidia-smi mig -cgi 14,14,14,14 -C

# Example: 7 partitions using profile ID 14
sudo nvidia-smi mig -cgi 14,14,14,14,14,14,14 -C
```

#### Step 5: List the MIG Device UUIDs

Each MIG partition gets a unique identifier (UUID). You will use these UUIDs to assign each training job to a specific partition.

```bash
nvidia-smi -L

# Example output (3-way split on H100):
# GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdef12-...)
#   MIG 2g.20gb Device 0: (UUID: MIG-11111111-...)
#   MIG 2g.20gb Device 1: (UUID: MIG-22222222-...)
#   MIG 2g.20gb Device 2: (UUID: MIG-33333333-...)
```

Copy all N MIG UUIDs (the lines starting with `MIG-`).

#### Step 6: Verify the Partitions

Confirm that all partitions are healthy:

```bash
nvidia-smi mig -lgi
# Should show N GPU instances with the expected memory

nvidia-smi mig -lci
# Should show one compute instance inside each GPU instance
```

#### Step 7: Launch Training Jobs

Set `CUDA_VISIBLE_DEVICES` to the UUID of a different MIG partition for each job. Each training script will see its partition as a single GPU (`cuda:0`).

**Job 1:**
```bash
export CUDA_VISIBLE_DEVICES=MIG-11111111-...
python train.py --config experiment_1.yaml --output-dir /data/runs/experiment_1
```

**Job 2:**
```bash
export CUDA_VISIBLE_DEVICES=MIG-22222222-...
python train.py --config experiment_2.yaml --output-dir /data/runs/experiment_2
```

**Job 3 ... N:** Repeat with the remaining MIG UUIDs.

> **Automating N jobs with a loop:**
> ```bash
> # Collect MIG UUIDs into an array
> UUIDS=($(nvidia-smi -L | grep "MIG" | grep -oP 'UUID: \K[^ )]+'))
>
> for i in "${!UUIDS[@]}"; do
>   JOB_NUM=$((i + 1))
>   CUDA_VISIBLE_DEVICES=${UUIDS[$i]} \
>   python train.py \
>       --config experiment_${JOB_NUM}.yaml \
>       --output-dir /data/runs/experiment_${JOB_NUM} &
> done
> wait
> ```

#### Step 8: Validate That All Jobs Are Running

```bash
nvidia-smi

# Each MIG device should show its own process
```

You can also verify from inside each training script:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should include the MIG profile name, e.g.: NVIDIA H100 80GB HBM3 MIG 2g.20gb
```

#### Step 9: Clean Up (Disable MIG When Done)

After all jobs complete, tear down the MIG partitions and return the GPU to normal mode:

```bash
# Destroy compute instances, then GPU instances
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi

# Disable MIG mode
sudo nvidia-smi -mig 0
sudo nvidia-smi --gpu-reset

# Verify
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader
# Expected output: Disabled
```

> **B200 / GB200 only:** After disabling MIG, verify that Fabric Manager is running to restore NVLink peer-to-peer connectivity: `sudo systemctl status nvidia-fabricmanager` (start it if needed: `sudo systemctl start nvidia-fabricmanager`).

---

### 5.7 Using MIG with Docker Containers

Pin each container to a specific MIG device. The `0:X` syntax means GPU 0, MIG device X:

```bash
# Job 1
docker run --gpus '"device=0:0"' --ipc=host --shm-size=16g \
    -v /data:/data my-training-image:latest \
    python train.py --config experiment_1.yaml

# Job 2
docker run --gpus '"device=0:1"' --ipc=host --shm-size=16g \
    -v /data:/data my-training-image:latest \
    python train.py --config experiment_2.yaml

# Job 3 ... N: increment the MIG device index (0:2, 0:3, ...)
```

The `--ipc=host` and `--shm-size` flags prevent PyTorch DataLoader shared memory errors.

---

### 5.8 Debugging Common Issues

| Problem | Solution |
|---|---|
| `"Unable to determine the device handle for GPU…"` | You set `CUDA_VISIBLE_DEVICES` to a UUID that does not exist. Re-run `nvidia-smi -L` and copy the exact MIG UUID. |
| `"nvidia-smi -mig 1"` says `"GPU not idle"` | Something is using the GPU. Run `sudo fuser -v /dev/nvidia*` to find processes, kill them, then retry. Also stop `dcgm`, `nvsm`, or any monitoring daemons. |
| `"Unable to create GPU instance"` or `"insufficient resources"` | You already have MIG instances consuming the GPU's capacity, or the requested combination of profiles does not fit. Run `sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi` to destroy existing instances, then try again. Use `nvidia-smi mig -lgip` to see which profiles are still available. |
| Training is much slower than expected | **GH200 only:** MIG disables the NVLink-C2C fast path, reducing CPU↔GPU copy bandwidth by ~8×. On all GPUs: with small MIG partitions (1g or 2g), each job has limited SMs — training will be slower per job by design. |
| `nvidia-smi` shows 0% GPU utilization but training is running | `nvidia-smi` does not report per-MIG-instance SM utilization. Use DCGM instead: `dcgmi dmon -e 1001,1002 -d 1000` to get per-instance metrics. |
| `"CUDA error: invalid device ordinal"` | Under MIG, each process should see only one device (`cuda:0`). If your training script tries to use `cuda:1` or references multiple GPUs, change it to use only `cuda:0`. |
| **B200:** `"MIG operation failed"` after disabling MIG | Ensure Fabric Manager is running: `sudo systemctl start nvidia-fabricmanager`. Blackwell GPUs require Fabric Manager for NVLink/NVSwitch re-initialisation. |
| **Profile combination rejected** | Not all MIG profile combinations are valid. Profiles must fit within the GPU's physical memory controller layout. Try a different combination, or use equal-sized profiles from the table in Section 5.3. |

---

## 6. Multi-GPU and Cluster Setups

### 6.1 Multiple GPUs in a Single Node

If your node has multiple GPUs (e.g. DGX A100 with 8× A100s, or an HGX H100 with 8× H100s), you can apply MIG or MPS to **each GPU independently**. This multiplies your total parallel capacity: 8 GPUs × N jobs per GPU = 8N concurrent experiments.

#### MIG on Multiple GPUs

Enable MIG and create partitions on each GPU by specifying the GPU index:

```bash
# Enable MIG on all GPUs
sudo nvidia-smi -mig 1
sudo nvidia-smi --gpu-reset

# Create partitions on each GPU (example: 3-way split)
for gpu in 0 1 2 3 4 5 6 7; do
  sudo nvidia-smi mig -i $gpu -cgi 5,5,5 -C
done

# List all MIG devices across all GPUs
nvidia-smi -L
```

Each MIG device gets its own UUID. Assign jobs to specific MIG UUIDs just as in the single-GPU case.

#### MPS on Multiple GPUs

Start a separate MPS daemon per GPU using different pipe directories:

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  mkdir -p /tmp/mps-gpu${gpu}
  CUDA_VISIBLE_DEVICES=$gpu CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-gpu${gpu} \
    nvidia-cuda-mps-control -d
done

# Then for each job, set CUDA_VISIBLE_DEVICES to the target GPU
# and CUDA_MPS_PIPE_DIRECTORY to that GPU's MPS directory
CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-gpu0 \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python train.py --config a.yaml &

CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-gpu0 \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python train.py --config b.yaml &

# ...repeat for GPUs 1-7
```

---

### 6.2 Kubernetes Clusters

For managed clusters, MIG is typically managed by the **NVIDIA GPU Operator** and its MIG Manager. This automates partitioning across all nodes.

#### Overview of the Kubernetes MIG Workflow

1. **Install the NVIDIA GPU Operator** with MIG support enabled:
   ```bash
   helm upgrade gpu-operator nvidia/gpu-operator \
     -n gpu-operator \
     --reuse-values \
     --set mig.strategy=mixed
   ```

2. **Create a MIG ConfigMap** defining your desired partition layout:
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: mig-config
     namespace: gpu-operator
   data:
     config.yaml: |
       version: v1
       mig-configs:
         all-disabled:
           - devices: all
             mig-enabled: false
         seven-way:
           - devices: all
             mig-enabled: true
             mig-devices:
               "1g.10gb": 7    # Adjust profile name for your GPU
         three-way:
           - devices: all
             mig-enabled: true
             mig-devices:
               "2g.20gb": 3
         two-way:
           - devices: all
             mig-enabled: true
             mig-devices:
               "3g.40gb": 2
   ```

3. **Label GPU nodes** with the desired MIG config:
   ```bash
   kubectl label node <GPU-NODE> nvidia.com/mig-config=three-way
   ```

4. **Deploy training pods** that request MIG devices:
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 1
   nodeSelector:
     nvidia.com/mig-partition-2g.20gb: "true"
   ```

The GPU Operator handles enabling MIG mode, creating instances, and advertising them to the Kubernetes scheduler — no manual `nvidia-smi` commands required on each node.

> **Note:** Kubernetes MIG setup varies by cloud provider (GKE, EKS, on-prem). Refer to the [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) for full instructions specific to your environment.

---

## 7. Best Practices (All Methods)

### 7.1 CPU and Memory Pinning (NUMA)

For best performance, split your CPU cores evenly between training jobs so they do not compete for the same CPU caches and memory channels.

First, check your system's core count and NUMA topology:

```bash
nproc               # total number of CPU cores
numactl -H          # NUMA node layout
nvidia-smi topo -m  # GPU-to-NUMA affinity
```

Then calculate the cores per job and assign ranges:

```bash
TOTAL_CORES=$(nproc)
CORES_PER_JOB=$((TOTAL_CORES / N))

# Job 1: cores 0 to (CORES_PER_JOB - 1)
numactl --physcpubind=0-$((CORES_PER_JOB - 1)) python train.py --config 1.yaml

# Job 2: cores CORES_PER_JOB to (2 * CORES_PER_JOB - 1)
numactl --physcpubind=${CORES_PER_JOB}-$((2 * CORES_PER_JOB - 1)) python train.py --config 2.yaml

# ...and so on for jobs 3 to N
```

> **Tip:** On multi-socket systems, pin each job to the NUMA node closest to the GPU it's using. Use `nvidia-smi topo -m` to check GPU-to-NUMA affinity. On the GH200, the Grace CPU is typically NUMA node 0.

---

### 7.2 PyTorch Optimisations

**Use bf16 mixed precision** to halve memory usage and improve training speed:

```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

**Cap DataLoader workers.** With N concurrent jobs, each job should use at most `total_cores / N / 2` workers to leave headroom:

```python
import os
num_workers = min(16, os.cpu_count() // (N * 2))  # Replace N with your job count
train_loader = DataLoader(dataset, batch_size=64, num_workers=num_workers, pin_memory=True)
```

**Add a memory guardrail** at the top of your training script (especially useful with MPS):

```python
torch.cuda.set_per_process_memory_fraction(0.85 / N)  # N = number of concurrent jobs
```

**Enable gradient checkpointing** if memory is tight (trades compute time for memory savings):

```python
from torch.utils.checkpoint import checkpoint
# Wrap memory-heavy layers: output = checkpoint(layer, input, use_reentrant=False)
```

---

### 7.3 Isolate All I/O State

Concurrent training runs must not write to the same directories. Separate everything:

| Resource | How to Isolate |
|---|---|
| Checkpoints | Separate `--output-dir` for each job |
| Dataset cache | Set `HF_HOME` and `HF_DATASETS_CACHE` to different paths per job |
| TensorBoard | Use different `--logdir` paths |
| W&B / MLflow | Use separate run names, groups, or experiment IDs |
| Ports | If jobs serve dashboards (e.g. TensorBoard), assign a different port per job (6006, 6007, 6008, ...) |

---

### 7.4 Monitoring

**`nvidia-smi dmon`** — simple GPU utilisation polling (works for both methods, but does not break down per-MIG instance):

```bash
nvidia-smi dmon -s u -d 2   # GPU/memory utilisation every 2 seconds
```

**nvitop** — interactive, colourised GPU monitor that supports MIG devices:

```bash
pip install nvitop && nvitop
```

**DCGM (Data Center GPU Manager)** — the only way to get proper per-MIG-instance SM utilisation metrics:

```bash
dcgmi dmon -e 1001,1002 -d 1000   # GPU util + mem util every second
```

---

### 7.5 Using tmux for Session Management

tmux lets you run multiple jobs in a persistent session that survives SSH disconnections:

```bash
# Create a session
tmux new-session -s training

# Split into N panes (Ctrl-b then % for vertical, " for horizontal)
# Launch one job per pane

# Detach (safe to close SSH): Ctrl-b then d
# Re-attach later: tmux attach -t training
```

> **Tip for many jobs:** Use a loop-based launcher (as shown in Steps 4 of each method) with `&` to background all jobs, rather than managing N separate tmux panes.

---

## 8. Quick-Start Cheat Sheet

### MPS Quick Start

```bash
# 1. Check your GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Set your parameters
N=<number_of_jobs>
THREAD_PCT=$((100 / N))
MEM_LIMIT="<see table in Section 4.1>"

# 3. Start MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
mkdir -p $CUDA_MPS_PIPE_DIRECTORY && nvidia-cuda-mps-control -d

# 4. Launch N jobs
for i in $(seq 1 $N); do
  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$THREAD_PCT \
  CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=${MEM_LIMIT}" \
  python train.py \
      --config experiment_${i}.yaml \
      --output-dir /data/runs/experiment_${i} &
done
wait

# 5. Cleanup
echo "quit" | nvidia-cuda-mps-control
```

---

### MIG Quick Start

```bash
# 1. Set your parameters
N=<number_of_jobs>
PROFILE_ID=<see table in Section 5.3>

# 2. Enable MIG and create N partitions
sudo nvidia-smi -mig 1 && sudo nvidia-smi --gpu-reset
PROFILES=$(printf "${PROFILE_ID}%.0s," $(seq 1 $N) | sed 's/,$//')
sudo nvidia-smi mig -cgi $PROFILES -C

# 3. Get MIG UUIDs and launch N jobs
UUIDS=($(nvidia-smi -L | grep "MIG" | grep -oP 'UUID: \K[^ )]+'))
for i in "${!UUIDS[@]}"; do
  CUDA_VISIBLE_DEVICES=${UUIDS[$i]} \
  python train.py \
      --config experiment_$((i+1)).yaml \
      --output-dir /data/runs/experiment_$((i+1)) &
done
wait

# 4. Cleanup
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
sudo nvidia-smi -mig 0 && sudo nvidia-smi --gpu-reset
```

---

## 9. GPU-Specific Notes

### A100

- MIG is well-supported and mature on A100. Profile ID 9 = `3g.20gb` (40GB) or `3g.40gb` (80GB) for two-way splits; Profile ID 14 = `1g.5gb`/`1g.10gb` for seven-way splits.
- A100 does **not** have NVLink-C2C; the CPU↔GPU link is PCIe Gen4. MIG does not affect CPU↔GPU bandwidth on A100.
- Older drivers (R450+) support MIG on A100, but R535+ is recommended for stability.

### H100

- MIG profiles and behaviour are very similar to the A100 80GB. Same profile IDs.
- H100 has higher SM performance and HBM3 bandwidth, so each MIG partition is significantly faster than an equivalent A100 partition.

### H200

- Same Hopper architecture as H100 but with 141 GB HBM3e. Use `3g.70gb` × 2, `2g.35gb` × 3, or `1g.18gb` × 7 depending on your split.
- The large VRAM makes MIG particularly useful — even the smallest `1g.18gb` partition has more memory than a `1g.5gb` on A100.

### GH200

- **Critical: MIG disables NVLink-C2C unified memory.** The NVLink-C2C link between the Grace CPU and Hopper GPU provides 900 GB/s bandwidth and supports unified memory. Enabling MIG breaks this path and reduces CPU↔GPU copy bandwidth by approximately 8×. For small models that fit entirely in GPU memory, this is unlikely to matter. For workloads that depend on CPU↔GPU data movement, prefer MPS.
- The Grace CPU is ARM-based (aarch64). Ensure your PyTorch build, CUDA toolkit, and any pip packages are compiled for ARM64.
- The 72-core Grace CPU divides cleanly for most job counts: 36 cores per job for N=2, 24 for N=3, 18 for N=4, 10 for N=7.

### B200

- B200 uses the Blackwell architecture with a different MIG profile structure. The largest equal two-way split is `2g.45gb` × 2, which uses only ~29% of SMs per partition (compared to ~43% on other GPUs). For seven-way splits, use `1g.23gb` × 7.
- A `4g.90gb` profile is available for asymmetric splits (e.g. one large job + smaller ones).
- **Fabric Manager is required.** On B200 and GB200 systems, the NVIDIA Fabric Manager service must be running for MIG operations and for restoring NVLink/NVSwitch connectivity after disabling MIG.
- B200 has 192 GB HBM3e (180 GB usable), so even small MIG partitions (`1g.23gb`) provide ample memory for most training workloads under 1B parameters.

---

*End of guide. For the latest MIG documentation, see: [docs.nvidia.com/datacenter/tesla/mig-user-guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide)*
