"""
LLM hardware usage calculator -- single-GPU pretraining.

Estimates peak VRAM, host RAM/vCPU recommendations, parameter count, and
training throughput ceiling from a transformer architecture spec.

Companion to llm-hardware-estimation-guide.docx -- see that guide for the
derivations and how to use the numbers when designing experiments.

Usage:
    # Inspect the current autoresearch train.py recipe
    python vram_calculator.py --preset autoresearch

    # Explore a hypothetical wider model
    python vram_calculator.py --depth 12 --aspect-ratio 96 --device-batch-size 64

    # JSON output for scripting
    python vram_calculator.py --preset autoresearch --json
"""

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# GPU presets -- peak BF16 FLOPS and memory for common single-GPU targets.
# Add new entries as needed; numbers from NVIDIA datasheets.
# ---------------------------------------------------------------------------

GPU_PRESETS = {
    # name: (peak BF16 TFLOPS dense, VRAM GB)
    "h100-sxm": (989.5, 80),
    "h100-pcie": (756.0, 80),
    "h200": (989.5, 141),
    "gh200-96": (989.5, 96),       # GH200 480GB variant has 96 GB HBM3
    "gh200-144": (989.5, 144),     # GH200 NVL variant has 144 GB HBM3e
    "a100-80": (312.0, 80),
    "a100-40": (312.0, 40),
    "l40s": (362.0, 48),
    "rtx-4090": (165.0, 24),
}

DTYPE_BYTES = {
    "fp32": 4,
    "tf32": 4,    # accumulator type; storage = fp32
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
}


# ---------------------------------------------------------------------------
# Architecture spec
# ---------------------------------------------------------------------------

@dataclass
class ArchSpec:
    depth: int                  # number of transformer blocks
    n_embd: int                 # model hidden dimension
    n_head: int                 # number of query heads
    n_kv_head: int              # number of KV heads (n_head for MHA, < for GQA)
    head_dim: int               # dimension per head
    seq_len: int                # context length
    vocab_size: int             # tokenizer vocab size
    mlp_mult: float = 4.0       # MLP hidden dim = mlp_mult * n_embd
    tied_embeddings: bool = False
    # Optional extras drawn from autoresearch's train.py -- set to 0 to ignore.
    value_embed_layers: int = 0     # number of layers with full value embedding
    value_embed_dim: Optional[int] = None  # kv_dim; defaults to n_kv_head * head_dim


@dataclass
class TrainSpec:
    device_batch_size: int      # micro-batch per forward/backward
    grad_accum_steps: int = 1   # number of micro-batches per optimizer step
    # Precision recipe
    param_dtype: str = "bf16"   # storage dtype for model params
    grad_dtype: str = "bf16"    # storage dtype for gradients
    optim_state_dtype: str = "bf16"  # zeros_like(p) matches param dtype in this repo
    master_weights: bool = False     # keep an fp32 copy of params (classic AMP)
    # Optimizer recipe -- affects state byte cost per parameter
    optimizer: str = "muon_adamw"    # "adamw" | "muon" | "muon_adamw" | "sgd" | "lion"
    activation_checkpointing: bool = False
    flash_attention: bool = True     # FA3/FA2 makes attn activations O(s*h), not O(s^2*h)
    # Activation memory tuning -- bytes held per layer per (batch * seq * hidden) unit.
    # Breakdown for a typical FA + bf16 transformer block (saved for backward):
    #   residual stream (2) + qkv outputs (~6) + norm outputs (2) + attn output (2)
    #   + MLP c_fc output (8 = 4*mlp_mult) + activation^2 (8) + projections (2-4)
    # ~= 30-40 bytes for codepaths with value embeddings / torch.compile.
    # Default 40 is calibrated against the autoresearch depth=8 b=128 s=2048 baseline
    # (observed peak ~45 GB on H100) and rounds up slightly to stay conservative for
    # OOM gating. For plain HF/transformers without value-embeds + torch.compile, you
    # can lower this to 14-20 via --activation-bytes-per-bsh once you have a measurement.
    activation_bytes_per_bsh: float = 40.0
    # CUDA workspace / fragmentation slack
    cuda_overhead_mb: float = 1024.0   # kernels, cuBLAS workspaces, NCCL, etc.
    fragmentation_factor: float = 1.10  # PyTorch caching allocator slack


@dataclass
class ParamBreakdown:
    embedding: int
    unembedding: int
    attention: int
    mlp: int
    value_embeddings: int
    misc: int
    total: int


@dataclass
class MemoryReport:
    params_mb: float
    grads_mb: float
    optim_state_mb: float
    master_weights_mb: float
    activations_mb: float
    cuda_overhead_mb: float
    peak_vram_mb: float
    peak_vram_gb: float


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(a: ArchSpec) -> ParamBreakdown:
    """Replicates the structure of autoresearch's GPT for accurate counting.
    For tied embeddings (lm_head shares wte), unembedding is folded into embedding.
    """
    embedding = a.vocab_size * a.n_embd
    unembedding = 0 if a.tied_embeddings else a.vocab_size * a.n_embd

    kv_dim = a.n_kv_head * a.head_dim
    q_dim = a.n_head * a.head_dim
    # Attention: c_q (E -> q_dim), c_k (E -> kv_dim), c_v (E -> kv_dim), c_proj (E -> E)
    attn_per_layer = (
        a.n_embd * q_dim       # c_q
        + a.n_embd * kv_dim    # c_k
        + a.n_embd * kv_dim    # c_v
        + a.n_embd * a.n_embd  # c_proj
    )
    # MLP: c_fc (E -> 4E), c_proj (4E -> E)  (SwiGLU users: bump mlp_mult ~2.67x3/2)
    mlp_hidden = int(round(a.mlp_mult * a.n_embd))
    mlp_per_layer = a.n_embd * mlp_hidden + mlp_hidden * a.n_embd
    attention = attn_per_layer * a.depth
    mlp = mlp_per_layer * a.depth

    ve_dim = a.value_embed_dim if a.value_embed_dim is not None else kv_dim
    value_embeddings = a.vocab_size * ve_dim * a.value_embed_layers

    # Per-layer residual scalars + ve gates etc -- negligible, lumped into misc.
    misc = 2 * a.depth + 32 * a.n_kv_head * max(0, a.value_embed_layers)

    total = embedding + unembedding + attention + mlp + value_embeddings + misc
    return ParamBreakdown(
        embedding=embedding,
        unembedding=unembedding,
        attention=attention,
        mlp=mlp,
        value_embeddings=value_embeddings,
        misc=misc,
        total=total,
    )


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def _bytes(dtype: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype {dtype!r}. Known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[dtype]


def _optim_state_multiplier(optimizer: str) -> float:
    """Number of optimizer-state tensors stored per parameter (each sized like the param)."""
    o = optimizer.lower()
    if o == "sgd":
        return 0.0
    if o == "sgd_momentum":
        return 1.0
    if o == "adamw" or o == "adam":
        return 2.0                          # exp_avg + exp_avg_sq
    if o == "lion":
        return 1.0                          # single momentum buffer
    if o == "muon":
        return 1.0 + 1e-3                   # momentum + a tiny second-moment buffer
    if o == "muon_adamw":
        return 1.7                          # blended: matrix params (most) use Muon, embed/scalar use AdamW
    raise ValueError(f"Unknown optimizer {optimizer!r}")


def estimate_memory(a: ArchSpec, t: TrainSpec, params: ParamBreakdown) -> MemoryReport:
    n = params.total
    pb = _bytes(t.param_dtype)
    gb = _bytes(t.grad_dtype)
    sb = _bytes(t.optim_state_dtype)

    params_mb = n * pb / 1024 / 1024
    grads_mb = n * gb / 1024 / 1024

    optim_mult = _optim_state_multiplier(t.optimizer)
    optim_state_mb = n * optim_mult * sb / 1024 / 1024
    master_mb = n * 4 / 1024 / 1024 if t.master_weights else 0.0

    # Activations: dominant term at large batch / long context.
    b = t.device_batch_size
    s = a.seq_len
    h = a.n_embd
    L = a.depth
    bsh = b * s * h
    if t.activation_checkpointing:
        # Recompute strategy: keep only block boundaries; save factor ~sqrt(L)
        per_layer_factor = t.activation_bytes_per_bsh / max(1.0, math.sqrt(L))
    else:
        per_layer_factor = t.activation_bytes_per_bsh

    activations_mb = bsh * per_layer_factor * L / 1024 / 1024
    if not t.flash_attention:
        # O(s^2) attention scores tensor per head, per layer, in fp32 for softmax stability
        attn_scores_mb = b * a.n_head * s * s * 4 * L / 1024 / 1024
        activations_mb += attn_scores_mb

    subtotal = params_mb + grads_mb + optim_state_mb + master_mb + activations_mb
    peak = (subtotal + t.cuda_overhead_mb) * t.fragmentation_factor

    return MemoryReport(
        params_mb=params_mb,
        grads_mb=grads_mb,
        optim_state_mb=optim_state_mb,
        master_weights_mb=master_mb,
        activations_mb=activations_mb,
        cuda_overhead_mb=t.cuda_overhead_mb,
        peak_vram_mb=peak,
        peak_vram_gb=peak / 1024,
    )


# ---------------------------------------------------------------------------
# Compute / throughput estimate
# ---------------------------------------------------------------------------

def estimate_flops_per_token(a: ArchSpec, params: ParamBreakdown) -> float:
    """Standard 6N + attention correction (Chinchilla form)."""
    nonembed = params.attention + params.mlp + params.misc
    # 12 * h * head_dim * s per layer is the attention FLOP contribution for full causal attn.
    attn_flops_per_token = 12 * a.n_head * a.head_dim * a.seq_len * a.depth
    return 6 * nonembed + attn_flops_per_token


def estimate_throughput(a: ArchSpec, t: TrainSpec, params: ParamBreakdown,
                        peak_tflops: float, target_mfu: float = 0.45) -> dict:
    flops_per_token = estimate_flops_per_token(a, params)
    achievable = peak_tflops * 1e12 * target_mfu
    tokens_per_sec = achievable / flops_per_token if flops_per_token > 0 else float("inf")
    tokens_per_step = t.device_batch_size * t.grad_accum_steps * a.seq_len
    step_time_s = tokens_per_step / tokens_per_sec if tokens_per_sec > 0 else float("inf")
    return {
        "flops_per_token": flops_per_token,
        "target_mfu": target_mfu,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_step": tokens_per_step,
        "step_time_s": step_time_s,
    }


# ---------------------------------------------------------------------------
# Host CPU / RAM
# ---------------------------------------------------------------------------

def estimate_host_resources(a: ArchSpec, t: TrainSpec) -> dict:
    """vCPU and RAM headroom for the DataLoader pipeline.

    Rule of thumb: ~2-4 vCPU per DataLoader worker (decompression, tokenization,
    collation), plus 1 main thread for the training loop, plus a few cores for
    CUDA driver/RPC overhead. RAM: each worker prefetches a few batches --
    budget ~2x (batch bytes) per worker as pinned-memory headroom.
    """
    # Recommend workers: enough to keep one micro-batch of tokens flowing per step
    # without blocking. Floor at 2, cap at 16 per GPU (more rarely helps).
    suggested_workers = min(16, max(2, t.device_batch_size // 16))
    vcpu_recommended = suggested_workers * 2 + 4

    # Pinned-memory batch footprint (long int64 ids: 8 bytes per token)
    batch_bytes = t.device_batch_size * a.seq_len * 8
    pinned_mb = (suggested_workers + 2) * batch_bytes / 1024 / 1024

    # Working RAM: tokenizer model + dataset shard cache + Python overhead.
    # ~1 GB Python+driver baseline, plus prefetch headroom, plus shard buffer.
    base_ram_mb = 1024
    shard_buffer_mb = 2048   # rough room for parquet row groups + tokenizer state
    ram_recommended_mb = base_ram_mb + shard_buffer_mb + pinned_mb

    return {
        "suggested_dataloader_workers": suggested_workers,
        "vcpu_recommended": vcpu_recommended,
        "pinned_memory_mb": pinned_mb,
        "ram_recommended_mb": ram_recommended_mb,
        "ram_recommended_gb": ram_recommended_mb / 1024,
    }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def preset_autoresearch(depth: int = 8, aspect_ratio: int = 64,
                         head_dim: int = 128) -> tuple[ArchSpec, TrainSpec]:
    """Mirrors train.py defaults and its model-dim derivation:
        base_dim = depth * ASPECT_RATIO
        n_embd   = round-up base_dim to a multiple of HEAD_DIM
        n_head   = n_embd / HEAD_DIM
    """
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    arch = ArchSpec(
        depth=depth,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_head,
        head_dim=head_dim,
        seq_len=2048,
        vocab_size=8192,
        mlp_mult=4.0,
        value_embed_layers=depth // 2,    # alternating layers have value embeddings
    )
    train = TrainSpec(
        device_batch_size=128,
        grad_accum_steps=2,         # TOTAL_BATCH_SIZE=2**19 / (128*2048) = 2
        param_dtype="bf16",
        grad_dtype="bf16",
        optim_state_dtype="bf16",
        master_weights=False,
        optimizer="muon_adamw",
        flash_attention=True,
        activation_checkpointing=False,
    )
    return arch, train


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_human(arch: ArchSpec, train: TrainSpec, params: ParamBreakdown,
                 mem: MemoryReport, throughput: dict, host: dict,
                 gpu_name: str, gpu_vram_gb: float, headroom_pct: float) -> str:
    lines = []
    h = lines.append
    h("=" * 70)
    h("LLM hardware estimate")
    h("=" * 70)
    h(f"Target GPU         : {gpu_name} ({gpu_vram_gb:.0f} GB VRAM)")
    h(f"Architecture       : depth={arch.depth}  n_embd={arch.n_embd}  "
      f"heads={arch.n_head}  kv_heads={arch.n_kv_head}  head_dim={arch.head_dim}")
    h(f"Sequence / vocab   : seq_len={arch.seq_len}  vocab={arch.vocab_size}")
    h(f"Batching           : micro_batch={train.device_batch_size}  "
      f"grad_accum={train.grad_accum_steps}  "
      f"tokens/step={train.device_batch_size*train.grad_accum_steps*arch.seq_len:,}")
    h(f"Recipe             : optimizer={train.optimizer}  "
      f"param/grad/optim={train.param_dtype}/{train.grad_dtype}/{train.optim_state_dtype}  "
      f"flash_attn={train.flash_attention}  ckpt={train.activation_checkpointing}")
    h("")
    h("Parameter count")
    h("-" * 70)
    h(f"  Embeddings (wte)      : {params.embedding:>14,}")
    h(f"  LM head (unembed)     : {params.unembedding:>14,}")
    h(f"  Attention             : {params.attention:>14,}")
    h(f"  MLP                   : {params.mlp:>14,}")
    h(f"  Value embeddings      : {params.value_embeddings:>14,}")
    h(f"  Misc (scalars/gates)  : {params.misc:>14,}")
    h(f"  TOTAL                 : {params.total:>14,}  ({params.total/1e6:.1f} M)")
    h("")
    h("Peak VRAM breakdown")
    h("-" * 70)
    h(f"  Model parameters      : {mem.params_mb:>10.1f} MB")
    h(f"  Gradients             : {mem.grads_mb:>10.1f} MB")
    h(f"  Optimizer state       : {mem.optim_state_mb:>10.1f} MB")
    if mem.master_weights_mb > 0:
        h(f"  Master weights (fp32) : {mem.master_weights_mb:>10.1f} MB")
    h(f"  Activations           : {mem.activations_mb:>10.1f} MB")
    h(f"  CUDA / workspace      : {mem.cuda_overhead_mb:>10.1f} MB")
    h(f"  --------------------- : ----------")
    h(f"  Estimated peak        : {mem.peak_vram_mb:>10.1f} MB  ({mem.peak_vram_gb:.1f} GB)")
    headroom_gb = gpu_vram_gb * (1 - headroom_pct / 100)
    fits = mem.peak_vram_gb <= headroom_gb
    verdict = "FITS" if fits else "WILL OOM"
    h(f"  Headroom target       : <= {headroom_gb:.1f} GB  ({headroom_pct:.0f}% reserved)")
    h(f"  Verdict               : {verdict}")
    h("")
    h("Compute / throughput")
    h("-" * 70)
    h(f"  FLOPs / token         : {throughput['flops_per_token']:.3e}")
    h(f"  Assumed MFU           : {throughput['target_mfu']*100:.0f}%")
    h(f"  Tokens / sec ceiling  : {throughput['tokens_per_sec']:,.0f}")
    h(f"  Tokens / opt step     : {throughput['tokens_per_step']:,}")
    h(f"  Step time @ MFU       : {throughput['step_time_s']*1000:.0f} ms")
    h("")
    h("Host (CPU / RAM)")
    h("-" * 70)
    h(f"  DataLoader workers    : {host['suggested_dataloader_workers']}")
    h(f"  vCPU recommended      : {host['vcpu_recommended']}")
    h(f"  RAM recommended       : {host['ram_recommended_gb']:.1f} GB")
    h(f"  Pinned-batch buffer   : {host['pinned_memory_mb']:.1f} MB")
    h("")
    h("Tips")
    h("-" * 70)
    if not fits:
        h("  - OOM risk: cut device_batch_size in half (single biggest knob).")
        h("  - Then try enabling activation_checkpointing (cost: ~30% slower).")
        h("  - Or shrink n_embd / depth -- params scale ~linearly, activations linearly with depth.")
    else:
        slack_gb = headroom_gb - mem.peak_vram_gb
        h(f"  - {slack_gb:.1f} GB of slack vs the {headroom_pct:.0f}% headroom target.")
        if slack_gb > 8:
            h("  - You can likely double device_batch_size for ~1.5-2x throughput.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_specs_from_args(args) -> tuple[ArchSpec, TrainSpec]:
    if args.preset == "autoresearch":
        # Rebuild the preset with the user's depth/aspect-ratio/head-dim overrides
        # so that --depth 12 actually widens the model the way train.py would.
        arch, train = preset_autoresearch(
            depth=args.depth or 8,
            aspect_ratio=args.aspect_ratio or 64,
            head_dim=args.head_dim or 128,
        )
    else:
        # Build from scratch -- head_dim required, derive heads from n_embd.
        if args.aspect_ratio is not None and args.depth is not None and args.head_dim is not None:
            base_dim = args.depth * args.aspect_ratio
            n_embd = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
        else:
            n_embd = args.n_embd
        if n_embd is None or args.depth is None or args.head_dim is None:
            raise SystemExit("Need --preset OR (--depth + --head-dim + (--aspect-ratio or --n-embd))")
        n_head = n_embd // args.head_dim
        arch = ArchSpec(
            depth=args.depth,
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=args.n_kv_head or n_head,
            head_dim=args.head_dim,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            mlp_mult=args.mlp_mult,
            tied_embeddings=args.tied_embeddings,
            value_embed_layers=args.value_embed_layers,
        )
        grad_accum = max(1, args.total_batch_size // (args.device_batch_size * args.seq_len)) \
            if args.total_batch_size else 1
        train = TrainSpec(
            device_batch_size=args.device_batch_size,
            grad_accum_steps=grad_accum,
            param_dtype=args.param_dtype,
            grad_dtype=args.grad_dtype,
            optim_state_dtype=args.optim_state_dtype,
            master_weights=args.master_weights,
            optimizer=args.optimizer,
            flash_attention=not args.no_flash_attention,
            activation_checkpointing=args.activation_checkpointing,
        )

    # Allow batch / seq overrides on top of preset (depth/aspect already applied).
    if args.preset == "autoresearch":
        if args.device_batch_size is not None:
            train.device_batch_size = args.device_batch_size
        if args.seq_len != 2048:
            arch.seq_len = args.seq_len
        if args.vocab_size != 8192:
            arch.vocab_size = args.vocab_size
    if args.activation_bytes_per_bsh is not None:
        train.activation_bytes_per_bsh = args.activation_bytes_per_bsh
    return arch, train


def main():
    p = argparse.ArgumentParser(
        description="Estimate peak VRAM, host RAM, and throughput for an LLM training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", choices=["autoresearch", "custom"], default="custom",
                   help="autoresearch = mirrors this repo's train.py defaults")
    p.add_argument("--gpu", choices=sorted(GPU_PRESETS.keys()), default="h100-sxm",
                   help="Target GPU (drives peak FLOPS and VRAM ceiling)")
    p.add_argument("--headroom-pct", type=float, default=15.0,
                   help="Fraction of VRAM to leave free for safety (%)")
    p.add_argument("--target-mfu", type=float, default=0.45,
                   help="Assumed Model FLOPs Utilization (0-1)")
    # Arch
    p.add_argument("--depth", type=int)
    p.add_argument("--aspect-ratio", type=int, help="n_embd = depth * aspect_ratio (rounded to head_dim)")
    p.add_argument("--head-dim", type=int)
    p.add_argument("--n-embd", type=int, help="(alternative to --aspect-ratio)")
    p.add_argument("--n-kv-head", type=int, help="GQA: KV heads (< n_head). Defaults to n_head (MHA).")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--mlp-mult", type=float, default=4.0)
    p.add_argument("--tied-embeddings", action="store_true")
    p.add_argument("--value-embed-layers", type=int, default=0)
    # Train
    p.add_argument("--device-batch-size", type=int)
    p.add_argument("--total-batch-size", type=int,
                   help="Tokens per optimizer step (drives gradient accumulation)")
    p.add_argument("--param-dtype", default="bf16", choices=list(DTYPE_BYTES))
    p.add_argument("--grad-dtype", default="bf16", choices=list(DTYPE_BYTES))
    p.add_argument("--optim-state-dtype", default="bf16", choices=list(DTYPE_BYTES))
    p.add_argument("--master-weights", action="store_true",
                   help="Maintain an fp32 master copy of params (classic AMP)")
    p.add_argument("--optimizer", default="muon_adamw",
                   choices=["adamw", "adam", "muon", "muon_adamw", "sgd", "sgd_momentum", "lion"])
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--no-flash-attention", action="store_true",
                   help="Disable FlashAttention assumption (much higher attn activations)")
    p.add_argument("--activation-bytes-per-bsh", type=float,
                   help="Override activation bytes per (batch * seq * hidden) per layer. "
                        "Calibrate by running once and dividing measured peak by b*s*h*L.")
    # Output
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    args = p.parse_args()

    arch, train = build_specs_from_args(args)
    params = count_parameters(arch)
    mem = estimate_memory(arch, train, params)
    peak_tflops, vram_gb = GPU_PRESETS[args.gpu]
    throughput = estimate_throughput(arch, train, params, peak_tflops, args.target_mfu)
    host = estimate_host_resources(arch, train)

    if args.json:
        payload = {
            "gpu": {"name": args.gpu, "peak_bf16_tflops": peak_tflops, "vram_gb": vram_gb},
            "arch": asdict(arch),
            "train": asdict(train),
            "params": asdict(params),
            "memory": asdict(mem),
            "throughput": throughput,
            "host": host,
            "verdict_fits": mem.peak_vram_gb <= vram_gb * (1 - args.headroom_pct / 100),
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print(format_human(arch, train, params, mem, throughput, host,
                           args.gpu, vram_gb, args.headroom_pct))


if __name__ == "__main__":
    main()
