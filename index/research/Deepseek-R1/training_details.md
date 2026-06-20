Here is the expanded, highly detailed report on the training stages of **DeepSeek-R1**, integrating the exact architectural foundations and the technical rationale ("why" and "what led to what") behind each design choice as detailed in the [DeepSeek-R1 Technical Report](https://arxiv.org/pdf/2501.12948).

---

## 0. Base Model and Architecture Foundations

Before launching into the training pipeline, it is essential to establish the architecture of the underlying engine:

* **Base Model:** **DeepSeek-V3-Base**.
* **Architecture:** A massive Mixture-of-Experts (MoE) Transformer featuring **671B total parameters** with **37B activated per token**.
* **Attention Mechanism:** Utilizes **Multi-Head Latent Attention (MLA)**, which compresses the Key-Value (KV) cache into a low-rank latent space ($d_c = 512$) to drastically reduce the VRAM footprint during long-context inference while maintaining standard multi-head attention performance.
* **Core Pre-training Exposure:** The base model was pre-trained on 14.8 trillion tokens, heavily weighted toward mathematics and code-related content. This exposure was critical; it equipped the model with a strong foundational repository of "reasoning traces" and factual knowledge, giving reinforcement learning a fertile starting ground to optimize.

### PyTorch/Python Implementation of the Base Block (MLA + MoE)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class MLAWithMoEBlock(nn.Module):
    """
    Simplified structural representation of a DeepSeek Transformer layer
    integrating Multi-Head Latent Attention (MLA) and DeepSeekMoE.
    """
    def __init__(self, d=7168, n_h=128, d_h=128, d_c=512, N_s=1, N_r=256, K_r=8, d_exp=2048):
        super().__init__()
        self.n_h, self.d_h, self.d_c = n_h, d_h, d_c
        self.N_r, self.K_r = N_r, K_r
        
        # MLA Projections
        self.W_DKV = nn.Linear(d, d_c, bias=False)
        self.W_UK = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_UV = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_DQ = nn.Linear(d, d_c, bias=False)
        self.W_UQ = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_O = nn.Linear(n_h * d_h, d, bias=False)
        self.attn_norm = RMSNorm(d)
        self.kv_compr_norm = RMSNorm(d_c)
        
        # MoE Components
        self.moe_norm = RMSNorm(d)
        self.shared_expert = nn.Sequential(
            nn.Linear(d, d_exp, bias=False), nn.SiLU(), nn.Linear(d_exp, d, bias=False)
        )
        self.routed_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_exp, bias=False), nn.SiLU(), nn.Linear(d_exp, d, bias=False))
            for _ in range(N_r)
        ])
        self.router_centroids = nn.Parameter(torch.randn(N_r, d))

    def forward(self, h_t):
        B, T, C = h_t.shape
        residual = h_t
        
        # --- Multi-Head Latent Attention (MLA) ---
        x_attn = self.attn_norm(h_t)
        c_kv = self.kv_compr_norm(self.W_DKV(x_attn))
        k = self.W_UK(c_kv).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        v = self.W_UV(c_kv).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        
        c_q = self.W_DQ(x_attn)
        q = self.W_UQ(c_q).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_h ** -0.5)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=h_t.device), diagonal=1)
        attn_weights = F.softmax(scores + mask.unsqueeze(0).unsqueeze(1), dim=-1)
        
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, -1)
        h_t = residual + self.W_O(context)
        
        # --- DeepSeekMoE Sub-block ---
        residual = h_t
        x_moe = self.moe_norm(h_t).view(-1, C)
        
        # Shared Expert output
        shared_out = self.shared_expert(x_moe)
        
        # Routing scoring via Sigmoid affinity
        norm_tokens = F.normalize(x_moe, dim=-1)
        norm_centroids = F.normalize(self.router_centroids, dim=-1)
        affinity = torch.sigmoid(torch.matmul(norm_tokens, norm_centroids.t()))
        
        topk_scores, topk_indices = torch.topk(affinity, self.K_r, dim=-1)
        gating_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        routed_out = torch.zeros_like(x_moe)
        for idx in range(self.N_r):
            mask = (topk_indices == idx)
            if not mask.any(): continue
            token_indices, choice_positions = torch.where(mask)
            exp_in = x_moe[token_indices]
            exp_out = self.routed_experts[idx](exp_in)
            weight = gating_weights[token_indices, choice_positions].unsqueeze(-1)
            routed_out[token_indices] += exp_out * weight
            
        output = (shared_out + routed_out).view(B, T, C)
        return residual + output

```

---

## 1. Experimental Precursor: DeepSeek-R1-Zero

DeepSeek-R1-Zero was an aggressive, structural experiment designed to isolate the power of pure reinforcement learning. The core hypothesis was that **Supervised Fine-Tuning (SFT) on human reasoning traces acts as a cognitive constraint**, forcing models to mimic shallow human formatting rather than discovering optimal, alien processing pathways.

### Data Used and Construction

* **Data Used:** A highly concentrated pool of roughly **thousands of hard, verifiable reasoning prompts** spanning advanced Competitive Mathematics (AIME, MATH), Algorithm Competitions (Codeforces, LeetCode), and hard logic/puzzle problems.
* **Construction:** Prompts were extracted from live operational platforms. Critically, **zero human-annotated chain-of-thought (CoT) descriptions or step-by-step target templates were used**. Only the raw query and a computer-verifiable solution key (e.g., a deterministic integer or a unit-test suite) were provided.

### Training Details (Compute, Evaluations, Loss Functions)

* **Compute:** Conducted on a cluster subset using **512 total H800 GPUs** (64 nodes of 8× H800) for approximately 198 continuous runtime hours (~101K total GPU hours).
* **Optimization Loop:** Leveraged **Group Relative Policy Optimization (GRPO)**.
* *Why GRPO?* Traditional PPO requires a secondary "Critic" model of equivalent size to evaluate state-values. For a 671B parameter model, hosting a twin Critic model doubles VRAM overhead and destroys pipeline memory efficiency. GRPO entirely avoids the Critic model by sampling a group of $G=16$ outputs per prompt and using the group's mathematical average and standard deviation to establish relative token advantages.


* **Hyper-parameters:** Base learning rate set to $3 \times 10^{-6}$. Max token sequence rollout scaled dynamically from 32,768 up to 65,536 tokens past the 8,200-step update boundary.
* **Loss Function:** GRPO clipped policy surrogate objective optimizing next-token probability ratios, penalized by an analytical KL divergence check against a reference policy ($\beta = 0.001$).
* **The Reward Function (Strict Rules Only):** No neural reward models were permitted because neural models are heavily susceptible to "reward hacking" (learning superficial prose styles that trick the validator without solving the underlying logic).
* *Accuracy Reward:* Check absolute correctness (e.g., evaluating a math answer string via python `SymPy` parsing or running code solutions inside an execution sandbox against hidden test cases).
* *Format Reward:* Evaluated if the output correctly utilized structural tags (`<think>...</think>` and `<answer>...</answer>`).



### What Changes Led to What Improvements & Architectural Rationale

* **The "Aha Moment" and Test-Time Compute Allocation:** By providing an open-ended token generation budget coupled with a reward signal that only cared about final correctness, the model naturally realized that **generating more intermediate tokens directly correlated with avoiding errors**. It autonomously discovered advanced problem-solving traits: structural self-correction, logical backtracking, and alternative pathway testing. This behavior emerged without explicitly telling the model to "think step by step."
* **Language Explosion Failure Mode:** Because the model had no language-style constraints, it discovered that mixing dialects (shifting dynamically between English and Chinese mid-sentence) was computationally optimal for structural token packing. This led to severe **readability degradation and language mixing**, prompting the realization that cold-start constraints are required for human-facing production software.

---

## 2. DeepSeek-R1 Production Pipeline: Stage 1 (Cold Start)

To maintain the raw intelligence discovered in R1-Zero while completely fixing its erratic readability, text formatting, and multi-dialect drift, the team designed the production **DeepSeek-R1** pipeline starting with a focused "Cold Start" grounding step.

### Data Used and Construction

* **Data Used:** Several thousand highly curated, gold-standard, human-readable **Long Chain-of-Thought (CoT) samples**.
* **Construction:** Sourced using a clever human-in-the-loop hybrid synthesis engine:
1. Prompts were processed through intermediate versions of R1-Zero to capture high-intensity reasoning paths.
2. Paths that achieved a perfect correctness score were isolated.
3. Human annotators manually refactored these paths into an explicit, reader-friendly structure using a first-person conversational framework (*"I should analyze this... wait, let me re-verify..."*).
4. They filtered out any mixed-language tokens and forced structural adherence to clean Markdown and standard LaTeX mathematical expressions.
5. A teacher model was then prompted with these high-quality examples to generate additional samples, which underwent a final strict human verification step.



### Training Details (Compute, Evaluations, Loss Functions)

* **Compute:** Minor computational footprint, consuming a small fraction of the broader **5K H800 GPU hours** set aside for data generation pipelines.
* **Training Framework:** Standard Supervised Fine-Tuning (SFT) executed over 2 to 3 training epochs.
* **Hyper-parameters:** Cosine learning rate scheduler dropping from a peak of $5 \times 10^{-5}$ down to a minimum floor of $5 \times 10^{-6}$. Sequence packing handled contexts up to 32,768 parameters using an isolated sample masking strategy.
* **Loss Function:** Standard token-level autoregressive Cross-Entropy.

### What Changes Led to What Improvements & Architectural Rationale

* **Establishing a Readability Prior:** Introducing this tiny sliver of structural initialization before scaling up reinforcement learning **drastically reduced formatting errors and dialect instability**, without degrading the underlying model capacity. It proved that a massive SFT corpus is not necessary; rather, providing a clear structural blueprint allows the subsequent RL loops to safely expand logic without structural degradation.

---

## 3. DeepSeek-R1 Production Pipeline: Stage 2 (Reasoning-Oriented RL)

This stage combined the structural formatting learned in the Cold Start phase with large-scale reinforcement learning to maximize logical capability across complex fields.

### Data Used and Construction

* **Data Used:** An expanded dataset of **65,000 highly structured reasoning prompts**.
* **Construction:** Compiled systematically across four primary categories: Quantitative Mathematics (26K), Algorithmic Coding + Bug Fixing (25K), STEM Multiple-Choice problems (22K), and Logic/Deduction Puzzles (15K).

### Training Details (Compute, Evaluations, Loss Functions)

* **Compute:** Processed as a major segment of the core **41K H800 GPU hour** reinforcement framework using 512 distributed GPUs over an 80-hour continuous training sprint.
* **Optimization Framework:** GRPO layout matching the R1-Zero layout, utilizing a group size of $G=16$ for advantage estimation, but enforcing a reduced sampling temperature of **0.7** to maintain logical coherence across long context windows.
* **Loss Function:** GRPO clipped surrogate policy reward loss.
* **Reward Design (The Multi-Component Shift):**

$$\text{Reward} = \text{Reward}_{\text{rule}} + \text{Reward}_{\text{language}}$$


* $\text{Reward}_{\text{rule}}$ applied the automated sandbox compiler and `SymPy` mathematical correctness checks.
* $\text{Reward}_{\text{language}}$ was explicitly added to solve the language mixing issue. It used an automated parsing metric that calculated the exact mathematical ratio of target-language tokens within the `<think>` block, strictly penalizing the model if it drifted into an alternative language.



### What Changes Led to What Improvements & Architectural Rationale

* **Enforcing Dialect Stability:** The addition of the explicit language consistency reward completely eliminated the language mixing issue discovered in R1-Zero, forcing the model to think in the exact language requested by the user prompt.
* **Why the Temperature Shift Mattered:** Lowering the exploration temperature from 1.0 to 0.7 during generation rollouts significantly stabilized long-context chains, preventing the model from spinning out into incoherent or repetitive semantic loops during deep, multi-thousand-token thought tasks.

---

## 4. DeepSeek-R1 Production Pipeline: Stage 3 (Rejection Sampling & Mixed SFT)

While Stage 2 produced an elite logical model, it suffered from a major flaw: it was overly specialized. It lost user-friendly capabilities for generic instructions, such as creative writing, role-play, or straightforward question answering. This stage set out to re-endow the model with versatile human alignment.

### Data Used and Construction

* **Data Used:** A large, multi-domain instruction-tuning dataset totaling **804,745 samples**.
* **Reasoning Data Construction (600K samples):** Generated using massive **rejection sampling** tracks run against the previous RL checkpoint. The model generated multiple long thought paths per prompt. Only responses that successfully passed the verifier *and* had clean, digestible, non-repetitive thought loops were saved.
* **Non-Reasoning Data Construction (200K samples):** Sourced from creative writing, factual QA, and software engineering data. For complex user prompts, developers forced DeepSeek-V3 to generate a clear, helpful "thought loop" *before* writing the final answer. Simple queries (e.g., *"Hello"*) skipped this process entirely to ensure the model wouldn't "overthink" conversational basics.

### Training Details (Compute, Evaluations, Loss Functions)

* **Compute:** Consumed standard cluster SFT compute cycles across 2 epochs.
* **Training Layout:** Learning rate scheduler scaled down from $5 \times 10^{-5}$ to $5 \times 10^{-6}$ with a global batch size of 128. Sample-masking kept parallel packed context sequences isolated.
* **Loss Function:** Standard Autoregressive Cross-Entropy.

### What Changes Led to What Improvements & Architectural Rationale

* **Resolving "Overthinking" on Simple Tasks:** By explicitly mapping simple queries to skip the internal `<think>` block during SFT data construction, developers successfully taught the model to **respond instantly to basic conversations**, while reserving deep test-time compute for genuinely difficult problems.
* **Formatting Cleanliness:** Filtering out chaotic or circular reasoning tracks during rejection sampling significantly cleaned up the readability of the model's inner thoughts.

---

## 5. DeepSeek-R1 Production Pipeline: Stage 4 (Diverse Preference Alignment RL)

The final post-training stage aimed to align the model with human ethical boundaries, safety standards, and nuanced stylistic preferences without undermining its hard-won logical capabilities.

### Data Used and Construction

* **Data Used:** A combined dataset pairing the reasoning prompt corpus with **66,000 helpfulness and harmlessness alignment prompts**.
* **Construction:** Prompts were processed to create preference pairs (Chosen vs. Rejected). Crucially, **helpfulness rewards only evaluated the final summary text**, minimizing interference with the model's underlying reasoning process. For **harmlessness, the entire generation (including the thought process) was evaluated** to flag safety violations, malicious intent, or jailbreak vulnerabilities.

### Training Details (Compute, Evaluations, Loss Functions)

* **Compute:** The final phase of the core 512-GPU optimization track.
* **Framework:** GRPO pipeline running for 1,700 total steps.
* **The Global Reward Formulation:**

$$\text{Reward} = \text{Reward}_{\text{reasoning}} + \text{Reward}_{\text{general}} + \text{Reward}_{\text{language}}$$


* $\text{Reward}_{\text{reasoning}}$ utilized rule-based verifiers for mathematical and coding correctness.
* $\text{Reward}_{\text{general}}$ utilized **Model-Based Reward Models** trained on human preference data to score open-ended helpfulness and harmlessness. Crucially, the reward model itself was forced to output a "chain-of-thought" before scoring, neutralizing its own vulnerability to reward hacking.


* **The Temporal Bias Change:** Preference-based alignment rewards were **exclusively introduced in the final 400 steps** of the 1,700-step training run.

### What Changes Led to What Improvements & Architectural Rationale

* **Mitigating Reward Hacking:** Training preference-based reward models to output their own chain of thought before scoring prevented the policy model from exploiting superficial text features to get high scores.
* **Preventing Performance Regressions (The 400-Step Gate):** Running preference-based rewards across the entire training run caused severe reward hacking and degraded performance on hard math tasks (as shown in Figure 6 of the report). Limiting these subjective preference signals to the **final 400 steps** allowed the model to maximize its logical reasoning capabilities first, before safely applying preference alignment.

---

## 6. Comprehensive Summary of the Training Stages

| Stage | Data Engine | Compute & Infrastructure | Core Objective / Loss | The "Why" & Rationale | Key Operational Outcome |
| --- | --- | --- | --- | --- | --- |
| **DeepSeek-R1-Zero** | Thousands of raw, hard reasoning queries; zero human examples. | 512 H800 GPUs for 198 Hours (~101K GPU hours). | GRPO clipped policy loss; rule-based accuracy + format rewards. | **Hypothesis:** Human SFT constraints limit exploration; pure RL allows models to discover optimal pathways autonomously. | Autonomous emergence of self-correction and backtracking ("Aha Moment"). Exploded in length; language mixing occurred. |
| **Stage 1: Cold Start** | Several thousand human-verified, clean, first-person long CoT trajectories. | Minor subset of the 5K total data creation GPU hours. | Next-Token Cross-Entropy SFT over 2-3 epochs. | **Rationale:** Establish a strong structural readability prior to solve R1-Zero's language-mixing and chaotic output formats. | Model grounded in human-readable prose styles, clean Markdown, and LaTeX formatting without losing reasoning potential. |
| **Stage 2: Reasoning RL** | 65,000 math, code, STEM, and logic prompts. | Part of the 41K GPU hour RL allocation; 512 GPUs over 80 hours. | GRPO policy loss; composite rule accuracy + language consistency rewards. | **Rationale:** Scale logical performance while using language consistency rewards to penalize multi-dialect drift. | Eliminated language mixing within thought tracks; stabilized inference generations by reducing sampling temperature to 0.7. |
| **Stage 3: Mixed SFT** | 804,745 samples (600K filtered reasoning tracks via rejection sampling + 200K non-reasoning tasks). | Standard cluster SFT cycles across 2 epochs. | Next-Token Cross-Entropy with sample isolation masking. | **Rationale:** Re-endow the model with versatile human capabilities (writing, basic QA) and prevent overthinking on simple tasks. | Unified elite reasoning with conversational helpfulness. Model learned to respond immediately to simple queries without thinking. |
| **Stage 4: Alignment RL** | Mixed reasoning prompts paired with 66K human preference alignment prompts. | Final stage of the 4-day, 512-GPU runtime loop. | GRPO policy loss; rule-based + model-based reward models. | **Rationale:** Align model behavior with safety and helpfulness guidelines while protecting technical capabilities. | **Strategic Gate:** Preference rewards restricted to final 400 steps to prevent reward hacking and avoid performance drop in math domains. Elite open-source chat variant produced. |