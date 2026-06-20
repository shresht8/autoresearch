Based on the provided technical report for [DeepSeek-V3](https://arxiv.org/pdf/2412.19437), the training process consists of three distinct stages: **Pre-Training**, **Long Context Extension**, and **Post-Training** (comprising Supervised Fine-Tuning and Reinforcement Learning).

Below are the comprehensive details for each step as requested:

---

## 1. Pre-Training Stage

### Data Used and How It Was Constructed

* **Data Volume:** Pre-trained on **14.8 trillion** high-quality, diverse, and multilingual tokens.


* **Composition Changes:** Optimized compared to DeepSeek-V2 by intentionally **enhancing the ratio of mathematical and programming samples**, alongside expanding broader multilingual token coverage beyond English and Chinese.


* **Processing & Integrity:** Refined data processing pipelines minimized redundancy while maintaining high diversity. Documents were packed sequentially using a specific **document packing method** to ensure data integrity. However, cross-sample attention masking was excluded during training.


* **FIM Strategy:** Integrated a **Fill-in-Middle (FIM)** strategy at a rate of **0.1** at the document level during pre-packing, utilizing a Prefix-Suffix-Middle (PSM) framework to allow middle text predictions based on contextual surrounding cues.


* **Tokenizer Modifications:** Employs a Byte-level BPE tokenizer with an expanded vocabulary size of **128K tokens**. To counteract "token boundary bias" stemming from new tokens combining punctuations and line breaks, a designated proportion of these combined structures were randomly split during training.



### Training Details (Compute, Evaluations, Loss Functions)

* **Compute Infrastructure:** Processed on a massive hardware cluster of **2,048 NVIDIA H800 GPUs**. The pre-training loop consumed **2,664K H800 GPU hours** (taking less than two months total on this cluster configuration).


* **Parallelism Layout:** Leveraged a 16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) distributed evenly across 8 cluster nodes, and ZeRO-1 Data Parallelism (DP).


* **Hyper-parameters:** Optimized via the AdamW optimizer with a cosine learning rate decay schedule starting from a peak of $2.2 \times 10^{-4}$ down to a minimum plateau of $7.3 \times 10^{-6}$. Batch sizing dynamically scaled from 3,072 to 15,360 tokens over the sequence lifespan.


* **Loss Functions:** Simultaneously optimized two primary distinct loss streams:
1. **Standard Next-Token Cross-Entropy Loss:** Guided the core language generation capabilities.


2. **Multi-Token Prediction (MTP) Loss ($\mathcal{L}_{MTP}$):** Average cross-entropy loss tracking the concurrent sequential prediction of additional lookahead future tokens, weighted by a factor of $\lambda = 0.3$ for the first 10T tokens and dropped to $\lambda = 0.1$ for the remaining 4.8T tokens.


3. **Complementary Sequence-Wise Balance Loss ($\mathcal{L}_{Bal}$):** Utilized a tiny scaling factor ($\alpha = 0.0001$) acting solely to protect against extreme local sequence imbalance across routed experts.





### Key Observations

* **High Economic Efficiency:** Driven by algorithmic co-design (like native fine-grained FP8 precision execution and DualPipe communication-computation overlapping), pre-training required a remarkably low footprint of **180K H800 GPU hours per trillion tokens**.


* **High Training Stability:** The overall training track was exceptionally clean; the architecture did not experience a single irrecoverable loss spike or require team engineers to perform rollbacks.


* **Exceptional Performance Foundations:** The generated base variant (*DeepSeek-V3-Base*) comprehensively outpaced state-of-the-art open-source peers (including Qwen2.5-72B and LLaMA-3.1-405B) on key mathematical, programming, and multilingual benchmarks.



---

## 2. Long Context Extension Stage

### Data Used and How It Was Constructed

* **Data Characteristics:** *None Explicitly Listed* (focuses on training sequences packed directly to extreme window footprints).
* **Construction Strategy:** Applied **YaRN (Yet another RoPE extensioN method)** context expansion exclusively targeting the model's decoupled shared position key track ($k^R_t$).



### Training Details (Compute, Evaluations, Loss Functions)

* **Compute Footprint:** Consumed **119K H800 GPU hours** split cleanly into two distinct 1,000-step extension intervals.


* **Phase Progressions:** * **Phase 1:** Extended sequence length window up to **32K** tokens, running with a micro-batch scale of 1,920.


* **Phase 2:** Further pushed sequence layout length to **128K** tokens, running with a tighter micro-batch scale of 480.




* **Hyper-parameters:** Ran with a fixed baseline learning rate matching the exact completion floor of the pre-training execution phase ($7.3 \times 10^{-6}$).


* **Evaluations:** Subjected to rigorous empirical testing using the standard **"Needle In A Haystack" (NIAH)** protocol across the entire spectrum of extended windows up to 128K parameters.


* **Loss Functions:** Standard next-token translation target objectives (inherited from basic autoregressive parameters).

### Key Observations

* **Consistent Window Robustness:** DeepSeek-V3 maintained a highly reliable, consistent context-window retrieval performance floor, exhibiting robust, uniform data command recall through all evaluation levels extending fully to the 128K context boundaries.



---

## 3. Post-Training Stage

This stage translates the raw base variant into aligned, highly interactive chat systems via sequential Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) operations.

### Data Used and How It Was Constructed

* **Data Volume:** Curated an instruction-tuning corpus spanning **1.5 million total samples**.


* **Reasoning Data Generation:** Specifically targeted mathematics, coding challenges, and logic tasks by leveraging data generated from an internal expert **DeepSeek-R1** variant.


* *Addressing Length Issues:* Since raw R1 outputs suffered from excessive length, structural overthinking, and messy styling, developers engineered targeted expert model data-generators via pipeline mixtures.


* *Format Mapping:* Formatted samples using two distinct SFT prompt layouts: `<problem, original response>` and `<system prompt, problem, R1 response>`. System prompts purposefully commanded reflection and verification modes.


* *Rejection Sampling:* Rejection sampling was executed at the conclusion of intermediate pipelines to maintain high accuracy, while strictly pruning excessive verbosity.




* **Non-Reasoning Data Generation:** Evaluated creative writing, role-play scenarios, and generic facts by generating prompts via **DeepSeek-V2.5** and processing them through rigorous human verification and curation passes.



### Training Details (Compute, Evaluations, Loss Functions)

* **Compute Footprint:** Highly economical, requiring only **5K H800 GPU hours** to execute.


* **SFT Settings:** Executed across **two training epochs** utilizing a sample-masking strategy to keep parallel packed context windows isolated and invisible from adjacent samples. Learning rates decayed smoothly from $5 \times 10^{-6}$ down to $1 \times 10^{-6}$.


* **RL Framework & Optimization Strategy:** Leveraged **Group Relative Policy Optimization (GRPO)**. GRPO significantly conserves compute footprint by bypassing the traditional, heavy critic model. Instead, it samples groups of alternative responses ($G$) for any given query, estimating baselines straight from the collective group's average performance output scores.


* **Evaluations:** Fully verified using open benchmark evaluation testbeds like IFEval (instruction tracking), Arena-Hard, AlpacaEval 2.0, GPQA, and RewardBench.


* **Loss / Reward Systems:** Driven by a multi-tier reinforcement evaluation setup:
1. **Rule-Based Reward Models (RM):** Deployed over deterministic tracks (e.g., verifying final math equations wrapped inside programmatic formatting boxes or executing generated source code directly inside isolated test compiler sandboxes).


2. **Model-Based Reward Models:** Checked open-ended questions. Checkpoints were trained directly using SFT preference rankings, and explicitly forced to output their own internal "chain-of-thought" leading to a score, successfully neutralizing systemic reward hacking.


3. **Constitutional Self-Rewarding:** Leveraged the voting outputs of DeepSeek-V3 itself to evaluate open subjective queries.





### Key Observations

* **Elite Closed-Source Parity:** Post-training unlocked top-tier performance, allowing DeepSeek-V3 to become the **first open-source language model to surpass an 85% win-rate on Arena-Hard**. This effectively closed the performance gap with elite frontier closed-source systems like GPT-4o and Claude-3.5-Sonnet.


* **Post-Training Distillation Trade-Offs:** Curation pipelines discovered that while long-CoT reasoning distillation from models like DeepSeek-R1 drastically boosts core intelligence and problem-solving capacities, it severely inflates text sequence generation length.


* **Speculative Decoding Speedups:** By discarding MTP modules or recycling them as standalone speculative decoding drivers during chat deployment, inference operations achieved an accelerated **1.8$\times$ output gain in Tokens Per Second (TPS)** with token verification acceptance rates staying high between 85% and 90%.