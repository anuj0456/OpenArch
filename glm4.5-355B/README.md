# GLM-4.5 (355B)

Zhipu AI / Z.ai's 2025 sparse Mixture-of-Experts decoder. 355B total parameters, ~32B active per token. Built in the DeepSeek/Kimi family lineage (sparse MoE + shared expert + first dense blocks + MTP), but with a deliberate "**deeper, not wider**" design choice: fewer experts and a smaller hidden dim than DeepSeek-V3, traded for more layers (92 vs 61) and ~2.5× more attention heads.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![GLM-4.5 architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/glm-4-5-355b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 355B total, ~32B active per token
- **Layers:** 92 transformer blocks (3 dense + 89 MoE) + 1 MTP layer
- **Embedding dim:** 5,120
- **Attention:** GQA with **96 query heads** and 8 KV heads (head dim 128)
- **MoE:** 160 routed experts + 1 shared expert, 8 routed active per token; expert hidden dim 1,536
- **Dense FFN hidden dim (first 3 blocks):** 12,288
- **Context length:** 128k
- **Vocab size:** 151k
- **Positional encoding:** Partial RoPE
- **Normalization:** RMSNorm (pre-norm), plus QK-Norm on Q/K
- **Activation:** SwiGLU
- **Routing:** Sigmoid-gated with loss-free balance routing

## What makes GLM-4.5 different

- **Deeper, not wider.** The GLM-4.5 paper explicitly argues that deeper models reason better. So compared to DeepSeek-V3 (61 layers, 7168 hidden, 256 experts) and Kimi K2 (61 layers, 7168 hidden, 384 experts), GLM-4.5 cuts width and grows depth: 92 layers, 5120 hidden, 160 experts.
- **2.5× more attention heads.** 96 heads at a 5120 hidden dim (head dim 128) — much higher than the usual ~32–64 heads at this scale. The authors report this doesn't lower training loss but consistently improves reasoning benchmarks like MMLU and BBH.
- **Sigmoid gating + loss-free balance routing.** Instead of softmax over experts, GLM-4.5 uses a sigmoid gate (each expert independently scored), combined with a bias term updated to keep expert load balanced without an auxiliary loss.
- **QK-Norm.** RMSNorm applied to queries and keys before the attention dot product, for training stability.
- **Multi-Token Prediction (MTP).** A single MTP layer is trained to predict the next-next token (and beyond), enabling speculative decoding at inference time for ~2× throughput gains.
- **Partial RoPE.** RoPE is applied to only a fraction of the head dim rather than the whole vector — a small efficiency / stability tweak inherited from earlier GLM models.

## What's in `model.py`

- **`InputLayer`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupedQueryAttention`** — GQA with optional QK-Norm on Q/K, RoPE applied to Q/K, and KV-head repetition to match query heads.
- **`FFN`** — SwiGLU feed-forward, used both as the dense first-block FFN and as the per-expert module inside MoE.
- **`MOE`** — top-k routed MoE: a linear router produces logits, top-k experts selected per token, outputs combined weighted by softmaxed router scores.
- **`MultiTokenPredictionHead`** — extra transformer block + LM head used to predict tokens beyond the next one. Includes a static loss helper that combines the standard next-token cross-entropy with auxiliary MTP losses at deeper offsets.
- **`SkipConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → (dense FFN or MoE) → residual`. A `use_dense` flag switches between dense and MoE for the early layers.
- **`GLM45Model`** — embeds tokens, stacks 92 transformer blocks (first 3 dense, rest MoE), applies a final RMSNorm, projects to vocab, and runs MTP heads when training labels are provided.

## References

- [GLM-4.5 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models (paper)](https://arxiv.org/pdf/2508.06471)
- [zai-org/GLM-4.5 config.json](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json)
- [Multi-Token Prediction (MTP) — Sebastian Raschka](https://sebastianraschka.com/llm-architecture-gallery/mtp/)
- [Mixture of Experts Architecture in Transformer Models — Machine Learning Mastery](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)