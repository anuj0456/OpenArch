# DeepSeek V3 / R1 (671B)

DeepSeek's 2024–2025 sparse Mixture-of-Experts decoder. 671B total parameters, but only ~37B active per token. Two architectural choices stand out: **Multi-head Latent Attention (MLA)**, which compresses Q/K/V through low-rank latent projections to shrink the KV cache, and a **sparse MoE** feed-forward where a router picks a small subset of experts per token.

![DeepSeek V3 / R1 architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/deepseek-v3-r1-671-billion.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 671B total, ~37B active per token
- **Layers:** 61 transformer blocks (first 3 are dense, the rest are MoE)
- **Embedding dim:** 7,168
- **Attention:** Multi-head Latent Attention, 128 heads
- **MoE:** 256 routed experts + 1 shared expert, 8 active per token; expert hidden dim 2,048
- **Dense FFN hidden dim (first 3 blocks):** 18,432
- **Context length:** 128k
- **Vocab size:** 129k
- **Positional encoding:** RoPE (applied to MLA)
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU

## What makes DeepSeek V3/R1 different

- **Multi-head Latent Attention (MLA).** Q, K, and V are first projected down to small *latent* dimensions, then projected up to per-head vectors when needed. This dramatically reduces the KV cache size at inference, since only the compressed latents need to be stored.
- **Sparse MoE with shared expert.** Each MoE layer has 256 routed experts plus 1 always-on shared expert. The router picks 8 of the 256 per token, so a tiny fraction of the model's parameters is active for any given step.
- **First 3 blocks are dense.** The early layers use a regular SwiGLU FFN (hidden dim 18,432) instead of MoE, since routing is less stable when token representations are still being formed.

## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embedding.
- **`MultiHeadLatentAttention`** — Q/K/V routed through low-rank latent projections (`q_latent_dim`, `kv_latent_dim`) before per-head expansion and attention.
- **`FeedForwardLayer`** — SwiGLU feed-forward, used in the dense early blocks and as the per-expert module inside MoE.
- **`MOELayer`** — top-k routed MoE: a linear router produces logits, top-k experts are selected per token, their outputs are combined weighted by softmaxed router scores.
- **`SkipConnection`** — residual add.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → MLA → residual → RMSNorm → (dense FFN or MoE) → residual`. The `skip_moe` flag switches between dense and MoE for the early layers.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`DeepseekR1Model`** — embeds tokens, stacks 61 transformer blocks (first 3 dense, rest MoE), applies a final RMSNorm, and projects to vocab.

## References

- [DeepSeek V3 / R1 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
- [deepseek-ai/DeepSeek-R1 config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json)
- [A Gentle Introduction to Multi-Head Latent Attention (MLA) — Machine Learning Mastery](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/)
- [Mixture of Experts Architecture in Transformer Models — Machine Learning Mastery](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)