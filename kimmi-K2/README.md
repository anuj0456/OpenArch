# Kimi K2 (1T)

Moonshot AI's 2025 sparse Mixture-of-Experts decoder. **1T total parameters, ~32B active per token.** Architecturally a direct successor to DeepSeek V3/R1: **Multi-head Latent Attention (MLA)** for cache-efficient attention, plus a **sparse MoE FFN with a shared expert**. Kimi K2 scales the design up with 384 routed experts (vs DeepSeek's 256) and 64 MLA heads, making it one of the largest fully-open MoE models published.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Kimi K2 architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/kimi-k2-1-trillion.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 1T total, ~32B active per token
- **Layers:** 61 transformer blocks (1 dense + 60 MoE)
- **Embedding dim:** 7,168
- **Attention:** Multi-head Latent Attention (MLA) with 64 heads
- **MoE:** 384 routed experts + 1 shared expert, 8 routed active per token; expert hidden dim 2,048
- **Dense FFN hidden dim (first block):** 18,432
- **Context length:** 128k
- **Vocab size:** 160k
- **Positional encoding:** RoPE (applied to MLA)
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU

## What makes Kimi K2 different

- **Multi-head Latent Attention (MLA).** Q, K, and V are routed through low-rank latent projections before per-head expansion. The KV cache only needs to store the small latent vectors, not full per-head K/V tensors — a substantial inference-time memory win at 1T scale.
- **Very wide MoE.** 384 routed experts is roughly 1.5× DeepSeek V3 and 3× Llama 4 Maverick. Combined with 8 active routed experts per token, this gives a fine-grained expert specialization pattern.
- **Shared expert.** Like DeepSeek V3 (and unlike Qwen3-MoE), every MoE layer has 1 always-on shared expert that runs in addition to the routed top-k.
- **First block is dense.** As in DeepSeek V3, the first transformer block uses a regular SwiGLU FFN (hidden dim 18,432) instead of MoE. Early-layer representations are typically less specialized, and routing instability hurts most at the bottom of the stack.

## What's in `model.py`

- **`InputLayer`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`MultiHeadLatentAttention`** — Q is projected directly to per-head; K/V are routed through a low-rank latent (`w_dkv`) and then up-projected to per-head K and V. Includes a simple latent KV cache (`cache_c_kv`) for autoregressive decoding.
- **`MLP`** — SwiGLU feed-forward expert: `silu(ll1(x)) * ll2(x)`, then a down-projection. Used both as the dense first-block FFN and as the per-expert module inside MoE.
- **`MOELayer`** — top-k routed MoE with 384 experts.
- **`SkipConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → MLA → residual → RMSNorm → MoE → residual`.
- **`KimmiK2Model`** — embeds tokens, runs the dense first block, stacks 60 MoE transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Kimi K2 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Kimi K2 Technical Report](https://arxiv.org/pdf/2507.20534)
- [moonshotai/Kimi-K2-Instruct config.json](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)
- [A Gentle Introduction to Multi-Head Latent Attention (MLA) — Machine Learning Mastery](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/)
- [Mixture of Experts Architecture in Transformer Models — Machine Learning Mastery](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)