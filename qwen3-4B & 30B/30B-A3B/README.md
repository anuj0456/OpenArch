# Qwen3 (30B-A3B)

Alibaba's 2025 sparse Mixture-of-Experts decoder. 30B total parameters, but only ~3B active per token (hence the "A3B" suffix — "Activated 3B"). Architecturally it's Qwen3-style — pre-norm RMSNorm, RoPE, GQA, QK-Norm, SwiGLU — with the dense FFN replaced by a 128-expert MoE that routes 8 experts per token.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Qwen3 30B-A3B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/qwen3-30b-a3b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 30.5B total, ~3.3B active per token
- **Layers:** 48 transformer blocks
- **Embedding dim:** 2,048
- **Attention:** GQA with 32 query heads and 4 KV heads (head dim 128)
- **MoE:** 128 routed experts, 8 active per token, no shared expert; expert hidden dim 768
- **Context length:** 128k
- **Vocab size:** 151,669 (~151k)
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-norm), plus QK-Norm on Q/K
- **Activation:** SwiGLU (in each expert)

## What makes Qwen3 30B-A3B different

- **Sparse MoE FFN.** Each block's feed-forward is a 128-expert layer; only the top 8 experts per token are active. This gives a 30B-parameter model the inference cost of a ~3B dense model.
- **No shared expert.** Unlike DeepSeek V3, Llama 4 Maverick, or Qwen2.5-MoE, Qwen3-MoE explicitly drops the always-on shared expert. The Qwen3 tech report attributes this to fine-grained expert segmentation plus a global-batch load-balancing loss, which together encourage expert specialization without needing a shared fallback path.
- **QK-Norm.** Same as the dense Qwen3 — RMSNorm applied to queries and keys before the attention dot product for training stability.

## What's in `model.py`

- **`InputLayer`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupQueryAttention`** — GQA with optional QK-Norm on Q/K, RoPE applied to Q/K, and KV-head repetition to match query heads.
- **`Expert`** — a single SwiGLU feed-forward expert.
- **`MOE`** — top-k routed MoE: a linear router produces logits, top-k experts are selected per token, their outputs are combined weighted by softmaxed router scores.
- **`SkipConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → MoE → residual`.
- **`Qwen3MoEModel`** — embeds tokens, stacks 48 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Qwen3 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- [Qwen/Qwen3-30B-A3B-Instruct-2507 config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json)
- [Mixture of Experts Architecture in Transformer Models — Machine Learning Mastery](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)