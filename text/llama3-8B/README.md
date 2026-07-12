# Llama 3 (8B)

Meta's 2024 dense decoder — a reference modern open-weight architecture using RMSNorm, RoPE, grouped-query attention, and a SwiGLU feed-forward.

![Llama 3 8B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/llama-3-8b.webp)

*Architecture diagram from [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/).*

## Key specs

- **Parameters:** 8B
- **Layers:** 32 transformer blocks
- **Embedding dim:** 4,096
- **Attention heads:** 32 query heads with grouped-query attention (8 KV heads)
- **FFN hidden dim:** 14,336
- **Context length:** 8,192
- **Vocab size:** 128k
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU

## What's in `model.py`

A from-scratch PyTorch implementation of the stack shown in the diagram above:

- **`InputEmbedding`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale (no bias, no mean-centering).
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupedQueryAttention`** — multi-head attention where K/V heads are shared across groups of query heads, reducing KV cache size.
- **`FeedForwardNetwork`** — SwiGLU feed-forward: two parallel linear projections gated by SiLU, then a final down-projection.
- **`SkipConnection`** — residual add.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → FFN → residual`.
- **`LLAMA3Model`** — embeds tokens, stacks 32 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Llama 3 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/#card-llama-3-8b)
- [The Llama 3 Herd of Models (tech report)](https://arxiv.org/pdf/2407.21783)
- [meta-llama/Meta-Llama-3-8B config.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)
- [Sebastian Raschka's from-scratch Llama implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama)