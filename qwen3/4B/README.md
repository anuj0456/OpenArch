# Qwen3 (4B)

Alibaba's 2025 dense decoder. A modern Llama-style stack with one distinctive choice from the OLMo-2 lineage: **QK-Norm** (RMSNorm applied to queries and keys before the attention dot product) for training stability. Otherwise: pre-norm RMSNorm, RoPE, GQA, SwiGLU.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Qwen3 4B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/qwen3-4b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 4B
- **Layers:** 36 transformer blocks
- **Embedding dim:** 2,560
- **Attention:** GQA with 32 query heads and 8 KV heads (head dim 128)
- **FFN hidden dim:** 9,728
- **Context length:** 32k
- **Vocab size:** 151k
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-norm), plus QK-Norm on Q/K
- **Activation:** SwiGLU

## What makes Qwen3 different

- **QK-Norm.** Each attention layer applies RMSNorm to queries and keys (after the linear projection and head split) before the dot product. Same idea as OLMo 2's QK-Norm — keeps attention logits well-scaled during training.
- **Everything else is conventional.** Pre-norm RMSNorm sandwiching attention and FFN, RoPE on Q/K, GQA with a 4:1 query-to-KV head ratio, and a SwiGLU FFN. The architecture is a clean, well-tuned modern baseline.

## What's in `model.py`

- **`InputLayer`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupQueryAttention`** — GQA with optional QK-Norm on Q/K, RoPE applied to Q/K, and a built-in causal mask path. The KV heads are repeated 4× to match the 32 query heads before the attention dot product.
- **`FeedForwardNetwork`** — SwiGLU feed-forward: `silu(ll1(x)) * ll2(x)`, then a down-projection.
- **`SkipConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → FFN → residual`.
- **`Qwen3Model`** — embeds tokens, stacks 36 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Qwen3 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- [Qwen/Qwen3-4B config.json](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)