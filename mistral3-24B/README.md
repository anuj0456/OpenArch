# Mistral Small 3.1 (24B)

Mistral AI's 2025 dense decoder. A clean, conventional modern-Llama-style architecture: pre-norm RMSNorm, RoPE, grouped-query attention, and a SwiGLU feed-forward. No sandwich norms, no QK-Norm, no sliding window — just a well-tuned baseline at 24B parameters with a 128k context.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Mistral Small 3.1 24B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/mistral-3-1-small-24b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 24B
- **Layers:** 40 transformer blocks
- **Embedding dim:** 5,120
- **Attention:** GQA with 32 query heads and 8 KV heads (head dim 128)
- **FFN hidden dim:** 32,768
- **Context length:** 128k
- **Vocab size:** 131k (the gallery diagram shows ~256k; the 3.1 text config is 131,072)
- **Positional encoding:** RoPE (theta = 1e9)
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU

## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup, scaled by √embed_size.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupQueryAttention`** — GQA: 32 query heads sharing 8 KV groups, with RoPE applied to Q/K. Each KV head is repeated 4× to match the query heads before the attention dot product.
- **`FeedForward`** — SwiGLU feed-forward: `silu(ff1(x)) * ff2(x)`, then a down-projection.
- **`SkipConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → FFN → residual`.
- **`Mistral3Model`** — embeds tokens, stacks 40 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Mistral Small 3.1 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Mistral Small 3.1 model card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- [mistralai/Mistral-Small-3.1-24B-Instruct-2503 config.json](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json)