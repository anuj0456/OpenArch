# Gemma 3 (27B)

Google DeepMind's 2025 dense decoder. Architecturally close to Llama, but with several distinctive choices: **sandwich normalization** (RMSNorm both *before* and *after* each sublayer), a **5:1 ratio of local sliding-window attention to global full attention** to keep long-context costs in check, **QK-Norm** on queries and keys, and **GQA** with 2× as many query heads as KV heads.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Gemma 3 27B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/gemma-3-27b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 27B
- **Layers:** 62 transformer blocks
- **Embedding dim:** 5,376
- **Attention:** GQA with 32 query heads and 16 KV heads
- **Attention pattern:** 5 local sliding-window layers for every 1 global full-attention layer
- **FFN hidden dim:** 21,504
- **Context length:** 128k
- **Vocab size:** 262k
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm in a sandwich configuration (pre- and post-norm around each sublayer), plus QK-Norm on Q/K
- **Activation:** GeGLU (GELU-gated)

## What makes Gemma 3 different

- **Sandwich norms.** Each sublayer has *both* a pre-norm and a post-norm: `RMSNorm → Attn → RMSNorm → residual`, and the same around the FFN. This is more stable than pure pre-norm or pure post-norm.
- **5:1 local:global attention.** Most blocks use sliding-window attention (cheap, local context), and every sixth block uses full global attention. Lets the model handle 128k context without quadratic blowup everywhere.
- **QK-Norm.** Queries and keys are RMS-normalized before the attention dot product, keeping attention logits well-scaled.
- **GeGLU FFN.** Uses GELU as the gating activation rather than SiLU (as in Llama/SwiGLU).

## What's in `model.py`

- **`InputEmbeddings`** — token embedding lookup, scaled by √embed_dim.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors, with offset support for KV-cached generation.
- **`GroupedQueryAttention`** — GQA with optional QK-Norm, RoPE applied to Q/K, configurable query pre-attention scaling, and a KV cache that supports both local (sliding-window) and global modes.
- **`FeedForward`** — GeGLU feed-forward: `gelu(ff1(x)) * ff2(x)`, then a down-projection.
- **`SkipConnection`** — residual add.
- **`TransformerBlock`** — sandwich-norm block: `pre-norm → Attn → post-norm → residual`, then `pre-norm → FFN → post-norm → residual`. The `attn_type` flag switches between sliding-window and global attention, and the sliding-window cache is trimmed to the window size after each step.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`Gemma3Model`** — embeds tokens, builds local and global causal masks, stacks 62 transformer blocks (mixed local/global), applies a final RMSNorm, and projects to vocab. Supports KV caching across generation steps.

## References

- [Gemma 3 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786)
- [google/gemma-3-27b-pt config.json](https://huggingface.co/google/gemma-3-27b-pt/blob/main/config.json)