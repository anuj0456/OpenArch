# Llama 4 Maverick (400B)

Meta's 2025 sparse Mixture-of-Experts decoder. 400B total parameters, but only ~17B active per token. Architecturally a Llama-style stack (RMSNorm, RoPE, GQA, SwiGLU experts) with two distinctive choices: **alternating dense and MoE blocks** (every other block uses a single dense SwiGLU FFN instead of MoE), and a **1 shared + 1 routed expert** MoE pattern across 128 experts — very sparse top-k routing.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Llama 4 Maverick architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/llama-4-maverick-400b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 400B total, ~17B active per token
- **Layers:** 48 transformer blocks (alternating dense / MoE every other block)
- **Embedding dim:** 5,120
- **Attention:** GQA + RoPE
- **MoE:** 128 routed experts + 1 shared expert, 1 routed expert active per token; expert hidden dim 8,192
- **Dense FFN hidden dim:** 16,384 (used in the non-MoE alternating blocks)
- **Context length:** 1M tokens
- **Vocab size:** 202k
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU (in experts and in the dense FFN)

## What makes Llama 4 Maverick different

- **Alternating dense / MoE blocks.** Every other transformer block uses a regular dense SwiGLU FFN (hidden dim 16,384) instead of an MoE layer. This balances the routing capacity of MoE with the more reliable representational density of dense blocks.
- **Very sparse routing.** Only 1 of the 128 routed experts is active per token (plus 1 always-on shared expert), making each forward pass cheap relative to the 400B total parameter count.
- **1M context.** Trained for extreme long-context use, much longer than Llama 3's 128k.

## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup, scaled by √embedding_dim.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupedQueryAttention`** — GQA with RoPE on Q/K and a built-in causal mask helper.
- **`Expert`** — a single SwiGLU feed-forward expert: `silu(linear1(x)) * linear2(x)`, then a down-projection.
- **`MOE`** — top-k routed MoE: a linear router produces logits, top-k experts are selected per token, their outputs are combined weighted by softmaxed router scores.
- **`SkipConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → MoE → residual`. The model alternates this with a dense-FFN variant.
- **`Llama4MaverickModel`** — embeds tokens, stacks 48 transformer blocks alternating dense and MoE, applies a final RMSNorm, and projects to vocab.

## References

- [Llama 4 Maverick in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Llama 4 announcement (Meta AI)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [meta-llama/Llama-4-Maverick-17B-128E-Instruct config.json](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/main/config.json)
- [Mixture of Experts Architecture in Transformer Models — Machine Learning Mastery](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)