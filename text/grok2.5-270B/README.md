# Grok 2.5 (270B)

xAI's 2024–25 sparse Mixture-of-Experts decoder, open-weighted in mid-2025. 270B total parameters with roughly 62B active per token. Architecturally close to modern Llama-family MoEs (RMSNorm, RoPE, GQA, SwiGLU experts), with one distinctive feature: **residual MoE** — a dense SwiGLU FFN runs in parallel with the MoE layer, and their outputs are summed. This is a much heavier "shared expert" than the single-expert versions in DeepSeek V3 or Kimi K2.

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![Grok 2.5 architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/grok-2-5-270b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 270B total, ~62B active per token
- **Layers:** 64 transformer blocks
- **Embedding dim:** 8,192
- **Attention:** GQA with 64 query heads and 8 KV heads (head dim 128)
- **MoE:** 8 experts, top-2 active per token; expert hidden dim 16,384
- **Dense (residual) FFN hidden dim:** 32,768 — runs in parallel with MoE
- **Context length:** 131k (`original_max_position_embeddings` = 8,192, extended to ~13k in config)
- **Vocab size:** 131,072
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU (in experts and in the parallel dense FFN)

## What makes Grok 2.5 different

- **Residual MoE.** The config flag `residual_moe: true` says it directly: in each transformer block, a full dense SwiGLU FFN (hidden dim 32,768) runs *in parallel* with the sparse MoE layer, and both outputs are added into the residual stream. So every token gets:
  - the routed MoE output (2 of 8 experts, hidden dim 16,384 each), plus
  - the always-on dense FFN output.
  This is architecturally similar to DeepSeek's "1 shared + N routed" pattern, but the shared side is much wider (32,768 vs one expert's 16,384).
- **Few, wide experts.** Only 8 experts total, with a wide 16,384 expert hidden dim. Compare DeepSeek V3 (256 experts × 2,048), Kimi K2 (384 × 2,048), Qwen3-30B (128 × 768) — Grok goes in the opposite direction: fewer specialists, each much bigger.
- **Very sparse routing.** top-2 out of 8 means 25% of experts fire per token. In practice, though, the residual dense path means the *effective* activation is still high (~62B active out of 270B).

## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors, with a small KV cache.
- **`GQA`** — grouped-query attention with 64 query heads / 8 KV heads, RoPE on Q/K, and a `use_cache` path for autoregressive decoding.
- **`MLP`** — SwiGLU feed-forward, used both as the always-on residual dense FFN and as the per-expert module inside MoE.
- **`MOE`** — top-k routed MoE: a linear router produces logits, top-k experts selected per token, outputs combined weighted by softmaxed router scores.
- **`ResidualConnection`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → (MoE + dense FFN) → residual`. The dense FFN and MoE run in parallel and their outputs are summed before the residual add.
- **`GroqModel`** — embeds tokens, stacks 64 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Grok 2.5 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [xai-org/grok-2 (Hugging Face model card)](https://huggingface.co/xai-org/grok-2)
- [xai-org/grok-1 (open-weights precursor)](https://github.com/xai-org/grok-1)
- [Mixture of Experts Architecture in Transformer Models — Machine Learning Mastery](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)