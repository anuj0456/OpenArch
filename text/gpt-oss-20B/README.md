# GPT-OSS (20B)

OpenAI's August 2025 open-weights MoE — their **first open-weights release since GPT-2**, under Apache 2.0. 20B total parameters with 3.6B active per token. Architecturally, GPT-OSS combines several modern ideas: **alternating banded sliding-window and full attention** (Gemma 3 style), **learned attention sinks** (a per-head bias inside the softmax denominator), and **sparse MoE without a shared expert** (Qwen3-MoE style, but with sigmoid-after-topk routing).

<!--
  To display the architecture diagram without committing the PNG to the repo:
  drag-drop the PNG into a GitHub issue comment, copy the generated
  https://github.com/user-attachments/... URL, and paste it below.
-->
![GPT-OSS 20B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/gpt-oss-20b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 20B total, ~3.6B active per token
- **Layers:** 24 transformer blocks
- **Embedding dim:** 2,880
- **Attention:** GQA with 64 query heads and 8 KV heads (head dim 64)
- **Attention pattern:** alternating banded sliding-window (window=128) and full attention
- **MoE:** 32 experts, **no shared expert**, top-4 routed per token; expert hidden dim 2,880
- **Context length:** 128k (RoPE + YaRN scaling)
- **Vocab size:** 201,088 (the o200k tokenizer used by GPT-4o)
- **Positional encoding:** RoPE with YaRN
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU (with clamping and a residual connection inside the FFN)

## What makes GPT-OSS different

- **Learned attention sinks.** Each attention head has a learned scalar bias appended to the softmax denominator. This lets a head "pay no attention to any tokens" if it wants to — useful for heads that don't need to attend on a given step. Implementation is a per-head learnable parameter, *not* a sink token added to the sequence.
- **Banded + full alternating attention.** Like Gemma 3, layers alternate between a cheap local-window attention (window=128 tokens) and a full-context layer. This keeps long-context costs in check while preserving the model's ability to attend globally where needed.
- **Top-k then softmax-over-selected MoE routing.** The router produces logits for all 32 experts, top-4 are selected, and softmax is applied *only over those 4*. (Compare DeepSeek V3, which uses sigmoid gates across all experts; Qwen3-MoE, which softmaxes over all then takes top-k.)
- **No shared expert.** Unlike DeepSeek V3 and Kimi K2, GPT-OSS goes the Qwen3-MoE route: all experts are routed, none always-on.
- **Wide-not-deep.** 24 layers with embedding dim 2,880 — wider and shallower than comparable models (Qwen3-30B-A3B has 48 layers at embed dim 2,048). Sebastian Raschka's writeup compares this directly.
- **First OpenAI open-weights release since GPT-2 (2019).** Apache 2.0 license.

## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`GroupedQueryAttention`** — GQA with 64 query heads and 8 KV heads; RoPE applied to Q/K.
- **`ResidualConnection`** — residual add.
- **`MLP`** — SwiGLU feed-forward expert: `silu(ff1(x)) * ff2(x)`, then a down-projection.
- **`MOE`** — top-k routed MoE: a linear router produces logits over experts, top-k are selected per token, outputs are combined weighted by softmaxed router scores.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBock`** — pre-norm block: `RMSNorm → GQA → residual → RMSNorm → MoE → residual`.
- **`GPTOSSModel`** — embeds tokens, stacks 24 transformer blocks, applies a final RMSNorm, and projects to vocab.

> Note: this implementation simplifies several GPT-OSS specifics — attention sinks, banded/full alternation, YaRN scaling, and SwiGLU clamping are not yet wired in. See the references below for full details.

## References

- [GPT-OSS in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [gpt-oss-120b & gpt-oss-20b Model Card (OpenAI, Aug 2025)](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
- [openai/gpt-oss-20b config.json](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)