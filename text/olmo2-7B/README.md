# OLMo 2 (7B)

Allen AI's 2024 fully-open dense decoder. Architecturally close to Llama, but distinguished by two choices: **post-norm** (RMSNorm applied to each sublayer's output before the residual add, rather than to its input) and **QK-Norm** (normalizing queries and keys before attention). Both improve training stability.

![OLMo 2 7B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/olmo-2-7b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/).*

## Key specs

- **Parameters:** 7B
- **Layers:** 32 transformer blocks
- **Embedding dim:** 4,096
- **Attention heads:** 32 (regular multi-head attention, 32 KV heads)
- **FFN hidden dim:** 11,008
- **Context length:** 4,096
- **Vocab size:** 100k
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm, applied **post-sublayer**; plus QK-Norm on queries and keys
- **Activation:** SwiGLU

## What makes OLMo 2 different

- **Post-norm.** Unlike the pre-norm used in Llama/GPT, OLMo 2 normalizes the *output* of attention and the feed-forward before adding the residual: `x = x + Norm(Attn(x))`. The norm sits between the sublayer and the skip connection.
- **QK-Norm.** Queries and keys are RMS-normalized before the attention dot product, which keeps attention logits well-scaled.

## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup.
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`MultiHeadAttention`** — multi-head self-attention with QK-Norm applied to queries/keys and RoPE on Q/K.
- **`PostRMSNorm`** — RMS normalization applied after each sublayer.
- **`FeedForwardNetwork`** — SwiGLU feed-forward: `silu(linear1(x)) * linear2(x)`, then a down-projection.
- **`SkipConnection`** — residual add.
- **`TransformerBlock`** — post-norm block: `Attn → Norm → residual`, then `FFN → Norm → residual`.
- **`OLMO2Model`** — embeds tokens, stacks 32 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [OLMo 2 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [2 OLMo 2 Furious (tech report)](https://arxiv.org/pdf/2501.00656)
- [allenai/OLMo-2-1124-7B config.json](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json)