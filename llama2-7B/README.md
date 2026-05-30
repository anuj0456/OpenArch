# Llama 2 (7B)

Meta's 2023 dense decoder — the predecessor to Llama 3, using RMSNorm, RoPE, multi-head attention, and a SwiGLU feed-forward.

![Llama 2 7B architecture diagram](

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/)(<img width="1672" height="941" alt="img" src="https://github.com/user-attachments/assets/23c9cc10-9ace-44e1-a98e-5bcb1502551b" />


## Key specs

- **Parameters:** 7B
- **Layers:** 32 transformer blocks
- **Embedding dim:** 4,096
- **Attention heads:** 32 (multi-head attention)
- **FFN hidden dim:** 11,008
- **Context length:** 4,096
- **Vocab size:** 32,000
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SwiGLU

## What's in `model.py`

A from-scratch PyTorch implementation of the stack shown in the diagram above:

- **`InputEmbedding`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale (no bias, no mean-centering).
- **`RoPE`** — rotary positional embeddings applied to query and key vectors.
- **`MultiHeadAttention`** — masked multi-head self-attention with a causal triangular mask and RoPE applied to Q/K.
- **`FeedForwardNetwork`** — SwiGLU feed-forward: `silu(linear1(x)) * linear2(x)`, then a down-projection.
- **`SkipAttention`** — residual add.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`TransformerBlock`** — pre-norm block: `RMSNorm → MHA → residual → RMSNorm → FFN → residual`.
- **`Llama2Model`** — embeds tokens, stacks 32 transformer blocks, applies a final RMSNorm, and projects to vocab.

## References

- [Llama 2 in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (paper)](https://arxiv.org/pdf/2307.09288)
- [meta-llama/Llama-2-7b-hf config.json](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json)
- [Sebastian Raschka's from-scratch Llama implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama)
