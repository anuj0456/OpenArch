# GPT-2 XL (1.5B)

OpenAI's late-2019 dense decoder, included as a reference baseline for how much LLM architectures have changed since.

![GPT-2 XL architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/gpt-2-xl.webp)

*Architecture diagram from [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/).*

## Key specs

- **Parameters:** 1.5B
- **Layers:** 48 transformer blocks
- **Embedding dim:** 1,600
- **Attention heads:** 25 (MHA)
- **FFN hidden dim:** 6,400
- **Context length:** 1,024
- **Vocab size:** 50,257
- **Positional encoding:** learned absolute
- **Normalization:** LayerNorm (pre-norm)
- **Activation:** GELU

## What's in `model.py`

A from-scratch PyTorch implementation of the stack shown in the diagram above:

- **`InputEmbedding`** — token embedding scaled by √d_model.
- **`PositionalEncoding`** — sinusoidal positional encoding added to embeddings. *Note: the original GPT-2 uses learned absolute position embeddings; this implementation uses sinusoidal as a simpler stand-in.*
- **`LayerNorm`** — standard LayerNorm with learnable scale and bias.
- **`MultiHeadAttention`** — masked multi-head self-attention with a causal triangular mask.
- **`GELU`** — tanh approximation of GELU, as used in the original GPT-2.
- **`FeedForward`** — two linear layers with GELU in between (the expanded hidden block on the right of the diagram).
- **`TransformerBlock`** — pre-norm block: `LayerNorm → MHA → residual → LayerNorm → FFN → residual`.
- **`GPT2Model`** — embeds tokens, adds positional encodings, stacks 48 transformer blocks, applies a final LayerNorm, and projects to vocab via a linear output head.

## References

- [GPT-2 XL in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/#card-gpt-2-xl-1-5b)
- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [openai-community/gpt2-xl config.json](https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json)
- [Umar Jamil - https://github.com/hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)