# Nemotron 3 Nano (4B)

NVIDIA's 2025 hybrid **Mamba-2 / Transformer** decoder — the small, **dense** member of the Nemotron 3 Nano family.Most layers are Mamba-2 state-space mixers, with a handful of Grouped-Query Attention layers and squared ReLU feed-forward layers interleaved between them.


![Nemotron 3 Nano 4B architecture diagram](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/nemotron-3-nano-4b.webp)

*Architecture diagram by [Sebastian Raschka](https://sebastianraschka.com/). (Verify/adjust the image filename against the gallery if it 404s.)*

## Key specs

- **Parameters:** ~4B total
- **Layers:** 42, hybrid — 21 Mamba-2, 4 attention, 17 feed-forward
- **Embedding dim:** 3,136
- **Attention:** GQA with 40 query heads and 8 KV heads
- **Mamba-2:** 96 heads, head dim 80, 8 groups, SSM state dim 128, conv kernel 4
- **Feed-forward hidden dim:** 12,544
- **Context length:** 262,144
- **Vocab size:** 131,072
- **Positional encoding:** None — position comes from the Mamba-2 layers
- **Normalization:** RMSNorm (pre-norm)
- **Activation:** squared ReLU in the FFN; SiLU gating in Mamba-2

## What makes Nemotron 3 Nano different

- **Hybrid Mamba-Transformer backbone.** Rather than a stack of identical attention+FFN blocks, each of the 42 layers is a *single* mixer of one of three types. Mamba-2 layers are linear-time in sequence length and cheap for long context; attention layers give precise long-range recall that pure state-space models are weaker at. Interleaving them keeps most of the throughput and context benefit of Mamba while retaining enough attention for recall.
- **NoPE attention.** No RoPE, no learned positional embeddings. In a hybrid model the recurrent Mamba-2 layers already encode order, so the attention layers omit positional encoding entirely — simpler than a standard transformer, not more complex.


## What's in `model.py`

- **`InputEmbedding`** — token embedding lookup.
- **`RMSNorm`** — root-mean-square normalization with a learnable scale.
- **`FeedForwardNetwork`** — non-gated squared-ReLU MLP
- **`GroupedQueryAttention`** — GQA with 40 query heads / 8 KV heads and a causal mask. No positional encoding (NoPE).
- **`Mamba2Block`** — Mamba-2 mixer: a single input projection splits into gate `z`, conv input `xBC`, and per-head timestep `dt`; a causal depthwise Conv1d + SiLU feeds a selective state-space scan with one `A`/`D` per head and grouped `B`/`C`; a gated RMSNorm and output projection close the block.
- **`NemotronBlock`** — one pre-norm residual sublayer that holds a single mixer (Mamba-2, attention, or FFN), chosen by its layer-type character: `x → RMSNorm → mixer → + residual`.
- **`OutputLayer`** — final linear projection to vocab logits.
- **`NemotronModel`** — embeds tokens, builds one `HybridBlock` per character in `hybrid_override_pattern`, applies a final RMSNorm, and projects to vocab.



## References

- [Nemotron 3 Nano in the LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (Hugging Face)](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b)
- [Nemotron 3 Nano technical report (arXiv:2512.20848)](https://arxiv.org/abs/2512.20848) — describes the 30B-A3B MoE variant; the 4B is the dense sibling in the same family
- [Transformers are SSMs: Mamba-2 / SSD (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
- [GQA: Grouped-Query Attention (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)