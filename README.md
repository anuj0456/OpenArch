# OpenArch

> Python implementations of modern open-source LLM architectures — written from scratch, one model at a time.

This repository contains hand-written PyTorch implementations of the model architectures cataloged in Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/). Each model is implemented to the best of my knowledge from the original papers, technical reports, reference `config.json` files, and the excellent writeups by Sebastian Raschka and Machine Learning Mastery.

The goal is not to compete with `transformers` or other production libraries. The goal is **clarity and learning**: a single readable file per architecture, with the structural choices (attention type, normalization, layer mix, MoE routing, positional encoding) made explicit and easy to compare side-by-side.

## Why this repo?

Modern LLM architectures share a common skeleton but differ in dozens of small, important choices:

- Attention: MHA, GQA, MQA, MLA, sliding-window, linear/DeltaNet hybrids
- Normalization: pre-norm, post-norm, QK-Norm, sandwich norm, RMSNorm
- Positional encodings: RoPE, NoPE, partial RoPE, YaRN
- Decoder type: dense vs sparse MoE (with or without shared experts), hybrid Mamba/attention
- Training-time tricks: MTP, latent experts, gated attention

Reading the official model code can be hard because production repos optimize for speed, sharding, and backward compatibility. This repo optimizes for **reading**.

## What's implemented (so far)

> ⚠️ Work in progress. Implementations marked ✅ are usable for forward passes; those marked 🚧 are under construction.

| Model | Status | Decoder | Attention |
|---|---|---|---|
| GPT-2 XL | ✅ | Dense | MHA |
| Llama 3 | ✅ | Dense | GQA + RoPE |
| OLMo 2 | 🚧 | Dense | MHA/GQA + QK-Norm |

The full target list mirrors the 72 architectures in the Architecture Gallery. Contributions toward any of them are welcome.

## Repository layout

```
OpenArch/
├── gpt2/
│   ├── model.py
│   └── README.md
├── llama3/
├── qwen3/
├── deepseek_v3/
├── README.md
└── requirements.txt


```

Each model lives in its own folder with respective `model.py` and a short `README.md` describing the architectural choices and references used.

## Contributing

**I am actively looking for contributors.** If you enjoy reading model papers, comparing `config.json` files, or just want to deepen your understanding of how modern LLMs are built, this is a friendly place to start.

Good first contributions:

- Pick an unimplemented model from the gallery and add a `model.py` for it
- Add a `README.md` for an existing model documenting its architectural choices
- Add a forward-pass test that loads the official weights and matches outputs on a few tokens
- Fix bugs, improve docstrings, or refactor shared components

Please open an issue before starting a large piece of work so we can avoid duplicating effort. Implementations should prioritize **readability over performance** — this is a learning resource first.

See `CONTRIBUTING.md` for more details.

## Acknowledgements

This repository would not exist without the work of two outstanding educators:

- **[Sebastian Raschka](https://sebastianraschka.com/)** — for the [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), the [Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) series, and the [LLMs From Scratch](https://github.com/rasbt/LLMs-from-scratch) book and codebase. The architecture diagrams, fact sheets, and side-by-side comparisons in the gallery are the primary reference behind every model in this repo.
- **Jason Brownlee and the team at [Machine Learning Mastery](https://machinelearningmastery.com/)** — for years of clear, accessible tutorials that have helped countless practitioners (myself included) build a working understanding of deep learning and transformer architectures from the ground up.

Any errors in the implementations here are entirely my own.

## License

This project is licensed under the Apache License 2.0 — see `LICENSE` for details. Individual model implementations follow the licenses of the original models where applicable; see each model's folder for specifics.

## Disclaimer

These implementations are written to the best of my knowledge based on publicly available papers, technical reports, configuration files, and educational material. They are intended as a **learning resource** and are not affiliated with or endorsed by the original model authors. For production use, please use the official implementations or `transformers`.