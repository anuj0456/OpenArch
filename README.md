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
- Training-time tricks: Multi-token-prediction, latent experts, gated attention

Reading the official model code can be hard because production repos optimize for speed, sharding, and backward compatibility. This repo optimizes for **reading**.

## What's implemented (so far)

> Implementations marked ✅ are usable for forward passes; those marked 🚧 are under construction.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Status</th>
      <th>Model Size</th>
      <th>Normalization</th>
      <th>Positional Encoding</th>
      <th>Attention</th>
      <th>Mixture of Expert</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GPT-2 XL</td>
      <td>✅</td>
      <td>1.5B</td>
      <td>-</td>
      <td>Absolute</td>
      <td>Multi Head Attention</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Llama 2</td>
      <td>✅</td>
      <td>7B</td>
      <td>RMS Norm</td>
      <td>RoPE</td>
      <td>Multi Head Attention</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Llama 3</td>
      <td>✅</td>
      <td>8B</td>
      <td>RMS Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention</td>
      <td>No</td>
    </tr>
    <tr>
      <td>OLMo 2</td>
      <td>✅</td>
      <td>7B</td>
      <td>RMS Norm & QK-Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention</td>
      <td>No</td>
    </tr>
    <tr>
      <td>DeepSeek R1</td>
      <td>✅</td>
      <td>671B</td>
      <td>RMS Norm & QK-Norm</td>
      <td>RoPE</td>
      <td>Multihead Latent Attention</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>Gemma 3</td>
      <td>✅</td>
      <td>27B</td>
      <td>RMS Norm & QK-Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention with Sliding Window</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Mistral 3</td>
      <td>✅</td>
      <td>24B</td>
      <td>RMS Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention with Sliding Window</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Llama 4 Maverick</td>
      <td>✅</td>
      <td>400B</td>
      <td>RMS Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen 3</td>
      <td rowspan="2">✅</td>
      <td>4B</td>
      <td>RMS Norm & QK-Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention</td>
      <td>No</td>
    </tr>
    <tr>
      <td>30B - A3B</td>
      <td>RMS Norm & QK-Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>Kimmi K2</td>
      <td>✅</td>
      <td>1T</td>
      <td>RMS Norm</td>
      <td>RoPE</td>
      <td>Multihead Latent Attention</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>GLM 4.5</td>
      <td>✅</td>
      <td>355B</td>
      <td>RMS Norm & QK-Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention & Multi-Token Prediction </td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>GPT-OSS</td>
      <td>🚧</td>
      <td>20B</td>
      <td>RMS Norm</td>
      <td>RoPE</td>
      <td>Grouped Query Attention with Sliding Window</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>

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