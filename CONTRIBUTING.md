# Contributing to OpenArch

Thanks for your interest in contributing! OpenArch is a learning-focused project — readable code matters more than fast code.

## Ways to help

- Implement a new architecture from [Sebastian Raschka's gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- Write or improve a per-model README
- Fix bugs or improve existing code
- Add tests

## Getting started

1. Fork the repo and clone your fork.
2. Create a branch: `git checkout -b add-<model-name>`.
3. Make your changes.
4. Open a pull request describing what you did and which references you used.

## Adding a new architecture

Open an issue first so we don't duplicate work. Each model lives in its own folder:

```
<model_name>/
├── model.py
└── README.md
```

The per-model README should link to the paper / tech report / `config.json` you implemented against.

## A few simple guidelines

- Prioritize readability over performance.
- Keep one model per `model.py` where possible.
- Cite your sources.
- Be kind in reviews.

That's it for now — more guidelines may come as the project grows. If anything is unclear, just open an issue and ask.
