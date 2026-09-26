"""Smoke test: every text/**/model.py must construct and return logits.

    python tests/smoke_test.py            # from the repo root
    python tests/smoke_test.py --root .   # or point it somewhere else

Exit code 0 if every model builds and returns a tensor, 1 otherwise.
Only torch is needed. This checks that a file *runs*, not that it is correct.
"""
import argparse, ast, glob, importlib.util, os, sys
import torch, torch.nn as nn

# Toy dims, small enough to build a 400B-shaped model on CPU. Constructor
# parameters are matched by name; anything not listed here falls back to 2.
# This is the weak part of the harness -- see the note at the bottom.
V = dict(
    vocab_size=64,
    embed_dim=32, d_model=32, embed_size=32, embedding_dim=32, input_embed=32,
    context_len=16, seq_len=16,
    num_heads=4, n_heads=4,
    head_dim=8,
    num_kv_groups=2, num_groups=2,
    hidden_dim=64,
    num_experts=4, top_k=2,
    q_latent_dim=16, kv_latent_dim=16, latent_dim=16,
    num_transformer_blocks=2, num_attention_blocks=2, transformer_layers=2,
    attention_blocks=2, num_trnfmr_blocks=2, num_transformer_block=2,
    num_attention_block=2, transformer_blocks=2,
    qk_norm=True, qkv_bias=False, dropout=0.0, mask=None,
    forward_token_count=1, mtp_depth=1, mtp_lambda=0.1,
)

# gemma3 takes a single cfg dict and splats it into TransformerBlock, so the
# keys have to match that signature exactly -- an extra key here shows up as a
# model failure that isn't one.
CFG = dict(input_embed=32, context_len=16, head_dim=8, num_kv_groups=2,
           num_heads=4, qk_norm=True, query_pre_attn_scalar=8,
           sliding_window=4)


def load(path):
    name = 'oa_' + path.replace(os.sep, '_').replace('-', '_').replace('.', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ctor_args(path, cls_name):
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == '__init__':
                    return [a.arg for a in fn.args.args if a.arg != 'self']
    return []


def check(path):
    """Return (status, detail). status == 'ok' means it returned a tensor."""
    try:
        mod = load(path)
    except Exception as e:
        return 'import', f'{type(e).__name__}: {e}'

    cls_name = next((n for n in dir(mod)
                     if n.endswith('Model') and isinstance(getattr(mod, n), type)
                     and issubclass(getattr(mod, n), nn.Module)), None)
    if cls_name is None:
        return 'no model class', ''

    kw = {n: (dict(CFG) if n == 'cfg' else V.get(n, 2)) for n in ctor_args(path, cls_name)}
    try:
        model = getattr(mod, cls_name)(**kw)
    except Exception as e:
        return 'construct', f'{type(e).__name__}: {e}'

    x = torch.randint(0, V['vocab_size'], (1, 8))
    last = None
    # Signatures differ: (x), (x, mask), (x, mask, cache). Try in order.
    for call in (lambda: model(x), lambda: model(x, None), lambda: model(x, None, None)):
        try:
            out = call()
        except TypeError as e:
            if 'positional argument' in str(e) or 'missing' in str(e):
                last = e
                continue
            return 'forward', f'TypeError: {e}'
        except Exception as e:
            return 'forward', f'{type(e).__name__}: {e}'
        if not torch.is_tensor(out):
            return 'forward', f'returned {type(out).__name__}, not a tensor'
        return 'ok', f'{cls_name} -> {tuple(out.shape)}'
    return 'forward', f'TypeError: {last}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    # Recursive: text/qwen3-4B & 30B/ nests one level deeper than the rest,
    # so text/*/model.py silently skips two of the fourteen models.
    paths = sorted(glob.glob(os.path.join(args.root, 'text', '**', 'model.py'), recursive=True))
    if not paths:
        print(f'no text/**/model.py under {args.root}', file=sys.stderr)
        return 2

    torch.manual_seed(0)
    results = []
    for path in paths:
        label = os.path.relpath(os.path.dirname(path), os.path.join(args.root, 'text'))
        results.append((label, *check(path)))

    w = max(len(r[0]) for r in results)
    for label, status, detail in results:
        mark = 'PASS' if status == 'ok' else 'FAIL'
        print(f'{mark}  {label:<{w}}  {status:<14} {detail[:100]}')

    failed = [r[0] for r in results if r[1] != 'ok']
    print(f'\n{len(results) - len(failed)}/{len(results)} returned logits '
          f'| torch {torch.__version__} | python {sys.version.split()[0]}')
    if failed:
        print('failing: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())