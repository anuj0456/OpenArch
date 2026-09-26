"""Behavioural checks on every text/**/model.py, one test per model per check.

    pytest tests/test_models.py -q

smoke_test.py only proves a file *runs*. These prove three things the smoke
test can't:

  causal      -- logits at position t don't change when a token at t' > t
                 changes. Catches a missing/wrong attention mask, a sliding
                 window that leaks forward, or an MTP head that peeks ahead.
  finite      -- no NaN/inf in the logits and shape == (B, T, vocab_size).
                 Catches bad RoPE scaling, RMSNorm eps=0, unclamped SwiGLU.
  backward    -- loss.backward() works and every parameter that should get a
                 gradient gets one. Catches params created but never used,
                 in-place ops on leaf tensors, and detach() in the graph.

Models whose construction/forward already fails in smoke_test are skipped
here so the failure is reported once, not four times.
"""
import glob, os, sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test import V, CFG, load, ctor_args  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = sorted(glob.glob(os.path.join(ROOT, 'text', '**', 'model.py'), recursive=True))
IDS = [os.path.relpath(os.path.dirname(p), os.path.join(ROOT, 'text')) for p in PATHS]

B, T = 2, 8


def _build(path):
    """Build the model and a callable forward; skip if smoke_test would fail."""
    import torch.nn as nn
    torch.manual_seed(0)
    try:
        mod = load(path)
    except Exception as e:
        pytest.skip(f'import failed (see smoke_test): {e}')
    cls_name = next((n for n in dir(mod)
                     if n.endswith('Model') and isinstance(getattr(mod, n), type)
                     and issubclass(getattr(mod, n), nn.Module)), None)
    if cls_name is None:
        pytest.skip('no *Model class')
    kw = {n: (dict(CFG) if n == 'cfg' else V.get(n, 2)) for n in ctor_args(path, cls_name)}
    try:
        model = getattr(mod, cls_name)(**kw)
    except Exception as e:
        pytest.skip(f'construct failed (see smoke_test): {e}')
    model.train(False)

    def fwd(x):
        last = None
        for call in (lambda: model(x), lambda: model(x, None), lambda: model(x, None, None)):
            try:
                return call()
            except TypeError as e:
                if 'positional argument' in str(e) or 'missing' in str(e):
                    last = e
                    continue
                raise
        raise last
    return model, fwd


@pytest.fixture(scope='module', params=PATHS, ids=IDS)
def built(request):
    return _build(request.param)


def test_finite_and_shape(built):
    _, fwd = built
    x = torch.randint(0, V['vocab_size'], (B, T))
    with torch.no_grad():
        out = fwd(x)
    assert torch.is_tensor(out), f'returned {type(out).__name__}'
    assert out.shape == (B, T, V['vocab_size']), out.shape
    assert torch.isfinite(out).all(), 'NaN/inf in logits'


def test_causal(built):
    _, fwd = built
    torch.manual_seed(1)
    x = torch.randint(0, V['vocab_size'], (1, T))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % V['vocab_size']      # only the LAST token differs
    with torch.no_grad():
        a, b = fwd(x), fwd(y)
    # Everything before the last position must be identical.
    diff = (a[0, :-1] - b[0, :-1]).abs().max().item()
    assert diff < 1e-5, f'future token leaked into earlier logits (max diff {diff:.3e})'
    # And the last position should actually see its own input (sanity: mask
    # isn't accidentally blanking everything).
    assert not torch.allclose(a[0, -1], b[0, -1]), 'last-position logits ignore the input token'


def test_batch_independence(built):
    """Row 0 of a batch must not depend on row 1 (catches wrong reshape/view)."""
    _, fwd = built
    torch.manual_seed(2)
    x = torch.randint(0, V['vocab_size'], (2, T))
    y = x.clone()
    y[1] = torch.randint(0, V['vocab_size'], (T,))
    with torch.no_grad():
        a, b = fwd(x), fwd(y)
    diff = (a[0] - b[0]).abs().max().item()
    assert diff < 1e-5, f'batch rows leak into each other (max diff {diff:.3e})'


def test_backward(built):
    model, fwd = built
    model.train(True)
    x = torch.randint(0, V['vocab_size'], (B, T))
    tgt = torch.randint(0, V['vocab_size'], (B, T))
    out = fwd(x)
    loss = torch.nn.functional.cross_entropy(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss), 'loss is NaN/inf'
    # Params with no grad are usually a bug (module defined but never called).
    # Exceptions: MoE experts that simply weren't routed to at this toy scale,
    # and MTP heads when depth=1. Report but only fail on non-expert params.
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None and 'expert' not in n.lower()]
    assert not dead, f'{len(dead)} param(s) got no gradient: {dead[:5]}'
    model.train(False)
