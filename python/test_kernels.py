"""Correctness of every kernel variant against the PyTorch equivalent.

Run with:  python python/test_kernels.py     (or: pytest python/test_kernels.py)
"""

import math

import torch

import llm_kernels as lk

torch.manual_seed(0)
# Plain FP32 matmul in PyTorch (no TF32), so both sides do the same arithmetic.
torch.backends.cuda.matmul.allow_tf32 = False


def test_matmul():
    for m, n, k in [(1, 1, 1), (128, 128, 128), (257, 129, 65), (1000, 777, 333), (2048, 2048, 2048)]:
        a = torch.rand(m, k, device="cuda") * 2 - 1
        b = torch.rand(k, n, device="cuda") * 2 - 1
        ref = a @ b
        for variant in lk.VARIANTS["matmul"]:
            out = lk.matmul(a, b, variant)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5 * k,
                                       msg=lambda s: f"matmul {variant} {m}x{n}x{k}: {s}")


def test_softmax():
    for shape in [(1, 1), (3, 31), (100, 1000), (2, 8, 128, 128), (64, 32000)]:
        x = torch.randn(*shape, device="cuda") * 10
        x[..., ::7] += 1000.0  # large values: needs the max subtraction
        x[..., shape[-1] // 2 + 1:] = -math.inf  # masked half (column 0 stays unmasked)
        ref = torch.softmax(x, dim=-1)
        for variant in lk.VARIANTS["softmax"]:
            out = lk.softmax(x, variant)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6,
                                       msg=lambda s: f"softmax {variant} {shape}: {s}")


def test_layer_norm():
    for shape in [(1, 1), (3, 31), (17, 768), (2, 128, 4096), (8, 20000)]:
        hidden = shape[-1]
        x = torch.randn(*shape, device="cuda") + 5.0
        w = torch.rand(hidden, device="cuda") + 0.5
        b = torch.randn(hidden, device="cuda")
        ref = torch.nn.functional.layer_norm(x, (hidden,), w, b, eps=1e-5)
        for variant in lk.VARIANTS["layernorm"]:
            out = lk.layer_norm(x, w, b, eps=1e-5, variant=variant)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4,
                                       msg=lambda s: f"layer_norm {variant} {shape}: {s}")


def test_stream_ordering():
    """Kernels run on PyTorch's current stream, ordered with torch ops on it."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        a = torch.randn(512, 512, device="cuda")
        out = lk.matmul(a, a)  # must see `a` fully written on stream s
        ref = a @ a
    s.synchronize()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5 * 512)


def test_bad_inputs():
    a = torch.randn(4, 4, device="cuda")
    for bad in (a.double(), a.cpu(), a.t()):
        try:
            lk.softmax(bad)
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError")
    try:
        lk.softmax(a, variant="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert lk.softmax(torch.empty(0, 8, device="cuda")).shape == (0, 8)


if __name__ == "__main__":
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("variants:", lk.VARIANTS)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: PASS")
    print("ALL TESTS PASSED")
