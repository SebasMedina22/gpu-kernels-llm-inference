"""Python bindings for the CUDA kernels, using ctypes over the C API in src/capi/.

The kernels run directly on PyTorch CUDA tensors: we pass the tensor's device
pointer and PyTorch's current CUDA stream, so nothing is copied and the
kernels are ordered correctly with surrounding PyTorch work.

    import torch, llm_kernels as lk
    a = torch.randn(1024, 1024, device="cuda")
    c = lk.matmul(a, a, variant="register_tiled")

Build the library first (cmake --build build --config Release). Set
LLM_KERNELS_LIB to its path if it is not in build/ or build/Release/.
"""

import ctypes
import os
import pathlib

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LIB_NAMES = ["llm_kernels_py.dll", "libllm_kernels_py.so"]


def _find_library() -> pathlib.Path:
    env = os.environ.get("LLM_KERNELS_LIB")
    if env:
        return pathlib.Path(env)
    for d in [_ROOT / "build" / "Release", _ROOT / "build"]:
        for name in _LIB_NAMES:
            if (d / name).exists():
                return d / name
    raise FileNotFoundError(
        "llm_kernels_py library not found; build it with "
        "'cmake -B build && cmake --build build --config Release' or set LLM_KERNELS_LIB"
    )


_lib = ctypes.CDLL(str(_find_library()))

_p = ctypes.c_void_p
_i = ctypes.c_int
_lib.lk_matmul.argtypes = [_i, _p, _p, _p, _i, _i, _i, _p]
_lib.lk_softmax.argtypes = [_i, _p, _p, _i, _i, _p]
_lib.lk_layernorm.argtypes = [_i, _p, _p, _p, _p, _i, _i, ctypes.c_float, _p]
for _f in (_lib.lk_matmul, _lib.lk_softmax, _lib.lk_layernorm):
    _f.restype = _i
_lib.lk_variant_name.argtypes = [ctypes.c_char_p, _i]
_lib.lk_variant_name.restype = ctypes.c_char_p
_lib.lk_error_string.argtypes = [_i]
_lib.lk_error_string.restype = ctypes.c_char_p


def _variants(op: str) -> dict:
    names, i = {}, 0
    while (name := _lib.lk_variant_name(op.encode(), i)) is not None:
        names[name.decode()] = i
        i += 1
    return names


VARIANTS = {op: _variants(op) for op in ("matmul", "softmax", "layernorm")}


def _variant_id(op: str, variant: str) -> int:
    try:
        return VARIANTS[op][variant]
    except KeyError:
        raise ValueError(f"unknown {op} variant {variant!r}; choose from {list(VARIANTS[op])}") from None


def _check(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous():
            raise TypeError("expected contiguous float32 CUDA tensors")
        if t.device != tensors[0].device:
            raise ValueError("all tensors must be on the same device")


def _raise_on_error(code: int) -> None:
    if code != 0:
        raise RuntimeError(f"CUDA error {code}: {_lib.lk_error_string(code).decode()}")


def _stream() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def matmul(a: torch.Tensor, b: torch.Tensor, variant: str = "register_tiled") -> torch.Tensor:
    """C = A @ B for 2D float32 tensors A [M, K] and B [K, N]."""
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"bad shapes {tuple(a.shape)} @ {tuple(b.shape)}")
    _check(a, b)
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    with torch.cuda.device(a.device):
        _raise_on_error(_lib.lk_matmul(_variant_id("matmul", variant), a.data_ptr(), b.data_ptr(),
                                       c.data_ptr(), m, n, k, _stream()))
    return c


def softmax(x: torch.Tensor, variant: str = "warp_online") -> torch.Tensor:
    """Softmax over the last dimension (any leading shape)."""
    _check(x)
    cols = x.shape[-1]
    rows = x.numel() // cols if cols else 0
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _raise_on_error(_lib.lk_softmax(_variant_id("softmax", variant), x.data_ptr(), y.data_ptr(),
                                        rows, cols, _stream()))
    return y


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5,
               variant: str = "warp") -> torch.Tensor:
    """LayerNorm over the last dimension, like torch.nn.functional.layer_norm."""
    hidden = x.shape[-1]
    if weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError("weight and bias must have shape (hidden,)")
    _check(x, weight, bias)
    rows = x.numel() // hidden if hidden else 0
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _raise_on_error(_lib.lk_layernorm(_variant_id("layernorm", variant), x.data_ptr(),
                                          weight.data_ptr(), bias.data_ptr(), y.data_ptr(),
                                          rows, hidden, eps, _stream()))
    return y
