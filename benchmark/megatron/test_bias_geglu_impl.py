# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Performance benchmark for bias_geglu_impl.

For standard GEGLU, the baseline is Megatron-LM's ``bias_geglu_impl`` when
installed. Its forward helpers use ``jit_fuser``, which is ``torch.compile``
on supported PyTorch versions. QuickGEGLU cases, and machines without
Megatron-LM, use the vendored forward implementation at the end of this file,
also compiled with ``torch.compile``. If a vendor backend cannot compile a
baseline, the affected variant explicitly falls back to the equivalent eager
formula and emits a warning. Such results must be reported as eager-baseline
measurements rather than compared directly with compiled-baseline results.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. One table covers standard GEGLU, QuickGEGLU, and QuickGEGLU with a
linear offset. The operator is forward-only, so there are no backward cases.

``torch.compile`` recompiles per shape/dtype/variant pattern. The default
Dynamo recompile limit is smaller than this benchmark's case count, so the
module raises it; otherwise the baseline could silently fall back to eager.
"""

import importlib.metadata
import os
import warnings

import pytest
import torch

import flag_train

from .. import base, consts

# Token counts and hidden widths used by a fused MLP activation. Last dim is even.
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]

# Enough variants for 7 shapes x 3 activations x 3 dtypes of forward cases
# without Dynamo falling back to eager halfway through the suite.
_DYNAMO_RECOMPILE_LIMIT = 128


def _mcore_version():
    try:
        return importlib.metadata.version("megatron-core")
    except Exception:
        return "not installed"


_MCORE_VERSION = _mcore_version()


def _stabilize_megatron_compile():
    """Keep the compiled baseline working on every backend.

    The vendored baseline always uses ``torch.compile``, even when
    Megatron-LM is absent. Therefore Dynamo must be configured regardless
    of whether Megatron-LM is installed; otherwise the baseline can silently
    fall back to eager after reaching its recompile limit.
    """
    # Inductor's async compile sends each Triton kernel to a worker process
    # and pickles the compiled kernel back. Some vendor Triton builds cannot
    # pickle JITFunction because it carries an RLock. One compile thread keeps
    # compilation in-process; it changes compile time only, not measured time.
    try:
        import torch._inductor.config

        if getattr(torch._inductor.config, "compile_threads", 1) != 1:
            torch._inductor.config.compile_threads = 1
    except ImportError:
        pass

    try:
        import torch._dynamo
    except ImportError:
        return

    if hasattr(torch._dynamo.config, "recompile_limit"):
        current = torch._dynamo.config.recompile_limit
    elif hasattr(torch._dynamo.config, "cache_size_limit"):
        current = torch._dynamo.config.cache_size_limit
    else:
        return
    if current >= _DYNAMO_RECOMPILE_LIMIT:
        return

    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = _DYNAMO_RECOMPILE_LIMIT
    else:
        torch._dynamo.config.cache_size_limit = _DYNAMO_RECOMPILE_LIMIT
    warnings.warn(
        "raised the torch._dynamo recompile limit to "
        f"{_DYNAMO_RECOMPILE_LIMIT} so the torch.compile baseline does not "
        "silently fall back to eager",
        RuntimeWarning,
        stacklevel=2,
    )


_stabilize_megatron_compile()


def _dtypes():
    """Use the same floating-point dtypes as the correctness suite."""
    dtypes = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class BiasGegluImplBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


# Each label becomes part of the benchmark row. The kwargs are accepted by
# both the FlagTrain operator and the vendored baseline.
_VARIANTS = {
    "gelu": {"activation": "gelu"},
    "quick_gelu": {"activation": "quick_gelu"},
    "quick_gelu_offset": {
        "activation": "quick_gelu",
        "linear_offset": 0.5,
    },
}


def bias_geglu_ref(x, bias=None, activation="gelu", linear_offset=0.0):
    """Eager formula used only when a compiled baseline is unavailable."""
    y = x if bias is None else x + bias
    gate, linear = torch.chunk(y, 2, dim=-1)
    if activation == "quick_gelu":
        return (gate * torch.sigmoid(1.702 * gate)) * (linear + linear_offset)
    return (
        gate
        * 0.5
        * (1.0 + torch.tanh(0.79788456 * gate * (1 + 0.044715 * gate * gate)))
    ) * linear


# Remember failed variants so Dynamo does not retry compilation for every row.
_EAGER_BASELINE_VARIANTS = set()


def bias_geglu_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    for variant in _VARIANTS:
        yield (x, bias, variant)


def torch_op(x, bias, variant):
    """Use the compiled baseline, with an explicit per-variant eager fallback."""
    if variant in _EAGER_BASELINE_VARIANTS:
        return bias_geglu_ref(x, bias, **_VARIANTS[variant])

    try:
        if variant == "gelu":
            return _BASELINE_IMPL(x, bias)
        return _vendored_bias_geglu_impl(x, bias, **_VARIANTS[variant])
    except Exception as exc:
        _EAGER_BASELINE_VARIANTS.add(variant)
        warnings.warn(
            f"compiled baseline failed for {variant!r}: "
            f"{type(exc).__name__}: {exc}; using the eager formula for all "
            "subsequent rows of this variant",
            RuntimeWarning,
            stacklevel=2,
        )
        return bias_geglu_ref(x, bias, **_VARIANTS[variant])


def train_op(x, bias, variant):
    """The operator under test."""
    return flag_train.megatron.bias_geglu_impl(x, bias, **_VARIANTS[variant])


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_perf():
    """Benchmark all activation variants for every shape in one table."""
    print(f"\nBaseline: {_BASELINE}")

    # On Ascend, enable parallel scheduling of all blocks to avoid excessive
    # logical cores and reduce scheduling overhead when the logical core count
    # exceeds 65,535.
    if flag_train.vendor_name == "ascend":
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

    bench = BiasGegluImplBenchmark(
        input_fn=bias_geglu_input_fn,
        op_name="bias_geglu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=False,
    )
    bench.set_train(train_op)
    bench.run()


# ===========================================================================
# Appendix: vendored Megatron-LM baseline
#
# Source: NVIDIA Megatron-LM, megatron/core/fusions/fused_bias_geglu.py
#   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_bias_geglu.py
#
# ``jit_fuser`` is replaced by ``torch.compile``. Autograd and backward-only
# helpers are omitted because this benchmark measures forward only. The
# forward formulas are unchanged. This is a performance baseline, not the
# correctness oracle used by the unit tests.
# ===========================================================================

jit_fuser = torch.compile


@jit_fuser
def geglu(y):
    """GEGLU using Megatron-LM's tanh approximation of GELU."""
    y_1, y_2 = torch.chunk(y, 2, -1)
    gate = y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1)))
    return gate * y_2


@jit_fuser
def quick_geglu(y, linear_offset=0.0):
    """QuickGEGLU: quick_gelu(y1) * (y2 + linear_offset)."""
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return y_1 * torch.sigmoid(1.702 * y_1) * (y_2 + linear_offset)


@jit_fuser
def _compiled_bias_geglu(y, bias, activation="gelu", linear_offset=0.0):
    """Compiled bias addition followed by the selected GLU activation."""
    if bias is not None:
        y = y + bias
    if activation == "quick_gelu":
        return quick_geglu(y, linear_offset)
    return geglu(y)


def _vendored_bias_geglu_impl(
    input,
    bias,
    activation="gelu",
    linear_offset=0.0,
):
    """Vendored forward baseline supporting two- and three-dimensional input."""
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    flat = input.view(-1, ori_shape[-1])
    out = _compiled_bias_geglu(flat, bias, activation, linear_offset)
    return out if len(ori_shape) == 2 else out.view(*ori_shape[:-1], -1)


def _load_baseline():
    """Load installed Megatron for standard GEGLU, otherwise use the appendix."""
    if _MCORE_VERSION != "not installed":
        try:
            from megatron.core.fusions.fused_bias_geglu import (
                bias_geglu_impl as installed,
            )
        except ImportError as exc:
            reason = f"megatron-core {_MCORE_VERSION} cannot import it: {exc}"
        else:
            return installed, (
                f"megatron-core {_MCORE_VERSION} for gelu; "
                "vendored compiled implementation for quick_gelu"
            )
    else:
        reason = "Megatron-LM is not installed"

    warnings.warn(
        f"{reason}; using the vendored compiled implementation as the "
        "standard GEGLU benchmark baseline",
        RuntimeWarning,
        stacklevel=2,
    )
    return _vendored_bias_geglu_impl, "vendored compiled implementation"


_BASELINE_IMPL, _BASELINE = _load_baseline()
