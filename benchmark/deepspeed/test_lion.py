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
"""Performance benchmark for the multi_tensor_lion operator.

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below:

* on a listed backend, DeepSpeed's ``multi_tensor_lion`` **is** the baseline.
  If it cannot be built there the module raises rather than falling back --
  measuring the torch reference and reporting it under the same name would
  answer a different question;
* on any other backend the baseline is ``lion_ref``, the plain-torch
  composition of the same contract, whether or not ``deepspeed`` is installed
  there. It is not a competitor, and the speedup there says how far the kernel
  is from *a* correct implementation rather than from the best one.

The benchmark times repeated steps on a fixed tensor set, which is the
training-loop pattern: the operator's device-side metadata is built on the
first call and served from the cache on every later call, so the timed region
is the single fused kernel launch. DeepSpeed's op rebuilds its host-side
pointer arrays on every call; that asymmetry is the optimization being
measured, not a benchmark artifact.
"""

import pytest
import torch

import flag_train
from flag_train.deepspeed import multi_tensor_lion

from .. import base

# (n_tensors, numel) cases: small tensor counts with tiny/large tensors up to
# a realistic training scale (256 tensors x 1M elements).
_LION_SHAPES = [
    (8, 1024),
    (64, 1024),
    (256, 1024),
    (1024, 1024),
    (8, 1048576),
    (64, 1048576),
    (256, 1048576),
]

# Fixed hyper-parameters shared by both implementations so the comparison is
# apples-to-apples. These mirror DeepSpeed's own FusedLion defaults.
_LR = 1e-4
_BETA1 = 0.9
_BETA2 = 0.99
_WEIGHT_DECAY = 0.01
_CHUNK_SIZE = 2048
_STEP = 1

# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be timed against a plain-torch composition of the
# same contract on any device. The DeepSpeed baseline needs the ``deepspeed``
# package and a CUDA device, so without a torch reference the operator would be
# unbenchmarkable off NVIDIA.
# ---------------------------------------------------------------------------


def lion_ref(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    step,
    weight_decay,
):
    """Reference for multi_tensor_lion, composed from plain torch ops."""
    g_list, p_list, m_list = tensor_lists
    for g, p, m in zip(g_list, p_list, m_list):
        gf = g.to(torch.float32)
        pf = p.to(torch.float32)
        mf = m.to(torch.float32)
        c = beta1 * mf + (1 - beta1) * gf
        upd = torch.where(c > 0, torch.full_like(c, -lr), torch.full_like(c, lr))
        p_new = pf * (1 - lr * weight_decay) + upd
        m_new = beta2 * mf + (1 - beta2) * gf
        p.copy_(p_new.to(p.dtype))
        m.copy_(m_new.to(m.dtype))


# Backends whose baseline is DeepSpeed. multi_tensor_lion ships as a CUDA op
# builder, so only a backend that can compile and execute one can host it. On
# these the baseline is not optional -- if it will not load, timing the torch
# reference and reporting it under the DeepSpeed baseline's name would answer
# another question.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_lion():
    """DeepSpeed's multi_tensor_lion, or ``None`` on other backends.

    ``FusedLionBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under
    ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.op_builder import FusedLionBuilder

        return FusedLionBuilder().load().multi_tensor_lion
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's multi_tensor_lion "
            f"as its baseline, but it could not be loaded: {exc!r}. Build "
            f"deepspeed, or drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


# Resolved once, so the first-use JIT compile is not counted in the measurement.
_deepspeed_lion = _load_deepspeed_lion()

_BASELINE = (
    "deepspeed multi_tensor_lion" if _deepspeed_lion is not None else "lion_ref (torch)"
)


class LionBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _LION_SHAPES

    def set_shapes(self, shape_file=None):
        # lion needs 3 lists of tensors per case (grads, params, exp_avgs);
        # keep the (n_tensors, numel) cases explicit to avoid OOM on CI GPUs.
        self.shapes = list(_LION_SHAPES)

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        # The inputs are (grads, params, exp_avgs) lists of n_tensors tensors;
        # the default deep_parse would print every tensor's shape, which is
        # unreadable at 1024 tensors. Compress to the case description.
        gs = args[0]
        return f"{len(gs)} tensors x {list(gs[0].shape)}"


def lion_input_fn(shape, dtype, device):
    n_tensors, numel = shape
    gs = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    ps = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    ms = [torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    yield gs, ps, ms


_NOOP_FLAG = None


def _noop_flag():
    global _NOOP_FLAG
    if _NOOP_FLAG is None:
        _NOOP_FLAG = torch.zeros(1, dtype=torch.int32)
    return _NOOP_FLAG


def _call(op, gs, ps, ms):
    return op(
        _CHUNK_SIZE,
        _noop_flag(),
        [gs, ps, ms],
        _LR,
        _BETA1,
        _BETA2,
        _STEP,
        _WEIGHT_DECAY,
    )


def torch_op(gs, ps, ms):
    """Baseline, chosen by platform. See the module docstring."""
    baseline = _deepspeed_lion if _deepspeed_lion is not None else lion_ref
    return _call(baseline, gs, ps, ms)


def train_op(gs, ps, ms):
    """The operator under test."""
    return _call(multi_tensor_lion, gs, ps, ms)


@pytest.mark.multi_tensor_lion
def test_lion_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = LionBenchmark(
        input_fn=lion_input_fn,
        op_name="multi_tensor_lion",
        torch_op=torch_op,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )
    bench.set_train(train_op)
    bench.run()
