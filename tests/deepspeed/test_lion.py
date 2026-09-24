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
"""Correctness tests for the multi_tensor_lion operator.

Two oracles, because they fail differently:

* ``lion_ref`` -- a plain-torch composition of the same contract. It is the
  primary check and runs on any device, so the operator is testable off NVIDIA.
  Under ``--ref cpu`` it is fed CPU operands and genuinely executes there, which
  makes it an independent oracle rather than a second reading of the same device
  arithmetic; see ``_reference_copy``.
* DeepSpeed's ``multi_tensor_lion`` -- the operator this implementation ports.
  It is an independent implementation and its version is recorded in
  ``_DEEPSPEED_VERSION``, as tests/deepspeed/README.md asks for, but it needs
  the ``deepspeed`` package, and on a backend in ``_DEEPSPEED_BASELINE_VENDORS``
  below it is *required* -- if it will not load there the module raises rather
  than losing the check quietly. On other backends it is an additional check
  that is skipped.

Checking both is not redundant: ``lion_ref`` is the same arithmetic written
from the same reading of the kernel, so a shared misreading of the contract
would survive it. DeepSpeed's operator is the only oracle that can disagree
with that reading.
"""

import pytest
import torch

import flag_train
from flag_train.deepspeed import multi_tensor_lion

from .. import accuracy_utils as utils

# Hyper-parameters shared by both implementations, mirroring the values
# DeepSpeed's FusedLion is exercised with.
_LR = 1e-4
_BETA1 = 0.9
_BETA2 = 0.99
_WEIGHT_DECAY = 0.01
_CHUNK_SIZE = 2048

# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be checked against a plain-torch composition of
# the same contract on any device. The DeepSpeed oracle needs the ``deepspeed``
# package and a CUDA device, so without a torch reference the operator would be
# untestable off NVIDIA.
# ---------------------------------------------------------------------------

def _reference_copy(tensor):
    """An independent copy of ``tensor``, on the device the reference must use.

    ``--ref cpu`` pushes the reference *computation* onto the CPU, not merely its
    result, so every operand ``lion_ref`` is handed comes from here and no kernel
    of the reference runs on the accelerator. That is what makes it an
    independent oracle -- a reference evaluated by the same device arithmetic it
    is checking can only confirm the arithmetic agrees with itself.
    ``lion_ref`` checks this rather than trusting it; the comparison helpers move
    the operator's own output across afterwards.

    ``lion_ref`` steps its parameters in place, so the copy is not optional: on a
    non-CPU reference ``to_reference`` returns its argument unchanged, and the
    reference would then corrupt the operator's operands.
    """
    return utils.to_reference(tensor).clone()

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
    """Reference for multi_tensor_lion, composed from plain torch ops.

    Steps ``p``/``m`` in place with fp32 intermediate math, matching the
    operator's contract including the c == 0 -> +lr tie-break, since the two
    are compared bit-for-bit-ish rather than through a fuzzy formula.
    """
    if utils.TO_CPU:
        # ``--ref cpu`` promises the reference *computation* is on the CPU, not
        # only its result. Asserting the precondition here rather than at the
        # call sites means a test that forgets ``_reference_copy`` fails loudly
        # instead of quietly becoming a second reading of the device arithmetic.
        for name, tensors in zip(("grads", "params", "exp_avgs"), tensor_lists):
            for operand in tensors:
                assert operand.device.type == "cpu", (
                    f"--ref cpu must run the reference on the CPU; a {name} "
                    f"operand is on {operand.device}"
                )

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

_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's multi_tensor_lion reference is unavailable; install the "
    "deepspeed package on a CUDA host to run this check."
)

# Backends whose reference is DeepSpeed. multi_tensor_lion ships as a CUDA op
# builder, so only a backend that can compile and execute one can host it. On
# these the reference is not optional -- a missing one is an environment fault,
# and skipping quietly would thin the suite without saying so.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}

def _load_deepspeed_lion():
    """``(op, version)`` for DeepSpeed's multi_tensor_lion, or ``(None, None)``.

    ``FusedLionBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under
    ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import FusedLionBuilder

        return FusedLionBuilder().load().multi_tensor_lion, deepspeed.__version__
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's multi_tensor_lion "
            f"as its reference, but it could not be loaded: {exc!r}. Build "
            f"deepspeed, or drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc

# Resolved once, at module import time.
_deepspeed_lion, _DEEPSPEED_VERSION = _load_deepspeed_lion()

# The torch reference is always available, so only the DeepSpeed checks skip.
requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_lion is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)

def _noop_flag():
    """The flag DeepSpeed's op takes but (with the check commented out) ignores."""
    return torch.zeros(1, dtype=torch.int32)

def _step(op, g_list, p_list, m_list, step):
    """Run one in-place multi-tensor step through ``op``.

    Every implementation updates its tensors in place, so callers pass tensors
    they own.
    """
    return op(
        _CHUNK_SIZE,
        _noop_flag(),
        [g_list, p_list, m_list],
        _LR,
        _BETA1,
        _BETA2,
        step,
        _WEIGHT_DECAY,
    )

def _make_triplet(n_tensors, numel, dtype, m_init="randn"):
    """Fresh (grads, params, exp_avgs) sharing one dtype."""
    device = flag_train.device
    gs = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    ps = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    if m_init == "zeros":
        ms = [
            torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n_tensors)
        ]
    else:
        ms = [
            torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)
        ]
    return gs, ps, ms

def _clone_lists(gs, ps, ms):
    """Every implementation steps in place, so each runs on its own copies."""
    return ([t.clone() for t in gs], [t.clone() for t in ps], [t.clone() for t in ms])

def _reference_lists(gs, ps, ms):
    """Copies on the device the reference must use (the CPU under --ref cpu)."""
    return (
        [_reference_copy(t) for t in gs],
        [_reference_copy(t) for t in ps],
        [_reference_copy(t) for t in ms],
    )

def _ds_atol(dtype):
    """Atol for comparisons against the DeepSpeed oracle. With fp16/bf16 the
    two independent builds can contract the fp32 FMAs differently and land on
    opposite sides of a rounding boundary, flipping the last ulp on isolated
    elements. The torch-reference checks pin our arithmetic exactly, so a
    couple-of-ulps atol is appropriate for this cross-implementation check."""
    return 1e-4 if dtype == torch.float32 else 2e-3

def _assert_step_matches(train_ps, train_ms, other_ps, other_ms, atol=1e-4):
    """Both the updated params and the updated moments must agree, per tensor."""
    for train_p, other_p in zip(train_ps, other_ps):
        utils.train_assert_close(
            utils.to_reference(train_p),
            utils.to_reference(other_p),
            train_p.dtype,
            atol=atol,
        )
    for train_m, other_m in zip(train_ms, other_ms):
        utils.train_assert_close(
            utils.to_reference(train_m),
            utils.to_reference(other_m),
            train_m.dtype,
            atol=atol,
        )

@pytest.mark.multi_tensor_lion
@pytest.mark.parametrize("n_tensors,numel", [(1, 1024), (4, 4096), (16, 16384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lion(n_tensors, numel, dtype):
    """A single Lion step must match the torch reference, and DeepSpeed's
    multi_tensor_lion when it is available."""
    gs, ps, ms = _make_triplet(n_tensors, numel, dtype)

    train_gs, train_ps, train_ms = _clone_lists(gs, ps, ms)
    _step(multi_tensor_lion, train_gs, train_ps, train_ms, 1)

    ref_gs, ref_ps, ref_ms = _reference_lists(gs, ps, ms)
    _step(lion_ref, ref_gs, ref_ps, ref_ms, 1)
    _assert_step_matches(train_ps, train_ms, ref_ps, ref_ms)

    if _deepspeed_lion is not None:
        ds_gs, ds_ps, ds_ms = _clone_lists(gs, ps, ms)
        _step(_deepspeed_lion, ds_gs, ds_ps, ds_ms, 1)
        _assert_step_matches(train_ps, train_ms, ds_ps, ds_ms, atol=_ds_atol(dtype))

@pytest.mark.multi_tensor_lion
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lion_matches_reference(dtype):
    """Run several Lion steps and compare against the torch reference.

    Multiple parameter tensors, fresh gradients each step, accumulating first
    moments, with the operator under test compared to the reference after all
    steps. Repeating the call on the same tensor set also exercises the
    metadata cache on every step after the first.
    """
    torch.manual_seed(0)
    n_tensors, numel, n_steps = 4, 4096, 5

    _, ps, ms = _make_triplet(n_tensors, numel, dtype, m_init="zeros")
    train_ps = [p.clone() for p in ps]
    train_ms = [m.clone() for m in ms]
    # Same values as the operator's operands -- ``to_reference`` copies rather
    # than converts -- but on whichever device the reference is meant to run.
    ref_ps = [_reference_copy(p) for p in ps]
    ref_ms = [_reference_copy(m) for m in ms]

    for step in range(1, n_steps + 1):
        grads = [
            torch.randn((numel,), dtype=dtype, device=flag_train.device)
            for _ in range(n_tensors)
        ]
        # Lion reads the gradient but never writes it, so the operator and the
        # reference may be handed the same values.
        _step(multi_tensor_lion, grads, train_ps, train_ms, step)
        _step(lion_ref, [_reference_copy(g) for g in grads], ref_ps, ref_ms, step)

    _assert_step_matches(train_ps, train_ms, ref_ps, ref_ms)

@pytest.mark.multi_tensor_lion
def test_lion_mixed_dtypes():
    """A single call may mix dtypes; the implementation groups by dtype and
    launches once per group (mirroring FusedLion.step). DeepSpeed's op itself
    is always called per dtype group, so the torch reference is the oracle."""
    gs32, ps32, ms32 = _make_triplet(2, 4096, torch.float32)
    gs16, ps16, ms16 = _make_triplet(2, 4096, torch.float16)
    gs, ps, ms = gs32 + gs16, ps32 + ps16, ms32 + ms16

    train_gs, train_ps, train_ms = _clone_lists(gs, ps, ms)
    _step(multi_tensor_lion, train_gs, train_ps, train_ms, 1)

    ref_gs, ref_ps, ref_ms = _reference_lists(gs, ps, ms)
    _step(lion_ref, ref_gs, ref_ps, ref_ms, 1)
    _assert_step_matches(train_ps, train_ms, ref_ps, ref_ms)

@pytest.mark.multi_tensor_lion
@requires_deepspeed_reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matches_deepspeed_oracle(dtype):
    """Pin the DeepSpeed oracle explicitly over multiple steps, so a run that
    quietly stopped reaching it (deepspeed missing) is visible rather than
    silently thinner."""
    gs, ps, ms = _make_triplet(8, 16384, dtype)

    train_gs, train_ps, train_ms = _clone_lists(gs, ps, ms)
    ds_gs, ds_ps, ds_ms = _clone_lists(gs, ps, ms)
    for step in range(1, 4):
        _step(multi_tensor_lion, train_gs, train_ps, train_ms, step)
        _step(_deepspeed_lion, ds_gs, ds_ps, ds_ms, step)
    _assert_step_matches(train_ps, train_ms, ds_ps, ds_ms, atol=_ds_atol(dtype))
