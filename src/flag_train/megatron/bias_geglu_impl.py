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

"""Fused bias + GeGLU.

Forward only. One token is one row; the last dimension is the concatenated
``[gate | linear]`` and the output keeps the gated half. Bias is added inside
the kernel.

The backward is a separate operator with its own implementation file, so this
module saves nothing for autograd and defines no ``torch.autograd.Function``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_NUM_WARPS = 4
_BLOCK_SIZE = 1024
_MIN_BLOCK_SIZE = 16
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Provided by the platform layer. ``tl.sigmoid`` is a core Triton builtin;
# ``tl_extra_shim.tanh`` maps to libdevice and falls back to an exp-based
# form on backends without it.
_sigmoid = tl.sigmoid
_tanh = tl_extra_shim.tanh


_BIAS_NONE = 0
_BIAS_VECTOR = 1
_BIAS_FULL = 2

# sqrt(2/pi) and the cubic coefficient of the tanh GELU approximation.
_GELU_SQRT_2_OVER_PI = tl.constexpr(0.79788456)
_GELU_COEFF = tl.constexpr(0.044715)


def _check_input(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} needs a tensor, got {type(tensor).__name__}")
    if tensor.dtype not in _DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _DTYPES)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _prepare_bias(bias, flat):
    """Return a kernel bias tensor and a mode.

    Mode vector is one row of ``hidden`` values shared by every token, which
    is the ``bias`` produced by ``skip_bias_add``. Mode full is a different
    bias per element and is only used when the tensor really depends on the
    token index.
    """
    if bias is None:
        return None, _BIAS_NONE
    _check_input(bias, "bias_geglu bias")
    if bias.device != flat.device:
        raise ValueError(f"bias_geglu bias is on {bias.device}, expected {flat.device}")
    if bias.dtype != flat.dtype:
        raise ValueError(
            "bias_geglu bias must have the input dtype, got "
            f"{bias.dtype} for {flat.dtype}"
        )

    n_rows, hidden = flat.shape
    flat_shape = (n_rows, hidden)
    try:
        expanded = torch.broadcast_to(bias, flat_shape)
    except RuntimeError:
        raise ValueError(
            "bias_geglu bias must broadcast with the flattened input "
            f"{flat_shape}, got {tuple(bias.shape)}"
        ) from None
    if n_rows == 0 or hidden == 0:
        return bias, _BIAS_VECTOR
    if expanded.stride(0) == 0:
        # Same values on every token. Copy one row, not the whole broadcast.
        return expanded[0].contiguous(), _BIAS_VECTOR
    return expanded.contiguous(), _BIAS_FULL


@triton.jit
def _activate(gate, linear):
    """Forward value of one gate/linear pair, in fp32."""
    # tanh approximation of GELU:
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
    t = _tanh(_GELU_SQRT_2_OVER_PI * gate * (1.0 + _GELU_COEFF * gate * gate))
    return 0.5 * gate * (1.0 + t) * linear


@libentry()
@triton.jit
def _bias_geglu_forward_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    BIAS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tle.program_id(0)
    hidden = n_cols * 2
    x_row = x_ptr + row * hidden
    y_row = y_ptr + row * n_cols
    bias_row = bias_ptr + row * hidden

    for start in tl.range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        linear = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        if BIAS_MODE == 1:
            gate += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            linear += tl.load(bias_ptr + n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        elif BIAS_MODE == 2:
            gate += tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
            linear += tl.load(bias_row + n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        out = _activate(gate, linear)
        tl.store(y_row + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


def _launch_config(n_cols):
    """Power-of-two column tile and the warps needed to cover it."""
    block = max(_MIN_BLOCK_SIZE, min(_BLOCK_SIZE, triton.next_power_of_2(n_cols)))
    if block <= 64:
        return block, 1
    if block <= 256:
        return block, 2
    return block, _NUM_WARPS


def _flatten_input(x):
    _check_input(x, "bias_geglu")
    if x.ndim not in (2, 3):
        raise ValueError(
            f"bias_geglu input must have rank 2 or 3, got rank {x.ndim} "
            f"shape {tuple(x.shape)}"
        )
    hidden = x.shape[-1]
    if hidden % 2 != 0:
        raise ValueError(
            f"bias_geglu input's last dimension must be even, got {hidden}"
        )
    flat = x.contiguous().view(-1, hidden)
    out_shape = (*x.shape[:-1], hidden // 2)
    return flat, out_shape


def _forward(flat, bias):
    """Run the kernel over a flattened ``[rows, hidden]`` view."""
    n_rows, hidden = flat.shape
    n_cols = hidden // 2
    bias_tensor, bias_mode = _prepare_bias(bias, flat)
    if n_rows == 0 or n_cols == 0:
        return flat.new_empty((n_rows, n_cols))

    block, num_warps = _launch_config(n_cols)
    y = flat.new_empty((n_rows, n_cols))
    with torch_device_fn.device(flat.device):
        _bias_geglu_forward_kernel[(n_rows,)](
            flat,
            # The constexpr mode stops the kernel from reading this pointer,
            # but Triton still type-checks the argument, so reuse ``flat``.
            flat if bias_tensor is None else bias_tensor,
            y,
            n_cols,
            BIAS_MODE=bias_mode,
            BLOCK=block,
            num_warps=num_warps,
        )
    return y


def bias_geglu_impl(input, bias):
    """Bias plus GeGLU.

    Forward only: the backward is a separate operator.

    Args:
        input (Tensor): rank-2 or rank-3 fp16, bf16 or fp32 tensor on the
            active device. The last dimension is ``[gate | linear]`` and must
            be even. A non-contiguous input is copied.
        bias (Tensor or None): added to ``input`` before the activation.
            ``None`` skips the add. A bias of shape ``[hidden]`` is the
            ``skip_bias_add`` case. It must have the same dtype as ``input``.

    Returns:
        Tensor: same rank and dtype as ``input``, with the last dimension halved.
    """
    logger.debug("TRAIN BIAS_GEGLU_IMPL")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"bias_geglu bias needs a tensor or None, got {type(bias).__name__}"
        )

    flat, out_shape = _flatten_input(input)
    return _forward(flat, bias).view(out_shape)
