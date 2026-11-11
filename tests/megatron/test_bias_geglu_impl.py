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
"""Correctness tests for fused bias + GeGLU (forward only).

The reference is Megatron-LM's own ``bias_geglu_impl`` when megatron-core is
installed. Otherwise, ``bias_geglu_ref`` evaluates the eager tanh-GELU formula
with the bias added by ordinary broadcasting.

Under ``--ref cpu`` the operands are moved to the CPU and upcast, so the
comparison is against a higher-precision CPU result. Tolerances are
``train_assert_close``'s defaults.

Only the forward output is checked: the backward is a separate operator.
``bias`` must have the same dtype as ``input``.
"""

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils


def _load_megatron_impl():
    """Installed ``bias_geglu_impl``, or ``None`` when it is unavailable.

    Older Megatron installs may not provide the fusions module or this public
    function; those installations use the formula reference. Other import
    failures still raise instead of silently hiding a broken installation.
    """
    try:
        importlib_metadata_version = __import__(
            "importlib.metadata", fromlist=["version"]
        )
    except ImportError:
        return None
    try:
        importlib_metadata_version.version("megatron-core")
    except Exception:
        return None
    try:
        from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
    except ModuleNotFoundError as exc:
        if exc.name in {
            "megatron.core.fusions",
            "megatron.core.fusions.fused_bias_geglu",
        }:
            return None
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    except ImportError as exc:
        if "cannot import name 'bias_geglu_impl'" in str(exc):
            return None
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    return bias_geglu_impl


_megatron_impl = _load_megatron_impl()

# Empty, unaligned (half = 3 and 17), and MLP-like ranks. The last dim is even.
_SHAPES = [
    (0, 8),
    (0, 4, 16),
    (1, 2),
    (8, 6),
    (5, 34),
    (17, 32),
    (64, 128),
    (2, 8, 64),
    (4, 7, 32),
]

# Bias is orthogonal to the shapes above, so it is sampled on representative
# cases only.
_BIAS_SHAPES = [
    (1, 2),
    (17, 32),
    (64, 128),
    (4, 7, 32),
]


def bias_geglu_ref(x, bias=None):
    y = x if bias is None else x + bias
    gate, linear = torch.chunk(y, 2, dim=-1)
    # tanh approximation of GELU, the same constants as Megatron-LM.
    return (
        gate
        * 0.5
        * (1.0 + torch.tanh(0.79788456 * gate * (1 + 0.044715 * gate * gate)))
    ) * linear


def _reference(tensor):
    if tensor is None:
        return None
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _expected_output(x, bias):
    """Use Megatron-LM when available, otherwise use the eager formula.

    The Megatron side gets contiguous operands: its ``input.view(-1, hidden)``
    rejects the transposed inputs this suite deliberately feeds the operator.
    Empty inputs also use the formula: Megatron restores a rank-3 result with
    ``view(..., -1)``, which is ambiguous when the output has zero elements.
    """
    if _megatron_impl is not None and not utils.TO_CPU and x.numel() > 0:
        return _megatron_impl(
            x.contiguous(),
            None if bias is None else bias.contiguous(),
        )
    return bias_geglu_ref(x, bias)


def _check_bias_geglu(x, bias):
    dtype = x.dtype
    y = flag_train.megatron.bias_geglu_impl(x, bias)
    assert y.shape[-1] * 2 == x.shape[-1]
    assert y.shape[:-1] == x.shape[:-1]
    assert y.dtype == dtype

    expected = _expected_output(_reference(x), _reference(bias))
    _assert_close(y, expected, dtype)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", _BIAS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    bias = torch.randn(shape[-1], dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias_shapes(dtype):
    utils.init_seed(4)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, torch.randn((1, 32), dtype=dtype, device=flag_train.device))
    _check_bias_geglu(x, torch.randn((8, 32), dtype=dtype, device=flag_train.device))
    _check_bias_geglu(x, torch.tensor(0.25, dtype=dtype, device=flag_train.device))


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_large_values(dtype):
    """Extreme magnitudes: check GELU/tanh saturation."""
    values = [-1000.0, -90.0, -20.0, -1.0, 0.0, 1.0, 20.0, 90.0, 1000.0, -8.0]
    # 10 gate values and 10 linear values.
    data = values + [-1000.0, -4.0, -1.0, 0.0, 0.5, 1.0, 4.0, 20.0, 90.0, 1000.0]
    x = torch.tensor(data, dtype=torch.float32, device=flag_train.device).to(dtype)
    x = x.view(1, -1)
    _check_bias_geglu(x, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_noncontiguous(dtype):
    utils.init_seed(5)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert x.shape == (64, 32)
    assert not x.is_contiguous()
    bias = torch.randn((64,), dtype=dtype, device=flag_train.device)[::2]
    assert bias.shape == (32,)
    assert not bias.is_contiguous()
    _check_bias_geglu(x, bias)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    assert transposed.shape[-1] % 2 == 0
    _check_bias_geglu(transposed, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((4, 8), device=device)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        flag_train.megatron.bias_geglu_impl(torch.randn((8,), device=device), None)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        flag_train.megatron.bias_geglu_impl(
            torch.randn((1, 1, 2, 4), device=device), None
        )
    with pytest.raises(ValueError, match="even"):
        flag_train.megatron.bias_geglu_impl(torch.randn((4, 7), device=device), None)
    with pytest.raises(ValueError, match="CPU"):
        flag_train.megatron.bias_geglu_impl(torch.randn((4, 8)), None)
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_geglu_impl(x, torch.randn((3,), device=device))
    # An empty input must not skip the bias checks.
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_geglu_impl(
            torch.empty((0, 8), device=device), torch.randn((3,), device=device)
        )


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_requires_matching_bias_dtype(dtype):
    """The kernel computes the add in fp32; a mixed-dtype bias would silently
    change the promoted dtype eager PyTorch produces, so it is rejected."""
    device = flag_train.device
    other = torch.float32 if dtype != torch.float32 else torch.float16
    x = torch.randn((4, 8), dtype=dtype, device=device)
    with pytest.raises(ValueError, match="input dtype"):
        flag_train.megatron.bias_geglu_impl(
            x, torch.randn((8,), dtype=other, device=device)
        )
