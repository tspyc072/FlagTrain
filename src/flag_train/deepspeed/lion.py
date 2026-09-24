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
import logging

import torch
import triton
import triton.language as tl

import flag_train
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Launch configuration swept on H20: BLOCK_SIZE 2048 with 16 warps keeps the
# per-thread work at 4 elements, matching the ILP=4 of DeepSpeed's CUDA kernel.
# Larger per-thread counts collapse occupancy and the kernel falls off a cliff.
_BLOCK_SIZE = 2048
_NUM_WARPS = 16

# Ascend rejects kernel launches with coreDim > 65535, so cap the grid and let
# each program stride over multiple blocks. On platforms with a larger limit
# (nvidia: 2**31) the cap is never hit and the loop below runs a single
# iteration, so this costs nothing where it is not needed.
_MAX_GRID = 65535


@libentry()
@triton.jit
def lion_fused_kernel(
    meta_ptr,
    n_tensors,
    n_blocks,
    lr,
    beta1,
    beta2,
    weight_decay,
    anchor,
    BLOCK_SIZE: tl.constexpr,
):
    """One Lion update block for one tensor, addressed through the meta buffer.

    Mirrors ``multi_tensor_lion_cuda`` in DeepSpeed: fp32 intermediate math,
    ``c = beta1*m + (1-beta1)*g``, ``update = c > 0 ? -lr : +lr`` (c == 0 takes
    +lr), ``p = p*(1 - lr*weight_decay) + update``, ``m = beta2*m + (1-beta2)*g``.
    A single launch covers every chunk of every tensor: each program looks its
    (tensor, chunk) pair up in the device-side meta buffer instead of receiving
    raw pointers as kernel arguments. The grid is capped at _MAX_GRID, so a
    program strides over several blocks when there are more blocks than the
    backend allows in one launch.
    """
    # meta (int64) layout:
    #   [0:n]                 g base addresses
    #   [n:2n]                p base addresses
    #   [2n:3n]               m base addresses
    #   [3n:4n]               numel of each tensor
    #   [4n + pid]            block -> tensor id
    #   [4n + n_blocks + pid] block -> chunk id within tensor
    num_programs = tl.num_programs(0)
    for pid in range(tle.program_id(0), n_blocks, num_programs):
        t = tl.load(meta_ptr + 4 * n_tensors + pid)
        c = tl.load(meta_ptr + 4 * n_tensors + n_blocks + pid)
        n = tl.load(meta_ptr + 3 * n_tensors + t)

        dtype = anchor.dtype.element_ty
        g_base = tl.load(meta_ptr + t).to(tl.pointer_type(dtype))
        p_base = tl.load(meta_ptr + n_tensors + t).to(tl.pointer_type(dtype))
        m_base = tl.load(meta_ptr + 2 * n_tensors + t).to(tl.pointer_type(dtype))

        # Hints so the compiler can vectorize the loads/stores: the chunk start
        # is a multiple of BLOCK_SIZE and the lane offsets are contiguous.
        start = tl.multiple_of(c * BLOCK_SIZE, BLOCK_SIZE)
        idx = tl.max_contiguous(start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        mask = idx < n

        # Match DeepSpeed's CUDA kernel: fp32 intermediate math (MATH_T = float).
        g = tl.load(g_base + idx, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_base + idx, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_base + idx, mask=mask, other=0.0).to(tl.float32)

        after_decay = 1.0 - lr * weight_decay
        cc = beta1 * m + (1.0 - beta1) * g
        upd = tl.where(
            cc > 0, -lr, lr
        )  # C++ semantics: c > 0 ? -lr : +lr (c==0 -> +lr)
        p_new = p * after_decay + upd
        m_new = beta2 * m + (1.0 - beta2) * g

        tl.store(p_base + idx, p_new.to(dtype), mask=mask)
        tl.store(m_base + idx, m_new.to(dtype), mask=mask)


def _build_metadata(g_list, p_list, m_list, block_size):
    """Pack every pointer/numel/block-mapping array into ONE int64 buffer.

    Packing means the device side costs a single H2D copy per rebuild instead
    of one small copy per array. The buffer is a pure function of the tensor
    set (addresses, numels, block_size), which is what makes it cacheable.
    """
    dev = p_list[0].device
    n_tensors = len(g_list)

    numels = [t.numel() for t in g_list]
    b2t = []
    b2c = []
    for i, n in enumerate(numels):
        nchunks = (n + block_size - 1) // block_size
        b2t.extend([i] * nchunks)
        b2c.extend(range(nchunks))
    ptrs = (
        [t.data_ptr() for t in g_list]
        + [t.data_ptr() for t in p_list]
        + [t.data_ptr() for t in m_list]
    )

    meta_cpu = torch.tensor(ptrs + numels + b2t + b2c, dtype=torch.int64)
    meta = meta_cpu.pin_memory().to(dev, non_blocking=True)
    return meta, n_tensors, len(b2t), g_list[0]


# Device-meta cache keyed by the exact tensor set. A hit always yields
# bit-identical content: the key contains every address and numel the buffer
# is built from, so even tensors freed and re-created at the same addresses
# with the same shapes produce the same key AND the same meta. Training loops
# call the optimizer on a fixed tensor set every step, so after the first step
# every call is a hit and the metadata cost disappears from the hot path.
_META_CACHE = {}


def _get_metadata(g_list, p_list, m_list, block_size):
    key = (
        g_list[0].dtype,
        block_size,
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.numel() for t in g_list),
    )
    entry = _META_CACHE.get(key)
    if entry is None:
        if len(_META_CACHE) >= 64:
            # Bound the device memory held by cached metas. Real workloads use
            # one tensor set, so an eviction here only costs one rebuild.
            cur = next(iter(_META_CACHE))
            del _META_CACHE[cur]
        entry = _build_metadata(g_list, p_list, m_list, block_size)
        _META_CACHE[key] = entry
    return entry


def multi_tensor_lion(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    step,
    weight_decay,
):
    """Fused Lion optimizer step over a list of parameter tensors.

    Faithful port of DeepSpeed's ``multi_tensor_lion`` CUDA operator. For each
    tensor, with gradient ``g``, parameter ``p`` and first moment ``m``::

        c      = beta1 * m + (1 - beta1) * g
        update = -lr if c > 0 else +lr      # sign update; c == 0 takes +lr
        p      = p * (1 - lr * weight_decay) + update
        m      = beta2 * m + (1 - beta2) * g

    Lion replaces Adam's adaptive scaling with the sign of the momentum, so the
    update magnitude is always exactly ``lr`` per element; it needs no second
    moment, which halves the optimizer-state memory.

    Unlike a per-tensor loop, all chunks of all tensors of one dtype are fused
    into a single kernel launch, addressed through a device-side meta buffer
    that is cached across calls (see ``_get_metadata``).

    Args:
        chunk_size (int): kept for interface parity with DeepSpeed; this
            implementation always launches with BLOCK_SIZE=2048.
        noop_flag (Tensor): kept for interface parity; DeepSpeed's own noop
            check is commented out in its source, so the flag is ignored here
            as well.
        tensor_lists (list): ``[grads, params, exp_avgs]`` -- three equally
            long lists of contiguous tensors, same dtype within each triplet.
        lr (float): learning rate; also the per-element update magnitude.
        beta1 (float): momentum coefficient used to build the update direction.
        beta2 (float): momentum coefficient for the exp_avg update itself.
        step (int): optimizer step; Lion has no bias correction, so the value
            is unused and accepted only for interface parity.
        weight_decay (float): decoupled weight decay coefficient.
    """
    logger.debug("TRAIN LION")

    g_list, p_list, m_list = tensor_lists
    assert len(g_list) == len(p_list) == len(m_list) and len(g_list) > 0
    for g, p, m in zip(g_list, p_list, m_list):
        assert g.is_contiguous() and p.is_contiguous() and m.is_contiguous()
        assert g.dtype == p.dtype == m.dtype
        # Accept whatever the active backend calls its device -- 'cuda' on
        # nvidia/hygon, 'npu' on ascend -- rather than hard-coding CUDA.
        assert (
            p.device.type == flag_train.device
        ), f"multi_tensor_lion only supports {flag_train.device} tensors"

    # Group by dtype, one launch per group (mirrors FusedLion.step).
    groups = {}
    for g, p, m in zip(g_list, p_list, m_list):
        groups.setdefault(p.dtype, []).append((g, p, m))
    for triples in groups.values():
        gs = [t[0] for t in triples]
        ps = [t[1] for t in triples]
        ms = [t[2] for t in triples]
        meta, n_tensors, n_blocks, anchor = _get_metadata(gs, ps, ms, _BLOCK_SIZE)
        if n_blocks == 0:
            continue
        with torch_device_fn.device(anchor.device):
            lion_fused_kernel[(min(n_blocks, _MAX_GRID),)](
                meta,
                n_tensors,
                n_blocks,
                lr,
                beta1,
                beta2,
                weight_decay,
                anchor=anchor,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=_NUM_WARPS,
            )
