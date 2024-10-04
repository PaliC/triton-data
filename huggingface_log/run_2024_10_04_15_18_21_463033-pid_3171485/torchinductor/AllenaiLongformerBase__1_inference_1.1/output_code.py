# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_sahanp/7v/c7vpbyseepx3s2rnbqzxxikru3ezvw3b7jqsqpo5xoxdnftygp5g.py
# Topologically Sorted Source Nodes: [is_index_global_attn, any_1, is_index_masked], Original ATen: [aten.gt, aten.any, aten.lt]
# Source node to ATen node mapping:
#   any_1 => any_1
#   is_index_global_attn => gt
#   is_index_masked => lt
# Graph fragment:
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.default](args = (%view,), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, 0), kwargs = {})
triton_red_fused_any_gt_lt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*i1', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {4: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_any_gt_lt_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 | tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp2, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp3, rmask)
    tmp5 = triton_helpers.any(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1024), (1024, 1), torch.bool)
        buf2 = empty_strided_cuda((4, 1024), (1024, 1), torch.bool)
        buf1 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [is_index_global_attn, any_1, is_index_masked], Original ATen: [aten.gt, aten.any, aten.lt]
        stream0 = get_raw_stream(0)
        triton_red_fused_any_gt_lt_0.run(arg0_1, buf0, buf2, buf1, 1, 4096, grid=grid(1), stream=stream0)
        del arg0_1
    return (buf1, buf2, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
