
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3444736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 14848) % 29
    x1 = (xindex // 512) % 29
    x3 = (xindex // 430592)
    x4 = xindex % 14848
    x0 = xindex % 512
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (401408*x3)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x1 + (28*x2) + (784*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (x1 + (28*x2) + (784*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp12 = 512.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tl.load(in_ptr4 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, float("-inf"), tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, None)