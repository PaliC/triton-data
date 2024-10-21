
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x2 = (xindex // 32) % 28
    x1 = (xindex // 2) % 16
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 28.000001
    tmp8 = tmp6 / tmp7
    tmp9 = 6.283185307179586
    tmp10 = tmp8 * tmp9
    tmp11 = 2*x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tmp18 = 0.03125
    tmp19 = tmp17 * tmp18
    tmp20 = 10000.0
    tmp21 = libdevice.pow(tmp20, tmp19)
    tmp22 = tmp10 / tmp21
    tmp23 = tl_math.sin(tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp27 = tl.full([1], 2, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = 1 + (2*x1)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp13
    tmp32 = libdevice.floor(tmp31)
    tmp33 = tmp32 * tmp16
    tmp34 = tmp33 * tmp18
    tmp35 = libdevice.pow(tmp20, tmp34)
    tmp36 = tmp10 / tmp35
    tmp37 = tl_math.cos(tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp26, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp25, tmp39)
    tl.store(out_ptr0 + (x6), tmp40, xmask)
