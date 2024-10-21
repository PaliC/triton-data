
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28) % 28
    y2 = (yindex // 784)
    y4 = yindex % 784
    tmp0 = tl.load(in_ptr0 + (x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (256 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14592 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14848 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15104 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (29184 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (29440 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (29696 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y4 + (784*x3) + (200704*y2)), tmp16, xmask & ymask)
