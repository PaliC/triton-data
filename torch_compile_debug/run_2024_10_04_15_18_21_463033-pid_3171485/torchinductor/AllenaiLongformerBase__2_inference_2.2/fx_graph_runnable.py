
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.5.0a0+git5380214
# torch cuda version: 12.1
# torch git version: 5380214107813f63c7c59f477487d5447085b45a


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0, 2])
        permute_1 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view = torch.ops.aten.view.default(clone, [4096, 768]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute_1);  view = permute_1 = None
        view_1 = torch.ops.aten.view.default(mm, [1024, 4, 768]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        permute_2 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        clone_1 = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view_2 = torch.ops.aten.view.default(clone_1, [4096, 768]);  clone_1 = None
        mm_1 = torch.ops.aten.mm.default(view_2, permute_2);  view_2 = permute_2 = None
        view_3 = torch.ops.aten.view.default(mm_1, [1024, 4, 768]);  mm_1 = None
        add_1 = torch.ops.aten.add.Tensor(view_3, arg4_1);  view_3 = arg4_1 = None
        permute_3 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        clone_2 = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4 = torch.ops.aten.view.default(clone_2, [4096, 768]);  clone_2 = None
        mm_2 = torch.ops.aten.mm.default(view_4, permute_3);  view_4 = permute_3 = None
        view_5 = torch.ops.aten.view.default(mm_2, [1024, 4, 768]);  mm_2 = None
        add_2 = torch.ops.aten.add.Tensor(view_5, arg6_1);  view_5 = arg6_1 = None
        div = torch.ops.aten.div.Tensor(add, 8.0);  add = None
        view_7 = torch.ops.aten.view.default(add_1, [1024, 4, 12, 64]);  add_1 = None
        permute_5 = torch.ops.aten.permute.default(view_7, [1, 0, 2, 3]);  view_7 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_9 = torch.ops.aten.view.default(permute_7, [48, 1024, 64]);  permute_7 = None
        view_11 = torch.ops.aten.view.default(view_9, [48, 2, 512, 64]);  view_9 = None
        as_strided_1 = torch.ops.aten.as_strided.default(view_11, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_11 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_9 = torch.ops.aten.permute.default(unsqueeze_1, [0, 1, 4, 2, 3]);  unsqueeze_1 = None
        view_12 = torch.ops.aten.view.default(div, [1024, 4, 12, 64]);  div = None
        permute_11 = torch.ops.aten.permute.default(view_12, [1, 0, 2, 3]);  view_12 = None
        permute_12 = torch.ops.aten.permute.default(permute_11, [0, 2, 1, 3]);  permute_11 = None
        view_13 = torch.ops.aten.view.default(permute_12, [48, 1024, 64]);  permute_12 = None
        view_14 = torch.ops.aten.view.default(view_13, [48, 2, 512, 64]);  view_13 = None
        as_strided_2 = torch.ops.aten.as_strided.default(view_14, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_14 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
        permute_13 = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 2, 4, 3]);  unsqueeze_2 = None
        permute_14 = torch.ops.aten.permute.default(permute_13, [0, 1, 2, 4, 3]);  permute_13 = None
        clone_3 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_15 = torch.ops.aten.view.default(clone_3, [144, 512, 64]);  clone_3 = None
        permute_15 = torch.ops.aten.permute.default(permute_9, [0, 1, 4, 3, 2]);  permute_9 = None
        clone_4 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_16 = torch.ops.aten.view.default(clone_4, [144, 64, 512]);  clone_4 = None
        bmm = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17 = torch.ops.aten.view.default(bmm, [48, 3, 512, 1, 512]);  bmm = None
        permute_16 = torch.ops.aten.permute.default(view_17, [0, 1, 2, 4, 3]);  view_17 = None
        view_18 = torch.ops.aten.view.default(permute_16, [48, 3, 512, 512]);  permute_16 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(view_18, [0, 0, 0, 1], 0.0);  view_18 = None
        view_19 = torch.ops.aten.view.default(constant_pad_nd, [48, 3, 512, 513]);  constant_pad_nd = None
        full = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_3 = torch.ops.aten.slice.Tensor(view_19, 2, 0, 256)
        slice_4 = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257);  slice_3 = None
        slice_6 = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_8 = torch.ops.aten.slice.Tensor(slice_6, 3, 256, 9223372036854775807);  slice_6 = None
        copy = torch.ops.aten.copy.default(slice_8, slice_4);  slice_8 = slice_4 = None
        slice_10 = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_scatter = torch.ops.aten.slice_scatter.default(slice_10, copy, 3, 256, 9223372036854775807);  slice_10 = copy = None
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  full = slice_scatter = None
        select = torch.ops.aten.select.int(view_19, 1, -1)
        slice_17 = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_18 = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 257);  slice_17 = None
        select_2 = torch.ops.aten.select.int(slice_scatter_2, 1, -1)
        slice_24 = torch.ops.aten.slice.Tensor(select_2, 2, 256, 9223372036854775807);  select_2 = None
        copy_1 = torch.ops.aten.copy.default(slice_24, slice_18);  slice_24 = slice_18 = None
        select_3 = torch.ops.aten.select.int(slice_scatter_2, 1, -1)
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(select_3, copy_1, 2, 256, 9223372036854775807);  select_3 = copy_1 = None
        select_scatter = torch.ops.aten.select_scatter.default(slice_scatter_2, slice_scatter_4, 1, -1);  slice_scatter_2 = slice_scatter_4 = None
        slice_32 = torch.ops.aten.slice.Tensor(view_19, 2, -257, -1)
        slice_33 = torch.ops.aten.slice.Tensor(slice_32, 3, 257, 9223372036854775807);  slice_32 = None
        slice_39 = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_41 = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 256);  slice_39 = None
        copy_2 = torch.ops.aten.copy.default(slice_41, slice_33);  slice_41 = slice_33 = None
        slice_43 = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_scatter_7 = torch.ops.aten.slice_scatter.default(slice_43, copy_2, 3, 0, 256);  slice_43 = copy_2 = None
        slice_scatter_9 = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_7, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_7 = None
        select_5 = torch.ops.aten.select.int(view_19, 1, 0);  view_19 = None
        slice_50 = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
        slice_51 = torch.ops.aten.slice.Tensor(slice_50, 2, -255, 9223372036854775807);  slice_50 = None
        select_7 = torch.ops.aten.select.int(slice_scatter_9, 1, 0)
        slice_56 = torch.ops.aten.slice.Tensor(select_7, 1, 1, 256);  select_7 = None
        slice_57 = torch.ops.aten.slice.Tensor(slice_56, 2, 1, 256);  slice_56 = None
        copy_3 = torch.ops.aten.copy.default(slice_57, slice_51);  slice_57 = slice_51 = None
        select_8 = torch.ops.aten.select.int(slice_scatter_9, 1, 0)
        slice_59 = torch.ops.aten.slice.Tensor(select_8, 1, 1, 256)
        slice_scatter_11 = torch.ops.aten.slice_scatter.default(slice_59, copy_3, 2, 1, 256);  slice_59 = copy_3 = None
        slice_scatter_12 = torch.ops.aten.slice_scatter.default(select_8, slice_scatter_11, 1, 1, 256);  select_8 = slice_scatter_11 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(slice_scatter_9, slice_scatter_12, 1, 0);  slice_scatter_9 = slice_scatter_12 = None
        full_default = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        le = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, full_default_1);  le = full_default = full_default_1 = None
        rev = torch.ops.prims.rev.default(where, [0]);  where = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None
        rev_1 = torch.ops.prims.rev.default(unsqueeze_6, [1, 3])
        expand = torch.ops.aten.expand.default(unsqueeze_6, [4, 256, 12, 257]);  unsqueeze_6 = None
        view_22 = torch.ops.aten.view.default(select_scatter_1, [4, 12, 1024, 513])
        permute_19 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        slice_70 = torch.ops.aten.slice.Tensor(permute_19, 1, 0, 256);  permute_19 = None
        slice_72 = torch.ops.aten.slice.Tensor(slice_70, 3, 0, 257);  slice_70 = None
        full_default_2 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_1 = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_72);  convert_element_type = full_default_2 = slice_72 = None
        view_23 = torch.ops.aten.view.default(select_scatter_1, [4, 12, 1024, 513])
        permute_20 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        slice_78 = torch.ops.aten.slice.Tensor(permute_20, 1, 0, 256);  permute_20 = None
        slice_80 = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 257);  slice_78 = None
        copy_4 = torch.ops.aten.copy.default(slice_80, where_1);  slice_80 = where_1 = None
        view_24 = torch.ops.aten.view.default(select_scatter_1, [4, 12, 1024, 513]);  select_scatter_1 = None
        permute_21 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        slice_82 = torch.ops.aten.slice.Tensor(permute_21, 1, 0, 256)
        slice_scatter_14 = torch.ops.aten.slice_scatter.default(slice_82, copy_4, 3, 0, 257);  slice_82 = copy_4 = None
        slice_scatter_16 = torch.ops.aten.slice_scatter.default(permute_21, slice_scatter_14, 1, 0, 256);  permute_21 = slice_scatter_14 = None
        permute_22 = torch.ops.aten.permute.default(slice_scatter_16, [0, 2, 1, 3]);  slice_scatter_16 = None
        view_25 = torch.ops.aten.view.default(permute_22, [48, 4, 256, 513]);  permute_22 = None
        expand_1 = torch.ops.aten.expand.default(rev_1, [4, 256, 12, 257]);  rev_1 = None
        view_27 = torch.ops.aten.view.default(view_25, [4, 12, 1024, 513])
        permute_24 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        slice_93 = torch.ops.aten.slice.Tensor(permute_24, 1, -256, 9223372036854775807);  permute_24 = None
        slice_95 = torch.ops.aten.slice.Tensor(slice_93, 3, -257, 9223372036854775807);  slice_93 = None
        full_default_3 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_2 = torch.ops.aten.where.self(convert_element_type_1, full_default_3, slice_95);  convert_element_type_1 = full_default_3 = slice_95 = None
        view_28 = torch.ops.aten.view.default(view_25, [4, 12, 1024, 513])
        permute_25 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        slice_101 = torch.ops.aten.slice.Tensor(permute_25, 1, -256, 9223372036854775807);  permute_25 = None
        slice_103 = torch.ops.aten.slice.Tensor(slice_101, 3, -257, 9223372036854775807);  slice_101 = None
        copy_5 = torch.ops.aten.copy.default(slice_103, where_2);  slice_103 = where_2 = None
        view_29 = torch.ops.aten.view.default(view_25, [4, 12, 1024, 513]);  view_25 = None
        permute_26 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        slice_105 = torch.ops.aten.slice.Tensor(permute_26, 1, -256, 9223372036854775807)
        slice_scatter_18 = torch.ops.aten.slice_scatter.default(slice_105, copy_5, 3, -257, 9223372036854775807);  slice_105 = copy_5 = None
        slice_scatter_20 = torch.ops.aten.slice_scatter.default(permute_26, slice_scatter_18, 1, -256, 9223372036854775807);  permute_26 = slice_scatter_18 = None
        permute_27 = torch.ops.aten.permute.default(slice_scatter_20, [0, 2, 1, 3]);  slice_scatter_20 = None
        view_30 = torch.ops.aten.view.default(permute_27, [48, 4, 256, 513]);  permute_27 = None
        ne = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.float32)
        full_default_4 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(unsqueeze_8, full_default_4, convert_element_type_2);  unsqueeze_8 = full_default_4 = convert_element_type_2 = None
        full_4 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_29 = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_32 = torch.ops.aten.view.default(permute_29, [4, 1024, 1]);  permute_29 = None
        permute_30 = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_33 = torch.ops.aten.view.default(permute_30, [4, 1024, 1]);  permute_30 = None
        view_34 = torch.ops.aten.view.default(view_32, [4, 2, 512, 1]);  view_32 = None
        as_strided_3 = torch.ops.aten.as_strided.default(view_34, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_34 = None
        view_35 = torch.ops.aten.view.default(view_33, [4, 2, 512, 1]);  view_33 = None
        as_strided_4 = torch.ops.aten.as_strided.default(view_35, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_35 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(as_strided_3, 4);  as_strided_3 = None
        permute_31 = torch.ops.aten.permute.default(unsqueeze_9, [0, 1, 2, 4, 3]);  unsqueeze_9 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(as_strided_4, 4);  as_strided_4 = None
        permute_32 = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 4, 2, 3]);  unsqueeze_10 = None
        mul = torch.ops.aten.mul.Tensor(permute_31, permute_32);  permute_31 = permute_32 = None
        view_36 = torch.ops.aten.view.default(mul, [4, 3, 512, 512]);  mul = None
        constant_pad_nd_1 = torch.ops.aten.constant_pad_nd.default(view_36, [0, 0, 0, 1], 0.0);  view_36 = None
        view_37 = torch.ops.aten.view.default(constant_pad_nd_1, [4, 3, 512, 513]);  constant_pad_nd_1 = None
        full_5 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_115 = torch.ops.aten.slice.Tensor(view_37, 2, 0, 256)
        slice_116 = torch.ops.aten.slice.Tensor(slice_115, 3, 0, 257);  slice_115 = None
        slice_118 = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_120 = torch.ops.aten.slice.Tensor(slice_118, 3, 256, 9223372036854775807);  slice_118 = None
        copy_6 = torch.ops.aten.copy.default(slice_120, slice_116);  slice_120 = slice_116 = None
        slice_122 = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_scatter_22 = torch.ops.aten.slice_scatter.default(slice_122, copy_6, 3, 256, 9223372036854775807);  slice_122 = copy_6 = None
        slice_scatter_24 = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_22, 1, 0, -1);  full_5 = slice_scatter_22 = None
        select_10 = torch.ops.aten.select.int(view_37, 1, -1)
        slice_129 = torch.ops.aten.slice.Tensor(select_10, 1, 256, 9223372036854775807);  select_10 = None
        slice_130 = torch.ops.aten.slice.Tensor(slice_129, 2, 0, 257);  slice_129 = None
        select_12 = torch.ops.aten.select.int(slice_scatter_24, 1, -1)
        slice_136 = torch.ops.aten.slice.Tensor(select_12, 2, 256, 9223372036854775807);  select_12 = None
        copy_7 = torch.ops.aten.copy.default(slice_136, slice_130);  slice_136 = slice_130 = None
        select_13 = torch.ops.aten.select.int(slice_scatter_24, 1, -1)
        slice_scatter_26 = torch.ops.aten.slice_scatter.default(select_13, copy_7, 2, 256, 9223372036854775807);  select_13 = copy_7 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(slice_scatter_24, slice_scatter_26, 1, -1);  slice_scatter_24 = slice_scatter_26 = None
        slice_144 = torch.ops.aten.slice.Tensor(view_37, 2, -257, -1)
        slice_145 = torch.ops.aten.slice.Tensor(slice_144, 3, 257, 9223372036854775807);  slice_144 = None
        slice_151 = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_153 = torch.ops.aten.slice.Tensor(slice_151, 3, 0, 256);  slice_151 = None
        copy_8 = torch.ops.aten.copy.default(slice_153, slice_145);  slice_153 = slice_145 = None
        slice_155 = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_scatter_29 = torch.ops.aten.slice_scatter.default(slice_155, copy_8, 3, 0, 256);  slice_155 = copy_8 = None
        slice_scatter_31 = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_29, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_29 = None
        select_15 = torch.ops.aten.select.int(view_37, 1, 0);  view_37 = None
        slice_162 = torch.ops.aten.slice.Tensor(select_15, 1, 0, 255);  select_15 = None
        slice_163 = torch.ops.aten.slice.Tensor(slice_162, 2, -255, 9223372036854775807);  slice_162 = None
        select_17 = torch.ops.aten.select.int(slice_scatter_31, 1, 0)
        slice_168 = torch.ops.aten.slice.Tensor(select_17, 1, 1, 256);  select_17 = None
        slice_169 = torch.ops.aten.slice.Tensor(slice_168, 2, 1, 256);  slice_168 = None
        copy_9 = torch.ops.aten.copy.default(slice_169, slice_163);  slice_169 = slice_163 = None
        select_18 = torch.ops.aten.select.int(slice_scatter_31, 1, 0)
        slice_171 = torch.ops.aten.slice.Tensor(select_18, 1, 1, 256)
        slice_scatter_33 = torch.ops.aten.slice_scatter.default(slice_171, copy_9, 2, 1, 256);  slice_171 = copy_9 = None
        slice_scatter_34 = torch.ops.aten.slice_scatter.default(select_18, slice_scatter_33, 1, 1, 256);  select_18 = slice_scatter_33 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(slice_scatter_31, slice_scatter_34, 1, 0);  slice_scatter_31 = slice_scatter_34 = None
        full_default_5 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_2 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(unsqueeze_11, unsqueeze_12);  unsqueeze_11 = unsqueeze_12 = None
        le_1 = torch.ops.aten.le.Scalar(sub_3, 0);  sub_3 = None
        full_default_6 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(le_1, full_default_5, full_default_6);  le_1 = full_default_5 = full_default_6 = None
        rev_2 = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        rev_3 = torch.ops.prims.rev.default(unsqueeze_14, [1, 3])
        expand_2 = torch.ops.aten.expand.default(unsqueeze_14, [4, 256, 1, 257]);  unsqueeze_14 = None
        view_40 = torch.ops.aten.view.default(select_scatter_3, [4, 1, 1024, 513])
        permute_35 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        slice_182 = torch.ops.aten.slice.Tensor(permute_35, 1, 0, 256);  permute_35 = None
        slice_184 = torch.ops.aten.slice.Tensor(slice_182, 3, 0, 257);  slice_182 = None
        full_default_7 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        where_5 = torch.ops.aten.where.self(convert_element_type_3, full_default_7, slice_184);  convert_element_type_3 = full_default_7 = slice_184 = None
        view_41 = torch.ops.aten.view.default(select_scatter_3, [4, 1, 1024, 513])
        permute_36 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        slice_190 = torch.ops.aten.slice.Tensor(permute_36, 1, 0, 256);  permute_36 = None
        slice_192 = torch.ops.aten.slice.Tensor(slice_190, 3, 0, 257);  slice_190 = None
        copy_10 = torch.ops.aten.copy.default(slice_192, where_5);  slice_192 = where_5 = None
        view_42 = torch.ops.aten.view.default(select_scatter_3, [4, 1, 1024, 513]);  select_scatter_3 = None
        permute_37 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        slice_194 = torch.ops.aten.slice.Tensor(permute_37, 1, 0, 256)
        slice_scatter_36 = torch.ops.aten.slice_scatter.default(slice_194, copy_10, 3, 0, 257);  slice_194 = copy_10 = None
        slice_scatter_38 = torch.ops.aten.slice_scatter.default(permute_37, slice_scatter_36, 1, 0, 256);  permute_37 = slice_scatter_36 = None
        permute_38 = torch.ops.aten.permute.default(slice_scatter_38, [0, 2, 1, 3]);  slice_scatter_38 = None
        view_43 = torch.ops.aten.view.default(permute_38, [4, 4, 256, 513]);  permute_38 = None
        expand_3 = torch.ops.aten.expand.default(rev_3, [4, 256, 1, 257]);  rev_3 = None
        view_45 = torch.ops.aten.view.default(view_43, [4, 1, 1024, 513])
        permute_40 = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        slice_205 = torch.ops.aten.slice.Tensor(permute_40, 1, -256, 9223372036854775807);  permute_40 = None
        slice_207 = torch.ops.aten.slice.Tensor(slice_205, 3, -257, 9223372036854775807);  slice_205 = None
        full_default_8 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        where_6 = torch.ops.aten.where.self(convert_element_type_4, full_default_8, slice_207);  convert_element_type_4 = full_default_8 = slice_207 = None
        view_46 = torch.ops.aten.view.default(view_43, [4, 1, 1024, 513])
        permute_41 = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        slice_213 = torch.ops.aten.slice.Tensor(permute_41, 1, -256, 9223372036854775807);  permute_41 = None
        slice_215 = torch.ops.aten.slice.Tensor(slice_213, 3, -257, 9223372036854775807);  slice_213 = None
        copy_11 = torch.ops.aten.copy.default(slice_215, where_6);  slice_215 = where_6 = None
        view_47 = torch.ops.aten.view.default(view_43, [4, 1, 1024, 513]);  view_43 = None
        permute_42 = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        slice_217 = torch.ops.aten.slice.Tensor(permute_42, 1, -256, 9223372036854775807)
        slice_scatter_40 = torch.ops.aten.slice_scatter.default(slice_217, copy_11, 3, -257, 9223372036854775807);  slice_217 = copy_11 = None
        slice_scatter_42 = torch.ops.aten.slice_scatter.default(permute_42, slice_scatter_40, 1, -256, 9223372036854775807);  permute_42 = slice_scatter_40 = None
        permute_43 = torch.ops.aten.permute.default(slice_scatter_42, [0, 2, 1, 3]);  slice_scatter_42 = None
        view_48 = torch.ops.aten.view.default(permute_43, [4, 4, 256, 513]);  permute_43 = None
        view_50 = torch.ops.aten.view.default(view_30, [4, 12, 1024, 513]);  view_30 = None
        permute_45 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        view_51 = torch.ops.aten.view.default(view_48, [4, 1, 1024, 513]);  view_48 = None
        permute_46 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        add_5 = torch.ops.aten.add.Tensor(permute_45, permute_46);  permute_45 = permute_46 = None
        permute_47 = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
        view_53 = torch.ops.aten.view.default(permute_47, [48, 4, 256, 513]);  permute_47 = None
        view_54 = torch.ops.aten.view.default(view_53, [4, 12, 1024, 513]);  view_53 = None
        permute_48 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_5 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        amax = torch.ops.aten.amax.default(clone_5, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(clone_5, amax);  clone_5 = amax = None
        exp = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(unsqueeze_15, 3);  unsqueeze_15 = None
        full_default_9 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(unsqueeze_16, full_default_9, div_7);  unsqueeze_16 = full_default_9 = div_7 = None
        view_55 = torch.ops.aten.view.default(add_2, [1024, 4, 12, 64]);  add_2 = None
        permute_49 = torch.ops.aten.permute.default(view_55, [1, 0, 2, 3]);  view_55 = None
        permute_50 = torch.ops.aten.permute.default(where_7, [0, 2, 1, 3]);  where_7 = None
        clone_7 = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_56 = torch.ops.aten.view.default(clone_7, [48, 4, 256, 513]);  clone_7 = None
        permute_51 = torch.ops.aten.permute.default(permute_49, [0, 2, 1, 3]);  permute_49 = None
        view_57 = torch.ops.aten.view.default(permute_51, [48, 1024, 64]);  permute_51 = None
        constant_pad_nd_2 = torch.ops.aten.constant_pad_nd.default(view_57, [0, 0, 256, 256], -1.0);  view_57 = None
        as_strided_5 = torch.ops.aten.as_strided.default(constant_pad_nd_2, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_2 = None
        constant_pad_nd_3 = torch.ops.aten.constant_pad_nd.default(view_56, [0, 257], 0.0);  view_56 = None
        view_58 = torch.ops.aten.view.default(constant_pad_nd_3, [48, 4, -1]);  constant_pad_nd_3 = None
        slice_227 = torch.ops.aten.slice.Tensor(view_58, 2, 0, -256);  view_58 = None
        view_59 = torch.ops.aten.view.default(slice_227, [48, 4, 256, 769]);  slice_227 = None
        slice_231 = torch.ops.aten.slice.Tensor(view_59, 3, 0, -1);  view_59 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(slice_231, 4);  slice_231 = None
        permute_52 = torch.ops.aten.permute.default(unsqueeze_17, [0, 1, 2, 4, 3]);  unsqueeze_17 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        permute_53 = torch.ops.aten.permute.default(unsqueeze_18, [0, 1, 4, 3, 2]);  unsqueeze_18 = None
        permute_54 = torch.ops.aten.permute.default(permute_52, [0, 1, 2, 4, 3]);  permute_52 = None
        view_60 = torch.ops.aten.view.default(permute_54, [192, 256, 768]);  permute_54 = None
        permute_55 = torch.ops.aten.permute.default(permute_53, [0, 1, 4, 3, 2]);  permute_53 = None
        clone_8 = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_61 = torch.ops.aten.view.default(clone_8, [192, 768, 64]);  clone_8 = None
        bmm_1 = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
        view_62 = torch.ops.aten.view.default(bmm_1, [48, 4, 256, 1, 64]);  bmm_1 = None
        permute_56 = torch.ops.aten.permute.default(view_62, [0, 1, 2, 4, 3]);  view_62 = None
        view_63 = torch.ops.aten.view.default(permute_56, [48, 4, 256, 64]);  permute_56 = None
        view_64 = torch.ops.aten.view.default(view_63, [4, 12, 1024, 64]);  view_63 = None
        permute_57 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        permute_58 = torch.ops.aten.permute.default(permute_57, [1, 0, 2, 3]);  permute_57 = None
        clone_9 = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_65 = torch.ops.aten.view.default(clone_9, [1024, 4, 768]);  clone_9 = None
        permute_59 = torch.ops.aten.permute.default(view_65, [1, 0, 2]);  view_65 = None
        permute_60 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        clone_10 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_66 = torch.ops.aten.view.default(clone_10, [4096, 768]);  clone_10 = None
        mm_3 = torch.ops.aten.mm.default(view_66, permute_60);  view_66 = permute_60 = None
        view_67 = torch.ops.aten.view.default(mm_3, [4, 1024, 768]);  mm_3 = None
        add_7 = torch.ops.aten.add.Tensor(view_67, arg10_1);  view_67 = arg10_1 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, arg0_1);  add_7 = arg0_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_9 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_8, getitem_1);  add_8 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        view_68 = torch.ops.aten.view.default(add_10, [4096, 768])
        permute_61 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm = torch.ops.aten.addmm.default(arg14_1, view_68, permute_61);  arg14_1 = view_68 = permute_61 = None
        view_69 = torch.ops.aten.view.default(addmm, [4, 1024, 3072]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_69, 0.5)
        mul_4 = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
        erf = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_11 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_3, add_11);  mul_3 = add_11 = None
        view_70 = torch.ops.aten.view.default(mul_5, [4096, 3072]);  mul_5 = None
        permute_62 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg16_1, view_70, permute_62);  arg16_1 = view_70 = permute_62 = None
        view_71 = torch.ops.aten.view.default(addmm_1, [4, 1024, 768]);  addmm_1 = None
        add_12 = torch.ops.aten.add.Tensor(view_71, add_10);  view_71 = add_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_12, getitem_3);  add_12 = getitem_3 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = rsqrt_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
        permute_63 = torch.ops.aten.permute.default(add_14, [1, 0, 2])
        permute_64 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        clone_13 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format)
        view_72 = torch.ops.aten.view.default(clone_13, [4096, 768]);  clone_13 = None
        mm_4 = torch.ops.aten.mm.default(view_72, permute_64);  view_72 = permute_64 = None
        view_73 = torch.ops.aten.view.default(mm_4, [1024, 4, 768]);  mm_4 = None
        add_15 = torch.ops.aten.add.Tensor(view_73, arg20_1);  view_73 = arg20_1 = None
        permute_65 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        clone_14 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format)
        view_74 = torch.ops.aten.view.default(clone_14, [4096, 768]);  clone_14 = None
        mm_5 = torch.ops.aten.mm.default(view_74, permute_65);  view_74 = permute_65 = None
        view_75 = torch.ops.aten.view.default(mm_5, [1024, 4, 768]);  mm_5 = None
        add_16 = torch.ops.aten.add.Tensor(view_75, arg22_1);  view_75 = arg22_1 = None
        permute_66 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        clone_15 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_76 = torch.ops.aten.view.default(clone_15, [4096, 768]);  clone_15 = None
        mm_6 = torch.ops.aten.mm.default(view_76, permute_66);  view_76 = permute_66 = None
        view_77 = torch.ops.aten.view.default(mm_6, [1024, 4, 768]);  mm_6 = None
        add_17 = torch.ops.aten.add.Tensor(view_77, arg24_1);  view_77 = arg24_1 = None
        div_10 = torch.ops.aten.div.Tensor(add_15, 8.0);  add_15 = None
        view_79 = torch.ops.aten.view.default(add_16, [1024, 4, 12, 64]);  add_16 = None
        permute_68 = torch.ops.aten.permute.default(view_79, [1, 0, 2, 3]);  view_79 = None
        permute_70 = torch.ops.aten.permute.default(permute_68, [0, 2, 1, 3]);  permute_68 = None
        view_81 = torch.ops.aten.view.default(permute_70, [48, 1024, 64]);  permute_70 = None
        view_83 = torch.ops.aten.view.default(view_81, [48, 2, 512, 64]);  view_81 = None
        as_strided_7 = torch.ops.aten.as_strided.default(view_83, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_83 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        permute_72 = torch.ops.aten.permute.default(unsqueeze_20, [0, 1, 4, 2, 3]);  unsqueeze_20 = None
        view_84 = torch.ops.aten.view.default(div_10, [1024, 4, 12, 64]);  div_10 = None
        permute_74 = torch.ops.aten.permute.default(view_84, [1, 0, 2, 3]);  view_84 = None
        permute_75 = torch.ops.aten.permute.default(permute_74, [0, 2, 1, 3]);  permute_74 = None
        view_85 = torch.ops.aten.view.default(permute_75, [48, 1024, 64]);  permute_75 = None
        view_86 = torch.ops.aten.view.default(view_85, [48, 2, 512, 64]);  view_85 = None
        as_strided_8 = torch.ops.aten.as_strided.default(view_86, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_86 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        permute_76 = torch.ops.aten.permute.default(unsqueeze_21, [0, 1, 2, 4, 3]);  unsqueeze_21 = None
        permute_77 = torch.ops.aten.permute.default(permute_76, [0, 1, 2, 4, 3]);  permute_76 = None
        clone_16 = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_87 = torch.ops.aten.view.default(clone_16, [144, 512, 64]);  clone_16 = None
        permute_78 = torch.ops.aten.permute.default(permute_72, [0, 1, 4, 3, 2]);  permute_72 = None
        clone_17 = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_88 = torch.ops.aten.view.default(clone_17, [144, 64, 512]);  clone_17 = None
        bmm_2 = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
        view_89 = torch.ops.aten.view.default(bmm_2, [48, 3, 512, 1, 512]);  bmm_2 = None
        permute_79 = torch.ops.aten.permute.default(view_89, [0, 1, 2, 4, 3]);  view_89 = None
        view_90 = torch.ops.aten.view.default(permute_79, [48, 3, 512, 512]);  permute_79 = None
        constant_pad_nd_4 = torch.ops.aten.constant_pad_nd.default(view_90, [0, 0, 0, 1], 0.0);  view_90 = None
        view_91 = torch.ops.aten.view.default(constant_pad_nd_4, [48, 3, 512, 513]);  constant_pad_nd_4 = None
        full_9 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_234 = torch.ops.aten.slice.Tensor(view_91, 2, 0, 256)
        slice_235 = torch.ops.aten.slice.Tensor(slice_234, 3, 0, 257);  slice_234 = None
        slice_237 = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_239 = torch.ops.aten.slice.Tensor(slice_237, 3, 256, 9223372036854775807);  slice_237 = None
        copy_12 = torch.ops.aten.copy.default(slice_239, slice_235);  slice_239 = slice_235 = None
        slice_241 = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_scatter_44 = torch.ops.aten.slice_scatter.default(slice_241, copy_12, 3, 256, 9223372036854775807);  slice_241 = copy_12 = None
        slice_scatter_46 = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_44, 1, 0, -1);  full_9 = slice_scatter_44 = None
        select_20 = torch.ops.aten.select.int(view_91, 1, -1)
        slice_248 = torch.ops.aten.slice.Tensor(select_20, 1, 256, 9223372036854775807);  select_20 = None
        slice_249 = torch.ops.aten.slice.Tensor(slice_248, 2, 0, 257);  slice_248 = None
        select_22 = torch.ops.aten.select.int(slice_scatter_46, 1, -1)
        slice_255 = torch.ops.aten.slice.Tensor(select_22, 2, 256, 9223372036854775807);  select_22 = None
        copy_13 = torch.ops.aten.copy.default(slice_255, slice_249);  slice_255 = slice_249 = None
        select_23 = torch.ops.aten.select.int(slice_scatter_46, 1, -1)
        slice_scatter_48 = torch.ops.aten.slice_scatter.default(select_23, copy_13, 2, 256, 9223372036854775807);  select_23 = copy_13 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(slice_scatter_46, slice_scatter_48, 1, -1);  slice_scatter_46 = slice_scatter_48 = None
        slice_263 = torch.ops.aten.slice.Tensor(view_91, 2, -257, -1)
        slice_264 = torch.ops.aten.slice.Tensor(slice_263, 3, 257, 9223372036854775807);  slice_263 = None
        slice_270 = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_272 = torch.ops.aten.slice.Tensor(slice_270, 3, 0, 256);  slice_270 = None
        copy_14 = torch.ops.aten.copy.default(slice_272, slice_264);  slice_272 = slice_264 = None
        slice_274 = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_scatter_51 = torch.ops.aten.slice_scatter.default(slice_274, copy_14, 3, 0, 256);  slice_274 = copy_14 = None
        slice_scatter_53 = torch.ops.aten.slice_scatter.default(select_scatter_4, slice_scatter_51, 1, 1, 9223372036854775807);  select_scatter_4 = slice_scatter_51 = None
        select_25 = torch.ops.aten.select.int(view_91, 1, 0);  view_91 = None
        slice_281 = torch.ops.aten.slice.Tensor(select_25, 1, 0, 255);  select_25 = None
        slice_282 = torch.ops.aten.slice.Tensor(slice_281, 2, -255, 9223372036854775807);  slice_281 = None
        select_27 = torch.ops.aten.select.int(slice_scatter_53, 1, 0)
        slice_287 = torch.ops.aten.slice.Tensor(select_27, 1, 1, 256);  select_27 = None
        slice_288 = torch.ops.aten.slice.Tensor(slice_287, 2, 1, 256);  slice_287 = None
        copy_15 = torch.ops.aten.copy.default(slice_288, slice_282);  slice_288 = slice_282 = None
        select_28 = torch.ops.aten.select.int(slice_scatter_53, 1, 0)
        slice_290 = torch.ops.aten.slice.Tensor(select_28, 1, 1, 256)
        slice_scatter_55 = torch.ops.aten.slice_scatter.default(slice_290, copy_15, 2, 1, 256);  slice_290 = copy_15 = None
        slice_scatter_56 = torch.ops.aten.slice_scatter.default(select_28, slice_scatter_55, 1, 1, 256);  select_28 = slice_scatter_55 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(slice_scatter_53, slice_scatter_56, 1, 0);  slice_scatter_53 = slice_scatter_56 = None
        full_default_10 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_4 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(iota_4, -2);  iota_4 = None
        iota_5 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        sub_9 = torch.ops.aten.sub.Tensor(unsqueeze_22, unsqueeze_23);  unsqueeze_22 = unsqueeze_23 = None
        le_2 = torch.ops.aten.le.Scalar(sub_9, 0);  sub_9 = None
        full_default_11 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8 = torch.ops.aten.where.self(le_2, full_default_10, full_default_11);  le_2 = full_default_10 = full_default_11 = None
        rev_4 = torch.ops.prims.rev.default(where_8, [0]);  where_8 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(rev_4, 0);  rev_4 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        rev_5 = torch.ops.prims.rev.default(unsqueeze_25, [1, 3])
        expand_4 = torch.ops.aten.expand.default(unsqueeze_25, [4, 256, 12, 257]);  unsqueeze_25 = None
        view_94 = torch.ops.aten.view.default(select_scatter_5, [4, 12, 1024, 513])
        permute_82 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        slice_301 = torch.ops.aten.slice.Tensor(permute_82, 1, 0, 256);  permute_82 = None
        slice_303 = torch.ops.aten.slice.Tensor(slice_301, 3, 0, 257);  slice_301 = None
        full_default_12 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(expand_4, torch.bool);  expand_4 = None
        where_9 = torch.ops.aten.where.self(convert_element_type_5, full_default_12, slice_303);  convert_element_type_5 = full_default_12 = slice_303 = None
        view_95 = torch.ops.aten.view.default(select_scatter_5, [4, 12, 1024, 513])
        permute_83 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        slice_309 = torch.ops.aten.slice.Tensor(permute_83, 1, 0, 256);  permute_83 = None
        slice_311 = torch.ops.aten.slice.Tensor(slice_309, 3, 0, 257);  slice_309 = None
        copy_16 = torch.ops.aten.copy.default(slice_311, where_9);  slice_311 = where_9 = None
        view_96 = torch.ops.aten.view.default(select_scatter_5, [4, 12, 1024, 513]);  select_scatter_5 = None
        permute_84 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        slice_313 = torch.ops.aten.slice.Tensor(permute_84, 1, 0, 256)
        slice_scatter_58 = torch.ops.aten.slice_scatter.default(slice_313, copy_16, 3, 0, 257);  slice_313 = copy_16 = None
        slice_scatter_60 = torch.ops.aten.slice_scatter.default(permute_84, slice_scatter_58, 1, 0, 256);  permute_84 = slice_scatter_58 = None
        permute_85 = torch.ops.aten.permute.default(slice_scatter_60, [0, 2, 1, 3]);  slice_scatter_60 = None
        view_97 = torch.ops.aten.view.default(permute_85, [48, 4, 256, 513]);  permute_85 = None
        expand_5 = torch.ops.aten.expand.default(rev_5, [4, 256, 12, 257]);  rev_5 = None
        view_99 = torch.ops.aten.view.default(view_97, [4, 12, 1024, 513])
        permute_87 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        slice_324 = torch.ops.aten.slice.Tensor(permute_87, 1, -256, 9223372036854775807);  permute_87 = None
        slice_326 = torch.ops.aten.slice.Tensor(slice_324, 3, -257, 9223372036854775807);  slice_324 = None
        full_default_13 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(expand_5, torch.bool);  expand_5 = None
        where_10 = torch.ops.aten.where.self(convert_element_type_6, full_default_13, slice_326);  convert_element_type_6 = full_default_13 = slice_326 = None
        view_100 = torch.ops.aten.view.default(view_97, [4, 12, 1024, 513])
        permute_88 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        slice_332 = torch.ops.aten.slice.Tensor(permute_88, 1, -256, 9223372036854775807);  permute_88 = None
        slice_334 = torch.ops.aten.slice.Tensor(slice_332, 3, -257, 9223372036854775807);  slice_332 = None
        copy_17 = torch.ops.aten.copy.default(slice_334, where_10);  slice_334 = where_10 = None
        view_101 = torch.ops.aten.view.default(view_97, [4, 12, 1024, 513]);  view_97 = None
        permute_89 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        slice_336 = torch.ops.aten.slice.Tensor(permute_89, 1, -256, 9223372036854775807)
        slice_scatter_62 = torch.ops.aten.slice_scatter.default(slice_336, copy_17, 3, -257, 9223372036854775807);  slice_336 = copy_17 = None
        slice_scatter_64 = torch.ops.aten.slice_scatter.default(permute_89, slice_scatter_62, 1, -256, 9223372036854775807);  permute_89 = slice_scatter_62 = None
        permute_90 = torch.ops.aten.permute.default(slice_scatter_64, [0, 2, 1, 3]);  slice_scatter_64 = None
        view_102 = torch.ops.aten.view.default(permute_90, [48, 4, 256, 513]);  permute_90 = None
        ne_1 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(ne_1, 2);  ne_1 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, 3);  unsqueeze_26 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(unsqueeze_27, torch.float32)
        full_default_14 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(unsqueeze_27, full_default_14, convert_element_type_7);  unsqueeze_27 = full_default_14 = convert_element_type_7 = None
        full_13 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_92 = torch.ops.aten.permute.default(full_13, [0, 2, 1, 3]);  full_13 = None
        view_104 = torch.ops.aten.view.default(permute_92, [4, 1024, 1]);  permute_92 = None
        permute_93 = torch.ops.aten.permute.default(where_11, [0, 2, 1, 3]);  where_11 = None
        view_105 = torch.ops.aten.view.default(permute_93, [4, 1024, 1]);  permute_93 = None
        view_106 = torch.ops.aten.view.default(view_104, [4, 2, 512, 1]);  view_104 = None
        as_strided_9 = torch.ops.aten.as_strided.default(view_106, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_106 = None
        view_107 = torch.ops.aten.view.default(view_105, [4, 2, 512, 1]);  view_105 = None
        as_strided_10 = torch.ops.aten.as_strided.default(view_107, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_107 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(as_strided_9, 4);  as_strided_9 = None
        permute_94 = torch.ops.aten.permute.default(unsqueeze_28, [0, 1, 2, 4, 3]);  unsqueeze_28 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        permute_95 = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 4, 2, 3]);  unsqueeze_29 = None
        mul_8 = torch.ops.aten.mul.Tensor(permute_94, permute_95);  permute_94 = permute_95 = None
        view_108 = torch.ops.aten.view.default(mul_8, [4, 3, 512, 512]);  mul_8 = None
        constant_pad_nd_5 = torch.ops.aten.constant_pad_nd.default(view_108, [0, 0, 0, 1], 0.0);  view_108 = None
        view_109 = torch.ops.aten.view.default(constant_pad_nd_5, [4, 3, 512, 513]);  constant_pad_nd_5 = None
        full_14 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_346 = torch.ops.aten.slice.Tensor(view_109, 2, 0, 256)
        slice_347 = torch.ops.aten.slice.Tensor(slice_346, 3, 0, 257);  slice_346 = None
        slice_349 = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_351 = torch.ops.aten.slice.Tensor(slice_349, 3, 256, 9223372036854775807);  slice_349 = None
        copy_18 = torch.ops.aten.copy.default(slice_351, slice_347);  slice_351 = slice_347 = None
        slice_353 = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_scatter_66 = torch.ops.aten.slice_scatter.default(slice_353, copy_18, 3, 256, 9223372036854775807);  slice_353 = copy_18 = None
        slice_scatter_68 = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_66, 1, 0, -1);  full_14 = slice_scatter_66 = None
        select_30 = torch.ops.aten.select.int(view_109, 1, -1)
        slice_360 = torch.ops.aten.slice.Tensor(select_30, 1, 256, 9223372036854775807);  select_30 = None
        slice_361 = torch.ops.aten.slice.Tensor(slice_360, 2, 0, 257);  slice_360 = None
        select_32 = torch.ops.aten.select.int(slice_scatter_68, 1, -1)
        slice_367 = torch.ops.aten.slice.Tensor(select_32, 2, 256, 9223372036854775807);  select_32 = None
        copy_19 = torch.ops.aten.copy.default(slice_367, slice_361);  slice_367 = slice_361 = None
        select_33 = torch.ops.aten.select.int(slice_scatter_68, 1, -1)
        slice_scatter_70 = torch.ops.aten.slice_scatter.default(select_33, copy_19, 2, 256, 9223372036854775807);  select_33 = copy_19 = None
        select_scatter_6 = torch.ops.aten.select_scatter.default(slice_scatter_68, slice_scatter_70, 1, -1);  slice_scatter_68 = slice_scatter_70 = None
        slice_375 = torch.ops.aten.slice.Tensor(view_109, 2, -257, -1)
        slice_376 = torch.ops.aten.slice.Tensor(slice_375, 3, 257, 9223372036854775807);  slice_375 = None
        slice_382 = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_384 = torch.ops.aten.slice.Tensor(slice_382, 3, 0, 256);  slice_382 = None
        copy_20 = torch.ops.aten.copy.default(slice_384, slice_376);  slice_384 = slice_376 = None
        slice_386 = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_scatter_73 = torch.ops.aten.slice_scatter.default(slice_386, copy_20, 3, 0, 256);  slice_386 = copy_20 = None
        slice_scatter_75 = torch.ops.aten.slice_scatter.default(select_scatter_6, slice_scatter_73, 1, 1, 9223372036854775807);  select_scatter_6 = slice_scatter_73 = None
        select_35 = torch.ops.aten.select.int(view_109, 1, 0);  view_109 = None
        slice_393 = torch.ops.aten.slice.Tensor(select_35, 1, 0, 255);  select_35 = None
        slice_394 = torch.ops.aten.slice.Tensor(slice_393, 2, -255, 9223372036854775807);  slice_393 = None
        select_37 = torch.ops.aten.select.int(slice_scatter_75, 1, 0)
        slice_399 = torch.ops.aten.slice.Tensor(select_37, 1, 1, 256);  select_37 = None
        slice_400 = torch.ops.aten.slice.Tensor(slice_399, 2, 1, 256);  slice_399 = None
        copy_21 = torch.ops.aten.copy.default(slice_400, slice_394);  slice_400 = slice_394 = None
        select_38 = torch.ops.aten.select.int(slice_scatter_75, 1, 0)
        slice_402 = torch.ops.aten.slice.Tensor(select_38, 1, 1, 256)
        slice_scatter_77 = torch.ops.aten.slice_scatter.default(slice_402, copy_21, 2, 1, 256);  slice_402 = copy_21 = None
        slice_scatter_78 = torch.ops.aten.slice_scatter.default(select_38, slice_scatter_77, 1, 1, 256);  select_38 = slice_scatter_77 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(slice_scatter_75, slice_scatter_78, 1, 0);  slice_scatter_75 = slice_scatter_78 = None
        full_default_15 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_6 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(iota_6, -2);  iota_6 = None
        iota_7 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        sub_11 = torch.ops.aten.sub.Tensor(unsqueeze_30, unsqueeze_31);  unsqueeze_30 = unsqueeze_31 = None
        le_3 = torch.ops.aten.le.Scalar(sub_11, 0);  sub_11 = None
        full_default_16 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12 = torch.ops.aten.where.self(le_3, full_default_15, full_default_16);  le_3 = full_default_15 = full_default_16 = None
        rev_6 = torch.ops.prims.rev.default(where_12, [0]);  where_12 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(rev_6, 0);  rev_6 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, 2);  unsqueeze_32 = None
        rev_7 = torch.ops.prims.rev.default(unsqueeze_33, [1, 3])
        expand_6 = torch.ops.aten.expand.default(unsqueeze_33, [4, 256, 1, 257]);  unsqueeze_33 = None
        view_112 = torch.ops.aten.view.default(select_scatter_7, [4, 1, 1024, 513])
        permute_98 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        slice_413 = torch.ops.aten.slice.Tensor(permute_98, 1, 0, 256);  permute_98 = None
        slice_415 = torch.ops.aten.slice.Tensor(slice_413, 3, 0, 257);  slice_413 = None
        full_default_17 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(expand_6, torch.bool);  expand_6 = None
        where_13 = torch.ops.aten.where.self(convert_element_type_8, full_default_17, slice_415);  convert_element_type_8 = full_default_17 = slice_415 = None
        view_113 = torch.ops.aten.view.default(select_scatter_7, [4, 1, 1024, 513])
        permute_99 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        slice_421 = torch.ops.aten.slice.Tensor(permute_99, 1, 0, 256);  permute_99 = None
        slice_423 = torch.ops.aten.slice.Tensor(slice_421, 3, 0, 257);  slice_421 = None
        copy_22 = torch.ops.aten.copy.default(slice_423, where_13);  slice_423 = where_13 = None
        view_114 = torch.ops.aten.view.default(select_scatter_7, [4, 1, 1024, 513]);  select_scatter_7 = None
        permute_100 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        slice_425 = torch.ops.aten.slice.Tensor(permute_100, 1, 0, 256)
        slice_scatter_80 = torch.ops.aten.slice_scatter.default(slice_425, copy_22, 3, 0, 257);  slice_425 = copy_22 = None
        slice_scatter_82 = torch.ops.aten.slice_scatter.default(permute_100, slice_scatter_80, 1, 0, 256);  permute_100 = slice_scatter_80 = None
        permute_101 = torch.ops.aten.permute.default(slice_scatter_82, [0, 2, 1, 3]);  slice_scatter_82 = None
        view_115 = torch.ops.aten.view.default(permute_101, [4, 4, 256, 513]);  permute_101 = None
        expand_7 = torch.ops.aten.expand.default(rev_7, [4, 256, 1, 257]);  rev_7 = None
        view_117 = torch.ops.aten.view.default(view_115, [4, 1, 1024, 513])
        permute_103 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        slice_436 = torch.ops.aten.slice.Tensor(permute_103, 1, -256, 9223372036854775807);  permute_103 = None
        slice_438 = torch.ops.aten.slice.Tensor(slice_436, 3, -257, 9223372036854775807);  slice_436 = None
        full_default_18 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(expand_7, torch.bool);  expand_7 = None
        where_14 = torch.ops.aten.where.self(convert_element_type_9, full_default_18, slice_438);  convert_element_type_9 = full_default_18 = slice_438 = None
        view_118 = torch.ops.aten.view.default(view_115, [4, 1, 1024, 513])
        permute_104 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        slice_444 = torch.ops.aten.slice.Tensor(permute_104, 1, -256, 9223372036854775807);  permute_104 = None
        slice_446 = torch.ops.aten.slice.Tensor(slice_444, 3, -257, 9223372036854775807);  slice_444 = None
        copy_23 = torch.ops.aten.copy.default(slice_446, where_14);  slice_446 = where_14 = None
        view_119 = torch.ops.aten.view.default(view_115, [4, 1, 1024, 513]);  view_115 = None
        permute_105 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        slice_448 = torch.ops.aten.slice.Tensor(permute_105, 1, -256, 9223372036854775807)
        slice_scatter_84 = torch.ops.aten.slice_scatter.default(slice_448, copy_23, 3, -257, 9223372036854775807);  slice_448 = copy_23 = None
        slice_scatter_86 = torch.ops.aten.slice_scatter.default(permute_105, slice_scatter_84, 1, -256, 9223372036854775807);  permute_105 = slice_scatter_84 = None
        permute_106 = torch.ops.aten.permute.default(slice_scatter_86, [0, 2, 1, 3]);  slice_scatter_86 = None
        view_120 = torch.ops.aten.view.default(permute_106, [4, 4, 256, 513]);  permute_106 = None
        view_122 = torch.ops.aten.view.default(view_102, [4, 12, 1024, 513]);  view_102 = None
        permute_108 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        view_123 = torch.ops.aten.view.default(view_120, [4, 1, 1024, 513]);  view_120 = None
        permute_109 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        add_20 = torch.ops.aten.add.Tensor(permute_108, permute_109);  permute_108 = permute_109 = None
        permute_110 = torch.ops.aten.permute.default(add_20, [0, 2, 1, 3]);  add_20 = None
        view_125 = torch.ops.aten.view.default(permute_110, [48, 4, 256, 513]);  permute_110 = None
        view_126 = torch.ops.aten.view.default(view_125, [4, 12, 1024, 513]);  view_125 = None
        permute_111 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_18 = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        amax_1 = torch.ops.aten.amax.default(clone_18, [-1], True)
        sub_12 = torch.ops.aten.sub.Tensor(clone_18, amax_1);  clone_18 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        full_default_19 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15 = torch.ops.aten.where.self(unsqueeze_35, full_default_19, div_17);  unsqueeze_35 = full_default_19 = div_17 = None
        view_127 = torch.ops.aten.view.default(add_17, [1024, 4, 12, 64]);  add_17 = None
        permute_112 = torch.ops.aten.permute.default(view_127, [1, 0, 2, 3]);  view_127 = None
        permute_113 = torch.ops.aten.permute.default(where_15, [0, 2, 1, 3]);  where_15 = None
        clone_20 = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_128 = torch.ops.aten.view.default(clone_20, [48, 4, 256, 513]);  clone_20 = None
        permute_114 = torch.ops.aten.permute.default(permute_112, [0, 2, 1, 3]);  permute_112 = None
        view_129 = torch.ops.aten.view.default(permute_114, [48, 1024, 64]);  permute_114 = None
        constant_pad_nd_6 = torch.ops.aten.constant_pad_nd.default(view_129, [0, 0, 256, 256], -1.0);  view_129 = None
        as_strided_11 = torch.ops.aten.as_strided.default(constant_pad_nd_6, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_6 = None
        constant_pad_nd_7 = torch.ops.aten.constant_pad_nd.default(view_128, [0, 257], 0.0);  view_128 = None
        view_130 = torch.ops.aten.view.default(constant_pad_nd_7, [48, 4, -1]);  constant_pad_nd_7 = None
        slice_458 = torch.ops.aten.slice.Tensor(view_130, 2, 0, -256);  view_130 = None
        view_131 = torch.ops.aten.view.default(slice_458, [48, 4, 256, 769]);  slice_458 = None
        slice_462 = torch.ops.aten.slice.Tensor(view_131, 3, 0, -1);  view_131 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(slice_462, 4);  slice_462 = None
        permute_115 = torch.ops.aten.permute.default(unsqueeze_36, [0, 1, 2, 4, 3]);  unsqueeze_36 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(as_strided_11, 4);  as_strided_11 = None
        permute_116 = torch.ops.aten.permute.default(unsqueeze_37, [0, 1, 4, 3, 2]);  unsqueeze_37 = None
        permute_117 = torch.ops.aten.permute.default(permute_115, [0, 1, 2, 4, 3]);  permute_115 = None
        view_132 = torch.ops.aten.view.default(permute_117, [192, 256, 768]);  permute_117 = None
        permute_118 = torch.ops.aten.permute.default(permute_116, [0, 1, 4, 3, 2]);  permute_116 = None
        clone_21 = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_133 = torch.ops.aten.view.default(clone_21, [192, 768, 64]);  clone_21 = None
        bmm_3 = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = view_133 = None
        view_134 = torch.ops.aten.view.default(bmm_3, [48, 4, 256, 1, 64]);  bmm_3 = None
        permute_119 = torch.ops.aten.permute.default(view_134, [0, 1, 2, 4, 3]);  view_134 = None
        view_135 = torch.ops.aten.view.default(permute_119, [48, 4, 256, 64]);  permute_119 = None
        view_136 = torch.ops.aten.view.default(view_135, [4, 12, 1024, 64]);  view_135 = None
        permute_120 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_121 = torch.ops.aten.permute.default(permute_120, [1, 0, 2, 3]);  permute_120 = None
        clone_22 = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_137 = torch.ops.aten.view.default(clone_22, [1024, 4, 768]);  clone_22 = None
        permute_122 = torch.ops.aten.permute.default(view_137, [1, 0, 2]);  view_137 = None
        permute_123 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        clone_23 = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_138 = torch.ops.aten.view.default(clone_23, [4096, 768]);  clone_23 = None
        mm_7 = torch.ops.aten.mm.default(view_138, permute_123);  view_138 = permute_123 = None
        view_139 = torch.ops.aten.view.default(mm_7, [4, 1024, 768]);  mm_7 = None
        add_22 = torch.ops.aten.add.Tensor(view_139, arg26_1);  view_139 = arg26_1 = None
        add_23 = torch.ops.aten.add.Tensor(add_22, add_14);  add_22 = add_14 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_23, getitem_5);  add_23 = getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg27_1);  mul_9 = arg27_1 = None
        add_25 = torch.ops.aten.add.Tensor(mul_10, arg28_1);  mul_10 = arg28_1 = None
        view_140 = torch.ops.aten.view.default(add_25, [4096, 768])
        permute_124 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg30_1, view_140, permute_124);  arg30_1 = view_140 = permute_124 = None
        view_141 = torch.ops.aten.view.default(addmm_2, [4, 1024, 3072]);  addmm_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_26 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_26);  mul_11 = add_26 = None
        view_142 = torch.ops.aten.view.default(mul_13, [4096, 3072]);  mul_13 = None
        permute_125 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg32_1, view_142, permute_125);  arg32_1 = view_142 = permute_125 = None
        view_143 = torch.ops.aten.view.default(addmm_3, [4, 1024, 768]);  addmm_3 = None
        add_27 = torch.ops.aten.add.Tensor(view_143, add_25);  view_143 = add_25 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_27, getitem_7);  add_27 = getitem_7 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = rsqrt_3 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        add_29 = torch.ops.aten.add.Tensor(mul_15, arg34_1);  mul_15 = arg34_1 = None
        permute_126 = torch.ops.aten.permute.default(add_29, [1, 0, 2])
        permute_127 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        clone_26 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format)
        view_144 = torch.ops.aten.view.default(clone_26, [4096, 768]);  clone_26 = None
        mm_8 = torch.ops.aten.mm.default(view_144, permute_127);  view_144 = permute_127 = None
        view_145 = torch.ops.aten.view.default(mm_8, [1024, 4, 768]);  mm_8 = None
        add_30 = torch.ops.aten.add.Tensor(view_145, arg36_1);  view_145 = arg36_1 = None
        permute_128 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        clone_27 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format)
        view_146 = torch.ops.aten.view.default(clone_27, [4096, 768]);  clone_27 = None
        mm_9 = torch.ops.aten.mm.default(view_146, permute_128);  view_146 = permute_128 = None
        view_147 = torch.ops.aten.view.default(mm_9, [1024, 4, 768]);  mm_9 = None
        add_31 = torch.ops.aten.add.Tensor(view_147, arg38_1);  view_147 = arg38_1 = None
        permute_129 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        clone_28 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_148 = torch.ops.aten.view.default(clone_28, [4096, 768]);  clone_28 = None
        mm_10 = torch.ops.aten.mm.default(view_148, permute_129);  view_148 = permute_129 = None
        view_149 = torch.ops.aten.view.default(mm_10, [1024, 4, 768]);  mm_10 = None
        add_32 = torch.ops.aten.add.Tensor(view_149, arg40_1);  view_149 = arg40_1 = None
        div_20 = torch.ops.aten.div.Tensor(add_30, 8.0);  add_30 = None
        view_151 = torch.ops.aten.view.default(add_31, [1024, 4, 12, 64]);  add_31 = None
        permute_131 = torch.ops.aten.permute.default(view_151, [1, 0, 2, 3]);  view_151 = None
        permute_133 = torch.ops.aten.permute.default(permute_131, [0, 2, 1, 3]);  permute_131 = None
        view_153 = torch.ops.aten.view.default(permute_133, [48, 1024, 64]);  permute_133 = None
        view_155 = torch.ops.aten.view.default(view_153, [48, 2, 512, 64]);  view_153 = None
        as_strided_13 = torch.ops.aten.as_strided.default(view_155, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_155 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(as_strided_13, 4);  as_strided_13 = None
        permute_135 = torch.ops.aten.permute.default(unsqueeze_39, [0, 1, 4, 2, 3]);  unsqueeze_39 = None
        view_156 = torch.ops.aten.view.default(div_20, [1024, 4, 12, 64]);  div_20 = None
        permute_137 = torch.ops.aten.permute.default(view_156, [1, 0, 2, 3]);  view_156 = None
        permute_138 = torch.ops.aten.permute.default(permute_137, [0, 2, 1, 3]);  permute_137 = None
        view_157 = torch.ops.aten.view.default(permute_138, [48, 1024, 64]);  permute_138 = None
        view_158 = torch.ops.aten.view.default(view_157, [48, 2, 512, 64]);  view_157 = None
        as_strided_14 = torch.ops.aten.as_strided.default(view_158, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_158 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        permute_139 = torch.ops.aten.permute.default(unsqueeze_40, [0, 1, 2, 4, 3]);  unsqueeze_40 = None
        permute_140 = torch.ops.aten.permute.default(permute_139, [0, 1, 2, 4, 3]);  permute_139 = None
        clone_29 = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_159 = torch.ops.aten.view.default(clone_29, [144, 512, 64]);  clone_29 = None
        permute_141 = torch.ops.aten.permute.default(permute_135, [0, 1, 4, 3, 2]);  permute_135 = None
        clone_30 = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_160 = torch.ops.aten.view.default(clone_30, [144, 64, 512]);  clone_30 = None
        bmm_4 = torch.ops.aten.bmm.default(view_159, view_160);  view_159 = view_160 = None
        view_161 = torch.ops.aten.view.default(bmm_4, [48, 3, 512, 1, 512]);  bmm_4 = None
        permute_142 = torch.ops.aten.permute.default(view_161, [0, 1, 2, 4, 3]);  view_161 = None
        view_162 = torch.ops.aten.view.default(permute_142, [48, 3, 512, 512]);  permute_142 = None
        constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(view_162, [0, 0, 0, 1], 0.0);  view_162 = None
        view_163 = torch.ops.aten.view.default(constant_pad_nd_8, [48, 3, 512, 513]);  constant_pad_nd_8 = None
        full_18 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_465 = torch.ops.aten.slice.Tensor(view_163, 2, 0, 256)
        slice_466 = torch.ops.aten.slice.Tensor(slice_465, 3, 0, 257);  slice_465 = None
        slice_468 = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_470 = torch.ops.aten.slice.Tensor(slice_468, 3, 256, 9223372036854775807);  slice_468 = None
        copy_24 = torch.ops.aten.copy.default(slice_470, slice_466);  slice_470 = slice_466 = None
        slice_472 = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_scatter_88 = torch.ops.aten.slice_scatter.default(slice_472, copy_24, 3, 256, 9223372036854775807);  slice_472 = copy_24 = None
        slice_scatter_90 = torch.ops.aten.slice_scatter.default(full_18, slice_scatter_88, 1, 0, -1);  full_18 = slice_scatter_88 = None
        select_40 = torch.ops.aten.select.int(view_163, 1, -1)
        slice_479 = torch.ops.aten.slice.Tensor(select_40, 1, 256, 9223372036854775807);  select_40 = None
        slice_480 = torch.ops.aten.slice.Tensor(slice_479, 2, 0, 257);  slice_479 = None
        select_42 = torch.ops.aten.select.int(slice_scatter_90, 1, -1)
        slice_486 = torch.ops.aten.slice.Tensor(select_42, 2, 256, 9223372036854775807);  select_42 = None
        copy_25 = torch.ops.aten.copy.default(slice_486, slice_480);  slice_486 = slice_480 = None
        select_43 = torch.ops.aten.select.int(slice_scatter_90, 1, -1)
        slice_scatter_92 = torch.ops.aten.slice_scatter.default(select_43, copy_25, 2, 256, 9223372036854775807);  select_43 = copy_25 = None
        select_scatter_8 = torch.ops.aten.select_scatter.default(slice_scatter_90, slice_scatter_92, 1, -1);  slice_scatter_90 = slice_scatter_92 = None
        slice_494 = torch.ops.aten.slice.Tensor(view_163, 2, -257, -1)
        slice_495 = torch.ops.aten.slice.Tensor(slice_494, 3, 257, 9223372036854775807);  slice_494 = None
        slice_501 = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_503 = torch.ops.aten.slice.Tensor(slice_501, 3, 0, 256);  slice_501 = None
        copy_26 = torch.ops.aten.copy.default(slice_503, slice_495);  slice_503 = slice_495 = None
        slice_505 = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_scatter_95 = torch.ops.aten.slice_scatter.default(slice_505, copy_26, 3, 0, 256);  slice_505 = copy_26 = None
        slice_scatter_97 = torch.ops.aten.slice_scatter.default(select_scatter_8, slice_scatter_95, 1, 1, 9223372036854775807);  select_scatter_8 = slice_scatter_95 = None
        select_45 = torch.ops.aten.select.int(view_163, 1, 0);  view_163 = None
        slice_512 = torch.ops.aten.slice.Tensor(select_45, 1, 0, 255);  select_45 = None
        slice_513 = torch.ops.aten.slice.Tensor(slice_512, 2, -255, 9223372036854775807);  slice_512 = None
        select_47 = torch.ops.aten.select.int(slice_scatter_97, 1, 0)
        slice_518 = torch.ops.aten.slice.Tensor(select_47, 1, 1, 256);  select_47 = None
        slice_519 = torch.ops.aten.slice.Tensor(slice_518, 2, 1, 256);  slice_518 = None
        copy_27 = torch.ops.aten.copy.default(slice_519, slice_513);  slice_519 = slice_513 = None
        select_48 = torch.ops.aten.select.int(slice_scatter_97, 1, 0)
        slice_521 = torch.ops.aten.slice.Tensor(select_48, 1, 1, 256)
        slice_scatter_99 = torch.ops.aten.slice_scatter.default(slice_521, copy_27, 2, 1, 256);  slice_521 = copy_27 = None
        slice_scatter_100 = torch.ops.aten.slice_scatter.default(select_48, slice_scatter_99, 1, 1, 256);  select_48 = slice_scatter_99 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(slice_scatter_97, slice_scatter_100, 1, 0);  slice_scatter_97 = slice_scatter_100 = None
        full_default_20 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_8 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(iota_8, -2);  iota_8 = None
        iota_9 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        sub_17 = torch.ops.aten.sub.Tensor(unsqueeze_41, unsqueeze_42);  unsqueeze_41 = unsqueeze_42 = None
        le_4 = torch.ops.aten.le.Scalar(sub_17, 0);  sub_17 = None
        full_default_21 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16 = torch.ops.aten.where.self(le_4, full_default_20, full_default_21);  le_4 = full_default_20 = full_default_21 = None
        rev_8 = torch.ops.prims.rev.default(where_16, [0]);  where_16 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(rev_8, 0);  rev_8 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(unsqueeze_43, 2);  unsqueeze_43 = None
        rev_9 = torch.ops.prims.rev.default(unsqueeze_44, [1, 3])
        expand_8 = torch.ops.aten.expand.default(unsqueeze_44, [4, 256, 12, 257]);  unsqueeze_44 = None
        view_166 = torch.ops.aten.view.default(select_scatter_9, [4, 12, 1024, 513])
        permute_145 = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        slice_532 = torch.ops.aten.slice.Tensor(permute_145, 1, 0, 256);  permute_145 = None
        slice_534 = torch.ops.aten.slice.Tensor(slice_532, 3, 0, 257);  slice_532 = None
        full_default_22 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(expand_8, torch.bool);  expand_8 = None
        where_17 = torch.ops.aten.where.self(convert_element_type_10, full_default_22, slice_534);  convert_element_type_10 = full_default_22 = slice_534 = None
        view_167 = torch.ops.aten.view.default(select_scatter_9, [4, 12, 1024, 513])
        permute_146 = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        slice_540 = torch.ops.aten.slice.Tensor(permute_146, 1, 0, 256);  permute_146 = None
        slice_542 = torch.ops.aten.slice.Tensor(slice_540, 3, 0, 257);  slice_540 = None
        copy_28 = torch.ops.aten.copy.default(slice_542, where_17);  slice_542 = where_17 = None
        view_168 = torch.ops.aten.view.default(select_scatter_9, [4, 12, 1024, 513]);  select_scatter_9 = None
        permute_147 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        slice_544 = torch.ops.aten.slice.Tensor(permute_147, 1, 0, 256)
        slice_scatter_102 = torch.ops.aten.slice_scatter.default(slice_544, copy_28, 3, 0, 257);  slice_544 = copy_28 = None
        slice_scatter_104 = torch.ops.aten.slice_scatter.default(permute_147, slice_scatter_102, 1, 0, 256);  permute_147 = slice_scatter_102 = None
        permute_148 = torch.ops.aten.permute.default(slice_scatter_104, [0, 2, 1, 3]);  slice_scatter_104 = None
        view_169 = torch.ops.aten.view.default(permute_148, [48, 4, 256, 513]);  permute_148 = None
        expand_9 = torch.ops.aten.expand.default(rev_9, [4, 256, 12, 257]);  rev_9 = None
        view_171 = torch.ops.aten.view.default(view_169, [4, 12, 1024, 513])
        permute_150 = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        slice_555 = torch.ops.aten.slice.Tensor(permute_150, 1, -256, 9223372036854775807);  permute_150 = None
        slice_557 = torch.ops.aten.slice.Tensor(slice_555, 3, -257, 9223372036854775807);  slice_555 = None
        full_default_23 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(expand_9, torch.bool);  expand_9 = None
        where_18 = torch.ops.aten.where.self(convert_element_type_11, full_default_23, slice_557);  convert_element_type_11 = full_default_23 = slice_557 = None
        view_172 = torch.ops.aten.view.default(view_169, [4, 12, 1024, 513])
        permute_151 = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        slice_563 = torch.ops.aten.slice.Tensor(permute_151, 1, -256, 9223372036854775807);  permute_151 = None
        slice_565 = torch.ops.aten.slice.Tensor(slice_563, 3, -257, 9223372036854775807);  slice_563 = None
        copy_29 = torch.ops.aten.copy.default(slice_565, where_18);  slice_565 = where_18 = None
        view_173 = torch.ops.aten.view.default(view_169, [4, 12, 1024, 513]);  view_169 = None
        permute_152 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        slice_567 = torch.ops.aten.slice.Tensor(permute_152, 1, -256, 9223372036854775807)
        slice_scatter_106 = torch.ops.aten.slice_scatter.default(slice_567, copy_29, 3, -257, 9223372036854775807);  slice_567 = copy_29 = None
        slice_scatter_108 = torch.ops.aten.slice_scatter.default(permute_152, slice_scatter_106, 1, -256, 9223372036854775807);  permute_152 = slice_scatter_106 = None
        permute_153 = torch.ops.aten.permute.default(slice_scatter_108, [0, 2, 1, 3]);  slice_scatter_108 = None
        view_174 = torch.ops.aten.view.default(permute_153, [48, 4, 256, 513]);  permute_153 = None
        ne_2 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(ne_2, 2);  ne_2 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(unsqueeze_45, 3);  unsqueeze_45 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(unsqueeze_46, torch.float32)
        full_default_24 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(unsqueeze_46, full_default_24, convert_element_type_12);  unsqueeze_46 = full_default_24 = convert_element_type_12 = None
        full_22 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_155 = torch.ops.aten.permute.default(full_22, [0, 2, 1, 3]);  full_22 = None
        view_176 = torch.ops.aten.view.default(permute_155, [4, 1024, 1]);  permute_155 = None
        permute_156 = torch.ops.aten.permute.default(where_19, [0, 2, 1, 3]);  where_19 = None
        view_177 = torch.ops.aten.view.default(permute_156, [4, 1024, 1]);  permute_156 = None
        view_178 = torch.ops.aten.view.default(view_176, [4, 2, 512, 1]);  view_176 = None
        as_strided_15 = torch.ops.aten.as_strided.default(view_178, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_178 = None
        view_179 = torch.ops.aten.view.default(view_177, [4, 2, 512, 1]);  view_177 = None
        as_strided_16 = torch.ops.aten.as_strided.default(view_179, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_179 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
        permute_157 = torch.ops.aten.permute.default(unsqueeze_47, [0, 1, 2, 4, 3]);  unsqueeze_47 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
        permute_158 = torch.ops.aten.permute.default(unsqueeze_48, [0, 1, 4, 2, 3]);  unsqueeze_48 = None
        mul_16 = torch.ops.aten.mul.Tensor(permute_157, permute_158);  permute_157 = permute_158 = None
        view_180 = torch.ops.aten.view.default(mul_16, [4, 3, 512, 512]);  mul_16 = None
        constant_pad_nd_9 = torch.ops.aten.constant_pad_nd.default(view_180, [0, 0, 0, 1], 0.0);  view_180 = None
        view_181 = torch.ops.aten.view.default(constant_pad_nd_9, [4, 3, 512, 513]);  constant_pad_nd_9 = None
        full_23 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_577 = torch.ops.aten.slice.Tensor(view_181, 2, 0, 256)
        slice_578 = torch.ops.aten.slice.Tensor(slice_577, 3, 0, 257);  slice_577 = None
        slice_580 = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_582 = torch.ops.aten.slice.Tensor(slice_580, 3, 256, 9223372036854775807);  slice_580 = None
        copy_30 = torch.ops.aten.copy.default(slice_582, slice_578);  slice_582 = slice_578 = None
        slice_584 = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_scatter_110 = torch.ops.aten.slice_scatter.default(slice_584, copy_30, 3, 256, 9223372036854775807);  slice_584 = copy_30 = None
        slice_scatter_112 = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_110, 1, 0, -1);  full_23 = slice_scatter_110 = None
        select_50 = torch.ops.aten.select.int(view_181, 1, -1)
        slice_591 = torch.ops.aten.slice.Tensor(select_50, 1, 256, 9223372036854775807);  select_50 = None
        slice_592 = torch.ops.aten.slice.Tensor(slice_591, 2, 0, 257);  slice_591 = None
        select_52 = torch.ops.aten.select.int(slice_scatter_112, 1, -1)
        slice_598 = torch.ops.aten.slice.Tensor(select_52, 2, 256, 9223372036854775807);  select_52 = None
        copy_31 = torch.ops.aten.copy.default(slice_598, slice_592);  slice_598 = slice_592 = None
        select_53 = torch.ops.aten.select.int(slice_scatter_112, 1, -1)
        slice_scatter_114 = torch.ops.aten.slice_scatter.default(select_53, copy_31, 2, 256, 9223372036854775807);  select_53 = copy_31 = None
        select_scatter_10 = torch.ops.aten.select_scatter.default(slice_scatter_112, slice_scatter_114, 1, -1);  slice_scatter_112 = slice_scatter_114 = None
        slice_606 = torch.ops.aten.slice.Tensor(view_181, 2, -257, -1)
        slice_607 = torch.ops.aten.slice.Tensor(slice_606, 3, 257, 9223372036854775807);  slice_606 = None
        slice_613 = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_615 = torch.ops.aten.slice.Tensor(slice_613, 3, 0, 256);  slice_613 = None
        copy_32 = torch.ops.aten.copy.default(slice_615, slice_607);  slice_615 = slice_607 = None
        slice_617 = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_scatter_117 = torch.ops.aten.slice_scatter.default(slice_617, copy_32, 3, 0, 256);  slice_617 = copy_32 = None
        slice_scatter_119 = torch.ops.aten.slice_scatter.default(select_scatter_10, slice_scatter_117, 1, 1, 9223372036854775807);  select_scatter_10 = slice_scatter_117 = None
        select_55 = torch.ops.aten.select.int(view_181, 1, 0);  view_181 = None
        slice_624 = torch.ops.aten.slice.Tensor(select_55, 1, 0, 255);  select_55 = None
        slice_625 = torch.ops.aten.slice.Tensor(slice_624, 2, -255, 9223372036854775807);  slice_624 = None
        select_57 = torch.ops.aten.select.int(slice_scatter_119, 1, 0)
        slice_630 = torch.ops.aten.slice.Tensor(select_57, 1, 1, 256);  select_57 = None
        slice_631 = torch.ops.aten.slice.Tensor(slice_630, 2, 1, 256);  slice_630 = None
        copy_33 = torch.ops.aten.copy.default(slice_631, slice_625);  slice_631 = slice_625 = None
        select_58 = torch.ops.aten.select.int(slice_scatter_119, 1, 0)
        slice_633 = torch.ops.aten.slice.Tensor(select_58, 1, 1, 256)
        slice_scatter_121 = torch.ops.aten.slice_scatter.default(slice_633, copy_33, 2, 1, 256);  slice_633 = copy_33 = None
        slice_scatter_122 = torch.ops.aten.slice_scatter.default(select_58, slice_scatter_121, 1, 1, 256);  select_58 = slice_scatter_121 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(slice_scatter_119, slice_scatter_122, 1, 0);  slice_scatter_119 = slice_scatter_122 = None
        full_default_25 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_10 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(iota_10, -2);  iota_10 = None
        iota_11 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        sub_19 = torch.ops.aten.sub.Tensor(unsqueeze_49, unsqueeze_50);  unsqueeze_49 = unsqueeze_50 = None
        le_5 = torch.ops.aten.le.Scalar(sub_19, 0);  sub_19 = None
        full_default_26 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20 = torch.ops.aten.where.self(le_5, full_default_25, full_default_26);  le_5 = full_default_25 = full_default_26 = None
        rev_10 = torch.ops.prims.rev.default(where_20, [0]);  where_20 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(rev_10, 0);  rev_10 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
        rev_11 = torch.ops.prims.rev.default(unsqueeze_52, [1, 3])
        expand_10 = torch.ops.aten.expand.default(unsqueeze_52, [4, 256, 1, 257]);  unsqueeze_52 = None
        view_184 = torch.ops.aten.view.default(select_scatter_11, [4, 1, 1024, 513])
        permute_161 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        slice_644 = torch.ops.aten.slice.Tensor(permute_161, 1, 0, 256);  permute_161 = None
        slice_646 = torch.ops.aten.slice.Tensor(slice_644, 3, 0, 257);  slice_644 = None
        full_default_27 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(expand_10, torch.bool);  expand_10 = None
        where_21 = torch.ops.aten.where.self(convert_element_type_13, full_default_27, slice_646);  convert_element_type_13 = full_default_27 = slice_646 = None
        view_185 = torch.ops.aten.view.default(select_scatter_11, [4, 1, 1024, 513])
        permute_162 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        slice_652 = torch.ops.aten.slice.Tensor(permute_162, 1, 0, 256);  permute_162 = None
        slice_654 = torch.ops.aten.slice.Tensor(slice_652, 3, 0, 257);  slice_652 = None
        copy_34 = torch.ops.aten.copy.default(slice_654, where_21);  slice_654 = where_21 = None
        view_186 = torch.ops.aten.view.default(select_scatter_11, [4, 1, 1024, 513]);  select_scatter_11 = None
        permute_163 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        slice_656 = torch.ops.aten.slice.Tensor(permute_163, 1, 0, 256)
        slice_scatter_124 = torch.ops.aten.slice_scatter.default(slice_656, copy_34, 3, 0, 257);  slice_656 = copy_34 = None
        slice_scatter_126 = torch.ops.aten.slice_scatter.default(permute_163, slice_scatter_124, 1, 0, 256);  permute_163 = slice_scatter_124 = None
        permute_164 = torch.ops.aten.permute.default(slice_scatter_126, [0, 2, 1, 3]);  slice_scatter_126 = None
        view_187 = torch.ops.aten.view.default(permute_164, [4, 4, 256, 513]);  permute_164 = None
        expand_11 = torch.ops.aten.expand.default(rev_11, [4, 256, 1, 257]);  rev_11 = None
        view_189 = torch.ops.aten.view.default(view_187, [4, 1, 1024, 513])
        permute_166 = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        slice_667 = torch.ops.aten.slice.Tensor(permute_166, 1, -256, 9223372036854775807);  permute_166 = None
        slice_669 = torch.ops.aten.slice.Tensor(slice_667, 3, -257, 9223372036854775807);  slice_667 = None
        full_default_28 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(expand_11, torch.bool);  expand_11 = None
        where_22 = torch.ops.aten.where.self(convert_element_type_14, full_default_28, slice_669);  convert_element_type_14 = full_default_28 = slice_669 = None
        view_190 = torch.ops.aten.view.default(view_187, [4, 1, 1024, 513])
        permute_167 = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        slice_675 = torch.ops.aten.slice.Tensor(permute_167, 1, -256, 9223372036854775807);  permute_167 = None
        slice_677 = torch.ops.aten.slice.Tensor(slice_675, 3, -257, 9223372036854775807);  slice_675 = None
        copy_35 = torch.ops.aten.copy.default(slice_677, where_22);  slice_677 = where_22 = None
        view_191 = torch.ops.aten.view.default(view_187, [4, 1, 1024, 513]);  view_187 = None
        permute_168 = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        slice_679 = torch.ops.aten.slice.Tensor(permute_168, 1, -256, 9223372036854775807)
        slice_scatter_128 = torch.ops.aten.slice_scatter.default(slice_679, copy_35, 3, -257, 9223372036854775807);  slice_679 = copy_35 = None
        slice_scatter_130 = torch.ops.aten.slice_scatter.default(permute_168, slice_scatter_128, 1, -256, 9223372036854775807);  permute_168 = slice_scatter_128 = None
        permute_169 = torch.ops.aten.permute.default(slice_scatter_130, [0, 2, 1, 3]);  slice_scatter_130 = None
        view_192 = torch.ops.aten.view.default(permute_169, [4, 4, 256, 513]);  permute_169 = None
        view_194 = torch.ops.aten.view.default(view_174, [4, 12, 1024, 513]);  view_174 = None
        permute_171 = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        view_195 = torch.ops.aten.view.default(view_192, [4, 1, 1024, 513]);  view_192 = None
        permute_172 = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        add_35 = torch.ops.aten.add.Tensor(permute_171, permute_172);  permute_171 = permute_172 = None
        permute_173 = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
        view_197 = torch.ops.aten.view.default(permute_173, [48, 4, 256, 513]);  permute_173 = None
        view_198 = torch.ops.aten.view.default(view_197, [4, 12, 1024, 513]);  view_197 = None
        permute_174 = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        clone_31 = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        amax_2 = torch.ops.aten.amax.default(clone_31, [-1], True)
        sub_20 = torch.ops.aten.sub.Tensor(clone_31, amax_2);  clone_31 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        full_default_29 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23 = torch.ops.aten.where.self(unsqueeze_54, full_default_29, div_27);  unsqueeze_54 = full_default_29 = div_27 = None
        view_199 = torch.ops.aten.view.default(add_32, [1024, 4, 12, 64]);  add_32 = None
        permute_175 = torch.ops.aten.permute.default(view_199, [1, 0, 2, 3]);  view_199 = None
        permute_176 = torch.ops.aten.permute.default(where_23, [0, 2, 1, 3]);  where_23 = None
        clone_33 = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_200 = torch.ops.aten.view.default(clone_33, [48, 4, 256, 513]);  clone_33 = None
        permute_177 = torch.ops.aten.permute.default(permute_175, [0, 2, 1, 3]);  permute_175 = None
        view_201 = torch.ops.aten.view.default(permute_177, [48, 1024, 64]);  permute_177 = None
        constant_pad_nd_10 = torch.ops.aten.constant_pad_nd.default(view_201, [0, 0, 256, 256], -1.0);  view_201 = None
        as_strided_17 = torch.ops.aten.as_strided.default(constant_pad_nd_10, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_10 = None
        constant_pad_nd_11 = torch.ops.aten.constant_pad_nd.default(view_200, [0, 257], 0.0);  view_200 = None
        view_202 = torch.ops.aten.view.default(constant_pad_nd_11, [48, 4, -1]);  constant_pad_nd_11 = None
        slice_689 = torch.ops.aten.slice.Tensor(view_202, 2, 0, -256);  view_202 = None
        view_203 = torch.ops.aten.view.default(slice_689, [48, 4, 256, 769]);  slice_689 = None
        slice_693 = torch.ops.aten.slice.Tensor(view_203, 3, 0, -1);  view_203 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(slice_693, 4);  slice_693 = None
        permute_178 = torch.ops.aten.permute.default(unsqueeze_55, [0, 1, 2, 4, 3]);  unsqueeze_55 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        permute_179 = torch.ops.aten.permute.default(unsqueeze_56, [0, 1, 4, 3, 2]);  unsqueeze_56 = None
        permute_180 = torch.ops.aten.permute.default(permute_178, [0, 1, 2, 4, 3]);  permute_178 = None
        view_204 = torch.ops.aten.view.default(permute_180, [192, 256, 768]);  permute_180 = None
        permute_181 = torch.ops.aten.permute.default(permute_179, [0, 1, 4, 3, 2]);  permute_179 = None
        clone_34 = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_205 = torch.ops.aten.view.default(clone_34, [192, 768, 64]);  clone_34 = None
        bmm_5 = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206 = torch.ops.aten.view.default(bmm_5, [48, 4, 256, 1, 64]);  bmm_5 = None
        permute_182 = torch.ops.aten.permute.default(view_206, [0, 1, 2, 4, 3]);  view_206 = None
        view_207 = torch.ops.aten.view.default(permute_182, [48, 4, 256, 64]);  permute_182 = None
        view_208 = torch.ops.aten.view.default(view_207, [4, 12, 1024, 64]);  view_207 = None
        permute_183 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        permute_184 = torch.ops.aten.permute.default(permute_183, [1, 0, 2, 3]);  permute_183 = None
        clone_35 = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_209 = torch.ops.aten.view.default(clone_35, [1024, 4, 768]);  clone_35 = None
        permute_185 = torch.ops.aten.permute.default(view_209, [1, 0, 2]);  view_209 = None
        permute_186 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        clone_36 = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_210 = torch.ops.aten.view.default(clone_36, [4096, 768]);  clone_36 = None
        mm_11 = torch.ops.aten.mm.default(view_210, permute_186);  view_210 = permute_186 = None
        view_211 = torch.ops.aten.view.default(mm_11, [4, 1024, 768]);  mm_11 = None
        add_37 = torch.ops.aten.add.Tensor(view_211, arg42_1);  view_211 = arg42_1 = None
        add_38 = torch.ops.aten.add.Tensor(add_37, add_29);  add_37 = add_29 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_38, getitem_9);  add_38 = getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = rsqrt_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg43_1);  mul_17 = arg43_1 = None
        add_40 = torch.ops.aten.add.Tensor(mul_18, arg44_1);  mul_18 = arg44_1 = None
        view_212 = torch.ops.aten.view.default(add_40, [4096, 768])
        permute_187 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg46_1, view_212, permute_187);  arg46_1 = view_212 = permute_187 = None
        view_213 = torch.ops.aten.view.default(addmm_4, [4, 1024, 3072]);  addmm_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_213, 0.5)
        mul_20 = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
        erf_2 = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_41 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_19, add_41);  mul_19 = add_41 = None
        view_214 = torch.ops.aten.view.default(mul_21, [4096, 3072]);  mul_21 = None
        permute_188 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg48_1, view_214, permute_188);  arg48_1 = view_214 = permute_188 = None
        view_215 = torch.ops.aten.view.default(addmm_5, [4, 1024, 768]);  addmm_5 = None
        add_42 = torch.ops.aten.add.Tensor(view_215, add_40);  view_215 = add_40 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_42, getitem_11);  add_42 = getitem_11 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = rsqrt_5 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg49_1);  mul_22 = arg49_1 = None
        add_44 = torch.ops.aten.add.Tensor(mul_23, arg50_1);  mul_23 = arg50_1 = None
        permute_189 = torch.ops.aten.permute.default(add_44, [1, 0, 2])
        permute_190 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        clone_39 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format)
        view_216 = torch.ops.aten.view.default(clone_39, [4096, 768]);  clone_39 = None
        mm_12 = torch.ops.aten.mm.default(view_216, permute_190);  view_216 = permute_190 = None
        view_217 = torch.ops.aten.view.default(mm_12, [1024, 4, 768]);  mm_12 = None
        add_45 = torch.ops.aten.add.Tensor(view_217, arg52_1);  view_217 = arg52_1 = None
        permute_191 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        clone_40 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format)
        view_218 = torch.ops.aten.view.default(clone_40, [4096, 768]);  clone_40 = None
        mm_13 = torch.ops.aten.mm.default(view_218, permute_191);  view_218 = permute_191 = None
        view_219 = torch.ops.aten.view.default(mm_13, [1024, 4, 768]);  mm_13 = None
        add_46 = torch.ops.aten.add.Tensor(view_219, arg54_1);  view_219 = arg54_1 = None
        permute_192 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        clone_41 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_220 = torch.ops.aten.view.default(clone_41, [4096, 768]);  clone_41 = None
        mm_14 = torch.ops.aten.mm.default(view_220, permute_192);  view_220 = permute_192 = None
        view_221 = torch.ops.aten.view.default(mm_14, [1024, 4, 768]);  mm_14 = None
        add_47 = torch.ops.aten.add.Tensor(view_221, arg56_1);  view_221 = arg56_1 = None
        div_30 = torch.ops.aten.div.Tensor(add_45, 8.0);  add_45 = None
        view_223 = torch.ops.aten.view.default(add_46, [1024, 4, 12, 64]);  add_46 = None
        permute_194 = torch.ops.aten.permute.default(view_223, [1, 0, 2, 3]);  view_223 = None
        permute_196 = torch.ops.aten.permute.default(permute_194, [0, 2, 1, 3]);  permute_194 = None
        view_225 = torch.ops.aten.view.default(permute_196, [48, 1024, 64]);  permute_196 = None
        view_227 = torch.ops.aten.view.default(view_225, [48, 2, 512, 64]);  view_225 = None
        as_strided_19 = torch.ops.aten.as_strided.default(view_227, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_227 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        permute_198 = torch.ops.aten.permute.default(unsqueeze_58, [0, 1, 4, 2, 3]);  unsqueeze_58 = None
        view_228 = torch.ops.aten.view.default(div_30, [1024, 4, 12, 64]);  div_30 = None
        permute_200 = torch.ops.aten.permute.default(view_228, [1, 0, 2, 3]);  view_228 = None
        permute_201 = torch.ops.aten.permute.default(permute_200, [0, 2, 1, 3]);  permute_200 = None
        view_229 = torch.ops.aten.view.default(permute_201, [48, 1024, 64]);  permute_201 = None
        view_230 = torch.ops.aten.view.default(view_229, [48, 2, 512, 64]);  view_229 = None
        as_strided_20 = torch.ops.aten.as_strided.default(view_230, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_230 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(as_strided_20, 4);  as_strided_20 = None
        permute_202 = torch.ops.aten.permute.default(unsqueeze_59, [0, 1, 2, 4, 3]);  unsqueeze_59 = None
        permute_203 = torch.ops.aten.permute.default(permute_202, [0, 1, 2, 4, 3]);  permute_202 = None
        clone_42 = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_231 = torch.ops.aten.view.default(clone_42, [144, 512, 64]);  clone_42 = None
        permute_204 = torch.ops.aten.permute.default(permute_198, [0, 1, 4, 3, 2]);  permute_198 = None
        clone_43 = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_232 = torch.ops.aten.view.default(clone_43, [144, 64, 512]);  clone_43 = None
        bmm_6 = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
        view_233 = torch.ops.aten.view.default(bmm_6, [48, 3, 512, 1, 512]);  bmm_6 = None
        permute_205 = torch.ops.aten.permute.default(view_233, [0, 1, 2, 4, 3]);  view_233 = None
        view_234 = torch.ops.aten.view.default(permute_205, [48, 3, 512, 512]);  permute_205 = None
        constant_pad_nd_12 = torch.ops.aten.constant_pad_nd.default(view_234, [0, 0, 0, 1], 0.0);  view_234 = None
        view_235 = torch.ops.aten.view.default(constant_pad_nd_12, [48, 3, 512, 513]);  constant_pad_nd_12 = None
        full_27 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_696 = torch.ops.aten.slice.Tensor(view_235, 2, 0, 256)
        slice_697 = torch.ops.aten.slice.Tensor(slice_696, 3, 0, 257);  slice_696 = None
        slice_699 = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_701 = torch.ops.aten.slice.Tensor(slice_699, 3, 256, 9223372036854775807);  slice_699 = None
        copy_36 = torch.ops.aten.copy.default(slice_701, slice_697);  slice_701 = slice_697 = None
        slice_703 = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_scatter_132 = torch.ops.aten.slice_scatter.default(slice_703, copy_36, 3, 256, 9223372036854775807);  slice_703 = copy_36 = None
        slice_scatter_134 = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_132, 1, 0, -1);  full_27 = slice_scatter_132 = None
        select_60 = torch.ops.aten.select.int(view_235, 1, -1)
        slice_710 = torch.ops.aten.slice.Tensor(select_60, 1, 256, 9223372036854775807);  select_60 = None
        slice_711 = torch.ops.aten.slice.Tensor(slice_710, 2, 0, 257);  slice_710 = None
        select_62 = torch.ops.aten.select.int(slice_scatter_134, 1, -1)
        slice_717 = torch.ops.aten.slice.Tensor(select_62, 2, 256, 9223372036854775807);  select_62 = None
        copy_37 = torch.ops.aten.copy.default(slice_717, slice_711);  slice_717 = slice_711 = None
        select_63 = torch.ops.aten.select.int(slice_scatter_134, 1, -1)
        slice_scatter_136 = torch.ops.aten.slice_scatter.default(select_63, copy_37, 2, 256, 9223372036854775807);  select_63 = copy_37 = None
        select_scatter_12 = torch.ops.aten.select_scatter.default(slice_scatter_134, slice_scatter_136, 1, -1);  slice_scatter_134 = slice_scatter_136 = None
        slice_725 = torch.ops.aten.slice.Tensor(view_235, 2, -257, -1)
        slice_726 = torch.ops.aten.slice.Tensor(slice_725, 3, 257, 9223372036854775807);  slice_725 = None
        slice_732 = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_734 = torch.ops.aten.slice.Tensor(slice_732, 3, 0, 256);  slice_732 = None
        copy_38 = torch.ops.aten.copy.default(slice_734, slice_726);  slice_734 = slice_726 = None
        slice_736 = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_scatter_139 = torch.ops.aten.slice_scatter.default(slice_736, copy_38, 3, 0, 256);  slice_736 = copy_38 = None
        slice_scatter_141 = torch.ops.aten.slice_scatter.default(select_scatter_12, slice_scatter_139, 1, 1, 9223372036854775807);  select_scatter_12 = slice_scatter_139 = None
        select_65 = torch.ops.aten.select.int(view_235, 1, 0);  view_235 = None
        slice_743 = torch.ops.aten.slice.Tensor(select_65, 1, 0, 255);  select_65 = None
        slice_744 = torch.ops.aten.slice.Tensor(slice_743, 2, -255, 9223372036854775807);  slice_743 = None
        select_67 = torch.ops.aten.select.int(slice_scatter_141, 1, 0)
        slice_749 = torch.ops.aten.slice.Tensor(select_67, 1, 1, 256);  select_67 = None
        slice_750 = torch.ops.aten.slice.Tensor(slice_749, 2, 1, 256);  slice_749 = None
        copy_39 = torch.ops.aten.copy.default(slice_750, slice_744);  slice_750 = slice_744 = None
        select_68 = torch.ops.aten.select.int(slice_scatter_141, 1, 0)
        slice_752 = torch.ops.aten.slice.Tensor(select_68, 1, 1, 256)
        slice_scatter_143 = torch.ops.aten.slice_scatter.default(slice_752, copy_39, 2, 1, 256);  slice_752 = copy_39 = None
        slice_scatter_144 = torch.ops.aten.slice_scatter.default(select_68, slice_scatter_143, 1, 1, 256);  select_68 = slice_scatter_143 = None
        select_scatter_13 = torch.ops.aten.select_scatter.default(slice_scatter_141, slice_scatter_144, 1, 0);  slice_scatter_141 = slice_scatter_144 = None
        full_default_30 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_12 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(iota_12, -2);  iota_12 = None
        iota_13 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        sub_25 = torch.ops.aten.sub.Tensor(unsqueeze_60, unsqueeze_61);  unsqueeze_60 = unsqueeze_61 = None
        le_6 = torch.ops.aten.le.Scalar(sub_25, 0);  sub_25 = None
        full_default_31 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24 = torch.ops.aten.where.self(le_6, full_default_30, full_default_31);  le_6 = full_default_30 = full_default_31 = None
        rev_12 = torch.ops.prims.rev.default(where_24, [0]);  where_24 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(rev_12, 0);  rev_12 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, 2);  unsqueeze_62 = None
        rev_13 = torch.ops.prims.rev.default(unsqueeze_63, [1, 3])
        expand_12 = torch.ops.aten.expand.default(unsqueeze_63, [4, 256, 12, 257]);  unsqueeze_63 = None
        view_238 = torch.ops.aten.view.default(select_scatter_13, [4, 12, 1024, 513])
        permute_208 = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        slice_763 = torch.ops.aten.slice.Tensor(permute_208, 1, 0, 256);  permute_208 = None
        slice_765 = torch.ops.aten.slice.Tensor(slice_763, 3, 0, 257);  slice_763 = None
        full_default_32 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(expand_12, torch.bool);  expand_12 = None
        where_25 = torch.ops.aten.where.self(convert_element_type_15, full_default_32, slice_765);  convert_element_type_15 = full_default_32 = slice_765 = None
        view_239 = torch.ops.aten.view.default(select_scatter_13, [4, 12, 1024, 513])
        permute_209 = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        slice_771 = torch.ops.aten.slice.Tensor(permute_209, 1, 0, 256);  permute_209 = None
        slice_773 = torch.ops.aten.slice.Tensor(slice_771, 3, 0, 257);  slice_771 = None
        copy_40 = torch.ops.aten.copy.default(slice_773, where_25);  slice_773 = where_25 = None
        view_240 = torch.ops.aten.view.default(select_scatter_13, [4, 12, 1024, 513]);  select_scatter_13 = None
        permute_210 = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        slice_775 = torch.ops.aten.slice.Tensor(permute_210, 1, 0, 256)
        slice_scatter_146 = torch.ops.aten.slice_scatter.default(slice_775, copy_40, 3, 0, 257);  slice_775 = copy_40 = None
        slice_scatter_148 = torch.ops.aten.slice_scatter.default(permute_210, slice_scatter_146, 1, 0, 256);  permute_210 = slice_scatter_146 = None
        permute_211 = torch.ops.aten.permute.default(slice_scatter_148, [0, 2, 1, 3]);  slice_scatter_148 = None
        view_241 = torch.ops.aten.view.default(permute_211, [48, 4, 256, 513]);  permute_211 = None
        expand_13 = torch.ops.aten.expand.default(rev_13, [4, 256, 12, 257]);  rev_13 = None
        view_243 = torch.ops.aten.view.default(view_241, [4, 12, 1024, 513])
        permute_213 = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        slice_786 = torch.ops.aten.slice.Tensor(permute_213, 1, -256, 9223372036854775807);  permute_213 = None
        slice_788 = torch.ops.aten.slice.Tensor(slice_786, 3, -257, 9223372036854775807);  slice_786 = None
        full_default_33 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(expand_13, torch.bool);  expand_13 = None
        where_26 = torch.ops.aten.where.self(convert_element_type_16, full_default_33, slice_788);  convert_element_type_16 = full_default_33 = slice_788 = None
        view_244 = torch.ops.aten.view.default(view_241, [4, 12, 1024, 513])
        permute_214 = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        slice_794 = torch.ops.aten.slice.Tensor(permute_214, 1, -256, 9223372036854775807);  permute_214 = None
        slice_796 = torch.ops.aten.slice.Tensor(slice_794, 3, -257, 9223372036854775807);  slice_794 = None
        copy_41 = torch.ops.aten.copy.default(slice_796, where_26);  slice_796 = where_26 = None
        view_245 = torch.ops.aten.view.default(view_241, [4, 12, 1024, 513]);  view_241 = None
        permute_215 = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        slice_798 = torch.ops.aten.slice.Tensor(permute_215, 1, -256, 9223372036854775807)
        slice_scatter_150 = torch.ops.aten.slice_scatter.default(slice_798, copy_41, 3, -257, 9223372036854775807);  slice_798 = copy_41 = None
        slice_scatter_152 = torch.ops.aten.slice_scatter.default(permute_215, slice_scatter_150, 1, -256, 9223372036854775807);  permute_215 = slice_scatter_150 = None
        permute_216 = torch.ops.aten.permute.default(slice_scatter_152, [0, 2, 1, 3]);  slice_scatter_152 = None
        view_246 = torch.ops.aten.view.default(permute_216, [48, 4, 256, 513]);  permute_216 = None
        ne_3 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(ne_3, 2);  ne_3 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(unsqueeze_65, torch.float32)
        full_default_34 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27 = torch.ops.aten.where.self(unsqueeze_65, full_default_34, convert_element_type_17);  unsqueeze_65 = full_default_34 = convert_element_type_17 = None
        full_31 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_218 = torch.ops.aten.permute.default(full_31, [0, 2, 1, 3]);  full_31 = None
        view_248 = torch.ops.aten.view.default(permute_218, [4, 1024, 1]);  permute_218 = None
        permute_219 = torch.ops.aten.permute.default(where_27, [0, 2, 1, 3]);  where_27 = None
        view_249 = torch.ops.aten.view.default(permute_219, [4, 1024, 1]);  permute_219 = None
        view_250 = torch.ops.aten.view.default(view_248, [4, 2, 512, 1]);  view_248 = None
        as_strided_21 = torch.ops.aten.as_strided.default(view_250, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_250 = None
        view_251 = torch.ops.aten.view.default(view_249, [4, 2, 512, 1]);  view_249 = None
        as_strided_22 = torch.ops.aten.as_strided.default(view_251, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_251 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(as_strided_21, 4);  as_strided_21 = None
        permute_220 = torch.ops.aten.permute.default(unsqueeze_66, [0, 1, 2, 4, 3]);  unsqueeze_66 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(as_strided_22, 4);  as_strided_22 = None
        permute_221 = torch.ops.aten.permute.default(unsqueeze_67, [0, 1, 4, 2, 3]);  unsqueeze_67 = None
        mul_24 = torch.ops.aten.mul.Tensor(permute_220, permute_221);  permute_220 = permute_221 = None
        view_252 = torch.ops.aten.view.default(mul_24, [4, 3, 512, 512]);  mul_24 = None
        constant_pad_nd_13 = torch.ops.aten.constant_pad_nd.default(view_252, [0, 0, 0, 1], 0.0);  view_252 = None
        view_253 = torch.ops.aten.view.default(constant_pad_nd_13, [4, 3, 512, 513]);  constant_pad_nd_13 = None
        full_32 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_808 = torch.ops.aten.slice.Tensor(view_253, 2, 0, 256)
        slice_809 = torch.ops.aten.slice.Tensor(slice_808, 3, 0, 257);  slice_808 = None
        slice_811 = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_813 = torch.ops.aten.slice.Tensor(slice_811, 3, 256, 9223372036854775807);  slice_811 = None
        copy_42 = torch.ops.aten.copy.default(slice_813, slice_809);  slice_813 = slice_809 = None
        slice_815 = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_scatter_154 = torch.ops.aten.slice_scatter.default(slice_815, copy_42, 3, 256, 9223372036854775807);  slice_815 = copy_42 = None
        slice_scatter_156 = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_154, 1, 0, -1);  full_32 = slice_scatter_154 = None
        select_70 = torch.ops.aten.select.int(view_253, 1, -1)
        slice_822 = torch.ops.aten.slice.Tensor(select_70, 1, 256, 9223372036854775807);  select_70 = None
        slice_823 = torch.ops.aten.slice.Tensor(slice_822, 2, 0, 257);  slice_822 = None
        select_72 = torch.ops.aten.select.int(slice_scatter_156, 1, -1)
        slice_829 = torch.ops.aten.slice.Tensor(select_72, 2, 256, 9223372036854775807);  select_72 = None
        copy_43 = torch.ops.aten.copy.default(slice_829, slice_823);  slice_829 = slice_823 = None
        select_73 = torch.ops.aten.select.int(slice_scatter_156, 1, -1)
        slice_scatter_158 = torch.ops.aten.slice_scatter.default(select_73, copy_43, 2, 256, 9223372036854775807);  select_73 = copy_43 = None
        select_scatter_14 = torch.ops.aten.select_scatter.default(slice_scatter_156, slice_scatter_158, 1, -1);  slice_scatter_156 = slice_scatter_158 = None
        slice_837 = torch.ops.aten.slice.Tensor(view_253, 2, -257, -1)
        slice_838 = torch.ops.aten.slice.Tensor(slice_837, 3, 257, 9223372036854775807);  slice_837 = None
        slice_844 = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_846 = torch.ops.aten.slice.Tensor(slice_844, 3, 0, 256);  slice_844 = None
        copy_44 = torch.ops.aten.copy.default(slice_846, slice_838);  slice_846 = slice_838 = None
        slice_848 = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_scatter_161 = torch.ops.aten.slice_scatter.default(slice_848, copy_44, 3, 0, 256);  slice_848 = copy_44 = None
        slice_scatter_163 = torch.ops.aten.slice_scatter.default(select_scatter_14, slice_scatter_161, 1, 1, 9223372036854775807);  select_scatter_14 = slice_scatter_161 = None
        select_75 = torch.ops.aten.select.int(view_253, 1, 0);  view_253 = None
        slice_855 = torch.ops.aten.slice.Tensor(select_75, 1, 0, 255);  select_75 = None
        slice_856 = torch.ops.aten.slice.Tensor(slice_855, 2, -255, 9223372036854775807);  slice_855 = None
        select_77 = torch.ops.aten.select.int(slice_scatter_163, 1, 0)
        slice_861 = torch.ops.aten.slice.Tensor(select_77, 1, 1, 256);  select_77 = None
        slice_862 = torch.ops.aten.slice.Tensor(slice_861, 2, 1, 256);  slice_861 = None
        copy_45 = torch.ops.aten.copy.default(slice_862, slice_856);  slice_862 = slice_856 = None
        select_78 = torch.ops.aten.select.int(slice_scatter_163, 1, 0)
        slice_864 = torch.ops.aten.slice.Tensor(select_78, 1, 1, 256)
        slice_scatter_165 = torch.ops.aten.slice_scatter.default(slice_864, copy_45, 2, 1, 256);  slice_864 = copy_45 = None
        slice_scatter_166 = torch.ops.aten.slice_scatter.default(select_78, slice_scatter_165, 1, 1, 256);  select_78 = slice_scatter_165 = None
        select_scatter_15 = torch.ops.aten.select_scatter.default(slice_scatter_163, slice_scatter_166, 1, 0);  slice_scatter_163 = slice_scatter_166 = None
        full_default_35 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_14 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(iota_14, -2);  iota_14 = None
        iota_15 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        sub_27 = torch.ops.aten.sub.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
        le_7 = torch.ops.aten.le.Scalar(sub_27, 0);  sub_27 = None
        full_default_36 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28 = torch.ops.aten.where.self(le_7, full_default_35, full_default_36);  le_7 = full_default_35 = full_default_36 = None
        rev_14 = torch.ops.prims.rev.default(where_28, [0]);  where_28 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(rev_14, 0);  rev_14 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, 2);  unsqueeze_70 = None
        rev_15 = torch.ops.prims.rev.default(unsqueeze_71, [1, 3])
        expand_14 = torch.ops.aten.expand.default(unsqueeze_71, [4, 256, 1, 257]);  unsqueeze_71 = None
        view_256 = torch.ops.aten.view.default(select_scatter_15, [4, 1, 1024, 513])
        permute_224 = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        slice_875 = torch.ops.aten.slice.Tensor(permute_224, 1, 0, 256);  permute_224 = None
        slice_877 = torch.ops.aten.slice.Tensor(slice_875, 3, 0, 257);  slice_875 = None
        full_default_37 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(expand_14, torch.bool);  expand_14 = None
        where_29 = torch.ops.aten.where.self(convert_element_type_18, full_default_37, slice_877);  convert_element_type_18 = full_default_37 = slice_877 = None
        view_257 = torch.ops.aten.view.default(select_scatter_15, [4, 1, 1024, 513])
        permute_225 = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        slice_883 = torch.ops.aten.slice.Tensor(permute_225, 1, 0, 256);  permute_225 = None
        slice_885 = torch.ops.aten.slice.Tensor(slice_883, 3, 0, 257);  slice_883 = None
        copy_46 = torch.ops.aten.copy.default(slice_885, where_29);  slice_885 = where_29 = None
        view_258 = torch.ops.aten.view.default(select_scatter_15, [4, 1, 1024, 513]);  select_scatter_15 = None
        permute_226 = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        slice_887 = torch.ops.aten.slice.Tensor(permute_226, 1, 0, 256)
        slice_scatter_168 = torch.ops.aten.slice_scatter.default(slice_887, copy_46, 3, 0, 257);  slice_887 = copy_46 = None
        slice_scatter_170 = torch.ops.aten.slice_scatter.default(permute_226, slice_scatter_168, 1, 0, 256);  permute_226 = slice_scatter_168 = None
        permute_227 = torch.ops.aten.permute.default(slice_scatter_170, [0, 2, 1, 3]);  slice_scatter_170 = None
        view_259 = torch.ops.aten.view.default(permute_227, [4, 4, 256, 513]);  permute_227 = None
        expand_15 = torch.ops.aten.expand.default(rev_15, [4, 256, 1, 257]);  rev_15 = None
        view_261 = torch.ops.aten.view.default(view_259, [4, 1, 1024, 513])
        permute_229 = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        slice_898 = torch.ops.aten.slice.Tensor(permute_229, 1, -256, 9223372036854775807);  permute_229 = None
        slice_900 = torch.ops.aten.slice.Tensor(slice_898, 3, -257, 9223372036854775807);  slice_898 = None
        full_default_38 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(expand_15, torch.bool);  expand_15 = None
        where_30 = torch.ops.aten.where.self(convert_element_type_19, full_default_38, slice_900);  convert_element_type_19 = full_default_38 = slice_900 = None
        view_262 = torch.ops.aten.view.default(view_259, [4, 1, 1024, 513])
        permute_230 = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        slice_906 = torch.ops.aten.slice.Tensor(permute_230, 1, -256, 9223372036854775807);  permute_230 = None
        slice_908 = torch.ops.aten.slice.Tensor(slice_906, 3, -257, 9223372036854775807);  slice_906 = None
        copy_47 = torch.ops.aten.copy.default(slice_908, where_30);  slice_908 = where_30 = None
        view_263 = torch.ops.aten.view.default(view_259, [4, 1, 1024, 513]);  view_259 = None
        permute_231 = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
        slice_910 = torch.ops.aten.slice.Tensor(permute_231, 1, -256, 9223372036854775807)
        slice_scatter_172 = torch.ops.aten.slice_scatter.default(slice_910, copy_47, 3, -257, 9223372036854775807);  slice_910 = copy_47 = None
        slice_scatter_174 = torch.ops.aten.slice_scatter.default(permute_231, slice_scatter_172, 1, -256, 9223372036854775807);  permute_231 = slice_scatter_172 = None
        permute_232 = torch.ops.aten.permute.default(slice_scatter_174, [0, 2, 1, 3]);  slice_scatter_174 = None
        view_264 = torch.ops.aten.view.default(permute_232, [4, 4, 256, 513]);  permute_232 = None
        view_266 = torch.ops.aten.view.default(view_246, [4, 12, 1024, 513]);  view_246 = None
        permute_234 = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        view_267 = torch.ops.aten.view.default(view_264, [4, 1, 1024, 513]);  view_264 = None
        permute_235 = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        add_50 = torch.ops.aten.add.Tensor(permute_234, permute_235);  permute_234 = permute_235 = None
        permute_236 = torch.ops.aten.permute.default(add_50, [0, 2, 1, 3]);  add_50 = None
        view_269 = torch.ops.aten.view.default(permute_236, [48, 4, 256, 513]);  permute_236 = None
        view_270 = torch.ops.aten.view.default(view_269, [4, 12, 1024, 513]);  view_269 = None
        permute_237 = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_44 = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        amax_3 = torch.ops.aten.amax.default(clone_44, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(clone_44, amax_3);  clone_44 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_37 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        full_default_39 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31 = torch.ops.aten.where.self(unsqueeze_73, full_default_39, div_37);  unsqueeze_73 = full_default_39 = div_37 = None
        view_271 = torch.ops.aten.view.default(add_47, [1024, 4, 12, 64]);  add_47 = None
        permute_238 = torch.ops.aten.permute.default(view_271, [1, 0, 2, 3]);  view_271 = None
        permute_239 = torch.ops.aten.permute.default(where_31, [0, 2, 1, 3]);  where_31 = None
        clone_46 = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_272 = torch.ops.aten.view.default(clone_46, [48, 4, 256, 513]);  clone_46 = None
        permute_240 = torch.ops.aten.permute.default(permute_238, [0, 2, 1, 3]);  permute_238 = None
        view_273 = torch.ops.aten.view.default(permute_240, [48, 1024, 64]);  permute_240 = None
        constant_pad_nd_14 = torch.ops.aten.constant_pad_nd.default(view_273, [0, 0, 256, 256], -1.0);  view_273 = None
        as_strided_23 = torch.ops.aten.as_strided.default(constant_pad_nd_14, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_14 = None
        constant_pad_nd_15 = torch.ops.aten.constant_pad_nd.default(view_272, [0, 257], 0.0);  view_272 = None
        view_274 = torch.ops.aten.view.default(constant_pad_nd_15, [48, 4, -1]);  constant_pad_nd_15 = None
        slice_920 = torch.ops.aten.slice.Tensor(view_274, 2, 0, -256);  view_274 = None
        view_275 = torch.ops.aten.view.default(slice_920, [48, 4, 256, 769]);  slice_920 = None
        slice_924 = torch.ops.aten.slice.Tensor(view_275, 3, 0, -1);  view_275 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(slice_924, 4);  slice_924 = None
        permute_241 = torch.ops.aten.permute.default(unsqueeze_74, [0, 1, 2, 4, 3]);  unsqueeze_74 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        permute_242 = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
        permute_243 = torch.ops.aten.permute.default(permute_241, [0, 1, 2, 4, 3]);  permute_241 = None
        view_276 = torch.ops.aten.view.default(permute_243, [192, 256, 768]);  permute_243 = None
        permute_244 = torch.ops.aten.permute.default(permute_242, [0, 1, 4, 3, 2]);  permute_242 = None
        clone_47 = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_277 = torch.ops.aten.view.default(clone_47, [192, 768, 64]);  clone_47 = None
        bmm_7 = torch.ops.aten.bmm.default(view_276, view_277);  view_276 = view_277 = None
        view_278 = torch.ops.aten.view.default(bmm_7, [48, 4, 256, 1, 64]);  bmm_7 = None
        permute_245 = torch.ops.aten.permute.default(view_278, [0, 1, 2, 4, 3]);  view_278 = None
        view_279 = torch.ops.aten.view.default(permute_245, [48, 4, 256, 64]);  permute_245 = None
        view_280 = torch.ops.aten.view.default(view_279, [4, 12, 1024, 64]);  view_279 = None
        permute_246 = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        permute_247 = torch.ops.aten.permute.default(permute_246, [1, 0, 2, 3]);  permute_246 = None
        clone_48 = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_281 = torch.ops.aten.view.default(clone_48, [1024, 4, 768]);  clone_48 = None
        permute_248 = torch.ops.aten.permute.default(view_281, [1, 0, 2]);  view_281 = None
        permute_249 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        clone_49 = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_282 = torch.ops.aten.view.default(clone_49, [4096, 768]);  clone_49 = None
        mm_15 = torch.ops.aten.mm.default(view_282, permute_249);  view_282 = permute_249 = None
        view_283 = torch.ops.aten.view.default(mm_15, [4, 1024, 768]);  mm_15 = None
        add_52 = torch.ops.aten.add.Tensor(view_283, arg58_1);  view_283 = arg58_1 = None
        add_53 = torch.ops.aten.add.Tensor(add_52, add_44);  add_52 = add_44 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_53, getitem_13);  add_53 = getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = rsqrt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg59_1);  mul_25 = arg59_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_26, arg60_1);  mul_26 = arg60_1 = None
        view_284 = torch.ops.aten.view.default(add_55, [4096, 768])
        permute_250 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg62_1, view_284, permute_250);  arg62_1 = view_284 = permute_250 = None
        view_285 = torch.ops.aten.view.default(addmm_6, [4, 1024, 3072]);  addmm_6 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_285, 0.5)
        mul_28 = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
        erf_3 = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_56 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_27, add_56);  mul_27 = add_56 = None
        view_286 = torch.ops.aten.view.default(mul_29, [4096, 3072]);  mul_29 = None
        permute_251 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg64_1, view_286, permute_251);  arg64_1 = view_286 = permute_251 = None
        view_287 = torch.ops.aten.view.default(addmm_7, [4, 1024, 768]);  addmm_7 = None
        add_57 = torch.ops.aten.add.Tensor(view_287, add_55);  view_287 = add_55 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_57, getitem_15);  add_57 = getitem_15 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = rsqrt_7 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg65_1);  mul_30 = arg65_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_31, arg66_1);  mul_31 = arg66_1 = None
        permute_252 = torch.ops.aten.permute.default(add_59, [1, 0, 2])
        permute_253 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        clone_52 = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format)
        view_288 = torch.ops.aten.view.default(clone_52, [4096, 768]);  clone_52 = None
        mm_16 = torch.ops.aten.mm.default(view_288, permute_253);  view_288 = permute_253 = None
        view_289 = torch.ops.aten.view.default(mm_16, [1024, 4, 768]);  mm_16 = None
        add_60 = torch.ops.aten.add.Tensor(view_289, arg68_1);  view_289 = arg68_1 = None
        permute_254 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        clone_53 = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format)
        view_290 = torch.ops.aten.view.default(clone_53, [4096, 768]);  clone_53 = None
        mm_17 = torch.ops.aten.mm.default(view_290, permute_254);  view_290 = permute_254 = None
        view_291 = torch.ops.aten.view.default(mm_17, [1024, 4, 768]);  mm_17 = None
        add_61 = torch.ops.aten.add.Tensor(view_291, arg70_1);  view_291 = arg70_1 = None
        permute_255 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        clone_54 = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_292 = torch.ops.aten.view.default(clone_54, [4096, 768]);  clone_54 = None
        mm_18 = torch.ops.aten.mm.default(view_292, permute_255);  view_292 = permute_255 = None
        view_293 = torch.ops.aten.view.default(mm_18, [1024, 4, 768]);  mm_18 = None
        add_62 = torch.ops.aten.add.Tensor(view_293, arg72_1);  view_293 = arg72_1 = None
        div_40 = torch.ops.aten.div.Tensor(add_60, 8.0);  add_60 = None
        view_295 = torch.ops.aten.view.default(add_61, [1024, 4, 12, 64]);  add_61 = None
        permute_257 = torch.ops.aten.permute.default(view_295, [1, 0, 2, 3]);  view_295 = None
        permute_259 = torch.ops.aten.permute.default(permute_257, [0, 2, 1, 3]);  permute_257 = None
        view_297 = torch.ops.aten.view.default(permute_259, [48, 1024, 64]);  permute_259 = None
        view_299 = torch.ops.aten.view.default(view_297, [48, 2, 512, 64]);  view_297 = None
        as_strided_25 = torch.ops.aten.as_strided.default(view_299, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_299 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
        permute_261 = torch.ops.aten.permute.default(unsqueeze_77, [0, 1, 4, 2, 3]);  unsqueeze_77 = None
        view_300 = torch.ops.aten.view.default(div_40, [1024, 4, 12, 64]);  div_40 = None
        permute_263 = torch.ops.aten.permute.default(view_300, [1, 0, 2, 3]);  view_300 = None
        permute_264 = torch.ops.aten.permute.default(permute_263, [0, 2, 1, 3]);  permute_263 = None
        view_301 = torch.ops.aten.view.default(permute_264, [48, 1024, 64]);  permute_264 = None
        view_302 = torch.ops.aten.view.default(view_301, [48, 2, 512, 64]);  view_301 = None
        as_strided_26 = torch.ops.aten.as_strided.default(view_302, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_302 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        permute_265 = torch.ops.aten.permute.default(unsqueeze_78, [0, 1, 2, 4, 3]);  unsqueeze_78 = None
        permute_266 = torch.ops.aten.permute.default(permute_265, [0, 1, 2, 4, 3]);  permute_265 = None
        clone_55 = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_303 = torch.ops.aten.view.default(clone_55, [144, 512, 64]);  clone_55 = None
        permute_267 = torch.ops.aten.permute.default(permute_261, [0, 1, 4, 3, 2]);  permute_261 = None
        clone_56 = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_304 = torch.ops.aten.view.default(clone_56, [144, 64, 512]);  clone_56 = None
        bmm_8 = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
        view_305 = torch.ops.aten.view.default(bmm_8, [48, 3, 512, 1, 512]);  bmm_8 = None
        permute_268 = torch.ops.aten.permute.default(view_305, [0, 1, 2, 4, 3]);  view_305 = None
        view_306 = torch.ops.aten.view.default(permute_268, [48, 3, 512, 512]);  permute_268 = None
        constant_pad_nd_16 = torch.ops.aten.constant_pad_nd.default(view_306, [0, 0, 0, 1], 0.0);  view_306 = None
        view_307 = torch.ops.aten.view.default(constant_pad_nd_16, [48, 3, 512, 513]);  constant_pad_nd_16 = None
        full_36 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_927 = torch.ops.aten.slice.Tensor(view_307, 2, 0, 256)
        slice_928 = torch.ops.aten.slice.Tensor(slice_927, 3, 0, 257);  slice_927 = None
        slice_930 = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_932 = torch.ops.aten.slice.Tensor(slice_930, 3, 256, 9223372036854775807);  slice_930 = None
        copy_48 = torch.ops.aten.copy.default(slice_932, slice_928);  slice_932 = slice_928 = None
        slice_934 = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_scatter_176 = torch.ops.aten.slice_scatter.default(slice_934, copy_48, 3, 256, 9223372036854775807);  slice_934 = copy_48 = None
        slice_scatter_178 = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_176, 1, 0, -1);  full_36 = slice_scatter_176 = None
        select_80 = torch.ops.aten.select.int(view_307, 1, -1)
        slice_941 = torch.ops.aten.slice.Tensor(select_80, 1, 256, 9223372036854775807);  select_80 = None
        slice_942 = torch.ops.aten.slice.Tensor(slice_941, 2, 0, 257);  slice_941 = None
        select_82 = torch.ops.aten.select.int(slice_scatter_178, 1, -1)
        slice_948 = torch.ops.aten.slice.Tensor(select_82, 2, 256, 9223372036854775807);  select_82 = None
        copy_49 = torch.ops.aten.copy.default(slice_948, slice_942);  slice_948 = slice_942 = None
        select_83 = torch.ops.aten.select.int(slice_scatter_178, 1, -1)
        slice_scatter_180 = torch.ops.aten.slice_scatter.default(select_83, copy_49, 2, 256, 9223372036854775807);  select_83 = copy_49 = None
        select_scatter_16 = torch.ops.aten.select_scatter.default(slice_scatter_178, slice_scatter_180, 1, -1);  slice_scatter_178 = slice_scatter_180 = None
        slice_956 = torch.ops.aten.slice.Tensor(view_307, 2, -257, -1)
        slice_957 = torch.ops.aten.slice.Tensor(slice_956, 3, 257, 9223372036854775807);  slice_956 = None
        slice_963 = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_965 = torch.ops.aten.slice.Tensor(slice_963, 3, 0, 256);  slice_963 = None
        copy_50 = torch.ops.aten.copy.default(slice_965, slice_957);  slice_965 = slice_957 = None
        slice_967 = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_scatter_183 = torch.ops.aten.slice_scatter.default(slice_967, copy_50, 3, 0, 256);  slice_967 = copy_50 = None
        slice_scatter_185 = torch.ops.aten.slice_scatter.default(select_scatter_16, slice_scatter_183, 1, 1, 9223372036854775807);  select_scatter_16 = slice_scatter_183 = None
        select_85 = torch.ops.aten.select.int(view_307, 1, 0);  view_307 = None
        slice_974 = torch.ops.aten.slice.Tensor(select_85, 1, 0, 255);  select_85 = None
        slice_975 = torch.ops.aten.slice.Tensor(slice_974, 2, -255, 9223372036854775807);  slice_974 = None
        select_87 = torch.ops.aten.select.int(slice_scatter_185, 1, 0)
        slice_980 = torch.ops.aten.slice.Tensor(select_87, 1, 1, 256);  select_87 = None
        slice_981 = torch.ops.aten.slice.Tensor(slice_980, 2, 1, 256);  slice_980 = None
        copy_51 = torch.ops.aten.copy.default(slice_981, slice_975);  slice_981 = slice_975 = None
        select_88 = torch.ops.aten.select.int(slice_scatter_185, 1, 0)
        slice_983 = torch.ops.aten.slice.Tensor(select_88, 1, 1, 256)
        slice_scatter_187 = torch.ops.aten.slice_scatter.default(slice_983, copy_51, 2, 1, 256);  slice_983 = copy_51 = None
        slice_scatter_188 = torch.ops.aten.slice_scatter.default(select_88, slice_scatter_187, 1, 1, 256);  select_88 = slice_scatter_187 = None
        select_scatter_17 = torch.ops.aten.select_scatter.default(slice_scatter_185, slice_scatter_188, 1, 0);  slice_scatter_185 = slice_scatter_188 = None
        full_default_40 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_16 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(iota_16, -2);  iota_16 = None
        iota_17 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        sub_33 = torch.ops.aten.sub.Tensor(unsqueeze_79, unsqueeze_80);  unsqueeze_79 = unsqueeze_80 = None
        le_8 = torch.ops.aten.le.Scalar(sub_33, 0);  sub_33 = None
        full_default_41 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32 = torch.ops.aten.where.self(le_8, full_default_40, full_default_41);  le_8 = full_default_40 = full_default_41 = None
        rev_16 = torch.ops.prims.rev.default(where_32, [0]);  where_32 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(rev_16, 0);  rev_16 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        rev_17 = torch.ops.prims.rev.default(unsqueeze_82, [1, 3])
        expand_16 = torch.ops.aten.expand.default(unsqueeze_82, [4, 256, 12, 257]);  unsqueeze_82 = None
        view_310 = torch.ops.aten.view.default(select_scatter_17, [4, 12, 1024, 513])
        permute_271 = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        slice_994 = torch.ops.aten.slice.Tensor(permute_271, 1, 0, 256);  permute_271 = None
        slice_996 = torch.ops.aten.slice.Tensor(slice_994, 3, 0, 257);  slice_994 = None
        full_default_42 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(expand_16, torch.bool);  expand_16 = None
        where_33 = torch.ops.aten.where.self(convert_element_type_20, full_default_42, slice_996);  convert_element_type_20 = full_default_42 = slice_996 = None
        view_311 = torch.ops.aten.view.default(select_scatter_17, [4, 12, 1024, 513])
        permute_272 = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
        slice_1002 = torch.ops.aten.slice.Tensor(permute_272, 1, 0, 256);  permute_272 = None
        slice_1004 = torch.ops.aten.slice.Tensor(slice_1002, 3, 0, 257);  slice_1002 = None
        copy_52 = torch.ops.aten.copy.default(slice_1004, where_33);  slice_1004 = where_33 = None
        view_312 = torch.ops.aten.view.default(select_scatter_17, [4, 12, 1024, 513]);  select_scatter_17 = None
        permute_273 = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        slice_1006 = torch.ops.aten.slice.Tensor(permute_273, 1, 0, 256)
        slice_scatter_190 = torch.ops.aten.slice_scatter.default(slice_1006, copy_52, 3, 0, 257);  slice_1006 = copy_52 = None
        slice_scatter_192 = torch.ops.aten.slice_scatter.default(permute_273, slice_scatter_190, 1, 0, 256);  permute_273 = slice_scatter_190 = None
        permute_274 = torch.ops.aten.permute.default(slice_scatter_192, [0, 2, 1, 3]);  slice_scatter_192 = None
        view_313 = torch.ops.aten.view.default(permute_274, [48, 4, 256, 513]);  permute_274 = None
        expand_17 = torch.ops.aten.expand.default(rev_17, [4, 256, 12, 257]);  rev_17 = None
        view_315 = torch.ops.aten.view.default(view_313, [4, 12, 1024, 513])
        permute_276 = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        slice_1017 = torch.ops.aten.slice.Tensor(permute_276, 1, -256, 9223372036854775807);  permute_276 = None
        slice_1019 = torch.ops.aten.slice.Tensor(slice_1017, 3, -257, 9223372036854775807);  slice_1017 = None
        full_default_43 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(expand_17, torch.bool);  expand_17 = None
        where_34 = torch.ops.aten.where.self(convert_element_type_21, full_default_43, slice_1019);  convert_element_type_21 = full_default_43 = slice_1019 = None
        view_316 = torch.ops.aten.view.default(view_313, [4, 12, 1024, 513])
        permute_277 = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        slice_1025 = torch.ops.aten.slice.Tensor(permute_277, 1, -256, 9223372036854775807);  permute_277 = None
        slice_1027 = torch.ops.aten.slice.Tensor(slice_1025, 3, -257, 9223372036854775807);  slice_1025 = None
        copy_53 = torch.ops.aten.copy.default(slice_1027, where_34);  slice_1027 = where_34 = None
        view_317 = torch.ops.aten.view.default(view_313, [4, 12, 1024, 513]);  view_313 = None
        permute_278 = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
        slice_1029 = torch.ops.aten.slice.Tensor(permute_278, 1, -256, 9223372036854775807)
        slice_scatter_194 = torch.ops.aten.slice_scatter.default(slice_1029, copy_53, 3, -257, 9223372036854775807);  slice_1029 = copy_53 = None
        slice_scatter_196 = torch.ops.aten.slice_scatter.default(permute_278, slice_scatter_194, 1, -256, 9223372036854775807);  permute_278 = slice_scatter_194 = None
        permute_279 = torch.ops.aten.permute.default(slice_scatter_196, [0, 2, 1, 3]);  slice_scatter_196 = None
        view_318 = torch.ops.aten.view.default(permute_279, [48, 4, 256, 513]);  permute_279 = None
        ne_4 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(ne_4, 2);  ne_4 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(unsqueeze_83, 3);  unsqueeze_83 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(unsqueeze_84, torch.float32)
        full_default_44 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35 = torch.ops.aten.where.self(unsqueeze_84, full_default_44, convert_element_type_22);  unsqueeze_84 = full_default_44 = convert_element_type_22 = None
        full_40 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_281 = torch.ops.aten.permute.default(full_40, [0, 2, 1, 3]);  full_40 = None
        view_320 = torch.ops.aten.view.default(permute_281, [4, 1024, 1]);  permute_281 = None
        permute_282 = torch.ops.aten.permute.default(where_35, [0, 2, 1, 3]);  where_35 = None
        view_321 = torch.ops.aten.view.default(permute_282, [4, 1024, 1]);  permute_282 = None
        view_322 = torch.ops.aten.view.default(view_320, [4, 2, 512, 1]);  view_320 = None
        as_strided_27 = torch.ops.aten.as_strided.default(view_322, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_322 = None
        view_323 = torch.ops.aten.view.default(view_321, [4, 2, 512, 1]);  view_321 = None
        as_strided_28 = torch.ops.aten.as_strided.default(view_323, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_323 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(as_strided_27, 4);  as_strided_27 = None
        permute_283 = torch.ops.aten.permute.default(unsqueeze_85, [0, 1, 2, 4, 3]);  unsqueeze_85 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        permute_284 = torch.ops.aten.permute.default(unsqueeze_86, [0, 1, 4, 2, 3]);  unsqueeze_86 = None
        mul_32 = torch.ops.aten.mul.Tensor(permute_283, permute_284);  permute_283 = permute_284 = None
        view_324 = torch.ops.aten.view.default(mul_32, [4, 3, 512, 512]);  mul_32 = None
        constant_pad_nd_17 = torch.ops.aten.constant_pad_nd.default(view_324, [0, 0, 0, 1], 0.0);  view_324 = None
        view_325 = torch.ops.aten.view.default(constant_pad_nd_17, [4, 3, 512, 513]);  constant_pad_nd_17 = None
        full_41 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1039 = torch.ops.aten.slice.Tensor(view_325, 2, 0, 256)
        slice_1040 = torch.ops.aten.slice.Tensor(slice_1039, 3, 0, 257);  slice_1039 = None
        slice_1042 = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_1044 = torch.ops.aten.slice.Tensor(slice_1042, 3, 256, 9223372036854775807);  slice_1042 = None
        copy_54 = torch.ops.aten.copy.default(slice_1044, slice_1040);  slice_1044 = slice_1040 = None
        slice_1046 = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_scatter_198 = torch.ops.aten.slice_scatter.default(slice_1046, copy_54, 3, 256, 9223372036854775807);  slice_1046 = copy_54 = None
        slice_scatter_200 = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_198, 1, 0, -1);  full_41 = slice_scatter_198 = None
        select_90 = torch.ops.aten.select.int(view_325, 1, -1)
        slice_1053 = torch.ops.aten.slice.Tensor(select_90, 1, 256, 9223372036854775807);  select_90 = None
        slice_1054 = torch.ops.aten.slice.Tensor(slice_1053, 2, 0, 257);  slice_1053 = None
        select_92 = torch.ops.aten.select.int(slice_scatter_200, 1, -1)
        slice_1060 = torch.ops.aten.slice.Tensor(select_92, 2, 256, 9223372036854775807);  select_92 = None
        copy_55 = torch.ops.aten.copy.default(slice_1060, slice_1054);  slice_1060 = slice_1054 = None
        select_93 = torch.ops.aten.select.int(slice_scatter_200, 1, -1)
        slice_scatter_202 = torch.ops.aten.slice_scatter.default(select_93, copy_55, 2, 256, 9223372036854775807);  select_93 = copy_55 = None
        select_scatter_18 = torch.ops.aten.select_scatter.default(slice_scatter_200, slice_scatter_202, 1, -1);  slice_scatter_200 = slice_scatter_202 = None
        slice_1068 = torch.ops.aten.slice.Tensor(view_325, 2, -257, -1)
        slice_1069 = torch.ops.aten.slice.Tensor(slice_1068, 3, 257, 9223372036854775807);  slice_1068 = None
        slice_1075 = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_1077 = torch.ops.aten.slice.Tensor(slice_1075, 3, 0, 256);  slice_1075 = None
        copy_56 = torch.ops.aten.copy.default(slice_1077, slice_1069);  slice_1077 = slice_1069 = None
        slice_1079 = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_scatter_205 = torch.ops.aten.slice_scatter.default(slice_1079, copy_56, 3, 0, 256);  slice_1079 = copy_56 = None
        slice_scatter_207 = torch.ops.aten.slice_scatter.default(select_scatter_18, slice_scatter_205, 1, 1, 9223372036854775807);  select_scatter_18 = slice_scatter_205 = None
        select_95 = torch.ops.aten.select.int(view_325, 1, 0);  view_325 = None
        slice_1086 = torch.ops.aten.slice.Tensor(select_95, 1, 0, 255);  select_95 = None
        slice_1087 = torch.ops.aten.slice.Tensor(slice_1086, 2, -255, 9223372036854775807);  slice_1086 = None
        select_97 = torch.ops.aten.select.int(slice_scatter_207, 1, 0)
        slice_1092 = torch.ops.aten.slice.Tensor(select_97, 1, 1, 256);  select_97 = None
        slice_1093 = torch.ops.aten.slice.Tensor(slice_1092, 2, 1, 256);  slice_1092 = None
        copy_57 = torch.ops.aten.copy.default(slice_1093, slice_1087);  slice_1093 = slice_1087 = None
        select_98 = torch.ops.aten.select.int(slice_scatter_207, 1, 0)
        slice_1095 = torch.ops.aten.slice.Tensor(select_98, 1, 1, 256)
        slice_scatter_209 = torch.ops.aten.slice_scatter.default(slice_1095, copy_57, 2, 1, 256);  slice_1095 = copy_57 = None
        slice_scatter_210 = torch.ops.aten.slice_scatter.default(select_98, slice_scatter_209, 1, 1, 256);  select_98 = slice_scatter_209 = None
        select_scatter_19 = torch.ops.aten.select_scatter.default(slice_scatter_207, slice_scatter_210, 1, 0);  slice_scatter_207 = slice_scatter_210 = None
        full_default_45 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_18 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(iota_18, -2);  iota_18 = None
        iota_19 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
        sub_35 = torch.ops.aten.sub.Tensor(unsqueeze_87, unsqueeze_88);  unsqueeze_87 = unsqueeze_88 = None
        le_9 = torch.ops.aten.le.Scalar(sub_35, 0);  sub_35 = None
        full_default_46 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36 = torch.ops.aten.where.self(le_9, full_default_45, full_default_46);  le_9 = full_default_45 = full_default_46 = None
        rev_18 = torch.ops.prims.rev.default(where_36, [0]);  where_36 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(rev_18, 0);  rev_18 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        rev_19 = torch.ops.prims.rev.default(unsqueeze_90, [1, 3])
        expand_18 = torch.ops.aten.expand.default(unsqueeze_90, [4, 256, 1, 257]);  unsqueeze_90 = None
        view_328 = torch.ops.aten.view.default(select_scatter_19, [4, 1, 1024, 513])
        permute_287 = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        slice_1106 = torch.ops.aten.slice.Tensor(permute_287, 1, 0, 256);  permute_287 = None
        slice_1108 = torch.ops.aten.slice.Tensor(slice_1106, 3, 0, 257);  slice_1106 = None
        full_default_47 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(expand_18, torch.bool);  expand_18 = None
        where_37 = torch.ops.aten.where.self(convert_element_type_23, full_default_47, slice_1108);  convert_element_type_23 = full_default_47 = slice_1108 = None
        view_329 = torch.ops.aten.view.default(select_scatter_19, [4, 1, 1024, 513])
        permute_288 = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        slice_1114 = torch.ops.aten.slice.Tensor(permute_288, 1, 0, 256);  permute_288 = None
        slice_1116 = torch.ops.aten.slice.Tensor(slice_1114, 3, 0, 257);  slice_1114 = None
        copy_58 = torch.ops.aten.copy.default(slice_1116, where_37);  slice_1116 = where_37 = None
        view_330 = torch.ops.aten.view.default(select_scatter_19, [4, 1, 1024, 513]);  select_scatter_19 = None
        permute_289 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        slice_1118 = torch.ops.aten.slice.Tensor(permute_289, 1, 0, 256)
        slice_scatter_212 = torch.ops.aten.slice_scatter.default(slice_1118, copy_58, 3, 0, 257);  slice_1118 = copy_58 = None
        slice_scatter_214 = torch.ops.aten.slice_scatter.default(permute_289, slice_scatter_212, 1, 0, 256);  permute_289 = slice_scatter_212 = None
        permute_290 = torch.ops.aten.permute.default(slice_scatter_214, [0, 2, 1, 3]);  slice_scatter_214 = None
        view_331 = torch.ops.aten.view.default(permute_290, [4, 4, 256, 513]);  permute_290 = None
        expand_19 = torch.ops.aten.expand.default(rev_19, [4, 256, 1, 257]);  rev_19 = None
        view_333 = torch.ops.aten.view.default(view_331, [4, 1, 1024, 513])
        permute_292 = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        slice_1129 = torch.ops.aten.slice.Tensor(permute_292, 1, -256, 9223372036854775807);  permute_292 = None
        slice_1131 = torch.ops.aten.slice.Tensor(slice_1129, 3, -257, 9223372036854775807);  slice_1129 = None
        full_default_48 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(expand_19, torch.bool);  expand_19 = None
        where_38 = torch.ops.aten.where.self(convert_element_type_24, full_default_48, slice_1131);  convert_element_type_24 = full_default_48 = slice_1131 = None
        view_334 = torch.ops.aten.view.default(view_331, [4, 1, 1024, 513])
        permute_293 = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        slice_1137 = torch.ops.aten.slice.Tensor(permute_293, 1, -256, 9223372036854775807);  permute_293 = None
        slice_1139 = torch.ops.aten.slice.Tensor(slice_1137, 3, -257, 9223372036854775807);  slice_1137 = None
        copy_59 = torch.ops.aten.copy.default(slice_1139, where_38);  slice_1139 = where_38 = None
        view_335 = torch.ops.aten.view.default(view_331, [4, 1, 1024, 513]);  view_331 = None
        permute_294 = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        slice_1141 = torch.ops.aten.slice.Tensor(permute_294, 1, -256, 9223372036854775807)
        slice_scatter_216 = torch.ops.aten.slice_scatter.default(slice_1141, copy_59, 3, -257, 9223372036854775807);  slice_1141 = copy_59 = None
        slice_scatter_218 = torch.ops.aten.slice_scatter.default(permute_294, slice_scatter_216, 1, -256, 9223372036854775807);  permute_294 = slice_scatter_216 = None
        permute_295 = torch.ops.aten.permute.default(slice_scatter_218, [0, 2, 1, 3]);  slice_scatter_218 = None
        view_336 = torch.ops.aten.view.default(permute_295, [4, 4, 256, 513]);  permute_295 = None
        view_338 = torch.ops.aten.view.default(view_318, [4, 12, 1024, 513]);  view_318 = None
        permute_297 = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        view_339 = torch.ops.aten.view.default(view_336, [4, 1, 1024, 513]);  view_336 = None
        permute_298 = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        add_65 = torch.ops.aten.add.Tensor(permute_297, permute_298);  permute_297 = permute_298 = None
        permute_299 = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
        view_341 = torch.ops.aten.view.default(permute_299, [48, 4, 256, 513]);  permute_299 = None
        view_342 = torch.ops.aten.view.default(view_341, [4, 12, 1024, 513]);  view_341 = None
        permute_300 = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        clone_57 = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
        amax_4 = torch.ops.aten.amax.default(clone_57, [-1], True)
        sub_36 = torch.ops.aten.sub.Tensor(clone_57, amax_4);  clone_57 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_47 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        full_default_49 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39 = torch.ops.aten.where.self(unsqueeze_92, full_default_49, div_47);  unsqueeze_92 = full_default_49 = div_47 = None
        view_343 = torch.ops.aten.view.default(add_62, [1024, 4, 12, 64]);  add_62 = None
        permute_301 = torch.ops.aten.permute.default(view_343, [1, 0, 2, 3]);  view_343 = None
        permute_302 = torch.ops.aten.permute.default(where_39, [0, 2, 1, 3]);  where_39 = None
        clone_59 = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_344 = torch.ops.aten.view.default(clone_59, [48, 4, 256, 513]);  clone_59 = None
        permute_303 = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
        view_345 = torch.ops.aten.view.default(permute_303, [48, 1024, 64]);  permute_303 = None
        constant_pad_nd_18 = torch.ops.aten.constant_pad_nd.default(view_345, [0, 0, 256, 256], -1.0);  view_345 = None
        as_strided_29 = torch.ops.aten.as_strided.default(constant_pad_nd_18, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_18 = None
        constant_pad_nd_19 = torch.ops.aten.constant_pad_nd.default(view_344, [0, 257], 0.0);  view_344 = None
        view_346 = torch.ops.aten.view.default(constant_pad_nd_19, [48, 4, -1]);  constant_pad_nd_19 = None
        slice_1151 = torch.ops.aten.slice.Tensor(view_346, 2, 0, -256);  view_346 = None
        view_347 = torch.ops.aten.view.default(slice_1151, [48, 4, 256, 769]);  slice_1151 = None
        slice_1155 = torch.ops.aten.slice.Tensor(view_347, 3, 0, -1);  view_347 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(slice_1155, 4);  slice_1155 = None
        permute_304 = torch.ops.aten.permute.default(unsqueeze_93, [0, 1, 2, 4, 3]);  unsqueeze_93 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(as_strided_29, 4);  as_strided_29 = None
        permute_305 = torch.ops.aten.permute.default(unsqueeze_94, [0, 1, 4, 3, 2]);  unsqueeze_94 = None
        permute_306 = torch.ops.aten.permute.default(permute_304, [0, 1, 2, 4, 3]);  permute_304 = None
        view_348 = torch.ops.aten.view.default(permute_306, [192, 256, 768]);  permute_306 = None
        permute_307 = torch.ops.aten.permute.default(permute_305, [0, 1, 4, 3, 2]);  permute_305 = None
        clone_60 = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
        view_349 = torch.ops.aten.view.default(clone_60, [192, 768, 64]);  clone_60 = None
        bmm_9 = torch.ops.aten.bmm.default(view_348, view_349);  view_348 = view_349 = None
        view_350 = torch.ops.aten.view.default(bmm_9, [48, 4, 256, 1, 64]);  bmm_9 = None
        permute_308 = torch.ops.aten.permute.default(view_350, [0, 1, 2, 4, 3]);  view_350 = None
        view_351 = torch.ops.aten.view.default(permute_308, [48, 4, 256, 64]);  permute_308 = None
        view_352 = torch.ops.aten.view.default(view_351, [4, 12, 1024, 64]);  view_351 = None
        permute_309 = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        permute_310 = torch.ops.aten.permute.default(permute_309, [1, 0, 2, 3]);  permute_309 = None
        clone_61 = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_353 = torch.ops.aten.view.default(clone_61, [1024, 4, 768]);  clone_61 = None
        permute_311 = torch.ops.aten.permute.default(view_353, [1, 0, 2]);  view_353 = None
        permute_312 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        clone_62 = torch.ops.aten.clone.default(permute_311, memory_format = torch.contiguous_format);  permute_311 = None
        view_354 = torch.ops.aten.view.default(clone_62, [4096, 768]);  clone_62 = None
        mm_19 = torch.ops.aten.mm.default(view_354, permute_312);  view_354 = permute_312 = None
        view_355 = torch.ops.aten.view.default(mm_19, [4, 1024, 768]);  mm_19 = None
        add_67 = torch.ops.aten.add.Tensor(view_355, arg74_1);  view_355 = arg74_1 = None
        add_68 = torch.ops.aten.add.Tensor(add_67, add_59);  add_67 = add_59 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_68, getitem_17);  add_68 = getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = rsqrt_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg75_1);  mul_33 = arg75_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_34, arg76_1);  mul_34 = arg76_1 = None
        view_356 = torch.ops.aten.view.default(add_70, [4096, 768])
        permute_313 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg78_1, view_356, permute_313);  arg78_1 = view_356 = permute_313 = None
        view_357 = torch.ops.aten.view.default(addmm_8, [4, 1024, 3072]);  addmm_8 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_36 = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_4 = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_71 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_35, add_71);  mul_35 = add_71 = None
        view_358 = torch.ops.aten.view.default(mul_37, [4096, 3072]);  mul_37 = None
        permute_314 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg80_1, view_358, permute_314);  arg80_1 = view_358 = permute_314 = None
        view_359 = torch.ops.aten.view.default(addmm_9, [4, 1024, 768]);  addmm_9 = None
        add_72 = torch.ops.aten.add.Tensor(view_359, add_70);  view_359 = add_70 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_73 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_72, getitem_19);  add_72 = getitem_19 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = rsqrt_9 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg81_1);  mul_38 = arg81_1 = None
        add_74 = torch.ops.aten.add.Tensor(mul_39, arg82_1);  mul_39 = arg82_1 = None
        permute_315 = torch.ops.aten.permute.default(add_74, [1, 0, 2])
        permute_316 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        clone_65 = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format)
        view_360 = torch.ops.aten.view.default(clone_65, [4096, 768]);  clone_65 = None
        mm_20 = torch.ops.aten.mm.default(view_360, permute_316);  view_360 = permute_316 = None
        view_361 = torch.ops.aten.view.default(mm_20, [1024, 4, 768]);  mm_20 = None
        add_75 = torch.ops.aten.add.Tensor(view_361, arg84_1);  view_361 = arg84_1 = None
        permute_317 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        clone_66 = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format)
        view_362 = torch.ops.aten.view.default(clone_66, [4096, 768]);  clone_66 = None
        mm_21 = torch.ops.aten.mm.default(view_362, permute_317);  view_362 = permute_317 = None
        view_363 = torch.ops.aten.view.default(mm_21, [1024, 4, 768]);  mm_21 = None
        add_76 = torch.ops.aten.add.Tensor(view_363, arg86_1);  view_363 = arg86_1 = None
        permute_318 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        clone_67 = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
        view_364 = torch.ops.aten.view.default(clone_67, [4096, 768]);  clone_67 = None
        mm_22 = torch.ops.aten.mm.default(view_364, permute_318);  view_364 = permute_318 = None
        view_365 = torch.ops.aten.view.default(mm_22, [1024, 4, 768]);  mm_22 = None
        add_77 = torch.ops.aten.add.Tensor(view_365, arg88_1);  view_365 = arg88_1 = None
        div_50 = torch.ops.aten.div.Tensor(add_75, 8.0);  add_75 = None
        view_367 = torch.ops.aten.view.default(add_76, [1024, 4, 12, 64]);  add_76 = None
        permute_320 = torch.ops.aten.permute.default(view_367, [1, 0, 2, 3]);  view_367 = None
        permute_322 = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
        view_369 = torch.ops.aten.view.default(permute_322, [48, 1024, 64]);  permute_322 = None
        view_371 = torch.ops.aten.view.default(view_369, [48, 2, 512, 64]);  view_369 = None
        as_strided_31 = torch.ops.aten.as_strided.default(view_371, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_371 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(as_strided_31, 4);  as_strided_31 = None
        permute_324 = torch.ops.aten.permute.default(unsqueeze_96, [0, 1, 4, 2, 3]);  unsqueeze_96 = None
        view_372 = torch.ops.aten.view.default(div_50, [1024, 4, 12, 64]);  div_50 = None
        permute_326 = torch.ops.aten.permute.default(view_372, [1, 0, 2, 3]);  view_372 = None
        permute_327 = torch.ops.aten.permute.default(permute_326, [0, 2, 1, 3]);  permute_326 = None
        view_373 = torch.ops.aten.view.default(permute_327, [48, 1024, 64]);  permute_327 = None
        view_374 = torch.ops.aten.view.default(view_373, [48, 2, 512, 64]);  view_373 = None
        as_strided_32 = torch.ops.aten.as_strided.default(view_374, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_374 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        permute_328 = torch.ops.aten.permute.default(unsqueeze_97, [0, 1, 2, 4, 3]);  unsqueeze_97 = None
        permute_329 = torch.ops.aten.permute.default(permute_328, [0, 1, 2, 4, 3]);  permute_328 = None
        clone_68 = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_375 = torch.ops.aten.view.default(clone_68, [144, 512, 64]);  clone_68 = None
        permute_330 = torch.ops.aten.permute.default(permute_324, [0, 1, 4, 3, 2]);  permute_324 = None
        clone_69 = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_376 = torch.ops.aten.view.default(clone_69, [144, 64, 512]);  clone_69 = None
        bmm_10 = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
        view_377 = torch.ops.aten.view.default(bmm_10, [48, 3, 512, 1, 512]);  bmm_10 = None
        permute_331 = torch.ops.aten.permute.default(view_377, [0, 1, 2, 4, 3]);  view_377 = None
        view_378 = torch.ops.aten.view.default(permute_331, [48, 3, 512, 512]);  permute_331 = None
        constant_pad_nd_20 = torch.ops.aten.constant_pad_nd.default(view_378, [0, 0, 0, 1], 0.0);  view_378 = None
        view_379 = torch.ops.aten.view.default(constant_pad_nd_20, [48, 3, 512, 513]);  constant_pad_nd_20 = None
        full_45 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1158 = torch.ops.aten.slice.Tensor(view_379, 2, 0, 256)
        slice_1159 = torch.ops.aten.slice.Tensor(slice_1158, 3, 0, 257);  slice_1158 = None
        slice_1161 = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_1163 = torch.ops.aten.slice.Tensor(slice_1161, 3, 256, 9223372036854775807);  slice_1161 = None
        copy_60 = torch.ops.aten.copy.default(slice_1163, slice_1159);  slice_1163 = slice_1159 = None
        slice_1165 = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_scatter_220 = torch.ops.aten.slice_scatter.default(slice_1165, copy_60, 3, 256, 9223372036854775807);  slice_1165 = copy_60 = None
        slice_scatter_222 = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_220, 1, 0, -1);  full_45 = slice_scatter_220 = None
        select_100 = torch.ops.aten.select.int(view_379, 1, -1)
        slice_1172 = torch.ops.aten.slice.Tensor(select_100, 1, 256, 9223372036854775807);  select_100 = None
        slice_1173 = torch.ops.aten.slice.Tensor(slice_1172, 2, 0, 257);  slice_1172 = None
        select_102 = torch.ops.aten.select.int(slice_scatter_222, 1, -1)
        slice_1179 = torch.ops.aten.slice.Tensor(select_102, 2, 256, 9223372036854775807);  select_102 = None
        copy_61 = torch.ops.aten.copy.default(slice_1179, slice_1173);  slice_1179 = slice_1173 = None
        select_103 = torch.ops.aten.select.int(slice_scatter_222, 1, -1)
        slice_scatter_224 = torch.ops.aten.slice_scatter.default(select_103, copy_61, 2, 256, 9223372036854775807);  select_103 = copy_61 = None
        select_scatter_20 = torch.ops.aten.select_scatter.default(slice_scatter_222, slice_scatter_224, 1, -1);  slice_scatter_222 = slice_scatter_224 = None
        slice_1187 = torch.ops.aten.slice.Tensor(view_379, 2, -257, -1)
        slice_1188 = torch.ops.aten.slice.Tensor(slice_1187, 3, 257, 9223372036854775807);  slice_1187 = None
        slice_1194 = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_1196 = torch.ops.aten.slice.Tensor(slice_1194, 3, 0, 256);  slice_1194 = None
        copy_62 = torch.ops.aten.copy.default(slice_1196, slice_1188);  slice_1196 = slice_1188 = None
        slice_1198 = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_scatter_227 = torch.ops.aten.slice_scatter.default(slice_1198, copy_62, 3, 0, 256);  slice_1198 = copy_62 = None
        slice_scatter_229 = torch.ops.aten.slice_scatter.default(select_scatter_20, slice_scatter_227, 1, 1, 9223372036854775807);  select_scatter_20 = slice_scatter_227 = None
        select_105 = torch.ops.aten.select.int(view_379, 1, 0);  view_379 = None
        slice_1205 = torch.ops.aten.slice.Tensor(select_105, 1, 0, 255);  select_105 = None
        slice_1206 = torch.ops.aten.slice.Tensor(slice_1205, 2, -255, 9223372036854775807);  slice_1205 = None
        select_107 = torch.ops.aten.select.int(slice_scatter_229, 1, 0)
        slice_1211 = torch.ops.aten.slice.Tensor(select_107, 1, 1, 256);  select_107 = None
        slice_1212 = torch.ops.aten.slice.Tensor(slice_1211, 2, 1, 256);  slice_1211 = None
        copy_63 = torch.ops.aten.copy.default(slice_1212, slice_1206);  slice_1212 = slice_1206 = None
        select_108 = torch.ops.aten.select.int(slice_scatter_229, 1, 0)
        slice_1214 = torch.ops.aten.slice.Tensor(select_108, 1, 1, 256)
        slice_scatter_231 = torch.ops.aten.slice_scatter.default(slice_1214, copy_63, 2, 1, 256);  slice_1214 = copy_63 = None
        slice_scatter_232 = torch.ops.aten.slice_scatter.default(select_108, slice_scatter_231, 1, 1, 256);  select_108 = slice_scatter_231 = None
        select_scatter_21 = torch.ops.aten.select_scatter.default(slice_scatter_229, slice_scatter_232, 1, 0);  slice_scatter_229 = slice_scatter_232 = None
        full_default_50 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_20 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(iota_20, -2);  iota_20 = None
        iota_21 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        sub_41 = torch.ops.aten.sub.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
        le_10 = torch.ops.aten.le.Scalar(sub_41, 0);  sub_41 = None
        full_default_51 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40 = torch.ops.aten.where.self(le_10, full_default_50, full_default_51);  le_10 = full_default_50 = full_default_51 = None
        rev_20 = torch.ops.prims.rev.default(where_40, [0]);  where_40 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(rev_20, 0);  rev_20 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, 2);  unsqueeze_100 = None
        rev_21 = torch.ops.prims.rev.default(unsqueeze_101, [1, 3])
        expand_20 = torch.ops.aten.expand.default(unsqueeze_101, [4, 256, 12, 257]);  unsqueeze_101 = None
        view_382 = torch.ops.aten.view.default(select_scatter_21, [4, 12, 1024, 513])
        permute_334 = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        slice_1225 = torch.ops.aten.slice.Tensor(permute_334, 1, 0, 256);  permute_334 = None
        slice_1227 = torch.ops.aten.slice.Tensor(slice_1225, 3, 0, 257);  slice_1225 = None
        full_default_52 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(expand_20, torch.bool);  expand_20 = None
        where_41 = torch.ops.aten.where.self(convert_element_type_25, full_default_52, slice_1227);  convert_element_type_25 = full_default_52 = slice_1227 = None
        view_383 = torch.ops.aten.view.default(select_scatter_21, [4, 12, 1024, 513])
        permute_335 = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        slice_1233 = torch.ops.aten.slice.Tensor(permute_335, 1, 0, 256);  permute_335 = None
        slice_1235 = torch.ops.aten.slice.Tensor(slice_1233, 3, 0, 257);  slice_1233 = None
        copy_64 = torch.ops.aten.copy.default(slice_1235, where_41);  slice_1235 = where_41 = None
        view_384 = torch.ops.aten.view.default(select_scatter_21, [4, 12, 1024, 513]);  select_scatter_21 = None
        permute_336 = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        slice_1237 = torch.ops.aten.slice.Tensor(permute_336, 1, 0, 256)
        slice_scatter_234 = torch.ops.aten.slice_scatter.default(slice_1237, copy_64, 3, 0, 257);  slice_1237 = copy_64 = None
        slice_scatter_236 = torch.ops.aten.slice_scatter.default(permute_336, slice_scatter_234, 1, 0, 256);  permute_336 = slice_scatter_234 = None
        permute_337 = torch.ops.aten.permute.default(slice_scatter_236, [0, 2, 1, 3]);  slice_scatter_236 = None
        view_385 = torch.ops.aten.view.default(permute_337, [48, 4, 256, 513]);  permute_337 = None
        expand_21 = torch.ops.aten.expand.default(rev_21, [4, 256, 12, 257]);  rev_21 = None
        view_387 = torch.ops.aten.view.default(view_385, [4, 12, 1024, 513])
        permute_339 = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        slice_1248 = torch.ops.aten.slice.Tensor(permute_339, 1, -256, 9223372036854775807);  permute_339 = None
        slice_1250 = torch.ops.aten.slice.Tensor(slice_1248, 3, -257, 9223372036854775807);  slice_1248 = None
        full_default_53 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(expand_21, torch.bool);  expand_21 = None
        where_42 = torch.ops.aten.where.self(convert_element_type_26, full_default_53, slice_1250);  convert_element_type_26 = full_default_53 = slice_1250 = None
        view_388 = torch.ops.aten.view.default(view_385, [4, 12, 1024, 513])
        permute_340 = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        slice_1256 = torch.ops.aten.slice.Tensor(permute_340, 1, -256, 9223372036854775807);  permute_340 = None
        slice_1258 = torch.ops.aten.slice.Tensor(slice_1256, 3, -257, 9223372036854775807);  slice_1256 = None
        copy_65 = torch.ops.aten.copy.default(slice_1258, where_42);  slice_1258 = where_42 = None
        view_389 = torch.ops.aten.view.default(view_385, [4, 12, 1024, 513]);  view_385 = None
        permute_341 = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
        slice_1260 = torch.ops.aten.slice.Tensor(permute_341, 1, -256, 9223372036854775807)
        slice_scatter_238 = torch.ops.aten.slice_scatter.default(slice_1260, copy_65, 3, -257, 9223372036854775807);  slice_1260 = copy_65 = None
        slice_scatter_240 = torch.ops.aten.slice_scatter.default(permute_341, slice_scatter_238, 1, -256, 9223372036854775807);  permute_341 = slice_scatter_238 = None
        permute_342 = torch.ops.aten.permute.default(slice_scatter_240, [0, 2, 1, 3]);  slice_scatter_240 = None
        view_390 = torch.ops.aten.view.default(permute_342, [48, 4, 256, 513]);  permute_342 = None
        ne_5 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(ne_5, 2);  ne_5 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(unsqueeze_103, torch.float32)
        full_default_54 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43 = torch.ops.aten.where.self(unsqueeze_103, full_default_54, convert_element_type_27);  unsqueeze_103 = full_default_54 = convert_element_type_27 = None
        full_49 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_344 = torch.ops.aten.permute.default(full_49, [0, 2, 1, 3]);  full_49 = None
        view_392 = torch.ops.aten.view.default(permute_344, [4, 1024, 1]);  permute_344 = None
        permute_345 = torch.ops.aten.permute.default(where_43, [0, 2, 1, 3]);  where_43 = None
        view_393 = torch.ops.aten.view.default(permute_345, [4, 1024, 1]);  permute_345 = None
        view_394 = torch.ops.aten.view.default(view_392, [4, 2, 512, 1]);  view_392 = None
        as_strided_33 = torch.ops.aten.as_strided.default(view_394, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_394 = None
        view_395 = torch.ops.aten.view.default(view_393, [4, 2, 512, 1]);  view_393 = None
        as_strided_34 = torch.ops.aten.as_strided.default(view_395, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_395 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
        permute_346 = torch.ops.aten.permute.default(unsqueeze_104, [0, 1, 2, 4, 3]);  unsqueeze_104 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
        permute_347 = torch.ops.aten.permute.default(unsqueeze_105, [0, 1, 4, 2, 3]);  unsqueeze_105 = None
        mul_40 = torch.ops.aten.mul.Tensor(permute_346, permute_347);  permute_346 = permute_347 = None
        view_396 = torch.ops.aten.view.default(mul_40, [4, 3, 512, 512]);  mul_40 = None
        constant_pad_nd_21 = torch.ops.aten.constant_pad_nd.default(view_396, [0, 0, 0, 1], 0.0);  view_396 = None
        view_397 = torch.ops.aten.view.default(constant_pad_nd_21, [4, 3, 512, 513]);  constant_pad_nd_21 = None
        full_50 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1270 = torch.ops.aten.slice.Tensor(view_397, 2, 0, 256)
        slice_1271 = torch.ops.aten.slice.Tensor(slice_1270, 3, 0, 257);  slice_1270 = None
        slice_1273 = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_1275 = torch.ops.aten.slice.Tensor(slice_1273, 3, 256, 9223372036854775807);  slice_1273 = None
        copy_66 = torch.ops.aten.copy.default(slice_1275, slice_1271);  slice_1275 = slice_1271 = None
        slice_1277 = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_scatter_242 = torch.ops.aten.slice_scatter.default(slice_1277, copy_66, 3, 256, 9223372036854775807);  slice_1277 = copy_66 = None
        slice_scatter_244 = torch.ops.aten.slice_scatter.default(full_50, slice_scatter_242, 1, 0, -1);  full_50 = slice_scatter_242 = None
        select_110 = torch.ops.aten.select.int(view_397, 1, -1)
        slice_1284 = torch.ops.aten.slice.Tensor(select_110, 1, 256, 9223372036854775807);  select_110 = None
        slice_1285 = torch.ops.aten.slice.Tensor(slice_1284, 2, 0, 257);  slice_1284 = None
        select_112 = torch.ops.aten.select.int(slice_scatter_244, 1, -1)
        slice_1291 = torch.ops.aten.slice.Tensor(select_112, 2, 256, 9223372036854775807);  select_112 = None
        copy_67 = torch.ops.aten.copy.default(slice_1291, slice_1285);  slice_1291 = slice_1285 = None
        select_113 = torch.ops.aten.select.int(slice_scatter_244, 1, -1)
        slice_scatter_246 = torch.ops.aten.slice_scatter.default(select_113, copy_67, 2, 256, 9223372036854775807);  select_113 = copy_67 = None
        select_scatter_22 = torch.ops.aten.select_scatter.default(slice_scatter_244, slice_scatter_246, 1, -1);  slice_scatter_244 = slice_scatter_246 = None
        slice_1299 = torch.ops.aten.slice.Tensor(view_397, 2, -257, -1)
        slice_1300 = torch.ops.aten.slice.Tensor(slice_1299, 3, 257, 9223372036854775807);  slice_1299 = None
        slice_1306 = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_1308 = torch.ops.aten.slice.Tensor(slice_1306, 3, 0, 256);  slice_1306 = None
        copy_68 = torch.ops.aten.copy.default(slice_1308, slice_1300);  slice_1308 = slice_1300 = None
        slice_1310 = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_scatter_249 = torch.ops.aten.slice_scatter.default(slice_1310, copy_68, 3, 0, 256);  slice_1310 = copy_68 = None
        slice_scatter_251 = torch.ops.aten.slice_scatter.default(select_scatter_22, slice_scatter_249, 1, 1, 9223372036854775807);  select_scatter_22 = slice_scatter_249 = None
        select_115 = torch.ops.aten.select.int(view_397, 1, 0);  view_397 = None
        slice_1317 = torch.ops.aten.slice.Tensor(select_115, 1, 0, 255);  select_115 = None
        slice_1318 = torch.ops.aten.slice.Tensor(slice_1317, 2, -255, 9223372036854775807);  slice_1317 = None
        select_117 = torch.ops.aten.select.int(slice_scatter_251, 1, 0)
        slice_1323 = torch.ops.aten.slice.Tensor(select_117, 1, 1, 256);  select_117 = None
        slice_1324 = torch.ops.aten.slice.Tensor(slice_1323, 2, 1, 256);  slice_1323 = None
        copy_69 = torch.ops.aten.copy.default(slice_1324, slice_1318);  slice_1324 = slice_1318 = None
        select_118 = torch.ops.aten.select.int(slice_scatter_251, 1, 0)
        slice_1326 = torch.ops.aten.slice.Tensor(select_118, 1, 1, 256)
        slice_scatter_253 = torch.ops.aten.slice_scatter.default(slice_1326, copy_69, 2, 1, 256);  slice_1326 = copy_69 = None
        slice_scatter_254 = torch.ops.aten.slice_scatter.default(select_118, slice_scatter_253, 1, 1, 256);  select_118 = slice_scatter_253 = None
        select_scatter_23 = torch.ops.aten.select_scatter.default(slice_scatter_251, slice_scatter_254, 1, 0);  slice_scatter_251 = slice_scatter_254 = None
        full_default_55 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_22 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(iota_22, -2);  iota_22 = None
        iota_23 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
        sub_43 = torch.ops.aten.sub.Tensor(unsqueeze_106, unsqueeze_107);  unsqueeze_106 = unsqueeze_107 = None
        le_11 = torch.ops.aten.le.Scalar(sub_43, 0);  sub_43 = None
        full_default_56 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44 = torch.ops.aten.where.self(le_11, full_default_55, full_default_56);  le_11 = full_default_55 = full_default_56 = None
        rev_22 = torch.ops.prims.rev.default(where_44, [0]);  where_44 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(rev_22, 0);  rev_22 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        rev_23 = torch.ops.prims.rev.default(unsqueeze_109, [1, 3])
        expand_22 = torch.ops.aten.expand.default(unsqueeze_109, [4, 256, 1, 257]);  unsqueeze_109 = None
        view_400 = torch.ops.aten.view.default(select_scatter_23, [4, 1, 1024, 513])
        permute_350 = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        slice_1337 = torch.ops.aten.slice.Tensor(permute_350, 1, 0, 256);  permute_350 = None
        slice_1339 = torch.ops.aten.slice.Tensor(slice_1337, 3, 0, 257);  slice_1337 = None
        full_default_57 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(expand_22, torch.bool);  expand_22 = None
        where_45 = torch.ops.aten.where.self(convert_element_type_28, full_default_57, slice_1339);  convert_element_type_28 = full_default_57 = slice_1339 = None
        view_401 = torch.ops.aten.view.default(select_scatter_23, [4, 1, 1024, 513])
        permute_351 = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
        slice_1345 = torch.ops.aten.slice.Tensor(permute_351, 1, 0, 256);  permute_351 = None
        slice_1347 = torch.ops.aten.slice.Tensor(slice_1345, 3, 0, 257);  slice_1345 = None
        copy_70 = torch.ops.aten.copy.default(slice_1347, where_45);  slice_1347 = where_45 = None
        view_402 = torch.ops.aten.view.default(select_scatter_23, [4, 1, 1024, 513]);  select_scatter_23 = None
        permute_352 = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        slice_1349 = torch.ops.aten.slice.Tensor(permute_352, 1, 0, 256)
        slice_scatter_256 = torch.ops.aten.slice_scatter.default(slice_1349, copy_70, 3, 0, 257);  slice_1349 = copy_70 = None
        slice_scatter_258 = torch.ops.aten.slice_scatter.default(permute_352, slice_scatter_256, 1, 0, 256);  permute_352 = slice_scatter_256 = None
        permute_353 = torch.ops.aten.permute.default(slice_scatter_258, [0, 2, 1, 3]);  slice_scatter_258 = None
        view_403 = torch.ops.aten.view.default(permute_353, [4, 4, 256, 513]);  permute_353 = None
        expand_23 = torch.ops.aten.expand.default(rev_23, [4, 256, 1, 257]);  rev_23 = None
        view_405 = torch.ops.aten.view.default(view_403, [4, 1, 1024, 513])
        permute_355 = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        slice_1360 = torch.ops.aten.slice.Tensor(permute_355, 1, -256, 9223372036854775807);  permute_355 = None
        slice_1362 = torch.ops.aten.slice.Tensor(slice_1360, 3, -257, 9223372036854775807);  slice_1360 = None
        full_default_58 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(expand_23, torch.bool);  expand_23 = None
        where_46 = torch.ops.aten.where.self(convert_element_type_29, full_default_58, slice_1362);  convert_element_type_29 = full_default_58 = slice_1362 = None
        view_406 = torch.ops.aten.view.default(view_403, [4, 1, 1024, 513])
        permute_356 = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        slice_1368 = torch.ops.aten.slice.Tensor(permute_356, 1, -256, 9223372036854775807);  permute_356 = None
        slice_1370 = torch.ops.aten.slice.Tensor(slice_1368, 3, -257, 9223372036854775807);  slice_1368 = None
        copy_71 = torch.ops.aten.copy.default(slice_1370, where_46);  slice_1370 = where_46 = None
        view_407 = torch.ops.aten.view.default(view_403, [4, 1, 1024, 513]);  view_403 = None
        permute_357 = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        slice_1372 = torch.ops.aten.slice.Tensor(permute_357, 1, -256, 9223372036854775807)
        slice_scatter_260 = torch.ops.aten.slice_scatter.default(slice_1372, copy_71, 3, -257, 9223372036854775807);  slice_1372 = copy_71 = None
        slice_scatter_262 = torch.ops.aten.slice_scatter.default(permute_357, slice_scatter_260, 1, -256, 9223372036854775807);  permute_357 = slice_scatter_260 = None
        permute_358 = torch.ops.aten.permute.default(slice_scatter_262, [0, 2, 1, 3]);  slice_scatter_262 = None
        view_408 = torch.ops.aten.view.default(permute_358, [4, 4, 256, 513]);  permute_358 = None
        view_410 = torch.ops.aten.view.default(view_390, [4, 12, 1024, 513]);  view_390 = None
        permute_360 = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        view_411 = torch.ops.aten.view.default(view_408, [4, 1, 1024, 513]);  view_408 = None
        permute_361 = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        add_80 = torch.ops.aten.add.Tensor(permute_360, permute_361);  permute_360 = permute_361 = None
        permute_362 = torch.ops.aten.permute.default(add_80, [0, 2, 1, 3]);  add_80 = None
        view_413 = torch.ops.aten.view.default(permute_362, [48, 4, 256, 513]);  permute_362 = None
        view_414 = torch.ops.aten.view.default(view_413, [4, 12, 1024, 513]);  view_413 = None
        permute_363 = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        clone_70 = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
        amax_5 = torch.ops.aten.amax.default(clone_70, [-1], True)
        sub_44 = torch.ops.aten.sub.Tensor(clone_70, amax_5);  clone_70 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_57 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, 3);  unsqueeze_110 = None
        full_default_59 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47 = torch.ops.aten.where.self(unsqueeze_111, full_default_59, div_57);  unsqueeze_111 = full_default_59 = div_57 = None
        view_415 = torch.ops.aten.view.default(add_77, [1024, 4, 12, 64]);  add_77 = None
        permute_364 = torch.ops.aten.permute.default(view_415, [1, 0, 2, 3]);  view_415 = None
        permute_365 = torch.ops.aten.permute.default(where_47, [0, 2, 1, 3]);  where_47 = None
        clone_72 = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_416 = torch.ops.aten.view.default(clone_72, [48, 4, 256, 513]);  clone_72 = None
        permute_366 = torch.ops.aten.permute.default(permute_364, [0, 2, 1, 3]);  permute_364 = None
        view_417 = torch.ops.aten.view.default(permute_366, [48, 1024, 64]);  permute_366 = None
        constant_pad_nd_22 = torch.ops.aten.constant_pad_nd.default(view_417, [0, 0, 256, 256], -1.0);  view_417 = None
        as_strided_35 = torch.ops.aten.as_strided.default(constant_pad_nd_22, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_22 = None
        constant_pad_nd_23 = torch.ops.aten.constant_pad_nd.default(view_416, [0, 257], 0.0);  view_416 = None
        view_418 = torch.ops.aten.view.default(constant_pad_nd_23, [48, 4, -1]);  constant_pad_nd_23 = None
        slice_1382 = torch.ops.aten.slice.Tensor(view_418, 2, 0, -256);  view_418 = None
        view_419 = torch.ops.aten.view.default(slice_1382, [48, 4, 256, 769]);  slice_1382 = None
        slice_1386 = torch.ops.aten.slice.Tensor(view_419, 3, 0, -1);  view_419 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(slice_1386, 4);  slice_1386 = None
        permute_367 = torch.ops.aten.permute.default(unsqueeze_112, [0, 1, 2, 4, 3]);  unsqueeze_112 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        permute_368 = torch.ops.aten.permute.default(unsqueeze_113, [0, 1, 4, 3, 2]);  unsqueeze_113 = None
        permute_369 = torch.ops.aten.permute.default(permute_367, [0, 1, 2, 4, 3]);  permute_367 = None
        view_420 = torch.ops.aten.view.default(permute_369, [192, 256, 768]);  permute_369 = None
        permute_370 = torch.ops.aten.permute.default(permute_368, [0, 1, 4, 3, 2]);  permute_368 = None
        clone_73 = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        view_421 = torch.ops.aten.view.default(clone_73, [192, 768, 64]);  clone_73 = None
        bmm_11 = torch.ops.aten.bmm.default(view_420, view_421);  view_420 = view_421 = None
        view_422 = torch.ops.aten.view.default(bmm_11, [48, 4, 256, 1, 64]);  bmm_11 = None
        permute_371 = torch.ops.aten.permute.default(view_422, [0, 1, 2, 4, 3]);  view_422 = None
        view_423 = torch.ops.aten.view.default(permute_371, [48, 4, 256, 64]);  permute_371 = None
        view_424 = torch.ops.aten.view.default(view_423, [4, 12, 1024, 64]);  view_423 = None
        permute_372 = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        permute_373 = torch.ops.aten.permute.default(permute_372, [1, 0, 2, 3]);  permute_372 = None
        clone_74 = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
        view_425 = torch.ops.aten.view.default(clone_74, [1024, 4, 768]);  clone_74 = None
        permute_374 = torch.ops.aten.permute.default(view_425, [1, 0, 2]);  view_425 = None
        permute_375 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        clone_75 = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        view_426 = torch.ops.aten.view.default(clone_75, [4096, 768]);  clone_75 = None
        mm_23 = torch.ops.aten.mm.default(view_426, permute_375);  view_426 = permute_375 = None
        view_427 = torch.ops.aten.view.default(mm_23, [4, 1024, 768]);  mm_23 = None
        add_82 = torch.ops.aten.add.Tensor(view_427, arg90_1);  view_427 = arg90_1 = None
        add_83 = torch.ops.aten.add.Tensor(add_82, add_74);  add_82 = add_74 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_83, getitem_21);  add_83 = getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = rsqrt_10 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg91_1);  mul_41 = arg91_1 = None
        add_85 = torch.ops.aten.add.Tensor(mul_42, arg92_1);  mul_42 = arg92_1 = None
        view_428 = torch.ops.aten.view.default(add_85, [4096, 768])
        permute_376 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg94_1, view_428, permute_376);  arg94_1 = view_428 = permute_376 = None
        view_429 = torch.ops.aten.view.default(addmm_10, [4, 1024, 3072]);  addmm_10 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_429, 0.5)
        mul_44 = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476);  view_429 = None
        erf_5 = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_86 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_43, add_86);  mul_43 = add_86 = None
        view_430 = torch.ops.aten.view.default(mul_45, [4096, 3072]);  mul_45 = None
        permute_377 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg96_1, view_430, permute_377);  arg96_1 = view_430 = permute_377 = None
        view_431 = torch.ops.aten.view.default(addmm_11, [4, 1024, 768]);  addmm_11 = None
        add_87 = torch.ops.aten.add.Tensor(view_431, add_85);  view_431 = add_85 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_87, getitem_23);  add_87 = getitem_23 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = rsqrt_11 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, arg97_1);  mul_46 = arg97_1 = None
        add_89 = torch.ops.aten.add.Tensor(mul_47, arg98_1);  mul_47 = arg98_1 = None
        permute_378 = torch.ops.aten.permute.default(add_89, [1, 0, 2])
        permute_379 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        clone_78 = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format)
        view_432 = torch.ops.aten.view.default(clone_78, [4096, 768]);  clone_78 = None
        mm_24 = torch.ops.aten.mm.default(view_432, permute_379);  view_432 = permute_379 = None
        view_433 = torch.ops.aten.view.default(mm_24, [1024, 4, 768]);  mm_24 = None
        add_90 = torch.ops.aten.add.Tensor(view_433, arg100_1);  view_433 = arg100_1 = None
        permute_380 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        clone_79 = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format)
        view_434 = torch.ops.aten.view.default(clone_79, [4096, 768]);  clone_79 = None
        mm_25 = torch.ops.aten.mm.default(view_434, permute_380);  view_434 = permute_380 = None
        view_435 = torch.ops.aten.view.default(mm_25, [1024, 4, 768]);  mm_25 = None
        add_91 = torch.ops.aten.add.Tensor(view_435, arg102_1);  view_435 = arg102_1 = None
        permute_381 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        clone_80 = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
        view_436 = torch.ops.aten.view.default(clone_80, [4096, 768]);  clone_80 = None
        mm_26 = torch.ops.aten.mm.default(view_436, permute_381);  view_436 = permute_381 = None
        view_437 = torch.ops.aten.view.default(mm_26, [1024, 4, 768]);  mm_26 = None
        add_92 = torch.ops.aten.add.Tensor(view_437, arg104_1);  view_437 = arg104_1 = None
        div_60 = torch.ops.aten.div.Tensor(add_90, 8.0);  add_90 = None
        view_439 = torch.ops.aten.view.default(add_91, [1024, 4, 12, 64]);  add_91 = None
        permute_383 = torch.ops.aten.permute.default(view_439, [1, 0, 2, 3]);  view_439 = None
        permute_385 = torch.ops.aten.permute.default(permute_383, [0, 2, 1, 3]);  permute_383 = None
        view_441 = torch.ops.aten.view.default(permute_385, [48, 1024, 64]);  permute_385 = None
        view_443 = torch.ops.aten.view.default(view_441, [48, 2, 512, 64]);  view_441 = None
        as_strided_37 = torch.ops.aten.as_strided.default(view_443, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_443 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        permute_387 = torch.ops.aten.permute.default(unsqueeze_115, [0, 1, 4, 2, 3]);  unsqueeze_115 = None
        view_444 = torch.ops.aten.view.default(div_60, [1024, 4, 12, 64]);  div_60 = None
        permute_389 = torch.ops.aten.permute.default(view_444, [1, 0, 2, 3]);  view_444 = None
        permute_390 = torch.ops.aten.permute.default(permute_389, [0, 2, 1, 3]);  permute_389 = None
        view_445 = torch.ops.aten.view.default(permute_390, [48, 1024, 64]);  permute_390 = None
        view_446 = torch.ops.aten.view.default(view_445, [48, 2, 512, 64]);  view_445 = None
        as_strided_38 = torch.ops.aten.as_strided.default(view_446, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_446 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(as_strided_38, 4);  as_strided_38 = None
        permute_391 = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None
        permute_392 = torch.ops.aten.permute.default(permute_391, [0, 1, 2, 4, 3]);  permute_391 = None
        clone_81 = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_447 = torch.ops.aten.view.default(clone_81, [144, 512, 64]);  clone_81 = None
        permute_393 = torch.ops.aten.permute.default(permute_387, [0, 1, 4, 3, 2]);  permute_387 = None
        clone_82 = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        view_448 = torch.ops.aten.view.default(clone_82, [144, 64, 512]);  clone_82 = None
        bmm_12 = torch.ops.aten.bmm.default(view_447, view_448);  view_447 = view_448 = None
        view_449 = torch.ops.aten.view.default(bmm_12, [48, 3, 512, 1, 512]);  bmm_12 = None
        permute_394 = torch.ops.aten.permute.default(view_449, [0, 1, 2, 4, 3]);  view_449 = None
        view_450 = torch.ops.aten.view.default(permute_394, [48, 3, 512, 512]);  permute_394 = None
        constant_pad_nd_24 = torch.ops.aten.constant_pad_nd.default(view_450, [0, 0, 0, 1], 0.0);  view_450 = None
        view_451 = torch.ops.aten.view.default(constant_pad_nd_24, [48, 3, 512, 513]);  constant_pad_nd_24 = None
        full_54 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1389 = torch.ops.aten.slice.Tensor(view_451, 2, 0, 256)
        slice_1390 = torch.ops.aten.slice.Tensor(slice_1389, 3, 0, 257);  slice_1389 = None
        slice_1392 = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_1394 = torch.ops.aten.slice.Tensor(slice_1392, 3, 256, 9223372036854775807);  slice_1392 = None
        copy_72 = torch.ops.aten.copy.default(slice_1394, slice_1390);  slice_1394 = slice_1390 = None
        slice_1396 = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_scatter_264 = torch.ops.aten.slice_scatter.default(slice_1396, copy_72, 3, 256, 9223372036854775807);  slice_1396 = copy_72 = None
        slice_scatter_266 = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_264, 1, 0, -1);  full_54 = slice_scatter_264 = None
        select_120 = torch.ops.aten.select.int(view_451, 1, -1)
        slice_1403 = torch.ops.aten.slice.Tensor(select_120, 1, 256, 9223372036854775807);  select_120 = None
        slice_1404 = torch.ops.aten.slice.Tensor(slice_1403, 2, 0, 257);  slice_1403 = None
        select_122 = torch.ops.aten.select.int(slice_scatter_266, 1, -1)
        slice_1410 = torch.ops.aten.slice.Tensor(select_122, 2, 256, 9223372036854775807);  select_122 = None
        copy_73 = torch.ops.aten.copy.default(slice_1410, slice_1404);  slice_1410 = slice_1404 = None
        select_123 = torch.ops.aten.select.int(slice_scatter_266, 1, -1)
        slice_scatter_268 = torch.ops.aten.slice_scatter.default(select_123, copy_73, 2, 256, 9223372036854775807);  select_123 = copy_73 = None
        select_scatter_24 = torch.ops.aten.select_scatter.default(slice_scatter_266, slice_scatter_268, 1, -1);  slice_scatter_266 = slice_scatter_268 = None
        slice_1418 = torch.ops.aten.slice.Tensor(view_451, 2, -257, -1)
        slice_1419 = torch.ops.aten.slice.Tensor(slice_1418, 3, 257, 9223372036854775807);  slice_1418 = None
        slice_1425 = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_1427 = torch.ops.aten.slice.Tensor(slice_1425, 3, 0, 256);  slice_1425 = None
        copy_74 = torch.ops.aten.copy.default(slice_1427, slice_1419);  slice_1427 = slice_1419 = None
        slice_1429 = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_scatter_271 = torch.ops.aten.slice_scatter.default(slice_1429, copy_74, 3, 0, 256);  slice_1429 = copy_74 = None
        slice_scatter_273 = torch.ops.aten.slice_scatter.default(select_scatter_24, slice_scatter_271, 1, 1, 9223372036854775807);  select_scatter_24 = slice_scatter_271 = None
        select_125 = torch.ops.aten.select.int(view_451, 1, 0);  view_451 = None
        slice_1436 = torch.ops.aten.slice.Tensor(select_125, 1, 0, 255);  select_125 = None
        slice_1437 = torch.ops.aten.slice.Tensor(slice_1436, 2, -255, 9223372036854775807);  slice_1436 = None
        select_127 = torch.ops.aten.select.int(slice_scatter_273, 1, 0)
        slice_1442 = torch.ops.aten.slice.Tensor(select_127, 1, 1, 256);  select_127 = None
        slice_1443 = torch.ops.aten.slice.Tensor(slice_1442, 2, 1, 256);  slice_1442 = None
        copy_75 = torch.ops.aten.copy.default(slice_1443, slice_1437);  slice_1443 = slice_1437 = None
        select_128 = torch.ops.aten.select.int(slice_scatter_273, 1, 0)
        slice_1445 = torch.ops.aten.slice.Tensor(select_128, 1, 1, 256)
        slice_scatter_275 = torch.ops.aten.slice_scatter.default(slice_1445, copy_75, 2, 1, 256);  slice_1445 = copy_75 = None
        slice_scatter_276 = torch.ops.aten.slice_scatter.default(select_128, slice_scatter_275, 1, 1, 256);  select_128 = slice_scatter_275 = None
        select_scatter_25 = torch.ops.aten.select_scatter.default(slice_scatter_273, slice_scatter_276, 1, 0);  slice_scatter_273 = slice_scatter_276 = None
        full_default_60 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_24 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(iota_24, -2);  iota_24 = None
        iota_25 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        sub_49 = torch.ops.aten.sub.Tensor(unsqueeze_117, unsqueeze_118);  unsqueeze_117 = unsqueeze_118 = None
        le_12 = torch.ops.aten.le.Scalar(sub_49, 0);  sub_49 = None
        full_default_61 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48 = torch.ops.aten.where.self(le_12, full_default_60, full_default_61);  le_12 = full_default_60 = full_default_61 = None
        rev_24 = torch.ops.prims.rev.default(where_48, [0]);  where_48 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(rev_24, 0);  rev_24 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
        rev_25 = torch.ops.prims.rev.default(unsqueeze_120, [1, 3])
        expand_24 = torch.ops.aten.expand.default(unsqueeze_120, [4, 256, 12, 257]);  unsqueeze_120 = None
        view_454 = torch.ops.aten.view.default(select_scatter_25, [4, 12, 1024, 513])
        permute_397 = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        slice_1456 = torch.ops.aten.slice.Tensor(permute_397, 1, 0, 256);  permute_397 = None
        slice_1458 = torch.ops.aten.slice.Tensor(slice_1456, 3, 0, 257);  slice_1456 = None
        full_default_62 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(expand_24, torch.bool);  expand_24 = None
        where_49 = torch.ops.aten.where.self(convert_element_type_30, full_default_62, slice_1458);  convert_element_type_30 = full_default_62 = slice_1458 = None
        view_455 = torch.ops.aten.view.default(select_scatter_25, [4, 12, 1024, 513])
        permute_398 = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        slice_1464 = torch.ops.aten.slice.Tensor(permute_398, 1, 0, 256);  permute_398 = None
        slice_1466 = torch.ops.aten.slice.Tensor(slice_1464, 3, 0, 257);  slice_1464 = None
        copy_76 = torch.ops.aten.copy.default(slice_1466, where_49);  slice_1466 = where_49 = None
        view_456 = torch.ops.aten.view.default(select_scatter_25, [4, 12, 1024, 513]);  select_scatter_25 = None
        permute_399 = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        slice_1468 = torch.ops.aten.slice.Tensor(permute_399, 1, 0, 256)
        slice_scatter_278 = torch.ops.aten.slice_scatter.default(slice_1468, copy_76, 3, 0, 257);  slice_1468 = copy_76 = None
        slice_scatter_280 = torch.ops.aten.slice_scatter.default(permute_399, slice_scatter_278, 1, 0, 256);  permute_399 = slice_scatter_278 = None
        permute_400 = torch.ops.aten.permute.default(slice_scatter_280, [0, 2, 1, 3]);  slice_scatter_280 = None
        view_457 = torch.ops.aten.view.default(permute_400, [48, 4, 256, 513]);  permute_400 = None
        expand_25 = torch.ops.aten.expand.default(rev_25, [4, 256, 12, 257]);  rev_25 = None
        view_459 = torch.ops.aten.view.default(view_457, [4, 12, 1024, 513])
        permute_402 = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        slice_1479 = torch.ops.aten.slice.Tensor(permute_402, 1, -256, 9223372036854775807);  permute_402 = None
        slice_1481 = torch.ops.aten.slice.Tensor(slice_1479, 3, -257, 9223372036854775807);  slice_1479 = None
        full_default_63 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(expand_25, torch.bool);  expand_25 = None
        where_50 = torch.ops.aten.where.self(convert_element_type_31, full_default_63, slice_1481);  convert_element_type_31 = full_default_63 = slice_1481 = None
        view_460 = torch.ops.aten.view.default(view_457, [4, 12, 1024, 513])
        permute_403 = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        slice_1487 = torch.ops.aten.slice.Tensor(permute_403, 1, -256, 9223372036854775807);  permute_403 = None
        slice_1489 = torch.ops.aten.slice.Tensor(slice_1487, 3, -257, 9223372036854775807);  slice_1487 = None
        copy_77 = torch.ops.aten.copy.default(slice_1489, where_50);  slice_1489 = where_50 = None
        view_461 = torch.ops.aten.view.default(view_457, [4, 12, 1024, 513]);  view_457 = None
        permute_404 = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None
        slice_1491 = torch.ops.aten.slice.Tensor(permute_404, 1, -256, 9223372036854775807)
        slice_scatter_282 = torch.ops.aten.slice_scatter.default(slice_1491, copy_77, 3, -257, 9223372036854775807);  slice_1491 = copy_77 = None
        slice_scatter_284 = torch.ops.aten.slice_scatter.default(permute_404, slice_scatter_282, 1, -256, 9223372036854775807);  permute_404 = slice_scatter_282 = None
        permute_405 = torch.ops.aten.permute.default(slice_scatter_284, [0, 2, 1, 3]);  slice_scatter_284 = None
        view_462 = torch.ops.aten.view.default(permute_405, [48, 4, 256, 513]);  permute_405 = None
        ne_6 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(ne_6, 2);  ne_6 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(unsqueeze_122, torch.float32)
        full_default_64 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_51 = torch.ops.aten.where.self(unsqueeze_122, full_default_64, convert_element_type_32);  unsqueeze_122 = full_default_64 = convert_element_type_32 = None
        full_58 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_407 = torch.ops.aten.permute.default(full_58, [0, 2, 1, 3]);  full_58 = None
        view_464 = torch.ops.aten.view.default(permute_407, [4, 1024, 1]);  permute_407 = None
        permute_408 = torch.ops.aten.permute.default(where_51, [0, 2, 1, 3]);  where_51 = None
        view_465 = torch.ops.aten.view.default(permute_408, [4, 1024, 1]);  permute_408 = None
        view_466 = torch.ops.aten.view.default(view_464, [4, 2, 512, 1]);  view_464 = None
        as_strided_39 = torch.ops.aten.as_strided.default(view_466, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_466 = None
        view_467 = torch.ops.aten.view.default(view_465, [4, 2, 512, 1]);  view_465 = None
        as_strided_40 = torch.ops.aten.as_strided.default(view_467, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_467 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(as_strided_39, 4);  as_strided_39 = None
        permute_409 = torch.ops.aten.permute.default(unsqueeze_123, [0, 1, 2, 4, 3]);  unsqueeze_123 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(as_strided_40, 4);  as_strided_40 = None
        permute_410 = torch.ops.aten.permute.default(unsqueeze_124, [0, 1, 4, 2, 3]);  unsqueeze_124 = None
        mul_48 = torch.ops.aten.mul.Tensor(permute_409, permute_410);  permute_409 = permute_410 = None
        view_468 = torch.ops.aten.view.default(mul_48, [4, 3, 512, 512]);  mul_48 = None
        constant_pad_nd_25 = torch.ops.aten.constant_pad_nd.default(view_468, [0, 0, 0, 1], 0.0);  view_468 = None
        view_469 = torch.ops.aten.view.default(constant_pad_nd_25, [4, 3, 512, 513]);  constant_pad_nd_25 = None
        full_59 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1501 = torch.ops.aten.slice.Tensor(view_469, 2, 0, 256)
        slice_1502 = torch.ops.aten.slice.Tensor(slice_1501, 3, 0, 257);  slice_1501 = None
        slice_1504 = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_1506 = torch.ops.aten.slice.Tensor(slice_1504, 3, 256, 9223372036854775807);  slice_1504 = None
        copy_78 = torch.ops.aten.copy.default(slice_1506, slice_1502);  slice_1506 = slice_1502 = None
        slice_1508 = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_scatter_286 = torch.ops.aten.slice_scatter.default(slice_1508, copy_78, 3, 256, 9223372036854775807);  slice_1508 = copy_78 = None
        slice_scatter_288 = torch.ops.aten.slice_scatter.default(full_59, slice_scatter_286, 1, 0, -1);  full_59 = slice_scatter_286 = None
        select_130 = torch.ops.aten.select.int(view_469, 1, -1)
        slice_1515 = torch.ops.aten.slice.Tensor(select_130, 1, 256, 9223372036854775807);  select_130 = None
        slice_1516 = torch.ops.aten.slice.Tensor(slice_1515, 2, 0, 257);  slice_1515 = None
        select_132 = torch.ops.aten.select.int(slice_scatter_288, 1, -1)
        slice_1522 = torch.ops.aten.slice.Tensor(select_132, 2, 256, 9223372036854775807);  select_132 = None
        copy_79 = torch.ops.aten.copy.default(slice_1522, slice_1516);  slice_1522 = slice_1516 = None
        select_133 = torch.ops.aten.select.int(slice_scatter_288, 1, -1)
        slice_scatter_290 = torch.ops.aten.slice_scatter.default(select_133, copy_79, 2, 256, 9223372036854775807);  select_133 = copy_79 = None
        select_scatter_26 = torch.ops.aten.select_scatter.default(slice_scatter_288, slice_scatter_290, 1, -1);  slice_scatter_288 = slice_scatter_290 = None
        slice_1530 = torch.ops.aten.slice.Tensor(view_469, 2, -257, -1)
        slice_1531 = torch.ops.aten.slice.Tensor(slice_1530, 3, 257, 9223372036854775807);  slice_1530 = None
        slice_1537 = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_1539 = torch.ops.aten.slice.Tensor(slice_1537, 3, 0, 256);  slice_1537 = None
        copy_80 = torch.ops.aten.copy.default(slice_1539, slice_1531);  slice_1539 = slice_1531 = None
        slice_1541 = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_scatter_293 = torch.ops.aten.slice_scatter.default(slice_1541, copy_80, 3, 0, 256);  slice_1541 = copy_80 = None
        slice_scatter_295 = torch.ops.aten.slice_scatter.default(select_scatter_26, slice_scatter_293, 1, 1, 9223372036854775807);  select_scatter_26 = slice_scatter_293 = None
        select_135 = torch.ops.aten.select.int(view_469, 1, 0);  view_469 = None
        slice_1548 = torch.ops.aten.slice.Tensor(select_135, 1, 0, 255);  select_135 = None
        slice_1549 = torch.ops.aten.slice.Tensor(slice_1548, 2, -255, 9223372036854775807);  slice_1548 = None
        select_137 = torch.ops.aten.select.int(slice_scatter_295, 1, 0)
        slice_1554 = torch.ops.aten.slice.Tensor(select_137, 1, 1, 256);  select_137 = None
        slice_1555 = torch.ops.aten.slice.Tensor(slice_1554, 2, 1, 256);  slice_1554 = None
        copy_81 = torch.ops.aten.copy.default(slice_1555, slice_1549);  slice_1555 = slice_1549 = None
        select_138 = torch.ops.aten.select.int(slice_scatter_295, 1, 0)
        slice_1557 = torch.ops.aten.slice.Tensor(select_138, 1, 1, 256)
        slice_scatter_297 = torch.ops.aten.slice_scatter.default(slice_1557, copy_81, 2, 1, 256);  slice_1557 = copy_81 = None
        slice_scatter_298 = torch.ops.aten.slice_scatter.default(select_138, slice_scatter_297, 1, 1, 256);  select_138 = slice_scatter_297 = None
        select_scatter_27 = torch.ops.aten.select_scatter.default(slice_scatter_295, slice_scatter_298, 1, 0);  slice_scatter_295 = slice_scatter_298 = None
        full_default_65 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_26 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(iota_26, -2);  iota_26 = None
        iota_27 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
        sub_51 = torch.ops.aten.sub.Tensor(unsqueeze_125, unsqueeze_126);  unsqueeze_125 = unsqueeze_126 = None
        le_13 = torch.ops.aten.le.Scalar(sub_51, 0);  sub_51 = None
        full_default_66 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_52 = torch.ops.aten.where.self(le_13, full_default_65, full_default_66);  le_13 = full_default_65 = full_default_66 = None
        rev_26 = torch.ops.prims.rev.default(where_52, [0]);  where_52 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(rev_26, 0);  rev_26 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        rev_27 = torch.ops.prims.rev.default(unsqueeze_128, [1, 3])
        expand_26 = torch.ops.aten.expand.default(unsqueeze_128, [4, 256, 1, 257]);  unsqueeze_128 = None
        view_472 = torch.ops.aten.view.default(select_scatter_27, [4, 1, 1024, 513])
        permute_413 = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        slice_1568 = torch.ops.aten.slice.Tensor(permute_413, 1, 0, 256);  permute_413 = None
        slice_1570 = torch.ops.aten.slice.Tensor(slice_1568, 3, 0, 257);  slice_1568 = None
        full_default_67 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(expand_26, torch.bool);  expand_26 = None
        where_53 = torch.ops.aten.where.self(convert_element_type_33, full_default_67, slice_1570);  convert_element_type_33 = full_default_67 = slice_1570 = None
        view_473 = torch.ops.aten.view.default(select_scatter_27, [4, 1, 1024, 513])
        permute_414 = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None
        slice_1576 = torch.ops.aten.slice.Tensor(permute_414, 1, 0, 256);  permute_414 = None
        slice_1578 = torch.ops.aten.slice.Tensor(slice_1576, 3, 0, 257);  slice_1576 = None
        copy_82 = torch.ops.aten.copy.default(slice_1578, where_53);  slice_1578 = where_53 = None
        view_474 = torch.ops.aten.view.default(select_scatter_27, [4, 1, 1024, 513]);  select_scatter_27 = None
        permute_415 = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
        slice_1580 = torch.ops.aten.slice.Tensor(permute_415, 1, 0, 256)
        slice_scatter_300 = torch.ops.aten.slice_scatter.default(slice_1580, copy_82, 3, 0, 257);  slice_1580 = copy_82 = None
        slice_scatter_302 = torch.ops.aten.slice_scatter.default(permute_415, slice_scatter_300, 1, 0, 256);  permute_415 = slice_scatter_300 = None
        permute_416 = torch.ops.aten.permute.default(slice_scatter_302, [0, 2, 1, 3]);  slice_scatter_302 = None
        view_475 = torch.ops.aten.view.default(permute_416, [4, 4, 256, 513]);  permute_416 = None
        expand_27 = torch.ops.aten.expand.default(rev_27, [4, 256, 1, 257]);  rev_27 = None
        view_477 = torch.ops.aten.view.default(view_475, [4, 1, 1024, 513])
        permute_418 = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        slice_1591 = torch.ops.aten.slice.Tensor(permute_418, 1, -256, 9223372036854775807);  permute_418 = None
        slice_1593 = torch.ops.aten.slice.Tensor(slice_1591, 3, -257, 9223372036854775807);  slice_1591 = None
        full_default_68 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(expand_27, torch.bool);  expand_27 = None
        where_54 = torch.ops.aten.where.self(convert_element_type_34, full_default_68, slice_1593);  convert_element_type_34 = full_default_68 = slice_1593 = None
        view_478 = torch.ops.aten.view.default(view_475, [4, 1, 1024, 513])
        permute_419 = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        slice_1599 = torch.ops.aten.slice.Tensor(permute_419, 1, -256, 9223372036854775807);  permute_419 = None
        slice_1601 = torch.ops.aten.slice.Tensor(slice_1599, 3, -257, 9223372036854775807);  slice_1599 = None
        copy_83 = torch.ops.aten.copy.default(slice_1601, where_54);  slice_1601 = where_54 = None
        view_479 = torch.ops.aten.view.default(view_475, [4, 1, 1024, 513]);  view_475 = None
        permute_420 = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        slice_1603 = torch.ops.aten.slice.Tensor(permute_420, 1, -256, 9223372036854775807)
        slice_scatter_304 = torch.ops.aten.slice_scatter.default(slice_1603, copy_83, 3, -257, 9223372036854775807);  slice_1603 = copy_83 = None
        slice_scatter_306 = torch.ops.aten.slice_scatter.default(permute_420, slice_scatter_304, 1, -256, 9223372036854775807);  permute_420 = slice_scatter_304 = None
        permute_421 = torch.ops.aten.permute.default(slice_scatter_306, [0, 2, 1, 3]);  slice_scatter_306 = None
        view_480 = torch.ops.aten.view.default(permute_421, [4, 4, 256, 513]);  permute_421 = None
        view_482 = torch.ops.aten.view.default(view_462, [4, 12, 1024, 513]);  view_462 = None
        permute_423 = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        view_483 = torch.ops.aten.view.default(view_480, [4, 1, 1024, 513]);  view_480 = None
        permute_424 = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
        add_95 = torch.ops.aten.add.Tensor(permute_423, permute_424);  permute_423 = permute_424 = None
        permute_425 = torch.ops.aten.permute.default(add_95, [0, 2, 1, 3]);  add_95 = None
        view_485 = torch.ops.aten.view.default(permute_425, [48, 4, 256, 513]);  permute_425 = None
        view_486 = torch.ops.aten.view.default(view_485, [4, 12, 1024, 513]);  view_485 = None
        permute_426 = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_83 = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
        amax_6 = torch.ops.aten.amax.default(clone_83, [-1], True)
        sub_52 = torch.ops.aten.sub.Tensor(clone_83, amax_6);  clone_83 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_67 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
        full_default_69 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_55 = torch.ops.aten.where.self(unsqueeze_130, full_default_69, div_67);  unsqueeze_130 = full_default_69 = div_67 = None
        view_487 = torch.ops.aten.view.default(add_92, [1024, 4, 12, 64]);  add_92 = None
        permute_427 = torch.ops.aten.permute.default(view_487, [1, 0, 2, 3]);  view_487 = None
        permute_428 = torch.ops.aten.permute.default(where_55, [0, 2, 1, 3]);  where_55 = None
        clone_85 = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
        view_488 = torch.ops.aten.view.default(clone_85, [48, 4, 256, 513]);  clone_85 = None
        permute_429 = torch.ops.aten.permute.default(permute_427, [0, 2, 1, 3]);  permute_427 = None
        view_489 = torch.ops.aten.view.default(permute_429, [48, 1024, 64]);  permute_429 = None
        constant_pad_nd_26 = torch.ops.aten.constant_pad_nd.default(view_489, [0, 0, 256, 256], -1.0);  view_489 = None
        as_strided_41 = torch.ops.aten.as_strided.default(constant_pad_nd_26, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_26 = None
        constant_pad_nd_27 = torch.ops.aten.constant_pad_nd.default(view_488, [0, 257], 0.0);  view_488 = None
        view_490 = torch.ops.aten.view.default(constant_pad_nd_27, [48, 4, -1]);  constant_pad_nd_27 = None
        slice_1613 = torch.ops.aten.slice.Tensor(view_490, 2, 0, -256);  view_490 = None
        view_491 = torch.ops.aten.view.default(slice_1613, [48, 4, 256, 769]);  slice_1613 = None
        slice_1617 = torch.ops.aten.slice.Tensor(view_491, 3, 0, -1);  view_491 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(slice_1617, 4);  slice_1617 = None
        permute_430 = torch.ops.aten.permute.default(unsqueeze_131, [0, 1, 2, 4, 3]);  unsqueeze_131 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
        permute_431 = torch.ops.aten.permute.default(unsqueeze_132, [0, 1, 4, 3, 2]);  unsqueeze_132 = None
        permute_432 = torch.ops.aten.permute.default(permute_430, [0, 1, 2, 4, 3]);  permute_430 = None
        view_492 = torch.ops.aten.view.default(permute_432, [192, 256, 768]);  permute_432 = None
        permute_433 = torch.ops.aten.permute.default(permute_431, [0, 1, 4, 3, 2]);  permute_431 = None
        clone_86 = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_493 = torch.ops.aten.view.default(clone_86, [192, 768, 64]);  clone_86 = None
        bmm_13 = torch.ops.aten.bmm.default(view_492, view_493);  view_492 = view_493 = None
        view_494 = torch.ops.aten.view.default(bmm_13, [48, 4, 256, 1, 64]);  bmm_13 = None
        permute_434 = torch.ops.aten.permute.default(view_494, [0, 1, 2, 4, 3]);  view_494 = None
        view_495 = torch.ops.aten.view.default(permute_434, [48, 4, 256, 64]);  permute_434 = None
        view_496 = torch.ops.aten.view.default(view_495, [4, 12, 1024, 64]);  view_495 = None
        permute_435 = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        permute_436 = torch.ops.aten.permute.default(permute_435, [1, 0, 2, 3]);  permute_435 = None
        clone_87 = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
        view_497 = torch.ops.aten.view.default(clone_87, [1024, 4, 768]);  clone_87 = None
        permute_437 = torch.ops.aten.permute.default(view_497, [1, 0, 2]);  view_497 = None
        permute_438 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        clone_88 = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
        view_498 = torch.ops.aten.view.default(clone_88, [4096, 768]);  clone_88 = None
        mm_27 = torch.ops.aten.mm.default(view_498, permute_438);  view_498 = permute_438 = None
        view_499 = torch.ops.aten.view.default(mm_27, [4, 1024, 768]);  mm_27 = None
        add_97 = torch.ops.aten.add.Tensor(view_499, arg106_1);  view_499 = arg106_1 = None
        add_98 = torch.ops.aten.add.Tensor(add_97, add_89);  add_97 = add_89 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_99 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_98, getitem_25);  add_98 = getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = rsqrt_12 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg107_1);  mul_49 = arg107_1 = None
        add_100 = torch.ops.aten.add.Tensor(mul_50, arg108_1);  mul_50 = arg108_1 = None
        view_500 = torch.ops.aten.view.default(add_100, [4096, 768])
        permute_439 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg110_1, view_500, permute_439);  arg110_1 = view_500 = permute_439 = None
        view_501 = torch.ops.aten.view.default(addmm_12, [4, 1024, 3072]);  addmm_12 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_52 = torch.ops.aten.mul.Tensor(view_501, 0.7071067811865476);  view_501 = None
        erf_6 = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_101 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_51, add_101);  mul_51 = add_101 = None
        view_502 = torch.ops.aten.view.default(mul_53, [4096, 3072]);  mul_53 = None
        permute_440 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg112_1, view_502, permute_440);  arg112_1 = view_502 = permute_440 = None
        view_503 = torch.ops.aten.view.default(addmm_13, [4, 1024, 768]);  addmm_13 = None
        add_102 = torch.ops.aten.add.Tensor(view_503, add_100);  view_503 = add_100 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_103 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_55 = torch.ops.aten.sub.Tensor(add_102, getitem_27);  add_102 = getitem_27 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = rsqrt_13 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, arg113_1);  mul_54 = arg113_1 = None
        add_104 = torch.ops.aten.add.Tensor(mul_55, arg114_1);  mul_55 = arg114_1 = None
        permute_441 = torch.ops.aten.permute.default(add_104, [1, 0, 2])
        permute_442 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        clone_91 = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format)
        view_504 = torch.ops.aten.view.default(clone_91, [4096, 768]);  clone_91 = None
        mm_28 = torch.ops.aten.mm.default(view_504, permute_442);  view_504 = permute_442 = None
        view_505 = torch.ops.aten.view.default(mm_28, [1024, 4, 768]);  mm_28 = None
        add_105 = torch.ops.aten.add.Tensor(view_505, arg116_1);  view_505 = arg116_1 = None
        permute_443 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        clone_92 = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format)
        view_506 = torch.ops.aten.view.default(clone_92, [4096, 768]);  clone_92 = None
        mm_29 = torch.ops.aten.mm.default(view_506, permute_443);  view_506 = permute_443 = None
        view_507 = torch.ops.aten.view.default(mm_29, [1024, 4, 768]);  mm_29 = None
        add_106 = torch.ops.aten.add.Tensor(view_507, arg118_1);  view_507 = arg118_1 = None
        permute_444 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        clone_93 = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format);  permute_441 = None
        view_508 = torch.ops.aten.view.default(clone_93, [4096, 768]);  clone_93 = None
        mm_30 = torch.ops.aten.mm.default(view_508, permute_444);  view_508 = permute_444 = None
        view_509 = torch.ops.aten.view.default(mm_30, [1024, 4, 768]);  mm_30 = None
        add_107 = torch.ops.aten.add.Tensor(view_509, arg120_1);  view_509 = arg120_1 = None
        div_70 = torch.ops.aten.div.Tensor(add_105, 8.0);  add_105 = None
        view_511 = torch.ops.aten.view.default(add_106, [1024, 4, 12, 64]);  add_106 = None
        permute_446 = torch.ops.aten.permute.default(view_511, [1, 0, 2, 3]);  view_511 = None
        permute_448 = torch.ops.aten.permute.default(permute_446, [0, 2, 1, 3]);  permute_446 = None
        view_513 = torch.ops.aten.view.default(permute_448, [48, 1024, 64]);  permute_448 = None
        view_515 = torch.ops.aten.view.default(view_513, [48, 2, 512, 64]);  view_513 = None
        as_strided_43 = torch.ops.aten.as_strided.default(view_515, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_515 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
        permute_450 = torch.ops.aten.permute.default(unsqueeze_134, [0, 1, 4, 2, 3]);  unsqueeze_134 = None
        view_516 = torch.ops.aten.view.default(div_70, [1024, 4, 12, 64]);  div_70 = None
        permute_452 = torch.ops.aten.permute.default(view_516, [1, 0, 2, 3]);  view_516 = None
        permute_453 = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
        view_517 = torch.ops.aten.view.default(permute_453, [48, 1024, 64]);  permute_453 = None
        view_518 = torch.ops.aten.view.default(view_517, [48, 2, 512, 64]);  view_517 = None
        as_strided_44 = torch.ops.aten.as_strided.default(view_518, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_518 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
        permute_454 = torch.ops.aten.permute.default(unsqueeze_135, [0, 1, 2, 4, 3]);  unsqueeze_135 = None
        permute_455 = torch.ops.aten.permute.default(permute_454, [0, 1, 2, 4, 3]);  permute_454 = None
        clone_94 = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_519 = torch.ops.aten.view.default(clone_94, [144, 512, 64]);  clone_94 = None
        permute_456 = torch.ops.aten.permute.default(permute_450, [0, 1, 4, 3, 2]);  permute_450 = None
        clone_95 = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
        view_520 = torch.ops.aten.view.default(clone_95, [144, 64, 512]);  clone_95 = None
        bmm_14 = torch.ops.aten.bmm.default(view_519, view_520);  view_519 = view_520 = None
        view_521 = torch.ops.aten.view.default(bmm_14, [48, 3, 512, 1, 512]);  bmm_14 = None
        permute_457 = torch.ops.aten.permute.default(view_521, [0, 1, 2, 4, 3]);  view_521 = None
        view_522 = torch.ops.aten.view.default(permute_457, [48, 3, 512, 512]);  permute_457 = None
        constant_pad_nd_28 = torch.ops.aten.constant_pad_nd.default(view_522, [0, 0, 0, 1], 0.0);  view_522 = None
        view_523 = torch.ops.aten.view.default(constant_pad_nd_28, [48, 3, 512, 513]);  constant_pad_nd_28 = None
        full_63 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1620 = torch.ops.aten.slice.Tensor(view_523, 2, 0, 256)
        slice_1621 = torch.ops.aten.slice.Tensor(slice_1620, 3, 0, 257);  slice_1620 = None
        slice_1623 = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_1625 = torch.ops.aten.slice.Tensor(slice_1623, 3, 256, 9223372036854775807);  slice_1623 = None
        copy_84 = torch.ops.aten.copy.default(slice_1625, slice_1621);  slice_1625 = slice_1621 = None
        slice_1627 = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_scatter_308 = torch.ops.aten.slice_scatter.default(slice_1627, copy_84, 3, 256, 9223372036854775807);  slice_1627 = copy_84 = None
        slice_scatter_310 = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_308, 1, 0, -1);  full_63 = slice_scatter_308 = None
        select_140 = torch.ops.aten.select.int(view_523, 1, -1)
        slice_1634 = torch.ops.aten.slice.Tensor(select_140, 1, 256, 9223372036854775807);  select_140 = None
        slice_1635 = torch.ops.aten.slice.Tensor(slice_1634, 2, 0, 257);  slice_1634 = None
        select_142 = torch.ops.aten.select.int(slice_scatter_310, 1, -1)
        slice_1641 = torch.ops.aten.slice.Tensor(select_142, 2, 256, 9223372036854775807);  select_142 = None
        copy_85 = torch.ops.aten.copy.default(slice_1641, slice_1635);  slice_1641 = slice_1635 = None
        select_143 = torch.ops.aten.select.int(slice_scatter_310, 1, -1)
        slice_scatter_312 = torch.ops.aten.slice_scatter.default(select_143, copy_85, 2, 256, 9223372036854775807);  select_143 = copy_85 = None
        select_scatter_28 = torch.ops.aten.select_scatter.default(slice_scatter_310, slice_scatter_312, 1, -1);  slice_scatter_310 = slice_scatter_312 = None
        slice_1649 = torch.ops.aten.slice.Tensor(view_523, 2, -257, -1)
        slice_1650 = torch.ops.aten.slice.Tensor(slice_1649, 3, 257, 9223372036854775807);  slice_1649 = None
        slice_1656 = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_1658 = torch.ops.aten.slice.Tensor(slice_1656, 3, 0, 256);  slice_1656 = None
        copy_86 = torch.ops.aten.copy.default(slice_1658, slice_1650);  slice_1658 = slice_1650 = None
        slice_1660 = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_scatter_315 = torch.ops.aten.slice_scatter.default(slice_1660, copy_86, 3, 0, 256);  slice_1660 = copy_86 = None
        slice_scatter_317 = torch.ops.aten.slice_scatter.default(select_scatter_28, slice_scatter_315, 1, 1, 9223372036854775807);  select_scatter_28 = slice_scatter_315 = None
        select_145 = torch.ops.aten.select.int(view_523, 1, 0);  view_523 = None
        slice_1667 = torch.ops.aten.slice.Tensor(select_145, 1, 0, 255);  select_145 = None
        slice_1668 = torch.ops.aten.slice.Tensor(slice_1667, 2, -255, 9223372036854775807);  slice_1667 = None
        select_147 = torch.ops.aten.select.int(slice_scatter_317, 1, 0)
        slice_1673 = torch.ops.aten.slice.Tensor(select_147, 1, 1, 256);  select_147 = None
        slice_1674 = torch.ops.aten.slice.Tensor(slice_1673, 2, 1, 256);  slice_1673 = None
        copy_87 = torch.ops.aten.copy.default(slice_1674, slice_1668);  slice_1674 = slice_1668 = None
        select_148 = torch.ops.aten.select.int(slice_scatter_317, 1, 0)
        slice_1676 = torch.ops.aten.slice.Tensor(select_148, 1, 1, 256)
        slice_scatter_319 = torch.ops.aten.slice_scatter.default(slice_1676, copy_87, 2, 1, 256);  slice_1676 = copy_87 = None
        slice_scatter_320 = torch.ops.aten.slice_scatter.default(select_148, slice_scatter_319, 1, 1, 256);  select_148 = slice_scatter_319 = None
        select_scatter_29 = torch.ops.aten.select_scatter.default(slice_scatter_317, slice_scatter_320, 1, 0);  slice_scatter_317 = slice_scatter_320 = None
        full_default_70 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_28 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(iota_28, -2);  iota_28 = None
        iota_29 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        sub_57 = torch.ops.aten.sub.Tensor(unsqueeze_136, unsqueeze_137);  unsqueeze_136 = unsqueeze_137 = None
        le_14 = torch.ops.aten.le.Scalar(sub_57, 0);  sub_57 = None
        full_default_71 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_56 = torch.ops.aten.where.self(le_14, full_default_70, full_default_71);  le_14 = full_default_70 = full_default_71 = None
        rev_28 = torch.ops.prims.rev.default(where_56, [0]);  where_56 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(rev_28, 0);  rev_28 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
        rev_29 = torch.ops.prims.rev.default(unsqueeze_139, [1, 3])
        expand_28 = torch.ops.aten.expand.default(unsqueeze_139, [4, 256, 12, 257]);  unsqueeze_139 = None
        view_526 = torch.ops.aten.view.default(select_scatter_29, [4, 12, 1024, 513])
        permute_460 = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        slice_1687 = torch.ops.aten.slice.Tensor(permute_460, 1, 0, 256);  permute_460 = None
        slice_1689 = torch.ops.aten.slice.Tensor(slice_1687, 3, 0, 257);  slice_1687 = None
        full_default_72 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(expand_28, torch.bool);  expand_28 = None
        where_57 = torch.ops.aten.where.self(convert_element_type_35, full_default_72, slice_1689);  convert_element_type_35 = full_default_72 = slice_1689 = None
        view_527 = torch.ops.aten.view.default(select_scatter_29, [4, 12, 1024, 513])
        permute_461 = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
        slice_1695 = torch.ops.aten.slice.Tensor(permute_461, 1, 0, 256);  permute_461 = None
        slice_1697 = torch.ops.aten.slice.Tensor(slice_1695, 3, 0, 257);  slice_1695 = None
        copy_88 = torch.ops.aten.copy.default(slice_1697, where_57);  slice_1697 = where_57 = None
        view_528 = torch.ops.aten.view.default(select_scatter_29, [4, 12, 1024, 513]);  select_scatter_29 = None
        permute_462 = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        slice_1699 = torch.ops.aten.slice.Tensor(permute_462, 1, 0, 256)
        slice_scatter_322 = torch.ops.aten.slice_scatter.default(slice_1699, copy_88, 3, 0, 257);  slice_1699 = copy_88 = None
        slice_scatter_324 = torch.ops.aten.slice_scatter.default(permute_462, slice_scatter_322, 1, 0, 256);  permute_462 = slice_scatter_322 = None
        permute_463 = torch.ops.aten.permute.default(slice_scatter_324, [0, 2, 1, 3]);  slice_scatter_324 = None
        view_529 = torch.ops.aten.view.default(permute_463, [48, 4, 256, 513]);  permute_463 = None
        expand_29 = torch.ops.aten.expand.default(rev_29, [4, 256, 12, 257]);  rev_29 = None
        view_531 = torch.ops.aten.view.default(view_529, [4, 12, 1024, 513])
        permute_465 = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
        slice_1710 = torch.ops.aten.slice.Tensor(permute_465, 1, -256, 9223372036854775807);  permute_465 = None
        slice_1712 = torch.ops.aten.slice.Tensor(slice_1710, 3, -257, 9223372036854775807);  slice_1710 = None
        full_default_73 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(expand_29, torch.bool);  expand_29 = None
        where_58 = torch.ops.aten.where.self(convert_element_type_36, full_default_73, slice_1712);  convert_element_type_36 = full_default_73 = slice_1712 = None
        view_532 = torch.ops.aten.view.default(view_529, [4, 12, 1024, 513])
        permute_466 = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        slice_1718 = torch.ops.aten.slice.Tensor(permute_466, 1, -256, 9223372036854775807);  permute_466 = None
        slice_1720 = torch.ops.aten.slice.Tensor(slice_1718, 3, -257, 9223372036854775807);  slice_1718 = None
        copy_89 = torch.ops.aten.copy.default(slice_1720, where_58);  slice_1720 = where_58 = None
        view_533 = torch.ops.aten.view.default(view_529, [4, 12, 1024, 513]);  view_529 = None
        permute_467 = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
        slice_1722 = torch.ops.aten.slice.Tensor(permute_467, 1, -256, 9223372036854775807)
        slice_scatter_326 = torch.ops.aten.slice_scatter.default(slice_1722, copy_89, 3, -257, 9223372036854775807);  slice_1722 = copy_89 = None
        slice_scatter_328 = torch.ops.aten.slice_scatter.default(permute_467, slice_scatter_326, 1, -256, 9223372036854775807);  permute_467 = slice_scatter_326 = None
        permute_468 = torch.ops.aten.permute.default(slice_scatter_328, [0, 2, 1, 3]);  slice_scatter_328 = None
        view_534 = torch.ops.aten.view.default(permute_468, [48, 4, 256, 513]);  permute_468 = None
        ne_7 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(ne_7, 2);  ne_7 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(unsqueeze_141, torch.float32)
        full_default_74 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_59 = torch.ops.aten.where.self(unsqueeze_141, full_default_74, convert_element_type_37);  unsqueeze_141 = full_default_74 = convert_element_type_37 = None
        full_67 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_470 = torch.ops.aten.permute.default(full_67, [0, 2, 1, 3]);  full_67 = None
        view_536 = torch.ops.aten.view.default(permute_470, [4, 1024, 1]);  permute_470 = None
        permute_471 = torch.ops.aten.permute.default(where_59, [0, 2, 1, 3]);  where_59 = None
        view_537 = torch.ops.aten.view.default(permute_471, [4, 1024, 1]);  permute_471 = None
        view_538 = torch.ops.aten.view.default(view_536, [4, 2, 512, 1]);  view_536 = None
        as_strided_45 = torch.ops.aten.as_strided.default(view_538, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_538 = None
        view_539 = torch.ops.aten.view.default(view_537, [4, 2, 512, 1]);  view_537 = None
        as_strided_46 = torch.ops.aten.as_strided.default(view_539, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_539 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(as_strided_45, 4);  as_strided_45 = None
        permute_472 = torch.ops.aten.permute.default(unsqueeze_142, [0, 1, 2, 4, 3]);  unsqueeze_142 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
        permute_473 = torch.ops.aten.permute.default(unsqueeze_143, [0, 1, 4, 2, 3]);  unsqueeze_143 = None
        mul_56 = torch.ops.aten.mul.Tensor(permute_472, permute_473);  permute_472 = permute_473 = None
        view_540 = torch.ops.aten.view.default(mul_56, [4, 3, 512, 512]);  mul_56 = None
        constant_pad_nd_29 = torch.ops.aten.constant_pad_nd.default(view_540, [0, 0, 0, 1], 0.0);  view_540 = None
        view_541 = torch.ops.aten.view.default(constant_pad_nd_29, [4, 3, 512, 513]);  constant_pad_nd_29 = None
        full_68 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1732 = torch.ops.aten.slice.Tensor(view_541, 2, 0, 256)
        slice_1733 = torch.ops.aten.slice.Tensor(slice_1732, 3, 0, 257);  slice_1732 = None
        slice_1735 = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_1737 = torch.ops.aten.slice.Tensor(slice_1735, 3, 256, 9223372036854775807);  slice_1735 = None
        copy_90 = torch.ops.aten.copy.default(slice_1737, slice_1733);  slice_1737 = slice_1733 = None
        slice_1739 = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_scatter_330 = torch.ops.aten.slice_scatter.default(slice_1739, copy_90, 3, 256, 9223372036854775807);  slice_1739 = copy_90 = None
        slice_scatter_332 = torch.ops.aten.slice_scatter.default(full_68, slice_scatter_330, 1, 0, -1);  full_68 = slice_scatter_330 = None
        select_150 = torch.ops.aten.select.int(view_541, 1, -1)
        slice_1746 = torch.ops.aten.slice.Tensor(select_150, 1, 256, 9223372036854775807);  select_150 = None
        slice_1747 = torch.ops.aten.slice.Tensor(slice_1746, 2, 0, 257);  slice_1746 = None
        select_152 = torch.ops.aten.select.int(slice_scatter_332, 1, -1)
        slice_1753 = torch.ops.aten.slice.Tensor(select_152, 2, 256, 9223372036854775807);  select_152 = None
        copy_91 = torch.ops.aten.copy.default(slice_1753, slice_1747);  slice_1753 = slice_1747 = None
        select_153 = torch.ops.aten.select.int(slice_scatter_332, 1, -1)
        slice_scatter_334 = torch.ops.aten.slice_scatter.default(select_153, copy_91, 2, 256, 9223372036854775807);  select_153 = copy_91 = None
        select_scatter_30 = torch.ops.aten.select_scatter.default(slice_scatter_332, slice_scatter_334, 1, -1);  slice_scatter_332 = slice_scatter_334 = None
        slice_1761 = torch.ops.aten.slice.Tensor(view_541, 2, -257, -1)
        slice_1762 = torch.ops.aten.slice.Tensor(slice_1761, 3, 257, 9223372036854775807);  slice_1761 = None
        slice_1768 = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_1770 = torch.ops.aten.slice.Tensor(slice_1768, 3, 0, 256);  slice_1768 = None
        copy_92 = torch.ops.aten.copy.default(slice_1770, slice_1762);  slice_1770 = slice_1762 = None
        slice_1772 = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_scatter_337 = torch.ops.aten.slice_scatter.default(slice_1772, copy_92, 3, 0, 256);  slice_1772 = copy_92 = None
        slice_scatter_339 = torch.ops.aten.slice_scatter.default(select_scatter_30, slice_scatter_337, 1, 1, 9223372036854775807);  select_scatter_30 = slice_scatter_337 = None
        select_155 = torch.ops.aten.select.int(view_541, 1, 0);  view_541 = None
        slice_1779 = torch.ops.aten.slice.Tensor(select_155, 1, 0, 255);  select_155 = None
        slice_1780 = torch.ops.aten.slice.Tensor(slice_1779, 2, -255, 9223372036854775807);  slice_1779 = None
        select_157 = torch.ops.aten.select.int(slice_scatter_339, 1, 0)
        slice_1785 = torch.ops.aten.slice.Tensor(select_157, 1, 1, 256);  select_157 = None
        slice_1786 = torch.ops.aten.slice.Tensor(slice_1785, 2, 1, 256);  slice_1785 = None
        copy_93 = torch.ops.aten.copy.default(slice_1786, slice_1780);  slice_1786 = slice_1780 = None
        select_158 = torch.ops.aten.select.int(slice_scatter_339, 1, 0)
        slice_1788 = torch.ops.aten.slice.Tensor(select_158, 1, 1, 256)
        slice_scatter_341 = torch.ops.aten.slice_scatter.default(slice_1788, copy_93, 2, 1, 256);  slice_1788 = copy_93 = None
        slice_scatter_342 = torch.ops.aten.slice_scatter.default(select_158, slice_scatter_341, 1, 1, 256);  select_158 = slice_scatter_341 = None
        select_scatter_31 = torch.ops.aten.select_scatter.default(slice_scatter_339, slice_scatter_342, 1, 0);  slice_scatter_339 = slice_scatter_342 = None
        full_default_75 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_30 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(iota_30, -2);  iota_30 = None
        iota_31 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
        sub_59 = torch.ops.aten.sub.Tensor(unsqueeze_144, unsqueeze_145);  unsqueeze_144 = unsqueeze_145 = None
        le_15 = torch.ops.aten.le.Scalar(sub_59, 0);  sub_59 = None
        full_default_76 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_60 = torch.ops.aten.where.self(le_15, full_default_75, full_default_76);  le_15 = full_default_75 = full_default_76 = None
        rev_30 = torch.ops.prims.rev.default(where_60, [0]);  where_60 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(rev_30, 0);  rev_30 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        rev_31 = torch.ops.prims.rev.default(unsqueeze_147, [1, 3])
        expand_30 = torch.ops.aten.expand.default(unsqueeze_147, [4, 256, 1, 257]);  unsqueeze_147 = None
        view_544 = torch.ops.aten.view.default(select_scatter_31, [4, 1, 1024, 513])
        permute_476 = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        slice_1799 = torch.ops.aten.slice.Tensor(permute_476, 1, 0, 256);  permute_476 = None
        slice_1801 = torch.ops.aten.slice.Tensor(slice_1799, 3, 0, 257);  slice_1799 = None
        full_default_77 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(expand_30, torch.bool);  expand_30 = None
        where_61 = torch.ops.aten.where.self(convert_element_type_38, full_default_77, slice_1801);  convert_element_type_38 = full_default_77 = slice_1801 = None
        view_545 = torch.ops.aten.view.default(select_scatter_31, [4, 1, 1024, 513])
        permute_477 = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
        slice_1807 = torch.ops.aten.slice.Tensor(permute_477, 1, 0, 256);  permute_477 = None
        slice_1809 = torch.ops.aten.slice.Tensor(slice_1807, 3, 0, 257);  slice_1807 = None
        copy_94 = torch.ops.aten.copy.default(slice_1809, where_61);  slice_1809 = where_61 = None
        view_546 = torch.ops.aten.view.default(select_scatter_31, [4, 1, 1024, 513]);  select_scatter_31 = None
        permute_478 = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        slice_1811 = torch.ops.aten.slice.Tensor(permute_478, 1, 0, 256)
        slice_scatter_344 = torch.ops.aten.slice_scatter.default(slice_1811, copy_94, 3, 0, 257);  slice_1811 = copy_94 = None
        slice_scatter_346 = torch.ops.aten.slice_scatter.default(permute_478, slice_scatter_344, 1, 0, 256);  permute_478 = slice_scatter_344 = None
        permute_479 = torch.ops.aten.permute.default(slice_scatter_346, [0, 2, 1, 3]);  slice_scatter_346 = None
        view_547 = torch.ops.aten.view.default(permute_479, [4, 4, 256, 513]);  permute_479 = None
        expand_31 = torch.ops.aten.expand.default(rev_31, [4, 256, 1, 257]);  rev_31 = None
        view_549 = torch.ops.aten.view.default(view_547, [4, 1, 1024, 513])
        permute_481 = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None
        slice_1822 = torch.ops.aten.slice.Tensor(permute_481, 1, -256, 9223372036854775807);  permute_481 = None
        slice_1824 = torch.ops.aten.slice.Tensor(slice_1822, 3, -257, 9223372036854775807);  slice_1822 = None
        full_default_78 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(expand_31, torch.bool);  expand_31 = None
        where_62 = torch.ops.aten.where.self(convert_element_type_39, full_default_78, slice_1824);  convert_element_type_39 = full_default_78 = slice_1824 = None
        view_550 = torch.ops.aten.view.default(view_547, [4, 1, 1024, 513])
        permute_482 = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
        slice_1830 = torch.ops.aten.slice.Tensor(permute_482, 1, -256, 9223372036854775807);  permute_482 = None
        slice_1832 = torch.ops.aten.slice.Tensor(slice_1830, 3, -257, 9223372036854775807);  slice_1830 = None
        copy_95 = torch.ops.aten.copy.default(slice_1832, where_62);  slice_1832 = where_62 = None
        view_551 = torch.ops.aten.view.default(view_547, [4, 1, 1024, 513]);  view_547 = None
        permute_483 = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
        slice_1834 = torch.ops.aten.slice.Tensor(permute_483, 1, -256, 9223372036854775807)
        slice_scatter_348 = torch.ops.aten.slice_scatter.default(slice_1834, copy_95, 3, -257, 9223372036854775807);  slice_1834 = copy_95 = None
        slice_scatter_350 = torch.ops.aten.slice_scatter.default(permute_483, slice_scatter_348, 1, -256, 9223372036854775807);  permute_483 = slice_scatter_348 = None
        permute_484 = torch.ops.aten.permute.default(slice_scatter_350, [0, 2, 1, 3]);  slice_scatter_350 = None
        view_552 = torch.ops.aten.view.default(permute_484, [4, 4, 256, 513]);  permute_484 = None
        view_554 = torch.ops.aten.view.default(view_534, [4, 12, 1024, 513]);  view_534 = None
        permute_486 = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        view_555 = torch.ops.aten.view.default(view_552, [4, 1, 1024, 513]);  view_552 = None
        permute_487 = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
        add_110 = torch.ops.aten.add.Tensor(permute_486, permute_487);  permute_486 = permute_487 = None
        permute_488 = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
        view_557 = torch.ops.aten.view.default(permute_488, [48, 4, 256, 513]);  permute_488 = None
        view_558 = torch.ops.aten.view.default(view_557, [4, 12, 1024, 513]);  view_557 = None
        permute_489 = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
        clone_96 = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
        amax_7 = torch.ops.aten.amax.default(clone_96, [-1], True)
        sub_60 = torch.ops.aten.sub.Tensor(clone_96, amax_7);  clone_96 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_77 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
        full_default_79 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_63 = torch.ops.aten.where.self(unsqueeze_149, full_default_79, div_77);  unsqueeze_149 = full_default_79 = div_77 = None
        view_559 = torch.ops.aten.view.default(add_107, [1024, 4, 12, 64]);  add_107 = None
        permute_490 = torch.ops.aten.permute.default(view_559, [1, 0, 2, 3]);  view_559 = None
        permute_491 = torch.ops.aten.permute.default(where_63, [0, 2, 1, 3]);  where_63 = None
        clone_98 = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_560 = torch.ops.aten.view.default(clone_98, [48, 4, 256, 513]);  clone_98 = None
        permute_492 = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
        view_561 = torch.ops.aten.view.default(permute_492, [48, 1024, 64]);  permute_492 = None
        constant_pad_nd_30 = torch.ops.aten.constant_pad_nd.default(view_561, [0, 0, 256, 256], -1.0);  view_561 = None
        as_strided_47 = torch.ops.aten.as_strided.default(constant_pad_nd_30, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_30 = None
        constant_pad_nd_31 = torch.ops.aten.constant_pad_nd.default(view_560, [0, 257], 0.0);  view_560 = None
        view_562 = torch.ops.aten.view.default(constant_pad_nd_31, [48, 4, -1]);  constant_pad_nd_31 = None
        slice_1844 = torch.ops.aten.slice.Tensor(view_562, 2, 0, -256);  view_562 = None
        view_563 = torch.ops.aten.view.default(slice_1844, [48, 4, 256, 769]);  slice_1844 = None
        slice_1848 = torch.ops.aten.slice.Tensor(view_563, 3, 0, -1);  view_563 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(slice_1848, 4);  slice_1848 = None
        permute_493 = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 2, 4, 3]);  unsqueeze_150 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(as_strided_47, 4);  as_strided_47 = None
        permute_494 = torch.ops.aten.permute.default(unsqueeze_151, [0, 1, 4, 3, 2]);  unsqueeze_151 = None
        permute_495 = torch.ops.aten.permute.default(permute_493, [0, 1, 2, 4, 3]);  permute_493 = None
        view_564 = torch.ops.aten.view.default(permute_495, [192, 256, 768]);  permute_495 = None
        permute_496 = torch.ops.aten.permute.default(permute_494, [0, 1, 4, 3, 2]);  permute_494 = None
        clone_99 = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
        view_565 = torch.ops.aten.view.default(clone_99, [192, 768, 64]);  clone_99 = None
        bmm_15 = torch.ops.aten.bmm.default(view_564, view_565);  view_564 = view_565 = None
        view_566 = torch.ops.aten.view.default(bmm_15, [48, 4, 256, 1, 64]);  bmm_15 = None
        permute_497 = torch.ops.aten.permute.default(view_566, [0, 1, 2, 4, 3]);  view_566 = None
        view_567 = torch.ops.aten.view.default(permute_497, [48, 4, 256, 64]);  permute_497 = None
        view_568 = torch.ops.aten.view.default(view_567, [4, 12, 1024, 64]);  view_567 = None
        permute_498 = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        permute_499 = torch.ops.aten.permute.default(permute_498, [1, 0, 2, 3]);  permute_498 = None
        clone_100 = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
        view_569 = torch.ops.aten.view.default(clone_100, [1024, 4, 768]);  clone_100 = None
        permute_500 = torch.ops.aten.permute.default(view_569, [1, 0, 2]);  view_569 = None
        permute_501 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        clone_101 = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_570 = torch.ops.aten.view.default(clone_101, [4096, 768]);  clone_101 = None
        mm_31 = torch.ops.aten.mm.default(view_570, permute_501);  view_570 = permute_501 = None
        view_571 = torch.ops.aten.view.default(mm_31, [4, 1024, 768]);  mm_31 = None
        add_112 = torch.ops.aten.add.Tensor(view_571, arg122_1);  view_571 = arg122_1 = None
        add_113 = torch.ops.aten.add.Tensor(add_112, add_104);  add_112 = add_104 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_113, getitem_29);  add_113 = getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = rsqrt_14 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg123_1);  mul_57 = arg123_1 = None
        add_115 = torch.ops.aten.add.Tensor(mul_58, arg124_1);  mul_58 = arg124_1 = None
        view_572 = torch.ops.aten.view.default(add_115, [4096, 768])
        permute_502 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg126_1, view_572, permute_502);  arg126_1 = view_572 = permute_502 = None
        view_573 = torch.ops.aten.view.default(addmm_14, [4, 1024, 3072]);  addmm_14 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_573, 0.5)
        mul_60 = torch.ops.aten.mul.Tensor(view_573, 0.7071067811865476);  view_573 = None
        erf_7 = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_116 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_59, add_116);  mul_59 = add_116 = None
        view_574 = torch.ops.aten.view.default(mul_61, [4096, 3072]);  mul_61 = None
        permute_503 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg128_1, view_574, permute_503);  arg128_1 = view_574 = permute_503 = None
        view_575 = torch.ops.aten.view.default(addmm_15, [4, 1024, 768]);  addmm_15 = None
        add_117 = torch.ops.aten.add.Tensor(view_575, add_115);  view_575 = add_115 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_63 = torch.ops.aten.sub.Tensor(add_117, getitem_31);  add_117 = getitem_31 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = rsqrt_15 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_62, arg129_1);  mul_62 = arg129_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_63, arg130_1);  mul_63 = arg130_1 = None
        permute_504 = torch.ops.aten.permute.default(add_119, [1, 0, 2])
        permute_505 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        clone_104 = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format)
        view_576 = torch.ops.aten.view.default(clone_104, [4096, 768]);  clone_104 = None
        mm_32 = torch.ops.aten.mm.default(view_576, permute_505);  view_576 = permute_505 = None
        view_577 = torch.ops.aten.view.default(mm_32, [1024, 4, 768]);  mm_32 = None
        add_120 = torch.ops.aten.add.Tensor(view_577, arg132_1);  view_577 = arg132_1 = None
        permute_506 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        clone_105 = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format)
        view_578 = torch.ops.aten.view.default(clone_105, [4096, 768]);  clone_105 = None
        mm_33 = torch.ops.aten.mm.default(view_578, permute_506);  view_578 = permute_506 = None
        view_579 = torch.ops.aten.view.default(mm_33, [1024, 4, 768]);  mm_33 = None
        add_121 = torch.ops.aten.add.Tensor(view_579, arg134_1);  view_579 = arg134_1 = None
        permute_507 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        clone_106 = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
        view_580 = torch.ops.aten.view.default(clone_106, [4096, 768]);  clone_106 = None
        mm_34 = torch.ops.aten.mm.default(view_580, permute_507);  view_580 = permute_507 = None
        view_581 = torch.ops.aten.view.default(mm_34, [1024, 4, 768]);  mm_34 = None
        add_122 = torch.ops.aten.add.Tensor(view_581, arg136_1);  view_581 = arg136_1 = None
        div_80 = torch.ops.aten.div.Tensor(add_120, 8.0);  add_120 = None
        view_583 = torch.ops.aten.view.default(add_121, [1024, 4, 12, 64]);  add_121 = None
        permute_509 = torch.ops.aten.permute.default(view_583, [1, 0, 2, 3]);  view_583 = None
        permute_511 = torch.ops.aten.permute.default(permute_509, [0, 2, 1, 3]);  permute_509 = None
        view_585 = torch.ops.aten.view.default(permute_511, [48, 1024, 64]);  permute_511 = None
        view_587 = torch.ops.aten.view.default(view_585, [48, 2, 512, 64]);  view_585 = None
        as_strided_49 = torch.ops.aten.as_strided.default(view_587, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_587 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(as_strided_49, 4);  as_strided_49 = None
        permute_513 = torch.ops.aten.permute.default(unsqueeze_153, [0, 1, 4, 2, 3]);  unsqueeze_153 = None
        view_588 = torch.ops.aten.view.default(div_80, [1024, 4, 12, 64]);  div_80 = None
        permute_515 = torch.ops.aten.permute.default(view_588, [1, 0, 2, 3]);  view_588 = None
        permute_516 = torch.ops.aten.permute.default(permute_515, [0, 2, 1, 3]);  permute_515 = None
        view_589 = torch.ops.aten.view.default(permute_516, [48, 1024, 64]);  permute_516 = None
        view_590 = torch.ops.aten.view.default(view_589, [48, 2, 512, 64]);  view_589 = None
        as_strided_50 = torch.ops.aten.as_strided.default(view_590, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_590 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
        permute_517 = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 2, 4, 3]);  unsqueeze_154 = None
        permute_518 = torch.ops.aten.permute.default(permute_517, [0, 1, 2, 4, 3]);  permute_517 = None
        clone_107 = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_591 = torch.ops.aten.view.default(clone_107, [144, 512, 64]);  clone_107 = None
        permute_519 = torch.ops.aten.permute.default(permute_513, [0, 1, 4, 3, 2]);  permute_513 = None
        clone_108 = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
        view_592 = torch.ops.aten.view.default(clone_108, [144, 64, 512]);  clone_108 = None
        bmm_16 = torch.ops.aten.bmm.default(view_591, view_592);  view_591 = view_592 = None
        view_593 = torch.ops.aten.view.default(bmm_16, [48, 3, 512, 1, 512]);  bmm_16 = None
        permute_520 = torch.ops.aten.permute.default(view_593, [0, 1, 2, 4, 3]);  view_593 = None
        view_594 = torch.ops.aten.view.default(permute_520, [48, 3, 512, 512]);  permute_520 = None
        constant_pad_nd_32 = torch.ops.aten.constant_pad_nd.default(view_594, [0, 0, 0, 1], 0.0);  view_594 = None
        view_595 = torch.ops.aten.view.default(constant_pad_nd_32, [48, 3, 512, 513]);  constant_pad_nd_32 = None
        full_72 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1851 = torch.ops.aten.slice.Tensor(view_595, 2, 0, 256)
        slice_1852 = torch.ops.aten.slice.Tensor(slice_1851, 3, 0, 257);  slice_1851 = None
        slice_1854 = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_1856 = torch.ops.aten.slice.Tensor(slice_1854, 3, 256, 9223372036854775807);  slice_1854 = None
        copy_96 = torch.ops.aten.copy.default(slice_1856, slice_1852);  slice_1856 = slice_1852 = None
        slice_1858 = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_scatter_352 = torch.ops.aten.slice_scatter.default(slice_1858, copy_96, 3, 256, 9223372036854775807);  slice_1858 = copy_96 = None
        slice_scatter_354 = torch.ops.aten.slice_scatter.default(full_72, slice_scatter_352, 1, 0, -1);  full_72 = slice_scatter_352 = None
        select_160 = torch.ops.aten.select.int(view_595, 1, -1)
        slice_1865 = torch.ops.aten.slice.Tensor(select_160, 1, 256, 9223372036854775807);  select_160 = None
        slice_1866 = torch.ops.aten.slice.Tensor(slice_1865, 2, 0, 257);  slice_1865 = None
        select_162 = torch.ops.aten.select.int(slice_scatter_354, 1, -1)
        slice_1872 = torch.ops.aten.slice.Tensor(select_162, 2, 256, 9223372036854775807);  select_162 = None
        copy_97 = torch.ops.aten.copy.default(slice_1872, slice_1866);  slice_1872 = slice_1866 = None
        select_163 = torch.ops.aten.select.int(slice_scatter_354, 1, -1)
        slice_scatter_356 = torch.ops.aten.slice_scatter.default(select_163, copy_97, 2, 256, 9223372036854775807);  select_163 = copy_97 = None
        select_scatter_32 = torch.ops.aten.select_scatter.default(slice_scatter_354, slice_scatter_356, 1, -1);  slice_scatter_354 = slice_scatter_356 = None
        slice_1880 = torch.ops.aten.slice.Tensor(view_595, 2, -257, -1)
        slice_1881 = torch.ops.aten.slice.Tensor(slice_1880, 3, 257, 9223372036854775807);  slice_1880 = None
        slice_1887 = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_1889 = torch.ops.aten.slice.Tensor(slice_1887, 3, 0, 256);  slice_1887 = None
        copy_98 = torch.ops.aten.copy.default(slice_1889, slice_1881);  slice_1889 = slice_1881 = None
        slice_1891 = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_scatter_359 = torch.ops.aten.slice_scatter.default(slice_1891, copy_98, 3, 0, 256);  slice_1891 = copy_98 = None
        slice_scatter_361 = torch.ops.aten.slice_scatter.default(select_scatter_32, slice_scatter_359, 1, 1, 9223372036854775807);  select_scatter_32 = slice_scatter_359 = None
        select_165 = torch.ops.aten.select.int(view_595, 1, 0);  view_595 = None
        slice_1898 = torch.ops.aten.slice.Tensor(select_165, 1, 0, 255);  select_165 = None
        slice_1899 = torch.ops.aten.slice.Tensor(slice_1898, 2, -255, 9223372036854775807);  slice_1898 = None
        select_167 = torch.ops.aten.select.int(slice_scatter_361, 1, 0)
        slice_1904 = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256);  select_167 = None
        slice_1905 = torch.ops.aten.slice.Tensor(slice_1904, 2, 1, 256);  slice_1904 = None
        copy_99 = torch.ops.aten.copy.default(slice_1905, slice_1899);  slice_1905 = slice_1899 = None
        select_168 = torch.ops.aten.select.int(slice_scatter_361, 1, 0)
        slice_1907 = torch.ops.aten.slice.Tensor(select_168, 1, 1, 256)
        slice_scatter_363 = torch.ops.aten.slice_scatter.default(slice_1907, copy_99, 2, 1, 256);  slice_1907 = copy_99 = None
        slice_scatter_364 = torch.ops.aten.slice_scatter.default(select_168, slice_scatter_363, 1, 1, 256);  select_168 = slice_scatter_363 = None
        select_scatter_33 = torch.ops.aten.select_scatter.default(slice_scatter_361, slice_scatter_364, 1, 0);  slice_scatter_361 = slice_scatter_364 = None
        full_default_80 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_32 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(iota_32, -2);  iota_32 = None
        iota_33 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        sub_65 = torch.ops.aten.sub.Tensor(unsqueeze_155, unsqueeze_156);  unsqueeze_155 = unsqueeze_156 = None
        le_16 = torch.ops.aten.le.Scalar(sub_65, 0);  sub_65 = None
        full_default_81 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_64 = torch.ops.aten.where.self(le_16, full_default_80, full_default_81);  le_16 = full_default_80 = full_default_81 = None
        rev_32 = torch.ops.prims.rev.default(where_64, [0]);  where_64 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(rev_32, 0);  rev_32 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        rev_33 = torch.ops.prims.rev.default(unsqueeze_158, [1, 3])
        expand_32 = torch.ops.aten.expand.default(unsqueeze_158, [4, 256, 12, 257]);  unsqueeze_158 = None
        view_598 = torch.ops.aten.view.default(select_scatter_33, [4, 12, 1024, 513])
        permute_523 = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        slice_1918 = torch.ops.aten.slice.Tensor(permute_523, 1, 0, 256);  permute_523 = None
        slice_1920 = torch.ops.aten.slice.Tensor(slice_1918, 3, 0, 257);  slice_1918 = None
        full_default_82 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(expand_32, torch.bool);  expand_32 = None
        where_65 = torch.ops.aten.where.self(convert_element_type_40, full_default_82, slice_1920);  convert_element_type_40 = full_default_82 = slice_1920 = None
        view_599 = torch.ops.aten.view.default(select_scatter_33, [4, 12, 1024, 513])
        permute_524 = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        slice_1926 = torch.ops.aten.slice.Tensor(permute_524, 1, 0, 256);  permute_524 = None
        slice_1928 = torch.ops.aten.slice.Tensor(slice_1926, 3, 0, 257);  slice_1926 = None
        copy_100 = torch.ops.aten.copy.default(slice_1928, where_65);  slice_1928 = where_65 = None
        view_600 = torch.ops.aten.view.default(select_scatter_33, [4, 12, 1024, 513]);  select_scatter_33 = None
        permute_525 = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
        slice_1930 = torch.ops.aten.slice.Tensor(permute_525, 1, 0, 256)
        slice_scatter_366 = torch.ops.aten.slice_scatter.default(slice_1930, copy_100, 3, 0, 257);  slice_1930 = copy_100 = None
        slice_scatter_368 = torch.ops.aten.slice_scatter.default(permute_525, slice_scatter_366, 1, 0, 256);  permute_525 = slice_scatter_366 = None
        permute_526 = torch.ops.aten.permute.default(slice_scatter_368, [0, 2, 1, 3]);  slice_scatter_368 = None
        view_601 = torch.ops.aten.view.default(permute_526, [48, 4, 256, 513]);  permute_526 = None
        expand_33 = torch.ops.aten.expand.default(rev_33, [4, 256, 12, 257]);  rev_33 = None
        view_603 = torch.ops.aten.view.default(view_601, [4, 12, 1024, 513])
        permute_528 = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
        slice_1941 = torch.ops.aten.slice.Tensor(permute_528, 1, -256, 9223372036854775807);  permute_528 = None
        slice_1943 = torch.ops.aten.slice.Tensor(slice_1941, 3, -257, 9223372036854775807);  slice_1941 = None
        full_default_83 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(expand_33, torch.bool);  expand_33 = None
        where_66 = torch.ops.aten.where.self(convert_element_type_41, full_default_83, slice_1943);  convert_element_type_41 = full_default_83 = slice_1943 = None
        view_604 = torch.ops.aten.view.default(view_601, [4, 12, 1024, 513])
        permute_529 = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
        slice_1949 = torch.ops.aten.slice.Tensor(permute_529, 1, -256, 9223372036854775807);  permute_529 = None
        slice_1951 = torch.ops.aten.slice.Tensor(slice_1949, 3, -257, 9223372036854775807);  slice_1949 = None
        copy_101 = torch.ops.aten.copy.default(slice_1951, where_66);  slice_1951 = where_66 = None
        view_605 = torch.ops.aten.view.default(view_601, [4, 12, 1024, 513]);  view_601 = None
        permute_530 = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None
        slice_1953 = torch.ops.aten.slice.Tensor(permute_530, 1, -256, 9223372036854775807)
        slice_scatter_370 = torch.ops.aten.slice_scatter.default(slice_1953, copy_101, 3, -257, 9223372036854775807);  slice_1953 = copy_101 = None
        slice_scatter_372 = torch.ops.aten.slice_scatter.default(permute_530, slice_scatter_370, 1, -256, 9223372036854775807);  permute_530 = slice_scatter_370 = None
        permute_531 = torch.ops.aten.permute.default(slice_scatter_372, [0, 2, 1, 3]);  slice_scatter_372 = None
        view_606 = torch.ops.aten.view.default(permute_531, [48, 4, 256, 513]);  permute_531 = None
        ne_8 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(ne_8, 2);  ne_8 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        convert_element_type_42 = torch.ops.prims.convert_element_type.default(unsqueeze_160, torch.float32)
        full_default_84 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_67 = torch.ops.aten.where.self(unsqueeze_160, full_default_84, convert_element_type_42);  unsqueeze_160 = full_default_84 = convert_element_type_42 = None
        full_76 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_533 = torch.ops.aten.permute.default(full_76, [0, 2, 1, 3]);  full_76 = None
        view_608 = torch.ops.aten.view.default(permute_533, [4, 1024, 1]);  permute_533 = None
        permute_534 = torch.ops.aten.permute.default(where_67, [0, 2, 1, 3]);  where_67 = None
        view_609 = torch.ops.aten.view.default(permute_534, [4, 1024, 1]);  permute_534 = None
        view_610 = torch.ops.aten.view.default(view_608, [4, 2, 512, 1]);  view_608 = None
        as_strided_51 = torch.ops.aten.as_strided.default(view_610, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_610 = None
        view_611 = torch.ops.aten.view.default(view_609, [4, 2, 512, 1]);  view_609 = None
        as_strided_52 = torch.ops.aten.as_strided.default(view_611, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_611 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(as_strided_51, 4);  as_strided_51 = None
        permute_535 = torch.ops.aten.permute.default(unsqueeze_161, [0, 1, 2, 4, 3]);  unsqueeze_161 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(as_strided_52, 4);  as_strided_52 = None
        permute_536 = torch.ops.aten.permute.default(unsqueeze_162, [0, 1, 4, 2, 3]);  unsqueeze_162 = None
        mul_64 = torch.ops.aten.mul.Tensor(permute_535, permute_536);  permute_535 = permute_536 = None
        view_612 = torch.ops.aten.view.default(mul_64, [4, 3, 512, 512]);  mul_64 = None
        constant_pad_nd_33 = torch.ops.aten.constant_pad_nd.default(view_612, [0, 0, 0, 1], 0.0);  view_612 = None
        view_613 = torch.ops.aten.view.default(constant_pad_nd_33, [4, 3, 512, 513]);  constant_pad_nd_33 = None
        full_77 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1963 = torch.ops.aten.slice.Tensor(view_613, 2, 0, 256)
        slice_1964 = torch.ops.aten.slice.Tensor(slice_1963, 3, 0, 257);  slice_1963 = None
        slice_1966 = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_1968 = torch.ops.aten.slice.Tensor(slice_1966, 3, 256, 9223372036854775807);  slice_1966 = None
        copy_102 = torch.ops.aten.copy.default(slice_1968, slice_1964);  slice_1968 = slice_1964 = None
        slice_1970 = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_scatter_374 = torch.ops.aten.slice_scatter.default(slice_1970, copy_102, 3, 256, 9223372036854775807);  slice_1970 = copy_102 = None
        slice_scatter_376 = torch.ops.aten.slice_scatter.default(full_77, slice_scatter_374, 1, 0, -1);  full_77 = slice_scatter_374 = None
        select_170 = torch.ops.aten.select.int(view_613, 1, -1)
        slice_1977 = torch.ops.aten.slice.Tensor(select_170, 1, 256, 9223372036854775807);  select_170 = None
        slice_1978 = torch.ops.aten.slice.Tensor(slice_1977, 2, 0, 257);  slice_1977 = None
        select_172 = torch.ops.aten.select.int(slice_scatter_376, 1, -1)
        slice_1984 = torch.ops.aten.slice.Tensor(select_172, 2, 256, 9223372036854775807);  select_172 = None
        copy_103 = torch.ops.aten.copy.default(slice_1984, slice_1978);  slice_1984 = slice_1978 = None
        select_173 = torch.ops.aten.select.int(slice_scatter_376, 1, -1)
        slice_scatter_378 = torch.ops.aten.slice_scatter.default(select_173, copy_103, 2, 256, 9223372036854775807);  select_173 = copy_103 = None
        select_scatter_34 = torch.ops.aten.select_scatter.default(slice_scatter_376, slice_scatter_378, 1, -1);  slice_scatter_376 = slice_scatter_378 = None
        slice_1992 = torch.ops.aten.slice.Tensor(view_613, 2, -257, -1)
        slice_1993 = torch.ops.aten.slice.Tensor(slice_1992, 3, 257, 9223372036854775807);  slice_1992 = None
        slice_1999 = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_2001 = torch.ops.aten.slice.Tensor(slice_1999, 3, 0, 256);  slice_1999 = None
        copy_104 = torch.ops.aten.copy.default(slice_2001, slice_1993);  slice_2001 = slice_1993 = None
        slice_2003 = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_scatter_381 = torch.ops.aten.slice_scatter.default(slice_2003, copy_104, 3, 0, 256);  slice_2003 = copy_104 = None
        slice_scatter_383 = torch.ops.aten.slice_scatter.default(select_scatter_34, slice_scatter_381, 1, 1, 9223372036854775807);  select_scatter_34 = slice_scatter_381 = None
        select_175 = torch.ops.aten.select.int(view_613, 1, 0);  view_613 = None
        slice_2010 = torch.ops.aten.slice.Tensor(select_175, 1, 0, 255);  select_175 = None
        slice_2011 = torch.ops.aten.slice.Tensor(slice_2010, 2, -255, 9223372036854775807);  slice_2010 = None
        select_177 = torch.ops.aten.select.int(slice_scatter_383, 1, 0)
        slice_2016 = torch.ops.aten.slice.Tensor(select_177, 1, 1, 256);  select_177 = None
        slice_2017 = torch.ops.aten.slice.Tensor(slice_2016, 2, 1, 256);  slice_2016 = None
        copy_105 = torch.ops.aten.copy.default(slice_2017, slice_2011);  slice_2017 = slice_2011 = None
        select_178 = torch.ops.aten.select.int(slice_scatter_383, 1, 0)
        slice_2019 = torch.ops.aten.slice.Tensor(select_178, 1, 1, 256)
        slice_scatter_385 = torch.ops.aten.slice_scatter.default(slice_2019, copy_105, 2, 1, 256);  slice_2019 = copy_105 = None
        slice_scatter_386 = torch.ops.aten.slice_scatter.default(select_178, slice_scatter_385, 1, 1, 256);  select_178 = slice_scatter_385 = None
        select_scatter_35 = torch.ops.aten.select_scatter.default(slice_scatter_383, slice_scatter_386, 1, 0);  slice_scatter_383 = slice_scatter_386 = None
        full_default_85 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_34 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(iota_34, -2);  iota_34 = None
        iota_35 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        sub_67 = torch.ops.aten.sub.Tensor(unsqueeze_163, unsqueeze_164);  unsqueeze_163 = unsqueeze_164 = None
        le_17 = torch.ops.aten.le.Scalar(sub_67, 0);  sub_67 = None
        full_default_86 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_68 = torch.ops.aten.where.self(le_17, full_default_85, full_default_86);  le_17 = full_default_85 = full_default_86 = None
        rev_34 = torch.ops.prims.rev.default(where_68, [0]);  where_68 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(rev_34, 0);  rev_34 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
        rev_35 = torch.ops.prims.rev.default(unsqueeze_166, [1, 3])
        expand_34 = torch.ops.aten.expand.default(unsqueeze_166, [4, 256, 1, 257]);  unsqueeze_166 = None
        view_616 = torch.ops.aten.view.default(select_scatter_35, [4, 1, 1024, 513])
        permute_539 = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        slice_2030 = torch.ops.aten.slice.Tensor(permute_539, 1, 0, 256);  permute_539 = None
        slice_2032 = torch.ops.aten.slice.Tensor(slice_2030, 3, 0, 257);  slice_2030 = None
        full_default_87 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(expand_34, torch.bool);  expand_34 = None
        where_69 = torch.ops.aten.where.self(convert_element_type_43, full_default_87, slice_2032);  convert_element_type_43 = full_default_87 = slice_2032 = None
        view_617 = torch.ops.aten.view.default(select_scatter_35, [4, 1, 1024, 513])
        permute_540 = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
        slice_2038 = torch.ops.aten.slice.Tensor(permute_540, 1, 0, 256);  permute_540 = None
        slice_2040 = torch.ops.aten.slice.Tensor(slice_2038, 3, 0, 257);  slice_2038 = None
        copy_106 = torch.ops.aten.copy.default(slice_2040, where_69);  slice_2040 = where_69 = None
        view_618 = torch.ops.aten.view.default(select_scatter_35, [4, 1, 1024, 513]);  select_scatter_35 = None
        permute_541 = torch.ops.aten.permute.default(view_618, [0, 2, 1, 3]);  view_618 = None
        slice_2042 = torch.ops.aten.slice.Tensor(permute_541, 1, 0, 256)
        slice_scatter_388 = torch.ops.aten.slice_scatter.default(slice_2042, copy_106, 3, 0, 257);  slice_2042 = copy_106 = None
        slice_scatter_390 = torch.ops.aten.slice_scatter.default(permute_541, slice_scatter_388, 1, 0, 256);  permute_541 = slice_scatter_388 = None
        permute_542 = torch.ops.aten.permute.default(slice_scatter_390, [0, 2, 1, 3]);  slice_scatter_390 = None
        view_619 = torch.ops.aten.view.default(permute_542, [4, 4, 256, 513]);  permute_542 = None
        expand_35 = torch.ops.aten.expand.default(rev_35, [4, 256, 1, 257]);  rev_35 = None
        view_621 = torch.ops.aten.view.default(view_619, [4, 1, 1024, 513])
        permute_544 = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None
        slice_2053 = torch.ops.aten.slice.Tensor(permute_544, 1, -256, 9223372036854775807);  permute_544 = None
        slice_2055 = torch.ops.aten.slice.Tensor(slice_2053, 3, -257, 9223372036854775807);  slice_2053 = None
        full_default_88 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(expand_35, torch.bool);  expand_35 = None
        where_70 = torch.ops.aten.where.self(convert_element_type_44, full_default_88, slice_2055);  convert_element_type_44 = full_default_88 = slice_2055 = None
        view_622 = torch.ops.aten.view.default(view_619, [4, 1, 1024, 513])
        permute_545 = torch.ops.aten.permute.default(view_622, [0, 2, 1, 3]);  view_622 = None
        slice_2061 = torch.ops.aten.slice.Tensor(permute_545, 1, -256, 9223372036854775807);  permute_545 = None
        slice_2063 = torch.ops.aten.slice.Tensor(slice_2061, 3, -257, 9223372036854775807);  slice_2061 = None
        copy_107 = torch.ops.aten.copy.default(slice_2063, where_70);  slice_2063 = where_70 = None
        view_623 = torch.ops.aten.view.default(view_619, [4, 1, 1024, 513]);  view_619 = None
        permute_546 = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None
        slice_2065 = torch.ops.aten.slice.Tensor(permute_546, 1, -256, 9223372036854775807)
        slice_scatter_392 = torch.ops.aten.slice_scatter.default(slice_2065, copy_107, 3, -257, 9223372036854775807);  slice_2065 = copy_107 = None
        slice_scatter_394 = torch.ops.aten.slice_scatter.default(permute_546, slice_scatter_392, 1, -256, 9223372036854775807);  permute_546 = slice_scatter_392 = None
        permute_547 = torch.ops.aten.permute.default(slice_scatter_394, [0, 2, 1, 3]);  slice_scatter_394 = None
        view_624 = torch.ops.aten.view.default(permute_547, [4, 4, 256, 513]);  permute_547 = None
        view_626 = torch.ops.aten.view.default(view_606, [4, 12, 1024, 513]);  view_606 = None
        permute_549 = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
        view_627 = torch.ops.aten.view.default(view_624, [4, 1, 1024, 513]);  view_624 = None
        permute_550 = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
        add_125 = torch.ops.aten.add.Tensor(permute_549, permute_550);  permute_549 = permute_550 = None
        permute_551 = torch.ops.aten.permute.default(add_125, [0, 2, 1, 3]);  add_125 = None
        view_629 = torch.ops.aten.view.default(permute_551, [48, 4, 256, 513]);  permute_551 = None
        view_630 = torch.ops.aten.view.default(view_629, [4, 12, 1024, 513]);  view_629 = None
        permute_552 = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        clone_109 = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
        amax_8 = torch.ops.aten.amax.default(clone_109, [-1], True)
        sub_68 = torch.ops.aten.sub.Tensor(clone_109, amax_8);  clone_109 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_87 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        full_default_89 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_71 = torch.ops.aten.where.self(unsqueeze_168, full_default_89, div_87);  unsqueeze_168 = full_default_89 = div_87 = None
        view_631 = torch.ops.aten.view.default(add_122, [1024, 4, 12, 64]);  add_122 = None
        permute_553 = torch.ops.aten.permute.default(view_631, [1, 0, 2, 3]);  view_631 = None
        permute_554 = torch.ops.aten.permute.default(where_71, [0, 2, 1, 3]);  where_71 = None
        clone_111 = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
        view_632 = torch.ops.aten.view.default(clone_111, [48, 4, 256, 513]);  clone_111 = None
        permute_555 = torch.ops.aten.permute.default(permute_553, [0, 2, 1, 3]);  permute_553 = None
        view_633 = torch.ops.aten.view.default(permute_555, [48, 1024, 64]);  permute_555 = None
        constant_pad_nd_34 = torch.ops.aten.constant_pad_nd.default(view_633, [0, 0, 256, 256], -1.0);  view_633 = None
        as_strided_53 = torch.ops.aten.as_strided.default(constant_pad_nd_34, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_34 = None
        constant_pad_nd_35 = torch.ops.aten.constant_pad_nd.default(view_632, [0, 257], 0.0);  view_632 = None
        view_634 = torch.ops.aten.view.default(constant_pad_nd_35, [48, 4, -1]);  constant_pad_nd_35 = None
        slice_2075 = torch.ops.aten.slice.Tensor(view_634, 2, 0, -256);  view_634 = None
        view_635 = torch.ops.aten.view.default(slice_2075, [48, 4, 256, 769]);  slice_2075 = None
        slice_2079 = torch.ops.aten.slice.Tensor(view_635, 3, 0, -1);  view_635 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(slice_2079, 4);  slice_2079 = None
        permute_556 = torch.ops.aten.permute.default(unsqueeze_169, [0, 1, 2, 4, 3]);  unsqueeze_169 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
        permute_557 = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 3, 2]);  unsqueeze_170 = None
        permute_558 = torch.ops.aten.permute.default(permute_556, [0, 1, 2, 4, 3]);  permute_556 = None
        view_636 = torch.ops.aten.view.default(permute_558, [192, 256, 768]);  permute_558 = None
        permute_559 = torch.ops.aten.permute.default(permute_557, [0, 1, 4, 3, 2]);  permute_557 = None
        clone_112 = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        view_637 = torch.ops.aten.view.default(clone_112, [192, 768, 64]);  clone_112 = None
        bmm_17 = torch.ops.aten.bmm.default(view_636, view_637);  view_636 = view_637 = None
        view_638 = torch.ops.aten.view.default(bmm_17, [48, 4, 256, 1, 64]);  bmm_17 = None
        permute_560 = torch.ops.aten.permute.default(view_638, [0, 1, 2, 4, 3]);  view_638 = None
        view_639 = torch.ops.aten.view.default(permute_560, [48, 4, 256, 64]);  permute_560 = None
        view_640 = torch.ops.aten.view.default(view_639, [4, 12, 1024, 64]);  view_639 = None
        permute_561 = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
        permute_562 = torch.ops.aten.permute.default(permute_561, [1, 0, 2, 3]);  permute_561 = None
        clone_113 = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
        view_641 = torch.ops.aten.view.default(clone_113, [1024, 4, 768]);  clone_113 = None
        permute_563 = torch.ops.aten.permute.default(view_641, [1, 0, 2]);  view_641 = None
        permute_564 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        clone_114 = torch.ops.aten.clone.default(permute_563, memory_format = torch.contiguous_format);  permute_563 = None
        view_642 = torch.ops.aten.view.default(clone_114, [4096, 768]);  clone_114 = None
        mm_35 = torch.ops.aten.mm.default(view_642, permute_564);  view_642 = permute_564 = None
        view_643 = torch.ops.aten.view.default(mm_35, [4, 1024, 768]);  mm_35 = None
        add_127 = torch.ops.aten.add.Tensor(view_643, arg138_1);  view_643 = arg138_1 = None
        add_128 = torch.ops.aten.add.Tensor(add_127, add_119);  add_127 = add_119 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_129 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_70 = torch.ops.aten.sub.Tensor(add_128, getitem_33);  add_128 = getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = rsqrt_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg139_1);  mul_65 = arg139_1 = None
        add_130 = torch.ops.aten.add.Tensor(mul_66, arg140_1);  mul_66 = arg140_1 = None
        view_644 = torch.ops.aten.view.default(add_130, [4096, 768])
        permute_565 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg142_1, view_644, permute_565);  arg142_1 = view_644 = permute_565 = None
        view_645 = torch.ops.aten.view.default(addmm_16, [4, 1024, 3072]);  addmm_16 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_645, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(view_645, 0.7071067811865476);  view_645 = None
        erf_8 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_131 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_131);  mul_67 = add_131 = None
        view_646 = torch.ops.aten.view.default(mul_69, [4096, 3072]);  mul_69 = None
        permute_566 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg144_1, view_646, permute_566);  arg144_1 = view_646 = permute_566 = None
        view_647 = torch.ops.aten.view.default(addmm_17, [4, 1024, 768]);  addmm_17 = None
        add_132 = torch.ops.aten.add.Tensor(view_647, add_130);  view_647 = add_130 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_133 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_132, getitem_35);  add_132 = getitem_35 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = rsqrt_17 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg145_1);  mul_70 = arg145_1 = None
        add_134 = torch.ops.aten.add.Tensor(mul_71, arg146_1);  mul_71 = arg146_1 = None
        permute_567 = torch.ops.aten.permute.default(add_134, [1, 0, 2])
        permute_568 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        clone_117 = torch.ops.aten.clone.default(permute_567, memory_format = torch.contiguous_format)
        view_648 = torch.ops.aten.view.default(clone_117, [4096, 768]);  clone_117 = None
        mm_36 = torch.ops.aten.mm.default(view_648, permute_568);  view_648 = permute_568 = None
        view_649 = torch.ops.aten.view.default(mm_36, [1024, 4, 768]);  mm_36 = None
        add_135 = torch.ops.aten.add.Tensor(view_649, arg148_1);  view_649 = arg148_1 = None
        permute_569 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        clone_118 = torch.ops.aten.clone.default(permute_567, memory_format = torch.contiguous_format)
        view_650 = torch.ops.aten.view.default(clone_118, [4096, 768]);  clone_118 = None
        mm_37 = torch.ops.aten.mm.default(view_650, permute_569);  view_650 = permute_569 = None
        view_651 = torch.ops.aten.view.default(mm_37, [1024, 4, 768]);  mm_37 = None
        add_136 = torch.ops.aten.add.Tensor(view_651, arg150_1);  view_651 = arg150_1 = None
        permute_570 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        clone_119 = torch.ops.aten.clone.default(permute_567, memory_format = torch.contiguous_format);  permute_567 = None
        view_652 = torch.ops.aten.view.default(clone_119, [4096, 768]);  clone_119 = None
        mm_38 = torch.ops.aten.mm.default(view_652, permute_570);  view_652 = permute_570 = None
        view_653 = torch.ops.aten.view.default(mm_38, [1024, 4, 768]);  mm_38 = None
        add_137 = torch.ops.aten.add.Tensor(view_653, arg152_1);  view_653 = arg152_1 = None
        div_90 = torch.ops.aten.div.Tensor(add_135, 8.0);  add_135 = None
        view_655 = torch.ops.aten.view.default(add_136, [1024, 4, 12, 64]);  add_136 = None
        permute_572 = torch.ops.aten.permute.default(view_655, [1, 0, 2, 3]);  view_655 = None
        permute_574 = torch.ops.aten.permute.default(permute_572, [0, 2, 1, 3]);  permute_572 = None
        view_657 = torch.ops.aten.view.default(permute_574, [48, 1024, 64]);  permute_574 = None
        view_659 = torch.ops.aten.view.default(view_657, [48, 2, 512, 64]);  view_657 = None
        as_strided_55 = torch.ops.aten.as_strided.default(view_659, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_659 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
        permute_576 = torch.ops.aten.permute.default(unsqueeze_172, [0, 1, 4, 2, 3]);  unsqueeze_172 = None
        view_660 = torch.ops.aten.view.default(div_90, [1024, 4, 12, 64]);  div_90 = None
        permute_578 = torch.ops.aten.permute.default(view_660, [1, 0, 2, 3]);  view_660 = None
        permute_579 = torch.ops.aten.permute.default(permute_578, [0, 2, 1, 3]);  permute_578 = None
        view_661 = torch.ops.aten.view.default(permute_579, [48, 1024, 64]);  permute_579 = None
        view_662 = torch.ops.aten.view.default(view_661, [48, 2, 512, 64]);  view_661 = None
        as_strided_56 = torch.ops.aten.as_strided.default(view_662, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_662 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(as_strided_56, 4);  as_strided_56 = None
        permute_580 = torch.ops.aten.permute.default(unsqueeze_173, [0, 1, 2, 4, 3]);  unsqueeze_173 = None
        permute_581 = torch.ops.aten.permute.default(permute_580, [0, 1, 2, 4, 3]);  permute_580 = None
        clone_120 = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
        view_663 = torch.ops.aten.view.default(clone_120, [144, 512, 64]);  clone_120 = None
        permute_582 = torch.ops.aten.permute.default(permute_576, [0, 1, 4, 3, 2]);  permute_576 = None
        clone_121 = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
        view_664 = torch.ops.aten.view.default(clone_121, [144, 64, 512]);  clone_121 = None
        bmm_18 = torch.ops.aten.bmm.default(view_663, view_664);  view_663 = view_664 = None
        view_665 = torch.ops.aten.view.default(bmm_18, [48, 3, 512, 1, 512]);  bmm_18 = None
        permute_583 = torch.ops.aten.permute.default(view_665, [0, 1, 2, 4, 3]);  view_665 = None
        view_666 = torch.ops.aten.view.default(permute_583, [48, 3, 512, 512]);  permute_583 = None
        constant_pad_nd_36 = torch.ops.aten.constant_pad_nd.default(view_666, [0, 0, 0, 1], 0.0);  view_666 = None
        view_667 = torch.ops.aten.view.default(constant_pad_nd_36, [48, 3, 512, 513]);  constant_pad_nd_36 = None
        full_81 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2082 = torch.ops.aten.slice.Tensor(view_667, 2, 0, 256)
        slice_2083 = torch.ops.aten.slice.Tensor(slice_2082, 3, 0, 257);  slice_2082 = None
        slice_2085 = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_2087 = torch.ops.aten.slice.Tensor(slice_2085, 3, 256, 9223372036854775807);  slice_2085 = None
        copy_108 = torch.ops.aten.copy.default(slice_2087, slice_2083);  slice_2087 = slice_2083 = None
        slice_2089 = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_scatter_396 = torch.ops.aten.slice_scatter.default(slice_2089, copy_108, 3, 256, 9223372036854775807);  slice_2089 = copy_108 = None
        slice_scatter_398 = torch.ops.aten.slice_scatter.default(full_81, slice_scatter_396, 1, 0, -1);  full_81 = slice_scatter_396 = None
        select_180 = torch.ops.aten.select.int(view_667, 1, -1)
        slice_2096 = torch.ops.aten.slice.Tensor(select_180, 1, 256, 9223372036854775807);  select_180 = None
        slice_2097 = torch.ops.aten.slice.Tensor(slice_2096, 2, 0, 257);  slice_2096 = None
        select_182 = torch.ops.aten.select.int(slice_scatter_398, 1, -1)
        slice_2103 = torch.ops.aten.slice.Tensor(select_182, 2, 256, 9223372036854775807);  select_182 = None
        copy_109 = torch.ops.aten.copy.default(slice_2103, slice_2097);  slice_2103 = slice_2097 = None
        select_183 = torch.ops.aten.select.int(slice_scatter_398, 1, -1)
        slice_scatter_400 = torch.ops.aten.slice_scatter.default(select_183, copy_109, 2, 256, 9223372036854775807);  select_183 = copy_109 = None
        select_scatter_36 = torch.ops.aten.select_scatter.default(slice_scatter_398, slice_scatter_400, 1, -1);  slice_scatter_398 = slice_scatter_400 = None
        slice_2111 = torch.ops.aten.slice.Tensor(view_667, 2, -257, -1)
        slice_2112 = torch.ops.aten.slice.Tensor(slice_2111, 3, 257, 9223372036854775807);  slice_2111 = None
        slice_2118 = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_2120 = torch.ops.aten.slice.Tensor(slice_2118, 3, 0, 256);  slice_2118 = None
        copy_110 = torch.ops.aten.copy.default(slice_2120, slice_2112);  slice_2120 = slice_2112 = None
        slice_2122 = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_scatter_403 = torch.ops.aten.slice_scatter.default(slice_2122, copy_110, 3, 0, 256);  slice_2122 = copy_110 = None
        slice_scatter_405 = torch.ops.aten.slice_scatter.default(select_scatter_36, slice_scatter_403, 1, 1, 9223372036854775807);  select_scatter_36 = slice_scatter_403 = None
        select_185 = torch.ops.aten.select.int(view_667, 1, 0);  view_667 = None
        slice_2129 = torch.ops.aten.slice.Tensor(select_185, 1, 0, 255);  select_185 = None
        slice_2130 = torch.ops.aten.slice.Tensor(slice_2129, 2, -255, 9223372036854775807);  slice_2129 = None
        select_187 = torch.ops.aten.select.int(slice_scatter_405, 1, 0)
        slice_2135 = torch.ops.aten.slice.Tensor(select_187, 1, 1, 256);  select_187 = None
        slice_2136 = torch.ops.aten.slice.Tensor(slice_2135, 2, 1, 256);  slice_2135 = None
        copy_111 = torch.ops.aten.copy.default(slice_2136, slice_2130);  slice_2136 = slice_2130 = None
        select_188 = torch.ops.aten.select.int(slice_scatter_405, 1, 0)
        slice_2138 = torch.ops.aten.slice.Tensor(select_188, 1, 1, 256)
        slice_scatter_407 = torch.ops.aten.slice_scatter.default(slice_2138, copy_111, 2, 1, 256);  slice_2138 = copy_111 = None
        slice_scatter_408 = torch.ops.aten.slice_scatter.default(select_188, slice_scatter_407, 1, 1, 256);  select_188 = slice_scatter_407 = None
        select_scatter_37 = torch.ops.aten.select_scatter.default(slice_scatter_405, slice_scatter_408, 1, 0);  slice_scatter_405 = slice_scatter_408 = None
        full_default_90 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_36 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(iota_36, -2);  iota_36 = None
        iota_37 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        sub_73 = torch.ops.aten.sub.Tensor(unsqueeze_174, unsqueeze_175);  unsqueeze_174 = unsqueeze_175 = None
        le_18 = torch.ops.aten.le.Scalar(sub_73, 0);  sub_73 = None
        full_default_91 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_72 = torch.ops.aten.where.self(le_18, full_default_90, full_default_91);  le_18 = full_default_90 = full_default_91 = None
        rev_36 = torch.ops.prims.rev.default(where_72, [0]);  where_72 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(rev_36, 0);  rev_36 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        rev_37 = torch.ops.prims.rev.default(unsqueeze_177, [1, 3])
        expand_36 = torch.ops.aten.expand.default(unsqueeze_177, [4, 256, 12, 257]);  unsqueeze_177 = None
        view_670 = torch.ops.aten.view.default(select_scatter_37, [4, 12, 1024, 513])
        permute_586 = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None
        slice_2149 = torch.ops.aten.slice.Tensor(permute_586, 1, 0, 256);  permute_586 = None
        slice_2151 = torch.ops.aten.slice.Tensor(slice_2149, 3, 0, 257);  slice_2149 = None
        full_default_92 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(expand_36, torch.bool);  expand_36 = None
        where_73 = torch.ops.aten.where.self(convert_element_type_45, full_default_92, slice_2151);  convert_element_type_45 = full_default_92 = slice_2151 = None
        view_671 = torch.ops.aten.view.default(select_scatter_37, [4, 12, 1024, 513])
        permute_587 = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
        slice_2157 = torch.ops.aten.slice.Tensor(permute_587, 1, 0, 256);  permute_587 = None
        slice_2159 = torch.ops.aten.slice.Tensor(slice_2157, 3, 0, 257);  slice_2157 = None
        copy_112 = torch.ops.aten.copy.default(slice_2159, where_73);  slice_2159 = where_73 = None
        view_672 = torch.ops.aten.view.default(select_scatter_37, [4, 12, 1024, 513]);  select_scatter_37 = None
        permute_588 = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
        slice_2161 = torch.ops.aten.slice.Tensor(permute_588, 1, 0, 256)
        slice_scatter_410 = torch.ops.aten.slice_scatter.default(slice_2161, copy_112, 3, 0, 257);  slice_2161 = copy_112 = None
        slice_scatter_412 = torch.ops.aten.slice_scatter.default(permute_588, slice_scatter_410, 1, 0, 256);  permute_588 = slice_scatter_410 = None
        permute_589 = torch.ops.aten.permute.default(slice_scatter_412, [0, 2, 1, 3]);  slice_scatter_412 = None
        view_673 = torch.ops.aten.view.default(permute_589, [48, 4, 256, 513]);  permute_589 = None
        expand_37 = torch.ops.aten.expand.default(rev_37, [4, 256, 12, 257]);  rev_37 = None
        view_675 = torch.ops.aten.view.default(view_673, [4, 12, 1024, 513])
        permute_591 = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None
        slice_2172 = torch.ops.aten.slice.Tensor(permute_591, 1, -256, 9223372036854775807);  permute_591 = None
        slice_2174 = torch.ops.aten.slice.Tensor(slice_2172, 3, -257, 9223372036854775807);  slice_2172 = None
        full_default_93 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(expand_37, torch.bool);  expand_37 = None
        where_74 = torch.ops.aten.where.self(convert_element_type_46, full_default_93, slice_2174);  convert_element_type_46 = full_default_93 = slice_2174 = None
        view_676 = torch.ops.aten.view.default(view_673, [4, 12, 1024, 513])
        permute_592 = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
        slice_2180 = torch.ops.aten.slice.Tensor(permute_592, 1, -256, 9223372036854775807);  permute_592 = None
        slice_2182 = torch.ops.aten.slice.Tensor(slice_2180, 3, -257, 9223372036854775807);  slice_2180 = None
        copy_113 = torch.ops.aten.copy.default(slice_2182, where_74);  slice_2182 = where_74 = None
        view_677 = torch.ops.aten.view.default(view_673, [4, 12, 1024, 513]);  view_673 = None
        permute_593 = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None
        slice_2184 = torch.ops.aten.slice.Tensor(permute_593, 1, -256, 9223372036854775807)
        slice_scatter_414 = torch.ops.aten.slice_scatter.default(slice_2184, copy_113, 3, -257, 9223372036854775807);  slice_2184 = copy_113 = None
        slice_scatter_416 = torch.ops.aten.slice_scatter.default(permute_593, slice_scatter_414, 1, -256, 9223372036854775807);  permute_593 = slice_scatter_414 = None
        permute_594 = torch.ops.aten.permute.default(slice_scatter_416, [0, 2, 1, 3]);  slice_scatter_416 = None
        view_678 = torch.ops.aten.view.default(permute_594, [48, 4, 256, 513]);  permute_594 = None
        ne_9 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(ne_9, 2);  ne_9 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(unsqueeze_179, torch.float32)
        full_default_94 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_75 = torch.ops.aten.where.self(unsqueeze_179, full_default_94, convert_element_type_47);  unsqueeze_179 = full_default_94 = convert_element_type_47 = None
        full_85 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_596 = torch.ops.aten.permute.default(full_85, [0, 2, 1, 3]);  full_85 = None
        view_680 = torch.ops.aten.view.default(permute_596, [4, 1024, 1]);  permute_596 = None
        permute_597 = torch.ops.aten.permute.default(where_75, [0, 2, 1, 3]);  where_75 = None
        view_681 = torch.ops.aten.view.default(permute_597, [4, 1024, 1]);  permute_597 = None
        view_682 = torch.ops.aten.view.default(view_680, [4, 2, 512, 1]);  view_680 = None
        as_strided_57 = torch.ops.aten.as_strided.default(view_682, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_682 = None
        view_683 = torch.ops.aten.view.default(view_681, [4, 2, 512, 1]);  view_681 = None
        as_strided_58 = torch.ops.aten.as_strided.default(view_683, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_683 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(as_strided_57, 4);  as_strided_57 = None
        permute_598 = torch.ops.aten.permute.default(unsqueeze_180, [0, 1, 2, 4, 3]);  unsqueeze_180 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(as_strided_58, 4);  as_strided_58 = None
        permute_599 = torch.ops.aten.permute.default(unsqueeze_181, [0, 1, 4, 2, 3]);  unsqueeze_181 = None
        mul_72 = torch.ops.aten.mul.Tensor(permute_598, permute_599);  permute_598 = permute_599 = None
        view_684 = torch.ops.aten.view.default(mul_72, [4, 3, 512, 512]);  mul_72 = None
        constant_pad_nd_37 = torch.ops.aten.constant_pad_nd.default(view_684, [0, 0, 0, 1], 0.0);  view_684 = None
        view_685 = torch.ops.aten.view.default(constant_pad_nd_37, [4, 3, 512, 513]);  constant_pad_nd_37 = None
        full_86 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2194 = torch.ops.aten.slice.Tensor(view_685, 2, 0, 256)
        slice_2195 = torch.ops.aten.slice.Tensor(slice_2194, 3, 0, 257);  slice_2194 = None
        slice_2197 = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_2199 = torch.ops.aten.slice.Tensor(slice_2197, 3, 256, 9223372036854775807);  slice_2197 = None
        copy_114 = torch.ops.aten.copy.default(slice_2199, slice_2195);  slice_2199 = slice_2195 = None
        slice_2201 = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_scatter_418 = torch.ops.aten.slice_scatter.default(slice_2201, copy_114, 3, 256, 9223372036854775807);  slice_2201 = copy_114 = None
        slice_scatter_420 = torch.ops.aten.slice_scatter.default(full_86, slice_scatter_418, 1, 0, -1);  full_86 = slice_scatter_418 = None
        select_190 = torch.ops.aten.select.int(view_685, 1, -1)
        slice_2208 = torch.ops.aten.slice.Tensor(select_190, 1, 256, 9223372036854775807);  select_190 = None
        slice_2209 = torch.ops.aten.slice.Tensor(slice_2208, 2, 0, 257);  slice_2208 = None
        select_192 = torch.ops.aten.select.int(slice_scatter_420, 1, -1)
        slice_2215 = torch.ops.aten.slice.Tensor(select_192, 2, 256, 9223372036854775807);  select_192 = None
        copy_115 = torch.ops.aten.copy.default(slice_2215, slice_2209);  slice_2215 = slice_2209 = None
        select_193 = torch.ops.aten.select.int(slice_scatter_420, 1, -1)
        slice_scatter_422 = torch.ops.aten.slice_scatter.default(select_193, copy_115, 2, 256, 9223372036854775807);  select_193 = copy_115 = None
        select_scatter_38 = torch.ops.aten.select_scatter.default(slice_scatter_420, slice_scatter_422, 1, -1);  slice_scatter_420 = slice_scatter_422 = None
        slice_2223 = torch.ops.aten.slice.Tensor(view_685, 2, -257, -1)
        slice_2224 = torch.ops.aten.slice.Tensor(slice_2223, 3, 257, 9223372036854775807);  slice_2223 = None
        slice_2230 = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_2232 = torch.ops.aten.slice.Tensor(slice_2230, 3, 0, 256);  slice_2230 = None
        copy_116 = torch.ops.aten.copy.default(slice_2232, slice_2224);  slice_2232 = slice_2224 = None
        slice_2234 = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_scatter_425 = torch.ops.aten.slice_scatter.default(slice_2234, copy_116, 3, 0, 256);  slice_2234 = copy_116 = None
        slice_scatter_427 = torch.ops.aten.slice_scatter.default(select_scatter_38, slice_scatter_425, 1, 1, 9223372036854775807);  select_scatter_38 = slice_scatter_425 = None
        select_195 = torch.ops.aten.select.int(view_685, 1, 0);  view_685 = None
        slice_2241 = torch.ops.aten.slice.Tensor(select_195, 1, 0, 255);  select_195 = None
        slice_2242 = torch.ops.aten.slice.Tensor(slice_2241, 2, -255, 9223372036854775807);  slice_2241 = None
        select_197 = torch.ops.aten.select.int(slice_scatter_427, 1, 0)
        slice_2247 = torch.ops.aten.slice.Tensor(select_197, 1, 1, 256);  select_197 = None
        slice_2248 = torch.ops.aten.slice.Tensor(slice_2247, 2, 1, 256);  slice_2247 = None
        copy_117 = torch.ops.aten.copy.default(slice_2248, slice_2242);  slice_2248 = slice_2242 = None
        select_198 = torch.ops.aten.select.int(slice_scatter_427, 1, 0)
        slice_2250 = torch.ops.aten.slice.Tensor(select_198, 1, 1, 256)
        slice_scatter_429 = torch.ops.aten.slice_scatter.default(slice_2250, copy_117, 2, 1, 256);  slice_2250 = copy_117 = None
        slice_scatter_430 = torch.ops.aten.slice_scatter.default(select_198, slice_scatter_429, 1, 1, 256);  select_198 = slice_scatter_429 = None
        select_scatter_39 = torch.ops.aten.select_scatter.default(slice_scatter_427, slice_scatter_430, 1, 0);  slice_scatter_427 = slice_scatter_430 = None
        full_default_95 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_38 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(iota_38, -2);  iota_38 = None
        iota_39 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        sub_75 = torch.ops.aten.sub.Tensor(unsqueeze_182, unsqueeze_183);  unsqueeze_182 = unsqueeze_183 = None
        le_19 = torch.ops.aten.le.Scalar(sub_75, 0);  sub_75 = None
        full_default_96 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_76 = torch.ops.aten.where.self(le_19, full_default_95, full_default_96);  le_19 = full_default_95 = full_default_96 = None
        rev_38 = torch.ops.prims.rev.default(where_76, [0]);  where_76 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(rev_38, 0);  rev_38 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
        rev_39 = torch.ops.prims.rev.default(unsqueeze_185, [1, 3])
        expand_38 = torch.ops.aten.expand.default(unsqueeze_185, [4, 256, 1, 257]);  unsqueeze_185 = None
        view_688 = torch.ops.aten.view.default(select_scatter_39, [4, 1, 1024, 513])
        permute_602 = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        slice_2261 = torch.ops.aten.slice.Tensor(permute_602, 1, 0, 256);  permute_602 = None
        slice_2263 = torch.ops.aten.slice.Tensor(slice_2261, 3, 0, 257);  slice_2261 = None
        full_default_97 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(expand_38, torch.bool);  expand_38 = None
        where_77 = torch.ops.aten.where.self(convert_element_type_48, full_default_97, slice_2263);  convert_element_type_48 = full_default_97 = slice_2263 = None
        view_689 = torch.ops.aten.view.default(select_scatter_39, [4, 1, 1024, 513])
        permute_603 = torch.ops.aten.permute.default(view_689, [0, 2, 1, 3]);  view_689 = None
        slice_2269 = torch.ops.aten.slice.Tensor(permute_603, 1, 0, 256);  permute_603 = None
        slice_2271 = torch.ops.aten.slice.Tensor(slice_2269, 3, 0, 257);  slice_2269 = None
        copy_118 = torch.ops.aten.copy.default(slice_2271, where_77);  slice_2271 = where_77 = None
        view_690 = torch.ops.aten.view.default(select_scatter_39, [4, 1, 1024, 513]);  select_scatter_39 = None
        permute_604 = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
        slice_2273 = torch.ops.aten.slice.Tensor(permute_604, 1, 0, 256)
        slice_scatter_432 = torch.ops.aten.slice_scatter.default(slice_2273, copy_118, 3, 0, 257);  slice_2273 = copy_118 = None
        slice_scatter_434 = torch.ops.aten.slice_scatter.default(permute_604, slice_scatter_432, 1, 0, 256);  permute_604 = slice_scatter_432 = None
        permute_605 = torch.ops.aten.permute.default(slice_scatter_434, [0, 2, 1, 3]);  slice_scatter_434 = None
        view_691 = torch.ops.aten.view.default(permute_605, [4, 4, 256, 513]);  permute_605 = None
        expand_39 = torch.ops.aten.expand.default(rev_39, [4, 256, 1, 257]);  rev_39 = None
        view_693 = torch.ops.aten.view.default(view_691, [4, 1, 1024, 513])
        permute_607 = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
        slice_2284 = torch.ops.aten.slice.Tensor(permute_607, 1, -256, 9223372036854775807);  permute_607 = None
        slice_2286 = torch.ops.aten.slice.Tensor(slice_2284, 3, -257, 9223372036854775807);  slice_2284 = None
        full_default_98 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(expand_39, torch.bool);  expand_39 = None
        where_78 = torch.ops.aten.where.self(convert_element_type_49, full_default_98, slice_2286);  convert_element_type_49 = full_default_98 = slice_2286 = None
        view_694 = torch.ops.aten.view.default(view_691, [4, 1, 1024, 513])
        permute_608 = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
        slice_2292 = torch.ops.aten.slice.Tensor(permute_608, 1, -256, 9223372036854775807);  permute_608 = None
        slice_2294 = torch.ops.aten.slice.Tensor(slice_2292, 3, -257, 9223372036854775807);  slice_2292 = None
        copy_119 = torch.ops.aten.copy.default(slice_2294, where_78);  slice_2294 = where_78 = None
        view_695 = torch.ops.aten.view.default(view_691, [4, 1, 1024, 513]);  view_691 = None
        permute_609 = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
        slice_2296 = torch.ops.aten.slice.Tensor(permute_609, 1, -256, 9223372036854775807)
        slice_scatter_436 = torch.ops.aten.slice_scatter.default(slice_2296, copy_119, 3, -257, 9223372036854775807);  slice_2296 = copy_119 = None
        slice_scatter_438 = torch.ops.aten.slice_scatter.default(permute_609, slice_scatter_436, 1, -256, 9223372036854775807);  permute_609 = slice_scatter_436 = None
        permute_610 = torch.ops.aten.permute.default(slice_scatter_438, [0, 2, 1, 3]);  slice_scatter_438 = None
        view_696 = torch.ops.aten.view.default(permute_610, [4, 4, 256, 513]);  permute_610 = None
        view_698 = torch.ops.aten.view.default(view_678, [4, 12, 1024, 513]);  view_678 = None
        permute_612 = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
        view_699 = torch.ops.aten.view.default(view_696, [4, 1, 1024, 513]);  view_696 = None
        permute_613 = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None
        add_140 = torch.ops.aten.add.Tensor(permute_612, permute_613);  permute_612 = permute_613 = None
        permute_614 = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
        view_701 = torch.ops.aten.view.default(permute_614, [48, 4, 256, 513]);  permute_614 = None
        view_702 = torch.ops.aten.view.default(view_701, [4, 12, 1024, 513]);  view_701 = None
        permute_615 = torch.ops.aten.permute.default(view_702, [0, 2, 1, 3]);  view_702 = None
        clone_122 = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
        amax_9 = torch.ops.aten.amax.default(clone_122, [-1], True)
        sub_76 = torch.ops.aten.sub.Tensor(clone_122, amax_9);  clone_122 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_97 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        full_default_99 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_79 = torch.ops.aten.where.self(unsqueeze_187, full_default_99, div_97);  unsqueeze_187 = full_default_99 = div_97 = None
        view_703 = torch.ops.aten.view.default(add_137, [1024, 4, 12, 64]);  add_137 = None
        permute_616 = torch.ops.aten.permute.default(view_703, [1, 0, 2, 3]);  view_703 = None
        permute_617 = torch.ops.aten.permute.default(where_79, [0, 2, 1, 3]);  where_79 = None
        clone_124 = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_704 = torch.ops.aten.view.default(clone_124, [48, 4, 256, 513]);  clone_124 = None
        permute_618 = torch.ops.aten.permute.default(permute_616, [0, 2, 1, 3]);  permute_616 = None
        view_705 = torch.ops.aten.view.default(permute_618, [48, 1024, 64]);  permute_618 = None
        constant_pad_nd_38 = torch.ops.aten.constant_pad_nd.default(view_705, [0, 0, 256, 256], -1.0);  view_705 = None
        as_strided_59 = torch.ops.aten.as_strided.default(constant_pad_nd_38, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_38 = None
        constant_pad_nd_39 = torch.ops.aten.constant_pad_nd.default(view_704, [0, 257], 0.0);  view_704 = None
        view_706 = torch.ops.aten.view.default(constant_pad_nd_39, [48, 4, -1]);  constant_pad_nd_39 = None
        slice_2306 = torch.ops.aten.slice.Tensor(view_706, 2, 0, -256);  view_706 = None
        view_707 = torch.ops.aten.view.default(slice_2306, [48, 4, 256, 769]);  slice_2306 = None
        slice_2310 = torch.ops.aten.slice.Tensor(view_707, 3, 0, -1);  view_707 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(slice_2310, 4);  slice_2310 = None
        permute_619 = torch.ops.aten.permute.default(unsqueeze_188, [0, 1, 2, 4, 3]);  unsqueeze_188 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
        permute_620 = torch.ops.aten.permute.default(unsqueeze_189, [0, 1, 4, 3, 2]);  unsqueeze_189 = None
        permute_621 = torch.ops.aten.permute.default(permute_619, [0, 1, 2, 4, 3]);  permute_619 = None
        view_708 = torch.ops.aten.view.default(permute_621, [192, 256, 768]);  permute_621 = None
        permute_622 = torch.ops.aten.permute.default(permute_620, [0, 1, 4, 3, 2]);  permute_620 = None
        clone_125 = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        view_709 = torch.ops.aten.view.default(clone_125, [192, 768, 64]);  clone_125 = None
        bmm_19 = torch.ops.aten.bmm.default(view_708, view_709);  view_708 = view_709 = None
        view_710 = torch.ops.aten.view.default(bmm_19, [48, 4, 256, 1, 64]);  bmm_19 = None
        permute_623 = torch.ops.aten.permute.default(view_710, [0, 1, 2, 4, 3]);  view_710 = None
        view_711 = torch.ops.aten.view.default(permute_623, [48, 4, 256, 64]);  permute_623 = None
        view_712 = torch.ops.aten.view.default(view_711, [4, 12, 1024, 64]);  view_711 = None
        permute_624 = torch.ops.aten.permute.default(view_712, [0, 2, 1, 3]);  view_712 = None
        permute_625 = torch.ops.aten.permute.default(permute_624, [1, 0, 2, 3]);  permute_624 = None
        clone_126 = torch.ops.aten.clone.default(permute_625, memory_format = torch.contiguous_format);  permute_625 = None
        view_713 = torch.ops.aten.view.default(clone_126, [1024, 4, 768]);  clone_126 = None
        permute_626 = torch.ops.aten.permute.default(view_713, [1, 0, 2]);  view_713 = None
        permute_627 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        clone_127 = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
        view_714 = torch.ops.aten.view.default(clone_127, [4096, 768]);  clone_127 = None
        mm_39 = torch.ops.aten.mm.default(view_714, permute_627);  view_714 = permute_627 = None
        view_715 = torch.ops.aten.view.default(mm_39, [4, 1024, 768]);  mm_39 = None
        add_142 = torch.ops.aten.add.Tensor(view_715, arg154_1);  view_715 = arg154_1 = None
        add_143 = torch.ops.aten.add.Tensor(add_142, add_134);  add_142 = add_134 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_144 = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_78 = torch.ops.aten.sub.Tensor(add_143, getitem_37);  add_143 = getitem_37 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = rsqrt_18 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg155_1);  mul_73 = arg155_1 = None
        add_145 = torch.ops.aten.add.Tensor(mul_74, arg156_1);  mul_74 = arg156_1 = None
        view_716 = torch.ops.aten.view.default(add_145, [4096, 768])
        permute_628 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg158_1, view_716, permute_628);  arg158_1 = view_716 = permute_628 = None
        view_717 = torch.ops.aten.view.default(addmm_18, [4, 1024, 3072]);  addmm_18 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_717, 0.5)
        mul_76 = torch.ops.aten.mul.Tensor(view_717, 0.7071067811865476);  view_717 = None
        erf_9 = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_146 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_75, add_146);  mul_75 = add_146 = None
        view_718 = torch.ops.aten.view.default(mul_77, [4096, 3072]);  mul_77 = None
        permute_629 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg160_1, view_718, permute_629);  arg160_1 = view_718 = permute_629 = None
        view_719 = torch.ops.aten.view.default(addmm_19, [4, 1024, 768]);  addmm_19 = None
        add_147 = torch.ops.aten.add.Tensor(view_719, add_145);  view_719 = add_145 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_148 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_79 = torch.ops.aten.sub.Tensor(add_147, getitem_39);  add_147 = getitem_39 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = rsqrt_19 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, arg161_1);  mul_78 = arg161_1 = None
        add_149 = torch.ops.aten.add.Tensor(mul_79, arg162_1);  mul_79 = arg162_1 = None
        permute_630 = torch.ops.aten.permute.default(add_149, [1, 0, 2])
        permute_631 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        clone_130 = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format)
        view_720 = torch.ops.aten.view.default(clone_130, [4096, 768]);  clone_130 = None
        mm_40 = torch.ops.aten.mm.default(view_720, permute_631);  view_720 = permute_631 = None
        view_721 = torch.ops.aten.view.default(mm_40, [1024, 4, 768]);  mm_40 = None
        add_150 = torch.ops.aten.add.Tensor(view_721, arg164_1);  view_721 = arg164_1 = None
        permute_632 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        clone_131 = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format)
        view_722 = torch.ops.aten.view.default(clone_131, [4096, 768]);  clone_131 = None
        mm_41 = torch.ops.aten.mm.default(view_722, permute_632);  view_722 = permute_632 = None
        view_723 = torch.ops.aten.view.default(mm_41, [1024, 4, 768]);  mm_41 = None
        add_151 = torch.ops.aten.add.Tensor(view_723, arg166_1);  view_723 = arg166_1 = None
        permute_633 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        clone_132 = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_724 = torch.ops.aten.view.default(clone_132, [4096, 768]);  clone_132 = None
        mm_42 = torch.ops.aten.mm.default(view_724, permute_633);  view_724 = permute_633 = None
        view_725 = torch.ops.aten.view.default(mm_42, [1024, 4, 768]);  mm_42 = None
        add_152 = torch.ops.aten.add.Tensor(view_725, arg168_1);  view_725 = arg168_1 = None
        div_100 = torch.ops.aten.div.Tensor(add_150, 8.0);  add_150 = None
        view_727 = torch.ops.aten.view.default(add_151, [1024, 4, 12, 64]);  add_151 = None
        permute_635 = torch.ops.aten.permute.default(view_727, [1, 0, 2, 3]);  view_727 = None
        permute_637 = torch.ops.aten.permute.default(permute_635, [0, 2, 1, 3]);  permute_635 = None
        view_729 = torch.ops.aten.view.default(permute_637, [48, 1024, 64]);  permute_637 = None
        view_731 = torch.ops.aten.view.default(view_729, [48, 2, 512, 64]);  view_729 = None
        as_strided_61 = torch.ops.aten.as_strided.default(view_731, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_731 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(as_strided_61, 4);  as_strided_61 = None
        permute_639 = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
        view_732 = torch.ops.aten.view.default(div_100, [1024, 4, 12, 64]);  div_100 = None
        permute_641 = torch.ops.aten.permute.default(view_732, [1, 0, 2, 3]);  view_732 = None
        permute_642 = torch.ops.aten.permute.default(permute_641, [0, 2, 1, 3]);  permute_641 = None
        view_733 = torch.ops.aten.view.default(permute_642, [48, 1024, 64]);  permute_642 = None
        view_734 = torch.ops.aten.view.default(view_733, [48, 2, 512, 64]);  view_733 = None
        as_strided_62 = torch.ops.aten.as_strided.default(view_734, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_734 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
        permute_643 = torch.ops.aten.permute.default(unsqueeze_192, [0, 1, 2, 4, 3]);  unsqueeze_192 = None
        permute_644 = torch.ops.aten.permute.default(permute_643, [0, 1, 2, 4, 3]);  permute_643 = None
        clone_133 = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
        view_735 = torch.ops.aten.view.default(clone_133, [144, 512, 64]);  clone_133 = None
        permute_645 = torch.ops.aten.permute.default(permute_639, [0, 1, 4, 3, 2]);  permute_639 = None
        clone_134 = torch.ops.aten.clone.default(permute_645, memory_format = torch.contiguous_format);  permute_645 = None
        view_736 = torch.ops.aten.view.default(clone_134, [144, 64, 512]);  clone_134 = None
        bmm_20 = torch.ops.aten.bmm.default(view_735, view_736);  view_735 = view_736 = None
        view_737 = torch.ops.aten.view.default(bmm_20, [48, 3, 512, 1, 512]);  bmm_20 = None
        permute_646 = torch.ops.aten.permute.default(view_737, [0, 1, 2, 4, 3]);  view_737 = None
        view_738 = torch.ops.aten.view.default(permute_646, [48, 3, 512, 512]);  permute_646 = None
        constant_pad_nd_40 = torch.ops.aten.constant_pad_nd.default(view_738, [0, 0, 0, 1], 0.0);  view_738 = None
        view_739 = torch.ops.aten.view.default(constant_pad_nd_40, [48, 3, 512, 513]);  constant_pad_nd_40 = None
        full_90 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2313 = torch.ops.aten.slice.Tensor(view_739, 2, 0, 256)
        slice_2314 = torch.ops.aten.slice.Tensor(slice_2313, 3, 0, 257);  slice_2313 = None
        slice_2316 = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_2318 = torch.ops.aten.slice.Tensor(slice_2316, 3, 256, 9223372036854775807);  slice_2316 = None
        copy_120 = torch.ops.aten.copy.default(slice_2318, slice_2314);  slice_2318 = slice_2314 = None
        slice_2320 = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_scatter_440 = torch.ops.aten.slice_scatter.default(slice_2320, copy_120, 3, 256, 9223372036854775807);  slice_2320 = copy_120 = None
        slice_scatter_442 = torch.ops.aten.slice_scatter.default(full_90, slice_scatter_440, 1, 0, -1);  full_90 = slice_scatter_440 = None
        select_200 = torch.ops.aten.select.int(view_739, 1, -1)
        slice_2327 = torch.ops.aten.slice.Tensor(select_200, 1, 256, 9223372036854775807);  select_200 = None
        slice_2328 = torch.ops.aten.slice.Tensor(slice_2327, 2, 0, 257);  slice_2327 = None
        select_202 = torch.ops.aten.select.int(slice_scatter_442, 1, -1)
        slice_2334 = torch.ops.aten.slice.Tensor(select_202, 2, 256, 9223372036854775807);  select_202 = None
        copy_121 = torch.ops.aten.copy.default(slice_2334, slice_2328);  slice_2334 = slice_2328 = None
        select_203 = torch.ops.aten.select.int(slice_scatter_442, 1, -1)
        slice_scatter_444 = torch.ops.aten.slice_scatter.default(select_203, copy_121, 2, 256, 9223372036854775807);  select_203 = copy_121 = None
        select_scatter_40 = torch.ops.aten.select_scatter.default(slice_scatter_442, slice_scatter_444, 1, -1);  slice_scatter_442 = slice_scatter_444 = None
        slice_2342 = torch.ops.aten.slice.Tensor(view_739, 2, -257, -1)
        slice_2343 = torch.ops.aten.slice.Tensor(slice_2342, 3, 257, 9223372036854775807);  slice_2342 = None
        slice_2349 = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_2351 = torch.ops.aten.slice.Tensor(slice_2349, 3, 0, 256);  slice_2349 = None
        copy_122 = torch.ops.aten.copy.default(slice_2351, slice_2343);  slice_2351 = slice_2343 = None
        slice_2353 = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_scatter_447 = torch.ops.aten.slice_scatter.default(slice_2353, copy_122, 3, 0, 256);  slice_2353 = copy_122 = None
        slice_scatter_449 = torch.ops.aten.slice_scatter.default(select_scatter_40, slice_scatter_447, 1, 1, 9223372036854775807);  select_scatter_40 = slice_scatter_447 = None
        select_205 = torch.ops.aten.select.int(view_739, 1, 0);  view_739 = None
        slice_2360 = torch.ops.aten.slice.Tensor(select_205, 1, 0, 255);  select_205 = None
        slice_2361 = torch.ops.aten.slice.Tensor(slice_2360, 2, -255, 9223372036854775807);  slice_2360 = None
        select_207 = torch.ops.aten.select.int(slice_scatter_449, 1, 0)
        slice_2366 = torch.ops.aten.slice.Tensor(select_207, 1, 1, 256);  select_207 = None
        slice_2367 = torch.ops.aten.slice.Tensor(slice_2366, 2, 1, 256);  slice_2366 = None
        copy_123 = torch.ops.aten.copy.default(slice_2367, slice_2361);  slice_2367 = slice_2361 = None
        select_208 = torch.ops.aten.select.int(slice_scatter_449, 1, 0)
        slice_2369 = torch.ops.aten.slice.Tensor(select_208, 1, 1, 256)
        slice_scatter_451 = torch.ops.aten.slice_scatter.default(slice_2369, copy_123, 2, 1, 256);  slice_2369 = copy_123 = None
        slice_scatter_452 = torch.ops.aten.slice_scatter.default(select_208, slice_scatter_451, 1, 1, 256);  select_208 = slice_scatter_451 = None
        select_scatter_41 = torch.ops.aten.select_scatter.default(slice_scatter_449, slice_scatter_452, 1, 0);  slice_scatter_449 = slice_scatter_452 = None
        full_default_100 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_40 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(iota_40, -2);  iota_40 = None
        iota_41 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        sub_81 = torch.ops.aten.sub.Tensor(unsqueeze_193, unsqueeze_194);  unsqueeze_193 = unsqueeze_194 = None
        le_20 = torch.ops.aten.le.Scalar(sub_81, 0);  sub_81 = None
        full_default_101 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_80 = torch.ops.aten.where.self(le_20, full_default_100, full_default_101);  le_20 = full_default_100 = full_default_101 = None
        rev_40 = torch.ops.prims.rev.default(where_80, [0]);  where_80 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(rev_40, 0);  rev_40 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        rev_41 = torch.ops.prims.rev.default(unsqueeze_196, [1, 3])
        expand_40 = torch.ops.aten.expand.default(unsqueeze_196, [4, 256, 12, 257]);  unsqueeze_196 = None
        view_742 = torch.ops.aten.view.default(select_scatter_41, [4, 12, 1024, 513])
        permute_649 = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
        slice_2380 = torch.ops.aten.slice.Tensor(permute_649, 1, 0, 256);  permute_649 = None
        slice_2382 = torch.ops.aten.slice.Tensor(slice_2380, 3, 0, 257);  slice_2380 = None
        full_default_102 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(expand_40, torch.bool);  expand_40 = None
        where_81 = torch.ops.aten.where.self(convert_element_type_50, full_default_102, slice_2382);  convert_element_type_50 = full_default_102 = slice_2382 = None
        view_743 = torch.ops.aten.view.default(select_scatter_41, [4, 12, 1024, 513])
        permute_650 = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None
        slice_2388 = torch.ops.aten.slice.Tensor(permute_650, 1, 0, 256);  permute_650 = None
        slice_2390 = torch.ops.aten.slice.Tensor(slice_2388, 3, 0, 257);  slice_2388 = None
        copy_124 = torch.ops.aten.copy.default(slice_2390, where_81);  slice_2390 = where_81 = None
        view_744 = torch.ops.aten.view.default(select_scatter_41, [4, 12, 1024, 513]);  select_scatter_41 = None
        permute_651 = torch.ops.aten.permute.default(view_744, [0, 2, 1, 3]);  view_744 = None
        slice_2392 = torch.ops.aten.slice.Tensor(permute_651, 1, 0, 256)
        slice_scatter_454 = torch.ops.aten.slice_scatter.default(slice_2392, copy_124, 3, 0, 257);  slice_2392 = copy_124 = None
        slice_scatter_456 = torch.ops.aten.slice_scatter.default(permute_651, slice_scatter_454, 1, 0, 256);  permute_651 = slice_scatter_454 = None
        permute_652 = torch.ops.aten.permute.default(slice_scatter_456, [0, 2, 1, 3]);  slice_scatter_456 = None
        view_745 = torch.ops.aten.view.default(permute_652, [48, 4, 256, 513]);  permute_652 = None
        expand_41 = torch.ops.aten.expand.default(rev_41, [4, 256, 12, 257]);  rev_41 = None
        view_747 = torch.ops.aten.view.default(view_745, [4, 12, 1024, 513])
        permute_654 = torch.ops.aten.permute.default(view_747, [0, 2, 1, 3]);  view_747 = None
        slice_2403 = torch.ops.aten.slice.Tensor(permute_654, 1, -256, 9223372036854775807);  permute_654 = None
        slice_2405 = torch.ops.aten.slice.Tensor(slice_2403, 3, -257, 9223372036854775807);  slice_2403 = None
        full_default_103 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(expand_41, torch.bool);  expand_41 = None
        where_82 = torch.ops.aten.where.self(convert_element_type_51, full_default_103, slice_2405);  convert_element_type_51 = full_default_103 = slice_2405 = None
        view_748 = torch.ops.aten.view.default(view_745, [4, 12, 1024, 513])
        permute_655 = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        slice_2411 = torch.ops.aten.slice.Tensor(permute_655, 1, -256, 9223372036854775807);  permute_655 = None
        slice_2413 = torch.ops.aten.slice.Tensor(slice_2411, 3, -257, 9223372036854775807);  slice_2411 = None
        copy_125 = torch.ops.aten.copy.default(slice_2413, where_82);  slice_2413 = where_82 = None
        view_749 = torch.ops.aten.view.default(view_745, [4, 12, 1024, 513]);  view_745 = None
        permute_656 = torch.ops.aten.permute.default(view_749, [0, 2, 1, 3]);  view_749 = None
        slice_2415 = torch.ops.aten.slice.Tensor(permute_656, 1, -256, 9223372036854775807)
        slice_scatter_458 = torch.ops.aten.slice_scatter.default(slice_2415, copy_125, 3, -257, 9223372036854775807);  slice_2415 = copy_125 = None
        slice_scatter_460 = torch.ops.aten.slice_scatter.default(permute_656, slice_scatter_458, 1, -256, 9223372036854775807);  permute_656 = slice_scatter_458 = None
        permute_657 = torch.ops.aten.permute.default(slice_scatter_460, [0, 2, 1, 3]);  slice_scatter_460 = None
        view_750 = torch.ops.aten.view.default(permute_657, [48, 4, 256, 513]);  permute_657 = None
        ne_10 = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(ne_10, 2);  ne_10 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(unsqueeze_198, torch.float32)
        full_default_104 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_83 = torch.ops.aten.where.self(unsqueeze_198, full_default_104, convert_element_type_52);  unsqueeze_198 = full_default_104 = convert_element_type_52 = None
        full_94 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_659 = torch.ops.aten.permute.default(full_94, [0, 2, 1, 3]);  full_94 = None
        view_752 = torch.ops.aten.view.default(permute_659, [4, 1024, 1]);  permute_659 = None
        permute_660 = torch.ops.aten.permute.default(where_83, [0, 2, 1, 3]);  where_83 = None
        view_753 = torch.ops.aten.view.default(permute_660, [4, 1024, 1]);  permute_660 = None
        view_754 = torch.ops.aten.view.default(view_752, [4, 2, 512, 1]);  view_752 = None
        as_strided_63 = torch.ops.aten.as_strided.default(view_754, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_754 = None
        view_755 = torch.ops.aten.view.default(view_753, [4, 2, 512, 1]);  view_753 = None
        as_strided_64 = torch.ops.aten.as_strided.default(view_755, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_755 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(as_strided_63, 4);  as_strided_63 = None
        permute_661 = torch.ops.aten.permute.default(unsqueeze_199, [0, 1, 2, 4, 3]);  unsqueeze_199 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(as_strided_64, 4);  as_strided_64 = None
        permute_662 = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 2, 3]);  unsqueeze_200 = None
        mul_80 = torch.ops.aten.mul.Tensor(permute_661, permute_662);  permute_661 = permute_662 = None
        view_756 = torch.ops.aten.view.default(mul_80, [4, 3, 512, 512]);  mul_80 = None
        constant_pad_nd_41 = torch.ops.aten.constant_pad_nd.default(view_756, [0, 0, 0, 1], 0.0);  view_756 = None
        view_757 = torch.ops.aten.view.default(constant_pad_nd_41, [4, 3, 512, 513]);  constant_pad_nd_41 = None
        full_95 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2425 = torch.ops.aten.slice.Tensor(view_757, 2, 0, 256)
        slice_2426 = torch.ops.aten.slice.Tensor(slice_2425, 3, 0, 257);  slice_2425 = None
        slice_2428 = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_2430 = torch.ops.aten.slice.Tensor(slice_2428, 3, 256, 9223372036854775807);  slice_2428 = None
        copy_126 = torch.ops.aten.copy.default(slice_2430, slice_2426);  slice_2430 = slice_2426 = None
        slice_2432 = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_scatter_462 = torch.ops.aten.slice_scatter.default(slice_2432, copy_126, 3, 256, 9223372036854775807);  slice_2432 = copy_126 = None
        slice_scatter_464 = torch.ops.aten.slice_scatter.default(full_95, slice_scatter_462, 1, 0, -1);  full_95 = slice_scatter_462 = None
        select_210 = torch.ops.aten.select.int(view_757, 1, -1)
        slice_2439 = torch.ops.aten.slice.Tensor(select_210, 1, 256, 9223372036854775807);  select_210 = None
        slice_2440 = torch.ops.aten.slice.Tensor(slice_2439, 2, 0, 257);  slice_2439 = None
        select_212 = torch.ops.aten.select.int(slice_scatter_464, 1, -1)
        slice_2446 = torch.ops.aten.slice.Tensor(select_212, 2, 256, 9223372036854775807);  select_212 = None
        copy_127 = torch.ops.aten.copy.default(slice_2446, slice_2440);  slice_2446 = slice_2440 = None
        select_213 = torch.ops.aten.select.int(slice_scatter_464, 1, -1)
        slice_scatter_466 = torch.ops.aten.slice_scatter.default(select_213, copy_127, 2, 256, 9223372036854775807);  select_213 = copy_127 = None
        select_scatter_42 = torch.ops.aten.select_scatter.default(slice_scatter_464, slice_scatter_466, 1, -1);  slice_scatter_464 = slice_scatter_466 = None
        slice_2454 = torch.ops.aten.slice.Tensor(view_757, 2, -257, -1)
        slice_2455 = torch.ops.aten.slice.Tensor(slice_2454, 3, 257, 9223372036854775807);  slice_2454 = None
        slice_2461 = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_2463 = torch.ops.aten.slice.Tensor(slice_2461, 3, 0, 256);  slice_2461 = None
        copy_128 = torch.ops.aten.copy.default(slice_2463, slice_2455);  slice_2463 = slice_2455 = None
        slice_2465 = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_scatter_469 = torch.ops.aten.slice_scatter.default(slice_2465, copy_128, 3, 0, 256);  slice_2465 = copy_128 = None
        slice_scatter_471 = torch.ops.aten.slice_scatter.default(select_scatter_42, slice_scatter_469, 1, 1, 9223372036854775807);  select_scatter_42 = slice_scatter_469 = None
        select_215 = torch.ops.aten.select.int(view_757, 1, 0);  view_757 = None
        slice_2472 = torch.ops.aten.slice.Tensor(select_215, 1, 0, 255);  select_215 = None
        slice_2473 = torch.ops.aten.slice.Tensor(slice_2472, 2, -255, 9223372036854775807);  slice_2472 = None
        select_217 = torch.ops.aten.select.int(slice_scatter_471, 1, 0)
        slice_2478 = torch.ops.aten.slice.Tensor(select_217, 1, 1, 256);  select_217 = None
        slice_2479 = torch.ops.aten.slice.Tensor(slice_2478, 2, 1, 256);  slice_2478 = None
        copy_129 = torch.ops.aten.copy.default(slice_2479, slice_2473);  slice_2479 = slice_2473 = None
        select_218 = torch.ops.aten.select.int(slice_scatter_471, 1, 0)
        slice_2481 = torch.ops.aten.slice.Tensor(select_218, 1, 1, 256)
        slice_scatter_473 = torch.ops.aten.slice_scatter.default(slice_2481, copy_129, 2, 1, 256);  slice_2481 = copy_129 = None
        slice_scatter_474 = torch.ops.aten.slice_scatter.default(select_218, slice_scatter_473, 1, 1, 256);  select_218 = slice_scatter_473 = None
        select_scatter_43 = torch.ops.aten.select_scatter.default(slice_scatter_471, slice_scatter_474, 1, 0);  slice_scatter_471 = slice_scatter_474 = None
        full_default_105 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_42 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(iota_42, -2);  iota_42 = None
        iota_43 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        sub_83 = torch.ops.aten.sub.Tensor(unsqueeze_201, unsqueeze_202);  unsqueeze_201 = unsqueeze_202 = None
        le_21 = torch.ops.aten.le.Scalar(sub_83, 0);  sub_83 = None
        full_default_106 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_84 = torch.ops.aten.where.self(le_21, full_default_105, full_default_106);  le_21 = full_default_105 = full_default_106 = None
        rev_42 = torch.ops.prims.rev.default(where_84, [0]);  where_84 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(rev_42, 0);  rev_42 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        rev_43 = torch.ops.prims.rev.default(unsqueeze_204, [1, 3])
        expand_42 = torch.ops.aten.expand.default(unsqueeze_204, [4, 256, 1, 257]);  unsqueeze_204 = None
        view_760 = torch.ops.aten.view.default(select_scatter_43, [4, 1, 1024, 513])
        permute_665 = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
        slice_2492 = torch.ops.aten.slice.Tensor(permute_665, 1, 0, 256);  permute_665 = None
        slice_2494 = torch.ops.aten.slice.Tensor(slice_2492, 3, 0, 257);  slice_2492 = None
        full_default_107 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(expand_42, torch.bool);  expand_42 = None
        where_85 = torch.ops.aten.where.self(convert_element_type_53, full_default_107, slice_2494);  convert_element_type_53 = full_default_107 = slice_2494 = None
        view_761 = torch.ops.aten.view.default(select_scatter_43, [4, 1, 1024, 513])
        permute_666 = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
        slice_2500 = torch.ops.aten.slice.Tensor(permute_666, 1, 0, 256);  permute_666 = None
        slice_2502 = torch.ops.aten.slice.Tensor(slice_2500, 3, 0, 257);  slice_2500 = None
        copy_130 = torch.ops.aten.copy.default(slice_2502, where_85);  slice_2502 = where_85 = None
        view_762 = torch.ops.aten.view.default(select_scatter_43, [4, 1, 1024, 513]);  select_scatter_43 = None
        permute_667 = torch.ops.aten.permute.default(view_762, [0, 2, 1, 3]);  view_762 = None
        slice_2504 = torch.ops.aten.slice.Tensor(permute_667, 1, 0, 256)
        slice_scatter_476 = torch.ops.aten.slice_scatter.default(slice_2504, copy_130, 3, 0, 257);  slice_2504 = copy_130 = None
        slice_scatter_478 = torch.ops.aten.slice_scatter.default(permute_667, slice_scatter_476, 1, 0, 256);  permute_667 = slice_scatter_476 = None
        permute_668 = torch.ops.aten.permute.default(slice_scatter_478, [0, 2, 1, 3]);  slice_scatter_478 = None
        view_763 = torch.ops.aten.view.default(permute_668, [4, 4, 256, 513]);  permute_668 = None
        expand_43 = torch.ops.aten.expand.default(rev_43, [4, 256, 1, 257]);  rev_43 = None
        view_765 = torch.ops.aten.view.default(view_763, [4, 1, 1024, 513])
        permute_670 = torch.ops.aten.permute.default(view_765, [0, 2, 1, 3]);  view_765 = None
        slice_2515 = torch.ops.aten.slice.Tensor(permute_670, 1, -256, 9223372036854775807);  permute_670 = None
        slice_2517 = torch.ops.aten.slice.Tensor(slice_2515, 3, -257, 9223372036854775807);  slice_2515 = None
        full_default_108 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(expand_43, torch.bool);  expand_43 = None
        where_86 = torch.ops.aten.where.self(convert_element_type_54, full_default_108, slice_2517);  convert_element_type_54 = full_default_108 = slice_2517 = None
        view_766 = torch.ops.aten.view.default(view_763, [4, 1, 1024, 513])
        permute_671 = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
        slice_2523 = torch.ops.aten.slice.Tensor(permute_671, 1, -256, 9223372036854775807);  permute_671 = None
        slice_2525 = torch.ops.aten.slice.Tensor(slice_2523, 3, -257, 9223372036854775807);  slice_2523 = None
        copy_131 = torch.ops.aten.copy.default(slice_2525, where_86);  slice_2525 = where_86 = None
        view_767 = torch.ops.aten.view.default(view_763, [4, 1, 1024, 513]);  view_763 = None
        permute_672 = torch.ops.aten.permute.default(view_767, [0, 2, 1, 3]);  view_767 = None
        slice_2527 = torch.ops.aten.slice.Tensor(permute_672, 1, -256, 9223372036854775807)
        slice_scatter_480 = torch.ops.aten.slice_scatter.default(slice_2527, copy_131, 3, -257, 9223372036854775807);  slice_2527 = copy_131 = None
        slice_scatter_482 = torch.ops.aten.slice_scatter.default(permute_672, slice_scatter_480, 1, -256, 9223372036854775807);  permute_672 = slice_scatter_480 = None
        permute_673 = torch.ops.aten.permute.default(slice_scatter_482, [0, 2, 1, 3]);  slice_scatter_482 = None
        view_768 = torch.ops.aten.view.default(permute_673, [4, 4, 256, 513]);  permute_673 = None
        view_770 = torch.ops.aten.view.default(view_750, [4, 12, 1024, 513]);  view_750 = None
        permute_675 = torch.ops.aten.permute.default(view_770, [0, 2, 1, 3]);  view_770 = None
        view_771 = torch.ops.aten.view.default(view_768, [4, 1, 1024, 513]);  view_768 = None
        permute_676 = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None
        add_155 = torch.ops.aten.add.Tensor(permute_675, permute_676);  permute_675 = permute_676 = None
        permute_677 = torch.ops.aten.permute.default(add_155, [0, 2, 1, 3]);  add_155 = None
        view_773 = torch.ops.aten.view.default(permute_677, [48, 4, 256, 513]);  permute_677 = None
        view_774 = torch.ops.aten.view.default(view_773, [4, 12, 1024, 513]);  view_773 = None
        permute_678 = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
        clone_135 = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
        amax_10 = torch.ops.aten.amax.default(clone_135, [-1], True)
        sub_84 = torch.ops.aten.sub.Tensor(clone_135, amax_10);  clone_135 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_107 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
        full_default_109 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_87 = torch.ops.aten.where.self(unsqueeze_206, full_default_109, div_107);  unsqueeze_206 = full_default_109 = div_107 = None
        view_775 = torch.ops.aten.view.default(add_152, [1024, 4, 12, 64]);  add_152 = None
        permute_679 = torch.ops.aten.permute.default(view_775, [1, 0, 2, 3]);  view_775 = None
        permute_680 = torch.ops.aten.permute.default(where_87, [0, 2, 1, 3]);  where_87 = None
        clone_137 = torch.ops.aten.clone.default(permute_680, memory_format = torch.contiguous_format);  permute_680 = None
        view_776 = torch.ops.aten.view.default(clone_137, [48, 4, 256, 513]);  clone_137 = None
        permute_681 = torch.ops.aten.permute.default(permute_679, [0, 2, 1, 3]);  permute_679 = None
        view_777 = torch.ops.aten.view.default(permute_681, [48, 1024, 64]);  permute_681 = None
        constant_pad_nd_42 = torch.ops.aten.constant_pad_nd.default(view_777, [0, 0, 256, 256], -1.0);  view_777 = None
        as_strided_65 = torch.ops.aten.as_strided.default(constant_pad_nd_42, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_42 = None
        constant_pad_nd_43 = torch.ops.aten.constant_pad_nd.default(view_776, [0, 257], 0.0);  view_776 = None
        view_778 = torch.ops.aten.view.default(constant_pad_nd_43, [48, 4, -1]);  constant_pad_nd_43 = None
        slice_2537 = torch.ops.aten.slice.Tensor(view_778, 2, 0, -256);  view_778 = None
        view_779 = torch.ops.aten.view.default(slice_2537, [48, 4, 256, 769]);  slice_2537 = None
        slice_2541 = torch.ops.aten.slice.Tensor(view_779, 3, 0, -1);  view_779 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(slice_2541, 4);  slice_2541 = None
        permute_682 = torch.ops.aten.permute.default(unsqueeze_207, [0, 1, 2, 4, 3]);  unsqueeze_207 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(as_strided_65, 4);  as_strided_65 = None
        permute_683 = torch.ops.aten.permute.default(unsqueeze_208, [0, 1, 4, 3, 2]);  unsqueeze_208 = None
        permute_684 = torch.ops.aten.permute.default(permute_682, [0, 1, 2, 4, 3]);  permute_682 = None
        view_780 = torch.ops.aten.view.default(permute_684, [192, 256, 768]);  permute_684 = None
        permute_685 = torch.ops.aten.permute.default(permute_683, [0, 1, 4, 3, 2]);  permute_683 = None
        clone_138 = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
        view_781 = torch.ops.aten.view.default(clone_138, [192, 768, 64]);  clone_138 = None
        bmm_21 = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782 = torch.ops.aten.view.default(bmm_21, [48, 4, 256, 1, 64]);  bmm_21 = None
        permute_686 = torch.ops.aten.permute.default(view_782, [0, 1, 2, 4, 3]);  view_782 = None
        view_783 = torch.ops.aten.view.default(permute_686, [48, 4, 256, 64]);  permute_686 = None
        view_784 = torch.ops.aten.view.default(view_783, [4, 12, 1024, 64]);  view_783 = None
        permute_687 = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
        permute_688 = torch.ops.aten.permute.default(permute_687, [1, 0, 2, 3]);  permute_687 = None
        clone_139 = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
        view_785 = torch.ops.aten.view.default(clone_139, [1024, 4, 768]);  clone_139 = None
        permute_689 = torch.ops.aten.permute.default(view_785, [1, 0, 2]);  view_785 = None
        permute_690 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        clone_140 = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
        view_786 = torch.ops.aten.view.default(clone_140, [4096, 768]);  clone_140 = None
        mm_43 = torch.ops.aten.mm.default(view_786, permute_690);  view_786 = permute_690 = None
        view_787 = torch.ops.aten.view.default(mm_43, [4, 1024, 768]);  mm_43 = None
        add_157 = torch.ops.aten.add.Tensor(view_787, arg170_1);  view_787 = arg170_1 = None
        add_158 = torch.ops.aten.add.Tensor(add_157, add_149);  add_157 = add_149 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_86 = torch.ops.aten.sub.Tensor(add_158, getitem_41);  add_158 = getitem_41 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = rsqrt_20 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, arg171_1);  mul_81 = arg171_1 = None
        add_160 = torch.ops.aten.add.Tensor(mul_82, arg172_1);  mul_82 = arg172_1 = None
        view_788 = torch.ops.aten.view.default(add_160, [4096, 768])
        permute_691 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg174_1, view_788, permute_691);  arg174_1 = view_788 = permute_691 = None
        view_789 = torch.ops.aten.view.default(addmm_20, [4, 1024, 3072]);  addmm_20 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_789, 0.5)
        mul_84 = torch.ops.aten.mul.Tensor(view_789, 0.7071067811865476);  view_789 = None
        erf_10 = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_161 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_83, add_161);  mul_83 = add_161 = None
        view_790 = torch.ops.aten.view.default(mul_85, [4096, 3072]);  mul_85 = None
        permute_692 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg176_1, view_790, permute_692);  arg176_1 = view_790 = permute_692 = None
        view_791 = torch.ops.aten.view.default(addmm_21, [4, 1024, 768]);  addmm_21 = None
        add_162 = torch.ops.aten.add.Tensor(view_791, add_160);  view_791 = add_160 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_163 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_87 = torch.ops.aten.sub.Tensor(add_162, getitem_43);  add_162 = getitem_43 = None
        mul_86 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = rsqrt_21 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_86, arg177_1);  mul_86 = arg177_1 = None
        add_164 = torch.ops.aten.add.Tensor(mul_87, arg178_1);  mul_87 = arg178_1 = None
        permute_693 = torch.ops.aten.permute.default(add_164, [1, 0, 2])
        permute_694 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        clone_143 = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format)
        view_792 = torch.ops.aten.view.default(clone_143, [4096, 768]);  clone_143 = None
        mm_44 = torch.ops.aten.mm.default(view_792, permute_694);  view_792 = permute_694 = None
        view_793 = torch.ops.aten.view.default(mm_44, [1024, 4, 768]);  mm_44 = None
        add_165 = torch.ops.aten.add.Tensor(view_793, arg180_1);  view_793 = arg180_1 = None
        permute_695 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        clone_144 = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format)
        view_794 = torch.ops.aten.view.default(clone_144, [4096, 768]);  clone_144 = None
        mm_45 = torch.ops.aten.mm.default(view_794, permute_695);  view_794 = permute_695 = None
        view_795 = torch.ops.aten.view.default(mm_45, [1024, 4, 768]);  mm_45 = None
        add_166 = torch.ops.aten.add.Tensor(view_795, arg182_1);  view_795 = arg182_1 = None
        permute_696 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        clone_145 = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format);  permute_693 = None
        view_796 = torch.ops.aten.view.default(clone_145, [4096, 768]);  clone_145 = None
        mm_46 = torch.ops.aten.mm.default(view_796, permute_696);  view_796 = permute_696 = None
        view_797 = torch.ops.aten.view.default(mm_46, [1024, 4, 768]);  mm_46 = None
        add_167 = torch.ops.aten.add.Tensor(view_797, arg184_1);  view_797 = arg184_1 = None
        div_110 = torch.ops.aten.div.Tensor(add_165, 8.0);  add_165 = None
        view_799 = torch.ops.aten.view.default(add_166, [1024, 4, 12, 64]);  add_166 = None
        permute_698 = torch.ops.aten.permute.default(view_799, [1, 0, 2, 3]);  view_799 = None
        permute_700 = torch.ops.aten.permute.default(permute_698, [0, 2, 1, 3]);  permute_698 = None
        view_801 = torch.ops.aten.view.default(permute_700, [48, 1024, 64]);  permute_700 = None
        view_803 = torch.ops.aten.view.default(view_801, [48, 2, 512, 64]);  view_801 = None
        as_strided_67 = torch.ops.aten.as_strided.default(view_803, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_803 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(as_strided_67, 4);  as_strided_67 = None
        permute_702 = torch.ops.aten.permute.default(unsqueeze_210, [0, 1, 4, 2, 3]);  unsqueeze_210 = None
        view_804 = torch.ops.aten.view.default(div_110, [1024, 4, 12, 64]);  div_110 = None
        permute_704 = torch.ops.aten.permute.default(view_804, [1, 0, 2, 3]);  view_804 = None
        permute_705 = torch.ops.aten.permute.default(permute_704, [0, 2, 1, 3]);  permute_704 = None
        view_805 = torch.ops.aten.view.default(permute_705, [48, 1024, 64]);  permute_705 = None
        view_806 = torch.ops.aten.view.default(view_805, [48, 2, 512, 64]);  view_805 = None
        as_strided_68 = torch.ops.aten.as_strided.default(view_806, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_806 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
        permute_706 = torch.ops.aten.permute.default(unsqueeze_211, [0, 1, 2, 4, 3]);  unsqueeze_211 = None
        permute_707 = torch.ops.aten.permute.default(permute_706, [0, 1, 2, 4, 3]);  permute_706 = None
        clone_146 = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
        view_807 = torch.ops.aten.view.default(clone_146, [144, 512, 64]);  clone_146 = None
        permute_708 = torch.ops.aten.permute.default(permute_702, [0, 1, 4, 3, 2]);  permute_702 = None
        clone_147 = torch.ops.aten.clone.default(permute_708, memory_format = torch.contiguous_format);  permute_708 = None
        view_808 = torch.ops.aten.view.default(clone_147, [144, 64, 512]);  clone_147 = None
        bmm_22 = torch.ops.aten.bmm.default(view_807, view_808);  view_807 = view_808 = None
        view_809 = torch.ops.aten.view.default(bmm_22, [48, 3, 512, 1, 512]);  bmm_22 = None
        permute_709 = torch.ops.aten.permute.default(view_809, [0, 1, 2, 4, 3]);  view_809 = None
        view_810 = torch.ops.aten.view.default(permute_709, [48, 3, 512, 512]);  permute_709 = None
        constant_pad_nd_44 = torch.ops.aten.constant_pad_nd.default(view_810, [0, 0, 0, 1], 0.0);  view_810 = None
        view_811 = torch.ops.aten.view.default(constant_pad_nd_44, [48, 3, 512, 513]);  constant_pad_nd_44 = None
        full_99 = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2544 = torch.ops.aten.slice.Tensor(view_811, 2, 0, 256)
        slice_2545 = torch.ops.aten.slice.Tensor(slice_2544, 3, 0, 257);  slice_2544 = None
        slice_2547 = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_2549 = torch.ops.aten.slice.Tensor(slice_2547, 3, 256, 9223372036854775807);  slice_2547 = None
        copy_132 = torch.ops.aten.copy.default(slice_2549, slice_2545);  slice_2549 = slice_2545 = None
        slice_2551 = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_scatter_484 = torch.ops.aten.slice_scatter.default(slice_2551, copy_132, 3, 256, 9223372036854775807);  slice_2551 = copy_132 = None
        slice_scatter_486 = torch.ops.aten.slice_scatter.default(full_99, slice_scatter_484, 1, 0, -1);  full_99 = slice_scatter_484 = None
        select_220 = torch.ops.aten.select.int(view_811, 1, -1)
        slice_2558 = torch.ops.aten.slice.Tensor(select_220, 1, 256, 9223372036854775807);  select_220 = None
        slice_2559 = torch.ops.aten.slice.Tensor(slice_2558, 2, 0, 257);  slice_2558 = None
        select_222 = torch.ops.aten.select.int(slice_scatter_486, 1, -1)
        slice_2565 = torch.ops.aten.slice.Tensor(select_222, 2, 256, 9223372036854775807);  select_222 = None
        copy_133 = torch.ops.aten.copy.default(slice_2565, slice_2559);  slice_2565 = slice_2559 = None
        select_223 = torch.ops.aten.select.int(slice_scatter_486, 1, -1)
        slice_scatter_488 = torch.ops.aten.slice_scatter.default(select_223, copy_133, 2, 256, 9223372036854775807);  select_223 = copy_133 = None
        select_scatter_44 = torch.ops.aten.select_scatter.default(slice_scatter_486, slice_scatter_488, 1, -1);  slice_scatter_486 = slice_scatter_488 = None
        slice_2573 = torch.ops.aten.slice.Tensor(view_811, 2, -257, -1)
        slice_2574 = torch.ops.aten.slice.Tensor(slice_2573, 3, 257, 9223372036854775807);  slice_2573 = None
        slice_2580 = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_2582 = torch.ops.aten.slice.Tensor(slice_2580, 3, 0, 256);  slice_2580 = None
        copy_134 = torch.ops.aten.copy.default(slice_2582, slice_2574);  slice_2582 = slice_2574 = None
        slice_2584 = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_scatter_491 = torch.ops.aten.slice_scatter.default(slice_2584, copy_134, 3, 0, 256);  slice_2584 = copy_134 = None
        slice_scatter_493 = torch.ops.aten.slice_scatter.default(select_scatter_44, slice_scatter_491, 1, 1, 9223372036854775807);  select_scatter_44 = slice_scatter_491 = None
        select_225 = torch.ops.aten.select.int(view_811, 1, 0);  view_811 = None
        slice_2591 = torch.ops.aten.slice.Tensor(select_225, 1, 0, 255);  select_225 = None
        slice_2592 = torch.ops.aten.slice.Tensor(slice_2591, 2, -255, 9223372036854775807);  slice_2591 = None
        select_227 = torch.ops.aten.select.int(slice_scatter_493, 1, 0)
        slice_2597 = torch.ops.aten.slice.Tensor(select_227, 1, 1, 256);  select_227 = None
        slice_2598 = torch.ops.aten.slice.Tensor(slice_2597, 2, 1, 256);  slice_2597 = None
        copy_135 = torch.ops.aten.copy.default(slice_2598, slice_2592);  slice_2598 = slice_2592 = None
        select_228 = torch.ops.aten.select.int(slice_scatter_493, 1, 0)
        slice_2600 = torch.ops.aten.slice.Tensor(select_228, 1, 1, 256)
        slice_scatter_495 = torch.ops.aten.slice_scatter.default(slice_2600, copy_135, 2, 1, 256);  slice_2600 = copy_135 = None
        slice_scatter_496 = torch.ops.aten.slice_scatter.default(select_228, slice_scatter_495, 1, 1, 256);  select_228 = slice_scatter_495 = None
        select_scatter_45 = torch.ops.aten.select_scatter.default(slice_scatter_493, slice_scatter_496, 1, 0);  slice_scatter_493 = slice_scatter_496 = None
        full_default_110 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_44 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(iota_44, -2);  iota_44 = None
        iota_45 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        sub_89 = torch.ops.aten.sub.Tensor(unsqueeze_212, unsqueeze_213);  unsqueeze_212 = unsqueeze_213 = None
        le_22 = torch.ops.aten.le.Scalar(sub_89, 0);  sub_89 = None
        full_default_111 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_88 = torch.ops.aten.where.self(le_22, full_default_110, full_default_111);  le_22 = full_default_110 = full_default_111 = None
        rev_44 = torch.ops.prims.rev.default(where_88, [0]);  where_88 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(rev_44, 0);  rev_44 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        rev_45 = torch.ops.prims.rev.default(unsqueeze_215, [1, 3])
        expand_44 = torch.ops.aten.expand.default(unsqueeze_215, [4, 256, 12, 257]);  unsqueeze_215 = None
        view_814 = torch.ops.aten.view.default(select_scatter_45, [4, 12, 1024, 513])
        permute_712 = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None
        slice_2611 = torch.ops.aten.slice.Tensor(permute_712, 1, 0, 256);  permute_712 = None
        slice_2613 = torch.ops.aten.slice.Tensor(slice_2611, 3, 0, 257);  slice_2611 = None
        full_default_112 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(expand_44, torch.bool);  expand_44 = None
        where_89 = torch.ops.aten.where.self(convert_element_type_55, full_default_112, slice_2613);  convert_element_type_55 = full_default_112 = slice_2613 = None
        view_815 = torch.ops.aten.view.default(select_scatter_45, [4, 12, 1024, 513])
        permute_713 = torch.ops.aten.permute.default(view_815, [0, 2, 1, 3]);  view_815 = None
        slice_2619 = torch.ops.aten.slice.Tensor(permute_713, 1, 0, 256);  permute_713 = None
        slice_2621 = torch.ops.aten.slice.Tensor(slice_2619, 3, 0, 257);  slice_2619 = None
        copy_136 = torch.ops.aten.copy.default(slice_2621, where_89);  slice_2621 = where_89 = None
        view_816 = torch.ops.aten.view.default(select_scatter_45, [4, 12, 1024, 513]);  select_scatter_45 = None
        permute_714 = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None
        slice_2623 = torch.ops.aten.slice.Tensor(permute_714, 1, 0, 256)
        slice_scatter_498 = torch.ops.aten.slice_scatter.default(slice_2623, copy_136, 3, 0, 257);  slice_2623 = copy_136 = None
        slice_scatter_500 = torch.ops.aten.slice_scatter.default(permute_714, slice_scatter_498, 1, 0, 256);  permute_714 = slice_scatter_498 = None
        permute_715 = torch.ops.aten.permute.default(slice_scatter_500, [0, 2, 1, 3]);  slice_scatter_500 = None
        view_817 = torch.ops.aten.view.default(permute_715, [48, 4, 256, 513]);  permute_715 = None
        expand_45 = torch.ops.aten.expand.default(rev_45, [4, 256, 12, 257]);  rev_45 = None
        view_819 = torch.ops.aten.view.default(view_817, [4, 12, 1024, 513])
        permute_717 = torch.ops.aten.permute.default(view_819, [0, 2, 1, 3]);  view_819 = None
        slice_2634 = torch.ops.aten.slice.Tensor(permute_717, 1, -256, 9223372036854775807);  permute_717 = None
        slice_2636 = torch.ops.aten.slice.Tensor(slice_2634, 3, -257, 9223372036854775807);  slice_2634 = None
        full_default_113 = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(expand_45, torch.bool);  expand_45 = None
        where_90 = torch.ops.aten.where.self(convert_element_type_56, full_default_113, slice_2636);  convert_element_type_56 = full_default_113 = slice_2636 = None
        view_820 = torch.ops.aten.view.default(view_817, [4, 12, 1024, 513])
        permute_718 = torch.ops.aten.permute.default(view_820, [0, 2, 1, 3]);  view_820 = None
        slice_2642 = torch.ops.aten.slice.Tensor(permute_718, 1, -256, 9223372036854775807);  permute_718 = None
        slice_2644 = torch.ops.aten.slice.Tensor(slice_2642, 3, -257, 9223372036854775807);  slice_2642 = None
        copy_137 = torch.ops.aten.copy.default(slice_2644, where_90);  slice_2644 = where_90 = None
        view_821 = torch.ops.aten.view.default(view_817, [4, 12, 1024, 513]);  view_817 = None
        permute_719 = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None
        slice_2646 = torch.ops.aten.slice.Tensor(permute_719, 1, -256, 9223372036854775807)
        slice_scatter_502 = torch.ops.aten.slice_scatter.default(slice_2646, copy_137, 3, -257, 9223372036854775807);  slice_2646 = copy_137 = None
        slice_scatter_504 = torch.ops.aten.slice_scatter.default(permute_719, slice_scatter_502, 1, -256, 9223372036854775807);  permute_719 = slice_scatter_502 = None
        permute_720 = torch.ops.aten.permute.default(slice_scatter_504, [0, 2, 1, 3]);  slice_scatter_504 = None
        view_822 = torch.ops.aten.view.default(permute_720, [48, 4, 256, 513]);  permute_720 = None
        ne_11 = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(ne_11, 2);  ne_11 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(unsqueeze_217, torch.float32)
        full_default_114 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_91 = torch.ops.aten.where.self(unsqueeze_217, full_default_114, convert_element_type_57);  unsqueeze_217 = full_default_114 = convert_element_type_57 = None
        full_103 = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_722 = torch.ops.aten.permute.default(full_103, [0, 2, 1, 3]);  full_103 = None
        view_824 = torch.ops.aten.view.default(permute_722, [4, 1024, 1]);  permute_722 = None
        permute_723 = torch.ops.aten.permute.default(where_91, [0, 2, 1, 3]);  where_91 = None
        view_825 = torch.ops.aten.view.default(permute_723, [4, 1024, 1]);  permute_723 = None
        view_826 = torch.ops.aten.view.default(view_824, [4, 2, 512, 1]);  view_824 = None
        as_strided_69 = torch.ops.aten.as_strided.default(view_826, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_826 = None
        view_827 = torch.ops.aten.view.default(view_825, [4, 2, 512, 1]);  view_825 = None
        as_strided_70 = torch.ops.aten.as_strided.default(view_827, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_827 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(as_strided_69, 4);  as_strided_69 = None
        permute_724 = torch.ops.aten.permute.default(unsqueeze_218, [0, 1, 2, 4, 3]);  unsqueeze_218 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(as_strided_70, 4);  as_strided_70 = None
        permute_725 = torch.ops.aten.permute.default(unsqueeze_219, [0, 1, 4, 2, 3]);  unsqueeze_219 = None
        mul_88 = torch.ops.aten.mul.Tensor(permute_724, permute_725);  permute_724 = permute_725 = None
        view_828 = torch.ops.aten.view.default(mul_88, [4, 3, 512, 512]);  mul_88 = None
        constant_pad_nd_45 = torch.ops.aten.constant_pad_nd.default(view_828, [0, 0, 0, 1], 0.0);  view_828 = None
        view_829 = torch.ops.aten.view.default(constant_pad_nd_45, [4, 3, 512, 513]);  constant_pad_nd_45 = None
        full_104 = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2656 = torch.ops.aten.slice.Tensor(view_829, 2, 0, 256)
        slice_2657 = torch.ops.aten.slice.Tensor(slice_2656, 3, 0, 257);  slice_2656 = None
        slice_2659 = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_2661 = torch.ops.aten.slice.Tensor(slice_2659, 3, 256, 9223372036854775807);  slice_2659 = None
        copy_138 = torch.ops.aten.copy.default(slice_2661, slice_2657);  slice_2661 = slice_2657 = None
        slice_2663 = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_scatter_506 = torch.ops.aten.slice_scatter.default(slice_2663, copy_138, 3, 256, 9223372036854775807);  slice_2663 = copy_138 = None
        slice_scatter_508 = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_506, 1, 0, -1);  full_104 = slice_scatter_506 = None
        select_230 = torch.ops.aten.select.int(view_829, 1, -1)
        slice_2670 = torch.ops.aten.slice.Tensor(select_230, 1, 256, 9223372036854775807);  select_230 = None
        slice_2671 = torch.ops.aten.slice.Tensor(slice_2670, 2, 0, 257);  slice_2670 = None
        select_232 = torch.ops.aten.select.int(slice_scatter_508, 1, -1)
        slice_2677 = torch.ops.aten.slice.Tensor(select_232, 2, 256, 9223372036854775807);  select_232 = None
        copy_139 = torch.ops.aten.copy.default(slice_2677, slice_2671);  slice_2677 = slice_2671 = None
        select_233 = torch.ops.aten.select.int(slice_scatter_508, 1, -1)
        slice_scatter_510 = torch.ops.aten.slice_scatter.default(select_233, copy_139, 2, 256, 9223372036854775807);  select_233 = copy_139 = None
        select_scatter_46 = torch.ops.aten.select_scatter.default(slice_scatter_508, slice_scatter_510, 1, -1);  slice_scatter_508 = slice_scatter_510 = None
        slice_2685 = torch.ops.aten.slice.Tensor(view_829, 2, -257, -1)
        slice_2686 = torch.ops.aten.slice.Tensor(slice_2685, 3, 257, 9223372036854775807);  slice_2685 = None
        slice_2692 = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_2694 = torch.ops.aten.slice.Tensor(slice_2692, 3, 0, 256);  slice_2692 = None
        copy_140 = torch.ops.aten.copy.default(slice_2694, slice_2686);  slice_2694 = slice_2686 = None
        slice_2696 = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_scatter_513 = torch.ops.aten.slice_scatter.default(slice_2696, copy_140, 3, 0, 256);  slice_2696 = copy_140 = None
        slice_scatter_515 = torch.ops.aten.slice_scatter.default(select_scatter_46, slice_scatter_513, 1, 1, 9223372036854775807);  select_scatter_46 = slice_scatter_513 = None
        select_235 = torch.ops.aten.select.int(view_829, 1, 0);  view_829 = None
        slice_2703 = torch.ops.aten.slice.Tensor(select_235, 1, 0, 255);  select_235 = None
        slice_2704 = torch.ops.aten.slice.Tensor(slice_2703, 2, -255, 9223372036854775807);  slice_2703 = None
        select_237 = torch.ops.aten.select.int(slice_scatter_515, 1, 0)
        slice_2709 = torch.ops.aten.slice.Tensor(select_237, 1, 1, 256);  select_237 = None
        slice_2710 = torch.ops.aten.slice.Tensor(slice_2709, 2, 1, 256);  slice_2709 = None
        copy_141 = torch.ops.aten.copy.default(slice_2710, slice_2704);  slice_2710 = slice_2704 = None
        select_238 = torch.ops.aten.select.int(slice_scatter_515, 1, 0)
        slice_2712 = torch.ops.aten.slice.Tensor(select_238, 1, 1, 256)
        slice_scatter_517 = torch.ops.aten.slice_scatter.default(slice_2712, copy_141, 2, 1, 256);  slice_2712 = copy_141 = None
        slice_scatter_518 = torch.ops.aten.slice_scatter.default(select_238, slice_scatter_517, 1, 1, 256);  select_238 = slice_scatter_517 = None
        select_scatter_47 = torch.ops.aten.select_scatter.default(slice_scatter_515, slice_scatter_518, 1, 0);  slice_scatter_515 = slice_scatter_518 = None
        full_default_115 = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_46 = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(iota_46, -2);  iota_46 = None
        iota_47 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        sub_91 = torch.ops.aten.sub.Tensor(unsqueeze_220, unsqueeze_221);  unsqueeze_220 = unsqueeze_221 = None
        le_23 = torch.ops.aten.le.Scalar(sub_91, 0);  sub_91 = None
        full_default_116 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_92 = torch.ops.aten.where.self(le_23, full_default_115, full_default_116);  le_23 = full_default_115 = full_default_116 = None
        rev_46 = torch.ops.prims.rev.default(where_92, [0]);  where_92 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(rev_46, 0);  rev_46 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
        rev_47 = torch.ops.prims.rev.default(unsqueeze_223, [1, 3])
        expand_46 = torch.ops.aten.expand.default(unsqueeze_223, [4, 256, 1, 257]);  unsqueeze_223 = None
        view_832 = torch.ops.aten.view.default(select_scatter_47, [4, 1, 1024, 513])
        permute_728 = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None
        slice_2723 = torch.ops.aten.slice.Tensor(permute_728, 1, 0, 256);  permute_728 = None
        slice_2725 = torch.ops.aten.slice.Tensor(slice_2723, 3, 0, 257);  slice_2723 = None
        full_default_117 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(expand_46, torch.bool);  expand_46 = None
        where_93 = torch.ops.aten.where.self(convert_element_type_58, full_default_117, slice_2725);  convert_element_type_58 = full_default_117 = slice_2725 = None
        view_833 = torch.ops.aten.view.default(select_scatter_47, [4, 1, 1024, 513])
        permute_729 = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
        slice_2731 = torch.ops.aten.slice.Tensor(permute_729, 1, 0, 256);  permute_729 = None
        slice_2733 = torch.ops.aten.slice.Tensor(slice_2731, 3, 0, 257);  slice_2731 = None
        copy_142 = torch.ops.aten.copy.default(slice_2733, where_93);  slice_2733 = where_93 = None
        view_834 = torch.ops.aten.view.default(select_scatter_47, [4, 1, 1024, 513]);  select_scatter_47 = None
        permute_730 = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None
        slice_2735 = torch.ops.aten.slice.Tensor(permute_730, 1, 0, 256)
        slice_scatter_520 = torch.ops.aten.slice_scatter.default(slice_2735, copy_142, 3, 0, 257);  slice_2735 = copy_142 = None
        slice_scatter_522 = torch.ops.aten.slice_scatter.default(permute_730, slice_scatter_520, 1, 0, 256);  permute_730 = slice_scatter_520 = None
        permute_731 = torch.ops.aten.permute.default(slice_scatter_522, [0, 2, 1, 3]);  slice_scatter_522 = None
        view_835 = torch.ops.aten.view.default(permute_731, [4, 4, 256, 513]);  permute_731 = None
        expand_47 = torch.ops.aten.expand.default(rev_47, [4, 256, 1, 257]);  rev_47 = None
        view_837 = torch.ops.aten.view.default(view_835, [4, 1, 1024, 513])
        permute_733 = torch.ops.aten.permute.default(view_837, [0, 2, 1, 3]);  view_837 = None
        slice_2746 = torch.ops.aten.slice.Tensor(permute_733, 1, -256, 9223372036854775807);  permute_733 = None
        slice_2748 = torch.ops.aten.slice.Tensor(slice_2746, 3, -257, 9223372036854775807);  slice_2746 = None
        full_default_118 = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(expand_47, torch.bool);  expand_47 = None
        where_94 = torch.ops.aten.where.self(convert_element_type_59, full_default_118, slice_2748);  convert_element_type_59 = full_default_118 = slice_2748 = None
        view_838 = torch.ops.aten.view.default(view_835, [4, 1, 1024, 513])
        permute_734 = torch.ops.aten.permute.default(view_838, [0, 2, 1, 3]);  view_838 = None
        slice_2754 = torch.ops.aten.slice.Tensor(permute_734, 1, -256, 9223372036854775807);  permute_734 = None
        slice_2756 = torch.ops.aten.slice.Tensor(slice_2754, 3, -257, 9223372036854775807);  slice_2754 = None
        copy_143 = torch.ops.aten.copy.default(slice_2756, where_94);  slice_2756 = where_94 = None
        view_839 = torch.ops.aten.view.default(view_835, [4, 1, 1024, 513]);  view_835 = None
        permute_735 = torch.ops.aten.permute.default(view_839, [0, 2, 1, 3]);  view_839 = None
        slice_2758 = torch.ops.aten.slice.Tensor(permute_735, 1, -256, 9223372036854775807)
        slice_scatter_524 = torch.ops.aten.slice_scatter.default(slice_2758, copy_143, 3, -257, 9223372036854775807);  slice_2758 = copy_143 = None
        slice_scatter_526 = torch.ops.aten.slice_scatter.default(permute_735, slice_scatter_524, 1, -256, 9223372036854775807);  permute_735 = slice_scatter_524 = None
        permute_736 = torch.ops.aten.permute.default(slice_scatter_526, [0, 2, 1, 3]);  slice_scatter_526 = None
        view_840 = torch.ops.aten.view.default(permute_736, [4, 4, 256, 513]);  permute_736 = None
        view_842 = torch.ops.aten.view.default(view_822, [4, 12, 1024, 513]);  view_822 = None
        permute_738 = torch.ops.aten.permute.default(view_842, [0, 2, 1, 3]);  view_842 = None
        view_843 = torch.ops.aten.view.default(view_840, [4, 1, 1024, 513]);  view_840 = None
        permute_739 = torch.ops.aten.permute.default(view_843, [0, 2, 1, 3]);  view_843 = None
        add_170 = torch.ops.aten.add.Tensor(permute_738, permute_739);  permute_738 = permute_739 = None
        permute_740 = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
        view_845 = torch.ops.aten.view.default(permute_740, [48, 4, 256, 513]);  permute_740 = None
        view_846 = torch.ops.aten.view.default(view_845, [4, 12, 1024, 513]);  view_845 = None
        permute_741 = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None
        clone_148 = torch.ops.aten.clone.default(permute_741, memory_format = torch.contiguous_format);  permute_741 = None
        amax_11 = torch.ops.aten.amax.default(clone_148, [-1], True)
        sub_92 = torch.ops.aten.sub.Tensor(clone_148, amax_11);  clone_148 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_92);  sub_92 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_117 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        full_default_119 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_95 = torch.ops.aten.where.self(unsqueeze_225, full_default_119, div_117);  unsqueeze_225 = full_default_119 = div_117 = None
        view_847 = torch.ops.aten.view.default(add_167, [1024, 4, 12, 64]);  add_167 = None
        permute_742 = torch.ops.aten.permute.default(view_847, [1, 0, 2, 3]);  view_847 = None
        permute_743 = torch.ops.aten.permute.default(where_95, [0, 2, 1, 3]);  where_95 = None
        clone_150 = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        view_848 = torch.ops.aten.view.default(clone_150, [48, 4, 256, 513]);  clone_150 = None
        permute_744 = torch.ops.aten.permute.default(permute_742, [0, 2, 1, 3]);  permute_742 = None
        view_849 = torch.ops.aten.view.default(permute_744, [48, 1024, 64]);  permute_744 = None
        constant_pad_nd_46 = torch.ops.aten.constant_pad_nd.default(view_849, [0, 0, 256, 256], -1.0);  view_849 = None
        as_strided_71 = torch.ops.aten.as_strided.default(constant_pad_nd_46, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_46 = None
        constant_pad_nd_47 = torch.ops.aten.constant_pad_nd.default(view_848, [0, 257], 0.0);  view_848 = None
        view_850 = torch.ops.aten.view.default(constant_pad_nd_47, [48, 4, -1]);  constant_pad_nd_47 = None
        slice_2768 = torch.ops.aten.slice.Tensor(view_850, 2, 0, -256);  view_850 = None
        view_851 = torch.ops.aten.view.default(slice_2768, [48, 4, 256, 769]);  slice_2768 = None
        slice_2772 = torch.ops.aten.slice.Tensor(view_851, 3, 0, -1);  view_851 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(slice_2772, 4);  slice_2772 = None
        permute_745 = torch.ops.aten.permute.default(unsqueeze_226, [0, 1, 2, 4, 3]);  unsqueeze_226 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
        permute_746 = torch.ops.aten.permute.default(unsqueeze_227, [0, 1, 4, 3, 2]);  unsqueeze_227 = None
        permute_747 = torch.ops.aten.permute.default(permute_745, [0, 1, 2, 4, 3]);  permute_745 = None
        view_852 = torch.ops.aten.view.default(permute_747, [192, 256, 768]);  permute_747 = None
        permute_748 = torch.ops.aten.permute.default(permute_746, [0, 1, 4, 3, 2]);  permute_746 = None
        clone_151 = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
        view_853 = torch.ops.aten.view.default(clone_151, [192, 768, 64]);  clone_151 = None
        bmm_23 = torch.ops.aten.bmm.default(view_852, view_853);  view_852 = view_853 = None
        view_854 = torch.ops.aten.view.default(bmm_23, [48, 4, 256, 1, 64]);  bmm_23 = None
        permute_749 = torch.ops.aten.permute.default(view_854, [0, 1, 2, 4, 3]);  view_854 = None
        view_855 = torch.ops.aten.view.default(permute_749, [48, 4, 256, 64]);  permute_749 = None
        view_856 = torch.ops.aten.view.default(view_855, [4, 12, 1024, 64]);  view_855 = None
        permute_750 = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None
        permute_751 = torch.ops.aten.permute.default(permute_750, [1, 0, 2, 3]);  permute_750 = None
        clone_152 = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
        view_857 = torch.ops.aten.view.default(clone_152, [1024, 4, 768]);  clone_152 = None
        permute_752 = torch.ops.aten.permute.default(view_857, [1, 0, 2]);  view_857 = None
        permute_753 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        clone_153 = torch.ops.aten.clone.default(permute_752, memory_format = torch.contiguous_format);  permute_752 = None
        view_858 = torch.ops.aten.view.default(clone_153, [4096, 768]);  clone_153 = None
        mm_47 = torch.ops.aten.mm.default(view_858, permute_753);  view_858 = permute_753 = None
        view_859 = torch.ops.aten.view.default(mm_47, [4, 1024, 768]);  mm_47 = None
        add_172 = torch.ops.aten.add.Tensor(view_859, arg186_1);  view_859 = arg186_1 = None
        add_173 = torch.ops.aten.add.Tensor(add_172, add_164);  add_172 = add_164 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_94 = torch.ops.aten.sub.Tensor(add_173, getitem_45);  add_173 = getitem_45 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = rsqrt_22 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, arg187_1);  mul_89 = arg187_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_90, arg188_1);  mul_90 = arg188_1 = None
        view_860 = torch.ops.aten.view.default(add_175, [4096, 768])
        permute_754 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg190_1, view_860, permute_754);  arg190_1 = view_860 = permute_754 = None
        view_861 = torch.ops.aten.view.default(addmm_22, [4, 1024, 3072]);  addmm_22 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_861, 0.5)
        mul_92 = torch.ops.aten.mul.Tensor(view_861, 0.7071067811865476);  view_861 = None
        erf_11 = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_176 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_91, add_176);  mul_91 = add_176 = None
        view_862 = torch.ops.aten.view.default(mul_93, [4096, 3072]);  mul_93 = None
        permute_755 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg192_1, view_862, permute_755);  arg192_1 = view_862 = permute_755 = None
        view_863 = torch.ops.aten.view.default(addmm_23, [4, 1024, 768]);  addmm_23 = None
        add_177 = torch.ops.aten.add.Tensor(view_863, add_175);  view_863 = add_175 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_178 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_95 = torch.ops.aten.sub.Tensor(add_177, getitem_47);  add_177 = getitem_47 = None
        mul_94 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = rsqrt_23 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, arg193_1);  mul_94 = arg193_1 = None
        add_179 = torch.ops.aten.add.Tensor(mul_95, arg194_1);  mul_95 = arg194_1 = None
        return (add_179,)
        
def load_args(reader):
    buf0 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (4, 1024, 768), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf7, (4, 1024), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf8, (4, 1024), dtype=torch.bool, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf13, (3072, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf14, (3072,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768, 3072), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768, 768), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768, 768), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768, 768), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf29, (3072, 768), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf30, (3072,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768, 3072), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768, 768), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 768), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768, 768), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf45, (3072, 768), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf46, (3072,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768, 3072), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768, 768), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768, 768), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768, 768), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf61, (3072, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf62, (3072,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768, 3072), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768, 768), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768, 768), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768, 768), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf77, (3072, 768), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf78, (3072,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768, 3072), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768, 768), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 768), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768, 768), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768, 768), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf93, (3072, 768), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf94, (3072,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768, 3072), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768, 768), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 768), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768, 768), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768, 768), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf109, (3072, 768), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf110, (3072,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768, 3072), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768, 768), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 768), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768, 768), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf125, (3072, 768), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf126, (3072,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768, 3072), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768, 768), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 768), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768, 768), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768, 768), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf141, (3072, 768), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf142, (3072,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768, 3072), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768, 768), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768, 768), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768, 768), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 768), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf157, (3072, 768), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf158, (3072,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768, 3072), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768, 768), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 768), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768, 768), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf173, (3072, 768), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf174, (3072,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768, 3072), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf179, (768, 768), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768, 768), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768, 768), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768, 768), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf189, (3072, 768), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf190, (3072,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768, 3072), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf192, (768,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)