
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
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 1024])
        embedding = torch.ops.aten.embedding.default(arg1_1, view);  view = None
        full = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default = torch.ops.aten.full.default([4, 1, 1, 1024], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(embedding, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1 = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(arg7_1, mul_1);  arg7_1 = mul_1 = None
        permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_1 = torch.ops.aten.view.default(mul_2, [4096, 512])
        mm = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2 = torch.ops.aten.view.default(mm, [4, 1024, 512]);  mm = None
        view_3 = torch.ops.aten.view.default(view_2, [4, -1, 8, 64]);  view_2 = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_2 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        view_4 = torch.ops.aten.view.default(mul_2, [4096, 512])
        mm_1 = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(mm_1, [4, 1024, 512]);  mm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [4, -1, 8, 64]);  view_5 = None
        permute_3 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_7 = torch.ops.aten.view.default(mul_2, [4096, 512]);  mul_2 = None
        mm_2 = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8 = torch.ops.aten.view.default(mm_2, [4, 1024, 512]);  mm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [4, -1, 8, 64]);  view_8 = None
        permute_5 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        permute_6 = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand = torch.ops.aten.expand.default(permute_1, [4, 8, 1024, 64]);  permute_1 = None
        clone_1 = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_10 = torch.ops.aten.view.default(clone_1, [32, 1024, 64]);  clone_1 = None
        expand_1 = torch.ops.aten.expand.default(permute_6, [4, 8, 64, 1024]);  permute_6 = None
        clone_2 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_11 = torch.ops.aten.view.default(clone_2, [32, 64, 1024]);  clone_2 = None
        bmm = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12 = torch.ops.aten.view.default(bmm, [4, 8, 1024, 1024]);  bmm = None
        iota = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        iota_1 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_2);  unsqueeze_3 = unsqueeze_2 = None
        gt = torch.ops.aten.gt.Scalar(sub_1, 0)
        convert_element_type = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
        add_1 = torch.ops.aten.add.Tensor(mul_3, 0);  mul_3 = None
        abs_1 = torch.ops.aten.abs.default(sub_1);  sub_1 = None
        lt = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
        log = torch.ops.aten.log.default(div);  div = None
        div_1 = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_4 = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_4, torch.int64);  mul_4 = None
        add_2 = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
        full_default_1 = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum = torch.ops.aten.minimum.default(add_2, full_default_1);  add_2 = full_default_1 = None
        where = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_3 = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
        embedding_1 = torch.ops.aten.embedding.default(arg6_1, add_3);  arg6_1 = add_3 = None
        permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        add_4 = torch.ops.aten.add.Tensor(unsqueeze_4, full_default);  unsqueeze_4 = full_default = None
        add_5 = torch.ops.aten.add.Tensor(view_12, add_4);  view_12 = None
        view_13 = torch.ops.aten.view.default(add_5, [32, 1024, 1024]);  add_5 = None
        view_14 = torch.ops.aten.view.default(view_13, [4, 8, 1024, 1024]);  view_13 = None
        amax = torch.ops.aten.amax.default(view_14, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2 = torch.ops.aten.expand.default(div_2, [4, 8, 1024, 1024]);  div_2 = None
        view_15 = torch.ops.aten.view.default(expand_2, [32, 1024, 1024]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_5, [4, 8, 1024, 64]);  permute_5 = None
        clone_4 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_16 = torch.ops.aten.view.default(clone_4, [32, 1024, 64]);  clone_4 = None
        bmm_1 = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17 = torch.ops.aten.view.default(bmm_1, [4, 8, 1024, 64]);  bmm_1 = None
        permute_8 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_5 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_18 = torch.ops.aten.view.default(clone_5, [4, -1, 512]);  clone_5 = None
        permute_9 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        view_19 = torch.ops.aten.view.default(view_18, [4096, 512]);  view_18 = None
        mm_3 = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20 = torch.ops.aten.view.default(mm_3, [4, 1024, 512]);  mm_3 = None
        add_6 = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_7 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_6, rsqrt_1);  rsqrt_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(arg10_1, mul_5);  arg10_1 = mul_5 = None
        permute_10 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        view_21 = torch.ops.aten.view.default(mul_6, [4096, 512]);  mul_6 = None
        mm_4 = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22 = torch.ops.aten.view.default(mm_4, [4, 1024, 2048]);  mm_4 = None
        relu = torch.ops.aten.relu.default(view_22);  view_22 = None
        permute_11 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        view_23 = torch.ops.aten.view.default(relu, [4096, 2048]);  relu = None
        mm_5 = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24 = torch.ops.aten.view.default(mm_5, [4, 1024, 512]);  mm_5 = None
        add_8 = torch.ops.aten.add.Tensor(add_6, view_24);  add_6 = view_24 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(add_8, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_9 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_8, rsqrt_2);  rsqrt_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(arg15_1, mul_7);  arg15_1 = mul_7 = None
        permute_12 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        view_25 = torch.ops.aten.view.default(mul_8, [4096, 512])
        mm_6 = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26 = torch.ops.aten.view.default(mm_6, [4, 1024, 512]);  mm_6 = None
        view_27 = torch.ops.aten.view.default(view_26, [4, -1, 8, 64]);  view_26 = None
        permute_13 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        permute_14 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        view_28 = torch.ops.aten.view.default(mul_8, [4096, 512])
        mm_7 = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
        view_29 = torch.ops.aten.view.default(mm_7, [4, 1024, 512]);  mm_7 = None
        view_30 = torch.ops.aten.view.default(view_29, [4, -1, 8, 64]);  view_29 = None
        permute_15 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        permute_16 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        view_31 = torch.ops.aten.view.default(mul_8, [4096, 512]);  mul_8 = None
        mm_8 = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
        view_32 = torch.ops.aten.view.default(mm_8, [4, 1024, 512]);  mm_8 = None
        view_33 = torch.ops.aten.view.default(view_32, [4, -1, 8, 64]);  view_32 = None
        permute_17 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        permute_18 = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_4 = torch.ops.aten.expand.default(permute_13, [4, 8, 1024, 64]);  permute_13 = None
        clone_9 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_34 = torch.ops.aten.view.default(clone_9, [32, 1024, 64]);  clone_9 = None
        expand_5 = torch.ops.aten.expand.default(permute_18, [4, 8, 64, 1024]);  permute_18 = None
        clone_10 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_35 = torch.ops.aten.view.default(clone_10, [32, 64, 1024]);  clone_10 = None
        bmm_2 = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36 = torch.ops.aten.view.default(bmm_2, [4, 8, 1024, 1024]);  bmm_2 = None
        add_10 = torch.ops.aten.add.Tensor(view_36, add_4);  view_36 = None
        view_37 = torch.ops.aten.view.default(add_10, [32, 1024, 1024]);  add_10 = None
        view_38 = torch.ops.aten.view.default(view_37, [4, 8, 1024, 1024]);  view_37 = None
        amax_1 = torch.ops.aten.amax.default(view_38, [-1], True)
        sub_3 = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        expand_6 = torch.ops.aten.expand.default(div_3, [4, 8, 1024, 1024]);  div_3 = None
        view_39 = torch.ops.aten.view.default(expand_6, [32, 1024, 1024]);  expand_6 = None
        expand_7 = torch.ops.aten.expand.default(permute_17, [4, 8, 1024, 64]);  permute_17 = None
        clone_12 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_40 = torch.ops.aten.view.default(clone_12, [32, 1024, 64]);  clone_12 = None
        bmm_3 = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
        view_41 = torch.ops.aten.view.default(bmm_3, [4, 8, 1024, 64]);  bmm_3 = None
        permute_19 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_13 = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_42 = torch.ops.aten.view.default(clone_13, [4, -1, 512]);  clone_13 = None
        permute_20 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        view_43 = torch.ops.aten.view.default(view_42, [4096, 512]);  view_42 = None
        mm_9 = torch.ops.aten.mm.default(view_43, permute_20);  view_43 = permute_20 = None
        view_44 = torch.ops.aten.view.default(mm_9, [4, 1024, 512]);  mm_9 = None
        add_11 = torch.ops.aten.add.Tensor(add_8, view_44);  add_8 = view_44 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_12 = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_11, rsqrt_3);  rsqrt_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(arg18_1, mul_9);  arg18_1 = mul_9 = None
        permute_21 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        view_45 = torch.ops.aten.view.default(mul_10, [4096, 512]);  mul_10 = None
        mm_10 = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46 = torch.ops.aten.view.default(mm_10, [4, 1024, 2048]);  mm_10 = None
        relu_1 = torch.ops.aten.relu.default(view_46);  view_46 = None
        permute_22 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        view_47 = torch.ops.aten.view.default(relu_1, [4096, 2048]);  relu_1 = None
        mm_11 = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48 = torch.ops.aten.view.default(mm_11, [4, 1024, 512]);  mm_11 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, view_48);  add_11 = view_48 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_14 = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_13, rsqrt_4);  rsqrt_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(arg23_1, mul_11);  arg23_1 = mul_11 = None
        permute_23 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        view_49 = torch.ops.aten.view.default(mul_12, [4096, 512])
        mm_12 = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50 = torch.ops.aten.view.default(mm_12, [4, 1024, 512]);  mm_12 = None
        view_51 = torch.ops.aten.view.default(view_50, [4, -1, 8, 64]);  view_50 = None
        permute_24 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        permute_25 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        view_52 = torch.ops.aten.view.default(mul_12, [4096, 512])
        mm_13 = torch.ops.aten.mm.default(view_52, permute_25);  view_52 = permute_25 = None
        view_53 = torch.ops.aten.view.default(mm_13, [4, 1024, 512]);  mm_13 = None
        view_54 = torch.ops.aten.view.default(view_53, [4, -1, 8, 64]);  view_53 = None
        permute_26 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        permute_27 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        view_55 = torch.ops.aten.view.default(mul_12, [4096, 512]);  mul_12 = None
        mm_14 = torch.ops.aten.mm.default(view_55, permute_27);  view_55 = permute_27 = None
        view_56 = torch.ops.aten.view.default(mm_14, [4, 1024, 512]);  mm_14 = None
        view_57 = torch.ops.aten.view.default(view_56, [4, -1, 8, 64]);  view_56 = None
        permute_28 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        permute_29 = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_8 = torch.ops.aten.expand.default(permute_24, [4, 8, 1024, 64]);  permute_24 = None
        clone_17 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_58 = torch.ops.aten.view.default(clone_17, [32, 1024, 64]);  clone_17 = None
        expand_9 = torch.ops.aten.expand.default(permute_29, [4, 8, 64, 1024]);  permute_29 = None
        clone_18 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_59 = torch.ops.aten.view.default(clone_18, [32, 64, 1024]);  clone_18 = None
        bmm_4 = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60 = torch.ops.aten.view.default(bmm_4, [4, 8, 1024, 1024]);  bmm_4 = None
        add_15 = torch.ops.aten.add.Tensor(view_60, add_4);  view_60 = None
        view_61 = torch.ops.aten.view.default(add_15, [32, 1024, 1024]);  add_15 = None
        view_62 = torch.ops.aten.view.default(view_61, [4, 8, 1024, 1024]);  view_61 = None
        amax_2 = torch.ops.aten.amax.default(view_62, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(view_62, amax_2);  view_62 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        expand_10 = torch.ops.aten.expand.default(div_4, [4, 8, 1024, 1024]);  div_4 = None
        view_63 = torch.ops.aten.view.default(expand_10, [32, 1024, 1024]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(permute_28, [4, 8, 1024, 64]);  permute_28 = None
        clone_20 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_64 = torch.ops.aten.view.default(clone_20, [32, 1024, 64]);  clone_20 = None
        bmm_5 = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
        view_65 = torch.ops.aten.view.default(bmm_5, [4, 8, 1024, 64]);  bmm_5 = None
        permute_30 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_21 = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_66 = torch.ops.aten.view.default(clone_21, [4, -1, 512]);  clone_21 = None
        permute_31 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        view_67 = torch.ops.aten.view.default(view_66, [4096, 512]);  view_66 = None
        mm_15 = torch.ops.aten.mm.default(view_67, permute_31);  view_67 = permute_31 = None
        view_68 = torch.ops.aten.view.default(mm_15, [4, 1024, 512]);  mm_15 = None
        add_16 = torch.ops.aten.add.Tensor(add_13, view_68);  add_13 = view_68 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_17 = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_13 = torch.ops.aten.mul.Tensor(add_16, rsqrt_5);  rsqrt_5 = None
        mul_14 = torch.ops.aten.mul.Tensor(arg26_1, mul_13);  arg26_1 = mul_13 = None
        permute_32 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        view_69 = torch.ops.aten.view.default(mul_14, [4096, 512]);  mul_14 = None
        mm_16 = torch.ops.aten.mm.default(view_69, permute_32);  view_69 = permute_32 = None
        view_70 = torch.ops.aten.view.default(mm_16, [4, 1024, 2048]);  mm_16 = None
        relu_2 = torch.ops.aten.relu.default(view_70);  view_70 = None
        permute_33 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        view_71 = torch.ops.aten.view.default(relu_2, [4096, 2048]);  relu_2 = None
        mm_17 = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72 = torch.ops.aten.view.default(mm_17, [4, 1024, 512]);  mm_17 = None
        add_18 = torch.ops.aten.add.Tensor(add_16, view_72);  add_16 = view_72 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_19 = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_18, rsqrt_6);  rsqrt_6 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg31_1, mul_15);  arg31_1 = mul_15 = None
        permute_34 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        view_73 = torch.ops.aten.view.default(mul_16, [4096, 512])
        mm_18 = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74 = torch.ops.aten.view.default(mm_18, [4, 1024, 512]);  mm_18 = None
        view_75 = torch.ops.aten.view.default(view_74, [4, -1, 8, 64]);  view_74 = None
        permute_35 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        permute_36 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        view_76 = torch.ops.aten.view.default(mul_16, [4096, 512])
        mm_19 = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
        view_77 = torch.ops.aten.view.default(mm_19, [4, 1024, 512]);  mm_19 = None
        view_78 = torch.ops.aten.view.default(view_77, [4, -1, 8, 64]);  view_77 = None
        permute_37 = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        permute_38 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        view_79 = torch.ops.aten.view.default(mul_16, [4096, 512]);  mul_16 = None
        mm_20 = torch.ops.aten.mm.default(view_79, permute_38);  view_79 = permute_38 = None
        view_80 = torch.ops.aten.view.default(mm_20, [4, 1024, 512]);  mm_20 = None
        view_81 = torch.ops.aten.view.default(view_80, [4, -1, 8, 64]);  view_80 = None
        permute_39 = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        permute_40 = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_12 = torch.ops.aten.expand.default(permute_35, [4, 8, 1024, 64]);  permute_35 = None
        clone_25 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_82 = torch.ops.aten.view.default(clone_25, [32, 1024, 64]);  clone_25 = None
        expand_13 = torch.ops.aten.expand.default(permute_40, [4, 8, 64, 1024]);  permute_40 = None
        clone_26 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_83 = torch.ops.aten.view.default(clone_26, [32, 64, 1024]);  clone_26 = None
        bmm_6 = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84 = torch.ops.aten.view.default(bmm_6, [4, 8, 1024, 1024]);  bmm_6 = None
        add_20 = torch.ops.aten.add.Tensor(view_84, add_4);  view_84 = None
        view_85 = torch.ops.aten.view.default(add_20, [32, 1024, 1024]);  add_20 = None
        view_86 = torch.ops.aten.view.default(view_85, [4, 8, 1024, 1024]);  view_85 = None
        amax_3 = torch.ops.aten.amax.default(view_86, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(view_86, amax_3);  view_86 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        expand_14 = torch.ops.aten.expand.default(div_5, [4, 8, 1024, 1024]);  div_5 = None
        view_87 = torch.ops.aten.view.default(expand_14, [32, 1024, 1024]);  expand_14 = None
        expand_15 = torch.ops.aten.expand.default(permute_39, [4, 8, 1024, 64]);  permute_39 = None
        clone_28 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_88 = torch.ops.aten.view.default(clone_28, [32, 1024, 64]);  clone_28 = None
        bmm_7 = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
        view_89 = torch.ops.aten.view.default(bmm_7, [4, 8, 1024, 64]);  bmm_7 = None
        permute_41 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_29 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_90 = torch.ops.aten.view.default(clone_29, [4, -1, 512]);  clone_29 = None
        permute_42 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        view_91 = torch.ops.aten.view.default(view_90, [4096, 512]);  view_90 = None
        mm_21 = torch.ops.aten.mm.default(view_91, permute_42);  view_91 = permute_42 = None
        view_92 = torch.ops.aten.view.default(mm_21, [4, 1024, 512]);  mm_21 = None
        add_21 = torch.ops.aten.add.Tensor(add_18, view_92);  add_18 = view_92 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_22 = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_17 = torch.ops.aten.mul.Tensor(add_21, rsqrt_7);  rsqrt_7 = None
        mul_18 = torch.ops.aten.mul.Tensor(arg34_1, mul_17);  arg34_1 = mul_17 = None
        permute_43 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        view_93 = torch.ops.aten.view.default(mul_18, [4096, 512]);  mul_18 = None
        mm_22 = torch.ops.aten.mm.default(view_93, permute_43);  view_93 = permute_43 = None
        view_94 = torch.ops.aten.view.default(mm_22, [4, 1024, 2048]);  mm_22 = None
        relu_3 = torch.ops.aten.relu.default(view_94);  view_94 = None
        permute_44 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        view_95 = torch.ops.aten.view.default(relu_3, [4096, 2048]);  relu_3 = None
        mm_23 = torch.ops.aten.mm.default(view_95, permute_44);  view_95 = permute_44 = None
        view_96 = torch.ops.aten.view.default(mm_23, [4, 1024, 512]);  mm_23 = None
        add_23 = torch.ops.aten.add.Tensor(add_21, view_96);  add_21 = view_96 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_24 = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_19 = torch.ops.aten.mul.Tensor(add_23, rsqrt_8);  rsqrt_8 = None
        mul_20 = torch.ops.aten.mul.Tensor(arg39_1, mul_19);  arg39_1 = mul_19 = None
        permute_45 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        view_97 = torch.ops.aten.view.default(mul_20, [4096, 512])
        mm_24 = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98 = torch.ops.aten.view.default(mm_24, [4, 1024, 512]);  mm_24 = None
        view_99 = torch.ops.aten.view.default(view_98, [4, -1, 8, 64]);  view_98 = None
        permute_46 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        permute_47 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        view_100 = torch.ops.aten.view.default(mul_20, [4096, 512])
        mm_25 = torch.ops.aten.mm.default(view_100, permute_47);  view_100 = permute_47 = None
        view_101 = torch.ops.aten.view.default(mm_25, [4, 1024, 512]);  mm_25 = None
        view_102 = torch.ops.aten.view.default(view_101, [4, -1, 8, 64]);  view_101 = None
        permute_48 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        permute_49 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        view_103 = torch.ops.aten.view.default(mul_20, [4096, 512]);  mul_20 = None
        mm_26 = torch.ops.aten.mm.default(view_103, permute_49);  view_103 = permute_49 = None
        view_104 = torch.ops.aten.view.default(mm_26, [4, 1024, 512]);  mm_26 = None
        view_105 = torch.ops.aten.view.default(view_104, [4, -1, 8, 64]);  view_104 = None
        permute_50 = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        permute_51 = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_16 = torch.ops.aten.expand.default(permute_46, [4, 8, 1024, 64]);  permute_46 = None
        clone_33 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_106 = torch.ops.aten.view.default(clone_33, [32, 1024, 64]);  clone_33 = None
        expand_17 = torch.ops.aten.expand.default(permute_51, [4, 8, 64, 1024]);  permute_51 = None
        clone_34 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_107 = torch.ops.aten.view.default(clone_34, [32, 64, 1024]);  clone_34 = None
        bmm_8 = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108 = torch.ops.aten.view.default(bmm_8, [4, 8, 1024, 1024]);  bmm_8 = None
        add_25 = torch.ops.aten.add.Tensor(view_108, add_4);  view_108 = None
        view_109 = torch.ops.aten.view.default(add_25, [32, 1024, 1024]);  add_25 = None
        view_110 = torch.ops.aten.view.default(view_109, [4, 8, 1024, 1024]);  view_109 = None
        amax_4 = torch.ops.aten.amax.default(view_110, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(view_110, amax_4);  view_110 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        expand_18 = torch.ops.aten.expand.default(div_6, [4, 8, 1024, 1024]);  div_6 = None
        view_111 = torch.ops.aten.view.default(expand_18, [32, 1024, 1024]);  expand_18 = None
        expand_19 = torch.ops.aten.expand.default(permute_50, [4, 8, 1024, 64]);  permute_50 = None
        clone_36 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_112 = torch.ops.aten.view.default(clone_36, [32, 1024, 64]);  clone_36 = None
        bmm_9 = torch.ops.aten.bmm.default(view_111, view_112);  view_111 = view_112 = None
        view_113 = torch.ops.aten.view.default(bmm_9, [4, 8, 1024, 64]);  bmm_9 = None
        permute_52 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_37 = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_114 = torch.ops.aten.view.default(clone_37, [4, -1, 512]);  clone_37 = None
        permute_53 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        view_115 = torch.ops.aten.view.default(view_114, [4096, 512]);  view_114 = None
        mm_27 = torch.ops.aten.mm.default(view_115, permute_53);  view_115 = permute_53 = None
        view_116 = torch.ops.aten.view.default(mm_27, [4, 1024, 512]);  mm_27 = None
        add_26 = torch.ops.aten.add.Tensor(add_23, view_116);  add_23 = view_116 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_27 = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_21 = torch.ops.aten.mul.Tensor(add_26, rsqrt_9);  rsqrt_9 = None
        mul_22 = torch.ops.aten.mul.Tensor(arg42_1, mul_21);  arg42_1 = mul_21 = None
        permute_54 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        view_117 = torch.ops.aten.view.default(mul_22, [4096, 512]);  mul_22 = None
        mm_28 = torch.ops.aten.mm.default(view_117, permute_54);  view_117 = permute_54 = None
        view_118 = torch.ops.aten.view.default(mm_28, [4, 1024, 2048]);  mm_28 = None
        relu_4 = torch.ops.aten.relu.default(view_118);  view_118 = None
        permute_55 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        view_119 = torch.ops.aten.view.default(relu_4, [4096, 2048]);  relu_4 = None
        mm_29 = torch.ops.aten.mm.default(view_119, permute_55);  view_119 = permute_55 = None
        view_120 = torch.ops.aten.view.default(mm_29, [4, 1024, 512]);  mm_29 = None
        add_28 = torch.ops.aten.add.Tensor(add_26, view_120);  add_26 = view_120 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(add_28, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_29 = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_28, rsqrt_10);  rsqrt_10 = None
        mul_24 = torch.ops.aten.mul.Tensor(arg47_1, mul_23);  arg47_1 = mul_23 = None
        permute_56 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        view_121 = torch.ops.aten.view.default(mul_24, [4096, 512])
        mm_30 = torch.ops.aten.mm.default(view_121, permute_56);  view_121 = permute_56 = None
        view_122 = torch.ops.aten.view.default(mm_30, [4, 1024, 512]);  mm_30 = None
        view_123 = torch.ops.aten.view.default(view_122, [4, -1, 8, 64]);  view_122 = None
        permute_57 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        permute_58 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        view_124 = torch.ops.aten.view.default(mul_24, [4096, 512])
        mm_31 = torch.ops.aten.mm.default(view_124, permute_58);  view_124 = permute_58 = None
        view_125 = torch.ops.aten.view.default(mm_31, [4, 1024, 512]);  mm_31 = None
        view_126 = torch.ops.aten.view.default(view_125, [4, -1, 8, 64]);  view_125 = None
        permute_59 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        permute_60 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        view_127 = torch.ops.aten.view.default(mul_24, [4096, 512]);  mul_24 = None
        mm_32 = torch.ops.aten.mm.default(view_127, permute_60);  view_127 = permute_60 = None
        view_128 = torch.ops.aten.view.default(mm_32, [4, 1024, 512]);  mm_32 = None
        view_129 = torch.ops.aten.view.default(view_128, [4, -1, 8, 64]);  view_128 = None
        permute_61 = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        permute_62 = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_20 = torch.ops.aten.expand.default(permute_57, [4, 8, 1024, 64]);  permute_57 = None
        clone_41 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_130 = torch.ops.aten.view.default(clone_41, [32, 1024, 64]);  clone_41 = None
        expand_21 = torch.ops.aten.expand.default(permute_62, [4, 8, 64, 1024]);  permute_62 = None
        clone_42 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_131 = torch.ops.aten.view.default(clone_42, [32, 64, 1024]);  clone_42 = None
        bmm_10 = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132 = torch.ops.aten.view.default(bmm_10, [4, 8, 1024, 1024]);  bmm_10 = None
        add_30 = torch.ops.aten.add.Tensor(view_132, add_4);  view_132 = add_4 = None
        view_133 = torch.ops.aten.view.default(add_30, [32, 1024, 1024]);  add_30 = None
        view_134 = torch.ops.aten.view.default(view_133, [4, 8, 1024, 1024]);  view_133 = None
        amax_5 = torch.ops.aten.amax.default(view_134, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(view_134, amax_5);  view_134 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        expand_22 = torch.ops.aten.expand.default(div_7, [4, 8, 1024, 1024]);  div_7 = None
        view_135 = torch.ops.aten.view.default(expand_22, [32, 1024, 1024]);  expand_22 = None
        expand_23 = torch.ops.aten.expand.default(permute_61, [4, 8, 1024, 64]);  permute_61 = None
        clone_44 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_136 = torch.ops.aten.view.default(clone_44, [32, 1024, 64]);  clone_44 = None
        bmm_11 = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
        view_137 = torch.ops.aten.view.default(bmm_11, [4, 8, 1024, 64]);  bmm_11 = None
        permute_63 = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_45 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_138 = torch.ops.aten.view.default(clone_45, [4, -1, 512]);  clone_45 = None
        permute_64 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        view_139 = torch.ops.aten.view.default(view_138, [4096, 512]);  view_138 = None
        mm_33 = torch.ops.aten.mm.default(view_139, permute_64);  view_139 = permute_64 = None
        view_140 = torch.ops.aten.view.default(mm_33, [4, 1024, 512]);  mm_33 = None
        add_31 = torch.ops.aten.add.Tensor(add_28, view_140);  add_28 = view_140 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_32 = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_25 = torch.ops.aten.mul.Tensor(add_31, rsqrt_11);  rsqrt_11 = None
        mul_26 = torch.ops.aten.mul.Tensor(arg50_1, mul_25);  arg50_1 = mul_25 = None
        permute_65 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        view_141 = torch.ops.aten.view.default(mul_26, [4096, 512]);  mul_26 = None
        mm_34 = torch.ops.aten.mm.default(view_141, permute_65);  view_141 = permute_65 = None
        view_142 = torch.ops.aten.view.default(mm_34, [4, 1024, 2048]);  mm_34 = None
        relu_5 = torch.ops.aten.relu.default(view_142);  view_142 = None
        permute_66 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        view_143 = torch.ops.aten.view.default(relu_5, [4096, 2048]);  relu_5 = None
        mm_35 = torch.ops.aten.mm.default(view_143, permute_66);  view_143 = permute_66 = None
        view_144 = torch.ops.aten.view.default(mm_35, [4, 1024, 512]);  mm_35 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, view_144);  add_31 = view_144 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(add_33, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_34 = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_27 = torch.ops.aten.mul.Tensor(add_33, rsqrt_12);  add_33 = rsqrt_12 = None
        mul_28 = torch.ops.aten.mul.Tensor(arg51_1, mul_27);  arg51_1 = mul_27 = None
        view_145 = torch.ops.aten.view.default(arg0_1, [-1, 1024]);  arg0_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg1_1, view_145);  view_145 = None
        full_2 = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_2 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(iota_2, 0)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        repeat = torch.ops.aten.repeat.default(unsqueeze_6, [4, 1024, 1]);  unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        le = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(full_2, 1);  full_2 = unsqueeze_10 = None
        full_default_2 = torch.ops.aten.full.default([4, 1, 1, 1024], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None
        sub_8 = torch.ops.aten.sub.Tensor(1.0, unsqueeze_9);  unsqueeze_9 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_8, -3.4028234663852886e+38);  sub_8 = None
        full_3 = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(full_3, 1);  full_3 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(unsqueeze_13, torch.float32);  unsqueeze_13 = None
        sub_9 = torch.ops.aten.sub.Tensor(1.0, convert_element_type_4);  convert_element_type_4 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_9, -3.4028234663852886e+38);  sub_9 = mul_31 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(embedding_2, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_35 = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_32 = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_13);  rsqrt_13 = None
        mul_33 = torch.ops.aten.mul.Tensor(arg58_1, mul_32);  arg58_1 = mul_32 = None
        permute_67 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        view_146 = torch.ops.aten.view.default(mul_33, [4096, 512])
        mm_36 = torch.ops.aten.mm.default(view_146, permute_67);  view_146 = permute_67 = None
        view_147 = torch.ops.aten.view.default(mm_36, [4, 1024, 512]);  mm_36 = None
        view_148 = torch.ops.aten.view.default(view_147, [4, -1, 8, 64]);  view_147 = None
        permute_68 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        permute_69 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        view_149 = torch.ops.aten.view.default(mul_33, [4096, 512])
        mm_37 = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
        view_150 = torch.ops.aten.view.default(mm_37, [4, 1024, 512]);  mm_37 = None
        view_151 = torch.ops.aten.view.default(view_150, [4, -1, 8, 64]);  view_150 = None
        permute_70 = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        permute_71 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        view_152 = torch.ops.aten.view.default(mul_33, [4096, 512]);  mul_33 = None
        mm_38 = torch.ops.aten.mm.default(view_152, permute_71);  view_152 = permute_71 = None
        view_153 = torch.ops.aten.view.default(mm_38, [4, 1024, 512]);  mm_38 = None
        view_154 = torch.ops.aten.view.default(view_153, [4, -1, 8, 64]);  view_153 = None
        permute_72 = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        permute_73 = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
        expand_24 = torch.ops.aten.expand.default(permute_68, [4, 8, 1024, 64]);  permute_68 = None
        clone_51 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_155 = torch.ops.aten.view.default(clone_51, [32, 1024, 64]);  clone_51 = None
        expand_25 = torch.ops.aten.expand.default(permute_73, [4, 8, 64, 1024]);  permute_73 = None
        clone_52 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_156 = torch.ops.aten.view.default(clone_52, [32, 64, 1024]);  clone_52 = None
        bmm_12 = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
        view_157 = torch.ops.aten.view.default(bmm_12, [4, 8, 1024, 1024]);  bmm_12 = None
        iota_3 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(iota_3, 1);  iota_3 = None
        iota_4 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        sub_10 = torch.ops.aten.sub.Tensor(unsqueeze_15, unsqueeze_14);  unsqueeze_15 = unsqueeze_14 = None
        full_default_3 = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1 = torch.ops.aten.minimum.default(sub_10, full_default_3);  sub_10 = full_default_3 = None
        neg = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        lt_1 = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_8 = torch.ops.aten.div.Tensor(convert_element_type_5, 16);  convert_element_type_5 = None
        log_1 = torch.ops.aten.log.default(div_8);  div_8 = None
        div_9 = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_34 = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_34, torch.int64);  mul_34 = None
        add_36 = torch.ops.aten.add.Tensor(convert_element_type_6, 16);  convert_element_type_6 = None
        full_default_4 = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_2 = torch.ops.aten.minimum.default(add_36, full_default_4);  add_36 = full_default_4 = None
        where_1 = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_37 = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
        embedding_3 = torch.ops.aten.embedding.default(arg57_1, add_37);  arg57_1 = add_37 = None
        permute_74 = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None
        add_38 = torch.ops.aten.add.Tensor(unsqueeze_16, mul_30);  unsqueeze_16 = mul_30 = None
        add_39 = torch.ops.aten.add.Tensor(view_157, add_38);  view_157 = None
        view_158 = torch.ops.aten.view.default(add_39, [32, 1024, 1024]);  add_39 = None
        view_159 = torch.ops.aten.view.default(view_158, [4, 8, 1024, 1024]);  view_158 = None
        amax_6 = torch.ops.aten.amax.default(view_159, [-1], True)
        sub_11 = torch.ops.aten.sub.Tensor(view_159, amax_6);  view_159 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_26 = torch.ops.aten.expand.default(div_10, [4, 8, 1024, 1024]);  div_10 = None
        view_160 = torch.ops.aten.view.default(expand_26, [32, 1024, 1024]);  expand_26 = None
        expand_27 = torch.ops.aten.expand.default(permute_72, [4, 8, 1024, 64])
        clone_54 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_161 = torch.ops.aten.view.default(clone_54, [32, 1024, 64]);  clone_54 = None
        bmm_13 = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
        view_162 = torch.ops.aten.view.default(bmm_13, [4, 8, 1024, 64]);  bmm_13 = None
        permute_75 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        clone_55 = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_163 = torch.ops.aten.view.default(clone_55, [4, -1, 512]);  clone_55 = None
        permute_76 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        view_164 = torch.ops.aten.view.default(view_163, [4096, 512]);  view_163 = None
        mm_39 = torch.ops.aten.mm.default(view_164, permute_76);  view_164 = permute_76 = None
        view_165 = torch.ops.aten.view.default(mm_39, [4, 1024, 512]);  mm_39 = None
        add_40 = torch.ops.aten.add.Tensor(embedding_2, view_165);  embedding_2 = view_165 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(add_40, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_41 = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_35 = torch.ops.aten.mul.Tensor(add_40, rsqrt_14);  rsqrt_14 = None
        mul_36 = torch.ops.aten.mul.Tensor(arg63_1, mul_35);  arg63_1 = mul_35 = None
        permute_77 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        view_166 = torch.ops.aten.view.default(mul_36, [4096, 512]);  mul_36 = None
        mm_40 = torch.ops.aten.mm.default(view_166, permute_77);  view_166 = permute_77 = None
        view_167 = torch.ops.aten.view.default(mm_40, [4, 1024, 512]);  mm_40 = None
        view_168 = torch.ops.aten.view.default(view_167, [4, -1, 8, 64]);  view_167 = None
        permute_78 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        permute_79 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        view_169 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_41 = torch.ops.aten.mm.default(view_169, permute_79);  view_169 = permute_79 = None
        view_170 = torch.ops.aten.view.default(mm_41, [4, 1024, 512]);  mm_41 = None
        view_171 = torch.ops.aten.view.default(view_170, [4, -1, 8, 64]);  view_170 = None
        permute_80 = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        permute_81 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        view_172 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_42 = torch.ops.aten.mm.default(view_172, permute_81);  view_172 = permute_81 = None
        view_173 = torch.ops.aten.view.default(mm_42, [4, 1024, 512]);  mm_42 = None
        view_174 = torch.ops.aten.view.default(view_173, [4, -1, 8, 64]);  view_173 = None
        permute_82 = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        permute_83 = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2])
        expand_28 = torch.ops.aten.expand.default(permute_78, [4, 8, 1024, 64]);  permute_78 = None
        clone_57 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_175 = torch.ops.aten.view.default(clone_57, [32, 1024, 64]);  clone_57 = None
        expand_29 = torch.ops.aten.expand.default(permute_83, [4, 8, 64, 1024]);  permute_83 = None
        clone_58 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_176 = torch.ops.aten.view.default(clone_58, [32, 64, 1024]);  clone_58 = None
        bmm_14 = torch.ops.aten.bmm.default(view_175, view_176);  view_175 = view_176 = None
        view_177 = torch.ops.aten.view.default(bmm_14, [4, 8, 1024, 1024]);  bmm_14 = None
        full_6 = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_6 = None
        full_default_5 = torch.ops.aten.full.default([4, 8, 1024, 1024], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_5 = None
        view_178 = torch.ops.aten.view.default(view_177, [32, 1024, 1024]);  view_177 = None
        view_179 = torch.ops.aten.view.default(view_178, [4, 8, 1024, 1024]);  view_178 = None
        amax_7 = torch.ops.aten.amax.default(view_179, [-1], True)
        sub_12 = torch.ops.aten.sub.Tensor(view_179, amax_7);  view_179 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_30 = torch.ops.aten.expand.default(div_11, [4, 8, 1024, 1024]);  div_11 = None
        view_180 = torch.ops.aten.view.default(expand_30, [32, 1024, 1024]);  expand_30 = None
        expand_31 = torch.ops.aten.expand.default(permute_82, [4, 8, 1024, 64])
        clone_60 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_181 = torch.ops.aten.view.default(clone_60, [32, 1024, 64]);  clone_60 = None
        bmm_15 = torch.ops.aten.bmm.default(view_180, view_181);  view_180 = view_181 = None
        view_182 = torch.ops.aten.view.default(bmm_15, [4, 8, 1024, 64]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_61 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_183 = torch.ops.aten.view.default(clone_61, [4, -1, 512]);  clone_61 = None
        permute_85 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        view_184 = torch.ops.aten.view.default(view_183, [4096, 512]);  view_183 = None
        mm_43 = torch.ops.aten.mm.default(view_184, permute_85);  view_184 = permute_85 = None
        view_185 = torch.ops.aten.view.default(mm_43, [4, 1024, 512]);  mm_43 = None
        add_44 = torch.ops.aten.add.Tensor(add_40, view_185);  add_40 = view_185 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_45 = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_37 = torch.ops.aten.mul.Tensor(add_44, rsqrt_15);  rsqrt_15 = None
        mul_38 = torch.ops.aten.mul.Tensor(arg66_1, mul_37);  arg66_1 = mul_37 = None
        permute_86 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        view_186 = torch.ops.aten.view.default(mul_38, [4096, 512]);  mul_38 = None
        mm_44 = torch.ops.aten.mm.default(view_186, permute_86);  view_186 = permute_86 = None
        view_187 = torch.ops.aten.view.default(mm_44, [4, 1024, 2048]);  mm_44 = None
        relu_6 = torch.ops.aten.relu.default(view_187);  view_187 = None
        permute_87 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        view_188 = torch.ops.aten.view.default(relu_6, [4096, 2048]);  relu_6 = None
        mm_45 = torch.ops.aten.mm.default(view_188, permute_87);  view_188 = permute_87 = None
        view_189 = torch.ops.aten.view.default(mm_45, [4, 1024, 512]);  mm_45 = None
        add_46 = torch.ops.aten.add.Tensor(add_44, view_189);  add_44 = view_189 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_47 = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_46, rsqrt_16);  rsqrt_16 = None
        mul_40 = torch.ops.aten.mul.Tensor(arg71_1, mul_39);  arg71_1 = mul_39 = None
        permute_88 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        view_190 = torch.ops.aten.view.default(mul_40, [4096, 512])
        mm_46 = torch.ops.aten.mm.default(view_190, permute_88);  view_190 = permute_88 = None
        view_191 = torch.ops.aten.view.default(mm_46, [4, 1024, 512]);  mm_46 = None
        view_192 = torch.ops.aten.view.default(view_191, [4, -1, 8, 64]);  view_191 = None
        permute_89 = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        permute_90 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        view_193 = torch.ops.aten.view.default(mul_40, [4096, 512])
        mm_47 = torch.ops.aten.mm.default(view_193, permute_90);  view_193 = permute_90 = None
        view_194 = torch.ops.aten.view.default(mm_47, [4, 1024, 512]);  mm_47 = None
        view_195 = torch.ops.aten.view.default(view_194, [4, -1, 8, 64]);  view_194 = None
        permute_91 = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        permute_92 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        view_196 = torch.ops.aten.view.default(mul_40, [4096, 512]);  mul_40 = None
        mm_48 = torch.ops.aten.mm.default(view_196, permute_92);  view_196 = permute_92 = None
        view_197 = torch.ops.aten.view.default(mm_48, [4, 1024, 512]);  mm_48 = None
        view_198 = torch.ops.aten.view.default(view_197, [4, -1, 8, 64]);  view_197 = None
        permute_93 = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        permute_94 = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2])
        expand_32 = torch.ops.aten.expand.default(permute_89, [4, 8, 1024, 64]);  permute_89 = None
        clone_65 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_199 = torch.ops.aten.view.default(clone_65, [32, 1024, 64]);  clone_65 = None
        expand_33 = torch.ops.aten.expand.default(permute_94, [4, 8, 64, 1024]);  permute_94 = None
        clone_66 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_200 = torch.ops.aten.view.default(clone_66, [32, 64, 1024]);  clone_66 = None
        bmm_16 = torch.ops.aten.bmm.default(view_199, view_200);  view_199 = view_200 = None
        view_201 = torch.ops.aten.view.default(bmm_16, [4, 8, 1024, 1024]);  bmm_16 = None
        add_48 = torch.ops.aten.add.Tensor(view_201, add_38);  view_201 = None
        view_202 = torch.ops.aten.view.default(add_48, [32, 1024, 1024]);  add_48 = None
        view_203 = torch.ops.aten.view.default(view_202, [4, 8, 1024, 1024]);  view_202 = None
        amax_8 = torch.ops.aten.amax.default(view_203, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(view_203, amax_8);  view_203 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_34 = torch.ops.aten.expand.default(div_12, [4, 8, 1024, 1024]);  div_12 = None
        view_204 = torch.ops.aten.view.default(expand_34, [32, 1024, 1024]);  expand_34 = None
        expand_35 = torch.ops.aten.expand.default(permute_93, [4, 8, 1024, 64])
        clone_68 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_205 = torch.ops.aten.view.default(clone_68, [32, 1024, 64]);  clone_68 = None
        bmm_17 = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206 = torch.ops.aten.view.default(bmm_17, [4, 8, 1024, 64]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_69 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_207 = torch.ops.aten.view.default(clone_69, [4, -1, 512]);  clone_69 = None
        permute_96 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        view_208 = torch.ops.aten.view.default(view_207, [4096, 512]);  view_207 = None
        mm_49 = torch.ops.aten.mm.default(view_208, permute_96);  view_208 = permute_96 = None
        view_209 = torch.ops.aten.view.default(mm_49, [4, 1024, 512]);  mm_49 = None
        add_49 = torch.ops.aten.add.Tensor(add_46, view_209);  add_46 = view_209 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        add_50 = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_41 = torch.ops.aten.mul.Tensor(add_49, rsqrt_17);  rsqrt_17 = None
        mul_42 = torch.ops.aten.mul.Tensor(arg76_1, mul_41);  arg76_1 = mul_41 = None
        permute_97 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        view_210 = torch.ops.aten.view.default(mul_42, [4096, 512]);  mul_42 = None
        mm_50 = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
        view_211 = torch.ops.aten.view.default(mm_50, [4, 1024, 512]);  mm_50 = None
        view_212 = torch.ops.aten.view.default(view_211, [4, -1, 8, 64]);  view_211 = None
        permute_98 = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        permute_99 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        view_213 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_51 = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
        view_214 = torch.ops.aten.view.default(mm_51, [4, 1024, 512]);  mm_51 = None
        view_215 = torch.ops.aten.view.default(view_214, [4, -1, 8, 64]);  view_214 = None
        permute_100 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        permute_101 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        view_216 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_52 = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
        view_217 = torch.ops.aten.view.default(mm_52, [4, 1024, 512]);  mm_52 = None
        view_218 = torch.ops.aten.view.default(view_217, [4, -1, 8, 64]);  view_217 = None
        permute_102 = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        permute_103 = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
        expand_36 = torch.ops.aten.expand.default(permute_98, [4, 8, 1024, 64]);  permute_98 = None
        clone_71 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_219 = torch.ops.aten.view.default(clone_71, [32, 1024, 64]);  clone_71 = None
        expand_37 = torch.ops.aten.expand.default(permute_103, [4, 8, 64, 1024]);  permute_103 = None
        clone_72 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_220 = torch.ops.aten.view.default(clone_72, [32, 64, 1024]);  clone_72 = None
        bmm_18 = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
        view_221 = torch.ops.aten.view.default(bmm_18, [4, 8, 1024, 1024]);  bmm_18 = None
        view_222 = torch.ops.aten.view.default(view_221, [32, 1024, 1024]);  view_221 = None
        view_223 = torch.ops.aten.view.default(view_222, [4, 8, 1024, 1024]);  view_222 = None
        amax_9 = torch.ops.aten.amax.default(view_223, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(view_223, amax_9);  view_223 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_38 = torch.ops.aten.expand.default(div_13, [4, 8, 1024, 1024]);  div_13 = None
        view_224 = torch.ops.aten.view.default(expand_38, [32, 1024, 1024]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(permute_102, [4, 8, 1024, 64])
        clone_74 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_225 = torch.ops.aten.view.default(clone_74, [32, 1024, 64]);  clone_74 = None
        bmm_19 = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
        view_226 = torch.ops.aten.view.default(bmm_19, [4, 8, 1024, 64]);  bmm_19 = None
        permute_104 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_75 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_227 = torch.ops.aten.view.default(clone_75, [4, -1, 512]);  clone_75 = None
        permute_105 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        view_228 = torch.ops.aten.view.default(view_227, [4096, 512]);  view_227 = None
        mm_53 = torch.ops.aten.mm.default(view_228, permute_105);  view_228 = permute_105 = None
        view_229 = torch.ops.aten.view.default(mm_53, [4, 1024, 512]);  mm_53 = None
        add_52 = torch.ops.aten.add.Tensor(add_49, view_229);  add_49 = view_229 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_53 = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_43 = torch.ops.aten.mul.Tensor(add_52, rsqrt_18);  rsqrt_18 = None
        mul_44 = torch.ops.aten.mul.Tensor(arg79_1, mul_43);  arg79_1 = mul_43 = None
        permute_106 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        view_230 = torch.ops.aten.view.default(mul_44, [4096, 512]);  mul_44 = None
        mm_54 = torch.ops.aten.mm.default(view_230, permute_106);  view_230 = permute_106 = None
        view_231 = torch.ops.aten.view.default(mm_54, [4, 1024, 2048]);  mm_54 = None
        relu_7 = torch.ops.aten.relu.default(view_231);  view_231 = None
        permute_107 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        view_232 = torch.ops.aten.view.default(relu_7, [4096, 2048]);  relu_7 = None
        mm_55 = torch.ops.aten.mm.default(view_232, permute_107);  view_232 = permute_107 = None
        view_233 = torch.ops.aten.view.default(mm_55, [4, 1024, 512]);  mm_55 = None
        add_54 = torch.ops.aten.add.Tensor(add_52, view_233);  add_52 = view_233 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_55 = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_45 = torch.ops.aten.mul.Tensor(add_54, rsqrt_19);  rsqrt_19 = None
        mul_46 = torch.ops.aten.mul.Tensor(arg84_1, mul_45);  arg84_1 = mul_45 = None
        permute_108 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        view_234 = torch.ops.aten.view.default(mul_46, [4096, 512])
        mm_56 = torch.ops.aten.mm.default(view_234, permute_108);  view_234 = permute_108 = None
        view_235 = torch.ops.aten.view.default(mm_56, [4, 1024, 512]);  mm_56 = None
        view_236 = torch.ops.aten.view.default(view_235, [4, -1, 8, 64]);  view_235 = None
        permute_109 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        permute_110 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        view_237 = torch.ops.aten.view.default(mul_46, [4096, 512])
        mm_57 = torch.ops.aten.mm.default(view_237, permute_110);  view_237 = permute_110 = None
        view_238 = torch.ops.aten.view.default(mm_57, [4, 1024, 512]);  mm_57 = None
        view_239 = torch.ops.aten.view.default(view_238, [4, -1, 8, 64]);  view_238 = None
        permute_111 = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        permute_112 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        view_240 = torch.ops.aten.view.default(mul_46, [4096, 512]);  mul_46 = None
        mm_58 = torch.ops.aten.mm.default(view_240, permute_112);  view_240 = permute_112 = None
        view_241 = torch.ops.aten.view.default(mm_58, [4, 1024, 512]);  mm_58 = None
        view_242 = torch.ops.aten.view.default(view_241, [4, -1, 8, 64]);  view_241 = None
        permute_113 = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
        permute_114 = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2])
        expand_40 = torch.ops.aten.expand.default(permute_109, [4, 8, 1024, 64]);  permute_109 = None
        clone_79 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_243 = torch.ops.aten.view.default(clone_79, [32, 1024, 64]);  clone_79 = None
        expand_41 = torch.ops.aten.expand.default(permute_114, [4, 8, 64, 1024]);  permute_114 = None
        clone_80 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_244 = torch.ops.aten.view.default(clone_80, [32, 64, 1024]);  clone_80 = None
        bmm_20 = torch.ops.aten.bmm.default(view_243, view_244);  view_243 = view_244 = None
        view_245 = torch.ops.aten.view.default(bmm_20, [4, 8, 1024, 1024]);  bmm_20 = None
        add_56 = torch.ops.aten.add.Tensor(view_245, add_38);  view_245 = None
        view_246 = torch.ops.aten.view.default(add_56, [32, 1024, 1024]);  add_56 = None
        view_247 = torch.ops.aten.view.default(view_246, [4, 8, 1024, 1024]);  view_246 = None
        amax_10 = torch.ops.aten.amax.default(view_247, [-1], True)
        sub_15 = torch.ops.aten.sub.Tensor(view_247, amax_10);  view_247 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_42 = torch.ops.aten.expand.default(div_14, [4, 8, 1024, 1024]);  div_14 = None
        view_248 = torch.ops.aten.view.default(expand_42, [32, 1024, 1024]);  expand_42 = None
        expand_43 = torch.ops.aten.expand.default(permute_113, [4, 8, 1024, 64])
        clone_82 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_249 = torch.ops.aten.view.default(clone_82, [32, 1024, 64]);  clone_82 = None
        bmm_21 = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
        view_250 = torch.ops.aten.view.default(bmm_21, [4, 8, 1024, 64]);  bmm_21 = None
        permute_115 = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        clone_83 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_251 = torch.ops.aten.view.default(clone_83, [4, -1, 512]);  clone_83 = None
        permute_116 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        view_252 = torch.ops.aten.view.default(view_251, [4096, 512]);  view_251 = None
        mm_59 = torch.ops.aten.mm.default(view_252, permute_116);  view_252 = permute_116 = None
        view_253 = torch.ops.aten.view.default(mm_59, [4, 1024, 512]);  mm_59 = None
        add_57 = torch.ops.aten.add.Tensor(add_54, view_253);  add_54 = view_253 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(add_57, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_58 = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_57, rsqrt_20);  rsqrt_20 = None
        mul_48 = torch.ops.aten.mul.Tensor(arg89_1, mul_47);  arg89_1 = mul_47 = None
        permute_117 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        view_254 = torch.ops.aten.view.default(mul_48, [4096, 512]);  mul_48 = None
        mm_60 = torch.ops.aten.mm.default(view_254, permute_117);  view_254 = permute_117 = None
        view_255 = torch.ops.aten.view.default(mm_60, [4, 1024, 512]);  mm_60 = None
        view_256 = torch.ops.aten.view.default(view_255, [4, -1, 8, 64]);  view_255 = None
        permute_118 = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        permute_119 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        view_257 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_61 = torch.ops.aten.mm.default(view_257, permute_119);  view_257 = permute_119 = None
        view_258 = torch.ops.aten.view.default(mm_61, [4, 1024, 512]);  mm_61 = None
        view_259 = torch.ops.aten.view.default(view_258, [4, -1, 8, 64]);  view_258 = None
        permute_120 = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        permute_121 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        view_260 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_62 = torch.ops.aten.mm.default(view_260, permute_121);  view_260 = permute_121 = None
        view_261 = torch.ops.aten.view.default(mm_62, [4, 1024, 512]);  mm_62 = None
        view_262 = torch.ops.aten.view.default(view_261, [4, -1, 8, 64]);  view_261 = None
        permute_122 = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        permute_123 = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2])
        expand_44 = torch.ops.aten.expand.default(permute_118, [4, 8, 1024, 64]);  permute_118 = None
        clone_85 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_263 = torch.ops.aten.view.default(clone_85, [32, 1024, 64]);  clone_85 = None
        expand_45 = torch.ops.aten.expand.default(permute_123, [4, 8, 64, 1024]);  permute_123 = None
        clone_86 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_264 = torch.ops.aten.view.default(clone_86, [32, 64, 1024]);  clone_86 = None
        bmm_22 = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = view_264 = None
        view_265 = torch.ops.aten.view.default(bmm_22, [4, 8, 1024, 1024]);  bmm_22 = None
        view_266 = torch.ops.aten.view.default(view_265, [32, 1024, 1024]);  view_265 = None
        view_267 = torch.ops.aten.view.default(view_266, [4, 8, 1024, 1024]);  view_266 = None
        amax_11 = torch.ops.aten.amax.default(view_267, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(view_267, amax_11);  view_267 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_46 = torch.ops.aten.expand.default(div_15, [4, 8, 1024, 1024]);  div_15 = None
        view_268 = torch.ops.aten.view.default(expand_46, [32, 1024, 1024]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(permute_122, [4, 8, 1024, 64])
        clone_88 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_269 = torch.ops.aten.view.default(clone_88, [32, 1024, 64]);  clone_88 = None
        bmm_23 = torch.ops.aten.bmm.default(view_268, view_269);  view_268 = view_269 = None
        view_270 = torch.ops.aten.view.default(bmm_23, [4, 8, 1024, 64]);  bmm_23 = None
        permute_124 = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_89 = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_271 = torch.ops.aten.view.default(clone_89, [4, -1, 512]);  clone_89 = None
        permute_125 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        view_272 = torch.ops.aten.view.default(view_271, [4096, 512]);  view_271 = None
        mm_63 = torch.ops.aten.mm.default(view_272, permute_125);  view_272 = permute_125 = None
        view_273 = torch.ops.aten.view.default(mm_63, [4, 1024, 512]);  mm_63 = None
        add_60 = torch.ops.aten.add.Tensor(add_57, view_273);  add_57 = view_273 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_61 = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_49 = torch.ops.aten.mul.Tensor(add_60, rsqrt_21);  rsqrt_21 = None
        mul_50 = torch.ops.aten.mul.Tensor(arg92_1, mul_49);  arg92_1 = mul_49 = None
        permute_126 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        view_274 = torch.ops.aten.view.default(mul_50, [4096, 512]);  mul_50 = None
        mm_64 = torch.ops.aten.mm.default(view_274, permute_126);  view_274 = permute_126 = None
        view_275 = torch.ops.aten.view.default(mm_64, [4, 1024, 2048]);  mm_64 = None
        relu_8 = torch.ops.aten.relu.default(view_275);  view_275 = None
        permute_127 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        view_276 = torch.ops.aten.view.default(relu_8, [4096, 2048]);  relu_8 = None
        mm_65 = torch.ops.aten.mm.default(view_276, permute_127);  view_276 = permute_127 = None
        view_277 = torch.ops.aten.view.default(mm_65, [4, 1024, 512]);  mm_65 = None
        add_62 = torch.ops.aten.add.Tensor(add_60, view_277);  add_60 = view_277 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_63 = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_51 = torch.ops.aten.mul.Tensor(add_62, rsqrt_22);  rsqrt_22 = None
        mul_52 = torch.ops.aten.mul.Tensor(arg97_1, mul_51);  arg97_1 = mul_51 = None
        permute_128 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        view_278 = torch.ops.aten.view.default(mul_52, [4096, 512])
        mm_66 = torch.ops.aten.mm.default(view_278, permute_128);  view_278 = permute_128 = None
        view_279 = torch.ops.aten.view.default(mm_66, [4, 1024, 512]);  mm_66 = None
        view_280 = torch.ops.aten.view.default(view_279, [4, -1, 8, 64]);  view_279 = None
        permute_129 = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        permute_130 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        view_281 = torch.ops.aten.view.default(mul_52, [4096, 512])
        mm_67 = torch.ops.aten.mm.default(view_281, permute_130);  view_281 = permute_130 = None
        view_282 = torch.ops.aten.view.default(mm_67, [4, 1024, 512]);  mm_67 = None
        view_283 = torch.ops.aten.view.default(view_282, [4, -1, 8, 64]);  view_282 = None
        permute_131 = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        permute_132 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        view_284 = torch.ops.aten.view.default(mul_52, [4096, 512]);  mul_52 = None
        mm_68 = torch.ops.aten.mm.default(view_284, permute_132);  view_284 = permute_132 = None
        view_285 = torch.ops.aten.view.default(mm_68, [4, 1024, 512]);  mm_68 = None
        view_286 = torch.ops.aten.view.default(view_285, [4, -1, 8, 64]);  view_285 = None
        permute_133 = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        permute_134 = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
        expand_48 = torch.ops.aten.expand.default(permute_129, [4, 8, 1024, 64]);  permute_129 = None
        clone_93 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_287 = torch.ops.aten.view.default(clone_93, [32, 1024, 64]);  clone_93 = None
        expand_49 = torch.ops.aten.expand.default(permute_134, [4, 8, 64, 1024]);  permute_134 = None
        clone_94 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_288 = torch.ops.aten.view.default(clone_94, [32, 64, 1024]);  clone_94 = None
        bmm_24 = torch.ops.aten.bmm.default(view_287, view_288);  view_287 = view_288 = None
        view_289 = torch.ops.aten.view.default(bmm_24, [4, 8, 1024, 1024]);  bmm_24 = None
        add_64 = torch.ops.aten.add.Tensor(view_289, add_38);  view_289 = None
        view_290 = torch.ops.aten.view.default(add_64, [32, 1024, 1024]);  add_64 = None
        view_291 = torch.ops.aten.view.default(view_290, [4, 8, 1024, 1024]);  view_290 = None
        amax_12 = torch.ops.aten.amax.default(view_291, [-1], True)
        sub_17 = torch.ops.aten.sub.Tensor(view_291, amax_12);  view_291 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        expand_50 = torch.ops.aten.expand.default(div_16, [4, 8, 1024, 1024]);  div_16 = None
        view_292 = torch.ops.aten.view.default(expand_50, [32, 1024, 1024]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(permute_133, [4, 8, 1024, 64])
        clone_96 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_293 = torch.ops.aten.view.default(clone_96, [32, 1024, 64]);  clone_96 = None
        bmm_25 = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
        view_294 = torch.ops.aten.view.default(bmm_25, [4, 8, 1024, 64]);  bmm_25 = None
        permute_135 = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_97 = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_295 = torch.ops.aten.view.default(clone_97, [4, -1, 512]);  clone_97 = None
        permute_136 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        view_296 = torch.ops.aten.view.default(view_295, [4096, 512]);  view_295 = None
        mm_69 = torch.ops.aten.mm.default(view_296, permute_136);  view_296 = permute_136 = None
        view_297 = torch.ops.aten.view.default(mm_69, [4, 1024, 512]);  mm_69 = None
        add_65 = torch.ops.aten.add.Tensor(add_62, view_297);  add_62 = view_297 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        add_66 = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_65, rsqrt_23);  rsqrt_23 = None
        mul_54 = torch.ops.aten.mul.Tensor(arg102_1, mul_53);  arg102_1 = mul_53 = None
        permute_137 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        view_298 = torch.ops.aten.view.default(mul_54, [4096, 512]);  mul_54 = None
        mm_70 = torch.ops.aten.mm.default(view_298, permute_137);  view_298 = permute_137 = None
        view_299 = torch.ops.aten.view.default(mm_70, [4, 1024, 512]);  mm_70 = None
        view_300 = torch.ops.aten.view.default(view_299, [4, -1, 8, 64]);  view_299 = None
        permute_138 = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        permute_139 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        view_301 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_71 = torch.ops.aten.mm.default(view_301, permute_139);  view_301 = permute_139 = None
        view_302 = torch.ops.aten.view.default(mm_71, [4, 1024, 512]);  mm_71 = None
        view_303 = torch.ops.aten.view.default(view_302, [4, -1, 8, 64]);  view_302 = None
        permute_140 = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        permute_141 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        view_304 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_72 = torch.ops.aten.mm.default(view_304, permute_141);  view_304 = permute_141 = None
        view_305 = torch.ops.aten.view.default(mm_72, [4, 1024, 512]);  mm_72 = None
        view_306 = torch.ops.aten.view.default(view_305, [4, -1, 8, 64]);  view_305 = None
        permute_142 = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        permute_143 = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2])
        expand_52 = torch.ops.aten.expand.default(permute_138, [4, 8, 1024, 64]);  permute_138 = None
        clone_99 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_307 = torch.ops.aten.view.default(clone_99, [32, 1024, 64]);  clone_99 = None
        expand_53 = torch.ops.aten.expand.default(permute_143, [4, 8, 64, 1024]);  permute_143 = None
        clone_100 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_308 = torch.ops.aten.view.default(clone_100, [32, 64, 1024]);  clone_100 = None
        bmm_26 = torch.ops.aten.bmm.default(view_307, view_308);  view_307 = view_308 = None
        view_309 = torch.ops.aten.view.default(bmm_26, [4, 8, 1024, 1024]);  bmm_26 = None
        view_310 = torch.ops.aten.view.default(view_309, [32, 1024, 1024]);  view_309 = None
        view_311 = torch.ops.aten.view.default(view_310, [4, 8, 1024, 1024]);  view_310 = None
        amax_13 = torch.ops.aten.amax.default(view_311, [-1], True)
        sub_18 = torch.ops.aten.sub.Tensor(view_311, amax_13);  view_311 = amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        expand_54 = torch.ops.aten.expand.default(div_17, [4, 8, 1024, 1024]);  div_17 = None
        view_312 = torch.ops.aten.view.default(expand_54, [32, 1024, 1024]);  expand_54 = None
        expand_55 = torch.ops.aten.expand.default(permute_142, [4, 8, 1024, 64])
        clone_102 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_313 = torch.ops.aten.view.default(clone_102, [32, 1024, 64]);  clone_102 = None
        bmm_27 = torch.ops.aten.bmm.default(view_312, view_313);  view_312 = view_313 = None
        view_314 = torch.ops.aten.view.default(bmm_27, [4, 8, 1024, 64]);  bmm_27 = None
        permute_144 = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_103 = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_315 = torch.ops.aten.view.default(clone_103, [4, -1, 512]);  clone_103 = None
        permute_145 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        view_316 = torch.ops.aten.view.default(view_315, [4096, 512]);  view_315 = None
        mm_73 = torch.ops.aten.mm.default(view_316, permute_145);  view_316 = permute_145 = None
        view_317 = torch.ops.aten.view.default(mm_73, [4, 1024, 512]);  mm_73 = None
        add_68 = torch.ops.aten.add.Tensor(add_65, view_317);  add_65 = view_317 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_69 = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_55 = torch.ops.aten.mul.Tensor(add_68, rsqrt_24);  rsqrt_24 = None
        mul_56 = torch.ops.aten.mul.Tensor(arg105_1, mul_55);  arg105_1 = mul_55 = None
        permute_146 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        view_318 = torch.ops.aten.view.default(mul_56, [4096, 512]);  mul_56 = None
        mm_74 = torch.ops.aten.mm.default(view_318, permute_146);  view_318 = permute_146 = None
        view_319 = torch.ops.aten.view.default(mm_74, [4, 1024, 2048]);  mm_74 = None
        relu_9 = torch.ops.aten.relu.default(view_319);  view_319 = None
        permute_147 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        view_320 = torch.ops.aten.view.default(relu_9, [4096, 2048]);  relu_9 = None
        mm_75 = torch.ops.aten.mm.default(view_320, permute_147);  view_320 = permute_147 = None
        view_321 = torch.ops.aten.view.default(mm_75, [4, 1024, 512]);  mm_75 = None
        add_70 = torch.ops.aten.add.Tensor(add_68, view_321);  add_68 = view_321 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_71 = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_57 = torch.ops.aten.mul.Tensor(add_70, rsqrt_25);  rsqrt_25 = None
        mul_58 = torch.ops.aten.mul.Tensor(arg110_1, mul_57);  arg110_1 = mul_57 = None
        permute_148 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        view_322 = torch.ops.aten.view.default(mul_58, [4096, 512])
        mm_76 = torch.ops.aten.mm.default(view_322, permute_148);  view_322 = permute_148 = None
        view_323 = torch.ops.aten.view.default(mm_76, [4, 1024, 512]);  mm_76 = None
        view_324 = torch.ops.aten.view.default(view_323, [4, -1, 8, 64]);  view_323 = None
        permute_149 = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        permute_150 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        view_325 = torch.ops.aten.view.default(mul_58, [4096, 512])
        mm_77 = torch.ops.aten.mm.default(view_325, permute_150);  view_325 = permute_150 = None
        view_326 = torch.ops.aten.view.default(mm_77, [4, 1024, 512]);  mm_77 = None
        view_327 = torch.ops.aten.view.default(view_326, [4, -1, 8, 64]);  view_326 = None
        permute_151 = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        permute_152 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        view_328 = torch.ops.aten.view.default(mul_58, [4096, 512]);  mul_58 = None
        mm_78 = torch.ops.aten.mm.default(view_328, permute_152);  view_328 = permute_152 = None
        view_329 = torch.ops.aten.view.default(mm_78, [4, 1024, 512]);  mm_78 = None
        view_330 = torch.ops.aten.view.default(view_329, [4, -1, 8, 64]);  view_329 = None
        permute_153 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        permute_154 = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2])
        expand_56 = torch.ops.aten.expand.default(permute_149, [4, 8, 1024, 64]);  permute_149 = None
        clone_107 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_331 = torch.ops.aten.view.default(clone_107, [32, 1024, 64]);  clone_107 = None
        expand_57 = torch.ops.aten.expand.default(permute_154, [4, 8, 64, 1024]);  permute_154 = None
        clone_108 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_332 = torch.ops.aten.view.default(clone_108, [32, 64, 1024]);  clone_108 = None
        bmm_28 = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
        view_333 = torch.ops.aten.view.default(bmm_28, [4, 8, 1024, 1024]);  bmm_28 = None
        add_72 = torch.ops.aten.add.Tensor(view_333, add_38);  view_333 = None
        view_334 = torch.ops.aten.view.default(add_72, [32, 1024, 1024]);  add_72 = None
        view_335 = torch.ops.aten.view.default(view_334, [4, 8, 1024, 1024]);  view_334 = None
        amax_14 = torch.ops.aten.amax.default(view_335, [-1], True)
        sub_19 = torch.ops.aten.sub.Tensor(view_335, amax_14);  view_335 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        expand_58 = torch.ops.aten.expand.default(div_18, [4, 8, 1024, 1024]);  div_18 = None
        view_336 = torch.ops.aten.view.default(expand_58, [32, 1024, 1024]);  expand_58 = None
        expand_59 = torch.ops.aten.expand.default(permute_153, [4, 8, 1024, 64])
        clone_110 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_337 = torch.ops.aten.view.default(clone_110, [32, 1024, 64]);  clone_110 = None
        bmm_29 = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
        view_338 = torch.ops.aten.view.default(bmm_29, [4, 8, 1024, 64]);  bmm_29 = None
        permute_155 = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_111 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_339 = torch.ops.aten.view.default(clone_111, [4, -1, 512]);  clone_111 = None
        permute_156 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        view_340 = torch.ops.aten.view.default(view_339, [4096, 512]);  view_339 = None
        mm_79 = torch.ops.aten.mm.default(view_340, permute_156);  view_340 = permute_156 = None
        view_341 = torch.ops.aten.view.default(mm_79, [4, 1024, 512]);  mm_79 = None
        add_73 = torch.ops.aten.add.Tensor(add_70, view_341);  add_70 = view_341 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(add_73, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_74 = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_59 = torch.ops.aten.mul.Tensor(add_73, rsqrt_26);  rsqrt_26 = None
        mul_60 = torch.ops.aten.mul.Tensor(arg115_1, mul_59);  arg115_1 = mul_59 = None
        permute_157 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        view_342 = torch.ops.aten.view.default(mul_60, [4096, 512]);  mul_60 = None
        mm_80 = torch.ops.aten.mm.default(view_342, permute_157);  view_342 = permute_157 = None
        view_343 = torch.ops.aten.view.default(mm_80, [4, 1024, 512]);  mm_80 = None
        view_344 = torch.ops.aten.view.default(view_343, [4, -1, 8, 64]);  view_343 = None
        permute_158 = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        permute_159 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        view_345 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_81 = torch.ops.aten.mm.default(view_345, permute_159);  view_345 = permute_159 = None
        view_346 = torch.ops.aten.view.default(mm_81, [4, 1024, 512]);  mm_81 = None
        view_347 = torch.ops.aten.view.default(view_346, [4, -1, 8, 64]);  view_346 = None
        permute_160 = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        permute_161 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        view_348 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_82 = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
        view_349 = torch.ops.aten.view.default(mm_82, [4, 1024, 512]);  mm_82 = None
        view_350 = torch.ops.aten.view.default(view_349, [4, -1, 8, 64]);  view_349 = None
        permute_162 = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        permute_163 = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2])
        expand_60 = torch.ops.aten.expand.default(permute_158, [4, 8, 1024, 64]);  permute_158 = None
        clone_113 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_351 = torch.ops.aten.view.default(clone_113, [32, 1024, 64]);  clone_113 = None
        expand_61 = torch.ops.aten.expand.default(permute_163, [4, 8, 64, 1024]);  permute_163 = None
        clone_114 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_352 = torch.ops.aten.view.default(clone_114, [32, 64, 1024]);  clone_114 = None
        bmm_30 = torch.ops.aten.bmm.default(view_351, view_352);  view_351 = view_352 = None
        view_353 = torch.ops.aten.view.default(bmm_30, [4, 8, 1024, 1024]);  bmm_30 = None
        view_354 = torch.ops.aten.view.default(view_353, [32, 1024, 1024]);  view_353 = None
        view_355 = torch.ops.aten.view.default(view_354, [4, 8, 1024, 1024]);  view_354 = None
        amax_15 = torch.ops.aten.amax.default(view_355, [-1], True)
        sub_20 = torch.ops.aten.sub.Tensor(view_355, amax_15);  view_355 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        expand_62 = torch.ops.aten.expand.default(div_19, [4, 8, 1024, 1024]);  div_19 = None
        view_356 = torch.ops.aten.view.default(expand_62, [32, 1024, 1024]);  expand_62 = None
        expand_63 = torch.ops.aten.expand.default(permute_162, [4, 8, 1024, 64])
        clone_116 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_357 = torch.ops.aten.view.default(clone_116, [32, 1024, 64]);  clone_116 = None
        bmm_31 = torch.ops.aten.bmm.default(view_356, view_357);  view_356 = view_357 = None
        view_358 = torch.ops.aten.view.default(bmm_31, [4, 8, 1024, 64]);  bmm_31 = None
        permute_164 = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_117 = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_359 = torch.ops.aten.view.default(clone_117, [4, -1, 512]);  clone_117 = None
        permute_165 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        view_360 = torch.ops.aten.view.default(view_359, [4096, 512]);  view_359 = None
        mm_83 = torch.ops.aten.mm.default(view_360, permute_165);  view_360 = permute_165 = None
        view_361 = torch.ops.aten.view.default(mm_83, [4, 1024, 512]);  mm_83 = None
        add_76 = torch.ops.aten.add.Tensor(add_73, view_361);  add_73 = view_361 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(add_76, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_77 = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_61 = torch.ops.aten.mul.Tensor(add_76, rsqrt_27);  rsqrt_27 = None
        mul_62 = torch.ops.aten.mul.Tensor(arg118_1, mul_61);  arg118_1 = mul_61 = None
        permute_166 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        view_362 = torch.ops.aten.view.default(mul_62, [4096, 512]);  mul_62 = None
        mm_84 = torch.ops.aten.mm.default(view_362, permute_166);  view_362 = permute_166 = None
        view_363 = torch.ops.aten.view.default(mm_84, [4, 1024, 2048]);  mm_84 = None
        relu_10 = torch.ops.aten.relu.default(view_363);  view_363 = None
        permute_167 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        view_364 = torch.ops.aten.view.default(relu_10, [4096, 2048]);  relu_10 = None
        mm_85 = torch.ops.aten.mm.default(view_364, permute_167);  view_364 = permute_167 = None
        view_365 = torch.ops.aten.view.default(mm_85, [4, 1024, 512]);  mm_85 = None
        add_78 = torch.ops.aten.add.Tensor(add_76, view_365);  add_76 = view_365 = None
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        add_79 = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_63 = torch.ops.aten.mul.Tensor(add_78, rsqrt_28);  rsqrt_28 = None
        mul_64 = torch.ops.aten.mul.Tensor(arg123_1, mul_63);  arg123_1 = mul_63 = None
        permute_168 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        view_366 = torch.ops.aten.view.default(mul_64, [4096, 512])
        mm_86 = torch.ops.aten.mm.default(view_366, permute_168);  view_366 = permute_168 = None
        view_367 = torch.ops.aten.view.default(mm_86, [4, 1024, 512]);  mm_86 = None
        view_368 = torch.ops.aten.view.default(view_367, [4, -1, 8, 64]);  view_367 = None
        permute_169 = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        permute_170 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        view_369 = torch.ops.aten.view.default(mul_64, [4096, 512])
        mm_87 = torch.ops.aten.mm.default(view_369, permute_170);  view_369 = permute_170 = None
        view_370 = torch.ops.aten.view.default(mm_87, [4, 1024, 512]);  mm_87 = None
        view_371 = torch.ops.aten.view.default(view_370, [4, -1, 8, 64]);  view_370 = None
        permute_171 = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        permute_172 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        view_372 = torch.ops.aten.view.default(mul_64, [4096, 512]);  mul_64 = None
        mm_88 = torch.ops.aten.mm.default(view_372, permute_172);  view_372 = permute_172 = None
        view_373 = torch.ops.aten.view.default(mm_88, [4, 1024, 512]);  mm_88 = None
        view_374 = torch.ops.aten.view.default(view_373, [4, -1, 8, 64]);  view_373 = None
        permute_173 = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        permute_174 = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2])
        expand_64 = torch.ops.aten.expand.default(permute_169, [4, 8, 1024, 64]);  permute_169 = None
        clone_121 = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_375 = torch.ops.aten.view.default(clone_121, [32, 1024, 64]);  clone_121 = None
        expand_65 = torch.ops.aten.expand.default(permute_174, [4, 8, 64, 1024]);  permute_174 = None
        clone_122 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_376 = torch.ops.aten.view.default(clone_122, [32, 64, 1024]);  clone_122 = None
        bmm_32 = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
        view_377 = torch.ops.aten.view.default(bmm_32, [4, 8, 1024, 1024]);  bmm_32 = None
        add_80 = torch.ops.aten.add.Tensor(view_377, add_38);  view_377 = add_38 = None
        view_378 = torch.ops.aten.view.default(add_80, [32, 1024, 1024]);  add_80 = None
        view_379 = torch.ops.aten.view.default(view_378, [4, 8, 1024, 1024]);  view_378 = None
        amax_16 = torch.ops.aten.amax.default(view_379, [-1], True)
        sub_21 = torch.ops.aten.sub.Tensor(view_379, amax_16);  view_379 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        expand_66 = torch.ops.aten.expand.default(div_20, [4, 8, 1024, 1024]);  div_20 = None
        view_380 = torch.ops.aten.view.default(expand_66, [32, 1024, 1024]);  expand_66 = None
        expand_67 = torch.ops.aten.expand.default(permute_173, [4, 8, 1024, 64])
        clone_124 = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_381 = torch.ops.aten.view.default(clone_124, [32, 1024, 64]);  clone_124 = None
        bmm_33 = torch.ops.aten.bmm.default(view_380, view_381);  view_380 = view_381 = None
        view_382 = torch.ops.aten.view.default(bmm_33, [4, 8, 1024, 64]);  bmm_33 = None
        permute_175 = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_125 = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_383 = torch.ops.aten.view.default(clone_125, [4, -1, 512]);  clone_125 = None
        permute_176 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        view_384 = torch.ops.aten.view.default(view_383, [4096, 512]);  view_383 = None
        mm_89 = torch.ops.aten.mm.default(view_384, permute_176);  view_384 = permute_176 = None
        view_385 = torch.ops.aten.view.default(mm_89, [4, 1024, 512]);  mm_89 = None
        add_81 = torch.ops.aten.add.Tensor(add_78, view_385);  add_78 = view_385 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_82 = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_65 = torch.ops.aten.mul.Tensor(add_81, rsqrt_29);  rsqrt_29 = None
        mul_66 = torch.ops.aten.mul.Tensor(arg128_1, mul_65);  arg128_1 = mul_65 = None
        permute_177 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        view_386 = torch.ops.aten.view.default(mul_66, [4096, 512]);  mul_66 = None
        mm_90 = torch.ops.aten.mm.default(view_386, permute_177);  view_386 = permute_177 = None
        view_387 = torch.ops.aten.view.default(mm_90, [4, 1024, 512]);  mm_90 = None
        view_388 = torch.ops.aten.view.default(view_387, [4, -1, 8, 64]);  view_387 = None
        permute_178 = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        permute_179 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        view_389 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_91 = torch.ops.aten.mm.default(view_389, permute_179);  view_389 = permute_179 = None
        view_390 = torch.ops.aten.view.default(mm_91, [4, 1024, 512]);  mm_91 = None
        view_391 = torch.ops.aten.view.default(view_390, [4, -1, 8, 64]);  view_390 = None
        permute_180 = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
        permute_181 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        view_392 = torch.ops.aten.view.default(mul_28, [4096, 512])
        mm_92 = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
        view_393 = torch.ops.aten.view.default(mm_92, [4, 1024, 512]);  mm_92 = None
        view_394 = torch.ops.aten.view.default(view_393, [4, -1, 8, 64]);  view_393 = None
        permute_182 = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
        permute_183 = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
        expand_68 = torch.ops.aten.expand.default(permute_178, [4, 8, 1024, 64]);  permute_178 = None
        clone_127 = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_395 = torch.ops.aten.view.default(clone_127, [32, 1024, 64]);  clone_127 = None
        expand_69 = torch.ops.aten.expand.default(permute_183, [4, 8, 64, 1024]);  permute_183 = None
        clone_128 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_396 = torch.ops.aten.view.default(clone_128, [32, 64, 1024]);  clone_128 = None
        bmm_34 = torch.ops.aten.bmm.default(view_395, view_396);  view_395 = view_396 = None
        view_397 = torch.ops.aten.view.default(bmm_34, [4, 8, 1024, 1024]);  bmm_34 = None
        view_398 = torch.ops.aten.view.default(view_397, [32, 1024, 1024]);  view_397 = None
        view_399 = torch.ops.aten.view.default(view_398, [4, 8, 1024, 1024]);  view_398 = None
        amax_17 = torch.ops.aten.amax.default(view_399, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(view_399, amax_17);  view_399 = amax_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        expand_70 = torch.ops.aten.expand.default(div_21, [4, 8, 1024, 1024]);  div_21 = None
        view_400 = torch.ops.aten.view.default(expand_70, [32, 1024, 1024]);  expand_70 = None
        expand_71 = torch.ops.aten.expand.default(permute_182, [4, 8, 1024, 64])
        clone_130 = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_401 = torch.ops.aten.view.default(clone_130, [32, 1024, 64]);  clone_130 = None
        bmm_35 = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
        view_402 = torch.ops.aten.view.default(bmm_35, [4, 8, 1024, 64]);  bmm_35 = None
        permute_184 = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_131 = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_403 = torch.ops.aten.view.default(clone_131, [4, -1, 512]);  clone_131 = None
        permute_185 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        view_404 = torch.ops.aten.view.default(view_403, [4096, 512]);  view_403 = None
        mm_93 = torch.ops.aten.mm.default(view_404, permute_185);  view_404 = permute_185 = None
        view_405 = torch.ops.aten.view.default(mm_93, [4, 1024, 512]);  mm_93 = None
        add_84 = torch.ops.aten.add.Tensor(add_81, view_405);  add_81 = view_405 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_85 = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_67 = torch.ops.aten.mul.Tensor(add_84, rsqrt_30);  rsqrt_30 = None
        mul_68 = torch.ops.aten.mul.Tensor(arg131_1, mul_67);  arg131_1 = mul_67 = None
        permute_186 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        view_406 = torch.ops.aten.view.default(mul_68, [4096, 512]);  mul_68 = None
        mm_94 = torch.ops.aten.mm.default(view_406, permute_186);  view_406 = permute_186 = None
        view_407 = torch.ops.aten.view.default(mm_94, [4, 1024, 2048]);  mm_94 = None
        relu_11 = torch.ops.aten.relu.default(view_407);  view_407 = None
        permute_187 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        view_408 = torch.ops.aten.view.default(relu_11, [4096, 2048]);  relu_11 = None
        mm_95 = torch.ops.aten.mm.default(view_408, permute_187);  view_408 = permute_187 = None
        view_409 = torch.ops.aten.view.default(mm_95, [4, 1024, 512]);  mm_95 = None
        add_86 = torch.ops.aten.add.Tensor(add_84, view_409);  add_84 = view_409 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_87 = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_69 = torch.ops.aten.mul.Tensor(add_86, rsqrt_31);  add_86 = rsqrt_31 = None
        mul_70 = torch.ops.aten.mul.Tensor(arg132_1, mul_69);  arg132_1 = mul_69 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, 0.04419417382415922);  mul_70 = None
        permute_188 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_410 = torch.ops.aten.view.default(mul_71, [4096, 512]);  mul_71 = None
        mm_96 = torch.ops.aten.mm.default(view_410, permute_188);  view_410 = permute_188 = None
        view_411 = torch.ops.aten.view.default(mm_96, [4, 1024, 32128]);  mm_96 = None
        view_412 = torch.ops.aten.view.default(view_411, [-1, 32128])
        view_413 = torch.ops.aten.view.default(arg52_1, [-1]);  arg52_1 = None
        amax_18 = torch.ops.aten.amax.default(view_412, [1], True)
        sub_23 = torch.ops.aten.sub.Tensor(view_412, amax_18);  view_412 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_23)
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
        log_2 = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_24 = torch.ops.aten.sub.Tensor(sub_23, log_2);  sub_23 = log_2 = None
        ne = torch.ops.aten.ne.Scalar(view_413, -100)
        full_default_6 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne, view_413, full_default_6);  ne = full_default_6 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather = torch.ops.aten.gather.default(sub_24, 1, unsqueeze_17);  sub_24 = unsqueeze_17 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1 = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_413, -100)
        full_default_7 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_1, neg_1, full_default_7);  ne_1 = neg_1 = full_default_7 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_413, -100);  view_413 = None
        sum_20 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        sum_21 = torch.ops.aten.sum.default(where_3);  where_3 = None
        div_22 = torch.ops.aten.div.Tensor(sum_21, convert_element_type_7);  sum_21 = convert_element_type_7 = None
        return (div_22, view_411, permute_70, permute_72, permute_80, permute_82, permute_91, permute_93, permute_100, permute_102, permute_111, permute_113, permute_120, permute_122, permute_131, permute_133, permute_140, permute_142, permute_151, permute_153, permute_160, permute_162, permute_171, permute_173, permute_180, permute_182, mul_28)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 65798144, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32128, 512), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 512), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 512), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf4, (512, 512), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 512), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf6, (32, 8), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2048, 512), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512, 2048), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512, 512), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512, 512), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512, 512), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512, 512), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf16, (2048, 512), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512, 2048), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512, 512), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512, 512), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512, 512), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512, 512), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf24, (2048, 512), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512, 2048), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512, 512), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512, 512), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512, 512), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512, 512), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf32, (2048, 512), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512, 2048), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512, 512), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512, 512), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512, 512), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512, 512), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf40, (2048, 512), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf41, (512, 2048), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512, 512), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512, 512), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512, 512), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512, 512), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf48, (2048, 512), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512, 2048), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf52, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf53, (512, 512), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf54, (512, 512), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512, 512), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512, 512), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (32, 8), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf59, (512, 512), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf60, (512, 512), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512, 512), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512, 512), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf64, (2048, 512), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512, 2048), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512, 512), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512, 512), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512, 512), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512, 512), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512, 512), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 512), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512, 512), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512, 512), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf77, (2048, 512), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512, 2048), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512, 512), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512, 512), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512, 512), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512, 512), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512, 512), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512, 512), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512, 512), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512, 512), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2048, 512), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512, 2048), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512, 512), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf94, (512, 512), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512, 512), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf96, (512, 512), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512, 512), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512, 512), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512, 512), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512, 512), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf103, (2048, 512), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512, 2048), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512, 512), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf107, (512, 512), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf108, (512, 512), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512, 512), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512, 512), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512, 512), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512, 512), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512, 512), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf116, (2048, 512), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512, 2048), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512, 512), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512, 512), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf121, (512, 512), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf122, (512, 512), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf123, (512,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf124, (512, 512), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf125, (512, 512), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf126, (512, 512), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf127, (512, 512), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf129, (2048, 512), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512, 2048), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf131, (512,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512,), is_leaf=True)  # arg132_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)