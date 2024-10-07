
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 128])
        embedding = torch.ops.aten.embedding.default(arg1_1, view);  view = None
        full = torch.ops.aten.full.default([16, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default = torch.ops.aten.full.default([16, 1, 1, 128], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(embedding, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1 = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(arg7_1, mul_1);  arg7_1 = mul_1 = None
        permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_1 = torch.ops.aten.view.default(mul_2, [2048, 512])
        mm = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2 = torch.ops.aten.view.default(mm, [16, 128, 384]);  mm = None
        view_3 = torch.ops.aten.view.default(view_2, [16, -1, 6, 64]);  view_2 = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_2 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        view_4 = torch.ops.aten.view.default(mul_2, [2048, 512])
        mm_1 = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(mm_1, [16, 128, 384]);  mm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [16, -1, 6, 64]);  view_5 = None
        permute_3 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_7 = torch.ops.aten.view.default(mul_2, [2048, 512]);  mul_2 = None
        mm_2 = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8 = torch.ops.aten.view.default(mm_2, [16, 128, 384]);  mm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [16, -1, 6, 64]);  view_8 = None
        permute_5 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        permute_6 = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand = torch.ops.aten.expand.default(permute_1, [16, 6, 128, 64]);  permute_1 = None
        clone_1 = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_10 = torch.ops.aten.view.default(clone_1, [96, 128, 64]);  clone_1 = None
        expand_1 = torch.ops.aten.expand.default(permute_6, [16, 6, 64, 128]);  permute_6 = None
        clone_2 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_11 = torch.ops.aten.view.default(clone_2, [96, 64, 128]);  clone_2 = None
        bmm = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12 = torch.ops.aten.view.default(bmm, [16, 6, 128, 128]);  bmm = None
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        iota_1 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
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
        full_default_1 = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum = torch.ops.aten.minimum.default(add_2, full_default_1);  add_2 = full_default_1 = None
        where = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_3 = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
        embedding_1 = torch.ops.aten.embedding.default(arg6_1, add_3);  arg6_1 = add_3 = None
        permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        add_4 = torch.ops.aten.add.Tensor(unsqueeze_4, full_default);  unsqueeze_4 = full_default = None
        add_5 = torch.ops.aten.add.Tensor(view_12, add_4);  view_12 = None
        view_13 = torch.ops.aten.view.default(add_5, [96, 128, 128]);  add_5 = None
        view_14 = torch.ops.aten.view.default(view_13, [16, 6, 128, 128]);  view_13 = None
        amax = torch.ops.aten.amax.default(view_14, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2 = torch.ops.aten.expand.default(div_2, [16, 6, 128, 128]);  div_2 = None
        view_15 = torch.ops.aten.view.default(expand_2, [96, 128, 128]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_5, [16, 6, 128, 64]);  permute_5 = None
        clone_4 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_16 = torch.ops.aten.view.default(clone_4, [96, 128, 64]);  clone_4 = None
        bmm_1 = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17 = torch.ops.aten.view.default(bmm_1, [16, 6, 128, 64]);  bmm_1 = None
        permute_8 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_5 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_18 = torch.ops.aten.view.default(clone_5, [16, -1, 384]);  clone_5 = None
        permute_9 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        view_19 = torch.ops.aten.view.default(view_18, [2048, 384]);  view_18 = None
        mm_3 = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20 = torch.ops.aten.view.default(mm_3, [16, 128, 512]);  mm_3 = None
        add_6 = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_7 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_6, rsqrt_1);  rsqrt_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(arg11_1, mul_5);  arg11_1 = mul_5 = None
        permute_10 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        view_21 = torch.ops.aten.view.default(mul_6, [2048, 512])
        mm_4 = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22 = torch.ops.aten.view.default(mm_4, [16, 128, 1024]);  mm_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_8 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_8 = torch.ops.aten.add.Tensor(view_22, mul_8);  view_22 = mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh = torch.ops.aten.tanh.default(mul_9);  mul_9 = None
        add_9 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
        permute_11 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        view_23 = torch.ops.aten.view.default(mul_6, [2048, 512]);  mul_6 = None
        mm_5 = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24 = torch.ops.aten.view.default(mm_5, [16, 128, 1024]);  mm_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, view_24);  mul_10 = view_24 = None
        permute_12 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        view_25 = torch.ops.aten.view.default(mul_11, [2048, 1024]);  mul_11 = None
        mm_6 = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26 = torch.ops.aten.view.default(mm_6, [16, 128, 512]);  mm_6 = None
        add_10 = torch.ops.aten.add.Tensor(add_6, view_26);  add_6 = view_26 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_10, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_11 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(add_10, rsqrt_2);  rsqrt_2 = None
        mul_13 = torch.ops.aten.mul.Tensor(arg16_1, mul_12);  arg16_1 = mul_12 = None
        permute_13 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        view_27 = torch.ops.aten.view.default(mul_13, [2048, 512])
        mm_7 = torch.ops.aten.mm.default(view_27, permute_13);  view_27 = permute_13 = None
        view_28 = torch.ops.aten.view.default(mm_7, [16, 128, 384]);  mm_7 = None
        view_29 = torch.ops.aten.view.default(view_28, [16, -1, 6, 64]);  view_28 = None
        permute_14 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        permute_15 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        view_30 = torch.ops.aten.view.default(mul_13, [2048, 512])
        mm_8 = torch.ops.aten.mm.default(view_30, permute_15);  view_30 = permute_15 = None
        view_31 = torch.ops.aten.view.default(mm_8, [16, 128, 384]);  mm_8 = None
        view_32 = torch.ops.aten.view.default(view_31, [16, -1, 6, 64]);  view_31 = None
        permute_16 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        permute_17 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        view_33 = torch.ops.aten.view.default(mul_13, [2048, 512]);  mul_13 = None
        mm_9 = torch.ops.aten.mm.default(view_33, permute_17);  view_33 = permute_17 = None
        view_34 = torch.ops.aten.view.default(mm_9, [16, 128, 384]);  mm_9 = None
        view_35 = torch.ops.aten.view.default(view_34, [16, -1, 6, 64]);  view_34 = None
        permute_18 = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        permute_19 = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_4 = torch.ops.aten.expand.default(permute_14, [16, 6, 128, 64]);  permute_14 = None
        clone_9 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_36 = torch.ops.aten.view.default(clone_9, [96, 128, 64]);  clone_9 = None
        expand_5 = torch.ops.aten.expand.default(permute_19, [16, 6, 64, 128]);  permute_19 = None
        clone_10 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_37 = torch.ops.aten.view.default(clone_10, [96, 64, 128]);  clone_10 = None
        bmm_2 = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38 = torch.ops.aten.view.default(bmm_2, [16, 6, 128, 128]);  bmm_2 = None
        add_12 = torch.ops.aten.add.Tensor(view_38, add_4);  view_38 = None
        view_39 = torch.ops.aten.view.default(add_12, [96, 128, 128]);  add_12 = None
        view_40 = torch.ops.aten.view.default(view_39, [16, 6, 128, 128]);  view_39 = None
        amax_1 = torch.ops.aten.amax.default(view_40, [-1], True)
        sub_3 = torch.ops.aten.sub.Tensor(view_40, amax_1);  view_40 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        expand_6 = torch.ops.aten.expand.default(div_3, [16, 6, 128, 128]);  div_3 = None
        view_41 = torch.ops.aten.view.default(expand_6, [96, 128, 128]);  expand_6 = None
        expand_7 = torch.ops.aten.expand.default(permute_18, [16, 6, 128, 64]);  permute_18 = None
        clone_12 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_42 = torch.ops.aten.view.default(clone_12, [96, 128, 64]);  clone_12 = None
        bmm_3 = torch.ops.aten.bmm.default(view_41, view_42);  view_41 = view_42 = None
        view_43 = torch.ops.aten.view.default(bmm_3, [16, 6, 128, 64]);  bmm_3 = None
        permute_20 = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        clone_13 = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_44 = torch.ops.aten.view.default(clone_13, [16, -1, 384]);  clone_13 = None
        permute_21 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        view_45 = torch.ops.aten.view.default(view_44, [2048, 384]);  view_44 = None
        mm_10 = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46 = torch.ops.aten.view.default(mm_10, [16, 128, 512]);  mm_10 = None
        add_13 = torch.ops.aten.add.Tensor(add_10, view_46);  add_10 = view_46 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_14 = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_13, rsqrt_3);  rsqrt_3 = None
        mul_15 = torch.ops.aten.mul.Tensor(arg20_1, mul_14);  arg20_1 = mul_14 = None
        permute_22 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        view_47 = torch.ops.aten.view.default(mul_15, [2048, 512])
        mm_11 = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48 = torch.ops.aten.view.default(mm_11, [16, 128, 1024]);  mm_11 = None
        mul_16 = torch.ops.aten.mul.Tensor(view_48, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_48, 3.0)
        mul_17 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_15 = torch.ops.aten.add.Tensor(view_48, mul_17);  view_48 = mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_16, add_16);  mul_16 = add_16 = None
        permute_23 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        view_49 = torch.ops.aten.view.default(mul_15, [2048, 512]);  mul_15 = None
        mm_12 = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50 = torch.ops.aten.view.default(mm_12, [16, 128, 1024]);  mm_12 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, view_50);  mul_19 = view_50 = None
        permute_24 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        view_51 = torch.ops.aten.view.default(mul_20, [2048, 1024]);  mul_20 = None
        mm_13 = torch.ops.aten.mm.default(view_51, permute_24);  view_51 = permute_24 = None
        view_52 = torch.ops.aten.view.default(mm_13, [16, 128, 512]);  mm_13 = None
        add_17 = torch.ops.aten.add.Tensor(add_13, view_52);  add_13 = view_52 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(add_17, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_18 = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_21 = torch.ops.aten.mul.Tensor(add_17, rsqrt_4);  rsqrt_4 = None
        mul_22 = torch.ops.aten.mul.Tensor(arg25_1, mul_21);  arg25_1 = mul_21 = None
        permute_25 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        view_53 = torch.ops.aten.view.default(mul_22, [2048, 512])
        mm_14 = torch.ops.aten.mm.default(view_53, permute_25);  view_53 = permute_25 = None
        view_54 = torch.ops.aten.view.default(mm_14, [16, 128, 384]);  mm_14 = None
        view_55 = torch.ops.aten.view.default(view_54, [16, -1, 6, 64]);  view_54 = None
        permute_26 = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        permute_27 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        view_56 = torch.ops.aten.view.default(mul_22, [2048, 512])
        mm_15 = torch.ops.aten.mm.default(view_56, permute_27);  view_56 = permute_27 = None
        view_57 = torch.ops.aten.view.default(mm_15, [16, 128, 384]);  mm_15 = None
        view_58 = torch.ops.aten.view.default(view_57, [16, -1, 6, 64]);  view_57 = None
        permute_28 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        permute_29 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        view_59 = torch.ops.aten.view.default(mul_22, [2048, 512]);  mul_22 = None
        mm_16 = torch.ops.aten.mm.default(view_59, permute_29);  view_59 = permute_29 = None
        view_60 = torch.ops.aten.view.default(mm_16, [16, 128, 384]);  mm_16 = None
        view_61 = torch.ops.aten.view.default(view_60, [16, -1, 6, 64]);  view_60 = None
        permute_30 = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        permute_31 = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_8 = torch.ops.aten.expand.default(permute_26, [16, 6, 128, 64]);  permute_26 = None
        clone_17 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_62 = torch.ops.aten.view.default(clone_17, [96, 128, 64]);  clone_17 = None
        expand_9 = torch.ops.aten.expand.default(permute_31, [16, 6, 64, 128]);  permute_31 = None
        clone_18 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_63 = torch.ops.aten.view.default(clone_18, [96, 64, 128]);  clone_18 = None
        bmm_4 = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
        view_64 = torch.ops.aten.view.default(bmm_4, [16, 6, 128, 128]);  bmm_4 = None
        add_19 = torch.ops.aten.add.Tensor(view_64, add_4);  view_64 = None
        view_65 = torch.ops.aten.view.default(add_19, [96, 128, 128]);  add_19 = None
        view_66 = torch.ops.aten.view.default(view_65, [16, 6, 128, 128]);  view_65 = None
        amax_2 = torch.ops.aten.amax.default(view_66, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(view_66, amax_2);  view_66 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        expand_10 = torch.ops.aten.expand.default(div_4, [16, 6, 128, 128]);  div_4 = None
        view_67 = torch.ops.aten.view.default(expand_10, [96, 128, 128]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(permute_30, [16, 6, 128, 64]);  permute_30 = None
        clone_20 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_68 = torch.ops.aten.view.default(clone_20, [96, 128, 64]);  clone_20 = None
        bmm_5 = torch.ops.aten.bmm.default(view_67, view_68);  view_67 = view_68 = None
        view_69 = torch.ops.aten.view.default(bmm_5, [16, 6, 128, 64]);  bmm_5 = None
        permute_32 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_21 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_70 = torch.ops.aten.view.default(clone_21, [16, -1, 384]);  clone_21 = None
        permute_33 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        view_71 = torch.ops.aten.view.default(view_70, [2048, 384]);  view_70 = None
        mm_17 = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72 = torch.ops.aten.view.default(mm_17, [16, 128, 512]);  mm_17 = None
        add_20 = torch.ops.aten.add.Tensor(add_17, view_72);  add_17 = view_72 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_20, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_21 = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_20, rsqrt_5);  rsqrt_5 = None
        mul_24 = torch.ops.aten.mul.Tensor(arg29_1, mul_23);  arg29_1 = mul_23 = None
        permute_34 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        view_73 = torch.ops.aten.view.default(mul_24, [2048, 512])
        mm_18 = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74 = torch.ops.aten.view.default(mm_18, [16, 128, 1024]);  mm_18 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_74, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_74, 3.0)
        mul_26 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_22 = torch.ops.aten.add.Tensor(view_74, mul_26);  view_74 = mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_27);  mul_27 = None
        add_23 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_25, add_23);  mul_25 = add_23 = None
        permute_35 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        view_75 = torch.ops.aten.view.default(mul_24, [2048, 512]);  mul_24 = None
        mm_19 = torch.ops.aten.mm.default(view_75, permute_35);  view_75 = permute_35 = None
        view_76 = torch.ops.aten.view.default(mm_19, [16, 128, 1024]);  mm_19 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, view_76);  mul_28 = view_76 = None
        permute_36 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        view_77 = torch.ops.aten.view.default(mul_29, [2048, 1024]);  mul_29 = None
        mm_20 = torch.ops.aten.mm.default(view_77, permute_36);  view_77 = permute_36 = None
        view_78 = torch.ops.aten.view.default(mm_20, [16, 128, 512]);  mm_20 = None
        add_24 = torch.ops.aten.add.Tensor(add_20, view_78);  add_20 = view_78 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_25 = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_24, rsqrt_6);  rsqrt_6 = None
        mul_31 = torch.ops.aten.mul.Tensor(arg34_1, mul_30);  arg34_1 = mul_30 = None
        permute_37 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        view_79 = torch.ops.aten.view.default(mul_31, [2048, 512])
        mm_21 = torch.ops.aten.mm.default(view_79, permute_37);  view_79 = permute_37 = None
        view_80 = torch.ops.aten.view.default(mm_21, [16, 128, 384]);  mm_21 = None
        view_81 = torch.ops.aten.view.default(view_80, [16, -1, 6, 64]);  view_80 = None
        permute_38 = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        permute_39 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        view_82 = torch.ops.aten.view.default(mul_31, [2048, 512])
        mm_22 = torch.ops.aten.mm.default(view_82, permute_39);  view_82 = permute_39 = None
        view_83 = torch.ops.aten.view.default(mm_22, [16, 128, 384]);  mm_22 = None
        view_84 = torch.ops.aten.view.default(view_83, [16, -1, 6, 64]);  view_83 = None
        permute_40 = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        permute_41 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        view_85 = torch.ops.aten.view.default(mul_31, [2048, 512]);  mul_31 = None
        mm_23 = torch.ops.aten.mm.default(view_85, permute_41);  view_85 = permute_41 = None
        view_86 = torch.ops.aten.view.default(mm_23, [16, 128, 384]);  mm_23 = None
        view_87 = torch.ops.aten.view.default(view_86, [16, -1, 6, 64]);  view_86 = None
        permute_42 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        permute_43 = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_12 = torch.ops.aten.expand.default(permute_38, [16, 6, 128, 64]);  permute_38 = None
        clone_25 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_88 = torch.ops.aten.view.default(clone_25, [96, 128, 64]);  clone_25 = None
        expand_13 = torch.ops.aten.expand.default(permute_43, [16, 6, 64, 128]);  permute_43 = None
        clone_26 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_89 = torch.ops.aten.view.default(clone_26, [96, 64, 128]);  clone_26 = None
        bmm_6 = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
        view_90 = torch.ops.aten.view.default(bmm_6, [16, 6, 128, 128]);  bmm_6 = None
        add_26 = torch.ops.aten.add.Tensor(view_90, add_4);  view_90 = None
        view_91 = torch.ops.aten.view.default(add_26, [96, 128, 128]);  add_26 = None
        view_92 = torch.ops.aten.view.default(view_91, [16, 6, 128, 128]);  view_91 = None
        amax_3 = torch.ops.aten.amax.default(view_92, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(view_92, amax_3);  view_92 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        expand_14 = torch.ops.aten.expand.default(div_5, [16, 6, 128, 128]);  div_5 = None
        view_93 = torch.ops.aten.view.default(expand_14, [96, 128, 128]);  expand_14 = None
        expand_15 = torch.ops.aten.expand.default(permute_42, [16, 6, 128, 64]);  permute_42 = None
        clone_28 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_94 = torch.ops.aten.view.default(clone_28, [96, 128, 64]);  clone_28 = None
        bmm_7 = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
        view_95 = torch.ops.aten.view.default(bmm_7, [16, 6, 128, 64]);  bmm_7 = None
        permute_44 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        clone_29 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_96 = torch.ops.aten.view.default(clone_29, [16, -1, 384]);  clone_29 = None
        permute_45 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        view_97 = torch.ops.aten.view.default(view_96, [2048, 384]);  view_96 = None
        mm_24 = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98 = torch.ops.aten.view.default(mm_24, [16, 128, 512]);  mm_24 = None
        add_27 = torch.ops.aten.add.Tensor(add_24, view_98);  add_24 = view_98 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_28 = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_32 = torch.ops.aten.mul.Tensor(add_27, rsqrt_7);  rsqrt_7 = None
        mul_33 = torch.ops.aten.mul.Tensor(arg38_1, mul_32);  arg38_1 = mul_32 = None
        permute_46 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        view_99 = torch.ops.aten.view.default(mul_33, [2048, 512])
        mm_25 = torch.ops.aten.mm.default(view_99, permute_46);  view_99 = permute_46 = None
        view_100 = torch.ops.aten.view.default(mm_25, [16, 128, 1024]);  mm_25 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_100, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_100, 3.0)
        mul_35 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_29 = torch.ops.aten.add.Tensor(view_100, mul_35);  view_100 = mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(add_29, 0.7978845608028654);  add_29 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
        add_30 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_34, add_30);  mul_34 = add_30 = None
        permute_47 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        view_101 = torch.ops.aten.view.default(mul_33, [2048, 512]);  mul_33 = None
        mm_26 = torch.ops.aten.mm.default(view_101, permute_47);  view_101 = permute_47 = None
        view_102 = torch.ops.aten.view.default(mm_26, [16, 128, 1024]);  mm_26 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, view_102);  mul_37 = view_102 = None
        permute_48 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        view_103 = torch.ops.aten.view.default(mul_38, [2048, 1024]);  mul_38 = None
        mm_27 = torch.ops.aten.mm.default(view_103, permute_48);  view_103 = permute_48 = None
        view_104 = torch.ops.aten.view.default(mm_27, [16, 128, 512]);  mm_27 = None
        add_31 = torch.ops.aten.add.Tensor(add_27, view_104);  add_27 = view_104 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_32 = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_31, rsqrt_8);  rsqrt_8 = None
        mul_40 = torch.ops.aten.mul.Tensor(arg43_1, mul_39);  arg43_1 = mul_39 = None
        permute_49 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        view_105 = torch.ops.aten.view.default(mul_40, [2048, 512])
        mm_28 = torch.ops.aten.mm.default(view_105, permute_49);  view_105 = permute_49 = None
        view_106 = torch.ops.aten.view.default(mm_28, [16, 128, 384]);  mm_28 = None
        view_107 = torch.ops.aten.view.default(view_106, [16, -1, 6, 64]);  view_106 = None
        permute_50 = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        permute_51 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        view_108 = torch.ops.aten.view.default(mul_40, [2048, 512])
        mm_29 = torch.ops.aten.mm.default(view_108, permute_51);  view_108 = permute_51 = None
        view_109 = torch.ops.aten.view.default(mm_29, [16, 128, 384]);  mm_29 = None
        view_110 = torch.ops.aten.view.default(view_109, [16, -1, 6, 64]);  view_109 = None
        permute_52 = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        permute_53 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        view_111 = torch.ops.aten.view.default(mul_40, [2048, 512]);  mul_40 = None
        mm_30 = torch.ops.aten.mm.default(view_111, permute_53);  view_111 = permute_53 = None
        view_112 = torch.ops.aten.view.default(mm_30, [16, 128, 384]);  mm_30 = None
        view_113 = torch.ops.aten.view.default(view_112, [16, -1, 6, 64]);  view_112 = None
        permute_54 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        permute_55 = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        expand_16 = torch.ops.aten.expand.default(permute_50, [16, 6, 128, 64]);  permute_50 = None
        clone_33 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_114 = torch.ops.aten.view.default(clone_33, [96, 128, 64]);  clone_33 = None
        expand_17 = torch.ops.aten.expand.default(permute_55, [16, 6, 64, 128]);  permute_55 = None
        clone_34 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_115 = torch.ops.aten.view.default(clone_34, [96, 64, 128]);  clone_34 = None
        bmm_8 = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
        view_116 = torch.ops.aten.view.default(bmm_8, [16, 6, 128, 128]);  bmm_8 = None
        add_33 = torch.ops.aten.add.Tensor(view_116, add_4);  view_116 = None
        view_117 = torch.ops.aten.view.default(add_33, [96, 128, 128]);  add_33 = None
        view_118 = torch.ops.aten.view.default(view_117, [16, 6, 128, 128]);  view_117 = None
        amax_4 = torch.ops.aten.amax.default(view_118, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(view_118, amax_4);  view_118 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        expand_18 = torch.ops.aten.expand.default(div_6, [16, 6, 128, 128]);  div_6 = None
        view_119 = torch.ops.aten.view.default(expand_18, [96, 128, 128]);  expand_18 = None
        expand_19 = torch.ops.aten.expand.default(permute_54, [16, 6, 128, 64]);  permute_54 = None
        clone_36 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_120 = torch.ops.aten.view.default(clone_36, [96, 128, 64]);  clone_36 = None
        bmm_9 = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121 = torch.ops.aten.view.default(bmm_9, [16, 6, 128, 64]);  bmm_9 = None
        permute_56 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_37 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_122 = torch.ops.aten.view.default(clone_37, [16, -1, 384]);  clone_37 = None
        permute_57 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        view_123 = torch.ops.aten.view.default(view_122, [2048, 384]);  view_122 = None
        mm_31 = torch.ops.aten.mm.default(view_123, permute_57);  view_123 = permute_57 = None
        view_124 = torch.ops.aten.view.default(mm_31, [16, 128, 512]);  mm_31 = None
        add_34 = torch.ops.aten.add.Tensor(add_31, view_124);  add_31 = view_124 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_35 = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_41 = torch.ops.aten.mul.Tensor(add_34, rsqrt_9);  rsqrt_9 = None
        mul_42 = torch.ops.aten.mul.Tensor(arg47_1, mul_41);  arg47_1 = mul_41 = None
        permute_58 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        view_125 = torch.ops.aten.view.default(mul_42, [2048, 512])
        mm_32 = torch.ops.aten.mm.default(view_125, permute_58);  view_125 = permute_58 = None
        view_126 = torch.ops.aten.view.default(mm_32, [16, 128, 1024]);  mm_32 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_126, 0.5)
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(view_126, 3.0)
        mul_44 = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_36 = torch.ops.aten.add.Tensor(view_126, mul_44);  view_126 = mul_44 = None
        mul_45 = torch.ops.aten.mul.Tensor(add_36, 0.7978845608028654);  add_36 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_45);  mul_45 = None
        add_37 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_43, add_37);  mul_43 = add_37 = None
        permute_59 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        view_127 = torch.ops.aten.view.default(mul_42, [2048, 512]);  mul_42 = None
        mm_33 = torch.ops.aten.mm.default(view_127, permute_59);  view_127 = permute_59 = None
        view_128 = torch.ops.aten.view.default(mm_33, [16, 128, 1024]);  mm_33 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, view_128);  mul_46 = view_128 = None
        permute_60 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        view_129 = torch.ops.aten.view.default(mul_47, [2048, 1024]);  mul_47 = None
        mm_34 = torch.ops.aten.mm.default(view_129, permute_60);  view_129 = permute_60 = None
        view_130 = torch.ops.aten.view.default(mm_34, [16, 128, 512]);  mm_34 = None
        add_38 = torch.ops.aten.add.Tensor(add_34, view_130);  add_34 = view_130 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(add_38, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_39 = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_48 = torch.ops.aten.mul.Tensor(add_38, rsqrt_10);  rsqrt_10 = None
        mul_49 = torch.ops.aten.mul.Tensor(arg52_1, mul_48);  arg52_1 = mul_48 = None
        permute_61 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        view_131 = torch.ops.aten.view.default(mul_49, [2048, 512])
        mm_35 = torch.ops.aten.mm.default(view_131, permute_61);  view_131 = permute_61 = None
        view_132 = torch.ops.aten.view.default(mm_35, [16, 128, 384]);  mm_35 = None
        view_133 = torch.ops.aten.view.default(view_132, [16, -1, 6, 64]);  view_132 = None
        permute_62 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        permute_63 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        view_134 = torch.ops.aten.view.default(mul_49, [2048, 512])
        mm_36 = torch.ops.aten.mm.default(view_134, permute_63);  view_134 = permute_63 = None
        view_135 = torch.ops.aten.view.default(mm_36, [16, 128, 384]);  mm_36 = None
        view_136 = torch.ops.aten.view.default(view_135, [16, -1, 6, 64]);  view_135 = None
        permute_64 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_65 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        view_137 = torch.ops.aten.view.default(mul_49, [2048, 512]);  mul_49 = None
        mm_37 = torch.ops.aten.mm.default(view_137, permute_65);  view_137 = permute_65 = None
        view_138 = torch.ops.aten.view.default(mm_37, [16, 128, 384]);  mm_37 = None
        view_139 = torch.ops.aten.view.default(view_138, [16, -1, 6, 64]);  view_138 = None
        permute_66 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        permute_67 = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        expand_20 = torch.ops.aten.expand.default(permute_62, [16, 6, 128, 64]);  permute_62 = None
        clone_41 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_140 = torch.ops.aten.view.default(clone_41, [96, 128, 64]);  clone_41 = None
        expand_21 = torch.ops.aten.expand.default(permute_67, [16, 6, 64, 128]);  permute_67 = None
        clone_42 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_141 = torch.ops.aten.view.default(clone_42, [96, 64, 128]);  clone_42 = None
        bmm_10 = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
        view_142 = torch.ops.aten.view.default(bmm_10, [16, 6, 128, 128]);  bmm_10 = None
        add_40 = torch.ops.aten.add.Tensor(view_142, add_4);  view_142 = None
        view_143 = torch.ops.aten.view.default(add_40, [96, 128, 128]);  add_40 = None
        view_144 = torch.ops.aten.view.default(view_143, [16, 6, 128, 128]);  view_143 = None
        amax_5 = torch.ops.aten.amax.default(view_144, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(view_144, amax_5);  view_144 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        expand_22 = torch.ops.aten.expand.default(div_7, [16, 6, 128, 128]);  div_7 = None
        view_145 = torch.ops.aten.view.default(expand_22, [96, 128, 128]);  expand_22 = None
        expand_23 = torch.ops.aten.expand.default(permute_66, [16, 6, 128, 64]);  permute_66 = None
        clone_44 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_146 = torch.ops.aten.view.default(clone_44, [96, 128, 64]);  clone_44 = None
        bmm_11 = torch.ops.aten.bmm.default(view_145, view_146);  view_145 = view_146 = None
        view_147 = torch.ops.aten.view.default(bmm_11, [16, 6, 128, 64]);  bmm_11 = None
        permute_68 = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
        clone_45 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_148 = torch.ops.aten.view.default(clone_45, [16, -1, 384]);  clone_45 = None
        permute_69 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        view_149 = torch.ops.aten.view.default(view_148, [2048, 384]);  view_148 = None
        mm_38 = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
        view_150 = torch.ops.aten.view.default(mm_38, [16, 128, 512]);  mm_38 = None
        add_41 = torch.ops.aten.add.Tensor(add_38, view_150);  add_38 = view_150 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(add_41, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_42 = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_50 = torch.ops.aten.mul.Tensor(add_41, rsqrt_11);  rsqrt_11 = None
        mul_51 = torch.ops.aten.mul.Tensor(arg56_1, mul_50);  arg56_1 = mul_50 = None
        permute_70 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        view_151 = torch.ops.aten.view.default(mul_51, [2048, 512])
        mm_39 = torch.ops.aten.mm.default(view_151, permute_70);  view_151 = permute_70 = None
        view_152 = torch.ops.aten.view.default(mm_39, [16, 128, 1024]);  mm_39 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_152, 0.5)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(view_152, 3.0)
        mul_53 = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_43 = torch.ops.aten.add.Tensor(view_152, mul_53);  view_152 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_44 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_52, add_44);  mul_52 = add_44 = None
        permute_71 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        view_153 = torch.ops.aten.view.default(mul_51, [2048, 512]);  mul_51 = None
        mm_40 = torch.ops.aten.mm.default(view_153, permute_71);  view_153 = permute_71 = None
        view_154 = torch.ops.aten.view.default(mm_40, [16, 128, 1024]);  mm_40 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, view_154);  mul_55 = view_154 = None
        permute_72 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        view_155 = torch.ops.aten.view.default(mul_56, [2048, 1024]);  mul_56 = None
        mm_41 = torch.ops.aten.mm.default(view_155, permute_72);  view_155 = permute_72 = None
        view_156 = torch.ops.aten.view.default(mm_41, [16, 128, 512]);  mm_41 = None
        add_45 = torch.ops.aten.add.Tensor(add_41, view_156);  add_41 = view_156 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(add_45, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_46 = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_57 = torch.ops.aten.mul.Tensor(add_45, rsqrt_12);  rsqrt_12 = None
        mul_58 = torch.ops.aten.mul.Tensor(arg61_1, mul_57);  arg61_1 = mul_57 = None
        permute_73 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        view_157 = torch.ops.aten.view.default(mul_58, [2048, 512])
        mm_42 = torch.ops.aten.mm.default(view_157, permute_73);  view_157 = permute_73 = None
        view_158 = torch.ops.aten.view.default(mm_42, [16, 128, 384]);  mm_42 = None
        view_159 = torch.ops.aten.view.default(view_158, [16, -1, 6, 64]);  view_158 = None
        permute_74 = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        permute_75 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        view_160 = torch.ops.aten.view.default(mul_58, [2048, 512])
        mm_43 = torch.ops.aten.mm.default(view_160, permute_75);  view_160 = permute_75 = None
        view_161 = torch.ops.aten.view.default(mm_43, [16, 128, 384]);  mm_43 = None
        view_162 = torch.ops.aten.view.default(view_161, [16, -1, 6, 64]);  view_161 = None
        permute_76 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_77 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        view_163 = torch.ops.aten.view.default(mul_58, [2048, 512]);  mul_58 = None
        mm_44 = torch.ops.aten.mm.default(view_163, permute_77);  view_163 = permute_77 = None
        view_164 = torch.ops.aten.view.default(mm_44, [16, 128, 384]);  mm_44 = None
        view_165 = torch.ops.aten.view.default(view_164, [16, -1, 6, 64]);  view_164 = None
        permute_78 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        permute_79 = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_24 = torch.ops.aten.expand.default(permute_74, [16, 6, 128, 64]);  permute_74 = None
        clone_49 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_166 = torch.ops.aten.view.default(clone_49, [96, 128, 64]);  clone_49 = None
        expand_25 = torch.ops.aten.expand.default(permute_79, [16, 6, 64, 128]);  permute_79 = None
        clone_50 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_167 = torch.ops.aten.view.default(clone_50, [96, 64, 128]);  clone_50 = None
        bmm_12 = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168 = torch.ops.aten.view.default(bmm_12, [16, 6, 128, 128]);  bmm_12 = None
        add_47 = torch.ops.aten.add.Tensor(view_168, add_4);  view_168 = None
        view_169 = torch.ops.aten.view.default(add_47, [96, 128, 128]);  add_47 = None
        view_170 = torch.ops.aten.view.default(view_169, [16, 6, 128, 128]);  view_169 = None
        amax_6 = torch.ops.aten.amax.default(view_170, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(view_170, amax_6);  view_170 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_26 = torch.ops.aten.expand.default(div_8, [16, 6, 128, 128]);  div_8 = None
        view_171 = torch.ops.aten.view.default(expand_26, [96, 128, 128]);  expand_26 = None
        expand_27 = torch.ops.aten.expand.default(permute_78, [16, 6, 128, 64]);  permute_78 = None
        clone_52 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_172 = torch.ops.aten.view.default(clone_52, [96, 128, 64]);  clone_52 = None
        bmm_13 = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
        view_173 = torch.ops.aten.view.default(bmm_13, [16, 6, 128, 64]);  bmm_13 = None
        permute_80 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        clone_53 = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_174 = torch.ops.aten.view.default(clone_53, [16, -1, 384]);  clone_53 = None
        permute_81 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        view_175 = torch.ops.aten.view.default(view_174, [2048, 384]);  view_174 = None
        mm_45 = torch.ops.aten.mm.default(view_175, permute_81);  view_175 = permute_81 = None
        view_176 = torch.ops.aten.view.default(mm_45, [16, 128, 512]);  mm_45 = None
        add_48 = torch.ops.aten.add.Tensor(add_45, view_176);  add_45 = view_176 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_49 = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_59 = torch.ops.aten.mul.Tensor(add_48, rsqrt_13);  rsqrt_13 = None
        mul_60 = torch.ops.aten.mul.Tensor(arg65_1, mul_59);  arg65_1 = mul_59 = None
        permute_82 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        view_177 = torch.ops.aten.view.default(mul_60, [2048, 512])
        mm_46 = torch.ops.aten.mm.default(view_177, permute_82);  view_177 = permute_82 = None
        view_178 = torch.ops.aten.view.default(mm_46, [16, 128, 1024]);  mm_46 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_178, 0.5)
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_62 = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_50 = torch.ops.aten.add.Tensor(view_178, mul_62);  view_178 = mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(add_50, 0.7978845608028654);  add_50 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
        add_51 = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_61, add_51);  mul_61 = add_51 = None
        permute_83 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        view_179 = torch.ops.aten.view.default(mul_60, [2048, 512]);  mul_60 = None
        mm_47 = torch.ops.aten.mm.default(view_179, permute_83);  view_179 = permute_83 = None
        view_180 = torch.ops.aten.view.default(mm_47, [16, 128, 1024]);  mm_47 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, view_180);  mul_64 = view_180 = None
        permute_84 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        view_181 = torch.ops.aten.view.default(mul_65, [2048, 1024]);  mul_65 = None
        mm_48 = torch.ops.aten.mm.default(view_181, permute_84);  view_181 = permute_84 = None
        view_182 = torch.ops.aten.view.default(mm_48, [16, 128, 512]);  mm_48 = None
        add_52 = torch.ops.aten.add.Tensor(add_48, view_182);  add_48 = view_182 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_53 = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_66 = torch.ops.aten.mul.Tensor(add_52, rsqrt_14);  rsqrt_14 = None
        mul_67 = torch.ops.aten.mul.Tensor(arg70_1, mul_66);  arg70_1 = mul_66 = None
        permute_85 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        view_183 = torch.ops.aten.view.default(mul_67, [2048, 512])
        mm_49 = torch.ops.aten.mm.default(view_183, permute_85);  view_183 = permute_85 = None
        view_184 = torch.ops.aten.view.default(mm_49, [16, 128, 384]);  mm_49 = None
        view_185 = torch.ops.aten.view.default(view_184, [16, -1, 6, 64]);  view_184 = None
        permute_86 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        permute_87 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        view_186 = torch.ops.aten.view.default(mul_67, [2048, 512])
        mm_50 = torch.ops.aten.mm.default(view_186, permute_87);  view_186 = permute_87 = None
        view_187 = torch.ops.aten.view.default(mm_50, [16, 128, 384]);  mm_50 = None
        view_188 = torch.ops.aten.view.default(view_187, [16, -1, 6, 64]);  view_187 = None
        permute_88 = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        permute_89 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        view_189 = torch.ops.aten.view.default(mul_67, [2048, 512]);  mul_67 = None
        mm_51 = torch.ops.aten.mm.default(view_189, permute_89);  view_189 = permute_89 = None
        view_190 = torch.ops.aten.view.default(mm_51, [16, 128, 384]);  mm_51 = None
        view_191 = torch.ops.aten.view.default(view_190, [16, -1, 6, 64]);  view_190 = None
        permute_90 = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        permute_91 = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        expand_28 = torch.ops.aten.expand.default(permute_86, [16, 6, 128, 64]);  permute_86 = None
        clone_57 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_192 = torch.ops.aten.view.default(clone_57, [96, 128, 64]);  clone_57 = None
        expand_29 = torch.ops.aten.expand.default(permute_91, [16, 6, 64, 128]);  permute_91 = None
        clone_58 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_193 = torch.ops.aten.view.default(clone_58, [96, 64, 128]);  clone_58 = None
        bmm_14 = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = view_193 = None
        view_194 = torch.ops.aten.view.default(bmm_14, [16, 6, 128, 128]);  bmm_14 = None
        add_54 = torch.ops.aten.add.Tensor(view_194, add_4);  view_194 = add_4 = None
        view_195 = torch.ops.aten.view.default(add_54, [96, 128, 128]);  add_54 = None
        view_196 = torch.ops.aten.view.default(view_195, [16, 6, 128, 128]);  view_195 = None
        amax_7 = torch.ops.aten.amax.default(view_196, [-1], True)
        sub_9 = torch.ops.aten.sub.Tensor(view_196, amax_7);  view_196 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_30 = torch.ops.aten.expand.default(div_9, [16, 6, 128, 128]);  div_9 = None
        view_197 = torch.ops.aten.view.default(expand_30, [96, 128, 128]);  expand_30 = None
        expand_31 = torch.ops.aten.expand.default(permute_90, [16, 6, 128, 64]);  permute_90 = None
        clone_60 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_198 = torch.ops.aten.view.default(clone_60, [96, 128, 64]);  clone_60 = None
        bmm_15 = torch.ops.aten.bmm.default(view_197, view_198);  view_197 = view_198 = None
        view_199 = torch.ops.aten.view.default(bmm_15, [16, 6, 128, 64]);  bmm_15 = None
        permute_92 = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        clone_61 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_200 = torch.ops.aten.view.default(clone_61, [16, -1, 384]);  clone_61 = None
        permute_93 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        view_201 = torch.ops.aten.view.default(view_200, [2048, 384]);  view_200 = None
        mm_52 = torch.ops.aten.mm.default(view_201, permute_93);  view_201 = permute_93 = None
        view_202 = torch.ops.aten.view.default(mm_52, [16, 128, 512]);  mm_52 = None
        add_55 = torch.ops.aten.add.Tensor(add_52, view_202);  add_52 = view_202 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(add_55, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_56 = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_68 = torch.ops.aten.mul.Tensor(add_55, rsqrt_15);  rsqrt_15 = None
        mul_69 = torch.ops.aten.mul.Tensor(arg74_1, mul_68);  arg74_1 = mul_68 = None
        permute_94 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        view_203 = torch.ops.aten.view.default(mul_69, [2048, 512])
        mm_53 = torch.ops.aten.mm.default(view_203, permute_94);  view_203 = permute_94 = None
        view_204 = torch.ops.aten.view.default(mm_53, [16, 128, 1024]);  mm_53 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_204, 0.5)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(view_204, 3.0)
        mul_71 = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_57 = torch.ops.aten.add.Tensor(view_204, mul_71);  view_204 = mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_72);  mul_72 = None
        add_58 = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_70, add_58);  mul_70 = add_58 = None
        permute_95 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        view_205 = torch.ops.aten.view.default(mul_69, [2048, 512]);  mul_69 = None
        mm_54 = torch.ops.aten.mm.default(view_205, permute_95);  view_205 = permute_95 = None
        view_206 = torch.ops.aten.view.default(mm_54, [16, 128, 1024]);  mm_54 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, view_206);  mul_73 = view_206 = None
        permute_96 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        view_207 = torch.ops.aten.view.default(mul_74, [2048, 1024]);  mul_74 = None
        mm_55 = torch.ops.aten.mm.default(view_207, permute_96);  view_207 = permute_96 = None
        view_208 = torch.ops.aten.view.default(mm_55, [16, 128, 512]);  mm_55 = None
        add_59 = torch.ops.aten.add.Tensor(add_55, view_208);  add_55 = view_208 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(add_59, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_60 = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_75 = torch.ops.aten.mul.Tensor(add_59, rsqrt_16);  add_59 = rsqrt_16 = None
        mul_76 = torch.ops.aten.mul.Tensor(arg75_1, mul_75);  arg75_1 = mul_75 = None
        view_209 = torch.ops.aten.view.default(arg0_1, [-1, 128]);  arg0_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg1_1, view_209);  arg1_1 = view_209 = None
        full_2 = torch.ops.aten.full.default([16, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_2 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(iota_2, 0)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        repeat = torch.ops.aten.repeat.default(unsqueeze_6, [16, 128, 1]);  unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        le = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(full_2, 1);  full_2 = unsqueeze_10 = None
        full_default_2 = torch.ops.aten.full.default([16, 1, 1, 128], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None
        sub_10 = torch.ops.aten.sub.Tensor(1.0, unsqueeze_9);  unsqueeze_9 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_10, -3.4028234663852886e+38);  sub_10 = None
        full_3 = torch.ops.aten.full.default([16, 128], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(full_3, 1);  full_3 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(unsqueeze_13, torch.float32);  unsqueeze_13 = None
        sub_11 = torch.ops.aten.sub.Tensor(1.0, convert_element_type_4);  convert_element_type_4 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_11, -3.4028234663852886e+38);  sub_11 = mul_79 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(embedding_2, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_61 = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_80 = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_17);  rsqrt_17 = None
        mul_81 = torch.ops.aten.mul.Tensor(arg82_1, mul_80);  arg82_1 = mul_80 = None
        permute_97 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        view_210 = torch.ops.aten.view.default(mul_81, [2048, 512])
        mm_56 = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
        view_211 = torch.ops.aten.view.default(mm_56, [16, 128, 384]);  mm_56 = None
        view_212 = torch.ops.aten.view.default(view_211, [16, -1, 6, 64]);  view_211 = None
        permute_98 = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        permute_99 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        view_213 = torch.ops.aten.view.default(mul_81, [2048, 512])
        mm_57 = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
        view_214 = torch.ops.aten.view.default(mm_57, [16, 128, 384]);  mm_57 = None
        view_215 = torch.ops.aten.view.default(view_214, [16, -1, 6, 64]);  view_214 = None
        permute_100 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        permute_101 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        view_216 = torch.ops.aten.view.default(mul_81, [2048, 512]);  mul_81 = None
        mm_58 = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
        view_217 = torch.ops.aten.view.default(mm_58, [16, 128, 384]);  mm_58 = None
        view_218 = torch.ops.aten.view.default(view_217, [16, -1, 6, 64]);  view_217 = None
        permute_102 = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        permute_103 = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
        expand_32 = torch.ops.aten.expand.default(permute_98, [16, 6, 128, 64]);  permute_98 = None
        clone_67 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_219 = torch.ops.aten.view.default(clone_67, [96, 128, 64]);  clone_67 = None
        expand_33 = torch.ops.aten.expand.default(permute_103, [16, 6, 64, 128]);  permute_103 = None
        clone_68 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_220 = torch.ops.aten.view.default(clone_68, [96, 64, 128]);  clone_68 = None
        bmm_16 = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
        view_221 = torch.ops.aten.view.default(bmm_16, [16, 6, 128, 128]);  bmm_16 = None
        iota_3 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(iota_3, 1);  iota_3 = None
        iota_4 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        sub_12 = torch.ops.aten.sub.Tensor(unsqueeze_15, unsqueeze_14);  unsqueeze_15 = unsqueeze_14 = None
        full_default_3 = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1 = torch.ops.aten.minimum.default(sub_12, full_default_3);  sub_12 = full_default_3 = None
        neg = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        lt_1 = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_10 = torch.ops.aten.div.Tensor(convert_element_type_5, 16);  convert_element_type_5 = None
        log_1 = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11 = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_82 = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_82, torch.int64);  mul_82 = None
        add_62 = torch.ops.aten.add.Tensor(convert_element_type_6, 16);  convert_element_type_6 = None
        full_default_4 = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_2 = torch.ops.aten.minimum.default(add_62, full_default_4);  add_62 = full_default_4 = None
        where_1 = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_63 = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
        embedding_3 = torch.ops.aten.embedding.default(arg81_1, add_63);  arg81_1 = add_63 = None
        permute_104 = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None
        add_64 = torch.ops.aten.add.Tensor(unsqueeze_16, mul_78);  unsqueeze_16 = mul_78 = None
        add_65 = torch.ops.aten.add.Tensor(view_221, add_64);  view_221 = None
        view_222 = torch.ops.aten.view.default(add_65, [96, 128, 128]);  add_65 = None
        view_223 = torch.ops.aten.view.default(view_222, [16, 6, 128, 128]);  view_222 = None
        amax_8 = torch.ops.aten.amax.default(view_223, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(view_223, amax_8);  view_223 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_34 = torch.ops.aten.expand.default(div_12, [16, 6, 128, 128]);  div_12 = None
        view_224 = torch.ops.aten.view.default(expand_34, [96, 128, 128]);  expand_34 = None
        expand_35 = torch.ops.aten.expand.default(permute_102, [16, 6, 128, 64])
        clone_70 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_225 = torch.ops.aten.view.default(clone_70, [96, 128, 64]);  clone_70 = None
        bmm_17 = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
        view_226 = torch.ops.aten.view.default(bmm_17, [16, 6, 128, 64]);  bmm_17 = None
        permute_105 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_71 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_227 = torch.ops.aten.view.default(clone_71, [16, -1, 384]);  clone_71 = None
        permute_106 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        view_228 = torch.ops.aten.view.default(view_227, [2048, 384]);  view_227 = None
        mm_59 = torch.ops.aten.mm.default(view_228, permute_106);  view_228 = permute_106 = None
        view_229 = torch.ops.aten.view.default(mm_59, [16, 128, 512]);  mm_59 = None
        add_66 = torch.ops.aten.add.Tensor(embedding_2, view_229);  embedding_2 = view_229 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(add_66, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_67 = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_83 = torch.ops.aten.mul.Tensor(add_66, rsqrt_18);  rsqrt_18 = None
        mul_84 = torch.ops.aten.mul.Tensor(arg87_1, mul_83);  arg87_1 = mul_83 = None
        permute_107 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        view_230 = torch.ops.aten.view.default(mul_84, [2048, 512]);  mul_84 = None
        mm_60 = torch.ops.aten.mm.default(view_230, permute_107);  view_230 = permute_107 = None
        view_231 = torch.ops.aten.view.default(mm_60, [16, 128, 384]);  mm_60 = None
        view_232 = torch.ops.aten.view.default(view_231, [16, -1, 6, 64]);  view_231 = None
        permute_108 = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        permute_109 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        view_233 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_61 = torch.ops.aten.mm.default(view_233, permute_109);  view_233 = permute_109 = None
        view_234 = torch.ops.aten.view.default(mm_61, [16, 128, 384]);  mm_61 = None
        view_235 = torch.ops.aten.view.default(view_234, [16, -1, 6, 64]);  view_234 = None
        permute_110 = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
        permute_111 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        view_236 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_62 = torch.ops.aten.mm.default(view_236, permute_111);  view_236 = permute_111 = None
        view_237 = torch.ops.aten.view.default(mm_62, [16, 128, 384]);  mm_62 = None
        view_238 = torch.ops.aten.view.default(view_237, [16, -1, 6, 64]);  view_237 = None
        permute_112 = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        permute_113 = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2])
        expand_36 = torch.ops.aten.expand.default(permute_108, [16, 6, 128, 64]);  permute_108 = None
        clone_73 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_239 = torch.ops.aten.view.default(clone_73, [96, 128, 64]);  clone_73 = None
        expand_37 = torch.ops.aten.expand.default(permute_113, [16, 6, 64, 128]);  permute_113 = None
        clone_74 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_240 = torch.ops.aten.view.default(clone_74, [96, 64, 128]);  clone_74 = None
        bmm_18 = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None
        view_241 = torch.ops.aten.view.default(bmm_18, [16, 6, 128, 128]);  bmm_18 = None
        full_6 = torch.ops.aten.full.default([1, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_6 = None
        full_default_5 = torch.ops.aten.full.default([16, 6, 128, 128], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_5 = None
        view_242 = torch.ops.aten.view.default(view_241, [96, 128, 128]);  view_241 = None
        view_243 = torch.ops.aten.view.default(view_242, [16, 6, 128, 128]);  view_242 = None
        amax_9 = torch.ops.aten.amax.default(view_243, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(view_243, amax_9);  view_243 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_38 = torch.ops.aten.expand.default(div_13, [16, 6, 128, 128]);  div_13 = None
        view_244 = torch.ops.aten.view.default(expand_38, [96, 128, 128]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(permute_112, [16, 6, 128, 64])
        clone_76 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_245 = torch.ops.aten.view.default(clone_76, [96, 128, 64]);  clone_76 = None
        bmm_19 = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
        view_246 = torch.ops.aten.view.default(bmm_19, [16, 6, 128, 64]);  bmm_19 = None
        permute_114 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_77 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_247 = torch.ops.aten.view.default(clone_77, [16, -1, 384]);  clone_77 = None
        permute_115 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        view_248 = torch.ops.aten.view.default(view_247, [2048, 384]);  view_247 = None
        mm_63 = torch.ops.aten.mm.default(view_248, permute_115);  view_248 = permute_115 = None
        view_249 = torch.ops.aten.view.default(mm_63, [16, 128, 512]);  mm_63 = None
        add_70 = torch.ops.aten.add.Tensor(add_66, view_249);  add_66 = view_249 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_71 = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_85 = torch.ops.aten.mul.Tensor(add_70, rsqrt_19);  rsqrt_19 = None
        mul_86 = torch.ops.aten.mul.Tensor(arg91_1, mul_85);  arg91_1 = mul_85 = None
        permute_116 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        view_250 = torch.ops.aten.view.default(mul_86, [2048, 512])
        mm_64 = torch.ops.aten.mm.default(view_250, permute_116);  view_250 = permute_116 = None
        view_251 = torch.ops.aten.view.default(mm_64, [16, 128, 1024]);  mm_64 = None
        mul_87 = torch.ops.aten.mul.Tensor(view_251, 0.5)
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
        mul_88 = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_72 = torch.ops.aten.add.Tensor(view_251, mul_88);  view_251 = mul_88 = None
        mul_89 = torch.ops.aten.mul.Tensor(add_72, 0.7978845608028654);  add_72 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_89);  mul_89 = None
        add_73 = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_87, add_73);  mul_87 = add_73 = None
        permute_117 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        view_252 = torch.ops.aten.view.default(mul_86, [2048, 512]);  mul_86 = None
        mm_65 = torch.ops.aten.mm.default(view_252, permute_117);  view_252 = permute_117 = None
        view_253 = torch.ops.aten.view.default(mm_65, [16, 128, 1024]);  mm_65 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, view_253);  mul_90 = view_253 = None
        permute_118 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        view_254 = torch.ops.aten.view.default(mul_91, [2048, 1024]);  mul_91 = None
        mm_66 = torch.ops.aten.mm.default(view_254, permute_118);  view_254 = permute_118 = None
        view_255 = torch.ops.aten.view.default(mm_66, [16, 128, 512]);  mm_66 = None
        add_74 = torch.ops.aten.add.Tensor(add_70, view_255);  add_70 = view_255 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_75 = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_92 = torch.ops.aten.mul.Tensor(add_74, rsqrt_20);  rsqrt_20 = None
        mul_93 = torch.ops.aten.mul.Tensor(arg96_1, mul_92);  arg96_1 = mul_92 = None
        permute_119 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        view_256 = torch.ops.aten.view.default(mul_93, [2048, 512])
        mm_67 = torch.ops.aten.mm.default(view_256, permute_119);  view_256 = permute_119 = None
        view_257 = torch.ops.aten.view.default(mm_67, [16, 128, 384]);  mm_67 = None
        view_258 = torch.ops.aten.view.default(view_257, [16, -1, 6, 64]);  view_257 = None
        permute_120 = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        permute_121 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        view_259 = torch.ops.aten.view.default(mul_93, [2048, 512])
        mm_68 = torch.ops.aten.mm.default(view_259, permute_121);  view_259 = permute_121 = None
        view_260 = torch.ops.aten.view.default(mm_68, [16, 128, 384]);  mm_68 = None
        view_261 = torch.ops.aten.view.default(view_260, [16, -1, 6, 64]);  view_260 = None
        permute_122 = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        permute_123 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        view_262 = torch.ops.aten.view.default(mul_93, [2048, 512]);  mul_93 = None
        mm_69 = torch.ops.aten.mm.default(view_262, permute_123);  view_262 = permute_123 = None
        view_263 = torch.ops.aten.view.default(mm_69, [16, 128, 384]);  mm_69 = None
        view_264 = torch.ops.aten.view.default(view_263, [16, -1, 6, 64]);  view_263 = None
        permute_124 = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        permute_125 = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2])
        expand_40 = torch.ops.aten.expand.default(permute_120, [16, 6, 128, 64]);  permute_120 = None
        clone_81 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_265 = torch.ops.aten.view.default(clone_81, [96, 128, 64]);  clone_81 = None
        expand_41 = torch.ops.aten.expand.default(permute_125, [16, 6, 64, 128]);  permute_125 = None
        clone_82 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_266 = torch.ops.aten.view.default(clone_82, [96, 64, 128]);  clone_82 = None
        bmm_20 = torch.ops.aten.bmm.default(view_265, view_266);  view_265 = view_266 = None
        view_267 = torch.ops.aten.view.default(bmm_20, [16, 6, 128, 128]);  bmm_20 = None
        add_76 = torch.ops.aten.add.Tensor(view_267, add_64);  view_267 = None
        view_268 = torch.ops.aten.view.default(add_76, [96, 128, 128]);  add_76 = None
        view_269 = torch.ops.aten.view.default(view_268, [16, 6, 128, 128]);  view_268 = None
        amax_10 = torch.ops.aten.amax.default(view_269, [-1], True)
        sub_15 = torch.ops.aten.sub.Tensor(view_269, amax_10);  view_269 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_42 = torch.ops.aten.expand.default(div_14, [16, 6, 128, 128]);  div_14 = None
        view_270 = torch.ops.aten.view.default(expand_42, [96, 128, 128]);  expand_42 = None
        expand_43 = torch.ops.aten.expand.default(permute_124, [16, 6, 128, 64])
        clone_84 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_271 = torch.ops.aten.view.default(clone_84, [96, 128, 64]);  clone_84 = None
        bmm_21 = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
        view_272 = torch.ops.aten.view.default(bmm_21, [16, 6, 128, 64]);  bmm_21 = None
        permute_126 = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_85 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_273 = torch.ops.aten.view.default(clone_85, [16, -1, 384]);  clone_85 = None
        permute_127 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        view_274 = torch.ops.aten.view.default(view_273, [2048, 384]);  view_273 = None
        mm_70 = torch.ops.aten.mm.default(view_274, permute_127);  view_274 = permute_127 = None
        view_275 = torch.ops.aten.view.default(mm_70, [16, 128, 512]);  mm_70 = None
        add_77 = torch.ops.aten.add.Tensor(add_74, view_275);  add_74 = view_275 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(add_77, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_78 = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_77, rsqrt_21);  rsqrt_21 = None
        mul_95 = torch.ops.aten.mul.Tensor(arg101_1, mul_94);  arg101_1 = mul_94 = None
        permute_128 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        view_276 = torch.ops.aten.view.default(mul_95, [2048, 512]);  mul_95 = None
        mm_71 = torch.ops.aten.mm.default(view_276, permute_128);  view_276 = permute_128 = None
        view_277 = torch.ops.aten.view.default(mm_71, [16, 128, 384]);  mm_71 = None
        view_278 = torch.ops.aten.view.default(view_277, [16, -1, 6, 64]);  view_277 = None
        permute_129 = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        permute_130 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        view_279 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_72 = torch.ops.aten.mm.default(view_279, permute_130);  view_279 = permute_130 = None
        view_280 = torch.ops.aten.view.default(mm_72, [16, 128, 384]);  mm_72 = None
        view_281 = torch.ops.aten.view.default(view_280, [16, -1, 6, 64]);  view_280 = None
        permute_131 = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
        permute_132 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        view_282 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_73 = torch.ops.aten.mm.default(view_282, permute_132);  view_282 = permute_132 = None
        view_283 = torch.ops.aten.view.default(mm_73, [16, 128, 384]);  mm_73 = None
        view_284 = torch.ops.aten.view.default(view_283, [16, -1, 6, 64]);  view_283 = None
        permute_133 = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        permute_134 = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
        expand_44 = torch.ops.aten.expand.default(permute_129, [16, 6, 128, 64]);  permute_129 = None
        clone_87 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_285 = torch.ops.aten.view.default(clone_87, [96, 128, 64]);  clone_87 = None
        expand_45 = torch.ops.aten.expand.default(permute_134, [16, 6, 64, 128]);  permute_134 = None
        clone_88 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_286 = torch.ops.aten.view.default(clone_88, [96, 64, 128]);  clone_88 = None
        bmm_22 = torch.ops.aten.bmm.default(view_285, view_286);  view_285 = view_286 = None
        view_287 = torch.ops.aten.view.default(bmm_22, [16, 6, 128, 128]);  bmm_22 = None
        view_288 = torch.ops.aten.view.default(view_287, [96, 128, 128]);  view_287 = None
        view_289 = torch.ops.aten.view.default(view_288, [16, 6, 128, 128]);  view_288 = None
        amax_11 = torch.ops.aten.amax.default(view_289, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(view_289, amax_11);  view_289 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_46 = torch.ops.aten.expand.default(div_15, [16, 6, 128, 128]);  div_15 = None
        view_290 = torch.ops.aten.view.default(expand_46, [96, 128, 128]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(permute_133, [16, 6, 128, 64])
        clone_90 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_291 = torch.ops.aten.view.default(clone_90, [96, 128, 64]);  clone_90 = None
        bmm_23 = torch.ops.aten.bmm.default(view_290, view_291);  view_290 = view_291 = None
        view_292 = torch.ops.aten.view.default(bmm_23, [16, 6, 128, 64]);  bmm_23 = None
        permute_135 = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_91 = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_293 = torch.ops.aten.view.default(clone_91, [16, -1, 384]);  clone_91 = None
        permute_136 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        view_294 = torch.ops.aten.view.default(view_293, [2048, 384]);  view_293 = None
        mm_74 = torch.ops.aten.mm.default(view_294, permute_136);  view_294 = permute_136 = None
        view_295 = torch.ops.aten.view.default(mm_74, [16, 128, 512]);  mm_74 = None
        add_80 = torch.ops.aten.add.Tensor(add_77, view_295);  add_77 = view_295 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(add_80, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_81 = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_96 = torch.ops.aten.mul.Tensor(add_80, rsqrt_22);  rsqrt_22 = None
        mul_97 = torch.ops.aten.mul.Tensor(arg105_1, mul_96);  arg105_1 = mul_96 = None
        permute_137 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        view_296 = torch.ops.aten.view.default(mul_97, [2048, 512])
        mm_75 = torch.ops.aten.mm.default(view_296, permute_137);  view_296 = permute_137 = None
        view_297 = torch.ops.aten.view.default(mm_75, [16, 128, 1024]);  mm_75 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_297, 0.5)
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(view_297, 3.0)
        mul_99 = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_82 = torch.ops.aten.add.Tensor(view_297, mul_99);  view_297 = mul_99 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
        add_83 = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_98, add_83);  mul_98 = add_83 = None
        permute_138 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        view_298 = torch.ops.aten.view.default(mul_97, [2048, 512]);  mul_97 = None
        mm_76 = torch.ops.aten.mm.default(view_298, permute_138);  view_298 = permute_138 = None
        view_299 = torch.ops.aten.view.default(mm_76, [16, 128, 1024]);  mm_76 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, view_299);  mul_101 = view_299 = None
        permute_139 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        view_300 = torch.ops.aten.view.default(mul_102, [2048, 1024]);  mul_102 = None
        mm_77 = torch.ops.aten.mm.default(view_300, permute_139);  view_300 = permute_139 = None
        view_301 = torch.ops.aten.view.default(mm_77, [16, 128, 512]);  mm_77 = None
        add_84 = torch.ops.aten.add.Tensor(add_80, view_301);  add_80 = view_301 = None
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        add_85 = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_103 = torch.ops.aten.mul.Tensor(add_84, rsqrt_23);  rsqrt_23 = None
        mul_104 = torch.ops.aten.mul.Tensor(arg110_1, mul_103);  arg110_1 = mul_103 = None
        permute_140 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        view_302 = torch.ops.aten.view.default(mul_104, [2048, 512])
        mm_78 = torch.ops.aten.mm.default(view_302, permute_140);  view_302 = permute_140 = None
        view_303 = torch.ops.aten.view.default(mm_78, [16, 128, 384]);  mm_78 = None
        view_304 = torch.ops.aten.view.default(view_303, [16, -1, 6, 64]);  view_303 = None
        permute_141 = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        permute_142 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        view_305 = torch.ops.aten.view.default(mul_104, [2048, 512])
        mm_79 = torch.ops.aten.mm.default(view_305, permute_142);  view_305 = permute_142 = None
        view_306 = torch.ops.aten.view.default(mm_79, [16, 128, 384]);  mm_79 = None
        view_307 = torch.ops.aten.view.default(view_306, [16, -1, 6, 64]);  view_306 = None
        permute_143 = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        permute_144 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        view_308 = torch.ops.aten.view.default(mul_104, [2048, 512]);  mul_104 = None
        mm_80 = torch.ops.aten.mm.default(view_308, permute_144);  view_308 = permute_144 = None
        view_309 = torch.ops.aten.view.default(mm_80, [16, 128, 384]);  mm_80 = None
        view_310 = torch.ops.aten.view.default(view_309, [16, -1, 6, 64]);  view_309 = None
        permute_145 = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        permute_146 = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2])
        expand_48 = torch.ops.aten.expand.default(permute_141, [16, 6, 128, 64]);  permute_141 = None
        clone_95 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_311 = torch.ops.aten.view.default(clone_95, [96, 128, 64]);  clone_95 = None
        expand_49 = torch.ops.aten.expand.default(permute_146, [16, 6, 64, 128]);  permute_146 = None
        clone_96 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_312 = torch.ops.aten.view.default(clone_96, [96, 64, 128]);  clone_96 = None
        bmm_24 = torch.ops.aten.bmm.default(view_311, view_312);  view_311 = view_312 = None
        view_313 = torch.ops.aten.view.default(bmm_24, [16, 6, 128, 128]);  bmm_24 = None
        add_86 = torch.ops.aten.add.Tensor(view_313, add_64);  view_313 = None
        view_314 = torch.ops.aten.view.default(add_86, [96, 128, 128]);  add_86 = None
        view_315 = torch.ops.aten.view.default(view_314, [16, 6, 128, 128]);  view_314 = None
        amax_12 = torch.ops.aten.amax.default(view_315, [-1], True)
        sub_17 = torch.ops.aten.sub.Tensor(view_315, amax_12);  view_315 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        expand_50 = torch.ops.aten.expand.default(div_16, [16, 6, 128, 128]);  div_16 = None
        view_316 = torch.ops.aten.view.default(expand_50, [96, 128, 128]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(permute_145, [16, 6, 128, 64])
        clone_98 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_317 = torch.ops.aten.view.default(clone_98, [96, 128, 64]);  clone_98 = None
        bmm_25 = torch.ops.aten.bmm.default(view_316, view_317);  view_316 = view_317 = None
        view_318 = torch.ops.aten.view.default(bmm_25, [16, 6, 128, 64]);  bmm_25 = None
        permute_147 = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_99 = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_319 = torch.ops.aten.view.default(clone_99, [16, -1, 384]);  clone_99 = None
        permute_148 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        view_320 = torch.ops.aten.view.default(view_319, [2048, 384]);  view_319 = None
        mm_81 = torch.ops.aten.mm.default(view_320, permute_148);  view_320 = permute_148 = None
        view_321 = torch.ops.aten.view.default(mm_81, [16, 128, 512]);  mm_81 = None
        add_87 = torch.ops.aten.add.Tensor(add_84, view_321);  add_84 = view_321 = None
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(add_87, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_88 = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_105 = torch.ops.aten.mul.Tensor(add_87, rsqrt_24);  rsqrt_24 = None
        mul_106 = torch.ops.aten.mul.Tensor(arg115_1, mul_105);  arg115_1 = mul_105 = None
        permute_149 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        view_322 = torch.ops.aten.view.default(mul_106, [2048, 512]);  mul_106 = None
        mm_82 = torch.ops.aten.mm.default(view_322, permute_149);  view_322 = permute_149 = None
        view_323 = torch.ops.aten.view.default(mm_82, [16, 128, 384]);  mm_82 = None
        view_324 = torch.ops.aten.view.default(view_323, [16, -1, 6, 64]);  view_323 = None
        permute_150 = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        permute_151 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        view_325 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_83 = torch.ops.aten.mm.default(view_325, permute_151);  view_325 = permute_151 = None
        view_326 = torch.ops.aten.view.default(mm_83, [16, 128, 384]);  mm_83 = None
        view_327 = torch.ops.aten.view.default(view_326, [16, -1, 6, 64]);  view_326 = None
        permute_152 = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        permute_153 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        view_328 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_84 = torch.ops.aten.mm.default(view_328, permute_153);  view_328 = permute_153 = None
        view_329 = torch.ops.aten.view.default(mm_84, [16, 128, 384]);  mm_84 = None
        view_330 = torch.ops.aten.view.default(view_329, [16, -1, 6, 64]);  view_329 = None
        permute_154 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        permute_155 = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2])
        expand_52 = torch.ops.aten.expand.default(permute_150, [16, 6, 128, 64]);  permute_150 = None
        clone_101 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_331 = torch.ops.aten.view.default(clone_101, [96, 128, 64]);  clone_101 = None
        expand_53 = torch.ops.aten.expand.default(permute_155, [16, 6, 64, 128]);  permute_155 = None
        clone_102 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_332 = torch.ops.aten.view.default(clone_102, [96, 64, 128]);  clone_102 = None
        bmm_26 = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
        view_333 = torch.ops.aten.view.default(bmm_26, [16, 6, 128, 128]);  bmm_26 = None
        view_334 = torch.ops.aten.view.default(view_333, [96, 128, 128]);  view_333 = None
        view_335 = torch.ops.aten.view.default(view_334, [16, 6, 128, 128]);  view_334 = None
        amax_13 = torch.ops.aten.amax.default(view_335, [-1], True)
        sub_18 = torch.ops.aten.sub.Tensor(view_335, amax_13);  view_335 = amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        expand_54 = torch.ops.aten.expand.default(div_17, [16, 6, 128, 128]);  div_17 = None
        view_336 = torch.ops.aten.view.default(expand_54, [96, 128, 128]);  expand_54 = None
        expand_55 = torch.ops.aten.expand.default(permute_154, [16, 6, 128, 64])
        clone_104 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_337 = torch.ops.aten.view.default(clone_104, [96, 128, 64]);  clone_104 = None
        bmm_27 = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
        view_338 = torch.ops.aten.view.default(bmm_27, [16, 6, 128, 64]);  bmm_27 = None
        permute_156 = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_105 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_339 = torch.ops.aten.view.default(clone_105, [16, -1, 384]);  clone_105 = None
        permute_157 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        view_340 = torch.ops.aten.view.default(view_339, [2048, 384]);  view_339 = None
        mm_85 = torch.ops.aten.mm.default(view_340, permute_157);  view_340 = permute_157 = None
        view_341 = torch.ops.aten.view.default(mm_85, [16, 128, 512]);  mm_85 = None
        add_90 = torch.ops.aten.add.Tensor(add_87, view_341);  add_87 = view_341 = None
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(add_90, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        add_91 = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_107 = torch.ops.aten.mul.Tensor(add_90, rsqrt_25);  rsqrt_25 = None
        mul_108 = torch.ops.aten.mul.Tensor(arg119_1, mul_107);  arg119_1 = mul_107 = None
        permute_158 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        view_342 = torch.ops.aten.view.default(mul_108, [2048, 512])
        mm_86 = torch.ops.aten.mm.default(view_342, permute_158);  view_342 = permute_158 = None
        view_343 = torch.ops.aten.view.default(mm_86, [16, 128, 1024]);  mm_86 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_343, 0.5)
        pow_37 = torch.ops.aten.pow.Tensor_Scalar(view_343, 3.0)
        mul_110 = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_92 = torch.ops.aten.add.Tensor(view_343, mul_110);  view_343 = mul_110 = None
        mul_111 = torch.ops.aten.mul.Tensor(add_92, 0.7978845608028654);  add_92 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_111);  mul_111 = None
        add_93 = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_109, add_93);  mul_109 = add_93 = None
        permute_159 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        view_344 = torch.ops.aten.view.default(mul_108, [2048, 512]);  mul_108 = None
        mm_87 = torch.ops.aten.mm.default(view_344, permute_159);  view_344 = permute_159 = None
        view_345 = torch.ops.aten.view.default(mm_87, [16, 128, 1024]);  mm_87 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, view_345);  mul_112 = view_345 = None
        permute_160 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        view_346 = torch.ops.aten.view.default(mul_113, [2048, 1024]);  mul_113 = None
        mm_88 = torch.ops.aten.mm.default(view_346, permute_160);  view_346 = permute_160 = None
        view_347 = torch.ops.aten.view.default(mm_88, [16, 128, 512]);  mm_88 = None
        add_94 = torch.ops.aten.add.Tensor(add_90, view_347);  add_90 = view_347 = None
        pow_38 = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        add_95 = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_114 = torch.ops.aten.mul.Tensor(add_94, rsqrt_26);  rsqrt_26 = None
        mul_115 = torch.ops.aten.mul.Tensor(arg124_1, mul_114);  arg124_1 = mul_114 = None
        permute_161 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        view_348 = torch.ops.aten.view.default(mul_115, [2048, 512])
        mm_89 = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
        view_349 = torch.ops.aten.view.default(mm_89, [16, 128, 384]);  mm_89 = None
        view_350 = torch.ops.aten.view.default(view_349, [16, -1, 6, 64]);  view_349 = None
        permute_162 = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        permute_163 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        view_351 = torch.ops.aten.view.default(mul_115, [2048, 512])
        mm_90 = torch.ops.aten.mm.default(view_351, permute_163);  view_351 = permute_163 = None
        view_352 = torch.ops.aten.view.default(mm_90, [16, 128, 384]);  mm_90 = None
        view_353 = torch.ops.aten.view.default(view_352, [16, -1, 6, 64]);  view_352 = None
        permute_164 = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        permute_165 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        view_354 = torch.ops.aten.view.default(mul_115, [2048, 512]);  mul_115 = None
        mm_91 = torch.ops.aten.mm.default(view_354, permute_165);  view_354 = permute_165 = None
        view_355 = torch.ops.aten.view.default(mm_91, [16, 128, 384]);  mm_91 = None
        view_356 = torch.ops.aten.view.default(view_355, [16, -1, 6, 64]);  view_355 = None
        permute_166 = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        permute_167 = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2])
        expand_56 = torch.ops.aten.expand.default(permute_162, [16, 6, 128, 64]);  permute_162 = None
        clone_109 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_357 = torch.ops.aten.view.default(clone_109, [96, 128, 64]);  clone_109 = None
        expand_57 = torch.ops.aten.expand.default(permute_167, [16, 6, 64, 128]);  permute_167 = None
        clone_110 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_358 = torch.ops.aten.view.default(clone_110, [96, 64, 128]);  clone_110 = None
        bmm_28 = torch.ops.aten.bmm.default(view_357, view_358);  view_357 = view_358 = None
        view_359 = torch.ops.aten.view.default(bmm_28, [16, 6, 128, 128]);  bmm_28 = None
        add_96 = torch.ops.aten.add.Tensor(view_359, add_64);  view_359 = None
        view_360 = torch.ops.aten.view.default(add_96, [96, 128, 128]);  add_96 = None
        view_361 = torch.ops.aten.view.default(view_360, [16, 6, 128, 128]);  view_360 = None
        amax_14 = torch.ops.aten.amax.default(view_361, [-1], True)
        sub_19 = torch.ops.aten.sub.Tensor(view_361, amax_14);  view_361 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        expand_58 = torch.ops.aten.expand.default(div_18, [16, 6, 128, 128]);  div_18 = None
        view_362 = torch.ops.aten.view.default(expand_58, [96, 128, 128]);  expand_58 = None
        expand_59 = torch.ops.aten.expand.default(permute_166, [16, 6, 128, 64])
        clone_112 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_363 = torch.ops.aten.view.default(clone_112, [96, 128, 64]);  clone_112 = None
        bmm_29 = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
        view_364 = torch.ops.aten.view.default(bmm_29, [16, 6, 128, 64]);  bmm_29 = None
        permute_168 = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        clone_113 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_365 = torch.ops.aten.view.default(clone_113, [16, -1, 384]);  clone_113 = None
        permute_169 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        view_366 = torch.ops.aten.view.default(view_365, [2048, 384]);  view_365 = None
        mm_92 = torch.ops.aten.mm.default(view_366, permute_169);  view_366 = permute_169 = None
        view_367 = torch.ops.aten.view.default(mm_92, [16, 128, 512]);  mm_92 = None
        add_97 = torch.ops.aten.add.Tensor(add_94, view_367);  add_94 = view_367 = None
        pow_39 = torch.ops.aten.pow.Tensor_Scalar(add_97, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_98 = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_116 = torch.ops.aten.mul.Tensor(add_97, rsqrt_27);  rsqrt_27 = None
        mul_117 = torch.ops.aten.mul.Tensor(arg129_1, mul_116);  arg129_1 = mul_116 = None
        permute_170 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        view_368 = torch.ops.aten.view.default(mul_117, [2048, 512]);  mul_117 = None
        mm_93 = torch.ops.aten.mm.default(view_368, permute_170);  view_368 = permute_170 = None
        view_369 = torch.ops.aten.view.default(mm_93, [16, 128, 384]);  mm_93 = None
        view_370 = torch.ops.aten.view.default(view_369, [16, -1, 6, 64]);  view_369 = None
        permute_171 = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        permute_172 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        view_371 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_94 = torch.ops.aten.mm.default(view_371, permute_172);  view_371 = permute_172 = None
        view_372 = torch.ops.aten.view.default(mm_94, [16, 128, 384]);  mm_94 = None
        view_373 = torch.ops.aten.view.default(view_372, [16, -1, 6, 64]);  view_372 = None
        permute_173 = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        permute_174 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        view_374 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_95 = torch.ops.aten.mm.default(view_374, permute_174);  view_374 = permute_174 = None
        view_375 = torch.ops.aten.view.default(mm_95, [16, 128, 384]);  mm_95 = None
        view_376 = torch.ops.aten.view.default(view_375, [16, -1, 6, 64]);  view_375 = None
        permute_175 = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        permute_176 = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2])
        expand_60 = torch.ops.aten.expand.default(permute_171, [16, 6, 128, 64]);  permute_171 = None
        clone_115 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_377 = torch.ops.aten.view.default(clone_115, [96, 128, 64]);  clone_115 = None
        expand_61 = torch.ops.aten.expand.default(permute_176, [16, 6, 64, 128]);  permute_176 = None
        clone_116 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_378 = torch.ops.aten.view.default(clone_116, [96, 64, 128]);  clone_116 = None
        bmm_30 = torch.ops.aten.bmm.default(view_377, view_378);  view_377 = view_378 = None
        view_379 = torch.ops.aten.view.default(bmm_30, [16, 6, 128, 128]);  bmm_30 = None
        view_380 = torch.ops.aten.view.default(view_379, [96, 128, 128]);  view_379 = None
        view_381 = torch.ops.aten.view.default(view_380, [16, 6, 128, 128]);  view_380 = None
        amax_15 = torch.ops.aten.amax.default(view_381, [-1], True)
        sub_20 = torch.ops.aten.sub.Tensor(view_381, amax_15);  view_381 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        expand_62 = torch.ops.aten.expand.default(div_19, [16, 6, 128, 128]);  div_19 = None
        view_382 = torch.ops.aten.view.default(expand_62, [96, 128, 128]);  expand_62 = None
        expand_63 = torch.ops.aten.expand.default(permute_175, [16, 6, 128, 64])
        clone_118 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_383 = torch.ops.aten.view.default(clone_118, [96, 128, 64]);  clone_118 = None
        bmm_31 = torch.ops.aten.bmm.default(view_382, view_383);  view_382 = view_383 = None
        view_384 = torch.ops.aten.view.default(bmm_31, [16, 6, 128, 64]);  bmm_31 = None
        permute_177 = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_119 = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_385 = torch.ops.aten.view.default(clone_119, [16, -1, 384]);  clone_119 = None
        permute_178 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        view_386 = torch.ops.aten.view.default(view_385, [2048, 384]);  view_385 = None
        mm_96 = torch.ops.aten.mm.default(view_386, permute_178);  view_386 = permute_178 = None
        view_387 = torch.ops.aten.view.default(mm_96, [16, 128, 512]);  mm_96 = None
        add_100 = torch.ops.aten.add.Tensor(add_97, view_387);  add_97 = view_387 = None
        pow_40 = torch.ops.aten.pow.Tensor_Scalar(add_100, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        add_101 = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_118 = torch.ops.aten.mul.Tensor(add_100, rsqrt_28);  rsqrt_28 = None
        mul_119 = torch.ops.aten.mul.Tensor(arg133_1, mul_118);  arg133_1 = mul_118 = None
        permute_179 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        view_388 = torch.ops.aten.view.default(mul_119, [2048, 512])
        mm_97 = torch.ops.aten.mm.default(view_388, permute_179);  view_388 = permute_179 = None
        view_389 = torch.ops.aten.view.default(mm_97, [16, 128, 1024]);  mm_97 = None
        mul_120 = torch.ops.aten.mul.Tensor(view_389, 0.5)
        pow_41 = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
        mul_121 = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_102 = torch.ops.aten.add.Tensor(view_389, mul_121);  view_389 = mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_122);  mul_122 = None
        add_103 = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_120, add_103);  mul_120 = add_103 = None
        permute_180 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        view_390 = torch.ops.aten.view.default(mul_119, [2048, 512]);  mul_119 = None
        mm_98 = torch.ops.aten.mm.default(view_390, permute_180);  view_390 = permute_180 = None
        view_391 = torch.ops.aten.view.default(mm_98, [16, 128, 1024]);  mm_98 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, view_391);  mul_123 = view_391 = None
        permute_181 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        view_392 = torch.ops.aten.view.default(mul_124, [2048, 1024]);  mul_124 = None
        mm_99 = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
        view_393 = torch.ops.aten.view.default(mm_99, [16, 128, 512]);  mm_99 = None
        add_104 = torch.ops.aten.add.Tensor(add_100, view_393);  add_100 = view_393 = None
        pow_42 = torch.ops.aten.pow.Tensor_Scalar(add_104, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        add_105 = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_125 = torch.ops.aten.mul.Tensor(add_104, rsqrt_29);  rsqrt_29 = None
        mul_126 = torch.ops.aten.mul.Tensor(arg138_1, mul_125);  arg138_1 = mul_125 = None
        permute_182 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        view_394 = torch.ops.aten.view.default(mul_126, [2048, 512])
        mm_100 = torch.ops.aten.mm.default(view_394, permute_182);  view_394 = permute_182 = None
        view_395 = torch.ops.aten.view.default(mm_100, [16, 128, 384]);  mm_100 = None
        view_396 = torch.ops.aten.view.default(view_395, [16, -1, 6, 64]);  view_395 = None
        permute_183 = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
        permute_184 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        view_397 = torch.ops.aten.view.default(mul_126, [2048, 512])
        mm_101 = torch.ops.aten.mm.default(view_397, permute_184);  view_397 = permute_184 = None
        view_398 = torch.ops.aten.view.default(mm_101, [16, 128, 384]);  mm_101 = None
        view_399 = torch.ops.aten.view.default(view_398, [16, -1, 6, 64]);  view_398 = None
        permute_185 = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        permute_186 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        view_400 = torch.ops.aten.view.default(mul_126, [2048, 512]);  mul_126 = None
        mm_102 = torch.ops.aten.mm.default(view_400, permute_186);  view_400 = permute_186 = None
        view_401 = torch.ops.aten.view.default(mm_102, [16, 128, 384]);  mm_102 = None
        view_402 = torch.ops.aten.view.default(view_401, [16, -1, 6, 64]);  view_401 = None
        permute_187 = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        permute_188 = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2])
        expand_64 = torch.ops.aten.expand.default(permute_183, [16, 6, 128, 64]);  permute_183 = None
        clone_123 = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_403 = torch.ops.aten.view.default(clone_123, [96, 128, 64]);  clone_123 = None
        expand_65 = torch.ops.aten.expand.default(permute_188, [16, 6, 64, 128]);  permute_188 = None
        clone_124 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_404 = torch.ops.aten.view.default(clone_124, [96, 64, 128]);  clone_124 = None
        bmm_32 = torch.ops.aten.bmm.default(view_403, view_404);  view_403 = view_404 = None
        view_405 = torch.ops.aten.view.default(bmm_32, [16, 6, 128, 128]);  bmm_32 = None
        add_106 = torch.ops.aten.add.Tensor(view_405, add_64);  view_405 = None
        view_406 = torch.ops.aten.view.default(add_106, [96, 128, 128]);  add_106 = None
        view_407 = torch.ops.aten.view.default(view_406, [16, 6, 128, 128]);  view_406 = None
        amax_16 = torch.ops.aten.amax.default(view_407, [-1], True)
        sub_21 = torch.ops.aten.sub.Tensor(view_407, amax_16);  view_407 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        expand_66 = torch.ops.aten.expand.default(div_20, [16, 6, 128, 128]);  div_20 = None
        view_408 = torch.ops.aten.view.default(expand_66, [96, 128, 128]);  expand_66 = None
        expand_67 = torch.ops.aten.expand.default(permute_187, [16, 6, 128, 64])
        clone_126 = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_409 = torch.ops.aten.view.default(clone_126, [96, 128, 64]);  clone_126 = None
        bmm_33 = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410 = torch.ops.aten.view.default(bmm_33, [16, 6, 128, 64]);  bmm_33 = None
        permute_189 = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_127 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_411 = torch.ops.aten.view.default(clone_127, [16, -1, 384]);  clone_127 = None
        permute_190 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        view_412 = torch.ops.aten.view.default(view_411, [2048, 384]);  view_411 = None
        mm_103 = torch.ops.aten.mm.default(view_412, permute_190);  view_412 = permute_190 = None
        view_413 = torch.ops.aten.view.default(mm_103, [16, 128, 512]);  mm_103 = None
        add_107 = torch.ops.aten.add.Tensor(add_104, view_413);  add_104 = view_413 = None
        pow_43 = torch.ops.aten.pow.Tensor_Scalar(add_107, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_108 = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_127 = torch.ops.aten.mul.Tensor(add_107, rsqrt_30);  rsqrt_30 = None
        mul_128 = torch.ops.aten.mul.Tensor(arg143_1, mul_127);  arg143_1 = mul_127 = None
        permute_191 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        view_414 = torch.ops.aten.view.default(mul_128, [2048, 512]);  mul_128 = None
        mm_104 = torch.ops.aten.mm.default(view_414, permute_191);  view_414 = permute_191 = None
        view_415 = torch.ops.aten.view.default(mm_104, [16, 128, 384]);  mm_104 = None
        view_416 = torch.ops.aten.view.default(view_415, [16, -1, 6, 64]);  view_415 = None
        permute_192 = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        permute_193 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        view_417 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_105 = torch.ops.aten.mm.default(view_417, permute_193);  view_417 = permute_193 = None
        view_418 = torch.ops.aten.view.default(mm_105, [16, 128, 384]);  mm_105 = None
        view_419 = torch.ops.aten.view.default(view_418, [16, -1, 6, 64]);  view_418 = None
        permute_194 = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
        permute_195 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        view_420 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_106 = torch.ops.aten.mm.default(view_420, permute_195);  view_420 = permute_195 = None
        view_421 = torch.ops.aten.view.default(mm_106, [16, 128, 384]);  mm_106 = None
        view_422 = torch.ops.aten.view.default(view_421, [16, -1, 6, 64]);  view_421 = None
        permute_196 = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        permute_197 = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2])
        expand_68 = torch.ops.aten.expand.default(permute_192, [16, 6, 128, 64]);  permute_192 = None
        clone_129 = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_423 = torch.ops.aten.view.default(clone_129, [96, 128, 64]);  clone_129 = None
        expand_69 = torch.ops.aten.expand.default(permute_197, [16, 6, 64, 128]);  permute_197 = None
        clone_130 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_424 = torch.ops.aten.view.default(clone_130, [96, 64, 128]);  clone_130 = None
        bmm_34 = torch.ops.aten.bmm.default(view_423, view_424);  view_423 = view_424 = None
        view_425 = torch.ops.aten.view.default(bmm_34, [16, 6, 128, 128]);  bmm_34 = None
        view_426 = torch.ops.aten.view.default(view_425, [96, 128, 128]);  view_425 = None
        view_427 = torch.ops.aten.view.default(view_426, [16, 6, 128, 128]);  view_426 = None
        amax_17 = torch.ops.aten.amax.default(view_427, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(view_427, amax_17);  view_427 = amax_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        expand_70 = torch.ops.aten.expand.default(div_21, [16, 6, 128, 128]);  div_21 = None
        view_428 = torch.ops.aten.view.default(expand_70, [96, 128, 128]);  expand_70 = None
        expand_71 = torch.ops.aten.expand.default(permute_196, [16, 6, 128, 64])
        clone_132 = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_429 = torch.ops.aten.view.default(clone_132, [96, 128, 64]);  clone_132 = None
        bmm_35 = torch.ops.aten.bmm.default(view_428, view_429);  view_428 = view_429 = None
        view_430 = torch.ops.aten.view.default(bmm_35, [16, 6, 128, 64]);  bmm_35 = None
        permute_198 = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_133 = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_431 = torch.ops.aten.view.default(clone_133, [16, -1, 384]);  clone_133 = None
        permute_199 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        view_432 = torch.ops.aten.view.default(view_431, [2048, 384]);  view_431 = None
        mm_107 = torch.ops.aten.mm.default(view_432, permute_199);  view_432 = permute_199 = None
        view_433 = torch.ops.aten.view.default(mm_107, [16, 128, 512]);  mm_107 = None
        add_110 = torch.ops.aten.add.Tensor(add_107, view_433);  add_107 = view_433 = None
        pow_44 = torch.ops.aten.pow.Tensor_Scalar(add_110, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        add_111 = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_129 = torch.ops.aten.mul.Tensor(add_110, rsqrt_31);  rsqrt_31 = None
        mul_130 = torch.ops.aten.mul.Tensor(arg147_1, mul_129);  arg147_1 = mul_129 = None
        permute_200 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        view_434 = torch.ops.aten.view.default(mul_130, [2048, 512])
        mm_108 = torch.ops.aten.mm.default(view_434, permute_200);  view_434 = permute_200 = None
        view_435 = torch.ops.aten.view.default(mm_108, [16, 128, 1024]);  mm_108 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_435, 0.5)
        pow_45 = torch.ops.aten.pow.Tensor_Scalar(view_435, 3.0)
        mul_132 = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_112 = torch.ops.aten.add.Tensor(view_435, mul_132);  view_435 = mul_132 = None
        mul_133 = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
        tanh_12 = torch.ops.aten.tanh.default(mul_133);  mul_133 = None
        add_113 = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_131, add_113);  mul_131 = add_113 = None
        permute_201 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        view_436 = torch.ops.aten.view.default(mul_130, [2048, 512]);  mul_130 = None
        mm_109 = torch.ops.aten.mm.default(view_436, permute_201);  view_436 = permute_201 = None
        view_437 = torch.ops.aten.view.default(mm_109, [16, 128, 1024]);  mm_109 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, view_437);  mul_134 = view_437 = None
        permute_202 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        view_438 = torch.ops.aten.view.default(mul_135, [2048, 1024]);  mul_135 = None
        mm_110 = torch.ops.aten.mm.default(view_438, permute_202);  view_438 = permute_202 = None
        view_439 = torch.ops.aten.view.default(mm_110, [16, 128, 512]);  mm_110 = None
        add_114 = torch.ops.aten.add.Tensor(add_110, view_439);  add_110 = view_439 = None
        pow_46 = torch.ops.aten.pow.Tensor_Scalar(add_114, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        add_115 = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_136 = torch.ops.aten.mul.Tensor(add_114, rsqrt_32);  rsqrt_32 = None
        mul_137 = torch.ops.aten.mul.Tensor(arg152_1, mul_136);  arg152_1 = mul_136 = None
        permute_203 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        view_440 = torch.ops.aten.view.default(mul_137, [2048, 512])
        mm_111 = torch.ops.aten.mm.default(view_440, permute_203);  view_440 = permute_203 = None
        view_441 = torch.ops.aten.view.default(mm_111, [16, 128, 384]);  mm_111 = None
        view_442 = torch.ops.aten.view.default(view_441, [16, -1, 6, 64]);  view_441 = None
        permute_204 = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        permute_205 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        view_443 = torch.ops.aten.view.default(mul_137, [2048, 512])
        mm_112 = torch.ops.aten.mm.default(view_443, permute_205);  view_443 = permute_205 = None
        view_444 = torch.ops.aten.view.default(mm_112, [16, 128, 384]);  mm_112 = None
        view_445 = torch.ops.aten.view.default(view_444, [16, -1, 6, 64]);  view_444 = None
        permute_206 = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        permute_207 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        view_446 = torch.ops.aten.view.default(mul_137, [2048, 512]);  mul_137 = None
        mm_113 = torch.ops.aten.mm.default(view_446, permute_207);  view_446 = permute_207 = None
        view_447 = torch.ops.aten.view.default(mm_113, [16, 128, 384]);  mm_113 = None
        view_448 = torch.ops.aten.view.default(view_447, [16, -1, 6, 64]);  view_447 = None
        permute_208 = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        permute_209 = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2])
        expand_72 = torch.ops.aten.expand.default(permute_204, [16, 6, 128, 64]);  permute_204 = None
        clone_137 = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_449 = torch.ops.aten.view.default(clone_137, [96, 128, 64]);  clone_137 = None
        expand_73 = torch.ops.aten.expand.default(permute_209, [16, 6, 64, 128]);  permute_209 = None
        clone_138 = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_450 = torch.ops.aten.view.default(clone_138, [96, 64, 128]);  clone_138 = None
        bmm_36 = torch.ops.aten.bmm.default(view_449, view_450);  view_449 = view_450 = None
        view_451 = torch.ops.aten.view.default(bmm_36, [16, 6, 128, 128]);  bmm_36 = None
        add_116 = torch.ops.aten.add.Tensor(view_451, add_64);  view_451 = None
        view_452 = torch.ops.aten.view.default(add_116, [96, 128, 128]);  add_116 = None
        view_453 = torch.ops.aten.view.default(view_452, [16, 6, 128, 128]);  view_452 = None
        amax_18 = torch.ops.aten.amax.default(view_453, [-1], True)
        sub_23 = torch.ops.aten.sub.Tensor(view_453, amax_18);  view_453 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        expand_74 = torch.ops.aten.expand.default(div_22, [16, 6, 128, 128]);  div_22 = None
        view_454 = torch.ops.aten.view.default(expand_74, [96, 128, 128]);  expand_74 = None
        expand_75 = torch.ops.aten.expand.default(permute_208, [16, 6, 128, 64])
        clone_140 = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_455 = torch.ops.aten.view.default(clone_140, [96, 128, 64]);  clone_140 = None
        bmm_37 = torch.ops.aten.bmm.default(view_454, view_455);  view_454 = view_455 = None
        view_456 = torch.ops.aten.view.default(bmm_37, [16, 6, 128, 64]);  bmm_37 = None
        permute_210 = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_141 = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_457 = torch.ops.aten.view.default(clone_141, [16, -1, 384]);  clone_141 = None
        permute_211 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        view_458 = torch.ops.aten.view.default(view_457, [2048, 384]);  view_457 = None
        mm_114 = torch.ops.aten.mm.default(view_458, permute_211);  view_458 = permute_211 = None
        view_459 = torch.ops.aten.view.default(mm_114, [16, 128, 512]);  mm_114 = None
        add_117 = torch.ops.aten.add.Tensor(add_114, view_459);  add_114 = view_459 = None
        pow_47 = torch.ops.aten.pow.Tensor_Scalar(add_117, 2)
        mean_33 = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_118 = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_138 = torch.ops.aten.mul.Tensor(add_117, rsqrt_33);  rsqrt_33 = None
        mul_139 = torch.ops.aten.mul.Tensor(arg157_1, mul_138);  arg157_1 = mul_138 = None
        permute_212 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        view_460 = torch.ops.aten.view.default(mul_139, [2048, 512]);  mul_139 = None
        mm_115 = torch.ops.aten.mm.default(view_460, permute_212);  view_460 = permute_212 = None
        view_461 = torch.ops.aten.view.default(mm_115, [16, 128, 384]);  mm_115 = None
        view_462 = torch.ops.aten.view.default(view_461, [16, -1, 6, 64]);  view_461 = None
        permute_213 = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        permute_214 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        view_463 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_116 = torch.ops.aten.mm.default(view_463, permute_214);  view_463 = permute_214 = None
        view_464 = torch.ops.aten.view.default(mm_116, [16, 128, 384]);  mm_116 = None
        view_465 = torch.ops.aten.view.default(view_464, [16, -1, 6, 64]);  view_464 = None
        permute_215 = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        permute_216 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        view_466 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_117 = torch.ops.aten.mm.default(view_466, permute_216);  view_466 = permute_216 = None
        view_467 = torch.ops.aten.view.default(mm_117, [16, 128, 384]);  mm_117 = None
        view_468 = torch.ops.aten.view.default(view_467, [16, -1, 6, 64]);  view_467 = None
        permute_217 = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        permute_218 = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2])
        expand_76 = torch.ops.aten.expand.default(permute_213, [16, 6, 128, 64]);  permute_213 = None
        clone_143 = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_469 = torch.ops.aten.view.default(clone_143, [96, 128, 64]);  clone_143 = None
        expand_77 = torch.ops.aten.expand.default(permute_218, [16, 6, 64, 128]);  permute_218 = None
        clone_144 = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_470 = torch.ops.aten.view.default(clone_144, [96, 64, 128]);  clone_144 = None
        bmm_38 = torch.ops.aten.bmm.default(view_469, view_470);  view_469 = view_470 = None
        view_471 = torch.ops.aten.view.default(bmm_38, [16, 6, 128, 128]);  bmm_38 = None
        view_472 = torch.ops.aten.view.default(view_471, [96, 128, 128]);  view_471 = None
        view_473 = torch.ops.aten.view.default(view_472, [16, 6, 128, 128]);  view_472 = None
        amax_19 = torch.ops.aten.amax.default(view_473, [-1], True)
        sub_24 = torch.ops.aten.sub.Tensor(view_473, amax_19);  view_473 = amax_19 = None
        exp_19 = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        expand_78 = torch.ops.aten.expand.default(div_23, [16, 6, 128, 128]);  div_23 = None
        view_474 = torch.ops.aten.view.default(expand_78, [96, 128, 128]);  expand_78 = None
        expand_79 = torch.ops.aten.expand.default(permute_217, [16, 6, 128, 64])
        clone_146 = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_475 = torch.ops.aten.view.default(clone_146, [96, 128, 64]);  clone_146 = None
        bmm_39 = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476 = torch.ops.aten.view.default(bmm_39, [16, 6, 128, 64]);  bmm_39 = None
        permute_219 = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_147 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_477 = torch.ops.aten.view.default(clone_147, [16, -1, 384]);  clone_147 = None
        permute_220 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        view_478 = torch.ops.aten.view.default(view_477, [2048, 384]);  view_477 = None
        mm_118 = torch.ops.aten.mm.default(view_478, permute_220);  view_478 = permute_220 = None
        view_479 = torch.ops.aten.view.default(mm_118, [16, 128, 512]);  mm_118 = None
        add_120 = torch.ops.aten.add.Tensor(add_117, view_479);  add_117 = view_479 = None
        pow_48 = torch.ops.aten.pow.Tensor_Scalar(add_120, 2)
        mean_34 = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        add_121 = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_140 = torch.ops.aten.mul.Tensor(add_120, rsqrt_34);  rsqrt_34 = None
        mul_141 = torch.ops.aten.mul.Tensor(arg161_1, mul_140);  arg161_1 = mul_140 = None
        permute_221 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        view_480 = torch.ops.aten.view.default(mul_141, [2048, 512])
        mm_119 = torch.ops.aten.mm.default(view_480, permute_221);  view_480 = permute_221 = None
        view_481 = torch.ops.aten.view.default(mm_119, [16, 128, 1024]);  mm_119 = None
        mul_142 = torch.ops.aten.mul.Tensor(view_481, 0.5)
        pow_49 = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_143 = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_122 = torch.ops.aten.add.Tensor(view_481, mul_143);  view_481 = mul_143 = None
        mul_144 = torch.ops.aten.mul.Tensor(add_122, 0.7978845608028654);  add_122 = None
        tanh_13 = torch.ops.aten.tanh.default(mul_144);  mul_144 = None
        add_123 = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_142, add_123);  mul_142 = add_123 = None
        permute_222 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        view_482 = torch.ops.aten.view.default(mul_141, [2048, 512]);  mul_141 = None
        mm_120 = torch.ops.aten.mm.default(view_482, permute_222);  view_482 = permute_222 = None
        view_483 = torch.ops.aten.view.default(mm_120, [16, 128, 1024]);  mm_120 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, view_483);  mul_145 = view_483 = None
        permute_223 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        view_484 = torch.ops.aten.view.default(mul_146, [2048, 1024]);  mul_146 = None
        mm_121 = torch.ops.aten.mm.default(view_484, permute_223);  view_484 = permute_223 = None
        view_485 = torch.ops.aten.view.default(mm_121, [16, 128, 512]);  mm_121 = None
        add_124 = torch.ops.aten.add.Tensor(add_120, view_485);  add_120 = view_485 = None
        pow_50 = torch.ops.aten.pow.Tensor_Scalar(add_124, 2)
        mean_35 = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        add_125 = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_147 = torch.ops.aten.mul.Tensor(add_124, rsqrt_35);  rsqrt_35 = None
        mul_148 = torch.ops.aten.mul.Tensor(arg166_1, mul_147);  arg166_1 = mul_147 = None
        permute_224 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        view_486 = torch.ops.aten.view.default(mul_148, [2048, 512])
        mm_122 = torch.ops.aten.mm.default(view_486, permute_224);  view_486 = permute_224 = None
        view_487 = torch.ops.aten.view.default(mm_122, [16, 128, 384]);  mm_122 = None
        view_488 = torch.ops.aten.view.default(view_487, [16, -1, 6, 64]);  view_487 = None
        permute_225 = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        permute_226 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        view_489 = torch.ops.aten.view.default(mul_148, [2048, 512])
        mm_123 = torch.ops.aten.mm.default(view_489, permute_226);  view_489 = permute_226 = None
        view_490 = torch.ops.aten.view.default(mm_123, [16, 128, 384]);  mm_123 = None
        view_491 = torch.ops.aten.view.default(view_490, [16, -1, 6, 64]);  view_490 = None
        permute_227 = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        permute_228 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        view_492 = torch.ops.aten.view.default(mul_148, [2048, 512]);  mul_148 = None
        mm_124 = torch.ops.aten.mm.default(view_492, permute_228);  view_492 = permute_228 = None
        view_493 = torch.ops.aten.view.default(mm_124, [16, 128, 384]);  mm_124 = None
        view_494 = torch.ops.aten.view.default(view_493, [16, -1, 6, 64]);  view_493 = None
        permute_229 = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        permute_230 = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2])
        expand_80 = torch.ops.aten.expand.default(permute_225, [16, 6, 128, 64]);  permute_225 = None
        clone_151 = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_495 = torch.ops.aten.view.default(clone_151, [96, 128, 64]);  clone_151 = None
        expand_81 = torch.ops.aten.expand.default(permute_230, [16, 6, 64, 128]);  permute_230 = None
        clone_152 = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_496 = torch.ops.aten.view.default(clone_152, [96, 64, 128]);  clone_152 = None
        bmm_40 = torch.ops.aten.bmm.default(view_495, view_496);  view_495 = view_496 = None
        view_497 = torch.ops.aten.view.default(bmm_40, [16, 6, 128, 128]);  bmm_40 = None
        add_126 = torch.ops.aten.add.Tensor(view_497, add_64);  view_497 = None
        view_498 = torch.ops.aten.view.default(add_126, [96, 128, 128]);  add_126 = None
        view_499 = torch.ops.aten.view.default(view_498, [16, 6, 128, 128]);  view_498 = None
        amax_20 = torch.ops.aten.amax.default(view_499, [-1], True)
        sub_25 = torch.ops.aten.sub.Tensor(view_499, amax_20);  view_499 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        expand_82 = torch.ops.aten.expand.default(div_24, [16, 6, 128, 128]);  div_24 = None
        view_500 = torch.ops.aten.view.default(expand_82, [96, 128, 128]);  expand_82 = None
        expand_83 = torch.ops.aten.expand.default(permute_229, [16, 6, 128, 64])
        clone_154 = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_501 = torch.ops.aten.view.default(clone_154, [96, 128, 64]);  clone_154 = None
        bmm_41 = torch.ops.aten.bmm.default(view_500, view_501);  view_500 = view_501 = None
        view_502 = torch.ops.aten.view.default(bmm_41, [16, 6, 128, 64]);  bmm_41 = None
        permute_231 = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
        clone_155 = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_503 = torch.ops.aten.view.default(clone_155, [16, -1, 384]);  clone_155 = None
        permute_232 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        view_504 = torch.ops.aten.view.default(view_503, [2048, 384]);  view_503 = None
        mm_125 = torch.ops.aten.mm.default(view_504, permute_232);  view_504 = permute_232 = None
        view_505 = torch.ops.aten.view.default(mm_125, [16, 128, 512]);  mm_125 = None
        add_127 = torch.ops.aten.add.Tensor(add_124, view_505);  add_124 = view_505 = None
        pow_51 = torch.ops.aten.pow.Tensor_Scalar(add_127, 2)
        mean_36 = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_128 = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_149 = torch.ops.aten.mul.Tensor(add_127, rsqrt_36);  rsqrt_36 = None
        mul_150 = torch.ops.aten.mul.Tensor(arg171_1, mul_149);  arg171_1 = mul_149 = None
        permute_233 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        view_506 = torch.ops.aten.view.default(mul_150, [2048, 512]);  mul_150 = None
        mm_126 = torch.ops.aten.mm.default(view_506, permute_233);  view_506 = permute_233 = None
        view_507 = torch.ops.aten.view.default(mm_126, [16, 128, 384]);  mm_126 = None
        view_508 = torch.ops.aten.view.default(view_507, [16, -1, 6, 64]);  view_507 = None
        permute_234 = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        permute_235 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        view_509 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_127 = torch.ops.aten.mm.default(view_509, permute_235);  view_509 = permute_235 = None
        view_510 = torch.ops.aten.view.default(mm_127, [16, 128, 384]);  mm_127 = None
        view_511 = torch.ops.aten.view.default(view_510, [16, -1, 6, 64]);  view_510 = None
        permute_236 = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        permute_237 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        view_512 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_128 = torch.ops.aten.mm.default(view_512, permute_237);  view_512 = permute_237 = None
        view_513 = torch.ops.aten.view.default(mm_128, [16, 128, 384]);  mm_128 = None
        view_514 = torch.ops.aten.view.default(view_513, [16, -1, 6, 64]);  view_513 = None
        permute_238 = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        permute_239 = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2])
        expand_84 = torch.ops.aten.expand.default(permute_234, [16, 6, 128, 64]);  permute_234 = None
        clone_157 = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_515 = torch.ops.aten.view.default(clone_157, [96, 128, 64]);  clone_157 = None
        expand_85 = torch.ops.aten.expand.default(permute_239, [16, 6, 64, 128]);  permute_239 = None
        clone_158 = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_516 = torch.ops.aten.view.default(clone_158, [96, 64, 128]);  clone_158 = None
        bmm_42 = torch.ops.aten.bmm.default(view_515, view_516);  view_515 = view_516 = None
        view_517 = torch.ops.aten.view.default(bmm_42, [16, 6, 128, 128]);  bmm_42 = None
        view_518 = torch.ops.aten.view.default(view_517, [96, 128, 128]);  view_517 = None
        view_519 = torch.ops.aten.view.default(view_518, [16, 6, 128, 128]);  view_518 = None
        amax_21 = torch.ops.aten.amax.default(view_519, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(view_519, amax_21);  view_519 = amax_21 = None
        exp_21 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        expand_86 = torch.ops.aten.expand.default(div_25, [16, 6, 128, 128]);  div_25 = None
        view_520 = torch.ops.aten.view.default(expand_86, [96, 128, 128]);  expand_86 = None
        expand_87 = torch.ops.aten.expand.default(permute_238, [16, 6, 128, 64])
        clone_160 = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_521 = torch.ops.aten.view.default(clone_160, [96, 128, 64]);  clone_160 = None
        bmm_43 = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
        view_522 = torch.ops.aten.view.default(bmm_43, [16, 6, 128, 64]);  bmm_43 = None
        permute_240 = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        clone_161 = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_523 = torch.ops.aten.view.default(clone_161, [16, -1, 384]);  clone_161 = None
        permute_241 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        view_524 = torch.ops.aten.view.default(view_523, [2048, 384]);  view_523 = None
        mm_129 = torch.ops.aten.mm.default(view_524, permute_241);  view_524 = permute_241 = None
        view_525 = torch.ops.aten.view.default(mm_129, [16, 128, 512]);  mm_129 = None
        add_130 = torch.ops.aten.add.Tensor(add_127, view_525);  add_127 = view_525 = None
        pow_52 = torch.ops.aten.pow.Tensor_Scalar(add_130, 2)
        mean_37 = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        add_131 = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_151 = torch.ops.aten.mul.Tensor(add_130, rsqrt_37);  rsqrt_37 = None
        mul_152 = torch.ops.aten.mul.Tensor(arg175_1, mul_151);  arg175_1 = mul_151 = None
        permute_242 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        view_526 = torch.ops.aten.view.default(mul_152, [2048, 512])
        mm_130 = torch.ops.aten.mm.default(view_526, permute_242);  view_526 = permute_242 = None
        view_527 = torch.ops.aten.view.default(mm_130, [16, 128, 1024]);  mm_130 = None
        mul_153 = torch.ops.aten.mul.Tensor(view_527, 0.5)
        pow_53 = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
        mul_154 = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_132 = torch.ops.aten.add.Tensor(view_527, mul_154);  view_527 = mul_154 = None
        mul_155 = torch.ops.aten.mul.Tensor(add_132, 0.7978845608028654);  add_132 = None
        tanh_14 = torch.ops.aten.tanh.default(mul_155);  mul_155 = None
        add_133 = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_153, add_133);  mul_153 = add_133 = None
        permute_243 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        view_528 = torch.ops.aten.view.default(mul_152, [2048, 512]);  mul_152 = None
        mm_131 = torch.ops.aten.mm.default(view_528, permute_243);  view_528 = permute_243 = None
        view_529 = torch.ops.aten.view.default(mm_131, [16, 128, 1024]);  mm_131 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, view_529);  mul_156 = view_529 = None
        permute_244 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        view_530 = torch.ops.aten.view.default(mul_157, [2048, 1024]);  mul_157 = None
        mm_132 = torch.ops.aten.mm.default(view_530, permute_244);  view_530 = permute_244 = None
        view_531 = torch.ops.aten.view.default(mm_132, [16, 128, 512]);  mm_132 = None
        add_134 = torch.ops.aten.add.Tensor(add_130, view_531);  add_130 = view_531 = None
        pow_54 = torch.ops.aten.pow.Tensor_Scalar(add_134, 2)
        mean_38 = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        add_135 = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_158 = torch.ops.aten.mul.Tensor(add_134, rsqrt_38);  rsqrt_38 = None
        mul_159 = torch.ops.aten.mul.Tensor(arg180_1, mul_158);  arg180_1 = mul_158 = None
        permute_245 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        view_532 = torch.ops.aten.view.default(mul_159, [2048, 512])
        mm_133 = torch.ops.aten.mm.default(view_532, permute_245);  view_532 = permute_245 = None
        view_533 = torch.ops.aten.view.default(mm_133, [16, 128, 384]);  mm_133 = None
        view_534 = torch.ops.aten.view.default(view_533, [16, -1, 6, 64]);  view_533 = None
        permute_246 = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        permute_247 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        view_535 = torch.ops.aten.view.default(mul_159, [2048, 512])
        mm_134 = torch.ops.aten.mm.default(view_535, permute_247);  view_535 = permute_247 = None
        view_536 = torch.ops.aten.view.default(mm_134, [16, 128, 384]);  mm_134 = None
        view_537 = torch.ops.aten.view.default(view_536, [16, -1, 6, 64]);  view_536 = None
        permute_248 = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
        permute_249 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        view_538 = torch.ops.aten.view.default(mul_159, [2048, 512]);  mul_159 = None
        mm_135 = torch.ops.aten.mm.default(view_538, permute_249);  view_538 = permute_249 = None
        view_539 = torch.ops.aten.view.default(mm_135, [16, 128, 384]);  mm_135 = None
        view_540 = torch.ops.aten.view.default(view_539, [16, -1, 6, 64]);  view_539 = None
        permute_250 = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        permute_251 = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2])
        expand_88 = torch.ops.aten.expand.default(permute_246, [16, 6, 128, 64]);  permute_246 = None
        clone_165 = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_541 = torch.ops.aten.view.default(clone_165, [96, 128, 64]);  clone_165 = None
        expand_89 = torch.ops.aten.expand.default(permute_251, [16, 6, 64, 128]);  permute_251 = None
        clone_166 = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_542 = torch.ops.aten.view.default(clone_166, [96, 64, 128]);  clone_166 = None
        bmm_44 = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
        view_543 = torch.ops.aten.view.default(bmm_44, [16, 6, 128, 128]);  bmm_44 = None
        add_136 = torch.ops.aten.add.Tensor(view_543, add_64);  view_543 = add_64 = None
        view_544 = torch.ops.aten.view.default(add_136, [96, 128, 128]);  add_136 = None
        view_545 = torch.ops.aten.view.default(view_544, [16, 6, 128, 128]);  view_544 = None
        amax_22 = torch.ops.aten.amax.default(view_545, [-1], True)
        sub_27 = torch.ops.aten.sub.Tensor(view_545, amax_22);  view_545 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        expand_90 = torch.ops.aten.expand.default(div_26, [16, 6, 128, 128]);  div_26 = None
        view_546 = torch.ops.aten.view.default(expand_90, [96, 128, 128]);  expand_90 = None
        expand_91 = torch.ops.aten.expand.default(permute_250, [16, 6, 128, 64])
        clone_168 = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_547 = torch.ops.aten.view.default(clone_168, [96, 128, 64]);  clone_168 = None
        bmm_45 = torch.ops.aten.bmm.default(view_546, view_547);  view_546 = view_547 = None
        view_548 = torch.ops.aten.view.default(bmm_45, [16, 6, 128, 64]);  bmm_45 = None
        permute_252 = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
        clone_169 = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_549 = torch.ops.aten.view.default(clone_169, [16, -1, 384]);  clone_169 = None
        permute_253 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        view_550 = torch.ops.aten.view.default(view_549, [2048, 384]);  view_549 = None
        mm_136 = torch.ops.aten.mm.default(view_550, permute_253);  view_550 = permute_253 = None
        view_551 = torch.ops.aten.view.default(mm_136, [16, 128, 512]);  mm_136 = None
        add_137 = torch.ops.aten.add.Tensor(add_134, view_551);  add_134 = view_551 = None
        pow_55 = torch.ops.aten.pow.Tensor_Scalar(add_137, 2)
        mean_39 = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_138 = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_160 = torch.ops.aten.mul.Tensor(add_137, rsqrt_39);  rsqrt_39 = None
        mul_161 = torch.ops.aten.mul.Tensor(arg185_1, mul_160);  arg185_1 = mul_160 = None
        permute_254 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        view_552 = torch.ops.aten.view.default(mul_161, [2048, 512]);  mul_161 = None
        mm_137 = torch.ops.aten.mm.default(view_552, permute_254);  view_552 = permute_254 = None
        view_553 = torch.ops.aten.view.default(mm_137, [16, 128, 384]);  mm_137 = None
        view_554 = torch.ops.aten.view.default(view_553, [16, -1, 6, 64]);  view_553 = None
        permute_255 = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        permute_256 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        view_555 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_138 = torch.ops.aten.mm.default(view_555, permute_256);  view_555 = permute_256 = None
        view_556 = torch.ops.aten.view.default(mm_138, [16, 128, 384]);  mm_138 = None
        view_557 = torch.ops.aten.view.default(view_556, [16, -1, 6, 64]);  view_556 = None
        permute_257 = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
        permute_258 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        view_558 = torch.ops.aten.view.default(mul_76, [2048, 512])
        mm_139 = torch.ops.aten.mm.default(view_558, permute_258);  view_558 = permute_258 = None
        view_559 = torch.ops.aten.view.default(mm_139, [16, 128, 384]);  mm_139 = None
        view_560 = torch.ops.aten.view.default(view_559, [16, -1, 6, 64]);  view_559 = None
        permute_259 = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        permute_260 = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
        expand_92 = torch.ops.aten.expand.default(permute_255, [16, 6, 128, 64]);  permute_255 = None
        clone_171 = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_561 = torch.ops.aten.view.default(clone_171, [96, 128, 64]);  clone_171 = None
        expand_93 = torch.ops.aten.expand.default(permute_260, [16, 6, 64, 128]);  permute_260 = None
        clone_172 = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_562 = torch.ops.aten.view.default(clone_172, [96, 64, 128]);  clone_172 = None
        bmm_46 = torch.ops.aten.bmm.default(view_561, view_562);  view_561 = view_562 = None
        view_563 = torch.ops.aten.view.default(bmm_46, [16, 6, 128, 128]);  bmm_46 = None
        view_564 = torch.ops.aten.view.default(view_563, [96, 128, 128]);  view_563 = None
        view_565 = torch.ops.aten.view.default(view_564, [16, 6, 128, 128]);  view_564 = None
        amax_23 = torch.ops.aten.amax.default(view_565, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(view_565, amax_23);  view_565 = amax_23 = None
        exp_23 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        expand_94 = torch.ops.aten.expand.default(div_27, [16, 6, 128, 128]);  div_27 = None
        view_566 = torch.ops.aten.view.default(expand_94, [96, 128, 128]);  expand_94 = None
        expand_95 = torch.ops.aten.expand.default(permute_259, [16, 6, 128, 64])
        clone_174 = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_567 = torch.ops.aten.view.default(clone_174, [96, 128, 64]);  clone_174 = None
        bmm_47 = torch.ops.aten.bmm.default(view_566, view_567);  view_566 = view_567 = None
        view_568 = torch.ops.aten.view.default(bmm_47, [16, 6, 128, 64]);  bmm_47 = None
        permute_261 = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        clone_175 = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_569 = torch.ops.aten.view.default(clone_175, [16, -1, 384]);  clone_175 = None
        permute_262 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        view_570 = torch.ops.aten.view.default(view_569, [2048, 384]);  view_569 = None
        mm_140 = torch.ops.aten.mm.default(view_570, permute_262);  view_570 = permute_262 = None
        view_571 = torch.ops.aten.view.default(mm_140, [16, 128, 512]);  mm_140 = None
        add_140 = torch.ops.aten.add.Tensor(add_137, view_571);  add_137 = view_571 = None
        pow_56 = torch.ops.aten.pow.Tensor_Scalar(add_140, 2)
        mean_40 = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        add_141 = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_162 = torch.ops.aten.mul.Tensor(add_140, rsqrt_40);  rsqrt_40 = None
        mul_163 = torch.ops.aten.mul.Tensor(arg189_1, mul_162);  arg189_1 = mul_162 = None
        permute_263 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        view_572 = torch.ops.aten.view.default(mul_163, [2048, 512])
        mm_141 = torch.ops.aten.mm.default(view_572, permute_263);  view_572 = permute_263 = None
        view_573 = torch.ops.aten.view.default(mm_141, [16, 128, 1024]);  mm_141 = None
        mul_164 = torch.ops.aten.mul.Tensor(view_573, 0.5)
        pow_57 = torch.ops.aten.pow.Tensor_Scalar(view_573, 3.0)
        mul_165 = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_142 = torch.ops.aten.add.Tensor(view_573, mul_165);  view_573 = mul_165 = None
        mul_166 = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
        tanh_15 = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_143 = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_164, add_143);  mul_164 = add_143 = None
        permute_264 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        view_574 = torch.ops.aten.view.default(mul_163, [2048, 512]);  mul_163 = None
        mm_142 = torch.ops.aten.mm.default(view_574, permute_264);  view_574 = permute_264 = None
        view_575 = torch.ops.aten.view.default(mm_142, [16, 128, 1024]);  mm_142 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_167, view_575);  mul_167 = view_575 = None
        permute_265 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        view_576 = torch.ops.aten.view.default(mul_168, [2048, 1024]);  mul_168 = None
        mm_143 = torch.ops.aten.mm.default(view_576, permute_265);  view_576 = permute_265 = None
        view_577 = torch.ops.aten.view.default(mm_143, [16, 128, 512]);  mm_143 = None
        add_144 = torch.ops.aten.add.Tensor(add_140, view_577);  add_140 = view_577 = None
        pow_58 = torch.ops.aten.pow.Tensor_Scalar(add_144, 2)
        mean_41 = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        add_145 = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_169 = torch.ops.aten.mul.Tensor(add_144, rsqrt_41);  add_144 = rsqrt_41 = None
        mul_170 = torch.ops.aten.mul.Tensor(arg190_1, mul_169);  arg190_1 = mul_169 = None
        permute_266 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        view_578 = torch.ops.aten.view.default(mul_170, [2048, 512]);  mul_170 = None
        mm_144 = torch.ops.aten.mm.default(view_578, permute_266);  view_578 = permute_266 = None
        view_579 = torch.ops.aten.view.default(mm_144, [16, 128, 250112]);  mm_144 = None
        view_580 = torch.ops.aten.view.default(view_579, [-1, 250112])
        view_581 = torch.ops.aten.view.default(arg76_1, [-1]);  arg76_1 = None
        amax_24 = torch.ops.aten.amax.default(view_580, [1], True)
        sub_29 = torch.ops.aten.sub.Tensor(view_580, amax_24);  view_580 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_29)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log_2 = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_30 = torch.ops.aten.sub.Tensor(sub_29, log_2);  sub_29 = log_2 = None
        ne = torch.ops.aten.ne.Scalar(view_581, -100)
        full_default_6 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne, view_581, full_default_6);  ne = full_default_6 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather = torch.ops.aten.gather.default(sub_30, 1, unsqueeze_17);  sub_30 = unsqueeze_17 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1 = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_581, -100)
        full_default_7 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_1, neg_1, full_default_7);  ne_1 = neg_1 = full_default_7 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_581, -100);  view_581 = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_3);  where_3 = None
        div_28 = torch.ops.aten.div.Tensor(sum_27, convert_element_type_7);  sum_27 = convert_element_type_7 = None
        return (div_28, view_579, permute_100, permute_102, permute_110, permute_112, permute_122, permute_124, permute_131, permute_133, permute_143, permute_145, permute_152, permute_154, permute_164, permute_166, permute_173, permute_175, permute_185, permute_187, permute_194, permute_196, permute_206, permute_208, permute_215, permute_217, permute_227, permute_229, permute_236, permute_238, permute_248, permute_250, permute_257, permute_259, mul_76)
        
def load_args(reader):
    buf0 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 512229376, device=device(type='cuda', index=0))
    reader.tensor(buf1, (250112, 512), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf2, (384, 512), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf3, (384, 512), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf4, (384, 512), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 384), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf6, (32, 6), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1024, 512), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024, 512), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512, 1024), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf12, (384, 512), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf13, (384, 512), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf14, (384, 512), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512, 384), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1024, 512), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1024, 512), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512, 1024), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf21, (384, 512), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf22, (384, 512), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf23, (384, 512), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 384), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1024, 512), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1024, 512), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512, 1024), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf30, (384, 512), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf31, (384, 512), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf32, (384, 512), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512, 384), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1024, 512), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024, 512), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512, 1024), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384, 512), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf40, (384, 512), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf41, (384, 512), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512, 384), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf44, (1024, 512), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1024, 512), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512, 1024), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf48, (384, 512), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf49, (384, 512), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf50, (384, 512), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512, 384), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1024, 512), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf54, (1024, 512), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512, 1024), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf57, (384, 512), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf58, (384, 512), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf59, (384, 512), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf60, (512, 384), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1024, 512), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1024, 512), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512, 1024), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf66, (384, 512), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf67, (384, 512), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf68, (384, 512), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512, 384), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1024, 512), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1024, 512), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 1024), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf76, (16, 128), dtype=torch.int64, is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf77, (384, 512), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384, 512), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf79, (384, 512), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512, 384), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf81, (32, 6), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf83, (384, 512), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf84, (384, 512), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf85, (384, 512), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512, 384), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1024, 512), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1024, 512), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512, 1024), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf92, (384, 512), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf93, (384, 512), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf94, (384, 512), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512, 384), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf96, (512,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf97, (384, 512), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf98, (384, 512), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384, 512), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512, 384), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1024, 512), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1024, 512), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512, 1024), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf106, (384, 512), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384, 512), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf108, (384, 512), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512, 384), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384, 512), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384, 512), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384, 512), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512, 384), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1024, 512), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024, 512), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512, 1024), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf120, (384, 512), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf121, (384, 512), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384, 512), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf123, (512, 384), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf124, (512,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384, 512), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384, 512), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384, 512), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512, 384), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1024, 512), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024, 512), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512, 1024), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384, 512), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf135, (384, 512), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf136, (384, 512), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512, 384), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf139, (384, 512), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf140, (384, 512), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf141, (384, 512), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf142, (512, 384), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf143, (512,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1024, 512), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1024, 512), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf146, (512, 1024), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf147, (512,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf148, (384, 512), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf149, (384, 512), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384, 512), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf151, (512, 384), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf152, (512,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf153, (384, 512), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf154, (384, 512), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf155, (384, 512), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512, 384), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf157, (512,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1024, 512), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1024, 512), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf160, (512, 1024), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf161, (512,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf162, (384, 512), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf163, (384, 512), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf164, (384, 512), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf165, (512, 384), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf166, (512,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf167, (384, 512), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf168, (384, 512), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf169, (384, 512), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf170, (512, 384), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf171, (512,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1024, 512), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1024, 512), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512, 1024), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf176, (384, 512), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf177, (384, 512), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf178, (384, 512), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf179, (512, 384), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf181, (384, 512), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf182, (384, 512), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf183, (384, 512), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf184, (512, 384), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf185, (512,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1024, 512), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1024, 512), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf188, (512, 1024), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf189, (512,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf190, (512,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 512229376, device=device(type='cuda', index=0))
    reader.tensor(buf191, (250112, 512), is_leaf=True)  # arg191_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)