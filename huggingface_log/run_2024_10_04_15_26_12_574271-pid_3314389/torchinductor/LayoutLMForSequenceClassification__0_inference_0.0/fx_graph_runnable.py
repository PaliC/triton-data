
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1):
        full = torch.ops.aten.full.default([16, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default = torch.ops.aten.full.default([16, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([16, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default_2 = torch.ops.aten.full.default([16, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, arg204_1);  arg2_1 = arg204_1 = None
        select = torch.ops.aten.select.int(full_default_1, 2, 0)
        embedding_2 = torch.ops.aten.embedding.default(arg3_1, select);  select = None
        select_1 = torch.ops.aten.select.int(full_default_1, 2, 1)
        embedding_3 = torch.ops.aten.embedding.default(arg4_1, select_1);  select_1 = None
        select_2 = torch.ops.aten.select.int(full_default_1, 2, 2)
        embedding_4 = torch.ops.aten.embedding.default(arg3_1, select_2);  arg3_1 = select_2 = None
        select_3 = torch.ops.aten.select.int(full_default_1, 2, 3)
        embedding_5 = torch.ops.aten.embedding.default(arg4_1, select_3);  arg4_1 = select_3 = None
        select_4 = torch.ops.aten.select.int(full_default_1, 2, 3)
        select_5 = torch.ops.aten.select.int(full_default_1, 2, 1)
        sub_1 = torch.ops.aten.sub.Tensor(select_4, select_5);  select_4 = select_5 = None
        embedding_6 = torch.ops.aten.embedding.default(arg5_1, sub_1);  arg5_1 = sub_1 = None
        select_6 = torch.ops.aten.select.int(full_default_1, 2, 2)
        select_7 = torch.ops.aten.select.int(full_default_1, 2, 0);  full_default_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(select_6, select_7);  select_6 = select_7 = None
        embedding_7 = torch.ops.aten.embedding.default(arg6_1, sub_2);  arg6_1 = sub_2 = None
        embedding_8 = torch.ops.aten.embedding.default(arg7_1, full_default);  arg7_1 = full_default = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None
        add_5 = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None
        add_6 = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None
        add_7 = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None
        var_mean = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_8 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg8_1);  mul_1 = arg8_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        view = torch.ops.aten.view.default(add_9, [8192, 768])
        permute = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm = torch.ops.aten.addmm.default(arg11_1, view, permute);  arg11_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [16, 512, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(add_9, [8192, 768])
        permute_1 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg13_1, view_2, permute_1);  arg13_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [16, 512, 768]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [16, 512, 12, 64]);  view_3 = None
        view_5 = torch.ops.aten.view.default(add_9, [8192, 768])
        permute_3 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg15_1, view_5, permute_3);  arg15_1 = view_5 = permute_3 = None
        view_6 = torch.ops.aten.view.default(addmm_2, [16, 512, 768]);  addmm_2 = None
        view_7 = torch.ops.aten.view.default(view_6, [16, 512, 12, 64]);  view_6 = None
        view_8 = torch.ops.aten.view.default(view_1, [16, 512, 12, 64]);  view_1 = None
        permute_default_33 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        permute_default_34 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_default_35 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_61 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        permute_7 = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15 = torch.ops.aten.view.default(clone_5, [16, 512, 768]);  clone_5 = None
        view_16 = torch.ops.aten.view.default(view_15, [8192, 768]);  view_15 = None
        permute_8 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg17_1, view_16, permute_8);  arg17_1 = view_16 = permute_8 = None
        view_17 = torch.ops.aten.view.default(addmm_3, [16, 512, 768]);  addmm_3 = None
        add_11 = torch.ops.aten.add.Tensor(view_17, add_9);  view_17 = add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_11, getitem_3);  add_11 = getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg18_1);  mul_3 = arg18_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_4, arg19_1);  mul_4 = arg19_1 = None
        view_18 = torch.ops.aten.view.default(add_13, [8192, 768])
        permute_9 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg21_1, view_18, permute_9);  arg21_1 = view_18 = permute_9 = None
        view_19 = torch.ops.aten.view.default(addmm_4, [16, 512, 3072]);  addmm_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_6 = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_14 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_14);  mul_5 = add_14 = None
        view_20 = torch.ops.aten.view.default(mul_7, [8192, 3072]);  mul_7 = None
        permute_10 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg23_1, view_20, permute_10);  arg23_1 = view_20 = permute_10 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [16, 512, 768]);  addmm_5 = None
        add_15 = torch.ops.aten.add.Tensor(view_21, add_13);  view_21 = add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg24_1);  mul_8 = arg24_1 = None
        add_17 = torch.ops.aten.add.Tensor(mul_9, arg25_1);  mul_9 = arg25_1 = None
        view_22 = torch.ops.aten.view.default(add_17, [8192, 768])
        permute_11 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg27_1, view_22, permute_11);  arg27_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [16, 512, 768]);  addmm_6 = None
        view_24 = torch.ops.aten.view.default(add_17, [8192, 768])
        permute_12 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg29_1, view_24, permute_12);  arg29_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [16, 512, 768]);  addmm_7 = None
        view_26 = torch.ops.aten.view.default(view_25, [16, 512, 12, 64]);  view_25 = None
        view_27 = torch.ops.aten.view.default(add_17, [8192, 768])
        permute_14 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg31_1, view_27, permute_14);  arg31_1 = view_27 = permute_14 = None
        view_28 = torch.ops.aten.view.default(addmm_8, [16, 512, 768]);  addmm_8 = None
        view_29 = torch.ops.aten.view.default(view_28, [16, 512, 12, 64]);  view_28 = None
        view_30 = torch.ops.aten.view.default(view_23, [16, 512, 12, 64]);  view_23 = None
        permute_default_30 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        permute_default_31 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        permute_default_32 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_60 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        permute_18 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        clone_12 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37 = torch.ops.aten.view.default(clone_12, [16, 512, 768]);  clone_12 = None
        view_38 = torch.ops.aten.view.default(view_37, [8192, 768]);  view_37 = None
        permute_19 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg33_1, view_38, permute_19);  arg33_1 = view_38 = permute_19 = None
        view_39 = torch.ops.aten.view.default(addmm_9, [16, 512, 768]);  addmm_9 = None
        add_19 = torch.ops.aten.add.Tensor(view_39, add_17);  view_39 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg34_1);  mul_10 = arg34_1 = None
        add_21 = torch.ops.aten.add.Tensor(mul_11, arg35_1);  mul_11 = arg35_1 = None
        view_40 = torch.ops.aten.view.default(add_21, [8192, 768])
        permute_20 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg37_1, view_40, permute_20);  arg37_1 = view_40 = permute_20 = None
        view_41 = torch.ops.aten.view.default(addmm_10, [16, 512, 3072]);  addmm_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_13 = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1 = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_22 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_12, add_22);  mul_12 = add_22 = None
        view_42 = torch.ops.aten.view.default(mul_14, [8192, 3072]);  mul_14 = None
        permute_21 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg39_1, view_42, permute_21);  arg39_1 = view_42 = permute_21 = None
        view_43 = torch.ops.aten.view.default(addmm_11, [16, 512, 768]);  addmm_11 = None
        add_23 = torch.ops.aten.add.Tensor(view_43, add_21);  view_43 = add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_23, getitem_9);  add_23 = getitem_9 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, arg40_1);  mul_15 = arg40_1 = None
        add_25 = torch.ops.aten.add.Tensor(mul_16, arg41_1);  mul_16 = arg41_1 = None
        view_44 = torch.ops.aten.view.default(add_25, [8192, 768])
        permute_22 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg43_1, view_44, permute_22);  arg43_1 = view_44 = permute_22 = None
        view_45 = torch.ops.aten.view.default(addmm_12, [16, 512, 768]);  addmm_12 = None
        view_46 = torch.ops.aten.view.default(add_25, [8192, 768])
        permute_23 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg45_1, view_46, permute_23);  arg45_1 = view_46 = permute_23 = None
        view_47 = torch.ops.aten.view.default(addmm_13, [16, 512, 768]);  addmm_13 = None
        view_48 = torch.ops.aten.view.default(view_47, [16, 512, 12, 64]);  view_47 = None
        view_49 = torch.ops.aten.view.default(add_25, [8192, 768])
        permute_25 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg47_1, view_49, permute_25);  arg47_1 = view_49 = permute_25 = None
        view_50 = torch.ops.aten.view.default(addmm_14, [16, 512, 768]);  addmm_14 = None
        view_51 = torch.ops.aten.view.default(view_50, [16, 512, 12, 64]);  view_50 = None
        view_52 = torch.ops.aten.view.default(view_45, [16, 512, 12, 64]);  view_45 = None
        permute_default_27 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        permute_default_28 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        permute_default_29 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_59 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        permute_29 = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        clone_19 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59 = torch.ops.aten.view.default(clone_19, [16, 512, 768]);  clone_19 = None
        view_60 = torch.ops.aten.view.default(view_59, [8192, 768]);  view_59 = None
        permute_30 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg49_1, view_60, permute_30);  arg49_1 = view_60 = permute_30 = None
        view_61 = torch.ops.aten.view.default(addmm_15, [16, 512, 768]);  addmm_15 = None
        add_27 = torch.ops.aten.add.Tensor(view_61, add_25);  view_61 = add_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_27, getitem_11);  add_27 = getitem_11 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = rsqrt_5 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg50_1);  mul_17 = arg50_1 = None
        add_29 = torch.ops.aten.add.Tensor(mul_18, arg51_1);  mul_18 = arg51_1 = None
        view_62 = torch.ops.aten.view.default(add_29, [8192, 768])
        permute_31 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg53_1, view_62, permute_31);  arg53_1 = view_62 = permute_31 = None
        view_63 = torch.ops.aten.view.default(addmm_16, [16, 512, 3072]);  addmm_16 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_20 = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2 = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_30 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_19, add_30);  mul_19 = add_30 = None
        view_64 = torch.ops.aten.view.default(mul_21, [8192, 3072]);  mul_21 = None
        permute_32 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg55_1, view_64, permute_32);  arg55_1 = view_64 = permute_32 = None
        view_65 = torch.ops.aten.view.default(addmm_17, [16, 512, 768]);  addmm_17 = None
        add_31 = torch.ops.aten.add.Tensor(view_65, add_29);  view_65 = add_29 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_31, getitem_13);  add_31 = getitem_13 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg56_1);  mul_22 = arg56_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_23, arg57_1);  mul_23 = arg57_1 = None
        view_66 = torch.ops.aten.view.default(add_33, [8192, 768])
        permute_33 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg59_1, view_66, permute_33);  arg59_1 = view_66 = permute_33 = None
        view_67 = torch.ops.aten.view.default(addmm_18, [16, 512, 768]);  addmm_18 = None
        view_68 = torch.ops.aten.view.default(add_33, [8192, 768])
        permute_34 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg61_1, view_68, permute_34);  arg61_1 = view_68 = permute_34 = None
        view_69 = torch.ops.aten.view.default(addmm_19, [16, 512, 768]);  addmm_19 = None
        view_70 = torch.ops.aten.view.default(view_69, [16, 512, 12, 64]);  view_69 = None
        view_71 = torch.ops.aten.view.default(add_33, [8192, 768])
        permute_36 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg63_1, view_71, permute_36);  arg63_1 = view_71 = permute_36 = None
        view_72 = torch.ops.aten.view.default(addmm_20, [16, 512, 768]);  addmm_20 = None
        view_73 = torch.ops.aten.view.default(view_72, [16, 512, 12, 64]);  view_72 = None
        view_74 = torch.ops.aten.view.default(view_67, [16, 512, 12, 64]);  view_67 = None
        permute_default_24 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_default_25 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        permute_default_26 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_58 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        permute_40 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_26 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81 = torch.ops.aten.view.default(clone_26, [16, 512, 768]);  clone_26 = None
        view_82 = torch.ops.aten.view.default(view_81, [8192, 768]);  view_81 = None
        permute_41 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg65_1, view_82, permute_41);  arg65_1 = view_82 = permute_41 = None
        view_83 = torch.ops.aten.view.default(addmm_21, [16, 512, 768]);  addmm_21 = None
        add_35 = torch.ops.aten.add.Tensor(view_83, add_33);  view_83 = add_33 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_35, getitem_15);  add_35 = getitem_15 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = rsqrt_7 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg66_1);  mul_24 = arg66_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_25, arg67_1);  mul_25 = arg67_1 = None
        view_84 = torch.ops.aten.view.default(add_37, [8192, 768])
        permute_42 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg69_1, view_84, permute_42);  arg69_1 = view_84 = permute_42 = None
        view_85 = torch.ops.aten.view.default(addmm_22, [16, 512, 3072]);  addmm_22 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_27 = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3 = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_38 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_26, add_38);  mul_26 = add_38 = None
        view_86 = torch.ops.aten.view.default(mul_28, [8192, 3072]);  mul_28 = None
        permute_43 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg71_1, view_86, permute_43);  arg71_1 = view_86 = permute_43 = None
        view_87 = torch.ops.aten.view.default(addmm_23, [16, 512, 768]);  addmm_23 = None
        add_39 = torch.ops.aten.add.Tensor(view_87, add_37);  view_87 = add_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_39, getitem_17);  add_39 = getitem_17 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_8);  sub_15 = rsqrt_8 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, arg72_1);  mul_29 = arg72_1 = None
        add_41 = torch.ops.aten.add.Tensor(mul_30, arg73_1);  mul_30 = arg73_1 = None
        view_88 = torch.ops.aten.view.default(add_41, [8192, 768])
        permute_44 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg75_1, view_88, permute_44);  arg75_1 = view_88 = permute_44 = None
        view_89 = torch.ops.aten.view.default(addmm_24, [16, 512, 768]);  addmm_24 = None
        view_90 = torch.ops.aten.view.default(add_41, [8192, 768])
        permute_45 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg77_1, view_90, permute_45);  arg77_1 = view_90 = permute_45 = None
        view_91 = torch.ops.aten.view.default(addmm_25, [16, 512, 768]);  addmm_25 = None
        view_92 = torch.ops.aten.view.default(view_91, [16, 512, 12, 64]);  view_91 = None
        view_93 = torch.ops.aten.view.default(add_41, [8192, 768])
        permute_47 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg79_1, view_93, permute_47);  arg79_1 = view_93 = permute_47 = None
        view_94 = torch.ops.aten.view.default(addmm_26, [16, 512, 768]);  addmm_26 = None
        view_95 = torch.ops.aten.view.default(view_94, [16, 512, 12, 64]);  view_94 = None
        view_96 = torch.ops.aten.view.default(view_89, [16, 512, 12, 64]);  view_89 = None
        permute_default_21 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        permute_default_22 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        permute_default_23 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_57 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        permute_51 = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_33 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103 = torch.ops.aten.view.default(clone_33, [16, 512, 768]);  clone_33 = None
        view_104 = torch.ops.aten.view.default(view_103, [8192, 768]);  view_103 = None
        permute_52 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg81_1, view_104, permute_52);  arg81_1 = view_104 = permute_52 = None
        view_105 = torch.ops.aten.view.default(addmm_27, [16, 512, 768]);  addmm_27 = None
        add_43 = torch.ops.aten.add.Tensor(view_105, add_41);  view_105 = add_41 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_43, getitem_19);  add_43 = getitem_19 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_9);  sub_17 = rsqrt_9 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, arg82_1);  mul_31 = arg82_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_32, arg83_1);  mul_32 = arg83_1 = None
        view_106 = torch.ops.aten.view.default(add_45, [8192, 768])
        permute_53 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg85_1, view_106, permute_53);  arg85_1 = view_106 = permute_53 = None
        view_107 = torch.ops.aten.view.default(addmm_28, [16, 512, 3072]);  addmm_28 = None
        mul_33 = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_34 = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4 = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_46 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_33, add_46);  mul_33 = add_46 = None
        view_108 = torch.ops.aten.view.default(mul_35, [8192, 3072]);  mul_35 = None
        permute_54 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg87_1, view_108, permute_54);  arg87_1 = view_108 = permute_54 = None
        view_109 = torch.ops.aten.view.default(addmm_29, [16, 512, 768]);  addmm_29 = None
        add_47 = torch.ops.aten.add.Tensor(view_109, add_45);  view_109 = add_45 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_47, getitem_21);  add_47 = getitem_21 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_10);  sub_18 = rsqrt_10 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg88_1);  mul_36 = arg88_1 = None
        add_49 = torch.ops.aten.add.Tensor(mul_37, arg89_1);  mul_37 = arg89_1 = None
        view_110 = torch.ops.aten.view.default(add_49, [8192, 768])
        permute_55 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg91_1, view_110, permute_55);  arg91_1 = view_110 = permute_55 = None
        view_111 = torch.ops.aten.view.default(addmm_30, [16, 512, 768]);  addmm_30 = None
        view_112 = torch.ops.aten.view.default(add_49, [8192, 768])
        permute_56 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg93_1, view_112, permute_56);  arg93_1 = view_112 = permute_56 = None
        view_113 = torch.ops.aten.view.default(addmm_31, [16, 512, 768]);  addmm_31 = None
        view_114 = torch.ops.aten.view.default(view_113, [16, 512, 12, 64]);  view_113 = None
        view_115 = torch.ops.aten.view.default(add_49, [8192, 768])
        permute_58 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg95_1, view_115, permute_58);  arg95_1 = view_115 = permute_58 = None
        view_116 = torch.ops.aten.view.default(addmm_32, [16, 512, 768]);  addmm_32 = None
        view_117 = torch.ops.aten.view.default(view_116, [16, 512, 12, 64]);  view_116 = None
        view_118 = torch.ops.aten.view.default(view_111, [16, 512, 12, 64]);  view_111 = None
        permute_default_18 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        permute_default_19 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_default_20 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_56 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        permute_62 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_40 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125 = torch.ops.aten.view.default(clone_40, [16, 512, 768]);  clone_40 = None
        view_126 = torch.ops.aten.view.default(view_125, [8192, 768]);  view_125 = None
        permute_63 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg97_1, view_126, permute_63);  arg97_1 = view_126 = permute_63 = None
        view_127 = torch.ops.aten.view.default(addmm_33, [16, 512, 768]);  addmm_33 = None
        add_51 = torch.ops.aten.add.Tensor(view_127, add_49);  view_127 = add_49 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_51, getitem_23);  add_51 = getitem_23 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_11);  sub_20 = rsqrt_11 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg98_1);  mul_38 = arg98_1 = None
        add_53 = torch.ops.aten.add.Tensor(mul_39, arg99_1);  mul_39 = arg99_1 = None
        view_128 = torch.ops.aten.view.default(add_53, [8192, 768])
        permute_64 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg101_1, view_128, permute_64);  arg101_1 = view_128 = permute_64 = None
        view_129 = torch.ops.aten.view.default(addmm_34, [16, 512, 3072]);  addmm_34 = None
        mul_40 = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_54 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_54);  mul_40 = add_54 = None
        view_130 = torch.ops.aten.view.default(mul_42, [8192, 3072]);  mul_42 = None
        permute_65 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg103_1, view_130, permute_65);  arg103_1 = view_130 = permute_65 = None
        view_131 = torch.ops.aten.view.default(addmm_35, [16, 512, 768]);  addmm_35 = None
        add_55 = torch.ops.aten.add.Tensor(view_131, add_53);  view_131 = add_53 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_12);  sub_21 = rsqrt_12 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, arg104_1);  mul_43 = arg104_1 = None
        add_57 = torch.ops.aten.add.Tensor(mul_44, arg105_1);  mul_44 = arg105_1 = None
        view_132 = torch.ops.aten.view.default(add_57, [8192, 768])
        permute_66 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg107_1, view_132, permute_66);  arg107_1 = view_132 = permute_66 = None
        view_133 = torch.ops.aten.view.default(addmm_36, [16, 512, 768]);  addmm_36 = None
        view_134 = torch.ops.aten.view.default(add_57, [8192, 768])
        permute_67 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg109_1, view_134, permute_67);  arg109_1 = view_134 = permute_67 = None
        view_135 = torch.ops.aten.view.default(addmm_37, [16, 512, 768]);  addmm_37 = None
        view_136 = torch.ops.aten.view.default(view_135, [16, 512, 12, 64]);  view_135 = None
        view_137 = torch.ops.aten.view.default(add_57, [8192, 768])
        permute_69 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg111_1, view_137, permute_69);  arg111_1 = view_137 = permute_69 = None
        view_138 = torch.ops.aten.view.default(addmm_38, [16, 512, 768]);  addmm_38 = None
        view_139 = torch.ops.aten.view.default(view_138, [16, 512, 12, 64]);  view_138 = None
        view_140 = torch.ops.aten.view.default(view_133, [16, 512, 12, 64]);  view_133 = None
        permute_default_15 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        permute_default_16 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_default_17 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_55 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        permute_73 = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        clone_47 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147 = torch.ops.aten.view.default(clone_47, [16, 512, 768]);  clone_47 = None
        view_148 = torch.ops.aten.view.default(view_147, [8192, 768]);  view_147 = None
        permute_74 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg113_1, view_148, permute_74);  arg113_1 = view_148 = permute_74 = None
        view_149 = torch.ops.aten.view.default(addmm_39, [16, 512, 768]);  addmm_39 = None
        add_59 = torch.ops.aten.add.Tensor(view_149, add_57);  view_149 = add_57 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = rsqrt_13 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, arg114_1);  mul_45 = arg114_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_46, arg115_1);  mul_46 = arg115_1 = None
        view_150 = torch.ops.aten.view.default(add_61, [8192, 768])
        permute_75 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg117_1, view_150, permute_75);  arg117_1 = view_150 = permute_75 = None
        view_151 = torch.ops.aten.view.default(addmm_40, [16, 512, 3072]);  addmm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_48 = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6 = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_62 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, add_62);  mul_47 = add_62 = None
        view_152 = torch.ops.aten.view.default(mul_49, [8192, 3072]);  mul_49 = None
        permute_76 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg119_1, view_152, permute_76);  arg119_1 = view_152 = permute_76 = None
        view_153 = torch.ops.aten.view.default(addmm_41, [16, 512, 768]);  addmm_41 = None
        add_63 = torch.ops.aten.add.Tensor(view_153, add_61);  view_153 = add_61 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_63, getitem_29);  add_63 = getitem_29 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_14);  sub_24 = rsqrt_14 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg120_1);  mul_50 = arg120_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_51, arg121_1);  mul_51 = arg121_1 = None
        view_154 = torch.ops.aten.view.default(add_65, [8192, 768])
        permute_77 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg123_1, view_154, permute_77);  arg123_1 = view_154 = permute_77 = None
        view_155 = torch.ops.aten.view.default(addmm_42, [16, 512, 768]);  addmm_42 = None
        view_156 = torch.ops.aten.view.default(add_65, [8192, 768])
        permute_78 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg125_1, view_156, permute_78);  arg125_1 = view_156 = permute_78 = None
        view_157 = torch.ops.aten.view.default(addmm_43, [16, 512, 768]);  addmm_43 = None
        view_158 = torch.ops.aten.view.default(view_157, [16, 512, 12, 64]);  view_157 = None
        view_159 = torch.ops.aten.view.default(add_65, [8192, 768])
        permute_80 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg127_1, view_159, permute_80);  arg127_1 = view_159 = permute_80 = None
        view_160 = torch.ops.aten.view.default(addmm_44, [16, 512, 768]);  addmm_44 = None
        view_161 = torch.ops.aten.view.default(view_160, [16, 512, 12, 64]);  view_160 = None
        view_162 = torch.ops.aten.view.default(view_155, [16, 512, 12, 64]);  view_155 = None
        permute_default_12 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_default_13 = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        permute_default_14 = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_54 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        permute_84 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_54 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169 = torch.ops.aten.view.default(clone_54, [16, 512, 768]);  clone_54 = None
        view_170 = torch.ops.aten.view.default(view_169, [8192, 768]);  view_169 = None
        permute_85 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg129_1, view_170, permute_85);  arg129_1 = view_170 = permute_85 = None
        view_171 = torch.ops.aten.view.default(addmm_45, [16, 512, 768]);  addmm_45 = None
        add_67 = torch.ops.aten.add.Tensor(view_171, add_65);  view_171 = add_65 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_67, getitem_31);  add_67 = getitem_31 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = rsqrt_15 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg130_1);  mul_52 = arg130_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_53, arg131_1);  mul_53 = arg131_1 = None
        view_172 = torch.ops.aten.view.default(add_69, [8192, 768])
        permute_86 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg133_1, view_172, permute_86);  arg133_1 = view_172 = permute_86 = None
        view_173 = torch.ops.aten.view.default(addmm_46, [16, 512, 3072]);  addmm_46 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_70 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_70);  mul_54 = add_70 = None
        view_174 = torch.ops.aten.view.default(mul_56, [8192, 3072]);  mul_56 = None
        permute_87 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg135_1, view_174, permute_87);  arg135_1 = view_174 = permute_87 = None
        view_175 = torch.ops.aten.view.default(addmm_47, [16, 512, 768]);  addmm_47 = None
        add_71 = torch.ops.aten.add.Tensor(view_175, add_69);  view_175 = add_69 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_71, getitem_33);  add_71 = getitem_33 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_16);  sub_27 = rsqrt_16 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg136_1);  mul_57 = arg136_1 = None
        add_73 = torch.ops.aten.add.Tensor(mul_58, arg137_1);  mul_58 = arg137_1 = None
        view_176 = torch.ops.aten.view.default(add_73, [8192, 768])
        permute_88 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg139_1, view_176, permute_88);  arg139_1 = view_176 = permute_88 = None
        view_177 = torch.ops.aten.view.default(addmm_48, [16, 512, 768]);  addmm_48 = None
        view_178 = torch.ops.aten.view.default(add_73, [8192, 768])
        permute_89 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg141_1, view_178, permute_89);  arg141_1 = view_178 = permute_89 = None
        view_179 = torch.ops.aten.view.default(addmm_49, [16, 512, 768]);  addmm_49 = None
        view_180 = torch.ops.aten.view.default(view_179, [16, 512, 12, 64]);  view_179 = None
        view_181 = torch.ops.aten.view.default(add_73, [8192, 768])
        permute_91 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg143_1, view_181, permute_91);  arg143_1 = view_181 = permute_91 = None
        view_182 = torch.ops.aten.view.default(addmm_50, [16, 512, 768]);  addmm_50 = None
        view_183 = torch.ops.aten.view.default(view_182, [16, 512, 12, 64]);  view_182 = None
        view_184 = torch.ops.aten.view.default(view_177, [16, 512, 12, 64]);  view_177 = None
        permute_default_9 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        permute_default_10 = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        permute_default_11 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_53 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        permute_95 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        clone_61 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191 = torch.ops.aten.view.default(clone_61, [16, 512, 768]);  clone_61 = None
        view_192 = torch.ops.aten.view.default(view_191, [8192, 768]);  view_191 = None
        permute_96 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg145_1, view_192, permute_96);  arg145_1 = view_192 = permute_96 = None
        view_193 = torch.ops.aten.view.default(addmm_51, [16, 512, 768]);  addmm_51 = None
        add_75 = torch.ops.aten.add.Tensor(view_193, add_73);  view_193 = add_73 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_75, getitem_35);  add_75 = getitem_35 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_17);  sub_29 = rsqrt_17 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg146_1);  mul_59 = arg146_1 = None
        add_77 = torch.ops.aten.add.Tensor(mul_60, arg147_1);  mul_60 = arg147_1 = None
        view_194 = torch.ops.aten.view.default(add_77, [8192, 768])
        permute_97 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg149_1, view_194, permute_97);  arg149_1 = view_194 = permute_97 = None
        view_195 = torch.ops.aten.view.default(addmm_52, [16, 512, 3072]);  addmm_52 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_62 = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8 = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_78 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_61, add_78);  mul_61 = add_78 = None
        view_196 = torch.ops.aten.view.default(mul_63, [8192, 3072]);  mul_63 = None
        permute_98 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg151_1, view_196, permute_98);  arg151_1 = view_196 = permute_98 = None
        view_197 = torch.ops.aten.view.default(addmm_53, [16, 512, 768]);  addmm_53 = None
        add_79 = torch.ops.aten.add.Tensor(view_197, add_77);  view_197 = add_77 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_79, getitem_37);  add_79 = getitem_37 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_18);  sub_30 = rsqrt_18 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg152_1);  mul_64 = arg152_1 = None
        add_81 = torch.ops.aten.add.Tensor(mul_65, arg153_1);  mul_65 = arg153_1 = None
        view_198 = torch.ops.aten.view.default(add_81, [8192, 768])
        permute_99 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg155_1, view_198, permute_99);  arg155_1 = view_198 = permute_99 = None
        view_199 = torch.ops.aten.view.default(addmm_54, [16, 512, 768]);  addmm_54 = None
        view_200 = torch.ops.aten.view.default(add_81, [8192, 768])
        permute_100 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg157_1, view_200, permute_100);  arg157_1 = view_200 = permute_100 = None
        view_201 = torch.ops.aten.view.default(addmm_55, [16, 512, 768]);  addmm_55 = None
        view_202 = torch.ops.aten.view.default(view_201, [16, 512, 12, 64]);  view_201 = None
        view_203 = torch.ops.aten.view.default(add_81, [8192, 768])
        permute_102 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg159_1, view_203, permute_102);  arg159_1 = view_203 = permute_102 = None
        view_204 = torch.ops.aten.view.default(addmm_56, [16, 512, 768]);  addmm_56 = None
        view_205 = torch.ops.aten.view.default(view_204, [16, 512, 12, 64]);  view_204 = None
        view_206 = torch.ops.aten.view.default(view_199, [16, 512, 12, 64]);  view_199 = None
        permute_default_6 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        permute_default_7 = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        permute_default_8 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_52 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        permute_106 = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_68 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213 = torch.ops.aten.view.default(clone_68, [16, 512, 768]);  clone_68 = None
        view_214 = torch.ops.aten.view.default(view_213, [8192, 768]);  view_213 = None
        permute_107 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg161_1, view_214, permute_107);  arg161_1 = view_214 = permute_107 = None
        view_215 = torch.ops.aten.view.default(addmm_57, [16, 512, 768]);  addmm_57 = None
        add_83 = torch.ops.aten.add.Tensor(view_215, add_81);  view_215 = add_81 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_83, getitem_39);  add_83 = getitem_39 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = rsqrt_19 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg162_1);  mul_66 = arg162_1 = None
        add_85 = torch.ops.aten.add.Tensor(mul_67, arg163_1);  mul_67 = arg163_1 = None
        view_216 = torch.ops.aten.view.default(add_85, [8192, 768])
        permute_108 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg165_1, view_216, permute_108);  arg165_1 = view_216 = permute_108 = None
        view_217 = torch.ops.aten.view.default(addmm_58, [16, 512, 3072]);  addmm_58 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_69 = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9 = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_86 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_68, add_86);  mul_68 = add_86 = None
        view_218 = torch.ops.aten.view.default(mul_70, [8192, 3072]);  mul_70 = None
        permute_109 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg167_1, view_218, permute_109);  arg167_1 = view_218 = permute_109 = None
        view_219 = torch.ops.aten.view.default(addmm_59, [16, 512, 768]);  addmm_59 = None
        add_87 = torch.ops.aten.add.Tensor(view_219, add_85);  view_219 = add_85 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_87, getitem_41);  add_87 = getitem_41 = None
        mul_71 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_20);  sub_33 = rsqrt_20 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_71, arg168_1);  mul_71 = arg168_1 = None
        add_89 = torch.ops.aten.add.Tensor(mul_72, arg169_1);  mul_72 = arg169_1 = None
        view_220 = torch.ops.aten.view.default(add_89, [8192, 768])
        permute_110 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg171_1, view_220, permute_110);  arg171_1 = view_220 = permute_110 = None
        view_221 = torch.ops.aten.view.default(addmm_60, [16, 512, 768]);  addmm_60 = None
        view_222 = torch.ops.aten.view.default(add_89, [8192, 768])
        permute_111 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg173_1, view_222, permute_111);  arg173_1 = view_222 = permute_111 = None
        view_223 = torch.ops.aten.view.default(addmm_61, [16, 512, 768]);  addmm_61 = None
        view_224 = torch.ops.aten.view.default(view_223, [16, 512, 12, 64]);  view_223 = None
        view_225 = torch.ops.aten.view.default(add_89, [8192, 768])
        permute_113 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg175_1, view_225, permute_113);  arg175_1 = view_225 = permute_113 = None
        view_226 = torch.ops.aten.view.default(addmm_62, [16, 512, 768]);  addmm_62 = None
        view_227 = torch.ops.aten.view.default(view_226, [16, 512, 12, 64]);  view_226 = None
        view_228 = torch.ops.aten.view.default(view_221, [16, 512, 12, 64]);  view_221 = None
        permute_default_3 = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        permute_default_4 = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        permute_default_5 = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_51 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        permute_117 = torch.ops.aten.permute.default(getitem_51, [0, 2, 1, 3]);  getitem_51 = None
        clone_75 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235 = torch.ops.aten.view.default(clone_75, [16, 512, 768]);  clone_75 = None
        view_236 = torch.ops.aten.view.default(view_235, [8192, 768]);  view_235 = None
        permute_118 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg177_1, view_236, permute_118);  arg177_1 = view_236 = permute_118 = None
        view_237 = torch.ops.aten.view.default(addmm_63, [16, 512, 768]);  addmm_63 = None
        add_91 = torch.ops.aten.add.Tensor(view_237, add_89);  view_237 = add_89 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_91, getitem_43);  add_91 = getitem_43 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_21);  sub_35 = rsqrt_21 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg178_1);  mul_73 = arg178_1 = None
        add_93 = torch.ops.aten.add.Tensor(mul_74, arg179_1);  mul_74 = arg179_1 = None
        view_238 = torch.ops.aten.view.default(add_93, [8192, 768])
        permute_119 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg181_1, view_238, permute_119);  arg181_1 = view_238 = permute_119 = None
        view_239 = torch.ops.aten.view.default(addmm_64, [16, 512, 3072]);  addmm_64 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_76 = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10 = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_94 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_75, add_94);  mul_75 = add_94 = None
        view_240 = torch.ops.aten.view.default(mul_77, [8192, 3072]);  mul_77 = None
        permute_120 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg183_1, view_240, permute_120);  arg183_1 = view_240 = permute_120 = None
        view_241 = torch.ops.aten.view.default(addmm_65, [16, 512, 768]);  addmm_65 = None
        add_95 = torch.ops.aten.add.Tensor(view_241, add_93);  view_241 = add_93 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_95, getitem_45);  add_95 = getitem_45 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = rsqrt_22 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, arg184_1);  mul_78 = arg184_1 = None
        add_97 = torch.ops.aten.add.Tensor(mul_79, arg185_1);  mul_79 = arg185_1 = None
        view_242 = torch.ops.aten.view.default(add_97, [8192, 768])
        permute_121 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg187_1, view_242, permute_121);  arg187_1 = view_242 = permute_121 = None
        view_243 = torch.ops.aten.view.default(addmm_66, [16, 512, 768]);  addmm_66 = None
        view_244 = torch.ops.aten.view.default(add_97, [8192, 768])
        permute_122 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg189_1, view_244, permute_122);  arg189_1 = view_244 = permute_122 = None
        view_245 = torch.ops.aten.view.default(addmm_67, [16, 512, 768]);  addmm_67 = None
        view_246 = torch.ops.aten.view.default(view_245, [16, 512, 12, 64]);  view_245 = None
        view_247 = torch.ops.aten.view.default(add_97, [8192, 768])
        permute_124 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg191_1, view_247, permute_124);  arg191_1 = view_247 = permute_124 = None
        view_248 = torch.ops.aten.view.default(addmm_68, [16, 512, 768]);  addmm_68 = None
        view_249 = torch.ops.aten.view.default(view_248, [16, 512, 12, 64]);  view_248 = None
        view_250 = torch.ops.aten.view.default(view_243, [16, 512, 12, 64]);  view_243 = None
        permute_default = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        permute_default_1 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        permute_default_2 = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_50 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        permute_128 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        clone_82 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257 = torch.ops.aten.view.default(clone_82, [16, 512, 768]);  clone_82 = None
        view_258 = torch.ops.aten.view.default(view_257, [8192, 768]);  view_257 = None
        permute_129 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg193_1, view_258, permute_129);  arg193_1 = view_258 = permute_129 = None
        view_259 = torch.ops.aten.view.default(addmm_69, [16, 512, 768]);  addmm_69 = None
        add_99 = torch.ops.aten.add.Tensor(view_259, add_97);  view_259 = add_97 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_100 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_99, getitem_47);  add_99 = getitem_47 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_23);  sub_38 = rsqrt_23 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg194_1);  mul_80 = arg194_1 = None
        add_101 = torch.ops.aten.add.Tensor(mul_81, arg195_1);  mul_81 = arg195_1 = None
        view_260 = torch.ops.aten.view.default(add_101, [8192, 768])
        permute_130 = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg197_1, view_260, permute_130);  arg197_1 = view_260 = permute_130 = None
        view_261 = torch.ops.aten.view.default(addmm_70, [16, 512, 3072]);  addmm_70 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_83 = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11 = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_102 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_82, add_102);  mul_82 = add_102 = None
        view_262 = torch.ops.aten.view.default(mul_84, [8192, 3072]);  mul_84 = None
        permute_131 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg199_1, view_262, permute_131);  arg199_1 = view_262 = permute_131 = None
        view_263 = torch.ops.aten.view.default(addmm_71, [16, 512, 768]);  addmm_71 = None
        add_103 = torch.ops.aten.add.Tensor(view_263, add_101);  view_263 = add_101 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_104 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_103, getitem_49);  add_103 = getitem_49 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = rsqrt_24 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, arg200_1);  mul_85 = arg200_1 = None
        add_105 = torch.ops.aten.add.Tensor(mul_86, arg201_1);  mul_86 = arg201_1 = None
        select_8 = torch.ops.aten.select.int(add_105, 1, 0);  add_105 = None
        permute_132 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg203_1, select_8, permute_132);  arg203_1 = select_8 = permute_132 = None
        tanh = torch.ops.aten.tanh.default(addmm_72);  addmm_72 = None
        permute_133 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        full_default_5 = torch.ops.aten.full.default([768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_133, full_default_5], 1);  permute_133 = full_default_5 = None
        full_default_6 = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default_1 = torch.ops.aten.cat.default([arg206_1, full_default_6]);  arg206_1 = full_default_6 = None
        addmm_default = torch.ops.aten.addmm.default(cat_default_1, tanh, cat_default);  cat_default_1 = tanh = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_264 = torch.ops.aten.view.default(slice_tensor, [-1, 2])
        view_265 = torch.ops.aten.view.default(arg207_1, [-1]);  arg207_1 = None
        amax_12 = torch.ops.aten.amax.default(view_264, [1], True)
        sub_40 = torch.ops.aten.sub.Tensor(view_264, amax_12);  view_264 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_40)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_41 = torch.ops.aten.sub.Tensor(sub_40, log);  sub_40 = log = None
        ne = torch.ops.aten.ne.Scalar(view_265, -100)
        full_default_3 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, view_265, full_default_3);  ne = full_default_3 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_2);  sub_41 = unsqueeze_2 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_265, -100)
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_4);  ne_1 = neg = full_default_4 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_265, -100);  view_265 = None
        sum_14 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_24 = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        return (div_24, slice_tensor)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 93763584, device=device(type='cuda', index=0))
    reader.tensor(buf1, (30522, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1024, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1024, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1024, 768), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2, 768), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768, 768), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768, 768), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768, 768), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768, 768), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf20, (3072, 768), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf21, (3072,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768, 3072), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 768), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768, 768), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 768), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (3072, 768), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf37, (3072,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 3072), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 768), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768, 768), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768, 768), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768, 768), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (3072, 768), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf53, (3072,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 3072), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768, 768), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768, 768), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768, 768), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768, 768), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf68, (3072, 768), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf69, (3072,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768, 3072), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 768), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768, 768), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768, 768), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 768), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf84, (3072, 768), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf85, (3072,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 3072), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 768), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768, 768), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768, 768), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768, 768), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf100, (3072, 768), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf101, (3072,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 3072), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 768), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 768), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768, 768), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768, 768), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf116, (3072, 768), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf117, (3072,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768, 3072), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768, 768), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768, 768), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768, 768), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 768), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf132, (3072, 768), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf133, (3072,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768, 3072), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768, 768), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768, 768), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768, 768), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768, 768), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf148, (3072, 768), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf149, (3072,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768, 3072), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768, 768), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768, 768), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768, 768), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768, 768), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf164, (3072, 768), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf165, (3072,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768, 3072), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768, 768), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768, 768), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf174, (768, 768), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768, 768), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf179, (768,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf180, (3072, 768), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf181, (3072,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768, 3072), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768, 768), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768, 768), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768, 768), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf192, (768, 768), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf195, (768,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf196, (3072, 768), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf197, (3072,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768, 3072), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf199, (768,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf200, (768,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf201, (768,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf202, (768, 768), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf203, (768,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf204, (1, 512), dtype=torch.int64, is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (2, 768), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf206, (2,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf207, (16,), dtype=torch.int64, is_leaf=True)  # arg207_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)