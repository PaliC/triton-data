
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1):
        expand = torch.ops.aten.expand.default(arg1_1, [16, 512]);  arg1_1 = None
        embedding = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg5_1, arg2_1);  arg5_1 = arg2_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        full = torch.ops.aten.full.default([16, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [16, 1, 512, 512]);  unsqueeze_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(convert_element_type, scalar_tensor, sub_1);  convert_element_type = scalar_tensor = sub_1 = None
        view = torch.ops.aten.view.default(add_3, [8192, 768])
        permute = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [16, 512, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [16, 512, 12, 64]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(add_3, [8192, 768])
        permute_2 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg11_1, view_3, permute_2);  arg11_1 = view_3 = permute_2 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [16, 512, 768]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [16, 512, 12, 64]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6 = torch.ops.aten.view.default(add_3, [8192, 768])
        permute_4 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg13_1, view_6, permute_4);  arg13_1 = view_6 = permute_4 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [16, 512, 768]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [16, 512, 12, 64]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        expand_2 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_2, False);  permute_1 = permute_3 = permute_5 = expand_2 = None
        getitem_2 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_9 = torch.ops.aten.view.default(permute_6, [16, 512, 768]);  permute_6 = None
        view_10 = torch.ops.aten.view.default(view_9, [8192, 768]);  view_9 = None
        permute_7 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg15_1, view_10, permute_7);  arg15_1 = view_10 = permute_7 = None
        view_11 = torch.ops.aten.view.default(addmm_3, [16, 512, 768]);  addmm_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg16_1);  mul_2 = arg16_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg17_1);  mul_3 = arg17_1 = None
        view_12 = torch.ops.aten.view.default(add_6, [8192, 768])
        permute_8 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg19_1, view_12, permute_8);  arg19_1 = view_12 = permute_8 = None
        view_13 = torch.ops.aten.view.default(addmm_4, [16, 512, 3072]);  addmm_4 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_14 = torch.ops.aten.view.default(mul_6, [8192, 3072]);  mul_6 = None
        permute_9 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg21_1, view_14, permute_9);  arg21_1 = view_14 = permute_9 = None
        view_15 = torch.ops.aten.view.default(addmm_5, [16, 512, 768]);  addmm_5 = None
        add_8 = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, arg22_1);  mul_7 = arg22_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_8, arg23_1);  mul_8 = arg23_1 = None
        view_16 = torch.ops.aten.view.default(add_10, [8192, 768])
        permute_10 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg25_1, view_16, permute_10);  arg25_1 = view_16 = permute_10 = None
        view_17 = torch.ops.aten.view.default(addmm_6, [16, 512, 768]);  addmm_6 = None
        view_18 = torch.ops.aten.view.default(view_17, [16, 512, 12, 64]);  view_17 = None
        permute_11 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        view_19 = torch.ops.aten.view.default(add_10, [8192, 768])
        permute_12 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg27_1, view_19, permute_12);  arg27_1 = view_19 = permute_12 = None
        view_20 = torch.ops.aten.view.default(addmm_7, [16, 512, 768]);  addmm_7 = None
        view_21 = torch.ops.aten.view.default(view_20, [16, 512, 12, 64]);  view_20 = None
        permute_13 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        view_22 = torch.ops.aten.view.default(add_10, [8192, 768])
        permute_14 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg29_1, view_22, permute_14);  arg29_1 = view_22 = permute_14 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [16, 512, 768]);  addmm_8 = None
        view_24 = torch.ops.aten.view.default(view_23, [16, 512, 12, 64]);  view_23 = None
        permute_15 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        expand_3 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_3, False);  permute_11 = permute_13 = permute_15 = expand_3 = None
        getitem_10 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_25 = torch.ops.aten.view.default(permute_16, [16, 512, 768]);  permute_16 = None
        view_26 = torch.ops.aten.view.default(view_25, [8192, 768]);  view_25 = None
        permute_17 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg31_1, view_26, permute_17);  arg31_1 = view_26 = permute_17 = None
        view_27 = torch.ops.aten.view.default(addmm_9, [16, 512, 768]);  addmm_9 = None
        add_11 = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg32_1);  mul_9 = arg32_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
        view_28 = torch.ops.aten.view.default(add_13, [8192, 768])
        permute_18 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg35_1, view_28, permute_18);  arg35_1 = view_28 = permute_18 = None
        view_29 = torch.ops.aten.view.default(addmm_10, [16, 512, 3072]);  addmm_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        view_30 = torch.ops.aten.view.default(mul_13, [8192, 3072]);  mul_13 = None
        permute_19 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg37_1, view_30, permute_19);  arg37_1 = view_30 = permute_19 = None
        view_31 = torch.ops.aten.view.default(addmm_11, [16, 512, 768]);  addmm_11 = None
        add_15 = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_15, getitem_17);  add_15 = getitem_17 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg38_1);  mul_14 = arg38_1 = None
        add_17 = torch.ops.aten.add.Tensor(mul_15, arg39_1);  mul_15 = arg39_1 = None
        view_32 = torch.ops.aten.view.default(add_17, [8192, 768])
        permute_20 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg41_1, view_32, permute_20);  arg41_1 = view_32 = permute_20 = None
        view_33 = torch.ops.aten.view.default(addmm_12, [16, 512, 768]);  addmm_12 = None
        view_34 = torch.ops.aten.view.default(view_33, [16, 512, 12, 64]);  view_33 = None
        permute_21 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        view_35 = torch.ops.aten.view.default(add_17, [8192, 768])
        permute_22 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg43_1, view_35, permute_22);  arg43_1 = view_35 = permute_22 = None
        view_36 = torch.ops.aten.view.default(addmm_13, [16, 512, 768]);  addmm_13 = None
        view_37 = torch.ops.aten.view.default(view_36, [16, 512, 12, 64]);  view_36 = None
        permute_23 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        view_38 = torch.ops.aten.view.default(add_17, [8192, 768])
        permute_24 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg45_1, view_38, permute_24);  arg45_1 = view_38 = permute_24 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [16, 512, 768]);  addmm_14 = None
        view_40 = torch.ops.aten.view.default(view_39, [16, 512, 12, 64]);  view_39 = None
        permute_25 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        expand_4 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_4, False);  permute_21 = permute_23 = permute_25 = expand_4 = None
        getitem_18 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_41 = torch.ops.aten.view.default(permute_26, [16, 512, 768]);  permute_26 = None
        view_42 = torch.ops.aten.view.default(view_41, [8192, 768]);  view_41 = None
        permute_27 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg47_1, view_42, permute_27);  arg47_1 = view_42 = permute_27 = None
        view_43 = torch.ops.aten.view.default(addmm_15, [16, 512, 768]);  addmm_15 = None
        add_18 = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_18, getitem_23);  add_18 = getitem_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg48_1);  mul_16 = arg48_1 = None
        add_20 = torch.ops.aten.add.Tensor(mul_17, arg49_1);  mul_17 = arg49_1 = None
        view_44 = torch.ops.aten.view.default(add_20, [8192, 768])
        permute_28 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg51_1, view_44, permute_28);  arg51_1 = view_44 = permute_28 = None
        view_45 = torch.ops.aten.view.default(addmm_16, [16, 512, 3072]);  addmm_16 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        view_46 = torch.ops.aten.view.default(mul_20, [8192, 3072]);  mul_20 = None
        permute_29 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg53_1, view_46, permute_29);  arg53_1 = view_46 = permute_29 = None
        view_47 = torch.ops.aten.view.default(addmm_17, [16, 512, 768]);  addmm_17 = None
        add_22 = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_22, getitem_25);  add_22 = getitem_25 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg54_1);  mul_21 = arg54_1 = None
        add_24 = torch.ops.aten.add.Tensor(mul_22, arg55_1);  mul_22 = arg55_1 = None
        view_48 = torch.ops.aten.view.default(add_24, [8192, 768])
        permute_30 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg57_1, view_48, permute_30);  arg57_1 = view_48 = permute_30 = None
        view_49 = torch.ops.aten.view.default(addmm_18, [16, 512, 768]);  addmm_18 = None
        view_50 = torch.ops.aten.view.default(view_49, [16, 512, 12, 64]);  view_49 = None
        permute_31 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        view_51 = torch.ops.aten.view.default(add_24, [8192, 768])
        permute_32 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg59_1, view_51, permute_32);  arg59_1 = view_51 = permute_32 = None
        view_52 = torch.ops.aten.view.default(addmm_19, [16, 512, 768]);  addmm_19 = None
        view_53 = torch.ops.aten.view.default(view_52, [16, 512, 12, 64]);  view_52 = None
        permute_33 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        view_54 = torch.ops.aten.view.default(add_24, [8192, 768])
        permute_34 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg61_1, view_54, permute_34);  arg61_1 = view_54 = permute_34 = None
        view_55 = torch.ops.aten.view.default(addmm_20, [16, 512, 768]);  addmm_20 = None
        view_56 = torch.ops.aten.view.default(view_55, [16, 512, 12, 64]);  view_55 = None
        permute_35 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_5 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_5, False);  permute_31 = permute_33 = permute_35 = expand_5 = None
        getitem_26 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_57 = torch.ops.aten.view.default(permute_36, [16, 512, 768]);  permute_36 = None
        view_58 = torch.ops.aten.view.default(view_57, [8192, 768]);  view_57 = None
        permute_37 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg63_1, view_58, permute_37);  arg63_1 = view_58 = permute_37 = None
        view_59 = torch.ops.aten.view.default(addmm_21, [16, 512, 768]);  addmm_21 = None
        add_25 = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_25, getitem_31);  add_25 = getitem_31 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, arg64_1);  mul_23 = arg64_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_24, arg65_1);  mul_24 = arg65_1 = None
        view_60 = torch.ops.aten.view.default(add_27, [8192, 768])
        permute_38 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg67_1, view_60, permute_38);  arg67_1 = view_60 = permute_38 = None
        view_61 = torch.ops.aten.view.default(addmm_22, [16, 512, 3072]);  addmm_22 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26 = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        view_62 = torch.ops.aten.view.default(mul_27, [8192, 3072]);  mul_27 = None
        permute_39 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg69_1, view_62, permute_39);  arg69_1 = view_62 = permute_39 = None
        view_63 = torch.ops.aten.view.default(addmm_23, [16, 512, 768]);  addmm_23 = None
        add_29 = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_29, getitem_33);  add_29 = getitem_33 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg70_1);  mul_28 = arg70_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_29, arg71_1);  mul_29 = arg71_1 = None
        view_64 = torch.ops.aten.view.default(add_31, [8192, 768])
        permute_40 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg73_1, view_64, permute_40);  arg73_1 = view_64 = permute_40 = None
        view_65 = torch.ops.aten.view.default(addmm_24, [16, 512, 768]);  addmm_24 = None
        view_66 = torch.ops.aten.view.default(view_65, [16, 512, 12, 64]);  view_65 = None
        permute_41 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        view_67 = torch.ops.aten.view.default(add_31, [8192, 768])
        permute_42 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg75_1, view_67, permute_42);  arg75_1 = view_67 = permute_42 = None
        view_68 = torch.ops.aten.view.default(addmm_25, [16, 512, 768]);  addmm_25 = None
        view_69 = torch.ops.aten.view.default(view_68, [16, 512, 12, 64]);  view_68 = None
        permute_43 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        view_70 = torch.ops.aten.view.default(add_31, [8192, 768])
        permute_44 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg77_1, view_70, permute_44);  arg77_1 = view_70 = permute_44 = None
        view_71 = torch.ops.aten.view.default(addmm_26, [16, 512, 768]);  addmm_26 = None
        view_72 = torch.ops.aten.view.default(view_71, [16, 512, 12, 64]);  view_71 = None
        permute_45 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        expand_6 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_6, False);  permute_41 = permute_43 = permute_45 = expand_6 = None
        getitem_34 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_73 = torch.ops.aten.view.default(permute_46, [16, 512, 768]);  permute_46 = None
        view_74 = torch.ops.aten.view.default(view_73, [8192, 768]);  view_73 = None
        permute_47 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg79_1, view_74, permute_47);  arg79_1 = view_74 = permute_47 = None
        view_75 = torch.ops.aten.view.default(addmm_27, [16, 512, 768]);  addmm_27 = None
        add_32 = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_32, getitem_39);  add_32 = getitem_39 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg80_1);  mul_30 = arg80_1 = None
        add_34 = torch.ops.aten.add.Tensor(mul_31, arg81_1);  mul_31 = arg81_1 = None
        view_76 = torch.ops.aten.view.default(add_34, [8192, 768])
        permute_48 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg83_1, view_76, permute_48);  arg83_1 = view_76 = permute_48 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [16, 512, 3072]);  addmm_28 = None
        mul_32 = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33 = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        view_78 = torch.ops.aten.view.default(mul_34, [8192, 3072]);  mul_34 = None
        permute_49 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg85_1, view_78, permute_49);  arg85_1 = view_78 = permute_49 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [16, 512, 768]);  addmm_29 = None
        add_36 = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_36, getitem_41);  add_36 = getitem_41 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, arg86_1);  mul_35 = arg86_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_36, arg87_1);  mul_36 = arg87_1 = None
        view_80 = torch.ops.aten.view.default(add_38, [8192, 768])
        permute_50 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg89_1, view_80, permute_50);  arg89_1 = view_80 = permute_50 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [16, 512, 768]);  addmm_30 = None
        view_82 = torch.ops.aten.view.default(view_81, [16, 512, 12, 64]);  view_81 = None
        permute_51 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        view_83 = torch.ops.aten.view.default(add_38, [8192, 768])
        permute_52 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg91_1, view_83, permute_52);  arg91_1 = view_83 = permute_52 = None
        view_84 = torch.ops.aten.view.default(addmm_31, [16, 512, 768]);  addmm_31 = None
        view_85 = torch.ops.aten.view.default(view_84, [16, 512, 12, 64]);  view_84 = None
        permute_53 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        view_86 = torch.ops.aten.view.default(add_38, [8192, 768])
        permute_54 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg93_1, view_86, permute_54);  arg93_1 = view_86 = permute_54 = None
        view_87 = torch.ops.aten.view.default(addmm_32, [16, 512, 768]);  addmm_32 = None
        view_88 = torch.ops.aten.view.default(view_87, [16, 512, 12, 64]);  view_87 = None
        permute_55 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        expand_7 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_7, False);  permute_51 = permute_53 = permute_55 = expand_7 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_89 = torch.ops.aten.view.default(permute_56, [16, 512, 768]);  permute_56 = None
        view_90 = torch.ops.aten.view.default(view_89, [8192, 768]);  view_89 = None
        permute_57 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg95_1, view_90, permute_57);  arg95_1 = view_90 = permute_57 = None
        view_91 = torch.ops.aten.view.default(addmm_33, [16, 512, 768]);  addmm_33 = None
        add_39 = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_39, getitem_47);  add_39 = getitem_47 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, arg96_1);  mul_37 = arg96_1 = None
        add_41 = torch.ops.aten.add.Tensor(mul_38, arg97_1);  mul_38 = arg97_1 = None
        view_92 = torch.ops.aten.view.default(add_41, [8192, 768])
        permute_58 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg99_1, view_92, permute_58);  arg99_1 = view_92 = permute_58 = None
        view_93 = torch.ops.aten.view.default(addmm_34, [16, 512, 3072]);  addmm_34 = None
        mul_39 = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        view_94 = torch.ops.aten.view.default(mul_41, [8192, 3072]);  mul_41 = None
        permute_59 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg101_1, view_94, permute_59);  arg101_1 = view_94 = permute_59 = None
        view_95 = torch.ops.aten.view.default(addmm_35, [16, 512, 768]);  addmm_35 = None
        add_43 = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_43, getitem_49);  add_43 = getitem_49 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg102_1);  mul_42 = arg102_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_43, arg103_1);  mul_43 = arg103_1 = None
        view_96 = torch.ops.aten.view.default(add_45, [8192, 768])
        permute_60 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg105_1, view_96, permute_60);  arg105_1 = view_96 = permute_60 = None
        view_97 = torch.ops.aten.view.default(addmm_36, [16, 512, 768]);  addmm_36 = None
        view_98 = torch.ops.aten.view.default(view_97, [16, 512, 12, 64]);  view_97 = None
        permute_61 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        view_99 = torch.ops.aten.view.default(add_45, [8192, 768])
        permute_62 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg107_1, view_99, permute_62);  arg107_1 = view_99 = permute_62 = None
        view_100 = torch.ops.aten.view.default(addmm_37, [16, 512, 768]);  addmm_37 = None
        view_101 = torch.ops.aten.view.default(view_100, [16, 512, 12, 64]);  view_100 = None
        permute_63 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        view_102 = torch.ops.aten.view.default(add_45, [8192, 768])
        permute_64 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg109_1, view_102, permute_64);  arg109_1 = view_102 = permute_64 = None
        view_103 = torch.ops.aten.view.default(addmm_38, [16, 512, 768]);  addmm_38 = None
        view_104 = torch.ops.aten.view.default(view_103, [16, 512, 12, 64]);  view_103 = None
        permute_65 = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        expand_8 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_8, False);  permute_61 = permute_63 = permute_65 = expand_8 = None
        getitem_50 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_66 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        view_105 = torch.ops.aten.view.default(permute_66, [16, 512, 768]);  permute_66 = None
        view_106 = torch.ops.aten.view.default(view_105, [8192, 768]);  view_105 = None
        permute_67 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg111_1, view_106, permute_67);  arg111_1 = view_106 = permute_67 = None
        view_107 = torch.ops.aten.view.default(addmm_39, [16, 512, 768]);  addmm_39 = None
        add_46 = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_13[0]
        getitem_55 = var_mean_13[1];  var_mean_13 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_46, getitem_55);  add_46 = getitem_55 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg112_1);  mul_44 = arg112_1 = None
        add_48 = torch.ops.aten.add.Tensor(mul_45, arg113_1);  mul_45 = arg113_1 = None
        view_108 = torch.ops.aten.view.default(add_48, [8192, 768])
        permute_68 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg115_1, view_108, permute_68);  arg115_1 = view_108 = permute_68 = None
        view_109 = torch.ops.aten.view.default(addmm_40, [16, 512, 3072]);  addmm_40 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        view_110 = torch.ops.aten.view.default(mul_48, [8192, 3072]);  mul_48 = None
        permute_69 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg117_1, view_110, permute_69);  arg117_1 = view_110 = permute_69 = None
        view_111 = torch.ops.aten.view.default(addmm_41, [16, 512, 768]);  addmm_41 = None
        add_50 = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_50, getitem_57);  add_50 = getitem_57 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg118_1);  mul_49 = arg118_1 = None
        add_52 = torch.ops.aten.add.Tensor(mul_50, arg119_1);  mul_50 = arg119_1 = None
        view_112 = torch.ops.aten.view.default(add_52, [8192, 768])
        permute_70 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg121_1, view_112, permute_70);  arg121_1 = view_112 = permute_70 = None
        view_113 = torch.ops.aten.view.default(addmm_42, [16, 512, 768]);  addmm_42 = None
        view_114 = torch.ops.aten.view.default(view_113, [16, 512, 12, 64]);  view_113 = None
        permute_71 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        view_115 = torch.ops.aten.view.default(add_52, [8192, 768])
        permute_72 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg123_1, view_115, permute_72);  arg123_1 = view_115 = permute_72 = None
        view_116 = torch.ops.aten.view.default(addmm_43, [16, 512, 768]);  addmm_43 = None
        view_117 = torch.ops.aten.view.default(view_116, [16, 512, 12, 64]);  view_116 = None
        permute_73 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        view_118 = torch.ops.aten.view.default(add_52, [8192, 768])
        permute_74 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg125_1, view_118, permute_74);  arg125_1 = view_118 = permute_74 = None
        view_119 = torch.ops.aten.view.default(addmm_44, [16, 512, 768]);  addmm_44 = None
        view_120 = torch.ops.aten.view.default(view_119, [16, 512, 12, 64]);  view_119 = None
        permute_75 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        expand_9 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_9, False);  permute_71 = permute_73 = permute_75 = expand_9 = None
        getitem_58 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_76 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        view_121 = torch.ops.aten.view.default(permute_76, [16, 512, 768]);  permute_76 = None
        view_122 = torch.ops.aten.view.default(view_121, [8192, 768]);  view_121 = None
        permute_77 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg127_1, view_122, permute_77);  arg127_1 = view_122 = permute_77 = None
        view_123 = torch.ops.aten.view.default(addmm_45, [16, 512, 768]);  addmm_45 = None
        add_53 = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_15[0]
        getitem_63 = var_mean_15[1];  var_mean_15 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_53, getitem_63);  add_53 = getitem_63 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, arg128_1);  mul_51 = arg128_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_52, arg129_1);  mul_52 = arg129_1 = None
        view_124 = torch.ops.aten.view.default(add_55, [8192, 768])
        permute_78 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg131_1, view_124, permute_78);  arg131_1 = view_124 = permute_78 = None
        view_125 = torch.ops.aten.view.default(addmm_46, [16, 512, 3072]);  addmm_46 = None
        mul_53 = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54 = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        view_126 = torch.ops.aten.view.default(mul_55, [8192, 3072]);  mul_55 = None
        permute_79 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg133_1, view_126, permute_79);  arg133_1 = view_126 = permute_79 = None
        view_127 = torch.ops.aten.view.default(addmm_47, [16, 512, 768]);  addmm_47 = None
        add_57 = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_57, getitem_65);  add_57 = getitem_65 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg134_1);  mul_56 = arg134_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg135_1);  mul_57 = arg135_1 = None
        view_128 = torch.ops.aten.view.default(add_59, [8192, 768])
        permute_80 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg137_1, view_128, permute_80);  arg137_1 = view_128 = permute_80 = None
        view_129 = torch.ops.aten.view.default(addmm_48, [16, 512, 768]);  addmm_48 = None
        view_130 = torch.ops.aten.view.default(view_129, [16, 512, 12, 64]);  view_129 = None
        permute_81 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        view_131 = torch.ops.aten.view.default(add_59, [8192, 768])
        permute_82 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg139_1, view_131, permute_82);  arg139_1 = view_131 = permute_82 = None
        view_132 = torch.ops.aten.view.default(addmm_49, [16, 512, 768]);  addmm_49 = None
        view_133 = torch.ops.aten.view.default(view_132, [16, 512, 12, 64]);  view_132 = None
        permute_83 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        view_134 = torch.ops.aten.view.default(add_59, [8192, 768])
        permute_84 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg141_1, view_134, permute_84);  arg141_1 = view_134 = permute_84 = None
        view_135 = torch.ops.aten.view.default(addmm_50, [16, 512, 768]);  addmm_50 = None
        view_136 = torch.ops.aten.view.default(view_135, [16, 512, 12, 64]);  view_135 = None
        permute_85 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        expand_10 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_10, False);  permute_81 = permute_83 = permute_85 = expand_10 = None
        getitem_66 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_86 = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        view_137 = torch.ops.aten.view.default(permute_86, [16, 512, 768]);  permute_86 = None
        view_138 = torch.ops.aten.view.default(view_137, [8192, 768]);  view_137 = None
        permute_87 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg143_1, view_138, permute_87);  arg143_1 = view_138 = permute_87 = None
        view_139 = torch.ops.aten.view.default(addmm_51, [16, 512, 768]);  addmm_51 = None
        add_60 = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_17[0]
        getitem_71 = var_mean_17[1];  var_mean_17 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_60, getitem_71);  add_60 = getitem_71 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg144_1);  mul_58 = arg144_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_59, arg145_1);  mul_59 = arg145_1 = None
        view_140 = torch.ops.aten.view.default(add_62, [8192, 768])
        permute_88 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg147_1, view_140, permute_88);  arg147_1 = view_140 = permute_88 = None
        view_141 = torch.ops.aten.view.default(addmm_52, [16, 512, 3072]);  addmm_52 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61 = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8 = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        view_142 = torch.ops.aten.view.default(mul_62, [8192, 3072]);  mul_62 = None
        permute_89 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg149_1, view_142, permute_89);  arg149_1 = view_142 = permute_89 = None
        view_143 = torch.ops.aten.view.default(addmm_53, [16, 512, 768]);  addmm_53 = None
        add_64 = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_64, getitem_73);  add_64 = getitem_73 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, arg150_1);  mul_63 = arg150_1 = None
        add_66 = torch.ops.aten.add.Tensor(mul_64, arg151_1);  mul_64 = arg151_1 = None
        view_144 = torch.ops.aten.view.default(add_66, [8192, 768])
        permute_90 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg153_1, view_144, permute_90);  arg153_1 = view_144 = permute_90 = None
        view_145 = torch.ops.aten.view.default(addmm_54, [16, 512, 768]);  addmm_54 = None
        view_146 = torch.ops.aten.view.default(view_145, [16, 512, 12, 64]);  view_145 = None
        permute_91 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        view_147 = torch.ops.aten.view.default(add_66, [8192, 768])
        permute_92 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg155_1, view_147, permute_92);  arg155_1 = view_147 = permute_92 = None
        view_148 = torch.ops.aten.view.default(addmm_55, [16, 512, 768]);  addmm_55 = None
        view_149 = torch.ops.aten.view.default(view_148, [16, 512, 12, 64]);  view_148 = None
        permute_93 = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        view_150 = torch.ops.aten.view.default(add_66, [8192, 768])
        permute_94 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg157_1, view_150, permute_94);  arg157_1 = view_150 = permute_94 = None
        view_151 = torch.ops.aten.view.default(addmm_56, [16, 512, 768]);  addmm_56 = None
        view_152 = torch.ops.aten.view.default(view_151, [16, 512, 12, 64]);  view_151 = None
        permute_95 = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        expand_11 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_11, False);  permute_91 = permute_93 = permute_95 = expand_11 = None
        getitem_74 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_96 = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        view_153 = torch.ops.aten.view.default(permute_96, [16, 512, 768]);  permute_96 = None
        view_154 = torch.ops.aten.view.default(view_153, [8192, 768]);  view_153 = None
        permute_97 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg159_1, view_154, permute_97);  arg159_1 = view_154 = permute_97 = None
        view_155 = torch.ops.aten.view.default(addmm_57, [16, 512, 768]);  addmm_57 = None
        add_67 = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_19[0]
        getitem_79 = var_mean_19[1];  var_mean_19 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_67, getitem_79);  add_67 = getitem_79 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg160_1);  mul_65 = arg160_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_66, arg161_1);  mul_66 = arg161_1 = None
        view_156 = torch.ops.aten.view.default(add_69, [8192, 768])
        permute_98 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg163_1, view_156, permute_98);  arg163_1 = view_156 = permute_98 = None
        view_157 = torch.ops.aten.view.default(addmm_58, [16, 512, 3072]);  addmm_58 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        view_158 = torch.ops.aten.view.default(mul_69, [8192, 3072]);  mul_69 = None
        permute_99 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg165_1, view_158, permute_99);  arg165_1 = view_158 = permute_99 = None
        view_159 = torch.ops.aten.view.default(addmm_59, [16, 512, 768]);  addmm_59 = None
        add_71 = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_71, getitem_81);  add_71 = getitem_81 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg166_1);  mul_70 = arg166_1 = None
        add_73 = torch.ops.aten.add.Tensor(mul_71, arg167_1);  mul_71 = arg167_1 = None
        view_160 = torch.ops.aten.view.default(add_73, [8192, 768])
        permute_100 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg169_1, view_160, permute_100);  arg169_1 = view_160 = permute_100 = None
        view_161 = torch.ops.aten.view.default(addmm_60, [16, 512, 768]);  addmm_60 = None
        view_162 = torch.ops.aten.view.default(view_161, [16, 512, 12, 64]);  view_161 = None
        permute_101 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        view_163 = torch.ops.aten.view.default(add_73, [8192, 768])
        permute_102 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg171_1, view_163, permute_102);  arg171_1 = view_163 = permute_102 = None
        view_164 = torch.ops.aten.view.default(addmm_61, [16, 512, 768]);  addmm_61 = None
        view_165 = torch.ops.aten.view.default(view_164, [16, 512, 12, 64]);  view_164 = None
        permute_103 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        view_166 = torch.ops.aten.view.default(add_73, [8192, 768])
        permute_104 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg173_1, view_166, permute_104);  arg173_1 = view_166 = permute_104 = None
        view_167 = torch.ops.aten.view.default(addmm_62, [16, 512, 768]);  addmm_62 = None
        view_168 = torch.ops.aten.view.default(view_167, [16, 512, 12, 64]);  view_167 = None
        permute_105 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        expand_12 = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_12, False);  permute_101 = permute_103 = permute_105 = expand_12 = None
        getitem_82 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_106 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_169 = torch.ops.aten.view.default(permute_106, [16, 512, 768]);  permute_106 = None
        view_170 = torch.ops.aten.view.default(view_169, [8192, 768]);  view_169 = None
        permute_107 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg175_1, view_170, permute_107);  arg175_1 = view_170 = permute_107 = None
        view_171 = torch.ops.aten.view.default(addmm_63, [16, 512, 768]);  addmm_63 = None
        add_74 = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_21[0]
        getitem_87 = var_mean_21[1];  var_mean_21 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_74, getitem_87);  add_74 = getitem_87 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg176_1);  mul_72 = arg176_1 = None
        add_76 = torch.ops.aten.add.Tensor(mul_73, arg177_1);  mul_73 = arg177_1 = None
        view_172 = torch.ops.aten.view.default(add_76, [8192, 768])
        permute_108 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg179_1, view_172, permute_108);  arg179_1 = view_172 = permute_108 = None
        view_173 = torch.ops.aten.view.default(addmm_64, [16, 512, 3072]);  addmm_64 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10 = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        view_174 = torch.ops.aten.view.default(mul_76, [8192, 3072]);  mul_76 = None
        permute_109 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg181_1, view_174, permute_109);  arg181_1 = view_174 = permute_109 = None
        view_175 = torch.ops.aten.view.default(addmm_65, [16, 512, 768]);  addmm_65 = None
        add_78 = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_79 = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_78, getitem_89);  add_78 = getitem_89 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, arg182_1);  mul_77 = arg182_1 = None
        add_80 = torch.ops.aten.add.Tensor(mul_78, arg183_1);  mul_78 = arg183_1 = None
        view_176 = torch.ops.aten.view.default(add_80, [8192, 768])
        permute_110 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg185_1, view_176, permute_110);  arg185_1 = view_176 = permute_110 = None
        view_177 = torch.ops.aten.view.default(addmm_66, [16, 512, 768]);  addmm_66 = None
        view_178 = torch.ops.aten.view.default(view_177, [16, 512, 12, 64]);  view_177 = None
        permute_111 = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        view_179 = torch.ops.aten.view.default(add_80, [8192, 768])
        permute_112 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg187_1, view_179, permute_112);  arg187_1 = view_179 = permute_112 = None
        view_180 = torch.ops.aten.view.default(addmm_67, [16, 512, 768]);  addmm_67 = None
        view_181 = torch.ops.aten.view.default(view_180, [16, 512, 12, 64]);  view_180 = None
        permute_113 = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        view_182 = torch.ops.aten.view.default(add_80, [8192, 768])
        permute_114 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg189_1, view_182, permute_114);  arg189_1 = view_182 = permute_114 = None
        view_183 = torch.ops.aten.view.default(addmm_68, [16, 512, 768]);  addmm_68 = None
        view_184 = torch.ops.aten.view.default(view_183, [16, 512, 12, 64]);  view_183 = None
        permute_115 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        expand_13 = torch.ops.aten.expand.default(where, [16, 12, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_13, False);  permute_111 = permute_113 = permute_115 = expand_13 = None
        getitem_90 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_116 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        view_185 = torch.ops.aten.view.default(permute_116, [16, 512, 768]);  permute_116 = None
        view_186 = torch.ops.aten.view.default(view_185, [8192, 768]);  view_185 = None
        permute_117 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg191_1, view_186, permute_117);  arg191_1 = view_186 = permute_117 = None
        view_187 = torch.ops.aten.view.default(addmm_69, [16, 512, 768]);  addmm_69 = None
        add_81 = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_23[0]
        getitem_95 = var_mean_23[1];  var_mean_23 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_81, getitem_95);  add_81 = getitem_95 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, arg192_1);  mul_79 = arg192_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_80, arg193_1);  mul_80 = arg193_1 = None
        view_188 = torch.ops.aten.view.default(add_83, [8192, 768])
        permute_118 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg195_1, view_188, permute_118);  arg195_1 = view_188 = permute_118 = None
        view_189 = torch.ops.aten.view.default(addmm_70, [16, 512, 3072]);  addmm_70 = None
        mul_81 = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82 = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11 = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        view_190 = torch.ops.aten.view.default(mul_83, [8192, 3072]);  mul_83 = None
        permute_119 = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg197_1, view_190, permute_119);  arg197_1 = view_190 = permute_119 = None
        view_191 = torch.ops.aten.view.default(addmm_71, [16, 512, 768]);  addmm_71 = None
        add_85 = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_85, getitem_97);  add_85 = getitem_97 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg198_1);  mul_84 = arg198_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_85, arg199_1);  mul_85 = arg199_1 = None
        view_192 = torch.ops.aten.view.default(add_87, [8192, 768]);  add_87 = None
        permute_120 = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg201_1, view_192, permute_120);  arg201_1 = view_192 = permute_120 = None
        view_193 = torch.ops.aten.view.default(addmm_72, [16, 512, 2]);  addmm_72 = None
        split = torch.ops.aten.split.Tensor(view_193, 1, -1);  view_193 = None
        getitem_98 = split[0]
        getitem_99 = split[1];  split = None
        squeeze = torch.ops.aten.squeeze.dim(getitem_98, -1);  getitem_98 = None
        clone_25 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1 = torch.ops.aten.squeeze.dim(getitem_99, -1);  getitem_99 = None
        clone_26 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(arg202_1, 0);  arg202_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(arg203_1, 0);  arg203_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        amax = torch.ops.aten.amax.default(clone_25, [1], True)
        sub_26 = torch.ops.aten.sub.Tensor(clone_25, amax);  amax = None
        exp = torch.ops.aten.exp.default(sub_26)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, clamp_max, full_default);  ne = full_default = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_2);  sub_27 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        ne_1 = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        amax_1 = torch.ops.aten.amax.default(clone_26, [1], True)
        sub_28 = torch.ops.aten.sub.Tensor(clone_26, amax_1);  amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_28)
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1 = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_29 = torch.ops.aten.sub.Tensor(sub_28, log_1);  sub_28 = log_1 = None
        ne_3 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_2);  ne_3 = full_default_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather_1 = torch.ops.aten.gather.default(sub_29, 1, unsqueeze_3);  sub_29 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_4 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(ne_4, neg_1, full_default_3);  ne_4 = neg_1 = full_default_3 = None
        ne_5 = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        sum_5 = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        sum_6 = torch.ops.aten.sum.default(where_4);  where_4 = None
        div_1 = torch.ops.aten.div.Tensor(sum_6, convert_element_type_2);  sum_6 = convert_element_type_2 = None
        add_88 = torch.ops.aten.add.Tensor(div, div_1);  div = div_1 = None
        div_2 = torch.ops.aten.div.Tensor(add_88, 2);  add_88 = None
        return (div_2, clone_25, clone_26)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 93763584, device=device(type='cuda', index=0))
    reader.tensor(buf3, (30522, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # arg8_1
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
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (3072, 768), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf19, (3072,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 3072), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768, 768), is_leaf=True)  # arg24_1
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
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf34, (3072, 768), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf35, (3072,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768, 3072), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768, 768), is_leaf=True)  # arg40_1
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
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf50, (3072, 768), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf51, (3072,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768, 3072), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # arg56_1
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
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf66, (3072, 768), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf67, (3072,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768, 3072), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768, 768), is_leaf=True)  # arg72_1
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
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf82, (3072, 768), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf83, (3072,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768, 3072), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768, 768), is_leaf=True)  # arg88_1
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
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf98, (3072, 768), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf99, (3072,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768, 3072), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768, 768), is_leaf=True)  # arg104_1
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
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf114, (3072, 768), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf115, (3072,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768, 3072), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768, 768), is_leaf=True)  # arg120_1
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
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf130, (3072, 768), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf131, (3072,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768, 3072), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768, 768), is_leaf=True)  # arg136_1
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
    buf144 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf146, (3072, 768), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf147, (3072,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768, 3072), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768, 768), is_leaf=True)  # arg152_1
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
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf162, (3072, 768), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf163, (3072,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768, 3072), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768, 768), is_leaf=True)  # arg168_1
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
    buf176 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf178, (3072, 768), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf179, (3072,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768, 3072), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768, 768), is_leaf=True)  # arg184_1
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
    buf192 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf192, (768,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf194, (3072, 768), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf195, (3072,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf196, (768, 3072), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf199, (768,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf200, (2, 768), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf201, (2,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf202, (16,), dtype=torch.int64, is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf203, (16,), dtype=torch.int64, is_leaf=True)  # arg203_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)