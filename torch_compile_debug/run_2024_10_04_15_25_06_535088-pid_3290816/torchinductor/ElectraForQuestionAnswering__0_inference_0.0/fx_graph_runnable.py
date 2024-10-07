
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1):
        full = torch.ops.aten.full.default([64, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand = torch.ops.aten.expand.default(arg1_1, [64, 512]);  arg1_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default = torch.ops.aten.full.default([64, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None
        embedding = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg3_1, arg201_1);  arg3_1 = arg201_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        view = torch.ops.aten.view.default(add_3, [32768, 128]);  add_3 = None
        permute = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [64, 512, 256]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [32768, 256])
        permute_1 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg10_1, view_2, permute_1);  arg10_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [64, 512, 256]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_1, [32768, 256])
        permute_2 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg12_1, view_4, permute_2);  arg12_1 = view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(addmm_2, [64, 512, 256]);  addmm_2 = None
        view_6 = torch.ops.aten.view.default(view_5, [64, 512, 4, 64]);  view_5 = None
        view_7 = torch.ops.aten.view.default(view_1, [32768, 256])
        permute_4 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg14_1, view_7, permute_4);  arg14_1 = view_7 = permute_4 = None
        view_8 = torch.ops.aten.view.default(addmm_3, [64, 512, 256]);  addmm_3 = None
        view_9 = torch.ops.aten.view.default(view_8, [64, 512, 4, 64]);  view_8 = None
        view_10 = torch.ops.aten.view.default(view_3, [64, 512, 4, 64]);  view_3 = None
        permute_default_33 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        permute_default_34 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_default_35 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_63 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        permute_8 = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        clone_5 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_17 = torch.ops.aten.view.default(clone_5, [64, 512, 256]);  clone_5 = None
        view_18 = torch.ops.aten.view.default(view_17, [32768, 256]);  view_17 = None
        permute_9 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg16_1, view_18, permute_9);  arg16_1 = view_18 = permute_9 = None
        view_19 = torch.ops.aten.view.default(addmm_4, [64, 512, 256]);  addmm_4 = None
        add_5 = torch.ops.aten.add.Tensor(view_19, view_1);  view_19 = view_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg17_1);  mul_3 = arg17_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_4, arg18_1);  mul_4 = arg18_1 = None
        view_20 = torch.ops.aten.view.default(add_7, [32768, 256])
        permute_10 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg20_1, view_20, permute_10);  arg20_1 = view_20 = permute_10 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [64, 512, 1024]);  addmm_5 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_6 = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        view_22 = torch.ops.aten.view.default(mul_7, [32768, 1024]);  mul_7 = None
        permute_11 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg22_1, view_22, permute_11);  arg22_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [64, 512, 256]);  addmm_6 = None
        add_9 = torch.ops.aten.add.Tensor(view_23, add_7);  view_23 = add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg23_1);  mul_8 = arg23_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg24_1);  mul_9 = arg24_1 = None
        view_24 = torch.ops.aten.view.default(add_11, [32768, 256])
        permute_12 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg26_1, view_24, permute_12);  arg26_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [64, 512, 256]);  addmm_7 = None
        view_26 = torch.ops.aten.view.default(add_11, [32768, 256])
        permute_13 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg28_1, view_26, permute_13);  arg28_1 = view_26 = permute_13 = None
        view_27 = torch.ops.aten.view.default(addmm_8, [64, 512, 256]);  addmm_8 = None
        view_28 = torch.ops.aten.view.default(view_27, [64, 512, 4, 64]);  view_27 = None
        view_29 = torch.ops.aten.view.default(add_11, [32768, 256])
        permute_15 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg30_1, view_29, permute_15);  arg30_1 = view_29 = permute_15 = None
        view_30 = torch.ops.aten.view.default(addmm_9, [64, 512, 256]);  addmm_9 = None
        view_31 = torch.ops.aten.view.default(view_30, [64, 512, 4, 64]);  view_30 = None
        view_32 = torch.ops.aten.view.default(view_25, [64, 512, 4, 64]);  view_25 = None
        permute_default_30 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        permute_default_31 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_default_32 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_62 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        permute_19 = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
        clone_12 = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_39 = torch.ops.aten.view.default(clone_12, [64, 512, 256]);  clone_12 = None
        view_40 = torch.ops.aten.view.default(view_39, [32768, 256]);  view_39 = None
        permute_20 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg32_1, view_40, permute_20);  arg32_1 = view_40 = permute_20 = None
        view_41 = torch.ops.aten.view.default(addmm_10, [64, 512, 256]);  addmm_10 = None
        add_13 = torch.ops.aten.add.Tensor(view_41, add_11);  view_41 = add_11 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_11, arg34_1);  mul_11 = arg34_1 = None
        view_42 = torch.ops.aten.view.default(add_15, [32768, 256])
        permute_21 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg36_1, view_42, permute_21);  arg36_1 = view_42 = permute_21 = None
        view_43 = torch.ops.aten.view.default(addmm_11, [64, 512, 1024]);  addmm_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_13 = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1 = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_44 = torch.ops.aten.view.default(mul_14, [32768, 1024]);  mul_14 = None
        permute_22 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg38_1, view_44, permute_22);  arg38_1 = view_44 = permute_22 = None
        view_45 = torch.ops.aten.view.default(addmm_12, [64, 512, 256]);  addmm_12 = None
        add_17 = torch.ops.aten.add.Tensor(view_45, add_15);  view_45 = add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, arg39_1);  mul_15 = arg39_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_16, arg40_1);  mul_16 = arg40_1 = None
        view_46 = torch.ops.aten.view.default(add_19, [32768, 256])
        permute_23 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg42_1, view_46, permute_23);  arg42_1 = view_46 = permute_23 = None
        view_47 = torch.ops.aten.view.default(addmm_13, [64, 512, 256]);  addmm_13 = None
        view_48 = torch.ops.aten.view.default(add_19, [32768, 256])
        permute_24 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg44_1, view_48, permute_24);  arg44_1 = view_48 = permute_24 = None
        view_49 = torch.ops.aten.view.default(addmm_14, [64, 512, 256]);  addmm_14 = None
        view_50 = torch.ops.aten.view.default(view_49, [64, 512, 4, 64]);  view_49 = None
        view_51 = torch.ops.aten.view.default(add_19, [32768, 256])
        permute_26 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg46_1, view_51, permute_26);  arg46_1 = view_51 = permute_26 = None
        view_52 = torch.ops.aten.view.default(addmm_15, [64, 512, 256]);  addmm_15 = None
        view_53 = torch.ops.aten.view.default(view_52, [64, 512, 4, 64]);  view_52 = None
        view_54 = torch.ops.aten.view.default(view_47, [64, 512, 4, 64]);  view_47 = None
        permute_default_27 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        permute_default_28 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_default_29 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_61 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        permute_30 = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
        clone_19 = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_61 = torch.ops.aten.view.default(clone_19, [64, 512, 256]);  clone_19 = None
        view_62 = torch.ops.aten.view.default(view_61, [32768, 256]);  view_61 = None
        permute_31 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg48_1, view_62, permute_31);  arg48_1 = view_62 = permute_31 = None
        view_63 = torch.ops.aten.view.default(addmm_16, [64, 512, 256]);  addmm_16 = None
        add_21 = torch.ops.aten.add.Tensor(view_63, add_19);  view_63 = add_19 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg49_1);  mul_17 = arg49_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_18, arg50_1);  mul_18 = arg50_1 = None
        view_64 = torch.ops.aten.view.default(add_23, [32768, 256])
        permute_32 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg52_1, view_64, permute_32);  arg52_1 = view_64 = permute_32 = None
        view_65 = torch.ops.aten.view.default(addmm_17, [64, 512, 1024]);  addmm_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_20 = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2 = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
        view_66 = torch.ops.aten.view.default(mul_21, [32768, 1024]);  mul_21 = None
        permute_33 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg54_1, view_66, permute_33);  arg54_1 = view_66 = permute_33 = None
        view_67 = torch.ops.aten.view.default(addmm_18, [64, 512, 256]);  addmm_18 = None
        add_25 = torch.ops.aten.add.Tensor(view_67, add_23);  view_67 = add_23 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg55_1);  mul_22 = arg55_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_23, arg56_1);  mul_23 = arg56_1 = None
        view_68 = torch.ops.aten.view.default(add_27, [32768, 256])
        permute_34 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg58_1, view_68, permute_34);  arg58_1 = view_68 = permute_34 = None
        view_69 = torch.ops.aten.view.default(addmm_19, [64, 512, 256]);  addmm_19 = None
        view_70 = torch.ops.aten.view.default(add_27, [32768, 256])
        permute_35 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg60_1, view_70, permute_35);  arg60_1 = view_70 = permute_35 = None
        view_71 = torch.ops.aten.view.default(addmm_20, [64, 512, 256]);  addmm_20 = None
        view_72 = torch.ops.aten.view.default(view_71, [64, 512, 4, 64]);  view_71 = None
        view_73 = torch.ops.aten.view.default(add_27, [32768, 256])
        permute_37 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg62_1, view_73, permute_37);  arg62_1 = view_73 = permute_37 = None
        view_74 = torch.ops.aten.view.default(addmm_21, [64, 512, 256]);  addmm_21 = None
        view_75 = torch.ops.aten.view.default(view_74, [64, 512, 4, 64]);  view_74 = None
        view_76 = torch.ops.aten.view.default(view_69, [64, 512, 4, 64]);  view_69 = None
        permute_default_24 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        permute_default_25 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        permute_default_26 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_60 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        permute_41 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        clone_26 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_83 = torch.ops.aten.view.default(clone_26, [64, 512, 256]);  clone_26 = None
        view_84 = torch.ops.aten.view.default(view_83, [32768, 256]);  view_83 = None
        permute_42 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg64_1, view_84, permute_42);  arg64_1 = view_84 = permute_42 = None
        view_85 = torch.ops.aten.view.default(addmm_22, [64, 512, 256]);  addmm_22 = None
        add_29 = torch.ops.aten.add.Tensor(view_85, add_27);  view_85 = add_27 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg65_1);  mul_24 = arg65_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_25, arg66_1);  mul_25 = arg66_1 = None
        view_86 = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_43 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg68_1, view_86, permute_43);  arg68_1 = view_86 = permute_43 = None
        view_87 = torch.ops.aten.view.default(addmm_23, [64, 512, 1024]);  addmm_23 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_27 = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3 = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        view_88 = torch.ops.aten.view.default(mul_28, [32768, 1024]);  mul_28 = None
        permute_44 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg70_1, view_88, permute_44);  arg70_1 = view_88 = permute_44 = None
        view_89 = torch.ops.aten.view.default(addmm_24, [64, 512, 256]);  addmm_24 = None
        add_33 = torch.ops.aten.add.Tensor(view_89, add_31);  view_89 = add_31 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, arg71_1);  mul_29 = arg71_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_30, arg72_1);  mul_30 = arg72_1 = None
        view_90 = torch.ops.aten.view.default(add_35, [32768, 256])
        permute_45 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg74_1, view_90, permute_45);  arg74_1 = view_90 = permute_45 = None
        view_91 = torch.ops.aten.view.default(addmm_25, [64, 512, 256]);  addmm_25 = None
        view_92 = torch.ops.aten.view.default(add_35, [32768, 256])
        permute_46 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg76_1, view_92, permute_46);  arg76_1 = view_92 = permute_46 = None
        view_93 = torch.ops.aten.view.default(addmm_26, [64, 512, 256]);  addmm_26 = None
        view_94 = torch.ops.aten.view.default(view_93, [64, 512, 4, 64]);  view_93 = None
        view_95 = torch.ops.aten.view.default(add_35, [32768, 256])
        permute_48 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg78_1, view_95, permute_48);  arg78_1 = view_95 = permute_48 = None
        view_96 = torch.ops.aten.view.default(addmm_27, [64, 512, 256]);  addmm_27 = None
        view_97 = torch.ops.aten.view.default(view_96, [64, 512, 4, 64]);  view_96 = None
        view_98 = torch.ops.aten.view.default(view_91, [64, 512, 4, 64]);  view_91 = None
        permute_default_21 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        permute_default_22 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_default_23 = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_59 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        permute_52 = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        clone_33 = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_105 = torch.ops.aten.view.default(clone_33, [64, 512, 256]);  clone_33 = None
        view_106 = torch.ops.aten.view.default(view_105, [32768, 256]);  view_105 = None
        permute_53 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg80_1, view_106, permute_53);  arg80_1 = view_106 = permute_53 = None
        view_107 = torch.ops.aten.view.default(addmm_28, [64, 512, 256]);  addmm_28 = None
        add_37 = torch.ops.aten.add.Tensor(view_107, add_35);  view_107 = add_35 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, arg81_1);  mul_31 = arg81_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_32, arg82_1);  mul_32 = arg82_1 = None
        view_108 = torch.ops.aten.view.default(add_39, [32768, 256])
        permute_54 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg84_1, view_108, permute_54);  arg84_1 = view_108 = permute_54 = None
        view_109 = torch.ops.aten.view.default(addmm_29, [64, 512, 1024]);  addmm_29 = None
        mul_33 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_34 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4 = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
        view_110 = torch.ops.aten.view.default(mul_35, [32768, 1024]);  mul_35 = None
        permute_55 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg86_1, view_110, permute_55);  arg86_1 = view_110 = permute_55 = None
        view_111 = torch.ops.aten.view.default(addmm_30, [64, 512, 256]);  addmm_30 = None
        add_41 = torch.ops.aten.add.Tensor(view_111, add_39);  view_111 = add_39 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg87_1);  mul_36 = arg87_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_37, arg88_1);  mul_37 = arg88_1 = None
        view_112 = torch.ops.aten.view.default(add_43, [32768, 256])
        permute_56 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg90_1, view_112, permute_56);  arg90_1 = view_112 = permute_56 = None
        view_113 = torch.ops.aten.view.default(addmm_31, [64, 512, 256]);  addmm_31 = None
        view_114 = torch.ops.aten.view.default(add_43, [32768, 256])
        permute_57 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg92_1, view_114, permute_57);  arg92_1 = view_114 = permute_57 = None
        view_115 = torch.ops.aten.view.default(addmm_32, [64, 512, 256]);  addmm_32 = None
        view_116 = torch.ops.aten.view.default(view_115, [64, 512, 4, 64]);  view_115 = None
        view_117 = torch.ops.aten.view.default(add_43, [32768, 256])
        permute_59 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg94_1, view_117, permute_59);  arg94_1 = view_117 = permute_59 = None
        view_118 = torch.ops.aten.view.default(addmm_33, [64, 512, 256]);  addmm_33 = None
        view_119 = torch.ops.aten.view.default(view_118, [64, 512, 4, 64]);  view_118 = None
        view_120 = torch.ops.aten.view.default(view_113, [64, 512, 4, 64]);  view_113 = None
        permute_default_18 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        permute_default_19 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        permute_default_20 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_58 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        permute_63 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_40 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_127 = torch.ops.aten.view.default(clone_40, [64, 512, 256]);  clone_40 = None
        view_128 = torch.ops.aten.view.default(view_127, [32768, 256]);  view_127 = None
        permute_64 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg96_1, view_128, permute_64);  arg96_1 = view_128 = permute_64 = None
        view_129 = torch.ops.aten.view.default(addmm_34, [64, 512, 256]);  addmm_34 = None
        add_45 = torch.ops.aten.add.Tensor(view_129, add_43);  view_129 = add_43 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg97_1);  mul_38 = arg97_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_39, arg98_1);  mul_39 = arg98_1 = None
        view_130 = torch.ops.aten.view.default(add_47, [32768, 256])
        permute_65 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg100_1, view_130, permute_65);  arg100_1 = view_130 = permute_65 = None
        view_131 = torch.ops.aten.view.default(addmm_35, [64, 512, 1024]);  addmm_35 = None
        mul_40 = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
        view_132 = torch.ops.aten.view.default(mul_42, [32768, 1024]);  mul_42 = None
        permute_66 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg102_1, view_132, permute_66);  arg102_1 = view_132 = permute_66 = None
        view_133 = torch.ops.aten.view.default(addmm_36, [64, 512, 256]);  addmm_36 = None
        add_49 = torch.ops.aten.add.Tensor(view_133, add_47);  view_133 = add_47 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, arg103_1);  mul_43 = arg103_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_44, arg104_1);  mul_44 = arg104_1 = None
        view_134 = torch.ops.aten.view.default(add_51, [32768, 256])
        permute_67 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg106_1, view_134, permute_67);  arg106_1 = view_134 = permute_67 = None
        view_135 = torch.ops.aten.view.default(addmm_37, [64, 512, 256]);  addmm_37 = None
        view_136 = torch.ops.aten.view.default(add_51, [32768, 256])
        permute_68 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg108_1, view_136, permute_68);  arg108_1 = view_136 = permute_68 = None
        view_137 = torch.ops.aten.view.default(addmm_38, [64, 512, 256]);  addmm_38 = None
        view_138 = torch.ops.aten.view.default(view_137, [64, 512, 4, 64]);  view_137 = None
        view_139 = torch.ops.aten.view.default(add_51, [32768, 256])
        permute_70 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg110_1, view_139, permute_70);  arg110_1 = view_139 = permute_70 = None
        view_140 = torch.ops.aten.view.default(addmm_39, [64, 512, 256]);  addmm_39 = None
        view_141 = torch.ops.aten.view.default(view_140, [64, 512, 4, 64]);  view_140 = None
        view_142 = torch.ops.aten.view.default(view_135, [64, 512, 4, 64]);  view_135 = None
        permute_default_15 = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        permute_default_16 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        permute_default_17 = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_57 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        permute_74 = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_47 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_149 = torch.ops.aten.view.default(clone_47, [64, 512, 256]);  clone_47 = None
        view_150 = torch.ops.aten.view.default(view_149, [32768, 256]);  view_149 = None
        permute_75 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg112_1, view_150, permute_75);  arg112_1 = view_150 = permute_75 = None
        view_151 = torch.ops.aten.view.default(addmm_40, [64, 512, 256]);  addmm_40 = None
        add_53 = torch.ops.aten.add.Tensor(view_151, add_51);  view_151 = add_51 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, arg113_1);  mul_45 = arg113_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_46, arg114_1);  mul_46 = arg114_1 = None
        view_152 = torch.ops.aten.view.default(add_55, [32768, 256])
        permute_76 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg116_1, view_152, permute_76);  arg116_1 = view_152 = permute_76 = None
        view_153 = torch.ops.aten.view.default(addmm_41, [64, 512, 1024]);  addmm_41 = None
        mul_47 = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_48 = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6 = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
        view_154 = torch.ops.aten.view.default(mul_49, [32768, 1024]);  mul_49 = None
        permute_77 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg118_1, view_154, permute_77);  arg118_1 = view_154 = permute_77 = None
        view_155 = torch.ops.aten.view.default(addmm_42, [64, 512, 256]);  addmm_42 = None
        add_57 = torch.ops.aten.add.Tensor(view_155, add_55);  view_155 = add_55 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg119_1);  mul_50 = arg119_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_51, arg120_1);  mul_51 = arg120_1 = None
        view_156 = torch.ops.aten.view.default(add_59, [32768, 256])
        permute_78 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg122_1, view_156, permute_78);  arg122_1 = view_156 = permute_78 = None
        view_157 = torch.ops.aten.view.default(addmm_43, [64, 512, 256]);  addmm_43 = None
        view_158 = torch.ops.aten.view.default(add_59, [32768, 256])
        permute_79 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg124_1, view_158, permute_79);  arg124_1 = view_158 = permute_79 = None
        view_159 = torch.ops.aten.view.default(addmm_44, [64, 512, 256]);  addmm_44 = None
        view_160 = torch.ops.aten.view.default(view_159, [64, 512, 4, 64]);  view_159 = None
        view_161 = torch.ops.aten.view.default(add_59, [32768, 256])
        permute_81 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg126_1, view_161, permute_81);  arg126_1 = view_161 = permute_81 = None
        view_162 = torch.ops.aten.view.default(addmm_45, [64, 512, 256]);  addmm_45 = None
        view_163 = torch.ops.aten.view.default(view_162, [64, 512, 4, 64]);  view_162 = None
        view_164 = torch.ops.aten.view.default(view_157, [64, 512, 4, 64]);  view_157 = None
        permute_default_12 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        permute_default_13 = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        permute_default_14 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_56 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        permute_85 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_54 = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_171 = torch.ops.aten.view.default(clone_54, [64, 512, 256]);  clone_54 = None
        view_172 = torch.ops.aten.view.default(view_171, [32768, 256]);  view_171 = None
        permute_86 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg128_1, view_172, permute_86);  arg128_1 = view_172 = permute_86 = None
        view_173 = torch.ops.aten.view.default(addmm_46, [64, 512, 256]);  addmm_46 = None
        add_61 = torch.ops.aten.add.Tensor(view_173, add_59);  view_173 = add_59 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg129_1);  mul_52 = arg129_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_53, arg130_1);  mul_53 = arg130_1 = None
        view_174 = torch.ops.aten.view.default(add_63, [32768, 256])
        permute_87 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg132_1, view_174, permute_87);  arg132_1 = view_174 = permute_87 = None
        view_175 = torch.ops.aten.view.default(addmm_47, [64, 512, 1024]);  addmm_47 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
        view_176 = torch.ops.aten.view.default(mul_56, [32768, 1024]);  mul_56 = None
        permute_88 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg134_1, view_176, permute_88);  arg134_1 = view_176 = permute_88 = None
        view_177 = torch.ops.aten.view.default(addmm_48, [64, 512, 256]);  addmm_48 = None
        add_65 = torch.ops.aten.add.Tensor(view_177, add_63);  view_177 = add_63 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg135_1);  mul_57 = arg135_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_58, arg136_1);  mul_58 = arg136_1 = None
        view_178 = torch.ops.aten.view.default(add_67, [32768, 256])
        permute_89 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg138_1, view_178, permute_89);  arg138_1 = view_178 = permute_89 = None
        view_179 = torch.ops.aten.view.default(addmm_49, [64, 512, 256]);  addmm_49 = None
        view_180 = torch.ops.aten.view.default(add_67, [32768, 256])
        permute_90 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg140_1, view_180, permute_90);  arg140_1 = view_180 = permute_90 = None
        view_181 = torch.ops.aten.view.default(addmm_50, [64, 512, 256]);  addmm_50 = None
        view_182 = torch.ops.aten.view.default(view_181, [64, 512, 4, 64]);  view_181 = None
        view_183 = torch.ops.aten.view.default(add_67, [32768, 256])
        permute_92 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg142_1, view_183, permute_92);  arg142_1 = view_183 = permute_92 = None
        view_184 = torch.ops.aten.view.default(addmm_51, [64, 512, 256]);  addmm_51 = None
        view_185 = torch.ops.aten.view.default(view_184, [64, 512, 4, 64]);  view_184 = None
        view_186 = torch.ops.aten.view.default(view_179, [64, 512, 4, 64]);  view_179 = None
        permute_default_9 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        permute_default_10 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        permute_default_11 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_55 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        permute_96 = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        clone_61 = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        view_193 = torch.ops.aten.view.default(clone_61, [64, 512, 256]);  clone_61 = None
        view_194 = torch.ops.aten.view.default(view_193, [32768, 256]);  view_193 = None
        permute_97 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg144_1, view_194, permute_97);  arg144_1 = view_194 = permute_97 = None
        view_195 = torch.ops.aten.view.default(addmm_52, [64, 512, 256]);  addmm_52 = None
        add_69 = torch.ops.aten.add.Tensor(view_195, add_67);  view_195 = add_67 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg145_1);  mul_59 = arg145_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_60, arg146_1);  mul_60 = arg146_1 = None
        view_196 = torch.ops.aten.view.default(add_71, [32768, 256])
        permute_98 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg148_1, view_196, permute_98);  arg148_1 = view_196 = permute_98 = None
        view_197 = torch.ops.aten.view.default(addmm_53, [64, 512, 1024]);  addmm_53 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_62 = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_8 = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
        view_198 = torch.ops.aten.view.default(mul_63, [32768, 1024]);  mul_63 = None
        permute_99 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg150_1, view_198, permute_99);  arg150_1 = view_198 = permute_99 = None
        view_199 = torch.ops.aten.view.default(addmm_54, [64, 512, 256]);  addmm_54 = None
        add_73 = torch.ops.aten.add.Tensor(view_199, add_71);  view_199 = add_71 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg151_1);  mul_64 = arg151_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_65, arg152_1);  mul_65 = arg152_1 = None
        view_200 = torch.ops.aten.view.default(add_75, [32768, 256])
        permute_100 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg154_1, view_200, permute_100);  arg154_1 = view_200 = permute_100 = None
        view_201 = torch.ops.aten.view.default(addmm_55, [64, 512, 256]);  addmm_55 = None
        view_202 = torch.ops.aten.view.default(add_75, [32768, 256])
        permute_101 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg156_1, view_202, permute_101);  arg156_1 = view_202 = permute_101 = None
        view_203 = torch.ops.aten.view.default(addmm_56, [64, 512, 256]);  addmm_56 = None
        view_204 = torch.ops.aten.view.default(view_203, [64, 512, 4, 64]);  view_203 = None
        view_205 = torch.ops.aten.view.default(add_75, [32768, 256])
        permute_103 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg158_1, view_205, permute_103);  arg158_1 = view_205 = permute_103 = None
        view_206 = torch.ops.aten.view.default(addmm_57, [64, 512, 256]);  addmm_57 = None
        view_207 = torch.ops.aten.view.default(view_206, [64, 512, 4, 64]);  view_206 = None
        view_208 = torch.ops.aten.view.default(view_201, [64, 512, 4, 64]);  view_201 = None
        permute_default_6 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        permute_default_7 = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        permute_default_8 = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_54 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        permute_107 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_68 = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
        view_215 = torch.ops.aten.view.default(clone_68, [64, 512, 256]);  clone_68 = None
        view_216 = torch.ops.aten.view.default(view_215, [32768, 256]);  view_215 = None
        permute_108 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg160_1, view_216, permute_108);  arg160_1 = view_216 = permute_108 = None
        view_217 = torch.ops.aten.view.default(addmm_58, [64, 512, 256]);  addmm_58 = None
        add_77 = torch.ops.aten.add.Tensor(view_217, add_75);  view_217 = add_75 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg161_1);  mul_66 = arg161_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_67, arg162_1);  mul_67 = arg162_1 = None
        view_218 = torch.ops.aten.view.default(add_79, [32768, 256])
        permute_109 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg164_1, view_218, permute_109);  arg164_1 = view_218 = permute_109 = None
        view_219 = torch.ops.aten.view.default(addmm_59, [64, 512, 1024]);  addmm_59 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_69 = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_9 = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
        view_220 = torch.ops.aten.view.default(mul_70, [32768, 1024]);  mul_70 = None
        permute_110 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg166_1, view_220, permute_110);  arg166_1 = view_220 = permute_110 = None
        view_221 = torch.ops.aten.view.default(addmm_60, [64, 512, 256]);  addmm_60 = None
        add_81 = torch.ops.aten.add.Tensor(view_221, add_79);  view_221 = add_79 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_71 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_71, arg167_1);  mul_71 = arg167_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_72, arg168_1);  mul_72 = arg168_1 = None
        view_222 = torch.ops.aten.view.default(add_83, [32768, 256])
        permute_111 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg170_1, view_222, permute_111);  arg170_1 = view_222 = permute_111 = None
        view_223 = torch.ops.aten.view.default(addmm_61, [64, 512, 256]);  addmm_61 = None
        view_224 = torch.ops.aten.view.default(add_83, [32768, 256])
        permute_112 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg172_1, view_224, permute_112);  arg172_1 = view_224 = permute_112 = None
        view_225 = torch.ops.aten.view.default(addmm_62, [64, 512, 256]);  addmm_62 = None
        view_226 = torch.ops.aten.view.default(view_225, [64, 512, 4, 64]);  view_225 = None
        view_227 = torch.ops.aten.view.default(add_83, [32768, 256])
        permute_114 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg174_1, view_227, permute_114);  arg174_1 = view_227 = permute_114 = None
        view_228 = torch.ops.aten.view.default(addmm_63, [64, 512, 256]);  addmm_63 = None
        view_229 = torch.ops.aten.view.default(view_228, [64, 512, 4, 64]);  view_228 = None
        view_230 = torch.ops.aten.view.default(view_223, [64, 512, 4, 64]);  view_223 = None
        permute_default_3 = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        permute_default_4 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        permute_default_5 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_53 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        permute_118 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        clone_75 = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_237 = torch.ops.aten.view.default(clone_75, [64, 512, 256]);  clone_75 = None
        view_238 = torch.ops.aten.view.default(view_237, [32768, 256]);  view_237 = None
        permute_119 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg176_1, view_238, permute_119);  arg176_1 = view_238 = permute_119 = None
        view_239 = torch.ops.aten.view.default(addmm_64, [64, 512, 256]);  addmm_64 = None
        add_85 = torch.ops.aten.add.Tensor(view_239, add_83);  view_239 = add_83 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg177_1);  mul_73 = arg177_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_74, arg178_1);  mul_74 = arg178_1 = None
        view_240 = torch.ops.aten.view.default(add_87, [32768, 256])
        permute_120 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg180_1, view_240, permute_120);  arg180_1 = view_240 = permute_120 = None
        view_241 = torch.ops.aten.view.default(addmm_65, [64, 512, 1024]);  addmm_65 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_76 = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
        erf_10 = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
        view_242 = torch.ops.aten.view.default(mul_77, [32768, 1024]);  mul_77 = None
        permute_121 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg182_1, view_242, permute_121);  arg182_1 = view_242 = permute_121 = None
        view_243 = torch.ops.aten.view.default(addmm_66, [64, 512, 256]);  addmm_66 = None
        add_89 = torch.ops.aten.add.Tensor(view_243, add_87);  view_243 = add_87 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, arg183_1);  mul_78 = arg183_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_79, arg184_1);  mul_79 = arg184_1 = None
        view_244 = torch.ops.aten.view.default(add_91, [32768, 256])
        permute_122 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg186_1, view_244, permute_122);  arg186_1 = view_244 = permute_122 = None
        view_245 = torch.ops.aten.view.default(addmm_67, [64, 512, 256]);  addmm_67 = None
        view_246 = torch.ops.aten.view.default(add_91, [32768, 256])
        permute_123 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg188_1, view_246, permute_123);  arg188_1 = view_246 = permute_123 = None
        view_247 = torch.ops.aten.view.default(addmm_68, [64, 512, 256]);  addmm_68 = None
        view_248 = torch.ops.aten.view.default(view_247, [64, 512, 4, 64]);  view_247 = None
        view_249 = torch.ops.aten.view.default(add_91, [32768, 256])
        permute_125 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg190_1, view_249, permute_125);  arg190_1 = view_249 = permute_125 = None
        view_250 = torch.ops.aten.view.default(addmm_69, [64, 512, 256]);  addmm_69 = None
        view_251 = torch.ops.aten.view.default(view_250, [64, 512, 4, 64]);  view_250 = None
        view_252 = torch.ops.aten.view.default(view_245, [64, 512, 4, 64]);  view_245 = None
        permute_default = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        permute_default_1 = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        permute_default_2 = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_52 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        permute_129 = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_82 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_259 = torch.ops.aten.view.default(clone_82, [64, 512, 256]);  clone_82 = None
        view_260 = torch.ops.aten.view.default(view_259, [32768, 256]);  view_259 = None
        permute_130 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg192_1, view_260, permute_130);  arg192_1 = view_260 = permute_130 = None
        view_261 = torch.ops.aten.view.default(addmm_70, [64, 512, 256]);  addmm_70 = None
        add_93 = torch.ops.aten.add.Tensor(view_261, add_91);  view_261 = add_91 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg193_1);  mul_80 = arg193_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_81, arg194_1);  mul_81 = arg194_1 = None
        view_262 = torch.ops.aten.view.default(add_95, [32768, 256])
        permute_131 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg196_1, view_262, permute_131);  arg196_1 = view_262 = permute_131 = None
        view_263 = torch.ops.aten.view.default(addmm_71, [64, 512, 1024]);  addmm_71 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_83 = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_11 = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
        view_264 = torch.ops.aten.view.default(mul_84, [32768, 1024]);  mul_84 = None
        permute_132 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg198_1, view_264, permute_132);  arg198_1 = view_264 = permute_132 = None
        view_265 = torch.ops.aten.view.default(addmm_72, [64, 512, 256]);  addmm_72 = None
        add_97 = torch.ops.aten.add.Tensor(view_265, add_95);  view_265 = add_95 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, arg199_1);  mul_85 = arg199_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_86, arg200_1);  mul_86 = arg200_1 = None
        view_266 = torch.ops.aten.view.default(add_99, [32768, 256]);  add_99 = None
        permute_133 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg203_1, view_266, permute_133);  arg203_1 = view_266 = permute_133 = None
        view_267 = torch.ops.aten.view.default(addmm_73, [64, 512, 2]);  addmm_73 = None
        split = torch.ops.aten.split.Tensor(view_267, 1, -1);  view_267 = None
        getitem_50 = split[0]
        getitem_51 = split[1];  split = None
        squeeze = torch.ops.aten.squeeze.dim(getitem_50, -1);  getitem_50 = None
        clone_85 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1 = torch.ops.aten.squeeze.dim(getitem_51, -1);  getitem_51 = None
        clone_86 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(arg204_1, 0);  arg204_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(arg205_1, 0);  arg205_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        amax_12 = torch.ops.aten.amax.default(clone_85, [1], True)
        sub_38 = torch.ops.aten.sub.Tensor(clone_85, amax_12);  amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_38)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        ne = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_1 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, clamp_max, full_default_1);  ne = full_default_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  sub_39 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        ne_1 = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_2);  ne_1 = neg = full_default_2 = None
        ne_2 = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        sum_14 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_24 = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        amax_13 = torch.ops.aten.amax.default(clone_86, [1], True)
        sub_40 = torch.ops.aten.sub.Tensor(clone_86, amax_13);  amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_40)
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
        log_1 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_41 = torch.ops.aten.sub.Tensor(sub_40, log_1);  sub_40 = log_1 = None
        ne_3 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_3 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_3);  ne_3 = full_default_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1 = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_3);  sub_41 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_4 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_4, neg_1, full_default_4);  ne_4 = neg_1 = full_default_4 = None
        ne_5 = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        sum_17 = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
        sum_18 = torch.ops.aten.sum.default(where_3);  where_3 = None
        div_25 = torch.ops.aten.div.Tensor(sum_18, convert_element_type_1);  sum_18 = convert_element_type_1 = None
        add_100 = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None
        div_26 = torch.ops.aten.div.Tensor(add_100, 2);  add_100 = None
        return (div_26, clone_85, clone_86)
        
def load_args(reader):
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (64, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 15627264, device=device(type='cuda', index=0))
    reader.tensor(buf2, (30522, 128), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 128), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 128), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (256, 128), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf8, (256,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf9, (256, 256), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256, 256), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256, 256), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256, 256), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf16, (256,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf18, (256,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024, 256), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 1024), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256, 256), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256, 256), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256, 256), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256, 256), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1024, 256), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256, 1024), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 256), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 256), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256, 256), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf46, (256,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256, 256), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (256,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024, 256), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1024,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256, 1024), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256, 256), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256, 256), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf61, (256, 256), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf62, (256,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf63, (256, 256), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1024, 256), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1024,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf69, (256, 1024), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf70, (256,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf71, (256,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256, 256), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256, 256), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf76, (256,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf77, (256, 256), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf78, (256,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf79, (256, 256), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf80, (256,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf82, (256,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1024, 256), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1024,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256, 1024), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf86, (256,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf87, (256,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf88, (256,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf89, (256, 256), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256, 256), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256, 256), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf95, (256, 256), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf96, (256,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf97, (256,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf98, (256,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1024, 256), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf101, (256, 1024), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf102, (256,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf103, (256,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf104, (256,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf105, (256, 256), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf106, (256,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf107, (256, 256), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf108, (256,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf109, (256, 256), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf110, (256,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256, 256), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf113, (256,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf114, (256,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1024, 256), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1024,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf117, (256, 1024), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf118, (256,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf119, (256,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 256), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256, 256), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256, 256), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256, 256), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf128, (256,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024, 256), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256, 1024), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (256,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf136, (256,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf137, (256, 256), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf138, (256,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (256, 256), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf140, (256,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (256, 256), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf142, (256,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256, 256), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf145, (256,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf146, (256,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1024, 256), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1024,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf149, (256, 1024), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf150, (256,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf151, (256,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf152, (256,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (256, 256), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf154, (256,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf155, (256, 256), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (256, 256), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf158, (256,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf159, (256, 256), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf160, (256,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf161, (256,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf162, (256,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1024, 256), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf164, (1024,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf165, (256, 1024), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf166, (256,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf167, (256,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf168, (256,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf169, (256, 256), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf170, (256,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf171, (256, 256), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf172, (256,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf173, (256, 256), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf174, (256,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf175, (256, 256), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf176, (256,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf177, (256,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf178, (256,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1024, 256), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1024,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf181, (256, 1024), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf182, (256,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf183, (256,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf184, (256,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf185, (256, 256), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf186, (256,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf187, (256, 256), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf188, (256,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf189, (256, 256), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf190, (256,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf191, (256, 256), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf192, (256,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf193, (256,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf194, (256,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1024, 256), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1024,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf197, (256, 1024), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf198, (256,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf199, (256,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf200, (256,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf201, (1, 512), dtype=torch.int64, is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf202, (2, 256), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf203, (2,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf204, (64,), dtype=torch.int64, is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf205, (64,), dtype=torch.int64, is_leaf=True)  # arg205_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)