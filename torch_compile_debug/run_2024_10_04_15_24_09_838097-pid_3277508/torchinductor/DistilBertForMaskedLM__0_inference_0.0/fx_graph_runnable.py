
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1):
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, slice_2);  arg3_1 = slice_2 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        full = torch.ops.aten.full.default([128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand = torch.ops.aten.expand.default(unsqueeze_1, [128, 1, 128, 128]);  unsqueeze_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(1.0, expand);  expand = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(convert_element_type, scalar_tensor, sub_1);  convert_element_type = scalar_tensor = sub_1 = None
        view = torch.ops.aten.view.default(add_2, [16384, 768])
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [128, 128, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [128, -1, 12, 64]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(add_2, [16384, 768])
        permute_2 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [128, 128, 768]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [128, -1, 12, 64]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6 = torch.ops.aten.view.default(add_2, [16384, 768])
        permute_4 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_6, permute_4);  arg11_1 = view_6 = permute_4 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [128, 128, 768]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [128, -1, 12, 64]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        expand_1 = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_1, False);  permute_1 = permute_3 = permute_5 = expand_1 = None
        getitem_2 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_9 = torch.ops.aten.view.default(permute_6, [128, -1, 768]);  permute_6 = None
        view_10 = torch.ops.aten.view.default(view_9, [16384, 768]);  view_9 = None
        permute_7 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_10, permute_7);  arg13_1 = view_10 = permute_7 = None
        view_11 = torch.ops.aten.view.default(addmm_3, [128, 128, 768]);  addmm_3 = None
        add_3 = torch.ops.aten.add.Tensor(view_11, add_2);  view_11 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_3, getitem_7);  add_3 = getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg14_1);  mul_2 = arg14_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None
        view_12 = torch.ops.aten.view.default(add_5, [16384, 768])
        permute_8 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_12, permute_8);  arg17_1 = view_12 = permute_8 = None
        view_13 = torch.ops.aten.view.default(addmm_4, [128, 128, 3072]);  addmm_4 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        view_14 = torch.ops.aten.view.default(mul_6, [16384, 3072]);  mul_6 = None
        permute_9 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_14, permute_9);  arg19_1 = view_14 = permute_9 = None
        view_15 = torch.ops.aten.view.default(addmm_5, [128, 128, 768]);  addmm_5 = None
        add_7 = torch.ops.aten.add.Tensor(view_15, add_5);  view_15 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_9);  add_7 = getitem_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, arg20_1);  mul_7 = arg20_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_8, arg21_1);  mul_8 = arg21_1 = None
        view_16 = torch.ops.aten.view.default(add_9, [16384, 768])
        permute_10 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_16, permute_10);  arg23_1 = view_16 = permute_10 = None
        view_17 = torch.ops.aten.view.default(addmm_6, [128, 128, 768]);  addmm_6 = None
        view_18 = torch.ops.aten.view.default(view_17, [128, -1, 12, 64]);  view_17 = None
        permute_11 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        view_19 = torch.ops.aten.view.default(add_9, [16384, 768])
        permute_12 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_19, permute_12);  arg25_1 = view_19 = permute_12 = None
        view_20 = torch.ops.aten.view.default(addmm_7, [128, 128, 768]);  addmm_7 = None
        view_21 = torch.ops.aten.view.default(view_20, [128, -1, 12, 64]);  view_20 = None
        permute_13 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        view_22 = torch.ops.aten.view.default(add_9, [16384, 768])
        permute_14 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg27_1, view_22, permute_14);  arg27_1 = view_22 = permute_14 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [128, 128, 768]);  addmm_8 = None
        view_24 = torch.ops.aten.view.default(view_23, [128, -1, 12, 64]);  view_23 = None
        permute_15 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        expand_2 = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_2, False);  permute_11 = permute_13 = permute_15 = expand_2 = None
        getitem_10 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_25 = torch.ops.aten.view.default(permute_16, [128, -1, 768]);  permute_16 = None
        view_26 = torch.ops.aten.view.default(view_25, [16384, 768]);  view_25 = None
        permute_17 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg29_1, view_26, permute_17);  arg29_1 = view_26 = permute_17 = None
        view_27 = torch.ops.aten.view.default(addmm_9, [128, 128, 768]);  addmm_9 = None
        add_10 = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_10, getitem_15);  add_10 = getitem_15 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg30_1);  mul_9 = arg30_1 = None
        add_12 = torch.ops.aten.add.Tensor(mul_10, arg31_1);  mul_10 = arg31_1 = None
        view_28 = torch.ops.aten.view.default(add_12, [16384, 768])
        permute_18 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg33_1, view_28, permute_18);  arg33_1 = view_28 = permute_18 = None
        view_29 = torch.ops.aten.view.default(addmm_10, [128, 128, 3072]);  addmm_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        view_30 = torch.ops.aten.view.default(mul_13, [16384, 3072]);  mul_13 = None
        permute_19 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg35_1, view_30, permute_19);  arg35_1 = view_30 = permute_19 = None
        view_31 = torch.ops.aten.view.default(addmm_11, [128, 128, 768]);  addmm_11 = None
        add_14 = torch.ops.aten.add.Tensor(view_31, add_12);  view_31 = add_12 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_14, getitem_17);  add_14 = getitem_17 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg36_1);  mul_14 = arg36_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
        view_32 = torch.ops.aten.view.default(add_16, [16384, 768])
        permute_20 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg39_1, view_32, permute_20);  arg39_1 = view_32 = permute_20 = None
        view_33 = torch.ops.aten.view.default(addmm_12, [128, 128, 768]);  addmm_12 = None
        view_34 = torch.ops.aten.view.default(view_33, [128, -1, 12, 64]);  view_33 = None
        permute_21 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        view_35 = torch.ops.aten.view.default(add_16, [16384, 768])
        permute_22 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg41_1, view_35, permute_22);  arg41_1 = view_35 = permute_22 = None
        view_36 = torch.ops.aten.view.default(addmm_13, [128, 128, 768]);  addmm_13 = None
        view_37 = torch.ops.aten.view.default(view_36, [128, -1, 12, 64]);  view_36 = None
        permute_23 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        view_38 = torch.ops.aten.view.default(add_16, [16384, 768])
        permute_24 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg43_1, view_38, permute_24);  arg43_1 = view_38 = permute_24 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [128, 128, 768]);  addmm_14 = None
        view_40 = torch.ops.aten.view.default(view_39, [128, -1, 12, 64]);  view_39 = None
        permute_25 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        expand_3 = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_3, False);  permute_21 = permute_23 = permute_25 = expand_3 = None
        getitem_18 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_41 = torch.ops.aten.view.default(permute_26, [128, -1, 768]);  permute_26 = None
        view_42 = torch.ops.aten.view.default(view_41, [16384, 768]);  view_41 = None
        permute_27 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg45_1, view_42, permute_27);  arg45_1 = view_42 = permute_27 = None
        view_43 = torch.ops.aten.view.default(addmm_15, [128, 128, 768]);  addmm_15 = None
        add_17 = torch.ops.aten.add.Tensor(view_43, add_16);  view_43 = add_16 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg46_1);  mul_16 = arg46_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, arg47_1);  mul_17 = arg47_1 = None
        view_44 = torch.ops.aten.view.default(add_19, [16384, 768])
        permute_28 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg49_1, view_44, permute_28);  arg49_1 = view_44 = permute_28 = None
        view_45 = torch.ops.aten.view.default(addmm_16, [128, 128, 3072]);  addmm_16 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
        view_46 = torch.ops.aten.view.default(mul_20, [16384, 3072]);  mul_20 = None
        permute_29 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg51_1, view_46, permute_29);  arg51_1 = view_46 = permute_29 = None
        view_47 = torch.ops.aten.view.default(addmm_17, [128, 128, 768]);  addmm_17 = None
        add_21 = torch.ops.aten.add.Tensor(view_47, add_19);  view_47 = add_19 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_21, getitem_25);  add_21 = getitem_25 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg52_1);  mul_21 = arg52_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
        view_48 = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_30 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_48, permute_30);  arg55_1 = view_48 = permute_30 = None
        view_49 = torch.ops.aten.view.default(addmm_18, [128, 128, 768]);  addmm_18 = None
        view_50 = torch.ops.aten.view.default(view_49, [128, -1, 12, 64]);  view_49 = None
        permute_31 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        view_51 = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_32 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_51, permute_32);  arg57_1 = view_51 = permute_32 = None
        view_52 = torch.ops.aten.view.default(addmm_19, [128, 128, 768]);  addmm_19 = None
        view_53 = torch.ops.aten.view.default(view_52, [128, -1, 12, 64]);  view_52 = None
        permute_33 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        view_54 = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_34 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_54, permute_34);  arg59_1 = view_54 = permute_34 = None
        view_55 = torch.ops.aten.view.default(addmm_20, [128, 128, 768]);  addmm_20 = None
        view_56 = torch.ops.aten.view.default(view_55, [128, -1, 12, 64]);  view_55 = None
        permute_35 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_4 = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_4, False);  permute_31 = permute_33 = permute_35 = expand_4 = None
        getitem_26 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_57 = torch.ops.aten.view.default(permute_36, [128, -1, 768]);  permute_36 = None
        view_58 = torch.ops.aten.view.default(view_57, [16384, 768]);  view_57 = None
        permute_37 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_58, permute_37);  arg61_1 = view_58 = permute_37 = None
        view_59 = torch.ops.aten.view.default(addmm_21, [128, 128, 768]);  addmm_21 = None
        add_24 = torch.ops.aten.add.Tensor(view_59, add_23);  view_59 = add_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_24, getitem_31);  add_24 = getitem_31 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, arg62_1);  mul_23 = arg62_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_24, arg63_1);  mul_24 = arg63_1 = None
        view_60 = torch.ops.aten.view.default(add_26, [16384, 768])
        permute_38 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg65_1, view_60, permute_38);  arg65_1 = view_60 = permute_38 = None
        view_61 = torch.ops.aten.view.default(addmm_22, [128, 128, 3072]);  addmm_22 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26 = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
        view_62 = torch.ops.aten.view.default(mul_27, [16384, 3072]);  mul_27 = None
        permute_39 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg67_1, view_62, permute_39);  arg67_1 = view_62 = permute_39 = None
        view_63 = torch.ops.aten.view.default(addmm_23, [128, 128, 768]);  addmm_23 = None
        add_28 = torch.ops.aten.add.Tensor(view_63, add_26);  view_63 = add_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_28, getitem_33);  add_28 = getitem_33 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg68_1);  mul_28 = arg68_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_29, arg69_1);  mul_29 = arg69_1 = None
        view_64 = torch.ops.aten.view.default(add_30, [16384, 768])
        permute_40 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg71_1, view_64, permute_40);  arg71_1 = view_64 = permute_40 = None
        view_65 = torch.ops.aten.view.default(addmm_24, [128, 128, 768]);  addmm_24 = None
        view_66 = torch.ops.aten.view.default(view_65, [128, -1, 12, 64]);  view_65 = None
        permute_41 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        view_67 = torch.ops.aten.view.default(add_30, [16384, 768])
        permute_42 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg73_1, view_67, permute_42);  arg73_1 = view_67 = permute_42 = None
        view_68 = torch.ops.aten.view.default(addmm_25, [128, 128, 768]);  addmm_25 = None
        view_69 = torch.ops.aten.view.default(view_68, [128, -1, 12, 64]);  view_68 = None
        permute_43 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        view_70 = torch.ops.aten.view.default(add_30, [16384, 768])
        permute_44 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg75_1, view_70, permute_44);  arg75_1 = view_70 = permute_44 = None
        view_71 = torch.ops.aten.view.default(addmm_26, [128, 128, 768]);  addmm_26 = None
        view_72 = torch.ops.aten.view.default(view_71, [128, -1, 12, 64]);  view_71 = None
        permute_45 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        expand_5 = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_5, False);  permute_41 = permute_43 = permute_45 = expand_5 = None
        getitem_34 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_73 = torch.ops.aten.view.default(permute_46, [128, -1, 768]);  permute_46 = None
        view_74 = torch.ops.aten.view.default(view_73, [16384, 768]);  view_73 = None
        permute_47 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg77_1, view_74, permute_47);  arg77_1 = view_74 = permute_47 = None
        view_75 = torch.ops.aten.view.default(addmm_27, [128, 128, 768]);  addmm_27 = None
        add_31 = torch.ops.aten.add.Tensor(view_75, add_30);  view_75 = add_30 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_31, getitem_39);  add_31 = getitem_39 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg78_1);  mul_30 = arg78_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
        view_76 = torch.ops.aten.view.default(add_33, [16384, 768])
        permute_48 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg81_1, view_76, permute_48);  arg81_1 = view_76 = permute_48 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [128, 128, 3072]);  addmm_28 = None
        mul_32 = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33 = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
        view_78 = torch.ops.aten.view.default(mul_34, [16384, 3072]);  mul_34 = None
        permute_49 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg83_1, view_78, permute_49);  arg83_1 = view_78 = permute_49 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [128, 128, 768]);  addmm_29 = None
        add_35 = torch.ops.aten.add.Tensor(view_79, add_33);  view_79 = add_33 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_35, getitem_41);  add_35 = getitem_41 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, arg84_1);  mul_35 = arg84_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_36, arg85_1);  mul_36 = arg85_1 = None
        view_80 = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_50 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg87_1, view_80, permute_50);  arg87_1 = view_80 = permute_50 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [128, 128, 768]);  addmm_30 = None
        view_82 = torch.ops.aten.view.default(view_81, [128, -1, 12, 64]);  view_81 = None
        permute_51 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        view_83 = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_52 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg89_1, view_83, permute_52);  arg89_1 = view_83 = permute_52 = None
        view_84 = torch.ops.aten.view.default(addmm_31, [128, 128, 768]);  addmm_31 = None
        view_85 = torch.ops.aten.view.default(view_84, [128, -1, 12, 64]);  view_84 = None
        permute_53 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        view_86 = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_54 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg91_1, view_86, permute_54);  arg91_1 = view_86 = permute_54 = None
        view_87 = torch.ops.aten.view.default(addmm_32, [128, 128, 768]);  addmm_32 = None
        view_88 = torch.ops.aten.view.default(view_87, [128, -1, 12, 64]);  view_87 = None
        permute_55 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        expand_6 = torch.ops.aten.expand.default(where, [128, 12, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_6, False);  permute_51 = permute_53 = permute_55 = expand_6 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_89 = torch.ops.aten.view.default(permute_56, [128, -1, 768]);  permute_56 = None
        view_90 = torch.ops.aten.view.default(view_89, [16384, 768]);  view_89 = None
        permute_57 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg93_1, view_90, permute_57);  arg93_1 = view_90 = permute_57 = None
        view_91 = torch.ops.aten.view.default(addmm_33, [128, 128, 768]);  addmm_33 = None
        add_38 = torch.ops.aten.add.Tensor(view_91, add_37);  view_91 = add_37 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_38, getitem_47);  add_38 = getitem_47 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, arg94_1);  mul_37 = arg94_1 = None
        add_40 = torch.ops.aten.add.Tensor(mul_38, arg95_1);  mul_38 = arg95_1 = None
        view_92 = torch.ops.aten.view.default(add_40, [16384, 768])
        permute_58 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg97_1, view_92, permute_58);  arg97_1 = view_92 = permute_58 = None
        view_93 = torch.ops.aten.view.default(addmm_34, [128, 128, 3072]);  addmm_34 = None
        mul_39 = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
        view_94 = torch.ops.aten.view.default(mul_41, [16384, 3072]);  mul_41 = None
        permute_59 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg99_1, view_94, permute_59);  arg99_1 = view_94 = permute_59 = None
        view_95 = torch.ops.aten.view.default(addmm_35, [128, 128, 768]);  addmm_35 = None
        add_42 = torch.ops.aten.add.Tensor(view_95, add_40);  view_95 = add_40 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_42, getitem_49);  add_42 = getitem_49 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg100_1);  mul_42 = arg100_1 = None
        add_44 = torch.ops.aten.add.Tensor(mul_43, arg101_1);  mul_43 = arg101_1 = None
        view_96 = torch.ops.aten.view.default(add_44, [16384, 768]);  add_44 = None
        permute_60 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg103_1, view_96, permute_60);  arg103_1 = view_96 = permute_60 = None
        view_97 = torch.ops.aten.view.default(addmm_36, [128, 128, 768]);  addmm_36 = None
        mul_44 = torch.ops.aten.mul.Tensor(view_97, 0.5)
        mul_45 = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476);  view_97 = None
        erf_6 = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_45 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_44, add_45);  mul_44 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(mul_46, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_13[0]
        getitem_51 = var_mean_13[1];  var_mean_13 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_14 = torch.ops.aten.sub.Tensor(mul_46, getitem_51);  mul_46 = getitem_51 = None
        mul_47 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, arg104_1);  mul_47 = arg104_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_48, arg105_1);  mul_48 = arg105_1 = None
        view_98 = torch.ops.aten.view.default(add_47, [16384, 768]);  add_47 = None
        permute_61 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        full_default_2 = torch.ops.aten.full.default([768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_61, full_default_2], 1);  permute_61 = full_default_2 = None
        full_default_3 = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default_1 = torch.ops.aten.cat.default([arg106_1, full_default_3]);  arg106_1 = full_default_3 = None
        addmm_default = torch.ops.aten.addmm.default(cat_default_1, view_98, cat_default);  cat_default_1 = view_98 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_99 = torch.ops.aten.view.default(slice_tensor, [128, 128, 30522]);  slice_tensor = None
        view_100 = torch.ops.aten.view.default(view_99, [-1, 30522])
        view_101 = torch.ops.aten.view.default(arg107_1, [-1]);  arg107_1 = None
        amax = torch.ops.aten.amax.default(view_100, [1], True)
        sub_15 = torch.ops.aten.sub.Tensor(view_100, amax);  view_100 = amax = None
        exp = torch.ops.aten.exp.default(sub_15)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_16 = torch.ops.aten.sub.Tensor(sub_15, log);  sub_15 = log = None
        ne = torch.ops.aten.ne.Scalar(view_101, -100)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_101, full_default);  ne = full_default = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_16, 1, unsqueeze_2);  sub_16 = unsqueeze_2 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_101, -100)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_101, -100);  view_101 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, view_99)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (128, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 93763584, device=device(type='cuda', index=0))
    reader.tensor(buf1, (30522, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768, 768), is_leaf=True)  # arg6_1
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
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3072, 768), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf17, (3072,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 3072), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768, 768), is_leaf=True)  # arg22_1
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
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf32, (3072, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf33, (3072,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768, 3072), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 768), is_leaf=True)  # arg38_1
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
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf48, (3072, 768), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf49, (3072,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768, 3072), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 768), is_leaf=True)  # arg54_1
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
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf64, (3072, 768), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf65, (3072,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 3072), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768, 768), is_leaf=True)  # arg70_1
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
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf80, (3072, 768), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf81, (3072,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768, 3072), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 768), is_leaf=True)  # arg86_1
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
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf96, (3072, 768), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf97, (3072,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768, 3072), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 768), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 122088, device=device(type='cuda', index=0))
    reader.tensor(buf106, (30522,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf107, (128, 128), dtype=torch.int64, is_leaf=True)  # arg107_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)