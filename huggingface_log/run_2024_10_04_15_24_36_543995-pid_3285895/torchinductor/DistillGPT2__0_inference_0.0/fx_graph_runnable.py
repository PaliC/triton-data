
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 512]);  arg0_1 = None
        iota = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        embedding = torch.ops.aten.embedding.default(arg1_1, view);  view = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        full_default = torch.ops.aten.full.default([512, 512], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_1 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1 = torch.ops.aten.add.Tensor(iota_1, 1)
        view_1 = torch.ops.aten.view.default(add_1, [512, 1]);  add_1 = None
        lt = torch.ops.aten.lt.Tensor(iota_1, view_1);  iota_1 = view_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        view_2 = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        addmm = torch.ops.aten.addmm.default(arg5_1, view_2, arg6_1);  arg5_1 = view_2 = arg6_1 = None
        view_3 = torch.ops.aten.view.default(addmm, [16, 512, 2304]);  addmm = None
        split = torch.ops.aten.split.Tensor(view_3, 768, 2);  view_3 = None
        getitem_2 = split[0]
        getitem_3 = split[1]
        getitem_4 = split[2];  split = None
        view_4 = torch.ops.aten.view.default(getitem_2, [16, 512, 12, 64]);  getitem_2 = None
        permute = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5 = torch.ops.aten.view.default(getitem_3, [16, 512, 12, 64]);  getitem_3 = None
        permute_1 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6 = torch.ops.aten.view.default(getitem_4, [16, 512, 12, 64]);  getitem_4 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_4, [16, 1, 512, 512]);  unsqueeze_4 = None
        expand_3 = torch.ops.aten.expand.default(expand_2, [16, 12, 512, 512]);  expand_2 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute, permute_1, permute_2, expand_3, False);  permute = expand_3 = None
        getitem_5 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_3 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_7 = torch.ops.aten.view.default(permute_3, [16, 512, 768]);  permute_3 = None
        view_8 = torch.ops.aten.view.default(view_7, [-1, 768]);  view_7 = None
        addmm_1 = torch.ops.aten.addmm.default(arg7_1, view_8, arg8_1);  arg7_1 = view_8 = arg8_1 = None
        view_9 = torch.ops.aten.view.default(addmm_1, [16, 512, 768]);  addmm_1 = None
        add_4 = torch.ops.aten.add.Tensor(view_9, add);  view_9 = add = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_1[0]
        getitem_10 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_4, getitem_10);  getitem_10 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        view_10 = torch.ops.aten.view.default(add_6, [-1, 768]);  add_6 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_10, arg12_1);  arg11_1 = view_10 = arg12_1 = None
        view_11 = torch.ops.aten.view.default(addmm_2, [16, 512, 3072]);  addmm_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_11, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7 = torch.ops.aten.add.Tensor(view_11, mul_5);  view_11 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        view_12 = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_12, arg14_1);  arg13_1 = view_12 = arg14_1 = None
        view_13 = torch.ops.aten.view.default(addmm_3, [16, 512, 768]);  addmm_3 = None
        add_9 = torch.ops.aten.add.Tensor(add_4, view_13);  add_4 = view_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_2[0]
        getitem_12 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_9, getitem_12);  getitem_12 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        view_14 = torch.ops.aten.view.default(add_11, [-1, 768]);  add_11 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_14, arg18_1);  arg17_1 = view_14 = arg18_1 = None
        view_15 = torch.ops.aten.view.default(addmm_4, [16, 512, 2304]);  addmm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_15, 768, 2);  view_15 = None
        getitem_13 = split_1[0]
        getitem_14 = split_1[1]
        getitem_15 = split_1[2];  split_1 = None
        view_16 = torch.ops.aten.view.default(getitem_13, [16, 512, 12, 64]);  getitem_13 = None
        permute_4 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        view_17 = torch.ops.aten.view.default(getitem_14, [16, 512, 12, 64]);  getitem_14 = None
        permute_5 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        view_18 = torch.ops.aten.view.default(getitem_15, [16, 512, 12, 64]);  getitem_15 = None
        permute_6 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_6, [16, 1, 512, 512]);  unsqueeze_6 = None
        expand_6 = torch.ops.aten.expand.default(expand_5, [16, 12, 512, 512]);  expand_5 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_4, permute_5, permute_6, expand_6, False);  permute_4 = expand_6 = None
        getitem_16 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_7 = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        view_19 = torch.ops.aten.view.default(permute_7, [16, 512, 768]);  permute_7 = None
        view_20 = torch.ops.aten.view.default(view_19, [-1, 768]);  view_19 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_20, arg20_1);  arg19_1 = view_20 = arg20_1 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [16, 512, 768]);  addmm_5 = None
        add_12 = torch.ops.aten.add.Tensor(view_21, add_9);  view_21 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_3[0]
        getitem_21 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_12, getitem_21);  getitem_21 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        view_22 = torch.ops.aten.view.default(add_14, [-1, 768]);  add_14 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_22, arg24_1);  arg23_1 = view_22 = arg24_1 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [16, 512, 3072]);  addmm_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_23, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_13 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15 = torch.ops.aten.add.Tensor(view_23, mul_13);  view_23 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_24 = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_24, arg26_1);  arg25_1 = view_24 = arg26_1 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [16, 512, 768]);  addmm_7 = None
        add_17 = torch.ops.aten.add.Tensor(add_12, view_25);  add_12 = view_25 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_4[0]
        getitem_23 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        view_26 = torch.ops.aten.view.default(add_19, [-1, 768]);  add_19 = None
        addmm_8 = torch.ops.aten.addmm.default(arg29_1, view_26, arg30_1);  arg29_1 = view_26 = arg30_1 = None
        view_27 = torch.ops.aten.view.default(addmm_8, [16, 512, 2304]);  addmm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_27, 768, 2);  view_27 = None
        getitem_24 = split_2[0]
        getitem_25 = split_2[1]
        getitem_26 = split_2[2];  split_2 = None
        view_28 = torch.ops.aten.view.default(getitem_24, [16, 512, 12, 64]);  getitem_24 = None
        permute_8 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        view_29 = torch.ops.aten.view.default(getitem_25, [16, 512, 12, 64]);  getitem_25 = None
        permute_9 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        view_30 = torch.ops.aten.view.default(getitem_26, [16, 512, 12, 64]);  getitem_26 = None
        permute_10 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_8, [16, 1, 512, 512]);  unsqueeze_8 = None
        expand_9 = torch.ops.aten.expand.default(expand_8, [16, 12, 512, 512]);  expand_8 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_8, permute_9, permute_10, expand_9, False);  permute_8 = expand_9 = None
        getitem_27 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_11 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        view_31 = torch.ops.aten.view.default(permute_11, [16, 512, 768]);  permute_11 = None
        view_32 = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
        addmm_9 = torch.ops.aten.addmm.default(arg31_1, view_32, arg32_1);  arg31_1 = view_32 = arg32_1 = None
        view_33 = torch.ops.aten.view.default(addmm_9, [16, 512, 768]);  addmm_9 = None
        add_20 = torch.ops.aten.add.Tensor(view_33, add_17);  view_33 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_31 = var_mean_5[0]
        getitem_32 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_20, getitem_32);  getitem_32 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        view_34 = torch.ops.aten.view.default(add_22, [-1, 768]);  add_22 = None
        addmm_10 = torch.ops.aten.addmm.default(arg35_1, view_34, arg36_1);  arg35_1 = view_34 = arg36_1 = None
        view_35 = torch.ops.aten.view.default(addmm_10, [16, 512, 3072]);  addmm_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_35, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
        mul_21 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23 = torch.ops.aten.add.Tensor(view_35, mul_21);  view_35 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        view_36 = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11 = torch.ops.aten.addmm.default(arg37_1, view_36, arg38_1);  arg37_1 = view_36 = arg38_1 = None
        view_37 = torch.ops.aten.view.default(addmm_11, [16, 512, 768]);  addmm_11 = None
        add_25 = torch.ops.aten.add.Tensor(add_20, view_37);  add_20 = view_37 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_33 = var_mean_6[0]
        getitem_34 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_25, getitem_34);  getitem_34 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        view_38 = torch.ops.aten.view.default(add_27, [-1, 768]);  add_27 = None
        addmm_12 = torch.ops.aten.addmm.default(arg41_1, view_38, arg42_1);  arg41_1 = view_38 = arg42_1 = None
        view_39 = torch.ops.aten.view.default(addmm_12, [16, 512, 2304]);  addmm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_39, 768, 2);  view_39 = None
        getitem_35 = split_3[0]
        getitem_36 = split_3[1]
        getitem_37 = split_3[2];  split_3 = None
        view_40 = torch.ops.aten.view.default(getitem_35, [16, 512, 12, 64]);  getitem_35 = None
        permute_12 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        view_41 = torch.ops.aten.view.default(getitem_36, [16, 512, 12, 64]);  getitem_36 = None
        permute_13 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        view_42 = torch.ops.aten.view.default(getitem_37, [16, 512, 12, 64]);  getitem_37 = None
        permute_14 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_10, [16, 1, 512, 512]);  unsqueeze_10 = None
        expand_12 = torch.ops.aten.expand.default(expand_11, [16, 12, 512, 512]);  expand_11 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, permute_13, permute_14, expand_12, False);  permute_12 = expand_12 = None
        getitem_38 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_15 = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        view_43 = torch.ops.aten.view.default(permute_15, [16, 512, 768]);  permute_15 = None
        view_44 = torch.ops.aten.view.default(view_43, [-1, 768]);  view_43 = None
        addmm_13 = torch.ops.aten.addmm.default(arg43_1, view_44, arg44_1);  arg43_1 = view_44 = arg44_1 = None
        view_45 = torch.ops.aten.view.default(addmm_13, [16, 512, 768]);  addmm_13 = None
        add_28 = torch.ops.aten.add.Tensor(view_45, add_25);  view_45 = add_25 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_7[0]
        getitem_43 = var_mean_7[1];  var_mean_7 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_28, getitem_43);  getitem_43 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        view_46 = torch.ops.aten.view.default(add_30, [-1, 768]);  add_30 = None
        addmm_14 = torch.ops.aten.addmm.default(arg47_1, view_46, arg48_1);  arg47_1 = view_46 = arg48_1 = None
        view_47 = torch.ops.aten.view.default(addmm_14, [16, 512, 3072]);  addmm_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_29 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31 = torch.ops.aten.add.Tensor(view_47, mul_29);  view_47 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        view_48 = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15 = torch.ops.aten.addmm.default(arg49_1, view_48, arg50_1);  arg49_1 = view_48 = arg50_1 = None
        view_49 = torch.ops.aten.view.default(addmm_15, [16, 512, 768]);  addmm_15 = None
        add_33 = torch.ops.aten.add.Tensor(add_28, view_49);  add_28 = view_49 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_8[0]
        getitem_45 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_33, getitem_45);  getitem_45 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        view_50 = torch.ops.aten.view.default(add_35, [-1, 768]);  add_35 = None
        addmm_16 = torch.ops.aten.addmm.default(arg53_1, view_50, arg54_1);  arg53_1 = view_50 = arg54_1 = None
        view_51 = torch.ops.aten.view.default(addmm_16, [16, 512, 2304]);  addmm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_51, 768, 2);  view_51 = None
        getitem_46 = split_4[0]
        getitem_47 = split_4[1]
        getitem_48 = split_4[2];  split_4 = None
        view_52 = torch.ops.aten.view.default(getitem_46, [16, 512, 12, 64]);  getitem_46 = None
        permute_16 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        view_53 = torch.ops.aten.view.default(getitem_47, [16, 512, 12, 64]);  getitem_47 = None
        permute_17 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        view_54 = torch.ops.aten.view.default(getitem_48, [16, 512, 12, 64]);  getitem_48 = None
        permute_18 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 1);  unsqueeze_11 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_12, [16, 1, 512, 512]);  unsqueeze_12 = None
        expand_15 = torch.ops.aten.expand.default(expand_14, [16, 12, 512, 512]);  expand_14 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_16, permute_17, permute_18, expand_15, False);  permute_16 = expand_15 = None
        getitem_49 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_19 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        view_55 = torch.ops.aten.view.default(permute_19, [16, 512, 768]);  permute_19 = None
        view_56 = torch.ops.aten.view.default(view_55, [-1, 768]);  view_55 = None
        addmm_17 = torch.ops.aten.addmm.default(arg55_1, view_56, arg56_1);  arg55_1 = view_56 = arg56_1 = None
        view_57 = torch.ops.aten.view.default(addmm_17, [16, 512, 768]);  addmm_17 = None
        add_36 = torch.ops.aten.add.Tensor(view_57, add_33);  view_57 = add_33 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_53 = var_mean_9[0]
        getitem_54 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_36, getitem_54);  getitem_54 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        view_58 = torch.ops.aten.view.default(add_38, [-1, 768]);  add_38 = None
        addmm_18 = torch.ops.aten.addmm.default(arg59_1, view_58, arg60_1);  arg59_1 = view_58 = arg60_1 = None
        view_59 = torch.ops.aten.view.default(addmm_18, [16, 512, 3072]);  addmm_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_59, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_59, 3.0)
        mul_37 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39 = torch.ops.aten.add.Tensor(view_59, mul_37);  view_59 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        view_60 = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19 = torch.ops.aten.addmm.default(arg61_1, view_60, arg62_1);  arg61_1 = view_60 = arg62_1 = None
        view_61 = torch.ops.aten.view.default(addmm_19, [16, 512, 768]);  addmm_19 = None
        add_41 = torch.ops.aten.add.Tensor(add_36, view_61);  add_36 = view_61 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_55 = var_mean_10[0]
        getitem_56 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_41, getitem_56);  getitem_56 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        view_62 = torch.ops.aten.view.default(add_43, [-1, 768]);  add_43 = None
        addmm_20 = torch.ops.aten.addmm.default(arg65_1, view_62, arg66_1);  arg65_1 = view_62 = arg66_1 = None
        view_63 = torch.ops.aten.view.default(addmm_20, [16, 512, 2304]);  addmm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_63, 768, 2);  view_63 = None
        getitem_57 = split_5[0]
        getitem_58 = split_5[1]
        getitem_59 = split_5[2];  split_5 = None
        view_64 = torch.ops.aten.view.default(getitem_57, [16, 512, 12, 64]);  getitem_57 = None
        permute_20 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        view_65 = torch.ops.aten.view.default(getitem_58, [16, 512, 12, 64]);  getitem_58 = None
        permute_21 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        view_66 = torch.ops.aten.view.default(getitem_59, [16, 512, 12, 64]);  getitem_59 = None
        permute_22 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        expand_17 = torch.ops.aten.expand.default(unsqueeze_14, [16, 1, 512, 512]);  unsqueeze_14 = None
        expand_18 = torch.ops.aten.expand.default(expand_17, [16, 12, 512, 512]);  expand_17 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_20, permute_21, permute_22, expand_18, False);  permute_20 = expand_18 = None
        getitem_60 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_23 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        view_67 = torch.ops.aten.view.default(permute_23, [16, 512, 768]);  permute_23 = None
        view_68 = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
        addmm_21 = torch.ops.aten.addmm.default(arg67_1, view_68, arg68_1);  arg67_1 = view_68 = arg68_1 = None
        view_69 = torch.ops.aten.view.default(addmm_21, [16, 512, 768]);  addmm_21 = None
        add_44 = torch.ops.aten.add.Tensor(view_69, add_41);  view_69 = add_41 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_11[0]
        getitem_65 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_44, getitem_65);  getitem_65 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_46 = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        view_70 = torch.ops.aten.view.default(add_46, [-1, 768]);  add_46 = None
        addmm_22 = torch.ops.aten.addmm.default(arg71_1, view_70, arg72_1);  arg71_1 = view_70 = arg72_1 = None
        view_71 = torch.ops.aten.view.default(addmm_22, [16, 512, 3072]);  addmm_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(view_71, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
        mul_45 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47 = torch.ops.aten.add.Tensor(view_71, mul_45);  view_71 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        view_72 = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23 = torch.ops.aten.addmm.default(arg73_1, view_72, arg74_1);  arg73_1 = view_72 = arg74_1 = None
        view_73 = torch.ops.aten.view.default(addmm_23, [16, 512, 768]);  addmm_23 = None
        add_49 = torch.ops.aten.add.Tensor(add_44, view_73);  add_44 = view_73 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_12[0]
        getitem_67 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_49, getitem_67);  add_49 = getitem_67 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        view_74 = torch.ops.aten.view.default(add_51, [-1, 512, 768]);  add_51 = None
        permute_24 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_75 = torch.ops.aten.view.default(view_74, [8192, 768]);  view_74 = None
        full_default_4 = torch.ops.aten.full.default([768, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_24, full_default_4], 1);  permute_24 = full_default_4 = None
        mm_default = torch.ops.aten.mm.default(view_75, cat_default);  view_75 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_76 = torch.ops.aten.view.default(slice_tensor, [16, 512, 50257]);  slice_tensor = None
        slice_15 = torch.ops.aten.slice.Tensor(view_76, 1, 0, -1)
        clone_13 = torch.ops.aten.clone.default(slice_15, memory_format = torch.contiguous_format);  slice_15 = None
        slice_17 = torch.ops.aten.slice.Tensor(arg77_1, 1, 1, 9223372036854775807);  arg77_1 = None
        clone_14 = torch.ops.aten.clone.default(slice_17, memory_format = torch.contiguous_format);  slice_17 = None
        view_77 = torch.ops.aten.view.default(clone_13, [-1, 50257]);  clone_13 = None
        view_78 = torch.ops.aten.view.default(clone_14, [-1]);  clone_14 = None
        amax = torch.ops.aten.amax.default(view_77, [1], True)
        sub_13 = torch.ops.aten.sub.Tensor(view_77, amax);  view_77 = amax = None
        exp = torch.ops.aten.exp.default(sub_13)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_14 = torch.ops.aten.sub.Tensor(sub_13, log);  sub_13 = log = None
        ne = torch.ops.aten.ne.Scalar(view_78, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_78, full_default_2);  ne = full_default_2 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_14, 1, unsqueeze_15);  sub_14 = unsqueeze_15 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_78, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_78, -100);  view_78 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_76, permute_1, permute_2, permute_5, permute_6, permute_9, permute_10, permute_13, permute_14, permute_17, permute_18, permute_21, permute_22)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 154389504, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50257, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1024, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2304,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768, 2304), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf11, (3072,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768, 3072), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf14, (3072, 768), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (2304,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 2304), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 768), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf23, (3072,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768, 3072), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (3072, 768), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf29, (2304,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 2304), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf35, (3072,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768, 3072), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (3072, 768), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf41, (2304,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 2304), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768, 768), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768, 3072), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf50, (3072, 768), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf53, (2304,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 2304), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf59, (3072,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768, 3072), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf62, (3072, 768), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf65, (2304,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 2304), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768, 768), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf71, (3072,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768, 3072), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf74, (3072, 768), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf77, (16, 512), dtype=torch.int64, is_leaf=True)  # arg77_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)