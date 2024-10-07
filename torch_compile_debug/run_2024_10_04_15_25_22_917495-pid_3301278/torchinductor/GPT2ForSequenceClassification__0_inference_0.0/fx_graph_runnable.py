
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 1024])
        iota = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        embedding = torch.ops.aten.embedding.default(arg1_1, view);  arg1_1 = view = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        full_default = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_1 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1 = torch.ops.aten.add.Tensor(iota_1, 1)
        view_1 = torch.ops.aten.view.default(add_1, [1024, 1]);  add_1 = None
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
        view_3 = torch.ops.aten.view.default(addmm, [4, 1024, 2304]);  addmm = None
        split = torch.ops.aten.split.Tensor(view_3, 768, 2);  view_3 = None
        getitem_2 = split[0]
        getitem_3 = split[1]
        getitem_4 = split[2];  split = None
        view_4 = torch.ops.aten.view.default(getitem_2, [4, 1024, 12, 64]);  getitem_2 = None
        permute = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5 = torch.ops.aten.view.default(getitem_3, [4, 1024, 12, 64]);  getitem_3 = None
        permute_1 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6 = torch.ops.aten.view.default(getitem_4, [4, 1024, 12, 64]);  getitem_4 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_4, [4, 1, 1024, 1024]);  unsqueeze_4 = None
        expand_3 = torch.ops.aten.expand.default(expand_2, [4, 12, 1024, 1024]);  expand_2 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute, permute_1, permute_2, expand_3, False);  permute = expand_3 = None
        getitem_5 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_3 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_7 = torch.ops.aten.view.default(permute_3, [4, 1024, 768]);  permute_3 = None
        view_8 = torch.ops.aten.view.default(view_7, [-1, 768]);  view_7 = None
        addmm_1 = torch.ops.aten.addmm.default(arg7_1, view_8, arg8_1);  arg7_1 = view_8 = arg8_1 = None
        view_9 = torch.ops.aten.view.default(addmm_1, [4, 1024, 768]);  addmm_1 = None
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
        view_11 = torch.ops.aten.view.default(addmm_2, [4, 1024, 3072]);  addmm_2 = None
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
        view_13 = torch.ops.aten.view.default(addmm_3, [4, 1024, 768]);  addmm_3 = None
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
        view_15 = torch.ops.aten.view.default(addmm_4, [4, 1024, 2304]);  addmm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_15, 768, 2);  view_15 = None
        getitem_13 = split_1[0]
        getitem_14 = split_1[1]
        getitem_15 = split_1[2];  split_1 = None
        view_16 = torch.ops.aten.view.default(getitem_13, [4, 1024, 12, 64]);  getitem_13 = None
        permute_4 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        view_17 = torch.ops.aten.view.default(getitem_14, [4, 1024, 12, 64]);  getitem_14 = None
        permute_5 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        view_18 = torch.ops.aten.view.default(getitem_15, [4, 1024, 12, 64]);  getitem_15 = None
        permute_6 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_6, [4, 1, 1024, 1024]);  unsqueeze_6 = None
        expand_6 = torch.ops.aten.expand.default(expand_5, [4, 12, 1024, 1024]);  expand_5 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_4, permute_5, permute_6, expand_6, False);  permute_4 = expand_6 = None
        getitem_16 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_7 = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        view_19 = torch.ops.aten.view.default(permute_7, [4, 1024, 768]);  permute_7 = None
        view_20 = torch.ops.aten.view.default(view_19, [-1, 768]);  view_19 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_20, arg20_1);  arg19_1 = view_20 = arg20_1 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [4, 1024, 768]);  addmm_5 = None
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
        view_23 = torch.ops.aten.view.default(addmm_6, [4, 1024, 3072]);  addmm_6 = None
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
        view_25 = torch.ops.aten.view.default(addmm_7, [4, 1024, 768]);  addmm_7 = None
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
        view_27 = torch.ops.aten.view.default(addmm_8, [4, 1024, 2304]);  addmm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_27, 768, 2);  view_27 = None
        getitem_24 = split_2[0]
        getitem_25 = split_2[1]
        getitem_26 = split_2[2];  split_2 = None
        view_28 = torch.ops.aten.view.default(getitem_24, [4, 1024, 12, 64]);  getitem_24 = None
        permute_8 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        view_29 = torch.ops.aten.view.default(getitem_25, [4, 1024, 12, 64]);  getitem_25 = None
        permute_9 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        view_30 = torch.ops.aten.view.default(getitem_26, [4, 1024, 12, 64]);  getitem_26 = None
        permute_10 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_8, [4, 1, 1024, 1024]);  unsqueeze_8 = None
        expand_9 = torch.ops.aten.expand.default(expand_8, [4, 12, 1024, 1024]);  expand_8 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_8, permute_9, permute_10, expand_9, False);  permute_8 = expand_9 = None
        getitem_27 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_11 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        view_31 = torch.ops.aten.view.default(permute_11, [4, 1024, 768]);  permute_11 = None
        view_32 = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
        addmm_9 = torch.ops.aten.addmm.default(arg31_1, view_32, arg32_1);  arg31_1 = view_32 = arg32_1 = None
        view_33 = torch.ops.aten.view.default(addmm_9, [4, 1024, 768]);  addmm_9 = None
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
        view_35 = torch.ops.aten.view.default(addmm_10, [4, 1024, 3072]);  addmm_10 = None
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
        view_37 = torch.ops.aten.view.default(addmm_11, [4, 1024, 768]);  addmm_11 = None
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
        view_39 = torch.ops.aten.view.default(addmm_12, [4, 1024, 2304]);  addmm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_39, 768, 2);  view_39 = None
        getitem_35 = split_3[0]
        getitem_36 = split_3[1]
        getitem_37 = split_3[2];  split_3 = None
        view_40 = torch.ops.aten.view.default(getitem_35, [4, 1024, 12, 64]);  getitem_35 = None
        permute_12 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        view_41 = torch.ops.aten.view.default(getitem_36, [4, 1024, 12, 64]);  getitem_36 = None
        permute_13 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        view_42 = torch.ops.aten.view.default(getitem_37, [4, 1024, 12, 64]);  getitem_37 = None
        permute_14 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_10, [4, 1, 1024, 1024]);  unsqueeze_10 = None
        expand_12 = torch.ops.aten.expand.default(expand_11, [4, 12, 1024, 1024]);  expand_11 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, permute_13, permute_14, expand_12, False);  permute_12 = expand_12 = None
        getitem_38 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_15 = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        view_43 = torch.ops.aten.view.default(permute_15, [4, 1024, 768]);  permute_15 = None
        view_44 = torch.ops.aten.view.default(view_43, [-1, 768]);  view_43 = None
        addmm_13 = torch.ops.aten.addmm.default(arg43_1, view_44, arg44_1);  arg43_1 = view_44 = arg44_1 = None
        view_45 = torch.ops.aten.view.default(addmm_13, [4, 1024, 768]);  addmm_13 = None
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
        view_47 = torch.ops.aten.view.default(addmm_14, [4, 1024, 3072]);  addmm_14 = None
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
        view_49 = torch.ops.aten.view.default(addmm_15, [4, 1024, 768]);  addmm_15 = None
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
        view_51 = torch.ops.aten.view.default(addmm_16, [4, 1024, 2304]);  addmm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_51, 768, 2);  view_51 = None
        getitem_46 = split_4[0]
        getitem_47 = split_4[1]
        getitem_48 = split_4[2];  split_4 = None
        view_52 = torch.ops.aten.view.default(getitem_46, [4, 1024, 12, 64]);  getitem_46 = None
        permute_16 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        view_53 = torch.ops.aten.view.default(getitem_47, [4, 1024, 12, 64]);  getitem_47 = None
        permute_17 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        view_54 = torch.ops.aten.view.default(getitem_48, [4, 1024, 12, 64]);  getitem_48 = None
        permute_18 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 1);  unsqueeze_11 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_12, [4, 1, 1024, 1024]);  unsqueeze_12 = None
        expand_15 = torch.ops.aten.expand.default(expand_14, [4, 12, 1024, 1024]);  expand_14 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_16, permute_17, permute_18, expand_15, False);  permute_16 = expand_15 = None
        getitem_49 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_19 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        view_55 = torch.ops.aten.view.default(permute_19, [4, 1024, 768]);  permute_19 = None
        view_56 = torch.ops.aten.view.default(view_55, [-1, 768]);  view_55 = None
        addmm_17 = torch.ops.aten.addmm.default(arg55_1, view_56, arg56_1);  arg55_1 = view_56 = arg56_1 = None
        view_57 = torch.ops.aten.view.default(addmm_17, [4, 1024, 768]);  addmm_17 = None
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
        view_59 = torch.ops.aten.view.default(addmm_18, [4, 1024, 3072]);  addmm_18 = None
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
        view_61 = torch.ops.aten.view.default(addmm_19, [4, 1024, 768]);  addmm_19 = None
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
        view_63 = torch.ops.aten.view.default(addmm_20, [4, 1024, 2304]);  addmm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_63, 768, 2);  view_63 = None
        getitem_57 = split_5[0]
        getitem_58 = split_5[1]
        getitem_59 = split_5[2];  split_5 = None
        view_64 = torch.ops.aten.view.default(getitem_57, [4, 1024, 12, 64]);  getitem_57 = None
        permute_20 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        view_65 = torch.ops.aten.view.default(getitem_58, [4, 1024, 12, 64]);  getitem_58 = None
        permute_21 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        view_66 = torch.ops.aten.view.default(getitem_59, [4, 1024, 12, 64]);  getitem_59 = None
        permute_22 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        expand_17 = torch.ops.aten.expand.default(unsqueeze_14, [4, 1, 1024, 1024]);  unsqueeze_14 = None
        expand_18 = torch.ops.aten.expand.default(expand_17, [4, 12, 1024, 1024]);  expand_17 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_20, permute_21, permute_22, expand_18, False);  permute_20 = expand_18 = None
        getitem_60 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_23 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        view_67 = torch.ops.aten.view.default(permute_23, [4, 1024, 768]);  permute_23 = None
        view_68 = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
        addmm_21 = torch.ops.aten.addmm.default(arg67_1, view_68, arg68_1);  arg67_1 = view_68 = arg68_1 = None
        view_69 = torch.ops.aten.view.default(addmm_21, [4, 1024, 768]);  addmm_21 = None
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
        view_71 = torch.ops.aten.view.default(addmm_22, [4, 1024, 3072]);  addmm_22 = None
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
        view_73 = torch.ops.aten.view.default(addmm_23, [4, 1024, 768]);  addmm_23 = None
        add_49 = torch.ops.aten.add.Tensor(add_44, view_73);  add_44 = view_73 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_12[0]
        getitem_67 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_49, getitem_67);  getitem_67 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        view_74 = torch.ops.aten.view.default(add_51, [-1, 768]);  add_51 = None
        addmm_24 = torch.ops.aten.addmm.default(arg77_1, view_74, arg78_1);  arg77_1 = view_74 = arg78_1 = None
        view_75 = torch.ops.aten.view.default(addmm_24, [4, 1024, 2304]);  addmm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_75, 768, 2);  view_75 = None
        getitem_68 = split_6[0]
        getitem_69 = split_6[1]
        getitem_70 = split_6[2];  split_6 = None
        view_76 = torch.ops.aten.view.default(getitem_68, [4, 1024, 12, 64]);  getitem_68 = None
        permute_24 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        view_77 = torch.ops.aten.view.default(getitem_69, [4, 1024, 12, 64]);  getitem_69 = None
        permute_25 = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        view_78 = torch.ops.aten.view.default(getitem_70, [4, 1024, 12, 64]);  getitem_70 = None
        permute_26 = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(unsqueeze_15, 1);  unsqueeze_15 = None
        expand_20 = torch.ops.aten.expand.default(unsqueeze_16, [4, 1, 1024, 1024]);  unsqueeze_16 = None
        expand_21 = torch.ops.aten.expand.default(expand_20, [4, 12, 1024, 1024]);  expand_20 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_24, permute_25, permute_26, expand_21, False);  permute_24 = expand_21 = None
        getitem_71 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_27 = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        view_79 = torch.ops.aten.view.default(permute_27, [4, 1024, 768]);  permute_27 = None
        view_80 = torch.ops.aten.view.default(view_79, [-1, 768]);  view_79 = None
        addmm_25 = torch.ops.aten.addmm.default(arg79_1, view_80, arg80_1);  arg79_1 = view_80 = arg80_1 = None
        view_81 = torch.ops.aten.view.default(addmm_25, [4, 1024, 768]);  addmm_25 = None
        add_52 = torch.ops.aten.add.Tensor(view_81, add_49);  view_81 = add_49 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_75 = var_mean_13[0]
        getitem_76 = var_mean_13[1];  var_mean_13 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_52, getitem_76);  getitem_76 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None
        view_82 = torch.ops.aten.view.default(add_54, [-1, 768]);  add_54 = None
        addmm_26 = torch.ops.aten.addmm.default(arg83_1, view_82, arg84_1);  arg83_1 = view_82 = arg84_1 = None
        view_83 = torch.ops.aten.view.default(addmm_26, [4, 1024, 3072]);  addmm_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_83, 0.5)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(view_83, 3.0)
        mul_53 = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55 = torch.ops.aten.add.Tensor(view_83, mul_53);  view_83 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        view_84 = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27 = torch.ops.aten.addmm.default(arg85_1, view_84, arg86_1);  arg85_1 = view_84 = arg86_1 = None
        view_85 = torch.ops.aten.view.default(addmm_27, [4, 1024, 768]);  addmm_27 = None
        add_57 = torch.ops.aten.add.Tensor(add_52, view_85);  add_52 = view_85 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_77 = var_mean_14[0]
        getitem_78 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_57, getitem_78);  getitem_78 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None
        view_86 = torch.ops.aten.view.default(add_59, [-1, 768]);  add_59 = None
        addmm_28 = torch.ops.aten.addmm.default(arg89_1, view_86, arg90_1);  arg89_1 = view_86 = arg90_1 = None
        view_87 = torch.ops.aten.view.default(addmm_28, [4, 1024, 2304]);  addmm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_87, 768, 2);  view_87 = None
        getitem_79 = split_7[0]
        getitem_80 = split_7[1]
        getitem_81 = split_7[2];  split_7 = None
        view_88 = torch.ops.aten.view.default(getitem_79, [4, 1024, 12, 64]);  getitem_79 = None
        permute_28 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        view_89 = torch.ops.aten.view.default(getitem_80, [4, 1024, 12, 64]);  getitem_80 = None
        permute_29 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        view_90 = torch.ops.aten.view.default(getitem_81, [4, 1024, 12, 64]);  getitem_81 = None
        permute_30 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 1);  unsqueeze_17 = None
        expand_23 = torch.ops.aten.expand.default(unsqueeze_18, [4, 1, 1024, 1024]);  unsqueeze_18 = None
        expand_24 = torch.ops.aten.expand.default(expand_23, [4, 12, 1024, 1024]);  expand_23 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_28, permute_29, permute_30, expand_24, False);  permute_28 = expand_24 = None
        getitem_82 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_31 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_91 = torch.ops.aten.view.default(permute_31, [4, 1024, 768]);  permute_31 = None
        view_92 = torch.ops.aten.view.default(view_91, [-1, 768]);  view_91 = None
        addmm_29 = torch.ops.aten.addmm.default(arg91_1, view_92, arg92_1);  arg91_1 = view_92 = arg92_1 = None
        view_93 = torch.ops.aten.view.default(addmm_29, [4, 1024, 768]);  addmm_29 = None
        add_60 = torch.ops.aten.add.Tensor(view_93, add_57);  view_93 = add_57 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_15[0]
        getitem_87 = var_mean_15[1];  var_mean_15 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_60, getitem_87);  getitem_87 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None
        view_94 = torch.ops.aten.view.default(add_62, [-1, 768]);  add_62 = None
        addmm_30 = torch.ops.aten.addmm.default(arg95_1, view_94, arg96_1);  arg95_1 = view_94 = arg96_1 = None
        view_95 = torch.ops.aten.view.default(addmm_30, [4, 1024, 3072]);  addmm_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_95, 0.5)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_95, 3.0)
        mul_61 = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63 = torch.ops.aten.add.Tensor(view_95, mul_61);  view_95 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64 = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        view_96 = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31 = torch.ops.aten.addmm.default(arg97_1, view_96, arg98_1);  arg97_1 = view_96 = arg98_1 = None
        view_97 = torch.ops.aten.view.default(addmm_31, [4, 1024, 768]);  addmm_31 = None
        add_65 = torch.ops.aten.add.Tensor(add_60, view_97);  add_60 = view_97 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_16[0]
        getitem_89 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_65, getitem_89);  getitem_89 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
        view_98 = torch.ops.aten.view.default(add_67, [-1, 768]);  add_67 = None
        addmm_32 = torch.ops.aten.addmm.default(arg101_1, view_98, arg102_1);  arg101_1 = view_98 = arg102_1 = None
        view_99 = torch.ops.aten.view.default(addmm_32, [4, 1024, 2304]);  addmm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_99, 768, 2);  view_99 = None
        getitem_90 = split_8[0]
        getitem_91 = split_8[1]
        getitem_92 = split_8[2];  split_8 = None
        view_100 = torch.ops.aten.view.default(getitem_90, [4, 1024, 12, 64]);  getitem_90 = None
        permute_32 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        view_101 = torch.ops.aten.view.default(getitem_91, [4, 1024, 12, 64]);  getitem_91 = None
        permute_33 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        view_102 = torch.ops.aten.view.default(getitem_92, [4, 1024, 12, 64]);  getitem_92 = None
        permute_34 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(unsqueeze_19, 1);  unsqueeze_19 = None
        expand_26 = torch.ops.aten.expand.default(unsqueeze_20, [4, 1, 1024, 1024]);  unsqueeze_20 = None
        expand_27 = torch.ops.aten.expand.default(expand_26, [4, 12, 1024, 1024]);  expand_26 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_32, permute_33, permute_34, expand_27, False);  permute_32 = expand_27 = None
        getitem_93 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_35 = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        view_103 = torch.ops.aten.view.default(permute_35, [4, 1024, 768]);  permute_35 = None
        view_104 = torch.ops.aten.view.default(view_103, [-1, 768]);  view_103 = None
        addmm_33 = torch.ops.aten.addmm.default(arg103_1, view_104, arg104_1);  arg103_1 = view_104 = arg104_1 = None
        view_105 = torch.ops.aten.view.default(addmm_33, [4, 1024, 768]);  addmm_33 = None
        add_68 = torch.ops.aten.add.Tensor(view_105, add_65);  view_105 = add_65 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_97 = var_mean_17[0]
        getitem_98 = var_mean_17[1];  var_mean_17 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_68, getitem_98);  getitem_98 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None
        view_106 = torch.ops.aten.view.default(add_70, [-1, 768]);  add_70 = None
        addmm_34 = torch.ops.aten.addmm.default(arg107_1, view_106, arg108_1);  arg107_1 = view_106 = arg108_1 = None
        view_107 = torch.ops.aten.view.default(addmm_34, [4, 1024, 3072]);  addmm_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_107, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
        mul_69 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71 = torch.ops.aten.add.Tensor(view_107, mul_69);  view_107 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72 = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        view_108 = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35 = torch.ops.aten.addmm.default(arg109_1, view_108, arg110_1);  arg109_1 = view_108 = arg110_1 = None
        view_109 = torch.ops.aten.view.default(addmm_35, [4, 1024, 768]);  addmm_35 = None
        add_73 = torch.ops.aten.add.Tensor(add_68, view_109);  add_68 = view_109 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_99 = var_mean_18[0]
        getitem_100 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_73, getitem_100);  getitem_100 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None
        view_110 = torch.ops.aten.view.default(add_75, [-1, 768]);  add_75 = None
        addmm_36 = torch.ops.aten.addmm.default(arg113_1, view_110, arg114_1);  arg113_1 = view_110 = arg114_1 = None
        view_111 = torch.ops.aten.view.default(addmm_36, [4, 1024, 2304]);  addmm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_111, 768, 2);  view_111 = None
        getitem_101 = split_9[0]
        getitem_102 = split_9[1]
        getitem_103 = split_9[2];  split_9 = None
        view_112 = torch.ops.aten.view.default(getitem_101, [4, 1024, 12, 64]);  getitem_101 = None
        permute_36 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        view_113 = torch.ops.aten.view.default(getitem_102, [4, 1024, 12, 64]);  getitem_102 = None
        permute_37 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        view_114 = torch.ops.aten.view.default(getitem_103, [4, 1024, 12, 64]);  getitem_103 = None
        permute_38 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(unsqueeze_21, 1);  unsqueeze_21 = None
        expand_29 = torch.ops.aten.expand.default(unsqueeze_22, [4, 1, 1024, 1024]);  unsqueeze_22 = None
        expand_30 = torch.ops.aten.expand.default(expand_29, [4, 12, 1024, 1024]);  expand_29 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_36, permute_37, permute_38, expand_30, False);  permute_36 = expand_30 = None
        getitem_104 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_39 = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        view_115 = torch.ops.aten.view.default(permute_39, [4, 1024, 768]);  permute_39 = None
        view_116 = torch.ops.aten.view.default(view_115, [-1, 768]);  view_115 = None
        addmm_37 = torch.ops.aten.addmm.default(arg115_1, view_116, arg116_1);  arg115_1 = view_116 = arg116_1 = None
        view_117 = torch.ops.aten.view.default(addmm_37, [4, 1024, 768]);  addmm_37 = None
        add_76 = torch.ops.aten.add.Tensor(view_117, add_73);  view_117 = add_73 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_108 = var_mean_19[0]
        getitem_109 = var_mean_19[1];  var_mean_19 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_76, getitem_109);  getitem_109 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_78 = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None
        view_118 = torch.ops.aten.view.default(add_78, [-1, 768]);  add_78 = None
        addmm_38 = torch.ops.aten.addmm.default(arg119_1, view_118, arg120_1);  arg119_1 = view_118 = arg120_1 = None
        view_119 = torch.ops.aten.view.default(addmm_38, [4, 1024, 3072]);  addmm_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_119, 0.5)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_119, 3.0)
        mul_77 = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79 = torch.ops.aten.add.Tensor(view_119, mul_77);  view_119 = mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80 = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        view_120 = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39 = torch.ops.aten.addmm.default(arg121_1, view_120, arg122_1);  arg121_1 = view_120 = arg122_1 = None
        view_121 = torch.ops.aten.view.default(addmm_39, [4, 1024, 768]);  addmm_39 = None
        add_81 = torch.ops.aten.add.Tensor(add_76, view_121);  add_76 = view_121 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_20[0]
        getitem_111 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_81, getitem_111);  getitem_111 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None
        view_122 = torch.ops.aten.view.default(add_83, [-1, 768]);  add_83 = None
        addmm_40 = torch.ops.aten.addmm.default(arg125_1, view_122, arg126_1);  arg125_1 = view_122 = arg126_1 = None
        view_123 = torch.ops.aten.view.default(addmm_40, [4, 1024, 2304]);  addmm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_123, 768, 2);  view_123 = None
        getitem_112 = split_10[0]
        getitem_113 = split_10[1]
        getitem_114 = split_10[2];  split_10 = None
        view_124 = torch.ops.aten.view.default(getitem_112, [4, 1024, 12, 64]);  getitem_112 = None
        permute_40 = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        view_125 = torch.ops.aten.view.default(getitem_113, [4, 1024, 12, 64]);  getitem_113 = None
        permute_41 = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        view_126 = torch.ops.aten.view.default(getitem_114, [4, 1024, 12, 64]);  getitem_114 = None
        permute_42 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(unsqueeze_23, 1);  unsqueeze_23 = None
        expand_32 = torch.ops.aten.expand.default(unsqueeze_24, [4, 1, 1024, 1024]);  unsqueeze_24 = None
        expand_33 = torch.ops.aten.expand.default(expand_32, [4, 12, 1024, 1024]);  expand_32 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_40, permute_41, permute_42, expand_33, False);  permute_40 = expand_33 = None
        getitem_115 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_43 = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        view_127 = torch.ops.aten.view.default(permute_43, [4, 1024, 768]);  permute_43 = None
        view_128 = torch.ops.aten.view.default(view_127, [-1, 768]);  view_127 = None
        addmm_41 = torch.ops.aten.addmm.default(arg127_1, view_128, arg128_1);  arg127_1 = view_128 = arg128_1 = None
        view_129 = torch.ops.aten.view.default(addmm_41, [4, 1024, 768]);  addmm_41 = None
        add_84 = torch.ops.aten.add.Tensor(view_129, add_81);  view_129 = add_81 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_119 = var_mean_21[0]
        getitem_120 = var_mean_21[1];  var_mean_21 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_84, getitem_120);  getitem_120 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None
        view_130 = torch.ops.aten.view.default(add_86, [-1, 768]);  add_86 = None
        addmm_42 = torch.ops.aten.addmm.default(arg131_1, view_130, arg132_1);  arg131_1 = view_130 = arg132_1 = None
        view_131 = torch.ops.aten.view.default(addmm_42, [4, 1024, 3072]);  addmm_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_131, 0.5)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
        mul_85 = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87 = torch.ops.aten.add.Tensor(view_131, mul_85);  view_131 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88 = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        view_132 = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43 = torch.ops.aten.addmm.default(arg133_1, view_132, arg134_1);  arg133_1 = view_132 = arg134_1 = None
        view_133 = torch.ops.aten.view.default(addmm_43, [4, 1024, 768]);  addmm_43 = None
        add_89 = torch.ops.aten.add.Tensor(add_84, view_133);  add_84 = view_133 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_121 = var_mean_22[0]
        getitem_122 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_89, getitem_122);  getitem_122 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None
        view_134 = torch.ops.aten.view.default(add_91, [-1, 768]);  add_91 = None
        addmm_44 = torch.ops.aten.addmm.default(arg137_1, view_134, arg138_1);  arg137_1 = view_134 = arg138_1 = None
        view_135 = torch.ops.aten.view.default(addmm_44, [4, 1024, 2304]);  addmm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_135, 768, 2);  view_135 = None
        getitem_123 = split_11[0]
        getitem_124 = split_11[1]
        getitem_125 = split_11[2];  split_11 = None
        view_136 = torch.ops.aten.view.default(getitem_123, [4, 1024, 12, 64]);  getitem_123 = None
        permute_44 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        view_137 = torch.ops.aten.view.default(getitem_124, [4, 1024, 12, 64]);  getitem_124 = None
        permute_45 = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        view_138 = torch.ops.aten.view.default(getitem_125, [4, 1024, 12, 64]);  getitem_125 = None
        permute_46 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(unsqueeze_25, 1);  unsqueeze_25 = None
        expand_35 = torch.ops.aten.expand.default(unsqueeze_26, [4, 1, 1024, 1024]);  unsqueeze_26 = None
        expand_36 = torch.ops.aten.expand.default(expand_35, [4, 12, 1024, 1024]);  expand_35 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_44, permute_45, permute_46, expand_36, False);  permute_44 = expand_36 = None
        getitem_126 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_47 = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        view_139 = torch.ops.aten.view.default(permute_47, [4, 1024, 768]);  permute_47 = None
        view_140 = torch.ops.aten.view.default(view_139, [-1, 768]);  view_139 = None
        addmm_45 = torch.ops.aten.addmm.default(arg139_1, view_140, arg140_1);  arg139_1 = view_140 = arg140_1 = None
        view_141 = torch.ops.aten.view.default(addmm_45, [4, 1024, 768]);  addmm_45 = None
        add_92 = torch.ops.aten.add.Tensor(view_141, add_89);  view_141 = add_89 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_130 = var_mean_23[0]
        getitem_131 = var_mean_23[1];  var_mean_23 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_92, getitem_131);  getitem_131 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None
        view_142 = torch.ops.aten.view.default(add_94, [-1, 768]);  add_94 = None
        addmm_46 = torch.ops.aten.addmm.default(arg143_1, view_142, arg144_1);  arg143_1 = view_142 = arg144_1 = None
        view_143 = torch.ops.aten.view.default(addmm_46, [4, 1024, 3072]);  addmm_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_143, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
        mul_93 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95 = torch.ops.aten.add.Tensor(view_143, mul_93);  view_143 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96 = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        view_144 = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47 = torch.ops.aten.addmm.default(arg145_1, view_144, arg146_1);  arg145_1 = view_144 = arg146_1 = None
        view_145 = torch.ops.aten.view.default(addmm_47, [4, 1024, 768]);  addmm_47 = None
        add_97 = torch.ops.aten.add.Tensor(add_92, view_145);  add_92 = view_145 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_132 = var_mean_24[0]
        getitem_133 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_97, getitem_133);  add_97 = getitem_133 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None
        view_146 = torch.ops.aten.view.default(add_99, [-1, 1024, 768]);  add_99 = None
        permute_48 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        view_147 = torch.ops.aten.view.default(view_146, [4096, 768]);  view_146 = None
        mm = torch.ops.aten.mm.default(view_147, permute_48);  view_147 = permute_48 = None
        view_148 = torch.ops.aten.view.default(mm, [4, 1024, 2]);  mm = None
        eq = torch.ops.aten.eq.Scalar(arg0_1, 0);  arg0_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(eq, torch.int32);  eq = None
        argmax = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
        sub_25 = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
        remainder = torch.ops.aten.remainder.Scalar(sub_25, 1024);  sub_25 = None
        iota_2 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index = torch.ops.aten.index.Tensor(view_148, [iota_2, remainder]);  view_148 = iota_2 = remainder = None
        view_149 = torch.ops.aten.view.default(index, [-1, 2])
        view_150 = torch.ops.aten.view.default(arg150_1, [-1]);  arg150_1 = None
        amax = torch.ops.aten.amax.default(view_149, [1], True)
        sub_26 = torch.ops.aten.sub.Tensor(view_149, amax);  view_149 = amax = None
        exp = torch.ops.aten.exp.default(sub_26)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne = torch.ops.aten.ne.Scalar(view_150, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_150, full_default_2);  ne = full_default_2 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_27);  sub_27 = unsqueeze_27 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_150, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_150, -100);  view_150 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, index, permute_1, permute_2, permute_5, permute_6, permute_9, permute_10, permute_13, permute_14, permute_17, permute_18, permute_21, permute_22, permute_25, permute_26, permute_29, permute_30, permute_33, permute_34, permute_37, permute_38, permute_41, permute_42, permute_45, permute_46)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg0_1
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
    buf77 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf77, (2304,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768, 2304), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 768), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf83, (3072,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768, 3072), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (3072, 768), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf89, (2304,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 2304), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768, 768), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf95, (3072,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768, 3072), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf98, (3072, 768), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf101, (2304,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 2304), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768, 768), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf107, (3072,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 3072), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf110, (3072, 768), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf113, (2304,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768, 2304), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768, 768), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf119, (3072,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768, 3072), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf122, (3072, 768), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf125, (2304,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768, 2304), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 768), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf131, (3072,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768, 3072), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf134, (3072, 768), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf137, (2304,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768, 2304), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768, 768), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf143, (3072,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768, 3072), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf146, (3072, 768), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (2, 768), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 32, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf150, (4,), dtype=torch.int64, is_leaf=True)  # arg150_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)