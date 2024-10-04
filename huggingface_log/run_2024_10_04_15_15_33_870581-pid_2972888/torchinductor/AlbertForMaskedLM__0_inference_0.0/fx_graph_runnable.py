
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1):
        full = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand = torch.ops.aten.expand.default(arg1_1, [4, 512]);  arg1_1 = None
        embedding = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg0_1 = None
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
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [4, 1, 512, 512]);  unsqueeze_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(convert_element_type, scalar_tensor, sub_1);  convert_element_type = scalar_tensor = sub_1 = None
        view = torch.ops.aten.view.default(add_3, [2048, 128]);  add_3 = None
        permute = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [4, 512, 4096]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [2048, 4096])
        permute_1 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(arg11_1, view_2, permute_1);  view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [4, 512, 4096]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [4, 512, 64, 64]);  view_3 = None
        permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5 = torch.ops.aten.view.default(view_1, [2048, 4096])
        permute_3 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(arg13_1, view_5, permute_3);  view_5 = permute_3 = None
        view_6 = torch.ops.aten.view.default(addmm_2, [4, 512, 4096]);  addmm_2 = None
        view_7 = torch.ops.aten.view.default(view_6, [4, 512, 64, 64]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        view_8 = torch.ops.aten.view.default(view_1, [2048, 4096])
        permute_5 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(arg15_1, view_8, permute_5);  view_8 = permute_5 = None
        view_9 = torch.ops.aten.view.default(addmm_3, [4, 512, 4096]);  addmm_3 = None
        view_10 = torch.ops.aten.view.default(view_9, [4, 512, 64, 64]);  view_9 = None
        permute_6 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        expand_2 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute_4, permute_6, expand_2, False);  permute_2 = permute_4 = permute_6 = expand_2 = None
        getitem_2 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_7 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_11 = torch.ops.aten.view.default(permute_7, [4, 512, 4096]);  permute_7 = None
        view_12 = torch.ops.aten.view.default(view_11, [2048, 4096]);  view_11 = None
        permute_8 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_12, permute_8);  view_12 = permute_8 = None
        view_13 = torch.ops.aten.view.default(addmm_4, [4, 512, 4096]);  addmm_4 = None
        add_4 = torch.ops.aten.add.Tensor(view_1, view_13);  view_1 = view_13 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg18_1);  mul_2 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg19_1);  mul_3 = None
        view_14 = torch.ops.aten.view.default(add_6, [2048, 4096])
        permute_9 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_5 = torch.ops.aten.addmm.default(arg21_1, view_14, permute_9);  view_14 = permute_9 = None
        view_15 = torch.ops.aten.view.default(addmm_5, [4, 512, 16384]);  addmm_5 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_15, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7 = torch.ops.aten.add.Tensor(view_15, mul_5);  view_15 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        view_16 = torch.ops.aten.view.default(mul_7, [2048, 16384]);  mul_7 = None
        permute_10 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_16, permute_10);  view_16 = permute_10 = None
        view_17 = torch.ops.aten.view.default(addmm_6, [4, 512, 4096]);  addmm_6 = None
        add_9 = torch.ops.aten.add.Tensor(view_17, add_6);  view_17 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg24_1);  mul_8 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg25_1);  mul_9 = None
        view_18 = torch.ops.aten.view.default(add_11, [2048, 4096])
        permute_11 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_7 = torch.ops.aten.addmm.default(arg11_1, view_18, permute_11);  view_18 = permute_11 = None
        view_19 = torch.ops.aten.view.default(addmm_7, [4, 512, 4096]);  addmm_7 = None
        view_20 = torch.ops.aten.view.default(view_19, [4, 512, 64, 64]);  view_19 = None
        permute_12 = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        view_21 = torch.ops.aten.view.default(add_11, [2048, 4096])
        permute_13 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_8 = torch.ops.aten.addmm.default(arg13_1, view_21, permute_13);  view_21 = permute_13 = None
        view_22 = torch.ops.aten.view.default(addmm_8, [4, 512, 4096]);  addmm_8 = None
        view_23 = torch.ops.aten.view.default(view_22, [4, 512, 64, 64]);  view_22 = None
        permute_14 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        view_24 = torch.ops.aten.view.default(add_11, [2048, 4096])
        permute_15 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_9 = torch.ops.aten.addmm.default(arg15_1, view_24, permute_15);  view_24 = permute_15 = None
        view_25 = torch.ops.aten.view.default(addmm_9, [4, 512, 4096]);  addmm_9 = None
        view_26 = torch.ops.aten.view.default(view_25, [4, 512, 64, 64]);  view_25 = None
        permute_16 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        expand_3 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, permute_14, permute_16, expand_3, False);  permute_12 = permute_14 = permute_16 = expand_3 = None
        getitem_10 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_17 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_27 = torch.ops.aten.view.default(permute_17, [4, 512, 4096]);  permute_17 = None
        view_28 = torch.ops.aten.view.default(view_27, [2048, 4096]);  view_27 = None
        permute_18 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_10 = torch.ops.aten.addmm.default(arg17_1, view_28, permute_18);  view_28 = permute_18 = None
        view_29 = torch.ops.aten.view.default(addmm_10, [4, 512, 4096]);  addmm_10 = None
        add_12 = torch.ops.aten.add.Tensor(add_11, view_29);  add_11 = view_29 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_12, getitem_15);  add_12 = getitem_15 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg18_1);  mul_10 = None
        add_14 = torch.ops.aten.add.Tensor(mul_11, arg19_1);  mul_11 = None
        view_30 = torch.ops.aten.view.default(add_14, [2048, 4096])
        permute_19 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_11 = torch.ops.aten.addmm.default(arg21_1, view_30, permute_19);  view_30 = permute_19 = None
        view_31 = torch.ops.aten.view.default(addmm_11, [4, 512, 16384]);  addmm_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_31, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
        mul_13 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15 = torch.ops.aten.add.Tensor(view_31, mul_13);  view_31 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_32 = torch.ops.aten.view.default(mul_15, [2048, 16384]);  mul_15 = None
        permute_20 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_12 = torch.ops.aten.addmm.default(arg23_1, view_32, permute_20);  view_32 = permute_20 = None
        view_33 = torch.ops.aten.view.default(addmm_12, [4, 512, 4096]);  addmm_12 = None
        add_17 = torch.ops.aten.add.Tensor(view_33, add_14);  view_33 = add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_17, getitem_17);  add_17 = getitem_17 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg24_1);  mul_16 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, arg25_1);  mul_17 = None
        view_34 = torch.ops.aten.view.default(add_19, [2048, 4096])
        permute_21 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_13 = torch.ops.aten.addmm.default(arg11_1, view_34, permute_21);  view_34 = permute_21 = None
        view_35 = torch.ops.aten.view.default(addmm_13, [4, 512, 4096]);  addmm_13 = None
        view_36 = torch.ops.aten.view.default(view_35, [4, 512, 64, 64]);  view_35 = None
        permute_22 = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        view_37 = torch.ops.aten.view.default(add_19, [2048, 4096])
        permute_23 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_14 = torch.ops.aten.addmm.default(arg13_1, view_37, permute_23);  view_37 = permute_23 = None
        view_38 = torch.ops.aten.view.default(addmm_14, [4, 512, 4096]);  addmm_14 = None
        view_39 = torch.ops.aten.view.default(view_38, [4, 512, 64, 64]);  view_38 = None
        permute_24 = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        view_40 = torch.ops.aten.view.default(add_19, [2048, 4096])
        permute_25 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_15 = torch.ops.aten.addmm.default(arg15_1, view_40, permute_25);  view_40 = permute_25 = None
        view_41 = torch.ops.aten.view.default(addmm_15, [4, 512, 4096]);  addmm_15 = None
        view_42 = torch.ops.aten.view.default(view_41, [4, 512, 64, 64]);  view_41 = None
        permute_26 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        expand_4 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_24, permute_26, expand_4, False);  permute_22 = permute_24 = permute_26 = expand_4 = None
        getitem_18 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_27 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_43 = torch.ops.aten.view.default(permute_27, [4, 512, 4096]);  permute_27 = None
        view_44 = torch.ops.aten.view.default(view_43, [2048, 4096]);  view_43 = None
        permute_28 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_16 = torch.ops.aten.addmm.default(arg17_1, view_44, permute_28);  view_44 = permute_28 = None
        view_45 = torch.ops.aten.view.default(addmm_16, [4, 512, 4096]);  addmm_16 = None
        add_20 = torch.ops.aten.add.Tensor(add_19, view_45);  add_19 = view_45 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_20, getitem_23);  add_20 = getitem_23 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg18_1);  mul_18 = None
        add_22 = torch.ops.aten.add.Tensor(mul_19, arg19_1);  mul_19 = None
        view_46 = torch.ops.aten.view.default(add_22, [2048, 4096])
        permute_29 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_17 = torch.ops.aten.addmm.default(arg21_1, view_46, permute_29);  view_46 = permute_29 = None
        view_47 = torch.ops.aten.view.default(addmm_17, [4, 512, 16384]);  addmm_17 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_21 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23 = torch.ops.aten.add.Tensor(view_47, mul_21);  view_47 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        view_48 = torch.ops.aten.view.default(mul_23, [2048, 16384]);  mul_23 = None
        permute_30 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_18 = torch.ops.aten.addmm.default(arg23_1, view_48, permute_30);  view_48 = permute_30 = None
        view_49 = torch.ops.aten.view.default(addmm_18, [4, 512, 4096]);  addmm_18 = None
        add_25 = torch.ops.aten.add.Tensor(view_49, add_22);  view_49 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_25, getitem_25);  add_25 = getitem_25 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg24_1);  mul_24 = None
        add_27 = torch.ops.aten.add.Tensor(mul_25, arg25_1);  mul_25 = None
        view_50 = torch.ops.aten.view.default(add_27, [2048, 4096])
        permute_31 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_19 = torch.ops.aten.addmm.default(arg11_1, view_50, permute_31);  view_50 = permute_31 = None
        view_51 = torch.ops.aten.view.default(addmm_19, [4, 512, 4096]);  addmm_19 = None
        view_52 = torch.ops.aten.view.default(view_51, [4, 512, 64, 64]);  view_51 = None
        permute_32 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        view_53 = torch.ops.aten.view.default(add_27, [2048, 4096])
        permute_33 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_20 = torch.ops.aten.addmm.default(arg13_1, view_53, permute_33);  view_53 = permute_33 = None
        view_54 = torch.ops.aten.view.default(addmm_20, [4, 512, 4096]);  addmm_20 = None
        view_55 = torch.ops.aten.view.default(view_54, [4, 512, 64, 64]);  view_54 = None
        permute_34 = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        view_56 = torch.ops.aten.view.default(add_27, [2048, 4096])
        permute_35 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_21 = torch.ops.aten.addmm.default(arg15_1, view_56, permute_35);  view_56 = permute_35 = None
        view_57 = torch.ops.aten.view.default(addmm_21, [4, 512, 4096]);  addmm_21 = None
        view_58 = torch.ops.aten.view.default(view_57, [4, 512, 64, 64]);  view_57 = None
        permute_36 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        expand_5 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_32, permute_34, permute_36, expand_5, False);  permute_32 = permute_34 = permute_36 = expand_5 = None
        getitem_26 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_37 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_59 = torch.ops.aten.view.default(permute_37, [4, 512, 4096]);  permute_37 = None
        view_60 = torch.ops.aten.view.default(view_59, [2048, 4096]);  view_59 = None
        permute_38 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_22 = torch.ops.aten.addmm.default(arg17_1, view_60, permute_38);  view_60 = permute_38 = None
        view_61 = torch.ops.aten.view.default(addmm_22, [4, 512, 4096]);  addmm_22 = None
        add_28 = torch.ops.aten.add.Tensor(add_27, view_61);  add_27 = view_61 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_28, getitem_31);  add_28 = getitem_31 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg18_1);  mul_26 = None
        add_30 = torch.ops.aten.add.Tensor(mul_27, arg19_1);  mul_27 = None
        view_62 = torch.ops.aten.view.default(add_30, [2048, 4096])
        permute_39 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_23 = torch.ops.aten.addmm.default(arg21_1, view_62, permute_39);  view_62 = permute_39 = None
        view_63 = torch.ops.aten.view.default(addmm_23, [4, 512, 16384]);  addmm_23 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_63, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_63, 3.0)
        mul_29 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31 = torch.ops.aten.add.Tensor(view_63, mul_29);  view_63 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        view_64 = torch.ops.aten.view.default(mul_31, [2048, 16384]);  mul_31 = None
        permute_40 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_24 = torch.ops.aten.addmm.default(arg23_1, view_64, permute_40);  view_64 = permute_40 = None
        view_65 = torch.ops.aten.view.default(addmm_24, [4, 512, 4096]);  addmm_24 = None
        add_33 = torch.ops.aten.add.Tensor(view_65, add_30);  view_65 = add_30 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_33, getitem_33);  add_33 = getitem_33 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg24_1);  mul_32 = None
        add_35 = torch.ops.aten.add.Tensor(mul_33, arg25_1);  mul_33 = None
        view_66 = torch.ops.aten.view.default(add_35, [2048, 4096])
        permute_41 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_25 = torch.ops.aten.addmm.default(arg11_1, view_66, permute_41);  view_66 = permute_41 = None
        view_67 = torch.ops.aten.view.default(addmm_25, [4, 512, 4096]);  addmm_25 = None
        view_68 = torch.ops.aten.view.default(view_67, [4, 512, 64, 64]);  view_67 = None
        permute_42 = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        view_69 = torch.ops.aten.view.default(add_35, [2048, 4096])
        permute_43 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_26 = torch.ops.aten.addmm.default(arg13_1, view_69, permute_43);  view_69 = permute_43 = None
        view_70 = torch.ops.aten.view.default(addmm_26, [4, 512, 4096]);  addmm_26 = None
        view_71 = torch.ops.aten.view.default(view_70, [4, 512, 64, 64]);  view_70 = None
        permute_44 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        view_72 = torch.ops.aten.view.default(add_35, [2048, 4096])
        permute_45 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_27 = torch.ops.aten.addmm.default(arg15_1, view_72, permute_45);  view_72 = permute_45 = None
        view_73 = torch.ops.aten.view.default(addmm_27, [4, 512, 4096]);  addmm_27 = None
        view_74 = torch.ops.aten.view.default(view_73, [4, 512, 64, 64]);  view_73 = None
        permute_46 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        expand_6 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_44, permute_46, expand_6, False);  permute_42 = permute_44 = permute_46 = expand_6 = None
        getitem_34 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_47 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_75 = torch.ops.aten.view.default(permute_47, [4, 512, 4096]);  permute_47 = None
        view_76 = torch.ops.aten.view.default(view_75, [2048, 4096]);  view_75 = None
        permute_48 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_28 = torch.ops.aten.addmm.default(arg17_1, view_76, permute_48);  view_76 = permute_48 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [4, 512, 4096]);  addmm_28 = None
        add_36 = torch.ops.aten.add.Tensor(add_35, view_77);  add_35 = view_77 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_36, getitem_39);  add_36 = getitem_39 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg18_1);  mul_34 = None
        add_38 = torch.ops.aten.add.Tensor(mul_35, arg19_1);  mul_35 = None
        view_78 = torch.ops.aten.view.default(add_38, [2048, 4096])
        permute_49 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_29 = torch.ops.aten.addmm.default(arg21_1, view_78, permute_49);  view_78 = permute_49 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [4, 512, 16384]);  addmm_29 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_79, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_79, 3.0)
        mul_37 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39 = torch.ops.aten.add.Tensor(view_79, mul_37);  view_79 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        view_80 = torch.ops.aten.view.default(mul_39, [2048, 16384]);  mul_39 = None
        permute_50 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_30 = torch.ops.aten.addmm.default(arg23_1, view_80, permute_50);  view_80 = permute_50 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [4, 512, 4096]);  addmm_30 = None
        add_41 = torch.ops.aten.add.Tensor(view_81, add_38);  view_81 = add_38 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_41, getitem_41);  add_41 = getitem_41 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg24_1);  mul_40 = None
        add_43 = torch.ops.aten.add.Tensor(mul_41, arg25_1);  mul_41 = None
        view_82 = torch.ops.aten.view.default(add_43, [2048, 4096])
        permute_51 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_31 = torch.ops.aten.addmm.default(arg11_1, view_82, permute_51);  view_82 = permute_51 = None
        view_83 = torch.ops.aten.view.default(addmm_31, [4, 512, 4096]);  addmm_31 = None
        view_84 = torch.ops.aten.view.default(view_83, [4, 512, 64, 64]);  view_83 = None
        permute_52 = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        view_85 = torch.ops.aten.view.default(add_43, [2048, 4096])
        permute_53 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_32 = torch.ops.aten.addmm.default(arg13_1, view_85, permute_53);  view_85 = permute_53 = None
        view_86 = torch.ops.aten.view.default(addmm_32, [4, 512, 4096]);  addmm_32 = None
        view_87 = torch.ops.aten.view.default(view_86, [4, 512, 64, 64]);  view_86 = None
        permute_54 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        view_88 = torch.ops.aten.view.default(add_43, [2048, 4096])
        permute_55 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_33 = torch.ops.aten.addmm.default(arg15_1, view_88, permute_55);  view_88 = permute_55 = None
        view_89 = torch.ops.aten.view.default(addmm_33, [4, 512, 4096]);  addmm_33 = None
        view_90 = torch.ops.aten.view.default(view_89, [4, 512, 64, 64]);  view_89 = None
        permute_56 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        expand_7 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_52, permute_54, permute_56, expand_7, False);  permute_52 = permute_54 = permute_56 = expand_7 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_57 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_91 = torch.ops.aten.view.default(permute_57, [4, 512, 4096]);  permute_57 = None
        view_92 = torch.ops.aten.view.default(view_91, [2048, 4096]);  view_91 = None
        permute_58 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_34 = torch.ops.aten.addmm.default(arg17_1, view_92, permute_58);  view_92 = permute_58 = None
        view_93 = torch.ops.aten.view.default(addmm_34, [4, 512, 4096]);  addmm_34 = None
        add_44 = torch.ops.aten.add.Tensor(add_43, view_93);  add_43 = view_93 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_44, getitem_47);  add_44 = getitem_47 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg18_1);  mul_42 = None
        add_46 = torch.ops.aten.add.Tensor(mul_43, arg19_1);  mul_43 = None
        view_94 = torch.ops.aten.view.default(add_46, [2048, 4096])
        permute_59 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_35 = torch.ops.aten.addmm.default(arg21_1, view_94, permute_59);  view_94 = permute_59 = None
        view_95 = torch.ops.aten.view.default(addmm_35, [4, 512, 16384]);  addmm_35 = None
        mul_44 = torch.ops.aten.mul.Tensor(view_95, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_95, 3.0)
        mul_45 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47 = torch.ops.aten.add.Tensor(view_95, mul_45);  view_95 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        view_96 = torch.ops.aten.view.default(mul_47, [2048, 16384]);  mul_47 = None
        permute_60 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_36 = torch.ops.aten.addmm.default(arg23_1, view_96, permute_60);  view_96 = permute_60 = None
        view_97 = torch.ops.aten.view.default(addmm_36, [4, 512, 4096]);  addmm_36 = None
        add_49 = torch.ops.aten.add.Tensor(view_97, add_46);  view_97 = add_46 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_49, getitem_49);  add_49 = getitem_49 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg24_1);  mul_48 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, arg25_1);  mul_49 = None
        view_98 = torch.ops.aten.view.default(add_51, [2048, 4096])
        permute_61 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_37 = torch.ops.aten.addmm.default(arg11_1, view_98, permute_61);  view_98 = permute_61 = None
        view_99 = torch.ops.aten.view.default(addmm_37, [4, 512, 4096]);  addmm_37 = None
        view_100 = torch.ops.aten.view.default(view_99, [4, 512, 64, 64]);  view_99 = None
        permute_62 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        view_101 = torch.ops.aten.view.default(add_51, [2048, 4096])
        permute_63 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_38 = torch.ops.aten.addmm.default(arg13_1, view_101, permute_63);  view_101 = permute_63 = None
        view_102 = torch.ops.aten.view.default(addmm_38, [4, 512, 4096]);  addmm_38 = None
        view_103 = torch.ops.aten.view.default(view_102, [4, 512, 64, 64]);  view_102 = None
        permute_64 = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        view_104 = torch.ops.aten.view.default(add_51, [2048, 4096])
        permute_65 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_39 = torch.ops.aten.addmm.default(arg15_1, view_104, permute_65);  view_104 = permute_65 = None
        view_105 = torch.ops.aten.view.default(addmm_39, [4, 512, 4096]);  addmm_39 = None
        view_106 = torch.ops.aten.view.default(view_105, [4, 512, 64, 64]);  view_105 = None
        permute_66 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        expand_8 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_62, permute_64, permute_66, expand_8, False);  permute_62 = permute_64 = permute_66 = expand_8 = None
        getitem_50 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_67 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        view_107 = torch.ops.aten.view.default(permute_67, [4, 512, 4096]);  permute_67 = None
        view_108 = torch.ops.aten.view.default(view_107, [2048, 4096]);  view_107 = None
        permute_68 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_40 = torch.ops.aten.addmm.default(arg17_1, view_108, permute_68);  view_108 = permute_68 = None
        view_109 = torch.ops.aten.view.default(addmm_40, [4, 512, 4096]);  addmm_40 = None
        add_52 = torch.ops.aten.add.Tensor(add_51, view_109);  add_51 = view_109 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_13[0]
        getitem_55 = var_mean_13[1];  var_mean_13 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_52, getitem_55);  add_52 = getitem_55 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg18_1);  mul_50 = None
        add_54 = torch.ops.aten.add.Tensor(mul_51, arg19_1);  mul_51 = None
        view_110 = torch.ops.aten.view.default(add_54, [2048, 4096])
        permute_69 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_41 = torch.ops.aten.addmm.default(arg21_1, view_110, permute_69);  view_110 = permute_69 = None
        view_111 = torch.ops.aten.view.default(addmm_41, [4, 512, 16384]);  addmm_41 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_111, 0.5)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(view_111, 3.0)
        mul_53 = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55 = torch.ops.aten.add.Tensor(view_111, mul_53);  view_111 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        view_112 = torch.ops.aten.view.default(mul_55, [2048, 16384]);  mul_55 = None
        permute_70 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_42 = torch.ops.aten.addmm.default(arg23_1, view_112, permute_70);  view_112 = permute_70 = None
        view_113 = torch.ops.aten.view.default(addmm_42, [4, 512, 4096]);  addmm_42 = None
        add_57 = torch.ops.aten.add.Tensor(view_113, add_54);  view_113 = add_54 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_57, getitem_57);  add_57 = getitem_57 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg24_1);  mul_56 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg25_1);  mul_57 = None
        view_114 = torch.ops.aten.view.default(add_59, [2048, 4096])
        permute_71 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_43 = torch.ops.aten.addmm.default(arg11_1, view_114, permute_71);  view_114 = permute_71 = None
        view_115 = torch.ops.aten.view.default(addmm_43, [4, 512, 4096]);  addmm_43 = None
        view_116 = torch.ops.aten.view.default(view_115, [4, 512, 64, 64]);  view_115 = None
        permute_72 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        view_117 = torch.ops.aten.view.default(add_59, [2048, 4096])
        permute_73 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_44 = torch.ops.aten.addmm.default(arg13_1, view_117, permute_73);  view_117 = permute_73 = None
        view_118 = torch.ops.aten.view.default(addmm_44, [4, 512, 4096]);  addmm_44 = None
        view_119 = torch.ops.aten.view.default(view_118, [4, 512, 64, 64]);  view_118 = None
        permute_74 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        view_120 = torch.ops.aten.view.default(add_59, [2048, 4096])
        permute_75 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_45 = torch.ops.aten.addmm.default(arg15_1, view_120, permute_75);  view_120 = permute_75 = None
        view_121 = torch.ops.aten.view.default(addmm_45, [4, 512, 4096]);  addmm_45 = None
        view_122 = torch.ops.aten.view.default(view_121, [4, 512, 64, 64]);  view_121 = None
        permute_76 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        expand_9 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_72, permute_74, permute_76, expand_9, False);  permute_72 = permute_74 = permute_76 = expand_9 = None
        getitem_58 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_77 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        view_123 = torch.ops.aten.view.default(permute_77, [4, 512, 4096]);  permute_77 = None
        view_124 = torch.ops.aten.view.default(view_123, [2048, 4096]);  view_123 = None
        permute_78 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_46 = torch.ops.aten.addmm.default(arg17_1, view_124, permute_78);  view_124 = permute_78 = None
        view_125 = torch.ops.aten.view.default(addmm_46, [4, 512, 4096]);  addmm_46 = None
        add_60 = torch.ops.aten.add.Tensor(add_59, view_125);  add_59 = view_125 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_15[0]
        getitem_63 = var_mean_15[1];  var_mean_15 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_60, getitem_63);  add_60 = getitem_63 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg18_1);  mul_58 = None
        add_62 = torch.ops.aten.add.Tensor(mul_59, arg19_1);  mul_59 = None
        view_126 = torch.ops.aten.view.default(add_62, [2048, 4096])
        permute_79 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_47 = torch.ops.aten.addmm.default(arg21_1, view_126, permute_79);  view_126 = permute_79 = None
        view_127 = torch.ops.aten.view.default(addmm_47, [4, 512, 16384]);  addmm_47 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_127, 0.5)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_127, 3.0)
        mul_61 = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63 = torch.ops.aten.add.Tensor(view_127, mul_61);  view_127 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64 = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        view_128 = torch.ops.aten.view.default(mul_63, [2048, 16384]);  mul_63 = None
        permute_80 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_48 = torch.ops.aten.addmm.default(arg23_1, view_128, permute_80);  view_128 = permute_80 = None
        view_129 = torch.ops.aten.view.default(addmm_48, [4, 512, 4096]);  addmm_48 = None
        add_65 = torch.ops.aten.add.Tensor(view_129, add_62);  view_129 = add_62 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_65, getitem_65);  add_65 = getitem_65 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg24_1);  mul_64 = None
        add_67 = torch.ops.aten.add.Tensor(mul_65, arg25_1);  mul_65 = None
        view_130 = torch.ops.aten.view.default(add_67, [2048, 4096])
        permute_81 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_49 = torch.ops.aten.addmm.default(arg11_1, view_130, permute_81);  view_130 = permute_81 = None
        view_131 = torch.ops.aten.view.default(addmm_49, [4, 512, 4096]);  addmm_49 = None
        view_132 = torch.ops.aten.view.default(view_131, [4, 512, 64, 64]);  view_131 = None
        permute_82 = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        view_133 = torch.ops.aten.view.default(add_67, [2048, 4096])
        permute_83 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_50 = torch.ops.aten.addmm.default(arg13_1, view_133, permute_83);  view_133 = permute_83 = None
        view_134 = torch.ops.aten.view.default(addmm_50, [4, 512, 4096]);  addmm_50 = None
        view_135 = torch.ops.aten.view.default(view_134, [4, 512, 64, 64]);  view_134 = None
        permute_84 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        view_136 = torch.ops.aten.view.default(add_67, [2048, 4096])
        permute_85 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_51 = torch.ops.aten.addmm.default(arg15_1, view_136, permute_85);  view_136 = permute_85 = None
        view_137 = torch.ops.aten.view.default(addmm_51, [4, 512, 4096]);  addmm_51 = None
        view_138 = torch.ops.aten.view.default(view_137, [4, 512, 64, 64]);  view_137 = None
        permute_86 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        expand_10 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_82, permute_84, permute_86, expand_10, False);  permute_82 = permute_84 = permute_86 = expand_10 = None
        getitem_66 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_87 = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        view_139 = torch.ops.aten.view.default(permute_87, [4, 512, 4096]);  permute_87 = None
        view_140 = torch.ops.aten.view.default(view_139, [2048, 4096]);  view_139 = None
        permute_88 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_52 = torch.ops.aten.addmm.default(arg17_1, view_140, permute_88);  view_140 = permute_88 = None
        view_141 = torch.ops.aten.view.default(addmm_52, [4, 512, 4096]);  addmm_52 = None
        add_68 = torch.ops.aten.add.Tensor(add_67, view_141);  add_67 = view_141 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_17[0]
        getitem_71 = var_mean_17[1];  var_mean_17 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_68, getitem_71);  add_68 = getitem_71 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg18_1);  mul_66 = None
        add_70 = torch.ops.aten.add.Tensor(mul_67, arg19_1);  mul_67 = None
        view_142 = torch.ops.aten.view.default(add_70, [2048, 4096])
        permute_89 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_53 = torch.ops.aten.addmm.default(arg21_1, view_142, permute_89);  view_142 = permute_89 = None
        view_143 = torch.ops.aten.view.default(addmm_53, [4, 512, 16384]);  addmm_53 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_143, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
        mul_69 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71 = torch.ops.aten.add.Tensor(view_143, mul_69);  view_143 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72 = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        view_144 = torch.ops.aten.view.default(mul_71, [2048, 16384]);  mul_71 = None
        permute_90 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_54 = torch.ops.aten.addmm.default(arg23_1, view_144, permute_90);  view_144 = permute_90 = None
        view_145 = torch.ops.aten.view.default(addmm_54, [4, 512, 4096]);  addmm_54 = None
        add_73 = torch.ops.aten.add.Tensor(view_145, add_70);  view_145 = add_70 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_73, getitem_73);  add_73 = getitem_73 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg24_1);  mul_72 = None
        add_75 = torch.ops.aten.add.Tensor(mul_73, arg25_1);  mul_73 = None
        view_146 = torch.ops.aten.view.default(add_75, [2048, 4096])
        permute_91 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_55 = torch.ops.aten.addmm.default(arg11_1, view_146, permute_91);  view_146 = permute_91 = None
        view_147 = torch.ops.aten.view.default(addmm_55, [4, 512, 4096]);  addmm_55 = None
        view_148 = torch.ops.aten.view.default(view_147, [4, 512, 64, 64]);  view_147 = None
        permute_92 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        view_149 = torch.ops.aten.view.default(add_75, [2048, 4096])
        permute_93 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_56 = torch.ops.aten.addmm.default(arg13_1, view_149, permute_93);  view_149 = permute_93 = None
        view_150 = torch.ops.aten.view.default(addmm_56, [4, 512, 4096]);  addmm_56 = None
        view_151 = torch.ops.aten.view.default(view_150, [4, 512, 64, 64]);  view_150 = None
        permute_94 = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        view_152 = torch.ops.aten.view.default(add_75, [2048, 4096])
        permute_95 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_57 = torch.ops.aten.addmm.default(arg15_1, view_152, permute_95);  view_152 = permute_95 = None
        view_153 = torch.ops.aten.view.default(addmm_57, [4, 512, 4096]);  addmm_57 = None
        view_154 = torch.ops.aten.view.default(view_153, [4, 512, 64, 64]);  view_153 = None
        permute_96 = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        expand_11 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_92, permute_94, permute_96, expand_11, False);  permute_92 = permute_94 = permute_96 = expand_11 = None
        getitem_74 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_97 = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        view_155 = torch.ops.aten.view.default(permute_97, [4, 512, 4096]);  permute_97 = None
        view_156 = torch.ops.aten.view.default(view_155, [2048, 4096]);  view_155 = None
        permute_98 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_58 = torch.ops.aten.addmm.default(arg17_1, view_156, permute_98);  view_156 = permute_98 = None
        view_157 = torch.ops.aten.view.default(addmm_58, [4, 512, 4096]);  addmm_58 = None
        add_76 = torch.ops.aten.add.Tensor(add_75, view_157);  add_75 = view_157 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_19[0]
        getitem_79 = var_mean_19[1];  var_mean_19 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_76, getitem_79);  add_76 = getitem_79 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg18_1);  mul_74 = None
        add_78 = torch.ops.aten.add.Tensor(mul_75, arg19_1);  mul_75 = None
        view_158 = torch.ops.aten.view.default(add_78, [2048, 4096])
        permute_99 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_59 = torch.ops.aten.addmm.default(arg21_1, view_158, permute_99);  view_158 = permute_99 = None
        view_159 = torch.ops.aten.view.default(addmm_59, [4, 512, 16384]);  addmm_59 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_159, 0.5)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_159, 3.0)
        mul_77 = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79 = torch.ops.aten.add.Tensor(view_159, mul_77);  view_159 = mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80 = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        view_160 = torch.ops.aten.view.default(mul_79, [2048, 16384]);  mul_79 = None
        permute_100 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_60 = torch.ops.aten.addmm.default(arg23_1, view_160, permute_100);  view_160 = permute_100 = None
        view_161 = torch.ops.aten.view.default(addmm_60, [4, 512, 4096]);  addmm_60 = None
        add_81 = torch.ops.aten.add.Tensor(view_161, add_78);  view_161 = add_78 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_81, getitem_81);  add_81 = getitem_81 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg24_1);  mul_80 = None
        add_83 = torch.ops.aten.add.Tensor(mul_81, arg25_1);  mul_81 = None
        view_162 = torch.ops.aten.view.default(add_83, [2048, 4096])
        permute_101 = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_61 = torch.ops.aten.addmm.default(arg11_1, view_162, permute_101);  view_162 = permute_101 = None
        view_163 = torch.ops.aten.view.default(addmm_61, [4, 512, 4096]);  addmm_61 = None
        view_164 = torch.ops.aten.view.default(view_163, [4, 512, 64, 64]);  view_163 = None
        permute_102 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        view_165 = torch.ops.aten.view.default(add_83, [2048, 4096])
        permute_103 = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_62 = torch.ops.aten.addmm.default(arg13_1, view_165, permute_103);  view_165 = permute_103 = None
        view_166 = torch.ops.aten.view.default(addmm_62, [4, 512, 4096]);  addmm_62 = None
        view_167 = torch.ops.aten.view.default(view_166, [4, 512, 64, 64]);  view_166 = None
        permute_104 = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        view_168 = torch.ops.aten.view.default(add_83, [2048, 4096])
        permute_105 = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_63 = torch.ops.aten.addmm.default(arg15_1, view_168, permute_105);  view_168 = permute_105 = None
        view_169 = torch.ops.aten.view.default(addmm_63, [4, 512, 4096]);  addmm_63 = None
        view_170 = torch.ops.aten.view.default(view_169, [4, 512, 64, 64]);  view_169 = None
        permute_106 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        expand_12 = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_102, permute_104, permute_106, expand_12, False);  permute_102 = permute_104 = permute_106 = expand_12 = None
        getitem_82 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_107 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_171 = torch.ops.aten.view.default(permute_107, [4, 512, 4096]);  permute_107 = None
        view_172 = torch.ops.aten.view.default(view_171, [2048, 4096]);  view_171 = None
        permute_108 = torch.ops.aten.permute.default(arg16_1, [1, 0])
        addmm_64 = torch.ops.aten.addmm.default(arg17_1, view_172, permute_108);  view_172 = permute_108 = None
        view_173 = torch.ops.aten.view.default(addmm_64, [4, 512, 4096]);  addmm_64 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, view_173);  add_83 = view_173 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_21[0]
        getitem_87 = var_mean_21[1];  var_mean_21 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_84, getitem_87);  add_84 = getitem_87 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg18_1);  mul_82 = None
        add_86 = torch.ops.aten.add.Tensor(mul_83, arg19_1);  mul_83 = None
        view_174 = torch.ops.aten.view.default(add_86, [2048, 4096])
        permute_109 = torch.ops.aten.permute.default(arg20_1, [1, 0])
        addmm_65 = torch.ops.aten.addmm.default(arg21_1, view_174, permute_109);  view_174 = permute_109 = None
        view_175 = torch.ops.aten.view.default(addmm_65, [4, 512, 16384]);  addmm_65 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
        mul_85 = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87 = torch.ops.aten.add.Tensor(view_175, mul_85);  view_175 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88 = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        view_176 = torch.ops.aten.view.default(mul_87, [2048, 16384]);  mul_87 = None
        permute_110 = torch.ops.aten.permute.default(arg22_1, [1, 0])
        addmm_66 = torch.ops.aten.addmm.default(arg23_1, view_176, permute_110);  view_176 = permute_110 = None
        view_177 = torch.ops.aten.view.default(addmm_66, [4, 512, 4096]);  addmm_66 = None
        add_89 = torch.ops.aten.add.Tensor(view_177, add_86);  view_177 = add_86 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_89, getitem_89);  add_89 = getitem_89 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg24_1);  mul_88 = None
        add_91 = torch.ops.aten.add.Tensor(mul_89, arg25_1);  mul_89 = None
        view_178 = torch.ops.aten.view.default(add_91, [2048, 4096])
        permute_111 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg11_1, view_178, permute_111);  arg11_1 = view_178 = permute_111 = None
        view_179 = torch.ops.aten.view.default(addmm_67, [4, 512, 4096]);  addmm_67 = None
        view_180 = torch.ops.aten.view.default(view_179, [4, 512, 64, 64]);  view_179 = None
        permute_112 = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        view_181 = torch.ops.aten.view.default(add_91, [2048, 4096])
        permute_113 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg13_1, view_181, permute_113);  arg13_1 = view_181 = permute_113 = None
        view_182 = torch.ops.aten.view.default(addmm_68, [4, 512, 4096]);  addmm_68 = None
        view_183 = torch.ops.aten.view.default(view_182, [4, 512, 64, 64]);  view_182 = None
        permute_114 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        view_184 = torch.ops.aten.view.default(add_91, [2048, 4096])
        permute_115 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg15_1, view_184, permute_115);  arg15_1 = view_184 = permute_115 = None
        view_185 = torch.ops.aten.view.default(addmm_69, [4, 512, 4096]);  addmm_69 = None
        view_186 = torch.ops.aten.view.default(view_185, [4, 512, 64, 64]);  view_185 = None
        permute_116 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        expand_13 = torch.ops.aten.expand.default(where, [4, 64, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_112, permute_114, permute_116, expand_13, False);  permute_112 = permute_114 = permute_116 = expand_13 = None
        getitem_90 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_117 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        view_187 = torch.ops.aten.view.default(permute_117, [4, 512, 4096]);  permute_117 = None
        view_188 = torch.ops.aten.view.default(view_187, [2048, 4096]);  view_187 = None
        permute_118 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg17_1, view_188, permute_118);  arg17_1 = view_188 = permute_118 = None
        view_189 = torch.ops.aten.view.default(addmm_70, [4, 512, 4096]);  addmm_70 = None
        add_92 = torch.ops.aten.add.Tensor(add_91, view_189);  add_91 = view_189 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_23[0]
        getitem_95 = var_mean_23[1];  var_mean_23 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_92, getitem_95);  add_92 = getitem_95 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg18_1);  mul_90 = arg18_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_91, arg19_1);  mul_91 = arg19_1 = None
        view_190 = torch.ops.aten.view.default(add_94, [2048, 4096])
        permute_119 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg21_1, view_190, permute_119);  arg21_1 = view_190 = permute_119 = None
        view_191 = torch.ops.aten.view.default(addmm_71, [4, 512, 16384]);  addmm_71 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_191, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_191, 3.0)
        mul_93 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95 = torch.ops.aten.add.Tensor(view_191, mul_93);  view_191 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96 = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        view_192 = torch.ops.aten.view.default(mul_95, [2048, 16384]);  mul_95 = None
        permute_120 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg23_1, view_192, permute_120);  arg23_1 = view_192 = permute_120 = None
        view_193 = torch.ops.aten.view.default(addmm_72, [4, 512, 4096]);  addmm_72 = None
        add_97 = torch.ops.aten.add.Tensor(view_193, add_94);  view_193 = add_94 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_97, getitem_97);  add_97 = getitem_97 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg24_1);  mul_96 = arg24_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_97, arg25_1);  mul_97 = arg25_1 = None
        view_194 = torch.ops.aten.view.default(add_99, [2048, 4096]);  add_99 = None
        permute_121 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg27_1, view_194, permute_121);  arg27_1 = view_194 = permute_121 = None
        view_195 = torch.ops.aten.view.default(addmm_73, [4, 512, 128]);  addmm_73 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_195, 0.5)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
        mul_99 = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_100 = torch.ops.aten.add.Tensor(view_195, mul_99);  view_195 = mul_99 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_12 = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
        add_101 = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_98, add_101);  mul_98 = add_101 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
        getitem_98 = var_mean_25[0]
        getitem_99 = var_mean_25[1];  var_mean_25 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_26 = torch.ops.aten.sub.Tensor(mul_101, getitem_99);  mul_101 = getitem_99 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_25);  sub_26 = rsqrt_25 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg28_1);  mul_102 = arg28_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_103, arg29_1);  mul_103 = arg29_1 = None
        view_196 = torch.ops.aten.view.default(add_103, [2048, 128]);  add_103 = None
        permute_122 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg30_1, view_196, permute_122);  arg30_1 = view_196 = permute_122 = None
        view_197 = torch.ops.aten.view.default(addmm_74, [4, 512, 30000]);  addmm_74 = None
        view_198 = torch.ops.aten.view.default(view_197, [-1, 30000])
        view_199 = torch.ops.aten.view.default(arg31_1, [-1]);  arg31_1 = None
        amax = torch.ops.aten.amax.default(view_198, [1], True)
        sub_27 = torch.ops.aten.sub.Tensor(view_198, amax);  view_198 = amax = None
        exp = torch.ops.aten.exp.default(sub_27)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28 = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        ne = torch.ops.aten.ne.Scalar(view_199, -100)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_199, full_default);  ne = full_default = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_28, 1, unsqueeze_2);  sub_28 = unsqueeze_2 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_199, -100)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_199, -100);  view_199 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, view_197)
        
def load_args(reader):
    buf0 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 15360000, device=device(type='cuda', index=0))
    reader.tensor(buf3, (30000, 128), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 128), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 128), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf8, (4096, 128), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf9, (4096,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf10, (4096, 4096), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf11, (4096,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf12, (4096, 4096), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf13, (4096,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4096, 4096), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf15, (4096,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf16, (4096, 4096), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf17, (4096,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf18, (4096,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf19, (4096,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 268435456, device=device(type='cuda', index=0))
    reader.tensor(buf20, (16384, 4096), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf21, (16384,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 268435456, device=device(type='cuda', index=0))
    reader.tensor(buf22, (4096, 16384), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf23, (4096,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf24, (4096,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf25, (4096,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128, 4096), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 120000, device=device(type='cuda', index=0))
    reader.tensor(buf30, (30000,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf31, (4, 512), dtype=torch.int64, is_leaf=True)  # arg31_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)