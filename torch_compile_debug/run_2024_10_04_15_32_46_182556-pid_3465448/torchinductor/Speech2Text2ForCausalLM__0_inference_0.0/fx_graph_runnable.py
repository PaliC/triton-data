
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 128]);  arg0_1 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, view, 1)
        mul = torch.ops.aten.mul.Tensor(embedding, 16.0);  embedding = None
        full_default = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota, 1)
        view_1 = torch.ops.aten.view.default(add, [128, 1]);  add = None
        lt = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        ne = torch.ops.aten.ne.Scalar(view, 1);  view = None
        convert_element_type = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        cumsum = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_1 = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, convert_element_type);  add_1 = convert_element_type = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_2 = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        view_2 = torch.ops.aten.view.default(add_2, [-1]);  add_2 = None
        index = torch.ops.aten.index.Tensor(arg2_1, [view_2]);  arg2_1 = view_2 = None
        view_3 = torch.ops.aten.view.default(index, [256, 128, -1]);  index = None
        add_3 = torch.ops.aten.add.Tensor(mul, view_3);  mul = view_3 = None
        view_4 = torch.ops.aten.view.default(add_3, [32768, 256])
        permute = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm = torch.ops.aten.addmm.default(arg4_1, view_4, permute);  arg4_1 = view_4 = permute = None
        view_5 = torch.ops.aten.view.default(addmm, [256, 128, 256]);  addmm = None
        mul_2 = torch.ops.aten.mul.Tensor(view_5, 0.125);  view_5 = None
        view_6 = torch.ops.aten.view.default(add_3, [32768, 256])
        permute_1 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg6_1, view_6, permute_1);  arg6_1 = view_6 = permute_1 = None
        view_7 = torch.ops.aten.view.default(addmm_1, [256, 128, 256]);  addmm_1 = None
        view_8 = torch.ops.aten.view.default(view_7, [256, -1, 4, 64]);  view_7 = None
        permute_2 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_9 = torch.ops.aten.view.default(add_3, [32768, 256])
        permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg8_1, view_9, permute_3);  arg8_1 = view_9 = permute_3 = None
        view_10 = torch.ops.aten.view.default(addmm_2, [256, 128, 256]);  addmm_2 = None
        view_11 = torch.ops.aten.view.default(view_10, [256, -1, 4, 64]);  view_10 = None
        permute_4 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_12 = torch.ops.aten.view.default(mul_2, [256, 128, 4, 64]);  mul_2 = None
        permute_5 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_13 = torch.ops.aten.view.default(clone_3, [1024, -1, 64]);  clone_3 = None
        view_14 = torch.ops.aten.view.default(clone_1, [1024, -1, 64])
        view_15 = torch.ops.aten.view.default(clone_2, [1024, -1, 64])
        permute_6 = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        bmm = torch.ops.aten.bmm.default(view_13, permute_6);  view_13 = permute_6 = None
        view_16 = torch.ops.aten.view.default(bmm, [256, 4, 128, 128]);  bmm = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [256, 1, 128, 128]);  unsqueeze_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_16, expand_1);  view_16 = None
        view_17 = torch.ops.aten.view.default(add_4, [1024, 128, 128]);  add_4 = None
        amax = torch.ops.aten.amax.default(view_17, [-1], True)
        sub = torch.ops.aten.sub.Tensor(view_17, amax);  view_17 = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm_1 = torch.ops.aten.bmm.default(div, view_15);  div = view_15 = None
        view_18 = torch.ops.aten.view.default(bmm_1, [256, 4, 128, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_19 = torch.ops.aten.view.default(clone_5, [256, 128, 256]);  clone_5 = None
        view_20 = torch.ops.aten.view.default(view_19, [32768, 256]);  view_19 = None
        permute_8 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg10_1, view_20, permute_8);  arg10_1 = view_20 = permute_8 = None
        view_21 = torch.ops.aten.view.default(addmm_3, [256, 128, 256]);  addmm_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, view_21);  add_3 = view_21 = None
        var_mean = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_6 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_5, getitem_1);  add_5 = getitem_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
        view_22 = torch.ops.aten.view.default(add_7, [32768, 256])
        permute_9 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg14_1, view_22, permute_9);  arg14_1 = view_22 = permute_9 = None
        view_23 = torch.ops.aten.view.default(addmm_4, [256, 128, 2048]);  addmm_4 = None
        relu = torch.ops.aten.relu.default(view_23);  view_23 = None
        view_24 = torch.ops.aten.view.default(relu, [32768, 2048]);  relu = None
        permute_10 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg16_1, view_24, permute_10);  arg16_1 = view_24 = permute_10 = None
        view_25 = torch.ops.aten.view.default(addmm_5, [256, 128, 256]);  addmm_5 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, view_25);  add_7 = view_25 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_3);  add_8 = getitem_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, arg17_1);  mul_5 = arg17_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_6, arg18_1);  mul_6 = arg18_1 = None
        view_26 = torch.ops.aten.view.default(add_10, [32768, 256])
        permute_11 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg20_1, view_26, permute_11);  arg20_1 = view_26 = permute_11 = None
        view_27 = torch.ops.aten.view.default(addmm_6, [256, 128, 256]);  addmm_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_27, 0.125);  view_27 = None
        view_28 = torch.ops.aten.view.default(add_10, [32768, 256])
        permute_12 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg22_1, view_28, permute_12);  arg22_1 = view_28 = permute_12 = None
        view_29 = torch.ops.aten.view.default(addmm_7, [256, 128, 256]);  addmm_7 = None
        view_30 = torch.ops.aten.view.default(view_29, [256, -1, 4, 64]);  view_29 = None
        permute_13 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_9 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_31 = torch.ops.aten.view.default(add_10, [32768, 256])
        permute_14 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg24_1, view_31, permute_14);  arg24_1 = view_31 = permute_14 = None
        view_32 = torch.ops.aten.view.default(addmm_8, [256, 128, 256]);  addmm_8 = None
        view_33 = torch.ops.aten.view.default(view_32, [256, -1, 4, 64]);  view_32 = None
        permute_15 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_10 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_34 = torch.ops.aten.view.default(mul_7, [256, 128, 4, 64]);  mul_7 = None
        permute_16 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_11 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_35 = torch.ops.aten.view.default(clone_11, [1024, -1, 64]);  clone_11 = None
        view_36 = torch.ops.aten.view.default(clone_9, [1024, -1, 64])
        view_37 = torch.ops.aten.view.default(clone_10, [1024, -1, 64])
        permute_17 = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        bmm_2 = torch.ops.aten.bmm.default(view_35, permute_17);  view_35 = permute_17 = None
        view_38 = torch.ops.aten.view.default(bmm_2, [256, 4, 128, 128]);  bmm_2 = None
        add_11 = torch.ops.aten.add.Tensor(view_38, expand_1);  view_38 = None
        view_39 = torch.ops.aten.view.default(add_11, [1024, 128, 128]);  add_11 = None
        amax_1 = torch.ops.aten.amax.default(view_39, [-1], True)
        sub_3 = torch.ops.aten.sub.Tensor(view_39, amax_1);  view_39 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        bmm_3 = torch.ops.aten.bmm.default(div_1, view_37);  div_1 = view_37 = None
        view_40 = torch.ops.aten.view.default(bmm_3, [256, 4, 128, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_13 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_41 = torch.ops.aten.view.default(clone_13, [256, 128, 256]);  clone_13 = None
        view_42 = torch.ops.aten.view.default(view_41, [32768, 256]);  view_41 = None
        permute_19 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg26_1, view_42, permute_19);  arg26_1 = view_42 = permute_19 = None
        view_43 = torch.ops.aten.view.default(addmm_9, [256, 128, 256]);  addmm_9 = None
        add_12 = torch.ops.aten.add.Tensor(add_10, view_43);  add_10 = view_43 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_12, getitem_5);  add_12 = getitem_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg27_1);  mul_8 = arg27_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_9, arg28_1);  mul_9 = arg28_1 = None
        view_44 = torch.ops.aten.view.default(add_14, [32768, 256])
        permute_20 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg30_1, view_44, permute_20);  arg30_1 = view_44 = permute_20 = None
        view_45 = torch.ops.aten.view.default(addmm_10, [256, 128, 2048]);  addmm_10 = None
        relu_1 = torch.ops.aten.relu.default(view_45);  view_45 = None
        view_46 = torch.ops.aten.view.default(relu_1, [32768, 2048]);  relu_1 = None
        permute_21 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg32_1, view_46, permute_21);  arg32_1 = view_46 = permute_21 = None
        view_47 = torch.ops.aten.view.default(addmm_11, [256, 128, 256]);  addmm_11 = None
        add_15 = torch.ops.aten.add.Tensor(add_14, view_47);  add_14 = view_47 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
        add_17 = torch.ops.aten.add.Tensor(mul_11, arg34_1);  mul_11 = arg34_1 = None
        view_48 = torch.ops.aten.view.default(add_17, [32768, 256])
        permute_22 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg36_1, view_48, permute_22);  arg36_1 = view_48 = permute_22 = None
        view_49 = torch.ops.aten.view.default(addmm_12, [256, 128, 256]);  addmm_12 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_49, 0.125);  view_49 = None
        view_50 = torch.ops.aten.view.default(add_17, [32768, 256])
        permute_23 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg38_1, view_50, permute_23);  arg38_1 = view_50 = permute_23 = None
        view_51 = torch.ops.aten.view.default(addmm_13, [256, 128, 256]);  addmm_13 = None
        view_52 = torch.ops.aten.view.default(view_51, [256, -1, 4, 64]);  view_51 = None
        permute_24 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_17 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_53 = torch.ops.aten.view.default(add_17, [32768, 256])
        permute_25 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg40_1, view_53, permute_25);  arg40_1 = view_53 = permute_25 = None
        view_54 = torch.ops.aten.view.default(addmm_14, [256, 128, 256]);  addmm_14 = None
        view_55 = torch.ops.aten.view.default(view_54, [256, -1, 4, 64]);  view_54 = None
        permute_26 = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        clone_18 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_56 = torch.ops.aten.view.default(mul_12, [256, 128, 4, 64]);  mul_12 = None
        permute_27 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_19 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_57 = torch.ops.aten.view.default(clone_19, [1024, -1, 64]);  clone_19 = None
        view_58 = torch.ops.aten.view.default(clone_17, [1024, -1, 64])
        view_59 = torch.ops.aten.view.default(clone_18, [1024, -1, 64])
        permute_28 = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        bmm_4 = torch.ops.aten.bmm.default(view_57, permute_28);  view_57 = permute_28 = None
        view_60 = torch.ops.aten.view.default(bmm_4, [256, 4, 128, 128]);  bmm_4 = None
        add_18 = torch.ops.aten.add.Tensor(view_60, expand_1);  view_60 = None
        view_61 = torch.ops.aten.view.default(add_18, [1024, 128, 128]);  add_18 = None
        amax_2 = torch.ops.aten.amax.default(view_61, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(view_61, amax_2);  view_61 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        bmm_5 = torch.ops.aten.bmm.default(div_2, view_59);  div_2 = view_59 = None
        view_62 = torch.ops.aten.view.default(bmm_5, [256, 4, 128, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        clone_21 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_63 = torch.ops.aten.view.default(clone_21, [256, 128, 256]);  clone_21 = None
        view_64 = torch.ops.aten.view.default(view_63, [32768, 256]);  view_63 = None
        permute_30 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg42_1, view_64, permute_30);  arg42_1 = view_64 = permute_30 = None
        view_65 = torch.ops.aten.view.default(addmm_15, [256, 128, 256]);  addmm_15 = None
        add_19 = torch.ops.aten.add.Tensor(add_17, view_65);  add_17 = view_65 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, arg43_1);  mul_13 = arg43_1 = None
        add_21 = torch.ops.aten.add.Tensor(mul_14, arg44_1);  mul_14 = arg44_1 = None
        view_66 = torch.ops.aten.view.default(add_21, [32768, 256])
        permute_31 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg46_1, view_66, permute_31);  arg46_1 = view_66 = permute_31 = None
        view_67 = torch.ops.aten.view.default(addmm_16, [256, 128, 2048]);  addmm_16 = None
        relu_2 = torch.ops.aten.relu.default(view_67);  view_67 = None
        view_68 = torch.ops.aten.view.default(relu_2, [32768, 2048]);  relu_2 = None
        permute_32 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg48_1, view_68, permute_32);  arg48_1 = view_68 = permute_32 = None
        view_69 = torch.ops.aten.view.default(addmm_17, [256, 128, 256]);  addmm_17 = None
        add_22 = torch.ops.aten.add.Tensor(add_21, view_69);  add_21 = view_69 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_22, getitem_11);  add_22 = getitem_11 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, arg49_1);  mul_15 = arg49_1 = None
        add_24 = torch.ops.aten.add.Tensor(mul_16, arg50_1);  mul_16 = arg50_1 = None
        view_70 = torch.ops.aten.view.default(add_24, [32768, 256])
        permute_33 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg52_1, view_70, permute_33);  arg52_1 = view_70 = permute_33 = None
        view_71 = torch.ops.aten.view.default(addmm_18, [256, 128, 256]);  addmm_18 = None
        mul_17 = torch.ops.aten.mul.Tensor(view_71, 0.125);  view_71 = None
        view_72 = torch.ops.aten.view.default(add_24, [32768, 256])
        permute_34 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg54_1, view_72, permute_34);  arg54_1 = view_72 = permute_34 = None
        view_73 = torch.ops.aten.view.default(addmm_19, [256, 128, 256]);  addmm_19 = None
        view_74 = torch.ops.aten.view.default(view_73, [256, -1, 4, 64]);  view_73 = None
        permute_35 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_75 = torch.ops.aten.view.default(add_24, [32768, 256])
        permute_36 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg56_1, view_75, permute_36);  arg56_1 = view_75 = permute_36 = None
        view_76 = torch.ops.aten.view.default(addmm_20, [256, 128, 256]);  addmm_20 = None
        view_77 = torch.ops.aten.view.default(view_76, [256, -1, 4, 64]);  view_76 = None
        permute_37 = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        clone_26 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_78 = torch.ops.aten.view.default(mul_17, [256, 128, 4, 64]);  mul_17 = None
        permute_38 = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        clone_27 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_79 = torch.ops.aten.view.default(clone_27, [1024, -1, 64]);  clone_27 = None
        view_80 = torch.ops.aten.view.default(clone_25, [1024, -1, 64])
        view_81 = torch.ops.aten.view.default(clone_26, [1024, -1, 64])
        permute_39 = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        bmm_6 = torch.ops.aten.bmm.default(view_79, permute_39);  view_79 = permute_39 = None
        view_82 = torch.ops.aten.view.default(bmm_6, [256, 4, 128, 128]);  bmm_6 = None
        add_25 = torch.ops.aten.add.Tensor(view_82, expand_1);  view_82 = None
        view_83 = torch.ops.aten.view.default(add_25, [1024, 128, 128]);  add_25 = None
        amax_3 = torch.ops.aten.amax.default(view_83, [-1], True)
        sub_9 = torch.ops.aten.sub.Tensor(view_83, amax_3);  view_83 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        bmm_7 = torch.ops.aten.bmm.default(div_3, view_81);  div_3 = view_81 = None
        view_84 = torch.ops.aten.view.default(bmm_7, [256, 4, 128, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_29 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85 = torch.ops.aten.view.default(clone_29, [256, 128, 256]);  clone_29 = None
        view_86 = torch.ops.aten.view.default(view_85, [32768, 256]);  view_85 = None
        permute_41 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg58_1, view_86, permute_41);  arg58_1 = view_86 = permute_41 = None
        view_87 = torch.ops.aten.view.default(addmm_21, [256, 128, 256]);  addmm_21 = None
        add_26 = torch.ops.aten.add.Tensor(add_24, view_87);  add_24 = view_87 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_26, getitem_13);  add_26 = getitem_13 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg59_1);  mul_18 = arg59_1 = None
        add_28 = torch.ops.aten.add.Tensor(mul_19, arg60_1);  mul_19 = arg60_1 = None
        view_88 = torch.ops.aten.view.default(add_28, [32768, 256])
        permute_42 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg62_1, view_88, permute_42);  arg62_1 = view_88 = permute_42 = None
        view_89 = torch.ops.aten.view.default(addmm_22, [256, 128, 2048]);  addmm_22 = None
        relu_3 = torch.ops.aten.relu.default(view_89);  view_89 = None
        view_90 = torch.ops.aten.view.default(relu_3, [32768, 2048]);  relu_3 = None
        permute_43 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg64_1, view_90, permute_43);  arg64_1 = view_90 = permute_43 = None
        view_91 = torch.ops.aten.view.default(addmm_23, [256, 128, 256]);  addmm_23 = None
        add_29 = torch.ops.aten.add.Tensor(add_28, view_91);  add_28 = view_91 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg65_1);  mul_20 = arg65_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_21, arg66_1);  mul_21 = arg66_1 = None
        view_92 = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_44 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg68_1, view_92, permute_44);  arg68_1 = view_92 = permute_44 = None
        view_93 = torch.ops.aten.view.default(addmm_24, [256, 128, 256]);  addmm_24 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_93, 0.125);  view_93 = None
        view_94 = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_45 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg70_1, view_94, permute_45);  arg70_1 = view_94 = permute_45 = None
        view_95 = torch.ops.aten.view.default(addmm_25, [256, 128, 256]);  addmm_25 = None
        view_96 = torch.ops.aten.view.default(view_95, [256, -1, 4, 64]);  view_95 = None
        permute_46 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_33 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_97 = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_47 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg72_1, view_97, permute_47);  arg72_1 = view_97 = permute_47 = None
        view_98 = torch.ops.aten.view.default(addmm_26, [256, 128, 256]);  addmm_26 = None
        view_99 = torch.ops.aten.view.default(view_98, [256, -1, 4, 64]);  view_98 = None
        permute_48 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_34 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_100 = torch.ops.aten.view.default(mul_22, [256, 128, 4, 64]);  mul_22 = None
        permute_49 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        clone_35 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_101 = torch.ops.aten.view.default(clone_35, [1024, -1, 64]);  clone_35 = None
        view_102 = torch.ops.aten.view.default(clone_33, [1024, -1, 64])
        view_103 = torch.ops.aten.view.default(clone_34, [1024, -1, 64])
        permute_50 = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        bmm_8 = torch.ops.aten.bmm.default(view_101, permute_50);  view_101 = permute_50 = None
        view_104 = torch.ops.aten.view.default(bmm_8, [256, 4, 128, 128]);  bmm_8 = None
        add_32 = torch.ops.aten.add.Tensor(view_104, expand_1);  view_104 = None
        view_105 = torch.ops.aten.view.default(add_32, [1024, 128, 128]);  add_32 = None
        amax_4 = torch.ops.aten.amax.default(view_105, [-1], True)
        sub_12 = torch.ops.aten.sub.Tensor(view_105, amax_4);  view_105 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        bmm_9 = torch.ops.aten.bmm.default(div_4, view_103);  div_4 = view_103 = None
        view_106 = torch.ops.aten.view.default(bmm_9, [256, 4, 128, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_37 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_107 = torch.ops.aten.view.default(clone_37, [256, 128, 256]);  clone_37 = None
        view_108 = torch.ops.aten.view.default(view_107, [32768, 256]);  view_107 = None
        permute_52 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg74_1, view_108, permute_52);  arg74_1 = view_108 = permute_52 = None
        view_109 = torch.ops.aten.view.default(addmm_27, [256, 128, 256]);  addmm_27 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, view_109);  add_31 = view_109 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, arg75_1);  mul_23 = arg75_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_24, arg76_1);  mul_24 = arg76_1 = None
        view_110 = torch.ops.aten.view.default(add_35, [32768, 256])
        permute_53 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg78_1, view_110, permute_53);  arg78_1 = view_110 = permute_53 = None
        view_111 = torch.ops.aten.view.default(addmm_28, [256, 128, 2048]);  addmm_28 = None
        relu_4 = torch.ops.aten.relu.default(view_111);  view_111 = None
        view_112 = torch.ops.aten.view.default(relu_4, [32768, 2048]);  relu_4 = None
        permute_54 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg80_1, view_112, permute_54);  arg80_1 = view_112 = permute_54 = None
        view_113 = torch.ops.aten.view.default(addmm_29, [256, 128, 256]);  addmm_29 = None
        add_36 = torch.ops.aten.add.Tensor(add_35, view_113);  add_35 = view_113 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_36, getitem_19);  add_36 = getitem_19 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg81_1);  mul_25 = arg81_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_26, arg82_1);  mul_26 = arg82_1 = None
        view_114 = torch.ops.aten.view.default(add_38, [32768, 256])
        permute_55 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg84_1, view_114, permute_55);  arg84_1 = view_114 = permute_55 = None
        view_115 = torch.ops.aten.view.default(addmm_30, [256, 128, 256]);  addmm_30 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_115, 0.125);  view_115 = None
        view_116 = torch.ops.aten.view.default(add_38, [32768, 256])
        permute_56 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg86_1, view_116, permute_56);  arg86_1 = view_116 = permute_56 = None
        view_117 = torch.ops.aten.view.default(addmm_31, [256, 128, 256]);  addmm_31 = None
        view_118 = torch.ops.aten.view.default(view_117, [256, -1, 4, 64]);  view_117 = None
        permute_57 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_41 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_119 = torch.ops.aten.view.default(add_38, [32768, 256])
        permute_58 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg88_1, view_119, permute_58);  arg88_1 = view_119 = permute_58 = None
        view_120 = torch.ops.aten.view.default(addmm_32, [256, 128, 256]);  addmm_32 = None
        view_121 = torch.ops.aten.view.default(view_120, [256, -1, 4, 64]);  view_120 = None
        permute_59 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_42 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_122 = torch.ops.aten.view.default(mul_27, [256, 128, 4, 64]);  mul_27 = None
        permute_60 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_43 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_123 = torch.ops.aten.view.default(clone_43, [1024, -1, 64]);  clone_43 = None
        view_124 = torch.ops.aten.view.default(clone_41, [1024, -1, 64])
        view_125 = torch.ops.aten.view.default(clone_42, [1024, -1, 64])
        permute_61 = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        bmm_10 = torch.ops.aten.bmm.default(view_123, permute_61);  view_123 = permute_61 = None
        view_126 = torch.ops.aten.view.default(bmm_10, [256, 4, 128, 128]);  bmm_10 = None
        add_39 = torch.ops.aten.add.Tensor(view_126, expand_1);  view_126 = expand_1 = None
        view_127 = torch.ops.aten.view.default(add_39, [1024, 128, 128]);  add_39 = None
        amax_5 = torch.ops.aten.amax.default(view_127, [-1], True)
        sub_15 = torch.ops.aten.sub.Tensor(view_127, amax_5);  view_127 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        bmm_11 = torch.ops.aten.bmm.default(div_5, view_125);  div_5 = view_125 = None
        view_128 = torch.ops.aten.view.default(bmm_11, [256, 4, 128, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        clone_45 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_129 = torch.ops.aten.view.default(clone_45, [256, 128, 256]);  clone_45 = None
        view_130 = torch.ops.aten.view.default(view_129, [32768, 256]);  view_129 = None
        permute_63 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg90_1, view_130, permute_63);  arg90_1 = view_130 = permute_63 = None
        view_131 = torch.ops.aten.view.default(addmm_33, [256, 128, 256]);  addmm_33 = None
        add_40 = torch.ops.aten.add.Tensor(add_38, view_131);  add_38 = view_131 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_41 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_40, getitem_21);  add_40 = getitem_21 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg91_1);  mul_28 = arg91_1 = None
        add_42 = torch.ops.aten.add.Tensor(mul_29, arg92_1);  mul_29 = arg92_1 = None
        view_132 = torch.ops.aten.view.default(add_42, [32768, 256])
        permute_64 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg94_1, view_132, permute_64);  arg94_1 = view_132 = permute_64 = None
        view_133 = torch.ops.aten.view.default(addmm_34, [256, 128, 2048]);  addmm_34 = None
        relu_5 = torch.ops.aten.relu.default(view_133);  view_133 = None
        view_134 = torch.ops.aten.view.default(relu_5, [32768, 2048]);  relu_5 = None
        permute_65 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg96_1, view_134, permute_65);  arg96_1 = view_134 = permute_65 = None
        view_135 = torch.ops.aten.view.default(addmm_35, [256, 128, 256]);  addmm_35 = None
        add_43 = torch.ops.aten.add.Tensor(add_42, view_135);  add_42 = view_135 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_43, getitem_23);  add_43 = getitem_23 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg97_1);  mul_30 = arg97_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_31, arg98_1);  mul_31 = arg98_1 = None
        permute_66 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_136 = torch.ops.aten.view.default(add_45, [32768, 256]);  add_45 = None
        mm = torch.ops.aten.mm.default(view_136, permute_66);  view_136 = permute_66 = None
        view_137 = torch.ops.aten.view.default(mm, [256, 128, 10000]);  mm = None
        view_138 = torch.ops.aten.view.default(view_137, [-1, 10000])
        view_139 = torch.ops.aten.view.default(arg99_1, [-1]);  arg99_1 = None
        amax_6 = torch.ops.aten.amax.default(view_138, [1], True)
        sub_18 = torch.ops.aten.sub.Tensor(view_138, amax_6);  view_138 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_18)
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_19 = torch.ops.aten.sub.Tensor(sub_18, log);  sub_18 = log = None
        ne_1 = torch.ops.aten.ne.Scalar(view_139, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, view_139, full_default_2);  ne_1 = full_default_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_19, 1, unsqueeze_4);  sub_19 = unsqueeze_4 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_2 = torch.ops.aten.ne.Scalar(view_139, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_2, neg, full_default_3);  ne_2 = neg = full_default_3 = None
        ne_3 = torch.ops.aten.ne.Scalar(view_139, -100);  view_139 = None
        sum_8 = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        sum_9 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_6 = torch.ops.aten.div.Tensor(sum_9, convert_element_type_3);  sum_9 = convert_element_type_3 = None
        return (div_6, view_137, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42)
        
def load_args(reader):
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (256, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 10240000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (10000, 256), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1050624, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1026, 256), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf3, (256, 256), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (256,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256, 256), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf6, (256,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (256, 256), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf8, (256,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf9, (256, 256), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf13, (2048, 256), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf14, (2048,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256, 2048), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf16, (256,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf18, (256,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf19, (256, 256), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf20, (256,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 256), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256, 256), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256, 256), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf29, (2048, 256), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf30, (2048,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256, 2048), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256, 256), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256, 256), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256, 256), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 256), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2048, 256), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2048,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256, 2048), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (256,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256, 256), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256, 256), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256, 256), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256, 256), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf61, (2048, 256), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf62, (2048,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf63, (256, 2048), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (256, 256), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf68, (256,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (256, 256), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf70, (256,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf71, (256, 256), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256, 256), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf76, (256,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf77, (2048, 256), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf78, (2048,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf79, (256, 2048), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf80, (256,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf82, (256,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (256, 256), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf84, (256,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256, 256), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf86, (256,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf87, (256, 256), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf88, (256,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf89, (256, 256), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf93, (2048, 256), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf94, (2048,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf95, (256, 2048), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf96, (256,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf97, (256,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf98, (256,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf99, (256, 128), dtype=torch.int64, is_leaf=True)  # arg99_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)