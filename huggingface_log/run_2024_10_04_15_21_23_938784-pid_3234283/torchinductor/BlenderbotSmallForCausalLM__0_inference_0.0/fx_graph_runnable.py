
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 128]);  arg0_1 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, view, 0);  view = None
        mul = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        full_default = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota, 1)
        view_1 = torch.ops.aten.view.default(add, [128, 1]);  add = None
        lt = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        iota_1 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, iota_1);  arg2_1 = iota_1 = None
        var_mean = torch.ops.aten.var_mean.correction(mul, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(mul, getitem_1);  mul = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, embedding_1);  add_2 = embedding_1 = None
        view_2 = torch.ops.aten.view.default(add_3, [8192, 512])
        permute = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3 = torch.ops.aten.view.default(addmm, [64, 128, 512]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_3, 0.1767766952966369);  view_3 = None
        view_4 = torch.ops.aten.view.default(add_3, [8192, 512])
        permute_1 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, view_4, permute_1);  arg8_1 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [64, 128, 512]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [64, -1, 16, 32]);  view_5 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7 = torch.ops.aten.view.default(add_3, [8192, 512])
        permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg10_1, view_7, permute_3);  arg10_1 = view_7 = permute_3 = None
        view_8 = torch.ops.aten.view.default(addmm_2, [64, 128, 512]);  addmm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [64, -1, 16, 32]);  view_8 = None
        permute_4 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_10 = torch.ops.aten.view.default(mul_3, [64, 128, 16, 32]);  mul_3 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11 = torch.ops.aten.view.default(clone_3, [1024, -1, 32]);  clone_3 = None
        view_12 = torch.ops.aten.view.default(clone_1, [1024, -1, 32])
        view_13 = torch.ops.aten.view.default(clone_2, [1024, -1, 32])
        permute_6 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
        view_14 = torch.ops.aten.view.default(bmm, [64, 16, 128, 128]);  bmm = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [64, 1, 128, 128]);  unsqueeze_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_14, expand_1);  view_14 = None
        view_15 = torch.ops.aten.view.default(add_4, [1024, 128, 128]);  add_4 = None
        amax = torch.ops.aten.amax.default(view_15, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm_1 = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
        view_16 = torch.ops.aten.view.default(bmm_1, [64, 16, 128, 32]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17 = torch.ops.aten.view.default(clone_5, [64, 128, 512]);  clone_5 = None
        view_18 = torch.ops.aten.view.default(view_17, [8192, 512]);  view_17 = None
        permute_8 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg12_1, view_18, permute_8);  arg12_1 = view_18 = permute_8 = None
        view_19 = torch.ops.aten.view.default(addmm_3, [64, 128, 512]);  addmm_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, view_19);  add_3 = view_19 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        view_20 = torch.ops.aten.view.default(add_7, [8192, 512])
        permute_9 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg16_1, view_20, permute_9);  arg16_1 = view_20 = permute_9 = None
        view_21 = torch.ops.aten.view.default(addmm_4, [64, 128, 2048]);  addmm_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_7 = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
        view_22 = torch.ops.aten.view.default(mul_8, [8192, 2048]);  mul_8 = None
        permute_10 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg18_1, view_22, permute_10);  arg18_1 = view_22 = permute_10 = None
        view_23 = torch.ops.aten.view.default(addmm_5, [64, 128, 512]);  addmm_5 = None
        add_9 = torch.ops.aten.add.Tensor(add_7, view_23);  add_7 = view_23 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        view_24 = torch.ops.aten.view.default(add_11, [8192, 512])
        permute_11 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg22_1, view_24, permute_11);  arg22_1 = view_24 = permute_11 = None
        view_25 = torch.ops.aten.view.default(addmm_6, [64, 128, 512]);  addmm_6 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_25, 0.1767766952966369);  view_25 = None
        view_26 = torch.ops.aten.view.default(add_11, [8192, 512])
        permute_12 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg24_1, view_26, permute_12);  arg24_1 = view_26 = permute_12 = None
        view_27 = torch.ops.aten.view.default(addmm_7, [64, 128, 512]);  addmm_7 = None
        view_28 = torch.ops.aten.view.default(view_27, [64, -1, 16, 32]);  view_27 = None
        permute_13 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_9 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_29 = torch.ops.aten.view.default(add_11, [8192, 512])
        permute_14 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg26_1, view_29, permute_14);  arg26_1 = view_29 = permute_14 = None
        view_30 = torch.ops.aten.view.default(addmm_8, [64, 128, 512]);  addmm_8 = None
        view_31 = torch.ops.aten.view.default(view_30, [64, -1, 16, 32]);  view_30 = None
        permute_15 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        clone_10 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_32 = torch.ops.aten.view.default(mul_11, [64, 128, 16, 32]);  mul_11 = None
        permute_16 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_11 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_33 = torch.ops.aten.view.default(clone_11, [1024, -1, 32]);  clone_11 = None
        view_34 = torch.ops.aten.view.default(clone_9, [1024, -1, 32])
        view_35 = torch.ops.aten.view.default(clone_10, [1024, -1, 32])
        permute_17 = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        bmm_2 = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
        view_36 = torch.ops.aten.view.default(bmm_2, [64, 16, 128, 128]);  bmm_2 = None
        add_12 = torch.ops.aten.add.Tensor(view_36, expand_1);  view_36 = None
        view_37 = torch.ops.aten.view.default(add_12, [1024, 128, 128]);  add_12 = None
        amax_1 = torch.ops.aten.amax.default(view_37, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        bmm_3 = torch.ops.aten.bmm.default(div_1, view_35);  div_1 = view_35 = None
        view_38 = torch.ops.aten.view.default(bmm_3, [64, 16, 128, 32]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_13 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39 = torch.ops.aten.view.default(clone_13, [64, 128, 512]);  clone_13 = None
        view_40 = torch.ops.aten.view.default(view_39, [8192, 512]);  view_39 = None
        permute_19 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg28_1, view_40, permute_19);  arg28_1 = view_40 = permute_19 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [64, 128, 512]);  addmm_9 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, view_41);  add_11 = view_41 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        view_42 = torch.ops.aten.view.default(add_15, [8192, 512])
        permute_20 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg32_1, view_42, permute_20);  arg32_1 = view_42 = permute_20 = None
        view_43 = torch.ops.aten.view.default(addmm_10, [64, 128, 2048]);  addmm_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_16);  mul_14 = add_16 = None
        view_44 = torch.ops.aten.view.default(mul_16, [8192, 2048]);  mul_16 = None
        permute_21 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg34_1, view_44, permute_21);  arg34_1 = view_44 = permute_21 = None
        view_45 = torch.ops.aten.view.default(addmm_11, [64, 128, 512]);  addmm_11 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, view_45);  add_15 = view_45 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        view_46 = torch.ops.aten.view.default(add_19, [8192, 512])
        permute_22 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg38_1, view_46, permute_22);  arg38_1 = view_46 = permute_22 = None
        view_47 = torch.ops.aten.view.default(addmm_12, [64, 128, 512]);  addmm_12 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_47, 0.1767766952966369);  view_47 = None
        view_48 = torch.ops.aten.view.default(add_19, [8192, 512])
        permute_23 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg40_1, view_48, permute_23);  arg40_1 = view_48 = permute_23 = None
        view_49 = torch.ops.aten.view.default(addmm_13, [64, 128, 512]);  addmm_13 = None
        view_50 = torch.ops.aten.view.default(view_49, [64, -1, 16, 32]);  view_49 = None
        permute_24 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_17 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_51 = torch.ops.aten.view.default(add_19, [8192, 512])
        permute_25 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg42_1, view_51, permute_25);  arg42_1 = view_51 = permute_25 = None
        view_52 = torch.ops.aten.view.default(addmm_14, [64, 128, 512]);  addmm_14 = None
        view_53 = torch.ops.aten.view.default(view_52, [64, -1, 16, 32]);  view_52 = None
        permute_26 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_18 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_54 = torch.ops.aten.view.default(mul_19, [64, 128, 16, 32]);  mul_19 = None
        permute_27 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_55 = torch.ops.aten.view.default(clone_19, [1024, -1, 32]);  clone_19 = None
        view_56 = torch.ops.aten.view.default(clone_17, [1024, -1, 32])
        view_57 = torch.ops.aten.view.default(clone_18, [1024, -1, 32])
        permute_28 = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        bmm_4 = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
        view_58 = torch.ops.aten.view.default(bmm_4, [64, 16, 128, 128]);  bmm_4 = None
        add_20 = torch.ops.aten.add.Tensor(view_58, expand_1);  view_58 = None
        view_59 = torch.ops.aten.view.default(add_20, [1024, 128, 128]);  add_20 = None
        amax_2 = torch.ops.aten.amax.default(view_59, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        bmm_5 = torch.ops.aten.bmm.default(div_2, view_57);  div_2 = view_57 = None
        view_60 = torch.ops.aten.view.default(bmm_5, [64, 16, 128, 32]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_21 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_61 = torch.ops.aten.view.default(clone_21, [64, 128, 512]);  clone_21 = None
        view_62 = torch.ops.aten.view.default(view_61, [8192, 512]);  view_61 = None
        permute_30 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg44_1, view_62, permute_30);  arg44_1 = view_62 = permute_30 = None
        view_63 = torch.ops.aten.view.default(addmm_15, [64, 128, 512]);  addmm_15 = None
        add_21 = torch.ops.aten.add.Tensor(add_19, view_63);  add_19 = view_63 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        view_64 = torch.ops.aten.view.default(add_23, [8192, 512])
        permute_31 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg48_1, view_64, permute_31);  arg48_1 = view_64 = permute_31 = None
        view_65 = torch.ops.aten.view.default(addmm_16, [64, 128, 2048]);  addmm_16 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_24);  mul_22 = add_24 = None
        view_66 = torch.ops.aten.view.default(mul_24, [8192, 2048]);  mul_24 = None
        permute_32 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg50_1, view_66, permute_32);  arg50_1 = view_66 = permute_32 = None
        view_67 = torch.ops.aten.view.default(addmm_17, [64, 128, 512]);  addmm_17 = None
        add_25 = torch.ops.aten.add.Tensor(add_23, view_67);  add_23 = view_67 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        view_68 = torch.ops.aten.view.default(add_27, [8192, 512])
        permute_33 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg54_1, view_68, permute_33);  arg54_1 = view_68 = permute_33 = None
        view_69 = torch.ops.aten.view.default(addmm_18, [64, 128, 512]);  addmm_18 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_69, 0.1767766952966369);  view_69 = None
        view_70 = torch.ops.aten.view.default(add_27, [8192, 512])
        permute_34 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg56_1, view_70, permute_34);  arg56_1 = view_70 = permute_34 = None
        view_71 = torch.ops.aten.view.default(addmm_19, [64, 128, 512]);  addmm_19 = None
        view_72 = torch.ops.aten.view.default(view_71, [64, -1, 16, 32]);  view_71 = None
        permute_35 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_73 = torch.ops.aten.view.default(add_27, [8192, 512])
        permute_36 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg58_1, view_73, permute_36);  arg58_1 = view_73 = permute_36 = None
        view_74 = torch.ops.aten.view.default(addmm_20, [64, 128, 512]);  addmm_20 = None
        view_75 = torch.ops.aten.view.default(view_74, [64, -1, 16, 32]);  view_74 = None
        permute_37 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_26 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_76 = torch.ops.aten.view.default(mul_27, [64, 128, 16, 32]);  mul_27 = None
        permute_38 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_27 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_77 = torch.ops.aten.view.default(clone_27, [1024, -1, 32]);  clone_27 = None
        view_78 = torch.ops.aten.view.default(clone_25, [1024, -1, 32])
        view_79 = torch.ops.aten.view.default(clone_26, [1024, -1, 32])
        permute_39 = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        bmm_6 = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
        view_80 = torch.ops.aten.view.default(bmm_6, [64, 16, 128, 128]);  bmm_6 = None
        add_28 = torch.ops.aten.add.Tensor(view_80, expand_1);  view_80 = None
        view_81 = torch.ops.aten.view.default(add_28, [1024, 128, 128]);  add_28 = None
        amax_3 = torch.ops.aten.amax.default(view_81, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        bmm_7 = torch.ops.aten.bmm.default(div_3, view_79);  div_3 = view_79 = None
        view_82 = torch.ops.aten.view.default(bmm_7, [64, 16, 128, 32]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_29 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_83 = torch.ops.aten.view.default(clone_29, [64, 128, 512]);  clone_29 = None
        view_84 = torch.ops.aten.view.default(view_83, [8192, 512]);  view_83 = None
        permute_41 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg60_1, view_84, permute_41);  arg60_1 = view_84 = permute_41 = None
        view_85 = torch.ops.aten.view.default(addmm_21, [64, 128, 512]);  addmm_21 = None
        add_29 = torch.ops.aten.add.Tensor(add_27, view_85);  add_27 = view_85 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        view_86 = torch.ops.aten.view.default(add_31, [8192, 512])
        permute_42 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg64_1, view_86, permute_42);  arg64_1 = view_86 = permute_42 = None
        view_87 = torch.ops.aten.view.default(addmm_22, [64, 128, 2048]);  addmm_22 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_32);  mul_30 = add_32 = None
        view_88 = torch.ops.aten.view.default(mul_32, [8192, 2048]);  mul_32 = None
        permute_43 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg66_1, view_88, permute_43);  arg66_1 = view_88 = permute_43 = None
        view_89 = torch.ops.aten.view.default(addmm_23, [64, 128, 512]);  addmm_23 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, view_89);  add_31 = view_89 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        view_90 = torch.ops.aten.view.default(add_35, [8192, 512])
        permute_44 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg70_1, view_90, permute_44);  arg70_1 = view_90 = permute_44 = None
        view_91 = torch.ops.aten.view.default(addmm_24, [64, 128, 512]);  addmm_24 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_91, 0.1767766952966369);  view_91 = None
        view_92 = torch.ops.aten.view.default(add_35, [8192, 512])
        permute_45 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg72_1, view_92, permute_45);  arg72_1 = view_92 = permute_45 = None
        view_93 = torch.ops.aten.view.default(addmm_25, [64, 128, 512]);  addmm_25 = None
        view_94 = torch.ops.aten.view.default(view_93, [64, -1, 16, 32]);  view_93 = None
        permute_46 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_33 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_95 = torch.ops.aten.view.default(add_35, [8192, 512])
        permute_47 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg74_1, view_95, permute_47);  arg74_1 = view_95 = permute_47 = None
        view_96 = torch.ops.aten.view.default(addmm_26, [64, 128, 512]);  addmm_26 = None
        view_97 = torch.ops.aten.view.default(view_96, [64, -1, 16, 32]);  view_96 = None
        permute_48 = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        clone_34 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_98 = torch.ops.aten.view.default(mul_35, [64, 128, 16, 32]);  mul_35 = None
        permute_49 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_35 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_99 = torch.ops.aten.view.default(clone_35, [1024, -1, 32]);  clone_35 = None
        view_100 = torch.ops.aten.view.default(clone_33, [1024, -1, 32])
        view_101 = torch.ops.aten.view.default(clone_34, [1024, -1, 32])
        permute_50 = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        bmm_8 = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
        view_102 = torch.ops.aten.view.default(bmm_8, [64, 16, 128, 128]);  bmm_8 = None
        add_36 = torch.ops.aten.add.Tensor(view_102, expand_1);  view_102 = None
        view_103 = torch.ops.aten.view.default(add_36, [1024, 128, 128]);  add_36 = None
        amax_4 = torch.ops.aten.amax.default(view_103, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        bmm_9 = torch.ops.aten.bmm.default(div_4, view_101);  div_4 = view_101 = None
        view_104 = torch.ops.aten.view.default(bmm_9, [64, 16, 128, 32]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_37 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_105 = torch.ops.aten.view.default(clone_37, [64, 128, 512]);  clone_37 = None
        view_106 = torch.ops.aten.view.default(view_105, [8192, 512]);  view_105 = None
        permute_52 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg76_1, view_106, permute_52);  arg76_1 = view_106 = permute_52 = None
        view_107 = torch.ops.aten.view.default(addmm_27, [64, 128, 512]);  addmm_27 = None
        add_37 = torch.ops.aten.add.Tensor(add_35, view_107);  add_35 = view_107 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        view_108 = torch.ops.aten.view.default(add_39, [8192, 512])
        permute_53 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg80_1, view_108, permute_53);  arg80_1 = view_108 = permute_53 = None
        view_109 = torch.ops.aten.view.default(addmm_28, [64, 128, 2048]);  addmm_28 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_39 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4 = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, add_40);  mul_38 = add_40 = None
        view_110 = torch.ops.aten.view.default(mul_40, [8192, 2048]);  mul_40 = None
        permute_54 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg82_1, view_110, permute_54);  arg82_1 = view_110 = permute_54 = None
        view_111 = torch.ops.aten.view.default(addmm_29, [64, 128, 512]);  addmm_29 = None
        add_41 = torch.ops.aten.add.Tensor(add_39, view_111);  add_39 = view_111 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        view_112 = torch.ops.aten.view.default(add_43, [8192, 512])
        permute_55 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg86_1, view_112, permute_55);  arg86_1 = view_112 = permute_55 = None
        view_113 = torch.ops.aten.view.default(addmm_30, [64, 128, 512]);  addmm_30 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_113, 0.1767766952966369);  view_113 = None
        view_114 = torch.ops.aten.view.default(add_43, [8192, 512])
        permute_56 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg88_1, view_114, permute_56);  arg88_1 = view_114 = permute_56 = None
        view_115 = torch.ops.aten.view.default(addmm_31, [64, 128, 512]);  addmm_31 = None
        view_116 = torch.ops.aten.view.default(view_115, [64, -1, 16, 32]);  view_115 = None
        permute_57 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_41 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_117 = torch.ops.aten.view.default(add_43, [8192, 512])
        permute_58 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg90_1, view_117, permute_58);  arg90_1 = view_117 = permute_58 = None
        view_118 = torch.ops.aten.view.default(addmm_32, [64, 128, 512]);  addmm_32 = None
        view_119 = torch.ops.aten.view.default(view_118, [64, -1, 16, 32]);  view_118 = None
        permute_59 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        clone_42 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_120 = torch.ops.aten.view.default(mul_43, [64, 128, 16, 32]);  mul_43 = None
        permute_60 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_43 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_121 = torch.ops.aten.view.default(clone_43, [1024, -1, 32]);  clone_43 = None
        view_122 = torch.ops.aten.view.default(clone_41, [1024, -1, 32])
        view_123 = torch.ops.aten.view.default(clone_42, [1024, -1, 32])
        permute_61 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        bmm_10 = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
        view_124 = torch.ops.aten.view.default(bmm_10, [64, 16, 128, 128]);  bmm_10 = None
        add_44 = torch.ops.aten.add.Tensor(view_124, expand_1);  view_124 = None
        view_125 = torch.ops.aten.view.default(add_44, [1024, 128, 128]);  add_44 = None
        amax_5 = torch.ops.aten.amax.default(view_125, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        bmm_11 = torch.ops.aten.bmm.default(div_5, view_123);  div_5 = view_123 = None
        view_126 = torch.ops.aten.view.default(bmm_11, [64, 16, 128, 32]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_45 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_127 = torch.ops.aten.view.default(clone_45, [64, 128, 512]);  clone_45 = None
        view_128 = torch.ops.aten.view.default(view_127, [8192, 512]);  view_127 = None
        permute_63 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg92_1, view_128, permute_63);  arg92_1 = view_128 = permute_63 = None
        view_129 = torch.ops.aten.view.default(addmm_33, [64, 128, 512]);  addmm_33 = None
        add_45 = torch.ops.aten.add.Tensor(add_43, view_129);  add_43 = view_129 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        view_130 = torch.ops.aten.view.default(add_47, [8192, 512])
        permute_64 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg96_1, view_130, permute_64);  arg96_1 = view_130 = permute_64 = None
        view_131 = torch.ops.aten.view.default(addmm_34, [64, 128, 2048]);  addmm_34 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
        view_132 = torch.ops.aten.view.default(mul_48, [8192, 2048]);  mul_48 = None
        permute_65 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg98_1, view_132, permute_65);  arg98_1 = view_132 = permute_65 = None
        view_133 = torch.ops.aten.view.default(addmm_35, [64, 128, 512]);  addmm_35 = None
        add_49 = torch.ops.aten.add.Tensor(add_47, view_133);  add_47 = view_133 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        view_134 = torch.ops.aten.view.default(add_51, [8192, 512])
        permute_66 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg102_1, view_134, permute_66);  arg102_1 = view_134 = permute_66 = None
        view_135 = torch.ops.aten.view.default(addmm_36, [64, 128, 512]);  addmm_36 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_135, 0.1767766952966369);  view_135 = None
        view_136 = torch.ops.aten.view.default(add_51, [8192, 512])
        permute_67 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg104_1, view_136, permute_67);  arg104_1 = view_136 = permute_67 = None
        view_137 = torch.ops.aten.view.default(addmm_37, [64, 128, 512]);  addmm_37 = None
        view_138 = torch.ops.aten.view.default(view_137, [64, -1, 16, 32]);  view_137 = None
        permute_68 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_49 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_139 = torch.ops.aten.view.default(add_51, [8192, 512])
        permute_69 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg106_1, view_139, permute_69);  arg106_1 = view_139 = permute_69 = None
        view_140 = torch.ops.aten.view.default(addmm_38, [64, 128, 512]);  addmm_38 = None
        view_141 = torch.ops.aten.view.default(view_140, [64, -1, 16, 32]);  view_140 = None
        permute_70 = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_50 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_142 = torch.ops.aten.view.default(mul_51, [64, 128, 16, 32]);  mul_51 = None
        permute_71 = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_51 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_143 = torch.ops.aten.view.default(clone_51, [1024, -1, 32]);  clone_51 = None
        view_144 = torch.ops.aten.view.default(clone_49, [1024, -1, 32])
        view_145 = torch.ops.aten.view.default(clone_50, [1024, -1, 32])
        permute_72 = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        bmm_12 = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
        view_146 = torch.ops.aten.view.default(bmm_12, [64, 16, 128, 128]);  bmm_12 = None
        add_52 = torch.ops.aten.add.Tensor(view_146, expand_1);  view_146 = None
        view_147 = torch.ops.aten.view.default(add_52, [1024, 128, 128]);  add_52 = None
        amax_6 = torch.ops.aten.amax.default(view_147, [-1], True)
        sub_19 = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        bmm_13 = torch.ops.aten.bmm.default(div_6, view_145);  div_6 = view_145 = None
        view_148 = torch.ops.aten.view.default(bmm_13, [64, 16, 128, 32]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_53 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_149 = torch.ops.aten.view.default(clone_53, [64, 128, 512]);  clone_53 = None
        view_150 = torch.ops.aten.view.default(view_149, [8192, 512]);  view_149 = None
        permute_74 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg108_1, view_150, permute_74);  arg108_1 = view_150 = permute_74 = None
        view_151 = torch.ops.aten.view.default(addmm_39, [64, 128, 512]);  addmm_39 = None
        add_53 = torch.ops.aten.add.Tensor(add_51, view_151);  add_51 = view_151 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        view_152 = torch.ops.aten.view.default(add_55, [8192, 512])
        permute_75 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg112_1, view_152, permute_75);  arg112_1 = view_152 = permute_75 = None
        view_153 = torch.ops.aten.view.default(addmm_40, [64, 128, 2048]);  addmm_40 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_56);  mul_54 = add_56 = None
        view_154 = torch.ops.aten.view.default(mul_56, [8192, 2048]);  mul_56 = None
        permute_76 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg114_1, view_154, permute_76);  arg114_1 = view_154 = permute_76 = None
        view_155 = torch.ops.aten.view.default(addmm_41, [64, 128, 512]);  addmm_41 = None
        add_57 = torch.ops.aten.add.Tensor(add_55, view_155);  add_55 = view_155 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        view_156 = torch.ops.aten.view.default(add_59, [8192, 512])
        permute_77 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg118_1, view_156, permute_77);  arg118_1 = view_156 = permute_77 = None
        view_157 = torch.ops.aten.view.default(addmm_42, [64, 128, 512]);  addmm_42 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_157, 0.1767766952966369);  view_157 = None
        view_158 = torch.ops.aten.view.default(add_59, [8192, 512])
        permute_78 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg120_1, view_158, permute_78);  arg120_1 = view_158 = permute_78 = None
        view_159 = torch.ops.aten.view.default(addmm_43, [64, 128, 512]);  addmm_43 = None
        view_160 = torch.ops.aten.view.default(view_159, [64, -1, 16, 32]);  view_159 = None
        permute_79 = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_57 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_161 = torch.ops.aten.view.default(add_59, [8192, 512])
        permute_80 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg122_1, view_161, permute_80);  arg122_1 = view_161 = permute_80 = None
        view_162 = torch.ops.aten.view.default(addmm_44, [64, 128, 512]);  addmm_44 = None
        view_163 = torch.ops.aten.view.default(view_162, [64, -1, 16, 32]);  view_162 = None
        permute_81 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_58 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_164 = torch.ops.aten.view.default(mul_59, [64, 128, 16, 32]);  mul_59 = None
        permute_82 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_59 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_165 = torch.ops.aten.view.default(clone_59, [1024, -1, 32]);  clone_59 = None
        view_166 = torch.ops.aten.view.default(clone_57, [1024, -1, 32])
        view_167 = torch.ops.aten.view.default(clone_58, [1024, -1, 32])
        permute_83 = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        bmm_14 = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
        view_168 = torch.ops.aten.view.default(bmm_14, [64, 16, 128, 128]);  bmm_14 = None
        add_60 = torch.ops.aten.add.Tensor(view_168, expand_1);  view_168 = expand_1 = None
        view_169 = torch.ops.aten.view.default(add_60, [1024, 128, 128]);  add_60 = None
        amax_7 = torch.ops.aten.amax.default(view_169, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        bmm_15 = torch.ops.aten.bmm.default(div_7, view_167);  div_7 = view_167 = None
        view_170 = torch.ops.aten.view.default(bmm_15, [64, 16, 128, 32]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_61 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_171 = torch.ops.aten.view.default(clone_61, [64, 128, 512]);  clone_61 = None
        view_172 = torch.ops.aten.view.default(view_171, [8192, 512]);  view_171 = None
        permute_85 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg124_1, view_172, permute_85);  arg124_1 = view_172 = permute_85 = None
        view_173 = torch.ops.aten.view.default(addmm_45, [64, 128, 512]);  addmm_45 = None
        add_61 = torch.ops.aten.add.Tensor(add_59, view_173);  add_59 = view_173 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        view_174 = torch.ops.aten.view.default(add_63, [8192, 512])
        permute_86 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg128_1, view_174, permute_86);  arg128_1 = view_174 = permute_86 = None
        view_175 = torch.ops.aten.view.default(addmm_46, [64, 128, 2048]);  addmm_46 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_63 = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7 = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_62, add_64);  mul_62 = add_64 = None
        view_176 = torch.ops.aten.view.default(mul_64, [8192, 2048]);  mul_64 = None
        permute_87 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg130_1, view_176, permute_87);  arg130_1 = view_176 = permute_87 = None
        view_177 = torch.ops.aten.view.default(addmm_47, [64, 128, 512]);  addmm_47 = None
        add_65 = torch.ops.aten.add.Tensor(add_63, view_177);  add_63 = view_177 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        permute_88 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_178 = torch.ops.aten.view.default(add_67, [8192, 512]);  add_67 = None
        full_default_4 = torch.ops.aten.full.default([512, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_88, full_default_4], 1);  permute_88 = full_default_4 = None
        mm_default = torch.ops.aten.mm.default(view_178, cat_default);  view_178 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_179 = torch.ops.aten.view.default(slice_tensor, [64, 128, 50265]);  slice_tensor = None
        view_180 = torch.ops.aten.view.default(view_179, [-1, 50265])
        view_181 = torch.ops.aten.view.default(arg133_1, [-1]);  arg133_1 = None
        amax_8 = torch.ops.aten.amax.default(view_180, [1], True)
        sub_25 = torch.ops.aten.sub.Tensor(view_180, amax_8);  view_180 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_25)
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log = torch.ops.aten.log.default(sum_9);  sum_9 = None
        sub_26 = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        ne = torch.ops.aten.ne.Scalar(view_181, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_181, full_default_2);  ne = full_default_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_4);  sub_26 = unsqueeze_4 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_181, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_181, -100);  view_181 = None
        sum_10 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None
        sum_11 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_8 = torch.ops.aten.div.Tensor(sum_11, convert_element_type);  sum_11 = convert_element_type = None
        return (div_8, view_179, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (64, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 102942720, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50265, 512), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 512), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf4, (512,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 512), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512, 512), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf8, (512,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512, 512), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512, 512), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf15, (2048, 512), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf16, (2048,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512, 2048), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512, 512), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512, 512), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512, 512), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512, 512), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf31, (2048, 512), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf32, (2048,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512, 2048), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512, 512), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512, 512), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf40, (512,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf41, (512, 512), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512, 512), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf47, (2048, 512), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf48, (2048,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512, 2048), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf53, (512, 512), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf54, (512,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512, 512), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512, 512), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf59, (512, 512), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf60, (512,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf63, (2048, 512), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf64, (2048,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512, 2048), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512, 512), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512, 512), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 512), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512, 512), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf79, (2048, 512), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf80, (2048,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512, 2048), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512, 512), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512, 512), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512, 512), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512, 512), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf94, (512,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf95, (2048, 512), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf96, (2048,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512, 2048), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512, 512), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf103, (512, 512), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512, 512), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf107, (512, 512), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf108, (512,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf109, (512,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf111, (2048, 512), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf112, (2048,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512, 2048), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512, 512), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512, 512), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf121, (512, 512), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf122, (512,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf123, (512, 512), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf124, (512,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf125, (512,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf126, (512,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf127, (2048, 512), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf128, (2048,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512, 2048), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf131, (512,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf133, (64, 128), dtype=torch.int64, is_leaf=True)  # arg133_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)