
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1):
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
        add_1 = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(add_1, getitem_1);  getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        view_2 = torch.ops.aten.view.default(add_3, [4096, 1024])
        permute = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3 = torch.ops.aten.view.default(addmm, [32, 128, 1024]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
        view_4 = torch.ops.aten.view.default(add_3, [4096, 1024])
        permute_1 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, view_4, permute_1);  arg8_1 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [32, 128, 1024]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [32, -1, 16, 64]);  view_5 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7 = torch.ops.aten.view.default(add_3, [4096, 1024]);  add_3 = None
        permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg10_1, view_7, permute_3);  arg10_1 = view_7 = permute_3 = None
        view_8 = torch.ops.aten.view.default(addmm_2, [32, 128, 1024]);  addmm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [32, -1, 16, 64]);  view_8 = None
        permute_4 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_10 = torch.ops.aten.view.default(mul_3, [32, 128, 16, 64]);  mul_3 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11 = torch.ops.aten.view.default(clone_3, [512, -1, 64]);  clone_3 = None
        view_12 = torch.ops.aten.view.default(clone_1, [512, -1, 64])
        view_13 = torch.ops.aten.view.default(clone_2, [512, -1, 64])
        permute_6 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
        view_14 = torch.ops.aten.view.default(bmm, [32, 16, 128, 128]);  bmm = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [32, 1, 128, 128]);  unsqueeze_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_14, expand_1);  view_14 = None
        view_15 = torch.ops.aten.view.default(add_4, [512, 128, 128]);  add_4 = None
        amax = torch.ops.aten.amax.default(view_15, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm_1 = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
        view_16 = torch.ops.aten.view.default(bmm_1, [32, 16, 128, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17 = torch.ops.aten.view.default(clone_5, [32, 128, 1024]);  clone_5 = None
        view_18 = torch.ops.aten.view.default(view_17, [4096, 1024]);  view_17 = None
        permute_8 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg12_1, view_18, permute_8);  arg12_1 = view_18 = permute_8 = None
        view_19 = torch.ops.aten.view.default(addmm_3, [32, 128, 1024]);  addmm_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        view_20 = torch.ops.aten.view.default(add_7, [4096, 1024]);  add_7 = None
        permute_9 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg16_1, view_20, permute_9);  arg16_1 = view_20 = permute_9 = None
        view_21 = torch.ops.aten.view.default(addmm_4, [32, 128, 4096]);  addmm_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_7 = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
        view_22 = torch.ops.aten.view.default(mul_8, [4096, 4096]);  mul_8 = None
        permute_10 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg18_1, view_22, permute_10);  arg18_1 = view_22 = permute_10 = None
        view_23 = torch.ops.aten.view.default(addmm_5, [32, 128, 1024]);  addmm_5 = None
        add_9 = torch.ops.aten.add.Tensor(add_5, view_23);  add_5 = view_23 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        view_24 = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_11 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg22_1, view_24, permute_11);  arg22_1 = view_24 = permute_11 = None
        view_25 = torch.ops.aten.view.default(addmm_6, [32, 128, 1024]);  addmm_6 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_25, 0.125);  view_25 = None
        view_26 = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_12 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg24_1, view_26, permute_12);  arg24_1 = view_26 = permute_12 = None
        view_27 = torch.ops.aten.view.default(addmm_7, [32, 128, 1024]);  addmm_7 = None
        view_28 = torch.ops.aten.view.default(view_27, [32, -1, 16, 64]);  view_27 = None
        permute_13 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_9 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_29 = torch.ops.aten.view.default(add_11, [4096, 1024]);  add_11 = None
        permute_14 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg26_1, view_29, permute_14);  arg26_1 = view_29 = permute_14 = None
        view_30 = torch.ops.aten.view.default(addmm_8, [32, 128, 1024]);  addmm_8 = None
        view_31 = torch.ops.aten.view.default(view_30, [32, -1, 16, 64]);  view_30 = None
        permute_15 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        clone_10 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_32 = torch.ops.aten.view.default(mul_11, [32, 128, 16, 64]);  mul_11 = None
        permute_16 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_11 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_33 = torch.ops.aten.view.default(clone_11, [512, -1, 64]);  clone_11 = None
        view_34 = torch.ops.aten.view.default(clone_9, [512, -1, 64])
        view_35 = torch.ops.aten.view.default(clone_10, [512, -1, 64])
        permute_17 = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        bmm_2 = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
        view_36 = torch.ops.aten.view.default(bmm_2, [32, 16, 128, 128]);  bmm_2 = None
        add_12 = torch.ops.aten.add.Tensor(view_36, expand_1);  view_36 = None
        view_37 = torch.ops.aten.view.default(add_12, [512, 128, 128]);  add_12 = None
        amax_1 = torch.ops.aten.amax.default(view_37, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        bmm_3 = torch.ops.aten.bmm.default(div_1, view_35);  div_1 = view_35 = None
        view_38 = torch.ops.aten.view.default(bmm_3, [32, 16, 128, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_13 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39 = torch.ops.aten.view.default(clone_13, [32, 128, 1024]);  clone_13 = None
        view_40 = torch.ops.aten.view.default(view_39, [4096, 1024]);  view_39 = None
        permute_19 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg28_1, view_40, permute_19);  arg28_1 = view_40 = permute_19 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [32, 128, 1024]);  addmm_9 = None
        add_13 = torch.ops.aten.add.Tensor(add_9, view_41);  add_9 = view_41 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        view_42 = torch.ops.aten.view.default(add_15, [4096, 1024]);  add_15 = None
        permute_20 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg32_1, view_42, permute_20);  arg32_1 = view_42 = permute_20 = None
        view_43 = torch.ops.aten.view.default(addmm_10, [32, 128, 4096]);  addmm_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_16);  mul_14 = add_16 = None
        view_44 = torch.ops.aten.view.default(mul_16, [4096, 4096]);  mul_16 = None
        permute_21 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg34_1, view_44, permute_21);  arg34_1 = view_44 = permute_21 = None
        view_45 = torch.ops.aten.view.default(addmm_11, [32, 128, 1024]);  addmm_11 = None
        add_17 = torch.ops.aten.add.Tensor(add_13, view_45);  add_13 = view_45 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        view_46 = torch.ops.aten.view.default(add_19, [4096, 1024])
        permute_22 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg38_1, view_46, permute_22);  arg38_1 = view_46 = permute_22 = None
        view_47 = torch.ops.aten.view.default(addmm_12, [32, 128, 1024]);  addmm_12 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_47, 0.125);  view_47 = None
        view_48 = torch.ops.aten.view.default(add_19, [4096, 1024])
        permute_23 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg40_1, view_48, permute_23);  arg40_1 = view_48 = permute_23 = None
        view_49 = torch.ops.aten.view.default(addmm_13, [32, 128, 1024]);  addmm_13 = None
        view_50 = torch.ops.aten.view.default(view_49, [32, -1, 16, 64]);  view_49 = None
        permute_24 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_17 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_51 = torch.ops.aten.view.default(add_19, [4096, 1024]);  add_19 = None
        permute_25 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg42_1, view_51, permute_25);  arg42_1 = view_51 = permute_25 = None
        view_52 = torch.ops.aten.view.default(addmm_14, [32, 128, 1024]);  addmm_14 = None
        view_53 = torch.ops.aten.view.default(view_52, [32, -1, 16, 64]);  view_52 = None
        permute_26 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_18 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_54 = torch.ops.aten.view.default(mul_19, [32, 128, 16, 64]);  mul_19 = None
        permute_27 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_55 = torch.ops.aten.view.default(clone_19, [512, -1, 64]);  clone_19 = None
        view_56 = torch.ops.aten.view.default(clone_17, [512, -1, 64])
        view_57 = torch.ops.aten.view.default(clone_18, [512, -1, 64])
        permute_28 = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        bmm_4 = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
        view_58 = torch.ops.aten.view.default(bmm_4, [32, 16, 128, 128]);  bmm_4 = None
        add_20 = torch.ops.aten.add.Tensor(view_58, expand_1);  view_58 = None
        view_59 = torch.ops.aten.view.default(add_20, [512, 128, 128]);  add_20 = None
        amax_2 = torch.ops.aten.amax.default(view_59, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        bmm_5 = torch.ops.aten.bmm.default(div_2, view_57);  div_2 = view_57 = None
        view_60 = torch.ops.aten.view.default(bmm_5, [32, 16, 128, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_21 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_61 = torch.ops.aten.view.default(clone_21, [32, 128, 1024]);  clone_21 = None
        view_62 = torch.ops.aten.view.default(view_61, [4096, 1024]);  view_61 = None
        permute_30 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg44_1, view_62, permute_30);  arg44_1 = view_62 = permute_30 = None
        view_63 = torch.ops.aten.view.default(addmm_15, [32, 128, 1024]);  addmm_15 = None
        add_21 = torch.ops.aten.add.Tensor(add_17, view_63);  add_17 = view_63 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_21, getitem_11);  getitem_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        view_64 = torch.ops.aten.view.default(add_23, [4096, 1024]);  add_23 = None
        permute_31 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg48_1, view_64, permute_31);  arg48_1 = view_64 = permute_31 = None
        view_65 = torch.ops.aten.view.default(addmm_16, [32, 128, 4096]);  addmm_16 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_24);  mul_22 = add_24 = None
        view_66 = torch.ops.aten.view.default(mul_24, [4096, 4096]);  mul_24 = None
        permute_32 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg50_1, view_66, permute_32);  arg50_1 = view_66 = permute_32 = None
        view_67 = torch.ops.aten.view.default(addmm_17, [32, 128, 1024]);  addmm_17 = None
        add_25 = torch.ops.aten.add.Tensor(add_21, view_67);  add_21 = view_67 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        view_68 = torch.ops.aten.view.default(add_27, [4096, 1024])
        permute_33 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg54_1, view_68, permute_33);  arg54_1 = view_68 = permute_33 = None
        view_69 = torch.ops.aten.view.default(addmm_18, [32, 128, 1024]);  addmm_18 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_69, 0.125);  view_69 = None
        view_70 = torch.ops.aten.view.default(add_27, [4096, 1024])
        permute_34 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg56_1, view_70, permute_34);  arg56_1 = view_70 = permute_34 = None
        view_71 = torch.ops.aten.view.default(addmm_19, [32, 128, 1024]);  addmm_19 = None
        view_72 = torch.ops.aten.view.default(view_71, [32, -1, 16, 64]);  view_71 = None
        permute_35 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_73 = torch.ops.aten.view.default(add_27, [4096, 1024]);  add_27 = None
        permute_36 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg58_1, view_73, permute_36);  arg58_1 = view_73 = permute_36 = None
        view_74 = torch.ops.aten.view.default(addmm_20, [32, 128, 1024]);  addmm_20 = None
        view_75 = torch.ops.aten.view.default(view_74, [32, -1, 16, 64]);  view_74 = None
        permute_37 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_26 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_76 = torch.ops.aten.view.default(mul_27, [32, 128, 16, 64]);  mul_27 = None
        permute_38 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_27 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_77 = torch.ops.aten.view.default(clone_27, [512, -1, 64]);  clone_27 = None
        view_78 = torch.ops.aten.view.default(clone_25, [512, -1, 64])
        view_79 = torch.ops.aten.view.default(clone_26, [512, -1, 64])
        permute_39 = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        bmm_6 = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
        view_80 = torch.ops.aten.view.default(bmm_6, [32, 16, 128, 128]);  bmm_6 = None
        add_28 = torch.ops.aten.add.Tensor(view_80, expand_1);  view_80 = None
        view_81 = torch.ops.aten.view.default(add_28, [512, 128, 128]);  add_28 = None
        amax_3 = torch.ops.aten.amax.default(view_81, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        bmm_7 = torch.ops.aten.bmm.default(div_3, view_79);  div_3 = view_79 = None
        view_82 = torch.ops.aten.view.default(bmm_7, [32, 16, 128, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_29 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_83 = torch.ops.aten.view.default(clone_29, [32, 128, 1024]);  clone_29 = None
        view_84 = torch.ops.aten.view.default(view_83, [4096, 1024]);  view_83 = None
        permute_41 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg60_1, view_84, permute_41);  arg60_1 = view_84 = permute_41 = None
        view_85 = torch.ops.aten.view.default(addmm_21, [32, 128, 1024]);  addmm_21 = None
        add_29 = torch.ops.aten.add.Tensor(add_25, view_85);  add_25 = view_85 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  getitem_15 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        view_86 = torch.ops.aten.view.default(add_31, [4096, 1024]);  add_31 = None
        permute_42 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg64_1, view_86, permute_42);  arg64_1 = view_86 = permute_42 = None
        view_87 = torch.ops.aten.view.default(addmm_22, [32, 128, 4096]);  addmm_22 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_32);  mul_30 = add_32 = None
        view_88 = torch.ops.aten.view.default(mul_32, [4096, 4096]);  mul_32 = None
        permute_43 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg66_1, view_88, permute_43);  arg66_1 = view_88 = permute_43 = None
        view_89 = torch.ops.aten.view.default(addmm_23, [32, 128, 1024]);  addmm_23 = None
        add_33 = torch.ops.aten.add.Tensor(add_29, view_89);  add_29 = view_89 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        view_90 = torch.ops.aten.view.default(add_35, [4096, 1024])
        permute_44 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg70_1, view_90, permute_44);  arg70_1 = view_90 = permute_44 = None
        view_91 = torch.ops.aten.view.default(addmm_24, [32, 128, 1024]);  addmm_24 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_91, 0.125);  view_91 = None
        view_92 = torch.ops.aten.view.default(add_35, [4096, 1024])
        permute_45 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg72_1, view_92, permute_45);  arg72_1 = view_92 = permute_45 = None
        view_93 = torch.ops.aten.view.default(addmm_25, [32, 128, 1024]);  addmm_25 = None
        view_94 = torch.ops.aten.view.default(view_93, [32, -1, 16, 64]);  view_93 = None
        permute_46 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_33 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_95 = torch.ops.aten.view.default(add_35, [4096, 1024]);  add_35 = None
        permute_47 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg74_1, view_95, permute_47);  arg74_1 = view_95 = permute_47 = None
        view_96 = torch.ops.aten.view.default(addmm_26, [32, 128, 1024]);  addmm_26 = None
        view_97 = torch.ops.aten.view.default(view_96, [32, -1, 16, 64]);  view_96 = None
        permute_48 = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        clone_34 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_98 = torch.ops.aten.view.default(mul_35, [32, 128, 16, 64]);  mul_35 = None
        permute_49 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_35 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_99 = torch.ops.aten.view.default(clone_35, [512, -1, 64]);  clone_35 = None
        view_100 = torch.ops.aten.view.default(clone_33, [512, -1, 64])
        view_101 = torch.ops.aten.view.default(clone_34, [512, -1, 64])
        permute_50 = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        bmm_8 = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
        view_102 = torch.ops.aten.view.default(bmm_8, [32, 16, 128, 128]);  bmm_8 = None
        add_36 = torch.ops.aten.add.Tensor(view_102, expand_1);  view_102 = None
        view_103 = torch.ops.aten.view.default(add_36, [512, 128, 128]);  add_36 = None
        amax_4 = torch.ops.aten.amax.default(view_103, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        bmm_9 = torch.ops.aten.bmm.default(div_4, view_101);  div_4 = view_101 = None
        view_104 = torch.ops.aten.view.default(bmm_9, [32, 16, 128, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_37 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_105 = torch.ops.aten.view.default(clone_37, [32, 128, 1024]);  clone_37 = None
        view_106 = torch.ops.aten.view.default(view_105, [4096, 1024]);  view_105 = None
        permute_52 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg76_1, view_106, permute_52);  arg76_1 = view_106 = permute_52 = None
        view_107 = torch.ops.aten.view.default(addmm_27, [32, 128, 1024]);  addmm_27 = None
        add_37 = torch.ops.aten.add.Tensor(add_33, view_107);  add_33 = view_107 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_37, getitem_19);  getitem_19 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        view_108 = torch.ops.aten.view.default(add_39, [4096, 1024]);  add_39 = None
        permute_53 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg80_1, view_108, permute_53);  arg80_1 = view_108 = permute_53 = None
        view_109 = torch.ops.aten.view.default(addmm_28, [32, 128, 4096]);  addmm_28 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_39 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4 = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, add_40);  mul_38 = add_40 = None
        view_110 = torch.ops.aten.view.default(mul_40, [4096, 4096]);  mul_40 = None
        permute_54 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg82_1, view_110, permute_54);  arg82_1 = view_110 = permute_54 = None
        view_111 = torch.ops.aten.view.default(addmm_29, [32, 128, 1024]);  addmm_29 = None
        add_41 = torch.ops.aten.add.Tensor(add_37, view_111);  add_37 = view_111 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        view_112 = torch.ops.aten.view.default(add_43, [4096, 1024])
        permute_55 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg86_1, view_112, permute_55);  arg86_1 = view_112 = permute_55 = None
        view_113 = torch.ops.aten.view.default(addmm_30, [32, 128, 1024]);  addmm_30 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
        view_114 = torch.ops.aten.view.default(add_43, [4096, 1024])
        permute_56 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg88_1, view_114, permute_56);  arg88_1 = view_114 = permute_56 = None
        view_115 = torch.ops.aten.view.default(addmm_31, [32, 128, 1024]);  addmm_31 = None
        view_116 = torch.ops.aten.view.default(view_115, [32, -1, 16, 64]);  view_115 = None
        permute_57 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_41 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_117 = torch.ops.aten.view.default(add_43, [4096, 1024]);  add_43 = None
        permute_58 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg90_1, view_117, permute_58);  arg90_1 = view_117 = permute_58 = None
        view_118 = torch.ops.aten.view.default(addmm_32, [32, 128, 1024]);  addmm_32 = None
        view_119 = torch.ops.aten.view.default(view_118, [32, -1, 16, 64]);  view_118 = None
        permute_59 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        clone_42 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_120 = torch.ops.aten.view.default(mul_43, [32, 128, 16, 64]);  mul_43 = None
        permute_60 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_43 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_121 = torch.ops.aten.view.default(clone_43, [512, -1, 64]);  clone_43 = None
        view_122 = torch.ops.aten.view.default(clone_41, [512, -1, 64])
        view_123 = torch.ops.aten.view.default(clone_42, [512, -1, 64])
        permute_61 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        bmm_10 = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
        view_124 = torch.ops.aten.view.default(bmm_10, [32, 16, 128, 128]);  bmm_10 = None
        add_44 = torch.ops.aten.add.Tensor(view_124, expand_1);  view_124 = None
        view_125 = torch.ops.aten.view.default(add_44, [512, 128, 128]);  add_44 = None
        amax_5 = torch.ops.aten.amax.default(view_125, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        bmm_11 = torch.ops.aten.bmm.default(div_5, view_123);  div_5 = view_123 = None
        view_126 = torch.ops.aten.view.default(bmm_11, [32, 16, 128, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_45 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_127 = torch.ops.aten.view.default(clone_45, [32, 128, 1024]);  clone_45 = None
        view_128 = torch.ops.aten.view.default(view_127, [4096, 1024]);  view_127 = None
        permute_63 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg92_1, view_128, permute_63);  arg92_1 = view_128 = permute_63 = None
        view_129 = torch.ops.aten.view.default(addmm_33, [32, 128, 1024]);  addmm_33 = None
        add_45 = torch.ops.aten.add.Tensor(add_41, view_129);  add_41 = view_129 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_45, getitem_23);  getitem_23 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        view_130 = torch.ops.aten.view.default(add_47, [4096, 1024]);  add_47 = None
        permute_64 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg96_1, view_130, permute_64);  arg96_1 = view_130 = permute_64 = None
        view_131 = torch.ops.aten.view.default(addmm_34, [32, 128, 4096]);  addmm_34 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
        view_132 = torch.ops.aten.view.default(mul_48, [4096, 4096]);  mul_48 = None
        permute_65 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg98_1, view_132, permute_65);  arg98_1 = view_132 = permute_65 = None
        view_133 = torch.ops.aten.view.default(addmm_35, [32, 128, 1024]);  addmm_35 = None
        add_49 = torch.ops.aten.add.Tensor(add_45, view_133);  add_45 = view_133 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        view_134 = torch.ops.aten.view.default(add_51, [4096, 1024])
        permute_66 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg102_1, view_134, permute_66);  arg102_1 = view_134 = permute_66 = None
        view_135 = torch.ops.aten.view.default(addmm_36, [32, 128, 1024]);  addmm_36 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_135, 0.125);  view_135 = None
        view_136 = torch.ops.aten.view.default(add_51, [4096, 1024])
        permute_67 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg104_1, view_136, permute_67);  arg104_1 = view_136 = permute_67 = None
        view_137 = torch.ops.aten.view.default(addmm_37, [32, 128, 1024]);  addmm_37 = None
        view_138 = torch.ops.aten.view.default(view_137, [32, -1, 16, 64]);  view_137 = None
        permute_68 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_49 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_139 = torch.ops.aten.view.default(add_51, [4096, 1024]);  add_51 = None
        permute_69 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg106_1, view_139, permute_69);  arg106_1 = view_139 = permute_69 = None
        view_140 = torch.ops.aten.view.default(addmm_38, [32, 128, 1024]);  addmm_38 = None
        view_141 = torch.ops.aten.view.default(view_140, [32, -1, 16, 64]);  view_140 = None
        permute_70 = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_50 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_142 = torch.ops.aten.view.default(mul_51, [32, 128, 16, 64]);  mul_51 = None
        permute_71 = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_51 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_143 = torch.ops.aten.view.default(clone_51, [512, -1, 64]);  clone_51 = None
        view_144 = torch.ops.aten.view.default(clone_49, [512, -1, 64])
        view_145 = torch.ops.aten.view.default(clone_50, [512, -1, 64])
        permute_72 = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        bmm_12 = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
        view_146 = torch.ops.aten.view.default(bmm_12, [32, 16, 128, 128]);  bmm_12 = None
        add_52 = torch.ops.aten.add.Tensor(view_146, expand_1);  view_146 = None
        view_147 = torch.ops.aten.view.default(add_52, [512, 128, 128]);  add_52 = None
        amax_6 = torch.ops.aten.amax.default(view_147, [-1], True)
        sub_19 = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        bmm_13 = torch.ops.aten.bmm.default(div_6, view_145);  div_6 = view_145 = None
        view_148 = torch.ops.aten.view.default(bmm_13, [32, 16, 128, 64]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_53 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_149 = torch.ops.aten.view.default(clone_53, [32, 128, 1024]);  clone_53 = None
        view_150 = torch.ops.aten.view.default(view_149, [4096, 1024]);  view_149 = None
        permute_74 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg108_1, view_150, permute_74);  arg108_1 = view_150 = permute_74 = None
        view_151 = torch.ops.aten.view.default(addmm_39, [32, 128, 1024]);  addmm_39 = None
        add_53 = torch.ops.aten.add.Tensor(add_49, view_151);  add_49 = view_151 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_53, getitem_27);  getitem_27 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        view_152 = torch.ops.aten.view.default(add_55, [4096, 1024]);  add_55 = None
        permute_75 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg112_1, view_152, permute_75);  arg112_1 = view_152 = permute_75 = None
        view_153 = torch.ops.aten.view.default(addmm_40, [32, 128, 4096]);  addmm_40 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_56);  mul_54 = add_56 = None
        view_154 = torch.ops.aten.view.default(mul_56, [4096, 4096]);  mul_56 = None
        permute_76 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg114_1, view_154, permute_76);  arg114_1 = view_154 = permute_76 = None
        view_155 = torch.ops.aten.view.default(addmm_41, [32, 128, 1024]);  addmm_41 = None
        add_57 = torch.ops.aten.add.Tensor(add_53, view_155);  add_53 = view_155 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        view_156 = torch.ops.aten.view.default(add_59, [4096, 1024])
        permute_77 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg118_1, view_156, permute_77);  arg118_1 = view_156 = permute_77 = None
        view_157 = torch.ops.aten.view.default(addmm_42, [32, 128, 1024]);  addmm_42 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
        view_158 = torch.ops.aten.view.default(add_59, [4096, 1024])
        permute_78 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg120_1, view_158, permute_78);  arg120_1 = view_158 = permute_78 = None
        view_159 = torch.ops.aten.view.default(addmm_43, [32, 128, 1024]);  addmm_43 = None
        view_160 = torch.ops.aten.view.default(view_159, [32, -1, 16, 64]);  view_159 = None
        permute_79 = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_57 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_161 = torch.ops.aten.view.default(add_59, [4096, 1024]);  add_59 = None
        permute_80 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg122_1, view_161, permute_80);  arg122_1 = view_161 = permute_80 = None
        view_162 = torch.ops.aten.view.default(addmm_44, [32, 128, 1024]);  addmm_44 = None
        view_163 = torch.ops.aten.view.default(view_162, [32, -1, 16, 64]);  view_162 = None
        permute_81 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_58 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_164 = torch.ops.aten.view.default(mul_59, [32, 128, 16, 64]);  mul_59 = None
        permute_82 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_59 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_165 = torch.ops.aten.view.default(clone_59, [512, -1, 64]);  clone_59 = None
        view_166 = torch.ops.aten.view.default(clone_57, [512, -1, 64])
        view_167 = torch.ops.aten.view.default(clone_58, [512, -1, 64])
        permute_83 = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        bmm_14 = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
        view_168 = torch.ops.aten.view.default(bmm_14, [32, 16, 128, 128]);  bmm_14 = None
        add_60 = torch.ops.aten.add.Tensor(view_168, expand_1);  view_168 = None
        view_169 = torch.ops.aten.view.default(add_60, [512, 128, 128]);  add_60 = None
        amax_7 = torch.ops.aten.amax.default(view_169, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        bmm_15 = torch.ops.aten.bmm.default(div_7, view_167);  div_7 = view_167 = None
        view_170 = torch.ops.aten.view.default(bmm_15, [32, 16, 128, 64]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_61 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_171 = torch.ops.aten.view.default(clone_61, [32, 128, 1024]);  clone_61 = None
        view_172 = torch.ops.aten.view.default(view_171, [4096, 1024]);  view_171 = None
        permute_85 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg124_1, view_172, permute_85);  arg124_1 = view_172 = permute_85 = None
        view_173 = torch.ops.aten.view.default(addmm_45, [32, 128, 1024]);  addmm_45 = None
        add_61 = torch.ops.aten.add.Tensor(add_57, view_173);  add_57 = view_173 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_61, getitem_31);  getitem_31 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        view_174 = torch.ops.aten.view.default(add_63, [4096, 1024]);  add_63 = None
        permute_86 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg128_1, view_174, permute_86);  arg128_1 = view_174 = permute_86 = None
        view_175 = torch.ops.aten.view.default(addmm_46, [32, 128, 4096]);  addmm_46 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_63 = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7 = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_62, add_64);  mul_62 = add_64 = None
        view_176 = torch.ops.aten.view.default(mul_64, [4096, 4096]);  mul_64 = None
        permute_87 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg130_1, view_176, permute_87);  arg130_1 = view_176 = permute_87 = None
        view_177 = torch.ops.aten.view.default(addmm_47, [32, 128, 1024]);  addmm_47 = None
        add_65 = torch.ops.aten.add.Tensor(add_61, view_177);  add_61 = view_177 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        view_178 = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_88 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg134_1, view_178, permute_88);  arg134_1 = view_178 = permute_88 = None
        view_179 = torch.ops.aten.view.default(addmm_48, [32, 128, 1024]);  addmm_48 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_179, 0.125);  view_179 = None
        view_180 = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_89 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg136_1, view_180, permute_89);  arg136_1 = view_180 = permute_89 = None
        view_181 = torch.ops.aten.view.default(addmm_49, [32, 128, 1024]);  addmm_49 = None
        view_182 = torch.ops.aten.view.default(view_181, [32, -1, 16, 64]);  view_181 = None
        permute_90 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_65 = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_183 = torch.ops.aten.view.default(add_67, [4096, 1024]);  add_67 = None
        permute_91 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg138_1, view_183, permute_91);  arg138_1 = view_183 = permute_91 = None
        view_184 = torch.ops.aten.view.default(addmm_50, [32, 128, 1024]);  addmm_50 = None
        view_185 = torch.ops.aten.view.default(view_184, [32, -1, 16, 64]);  view_184 = None
        permute_92 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_66 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_186 = torch.ops.aten.view.default(mul_67, [32, 128, 16, 64]);  mul_67 = None
        permute_93 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_67 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_187 = torch.ops.aten.view.default(clone_67, [512, -1, 64]);  clone_67 = None
        view_188 = torch.ops.aten.view.default(clone_65, [512, -1, 64])
        view_189 = torch.ops.aten.view.default(clone_66, [512, -1, 64])
        permute_94 = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        bmm_16 = torch.ops.aten.bmm.default(view_187, permute_94);  view_187 = permute_94 = None
        view_190 = torch.ops.aten.view.default(bmm_16, [32, 16, 128, 128]);  bmm_16 = None
        add_68 = torch.ops.aten.add.Tensor(view_190, expand_1);  view_190 = None
        view_191 = torch.ops.aten.view.default(add_68, [512, 128, 128]);  add_68 = None
        amax_8 = torch.ops.aten.amax.default(view_191, [-1], True)
        sub_25 = torch.ops.aten.sub.Tensor(view_191, amax_8);  view_191 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        bmm_17 = torch.ops.aten.bmm.default(div_8, view_189);  div_8 = view_189 = None
        view_192 = torch.ops.aten.view.default(bmm_17, [32, 16, 128, 64]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_69 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_193 = torch.ops.aten.view.default(clone_69, [32, 128, 1024]);  clone_69 = None
        view_194 = torch.ops.aten.view.default(view_193, [4096, 1024]);  view_193 = None
        permute_96 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg140_1, view_194, permute_96);  arg140_1 = view_194 = permute_96 = None
        view_195 = torch.ops.aten.view.default(addmm_51, [32, 128, 1024]);  addmm_51 = None
        add_69 = torch.ops.aten.add.Tensor(add_65, view_195);  add_65 = view_195 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_69, getitem_35);  getitem_35 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        view_196 = torch.ops.aten.view.default(add_71, [4096, 1024]);  add_71 = None
        permute_97 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg144_1, view_196, permute_97);  arg144_1 = view_196 = permute_97 = None
        view_197 = torch.ops.aten.view.default(addmm_52, [32, 128, 4096]);  addmm_52 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_71 = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_8 = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_70, add_72);  mul_70 = add_72 = None
        view_198 = torch.ops.aten.view.default(mul_72, [4096, 4096]);  mul_72 = None
        permute_98 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg146_1, view_198, permute_98);  arg146_1 = view_198 = permute_98 = None
        view_199 = torch.ops.aten.view.default(addmm_53, [32, 128, 1024]);  addmm_53 = None
        add_73 = torch.ops.aten.add.Tensor(add_69, view_199);  add_69 = view_199 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_73, getitem_37);  getitem_37 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg147_1);  mul_73 = arg147_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        view_200 = torch.ops.aten.view.default(add_75, [4096, 1024])
        permute_99 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg150_1, view_200, permute_99);  arg150_1 = view_200 = permute_99 = None
        view_201 = torch.ops.aten.view.default(addmm_54, [32, 128, 1024]);  addmm_54 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_201, 0.125);  view_201 = None
        view_202 = torch.ops.aten.view.default(add_75, [4096, 1024])
        permute_100 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg152_1, view_202, permute_100);  arg152_1 = view_202 = permute_100 = None
        view_203 = torch.ops.aten.view.default(addmm_55, [32, 128, 1024]);  addmm_55 = None
        view_204 = torch.ops.aten.view.default(view_203, [32, -1, 16, 64]);  view_203 = None
        permute_101 = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        clone_73 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_205 = torch.ops.aten.view.default(add_75, [4096, 1024]);  add_75 = None
        permute_102 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg154_1, view_205, permute_102);  arg154_1 = view_205 = permute_102 = None
        view_206 = torch.ops.aten.view.default(addmm_56, [32, 128, 1024]);  addmm_56 = None
        view_207 = torch.ops.aten.view.default(view_206, [32, -1, 16, 64]);  view_206 = None
        permute_103 = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        clone_74 = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_208 = torch.ops.aten.view.default(mul_75, [32, 128, 16, 64]);  mul_75 = None
        permute_104 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_75 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_209 = torch.ops.aten.view.default(clone_75, [512, -1, 64]);  clone_75 = None
        view_210 = torch.ops.aten.view.default(clone_73, [512, -1, 64])
        view_211 = torch.ops.aten.view.default(clone_74, [512, -1, 64])
        permute_105 = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        bmm_18 = torch.ops.aten.bmm.default(view_209, permute_105);  view_209 = permute_105 = None
        view_212 = torch.ops.aten.view.default(bmm_18, [32, 16, 128, 128]);  bmm_18 = None
        add_76 = torch.ops.aten.add.Tensor(view_212, expand_1);  view_212 = None
        view_213 = torch.ops.aten.view.default(add_76, [512, 128, 128]);  add_76 = None
        amax_9 = torch.ops.aten.amax.default(view_213, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(view_213, amax_9);  view_213 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        bmm_19 = torch.ops.aten.bmm.default(div_9, view_211);  div_9 = view_211 = None
        view_214 = torch.ops.aten.view.default(bmm_19, [32, 16, 128, 64]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_77 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_215 = torch.ops.aten.view.default(clone_77, [32, 128, 1024]);  clone_77 = None
        view_216 = torch.ops.aten.view.default(view_215, [4096, 1024]);  view_215 = None
        permute_107 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg156_1, view_216, permute_107);  arg156_1 = view_216 = permute_107 = None
        view_217 = torch.ops.aten.view.default(addmm_57, [32, 128, 1024]);  addmm_57 = None
        add_77 = torch.ops.aten.add.Tensor(add_73, view_217);  add_73 = view_217 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_77, getitem_39);  getitem_39 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, arg157_1);  mul_76 = arg157_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        view_218 = torch.ops.aten.view.default(add_79, [4096, 1024]);  add_79 = None
        permute_108 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg160_1, view_218, permute_108);  arg160_1 = view_218 = permute_108 = None
        view_219 = torch.ops.aten.view.default(addmm_58, [32, 128, 4096]);  addmm_58 = None
        mul_78 = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_79 = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_9 = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_78, add_80);  mul_78 = add_80 = None
        view_220 = torch.ops.aten.view.default(mul_80, [4096, 4096]);  mul_80 = None
        permute_109 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg162_1, view_220, permute_109);  arg162_1 = view_220 = permute_109 = None
        view_221 = torch.ops.aten.view.default(addmm_59, [32, 128, 1024]);  addmm_59 = None
        add_81 = torch.ops.aten.add.Tensor(add_77, view_221);  add_77 = view_221 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_81, getitem_41);  getitem_41 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, arg163_1);  mul_81 = arg163_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        view_222 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_110 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg166_1, view_222, permute_110);  arg166_1 = view_222 = permute_110 = None
        view_223 = torch.ops.aten.view.default(addmm_60, [32, 128, 1024]);  addmm_60 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
        view_224 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_111 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg168_1, view_224, permute_111);  arg168_1 = view_224 = permute_111 = None
        view_225 = torch.ops.aten.view.default(addmm_61, [32, 128, 1024]);  addmm_61 = None
        view_226 = torch.ops.aten.view.default(view_225, [32, -1, 16, 64]);  view_225 = None
        permute_112 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_81 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_227 = torch.ops.aten.view.default(add_83, [4096, 1024]);  add_83 = None
        permute_113 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg170_1, view_227, permute_113);  arg170_1 = view_227 = permute_113 = None
        view_228 = torch.ops.aten.view.default(addmm_62, [32, 128, 1024]);  addmm_62 = None
        view_229 = torch.ops.aten.view.default(view_228, [32, -1, 16, 64]);  view_228 = None
        permute_114 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_82 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_230 = torch.ops.aten.view.default(mul_83, [32, 128, 16, 64]);  mul_83 = None
        permute_115 = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_83 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_231 = torch.ops.aten.view.default(clone_83, [512, -1, 64]);  clone_83 = None
        view_232 = torch.ops.aten.view.default(clone_81, [512, -1, 64])
        view_233 = torch.ops.aten.view.default(clone_82, [512, -1, 64])
        permute_116 = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        bmm_20 = torch.ops.aten.bmm.default(view_231, permute_116);  view_231 = permute_116 = None
        view_234 = torch.ops.aten.view.default(bmm_20, [32, 16, 128, 128]);  bmm_20 = None
        add_84 = torch.ops.aten.add.Tensor(view_234, expand_1);  view_234 = None
        view_235 = torch.ops.aten.view.default(add_84, [512, 128, 128]);  add_84 = None
        amax_10 = torch.ops.aten.amax.default(view_235, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(view_235, amax_10);  view_235 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        bmm_21 = torch.ops.aten.bmm.default(div_10, view_233);  div_10 = view_233 = None
        view_236 = torch.ops.aten.view.default(bmm_21, [32, 16, 128, 64]);  bmm_21 = None
        permute_117 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_85 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_237 = torch.ops.aten.view.default(clone_85, [32, 128, 1024]);  clone_85 = None
        view_238 = torch.ops.aten.view.default(view_237, [4096, 1024]);  view_237 = None
        permute_118 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg172_1, view_238, permute_118);  arg172_1 = view_238 = permute_118 = None
        view_239 = torch.ops.aten.view.default(addmm_63, [32, 128, 1024]);  addmm_63 = None
        add_85 = torch.ops.aten.add.Tensor(add_81, view_239);  add_81 = view_239 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_85, getitem_43);  getitem_43 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg173_1);  mul_84 = arg173_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        view_240 = torch.ops.aten.view.default(add_87, [4096, 1024]);  add_87 = None
        permute_119 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg176_1, view_240, permute_119);  arg176_1 = view_240 = permute_119 = None
        view_241 = torch.ops.aten.view.default(addmm_64, [32, 128, 4096]);  addmm_64 = None
        mul_86 = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_87 = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
        erf_10 = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_86, add_88);  mul_86 = add_88 = None
        view_242 = torch.ops.aten.view.default(mul_88, [4096, 4096]);  mul_88 = None
        permute_120 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg178_1, view_242, permute_120);  arg178_1 = view_242 = permute_120 = None
        view_243 = torch.ops.aten.view.default(addmm_65, [32, 128, 1024]);  addmm_65 = None
        add_89 = torch.ops.aten.add.Tensor(add_85, view_243);  add_85 = view_243 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_89, getitem_45);  getitem_45 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, arg179_1);  mul_89 = arg179_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        view_244 = torch.ops.aten.view.default(add_91, [4096, 1024])
        permute_121 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg182_1, view_244, permute_121);  arg182_1 = view_244 = permute_121 = None
        view_245 = torch.ops.aten.view.default(addmm_66, [32, 128, 1024]);  addmm_66 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_245, 0.125);  view_245 = None
        view_246 = torch.ops.aten.view.default(add_91, [4096, 1024])
        permute_122 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg184_1, view_246, permute_122);  arg184_1 = view_246 = permute_122 = None
        view_247 = torch.ops.aten.view.default(addmm_67, [32, 128, 1024]);  addmm_67 = None
        view_248 = torch.ops.aten.view.default(view_247, [32, -1, 16, 64]);  view_247 = None
        permute_123 = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_89 = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        view_249 = torch.ops.aten.view.default(add_91, [4096, 1024]);  add_91 = None
        permute_124 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg186_1, view_249, permute_124);  arg186_1 = view_249 = permute_124 = None
        view_250 = torch.ops.aten.view.default(addmm_68, [32, 128, 1024]);  addmm_68 = None
        view_251 = torch.ops.aten.view.default(view_250, [32, -1, 16, 64]);  view_250 = None
        permute_125 = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        clone_90 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_252 = torch.ops.aten.view.default(mul_91, [32, 128, 16, 64]);  mul_91 = None
        permute_126 = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_91 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_253 = torch.ops.aten.view.default(clone_91, [512, -1, 64]);  clone_91 = None
        view_254 = torch.ops.aten.view.default(clone_89, [512, -1, 64])
        view_255 = torch.ops.aten.view.default(clone_90, [512, -1, 64])
        permute_127 = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        bmm_22 = torch.ops.aten.bmm.default(view_253, permute_127);  view_253 = permute_127 = None
        view_256 = torch.ops.aten.view.default(bmm_22, [32, 16, 128, 128]);  bmm_22 = None
        add_92 = torch.ops.aten.add.Tensor(view_256, expand_1);  view_256 = expand_1 = None
        view_257 = torch.ops.aten.view.default(add_92, [512, 128, 128]);  add_92 = None
        amax_11 = torch.ops.aten.amax.default(view_257, [-1], True)
        sub_34 = torch.ops.aten.sub.Tensor(view_257, amax_11);  view_257 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        bmm_23 = torch.ops.aten.bmm.default(div_11, view_255);  div_11 = view_255 = None
        view_258 = torch.ops.aten.view.default(bmm_23, [32, 16, 128, 64]);  bmm_23 = None
        permute_128 = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_93 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_259 = torch.ops.aten.view.default(clone_93, [32, 128, 1024]);  clone_93 = None
        view_260 = torch.ops.aten.view.default(view_259, [4096, 1024]);  view_259 = None
        permute_129 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg188_1, view_260, permute_129);  arg188_1 = view_260 = permute_129 = None
        view_261 = torch.ops.aten.view.default(addmm_69, [32, 128, 1024]);  addmm_69 = None
        add_93 = torch.ops.aten.add.Tensor(add_89, view_261);  add_89 = view_261 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_93, getitem_47);  getitem_47 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, arg189_1);  mul_92 = arg189_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        view_262 = torch.ops.aten.view.default(add_95, [4096, 1024]);  add_95 = None
        permute_130 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg192_1, view_262, permute_130);  arg192_1 = view_262 = permute_130 = None
        view_263 = torch.ops.aten.view.default(addmm_70, [32, 128, 4096]);  addmm_70 = None
        mul_94 = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_95 = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_11 = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_94, add_96);  mul_94 = add_96 = None
        view_264 = torch.ops.aten.view.default(mul_96, [4096, 4096]);  mul_96 = None
        permute_131 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg194_1, view_264, permute_131);  arg194_1 = view_264 = permute_131 = None
        view_265 = torch.ops.aten.view.default(addmm_71, [32, 128, 1024]);  addmm_71 = None
        add_97 = torch.ops.aten.add.Tensor(add_93, view_265);  add_93 = view_265 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        permute_132 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_266 = torch.ops.aten.view.default(add_99, [4096, 1024]);  add_99 = None
        full_default_4 = torch.ops.aten.full.default([1024, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_132, full_default_4], 1);  permute_132 = full_default_4 = None
        mm_default = torch.ops.aten.mm.default(view_266, cat_default);  view_266 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_267 = torch.ops.aten.view.default(slice_tensor, [32, 128, 50265]);  slice_tensor = None
        view_268 = torch.ops.aten.view.default(view_267, [-1, 50265])
        view_269 = torch.ops.aten.view.default(arg197_1, [-1]);  arg197_1 = None
        amax_12 = torch.ops.aten.amax.default(view_268, [1], True)
        sub_37 = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_37)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_38 = torch.ops.aten.sub.Tensor(sub_37, log);  sub_37 = log = None
        ne = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_269, full_default_2);  ne = full_default_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_38, 1, unsqueeze_4);  sub_38 = unsqueeze_4 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
        sum_14 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_12 = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        return (div_12, view_267, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58, clone_65, clone_66, clone_73, clone_74, clone_81, clone_82, clone_89, clone_90)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (32, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 205885440, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50265, 1024), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1024, 1024), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1024,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1024, 1024), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1024,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1024, 1024), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1024,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024, 1024), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1024,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1024, 1024), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1024,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1024,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1024,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf15, (4096, 1024), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf16, (4096,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1024, 4096), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1024,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1024, 1024), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1024,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1024, 1024), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1024,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1024, 1024), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1024,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1024, 1024), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1024,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1024,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1024,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf31, (4096, 1024), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4096,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1024, 4096), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1024,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1024,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1024, 1024), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1024,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1024, 1024), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1024,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1024, 1024), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1024,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf43, (1024, 1024), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf44, (1024,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1024,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1024,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf47, (4096, 1024), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf48, (4096,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf49, (1024, 4096), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1024,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1024,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1024, 1024), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf54, (1024,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf55, (1024, 1024), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf56, (1024,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf57, (1024, 1024), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf58, (1024,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1024, 1024), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1024,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1024,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1024,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf63, (4096, 1024), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf64, (4096,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1024, 4096), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf66, (1024,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1024,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1024,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1024, 1024), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1024,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1024, 1024), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1024,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf73, (1024, 1024), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf74, (1024,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf75, (1024, 1024), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1024,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1024,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1024,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf79, (4096, 1024), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf80, (4096,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf81, (1024, 4096), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1024,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1024,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1024,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1024, 1024), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1024,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1024, 1024), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1024,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1024, 1024), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf90, (1024,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1024, 1024), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1024,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1024,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf94, (1024,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf95, (4096, 1024), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf96, (4096,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1024, 4096), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf98, (1024,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1024,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1024, 1024), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1024,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1024, 1024), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf104, (1024,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1024, 1024), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1024,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1024, 1024), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1024,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1024,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1024,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf111, (4096, 1024), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf112, (4096,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf113, (1024, 4096), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1024,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1024,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1024,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024, 1024), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1024,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1024, 1024), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1024,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1024, 1024), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1024,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf123, (1024, 1024), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1024,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1024,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1024,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf127, (4096, 1024), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf128, (4096,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf129, (1024, 4096), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1024,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024, 1024), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1024,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024, 1024), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024, 1024), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1024,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1024, 1024), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1024,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1024,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1024,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf143, (4096, 1024), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf144, (4096,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1024, 4096), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1024,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1024,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1024,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1024, 1024), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1024,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1024, 1024), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1024,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1024, 1024), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf154, (1024,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1024, 1024), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1024,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1024,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1024,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf159, (4096, 1024), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf160, (4096,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1024, 4096), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1024,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1024,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf164, (1024,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf165, (1024, 1024), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1024,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1024, 1024), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1024,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1024, 1024), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1024,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1024, 1024), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1024,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1024,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1024,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf175, (4096, 1024), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf176, (4096,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1024, 4096), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1024,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1024,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1024,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1024, 1024), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1024,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1024, 1024), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1024,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1024, 1024), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1024,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1024, 1024), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf188, (1024,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1024,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf190, (1024,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf191, (4096, 1024), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf192, (4096,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf193, (1024, 4096), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf194, (1024,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1024,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1024,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf197, (32, 128), dtype=torch.int64, is_leaf=True)  # arg197_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)