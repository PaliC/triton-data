
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 1024]);  arg0_1 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, view, 1);  view = None
        mul = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        full_default = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota, 1)
        view_1 = torch.ops.aten.view.default(add, [1024, 1]);  add = None
        lt = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        iota_1 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand_1 = torch.ops.aten.expand.default(iota_1, [4, -1]);  iota_1 = None
        add_1 = torch.ops.aten.add.Tensor(expand_1, 2);  expand_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, add_1);  arg2_1 = add_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_4, getitem_3);  getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_4, arg6_1);  mul_4 = arg6_1 = None
        view_2 = torch.ops.aten.view.default(add_6, [4096, 1024])
        permute = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm = torch.ops.aten.addmm.default(arg8_1, view_2, permute);  arg8_1 = view_2 = permute = None
        view_3 = torch.ops.aten.view.default(addmm, [4, 1024, 1024]);  addmm = None
        view_4 = torch.ops.aten.view.default(add_6, [4096, 1024])
        permute_1 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg10_1, view_4, permute_1);  arg10_1 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [4, 1024, 1024]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [4, -1, 16, 64]);  view_5 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7 = torch.ops.aten.view.default(add_6, [4096, 1024]);  add_6 = None
        permute_3 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg12_1, view_7, permute_3);  arg12_1 = view_7 = permute_3 = None
        view_8 = torch.ops.aten.view.default(addmm_2, [4, 1024, 1024]);  addmm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [4, -1, 16, 64]);  view_8 = None
        permute_4 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_10 = torch.ops.aten.view.default(view_3, [4, 1024, 16, 64]);  view_3 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_3, [4, 1, 1024, 1024]);  unsqueeze_3 = None
        expand_4 = torch.ops.aten.expand.default(expand_3, [4, 16, 1024, 1024]);  expand_3 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_3, clone_1, clone_2, expand_4, False);  clone_3 = expand_4 = None
        getitem_4 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        view_11 = torch.ops.aten.view.default(permute_6, [4, 1024, 1024]);  permute_6 = None
        view_12 = torch.ops.aten.view.default(view_11, [4096, 1024]);  view_11 = None
        permute_7 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg14_1, view_12, permute_7);  arg14_1 = view_12 = permute_7 = None
        view_13 = torch.ops.aten.view.default(addmm_3, [4, 1024, 1024]);  addmm_3 = None
        add_7 = torch.ops.aten.add.Tensor(add_4, view_13);  add_4 = view_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_7, getitem_9);  getitem_9 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_6, arg16_1);  mul_6 = arg16_1 = None
        view_14 = torch.ops.aten.view.default(add_9, [4096, 1024]);  add_9 = None
        permute_8 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg18_1, view_14, permute_8);  arg18_1 = view_14 = permute_8 = None
        view_15 = torch.ops.aten.view.default(addmm_4, [4, 1024, 4096]);  addmm_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_8 = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
        erf = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_10 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, add_10);  mul_7 = add_10 = None
        view_16 = torch.ops.aten.view.default(mul_9, [4096, 4096]);  mul_9 = None
        permute_9 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg20_1, view_16, permute_9);  arg20_1 = view_16 = permute_9 = None
        view_17 = torch.ops.aten.view.default(addmm_5, [4, 1024, 1024]);  addmm_5 = None
        add_11 = torch.ops.aten.add.Tensor(add_7, view_17);  add_7 = view_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_3[0]
        getitem_11 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_11, getitem_11);  getitem_11 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        view_18 = torch.ops.aten.view.default(add_13, [4096, 1024])
        permute_10 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg24_1, view_18, permute_10);  arg24_1 = view_18 = permute_10 = None
        view_19 = torch.ops.aten.view.default(addmm_6, [4, 1024, 1024]);  addmm_6 = None
        view_20 = torch.ops.aten.view.default(add_13, [4096, 1024])
        permute_11 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg26_1, view_20, permute_11);  arg26_1 = view_20 = permute_11 = None
        view_21 = torch.ops.aten.view.default(addmm_7, [4, 1024, 1024]);  addmm_7 = None
        view_22 = torch.ops.aten.view.default(view_21, [4, -1, 16, 64]);  view_21 = None
        permute_12 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        clone_7 = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_23 = torch.ops.aten.view.default(add_13, [4096, 1024]);  add_13 = None
        permute_13 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg28_1, view_23, permute_13);  arg28_1 = view_23 = permute_13 = None
        view_24 = torch.ops.aten.view.default(addmm_8, [4, 1024, 1024]);  addmm_8 = None
        view_25 = torch.ops.aten.view.default(view_24, [4, -1, 16, 64]);  view_24 = None
        permute_14 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_8 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_26 = torch.ops.aten.view.default(view_19, [4, 1024, 16, 64]);  view_19 = None
        permute_15 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_9 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        expand_6 = torch.ops.aten.expand.default(unsqueeze_5, [4, 1, 1024, 1024]);  unsqueeze_5 = None
        expand_7 = torch.ops.aten.expand.default(expand_6, [4, 16, 1024, 1024]);  expand_6 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_9, clone_7, clone_8, expand_7, False);  clone_9 = expand_7 = None
        getitem_12 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        view_27 = torch.ops.aten.view.default(permute_16, [4, 1024, 1024]);  permute_16 = None
        view_28 = torch.ops.aten.view.default(view_27, [4096, 1024]);  view_27 = None
        permute_17 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg30_1, view_28, permute_17);  arg30_1 = view_28 = permute_17 = None
        view_29 = torch.ops.aten.view.default(addmm_9, [4, 1024, 1024]);  addmm_9 = None
        add_14 = torch.ops.aten.add.Tensor(add_11, view_29);  add_11 = view_29 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_14, getitem_17);  getitem_17 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg31_1);  mul_12 = arg31_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_13, arg32_1);  mul_13 = arg32_1 = None
        view_30 = torch.ops.aten.view.default(add_16, [4096, 1024]);  add_16 = None
        permute_18 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg34_1, view_30, permute_18);  arg34_1 = view_30 = permute_18 = None
        view_31 = torch.ops.aten.view.default(addmm_10, [4, 1024, 4096]);  addmm_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_31, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_17 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_17);  mul_14 = add_17 = None
        view_32 = torch.ops.aten.view.default(mul_16, [4096, 4096]);  mul_16 = None
        permute_19 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg36_1, view_32, permute_19);  arg36_1 = view_32 = permute_19 = None
        view_33 = torch.ops.aten.view.default(addmm_11, [4, 1024, 1024]);  addmm_11 = None
        add_18 = torch.ops.aten.add.Tensor(add_14, view_33);  add_14 = view_33 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_5[0]
        getitem_19 = var_mean_5[1];  var_mean_5 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_18, getitem_19);  getitem_19 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg37_1);  mul_17 = arg37_1 = None
        add_20 = torch.ops.aten.add.Tensor(mul_18, arg38_1);  mul_18 = arg38_1 = None
        view_34 = torch.ops.aten.view.default(add_20, [4096, 1024])
        permute_20 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg40_1, view_34, permute_20);  arg40_1 = view_34 = permute_20 = None
        view_35 = torch.ops.aten.view.default(addmm_12, [4, 1024, 1024]);  addmm_12 = None
        view_36 = torch.ops.aten.view.default(add_20, [4096, 1024])
        permute_21 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg42_1, view_36, permute_21);  arg42_1 = view_36 = permute_21 = None
        view_37 = torch.ops.aten.view.default(addmm_13, [4, 1024, 1024]);  addmm_13 = None
        view_38 = torch.ops.aten.view.default(view_37, [4, -1, 16, 64]);  view_37 = None
        permute_22 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_13 = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_39 = torch.ops.aten.view.default(add_20, [4096, 1024]);  add_20 = None
        permute_23 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg44_1, view_39, permute_23);  arg44_1 = view_39 = permute_23 = None
        view_40 = torch.ops.aten.view.default(addmm_14, [4, 1024, 1024]);  addmm_14 = None
        view_41 = torch.ops.aten.view.default(view_40, [4, -1, 16, 64]);  view_40 = None
        permute_24 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_14 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_42 = torch.ops.aten.view.default(view_35, [4, 1024, 16, 64]);  view_35 = None
        permute_25 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_15 = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        expand_9 = torch.ops.aten.expand.default(unsqueeze_7, [4, 1, 1024, 1024]);  unsqueeze_7 = None
        expand_10 = torch.ops.aten.expand.default(expand_9, [4, 16, 1024, 1024]);  expand_9 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_15, clone_13, clone_14, expand_10, False);  clone_15 = expand_10 = None
        getitem_20 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
        view_43 = torch.ops.aten.view.default(permute_26, [4, 1024, 1024]);  permute_26 = None
        view_44 = torch.ops.aten.view.default(view_43, [4096, 1024]);  view_43 = None
        permute_27 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg46_1, view_44, permute_27);  arg46_1 = view_44 = permute_27 = None
        view_45 = torch.ops.aten.view.default(addmm_15, [4, 1024, 1024]);  addmm_15 = None
        add_21 = torch.ops.aten.add.Tensor(add_18, view_45);  add_18 = view_45 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_21, getitem_25);  getitem_25 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, arg47_1);  mul_19 = arg47_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_20, arg48_1);  mul_20 = arg48_1 = None
        view_46 = torch.ops.aten.view.default(add_23, [4096, 1024]);  add_23 = None
        permute_28 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg50_1, view_46, permute_28);  arg50_1 = view_46 = permute_28 = None
        view_47 = torch.ops.aten.view.default(addmm_16, [4, 1024, 4096]);  addmm_16 = None
        mul_21 = torch.ops.aten.mul.Tensor(view_47, 0.5)
        mul_22 = torch.ops.aten.mul.Tensor(view_47, 0.7071067811865476);  view_47 = None
        erf_2 = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_21, add_24);  mul_21 = add_24 = None
        view_48 = torch.ops.aten.view.default(mul_23, [4096, 4096]);  mul_23 = None
        permute_29 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg52_1, view_48, permute_29);  arg52_1 = view_48 = permute_29 = None
        view_49 = torch.ops.aten.view.default(addmm_17, [4, 1024, 1024]);  addmm_17 = None
        add_25 = torch.ops.aten.add.Tensor(add_21, view_49);  add_21 = view_49 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_7[0]
        getitem_27 = var_mean_7[1];  var_mean_7 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_25, getitem_27);  getitem_27 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg53_1);  mul_24 = arg53_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_25, arg54_1);  mul_25 = arg54_1 = None
        view_50 = torch.ops.aten.view.default(add_27, [4096, 1024])
        permute_30 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg56_1, view_50, permute_30);  arg56_1 = view_50 = permute_30 = None
        view_51 = torch.ops.aten.view.default(addmm_18, [4, 1024, 1024]);  addmm_18 = None
        view_52 = torch.ops.aten.view.default(add_27, [4096, 1024])
        permute_31 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg58_1, view_52, permute_31);  arg58_1 = view_52 = permute_31 = None
        view_53 = torch.ops.aten.view.default(addmm_19, [4, 1024, 1024]);  addmm_19 = None
        view_54 = torch.ops.aten.view.default(view_53, [4, -1, 16, 64]);  view_53 = None
        permute_32 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_55 = torch.ops.aten.view.default(add_27, [4096, 1024]);  add_27 = None
        permute_33 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg60_1, view_55, permute_33);  arg60_1 = view_55 = permute_33 = None
        view_56 = torch.ops.aten.view.default(addmm_20, [4, 1024, 1024]);  addmm_20 = None
        view_57 = torch.ops.aten.view.default(view_56, [4, -1, 16, 64]);  view_56 = None
        permute_34 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        clone_20 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_58 = torch.ops.aten.view.default(view_51, [4, 1024, 16, 64]);  view_51 = None
        permute_35 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_21 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
        expand_12 = torch.ops.aten.expand.default(unsqueeze_9, [4, 1, 1024, 1024]);  unsqueeze_9 = None
        expand_13 = torch.ops.aten.expand.default(expand_12, [4, 16, 1024, 1024]);  expand_12 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_21, clone_19, clone_20, expand_13, False);  clone_21 = expand_13 = None
        getitem_28 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None
        view_59 = torch.ops.aten.view.default(permute_36, [4, 1024, 1024]);  permute_36 = None
        view_60 = torch.ops.aten.view.default(view_59, [4096, 1024]);  view_59 = None
        permute_37 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg62_1, view_60, permute_37);  arg62_1 = view_60 = permute_37 = None
        view_61 = torch.ops.aten.view.default(addmm_21, [4, 1024, 1024]);  addmm_21 = None
        add_28 = torch.ops.aten.add.Tensor(add_25, view_61);  add_25 = view_61 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_28, getitem_33);  getitem_33 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg63_1);  mul_26 = arg63_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_27, arg64_1);  mul_27 = arg64_1 = None
        view_62 = torch.ops.aten.view.default(add_30, [4096, 1024]);  add_30 = None
        permute_38 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg66_1, view_62, permute_38);  arg66_1 = view_62 = permute_38 = None
        view_63 = torch.ops.aten.view.default(addmm_22, [4, 1024, 4096]);  addmm_22 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_29 = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_3 = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_31 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
        view_64 = torch.ops.aten.view.default(mul_30, [4096, 4096]);  mul_30 = None
        permute_39 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg68_1, view_64, permute_39);  arg68_1 = view_64 = permute_39 = None
        view_65 = torch.ops.aten.view.default(addmm_23, [4, 1024, 1024]);  addmm_23 = None
        add_32 = torch.ops.aten.add.Tensor(add_28, view_65);  add_28 = view_65 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_9[0]
        getitem_35 = var_mean_9[1];  var_mean_9 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_32, getitem_35);  getitem_35 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, arg69_1);  mul_31 = arg69_1 = None
        add_34 = torch.ops.aten.add.Tensor(mul_32, arg70_1);  mul_32 = arg70_1 = None
        view_66 = torch.ops.aten.view.default(add_34, [4096, 1024])
        permute_40 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg72_1, view_66, permute_40);  arg72_1 = view_66 = permute_40 = None
        view_67 = torch.ops.aten.view.default(addmm_24, [4, 1024, 1024]);  addmm_24 = None
        view_68 = torch.ops.aten.view.default(add_34, [4096, 1024])
        permute_41 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg74_1, view_68, permute_41);  arg74_1 = view_68 = permute_41 = None
        view_69 = torch.ops.aten.view.default(addmm_25, [4, 1024, 1024]);  addmm_25 = None
        view_70 = torch.ops.aten.view.default(view_69, [4, -1, 16, 64]);  view_69 = None
        permute_42 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_25 = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_71 = torch.ops.aten.view.default(add_34, [4096, 1024]);  add_34 = None
        permute_43 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg76_1, view_71, permute_43);  arg76_1 = view_71 = permute_43 = None
        view_72 = torch.ops.aten.view.default(addmm_26, [4, 1024, 1024]);  addmm_26 = None
        view_73 = torch.ops.aten.view.default(view_72, [4, -1, 16, 64]);  view_72 = None
        permute_44 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        clone_26 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_74 = torch.ops.aten.view.default(view_67, [4, 1024, 16, 64]);  view_67 = None
        permute_45 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_27 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        expand_15 = torch.ops.aten.expand.default(unsqueeze_11, [4, 1, 1024, 1024]);  unsqueeze_11 = None
        expand_16 = torch.ops.aten.expand.default(expand_15, [4, 16, 1024, 1024]);  expand_15 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_27, clone_25, clone_26, expand_16, False);  clone_27 = expand_16 = None
        getitem_36 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None
        view_75 = torch.ops.aten.view.default(permute_46, [4, 1024, 1024]);  permute_46 = None
        view_76 = torch.ops.aten.view.default(view_75, [4096, 1024]);  view_75 = None
        permute_47 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg78_1, view_76, permute_47);  arg78_1 = view_76 = permute_47 = None
        view_77 = torch.ops.aten.view.default(addmm_27, [4, 1024, 1024]);  addmm_27 = None
        add_35 = torch.ops.aten.add.Tensor(add_32, view_77);  add_32 = view_77 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_35, getitem_41);  getitem_41 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg79_1);  mul_33 = arg79_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_34, arg80_1);  mul_34 = arg80_1 = None
        view_78 = torch.ops.aten.view.default(add_37, [4096, 1024]);  add_37 = None
        permute_48 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg82_1, view_78, permute_48);  arg82_1 = view_78 = permute_48 = None
        view_79 = torch.ops.aten.view.default(addmm_28, [4, 1024, 4096]);  addmm_28 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_79, 0.5)
        mul_36 = torch.ops.aten.mul.Tensor(view_79, 0.7071067811865476);  view_79 = None
        erf_4 = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_38 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_35, add_38);  mul_35 = add_38 = None
        view_80 = torch.ops.aten.view.default(mul_37, [4096, 4096]);  mul_37 = None
        permute_49 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg84_1, view_80, permute_49);  arg84_1 = view_80 = permute_49 = None
        view_81 = torch.ops.aten.view.default(addmm_29, [4, 1024, 1024]);  addmm_29 = None
        add_39 = torch.ops.aten.add.Tensor(add_35, view_81);  add_35 = view_81 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_11[0]
        getitem_43 = var_mean_11[1];  var_mean_11 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_39, getitem_43);  getitem_43 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg85_1);  mul_38 = arg85_1 = None
        add_41 = torch.ops.aten.add.Tensor(mul_39, arg86_1);  mul_39 = arg86_1 = None
        view_82 = torch.ops.aten.view.default(add_41, [4096, 1024])
        permute_50 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg88_1, view_82, permute_50);  arg88_1 = view_82 = permute_50 = None
        view_83 = torch.ops.aten.view.default(addmm_30, [4, 1024, 1024]);  addmm_30 = None
        view_84 = torch.ops.aten.view.default(add_41, [4096, 1024])
        permute_51 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg90_1, view_84, permute_51);  arg90_1 = view_84 = permute_51 = None
        view_85 = torch.ops.aten.view.default(addmm_31, [4, 1024, 1024]);  addmm_31 = None
        view_86 = torch.ops.aten.view.default(view_85, [4, -1, 16, 64]);  view_85 = None
        permute_52 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        clone_31 = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_87 = torch.ops.aten.view.default(add_41, [4096, 1024]);  add_41 = None
        permute_53 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg92_1, view_87, permute_53);  arg92_1 = view_87 = permute_53 = None
        view_88 = torch.ops.aten.view.default(addmm_32, [4, 1024, 1024]);  addmm_32 = None
        view_89 = torch.ops.aten.view.default(view_88, [4, -1, 16, 64]);  view_88 = None
        permute_54 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_32 = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_90 = torch.ops.aten.view.default(view_83, [4, 1024, 16, 64]);  view_83 = None
        permute_55 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_33 = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        expand_18 = torch.ops.aten.expand.default(unsqueeze_13, [4, 1, 1024, 1024]);  unsqueeze_13 = None
        expand_19 = torch.ops.aten.expand.default(expand_18, [4, 16, 1024, 1024]);  expand_18 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_33, clone_31, clone_32, expand_19, False);  clone_33 = expand_19 = None
        getitem_44 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_44, [0, 2, 1, 3]);  getitem_44 = None
        view_91 = torch.ops.aten.view.default(permute_56, [4, 1024, 1024]);  permute_56 = None
        view_92 = torch.ops.aten.view.default(view_91, [4096, 1024]);  view_91 = None
        permute_57 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg94_1, view_92, permute_57);  arg94_1 = view_92 = permute_57 = None
        view_93 = torch.ops.aten.view.default(addmm_33, [4, 1024, 1024]);  addmm_33 = None
        add_42 = torch.ops.aten.add.Tensor(add_39, view_93);  add_39 = view_93 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_42, getitem_49);  getitem_49 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg95_1);  mul_40 = arg95_1 = None
        add_44 = torch.ops.aten.add.Tensor(mul_41, arg96_1);  mul_41 = arg96_1 = None
        view_94 = torch.ops.aten.view.default(add_44, [4096, 1024]);  add_44 = None
        permute_58 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg98_1, view_94, permute_58);  arg98_1 = view_94 = permute_58 = None
        view_95 = torch.ops.aten.view.default(addmm_34, [4, 1024, 4096]);  addmm_34 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_95, 0.5)
        mul_43 = torch.ops.aten.mul.Tensor(view_95, 0.7071067811865476);  view_95 = None
        erf_5 = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_45 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_42, add_45);  mul_42 = add_45 = None
        view_96 = torch.ops.aten.view.default(mul_44, [4096, 4096]);  mul_44 = None
        permute_59 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg100_1, view_96, permute_59);  arg100_1 = view_96 = permute_59 = None
        view_97 = torch.ops.aten.view.default(addmm_35, [4, 1024, 1024]);  addmm_35 = None
        add_46 = torch.ops.aten.add.Tensor(add_42, view_97);  add_42 = view_97 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_13[0]
        getitem_51 = var_mean_13[1];  var_mean_13 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_46, getitem_51);  getitem_51 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, arg101_1);  mul_45 = arg101_1 = None
        add_48 = torch.ops.aten.add.Tensor(mul_46, arg102_1);  mul_46 = arg102_1 = None
        view_98 = torch.ops.aten.view.default(add_48, [4096, 1024])
        permute_60 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg104_1, view_98, permute_60);  arg104_1 = view_98 = permute_60 = None
        view_99 = torch.ops.aten.view.default(addmm_36, [4, 1024, 1024]);  addmm_36 = None
        view_100 = torch.ops.aten.view.default(add_48, [4096, 1024])
        permute_61 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg106_1, view_100, permute_61);  arg106_1 = view_100 = permute_61 = None
        view_101 = torch.ops.aten.view.default(addmm_37, [4, 1024, 1024]);  addmm_37 = None
        view_102 = torch.ops.aten.view.default(view_101, [4, -1, 16, 64]);  view_101 = None
        permute_62 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_37 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_103 = torch.ops.aten.view.default(add_48, [4096, 1024]);  add_48 = None
        permute_63 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg108_1, view_103, permute_63);  arg108_1 = view_103 = permute_63 = None
        view_104 = torch.ops.aten.view.default(addmm_38, [4, 1024, 1024]);  addmm_38 = None
        view_105 = torch.ops.aten.view.default(view_104, [4, -1, 16, 64]);  view_104 = None
        permute_64 = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        clone_38 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_106 = torch.ops.aten.view.default(view_99, [4, 1024, 16, 64]);  view_99 = None
        permute_65 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_39 = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
        expand_21 = torch.ops.aten.expand.default(unsqueeze_15, [4, 1, 1024, 1024]);  unsqueeze_15 = None
        expand_22 = torch.ops.aten.expand.default(expand_21, [4, 16, 1024, 1024]);  expand_21 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_39, clone_37, clone_38, expand_22, False);  clone_39 = expand_22 = None
        getitem_52 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_66 = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        view_107 = torch.ops.aten.view.default(permute_66, [4, 1024, 1024]);  permute_66 = None
        view_108 = torch.ops.aten.view.default(view_107, [4096, 1024]);  view_107 = None
        permute_67 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg110_1, view_108, permute_67);  arg110_1 = view_108 = permute_67 = None
        view_109 = torch.ops.aten.view.default(addmm_39, [4, 1024, 1024]);  addmm_39 = None
        add_49 = torch.ops.aten.add.Tensor(add_46, view_109);  add_46 = view_109 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_49, getitem_57);  getitem_57 = None
        mul_47 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, arg111_1);  mul_47 = arg111_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_48, arg112_1);  mul_48 = arg112_1 = None
        view_110 = torch.ops.aten.view.default(add_51, [4096, 1024]);  add_51 = None
        permute_68 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg114_1, view_110, permute_68);  arg114_1 = view_110 = permute_68 = None
        view_111 = torch.ops.aten.view.default(addmm_40, [4, 1024, 4096]);  addmm_40 = None
        mul_49 = torch.ops.aten.mul.Tensor(view_111, 0.5)
        mul_50 = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
        erf_6 = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_52 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_49, add_52);  mul_49 = add_52 = None
        view_112 = torch.ops.aten.view.default(mul_51, [4096, 4096]);  mul_51 = None
        permute_69 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg116_1, view_112, permute_69);  arg116_1 = view_112 = permute_69 = None
        view_113 = torch.ops.aten.view.default(addmm_41, [4, 1024, 1024]);  addmm_41 = None
        add_53 = torch.ops.aten.add.Tensor(add_49, view_113);  add_49 = view_113 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_15[0]
        getitem_59 = var_mean_15[1];  var_mean_15 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_53, getitem_59);  getitem_59 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg117_1);  mul_52 = arg117_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_53, arg118_1);  mul_53 = arg118_1 = None
        view_114 = torch.ops.aten.view.default(add_55, [4096, 1024])
        permute_70 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg120_1, view_114, permute_70);  arg120_1 = view_114 = permute_70 = None
        view_115 = torch.ops.aten.view.default(addmm_42, [4, 1024, 1024]);  addmm_42 = None
        view_116 = torch.ops.aten.view.default(add_55, [4096, 1024])
        permute_71 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg122_1, view_116, permute_71);  arg122_1 = view_116 = permute_71 = None
        view_117 = torch.ops.aten.view.default(addmm_43, [4, 1024, 1024]);  addmm_43 = None
        view_118 = torch.ops.aten.view.default(view_117, [4, -1, 16, 64]);  view_117 = None
        permute_72 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_43 = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_119 = torch.ops.aten.view.default(add_55, [4096, 1024]);  add_55 = None
        permute_73 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg124_1, view_119, permute_73);  arg124_1 = view_119 = permute_73 = None
        view_120 = torch.ops.aten.view.default(addmm_44, [4, 1024, 1024]);  addmm_44 = None
        view_121 = torch.ops.aten.view.default(view_120, [4, -1, 16, 64]);  view_120 = None
        permute_74 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_44 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_122 = torch.ops.aten.view.default(view_115, [4, 1024, 16, 64]);  view_115 = None
        permute_75 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_45 = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        expand_24 = torch.ops.aten.expand.default(unsqueeze_17, [4, 1, 1024, 1024]);  unsqueeze_17 = None
        expand_25 = torch.ops.aten.expand.default(expand_24, [4, 16, 1024, 1024]);  expand_24 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_45, clone_43, clone_44, expand_25, False);  clone_45 = expand_25 = None
        getitem_60 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_76 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        view_123 = torch.ops.aten.view.default(permute_76, [4, 1024, 1024]);  permute_76 = None
        view_124 = torch.ops.aten.view.default(view_123, [4096, 1024]);  view_123 = None
        permute_77 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg126_1, view_124, permute_77);  arg126_1 = view_124 = permute_77 = None
        view_125 = torch.ops.aten.view.default(addmm_45, [4, 1024, 1024]);  addmm_45 = None
        add_56 = torch.ops.aten.add.Tensor(add_53, view_125);  add_53 = view_125 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_56, getitem_65);  getitem_65 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, arg127_1);  mul_54 = arg127_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_55, arg128_1);  mul_55 = arg128_1 = None
        view_126 = torch.ops.aten.view.default(add_58, [4096, 1024]);  add_58 = None
        permute_78 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg130_1, view_126, permute_78);  arg130_1 = view_126 = permute_78 = None
        view_127 = torch.ops.aten.view.default(addmm_46, [4, 1024, 4096]);  addmm_46 = None
        mul_56 = torch.ops.aten.mul.Tensor(view_127, 0.5)
        mul_57 = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
        erf_7 = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_59 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_56, add_59);  mul_56 = add_59 = None
        view_128 = torch.ops.aten.view.default(mul_58, [4096, 4096]);  mul_58 = None
        permute_79 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg132_1, view_128, permute_79);  arg132_1 = view_128 = permute_79 = None
        view_129 = torch.ops.aten.view.default(addmm_47, [4, 1024, 1024]);  addmm_47 = None
        add_60 = torch.ops.aten.add.Tensor(add_56, view_129);  add_56 = view_129 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_17[0]
        getitem_67 = var_mean_17[1];  var_mean_17 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_60, getitem_67);  getitem_67 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg133_1);  mul_59 = arg133_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_60, arg134_1);  mul_60 = arg134_1 = None
        view_130 = torch.ops.aten.view.default(add_62, [4096, 1024])
        permute_80 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg136_1, view_130, permute_80);  arg136_1 = view_130 = permute_80 = None
        view_131 = torch.ops.aten.view.default(addmm_48, [4, 1024, 1024]);  addmm_48 = None
        view_132 = torch.ops.aten.view.default(add_62, [4096, 1024])
        permute_81 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg138_1, view_132, permute_81);  arg138_1 = view_132 = permute_81 = None
        view_133 = torch.ops.aten.view.default(addmm_49, [4, 1024, 1024]);  addmm_49 = None
        view_134 = torch.ops.aten.view.default(view_133, [4, -1, 16, 64]);  view_133 = None
        permute_82 = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        clone_49 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_135 = torch.ops.aten.view.default(add_62, [4096, 1024]);  add_62 = None
        permute_83 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg140_1, view_135, permute_83);  arg140_1 = view_135 = permute_83 = None
        view_136 = torch.ops.aten.view.default(addmm_50, [4, 1024, 1024]);  addmm_50 = None
        view_137 = torch.ops.aten.view.default(view_136, [4, -1, 16, 64]);  view_136 = None
        permute_84 = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_50 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_138 = torch.ops.aten.view.default(view_131, [4, 1024, 16, 64]);  view_131 = None
        permute_85 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_51 = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
        expand_27 = torch.ops.aten.expand.default(unsqueeze_19, [4, 1, 1024, 1024]);  unsqueeze_19 = None
        expand_28 = torch.ops.aten.expand.default(expand_27, [4, 16, 1024, 1024]);  expand_27 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_51, clone_49, clone_50, expand_28, False);  clone_51 = expand_28 = None
        getitem_68 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_86 = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
        view_139 = torch.ops.aten.view.default(permute_86, [4, 1024, 1024]);  permute_86 = None
        view_140 = torch.ops.aten.view.default(view_139, [4096, 1024]);  view_139 = None
        permute_87 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg142_1, view_140, permute_87);  arg142_1 = view_140 = permute_87 = None
        view_141 = torch.ops.aten.view.default(addmm_51, [4, 1024, 1024]);  addmm_51 = None
        add_63 = torch.ops.aten.add.Tensor(add_60, view_141);  add_60 = view_141 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_63, getitem_73);  getitem_73 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, arg143_1);  mul_61 = arg143_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_62, arg144_1);  mul_62 = arg144_1 = None
        view_142 = torch.ops.aten.view.default(add_65, [4096, 1024]);  add_65 = None
        permute_88 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg146_1, view_142, permute_88);  arg146_1 = view_142 = permute_88 = None
        view_143 = torch.ops.aten.view.default(addmm_52, [4, 1024, 4096]);  addmm_52 = None
        mul_63 = torch.ops.aten.mul.Tensor(view_143, 0.5)
        mul_64 = torch.ops.aten.mul.Tensor(view_143, 0.7071067811865476);  view_143 = None
        erf_8 = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_66 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_63, add_66);  mul_63 = add_66 = None
        view_144 = torch.ops.aten.view.default(mul_65, [4096, 4096]);  mul_65 = None
        permute_89 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg148_1, view_144, permute_89);  arg148_1 = view_144 = permute_89 = None
        view_145 = torch.ops.aten.view.default(addmm_53, [4, 1024, 1024]);  addmm_53 = None
        add_67 = torch.ops.aten.add.Tensor(add_63, view_145);  add_63 = view_145 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_19[0]
        getitem_75 = var_mean_19[1];  var_mean_19 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_67, getitem_75);  getitem_75 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg149_1);  mul_66 = arg149_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_67, arg150_1);  mul_67 = arg150_1 = None
        view_146 = torch.ops.aten.view.default(add_69, [4096, 1024])
        permute_90 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg152_1, view_146, permute_90);  arg152_1 = view_146 = permute_90 = None
        view_147 = torch.ops.aten.view.default(addmm_54, [4, 1024, 1024]);  addmm_54 = None
        view_148 = torch.ops.aten.view.default(add_69, [4096, 1024])
        permute_91 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg154_1, view_148, permute_91);  arg154_1 = view_148 = permute_91 = None
        view_149 = torch.ops.aten.view.default(addmm_55, [4, 1024, 1024]);  addmm_55 = None
        view_150 = torch.ops.aten.view.default(view_149, [4, -1, 16, 64]);  view_149 = None
        permute_92 = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        clone_55 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_151 = torch.ops.aten.view.default(add_69, [4096, 1024]);  add_69 = None
        permute_93 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg156_1, view_151, permute_93);  arg156_1 = view_151 = permute_93 = None
        view_152 = torch.ops.aten.view.default(addmm_56, [4, 1024, 1024]);  addmm_56 = None
        view_153 = torch.ops.aten.view.default(view_152, [4, -1, 16, 64]);  view_152 = None
        permute_94 = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        clone_56 = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_154 = torch.ops.aten.view.default(view_147, [4, 1024, 16, 64]);  view_147 = None
        permute_95 = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_57 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
        expand_30 = torch.ops.aten.expand.default(unsqueeze_21, [4, 1, 1024, 1024]);  unsqueeze_21 = None
        expand_31 = torch.ops.aten.expand.default(expand_30, [4, 16, 1024, 1024]);  expand_30 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_57, clone_55, clone_56, expand_31, False);  clone_57 = expand_31 = None
        getitem_76 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_96 = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3]);  getitem_76 = None
        view_155 = torch.ops.aten.view.default(permute_96, [4, 1024, 1024]);  permute_96 = None
        view_156 = torch.ops.aten.view.default(view_155, [4096, 1024]);  view_155 = None
        permute_97 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg158_1, view_156, permute_97);  arg158_1 = view_156 = permute_97 = None
        view_157 = torch.ops.aten.view.default(addmm_57, [4, 1024, 1024]);  addmm_57 = None
        add_70 = torch.ops.aten.add.Tensor(add_67, view_157);  add_67 = view_157 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_70, getitem_81);  getitem_81 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg159_1);  mul_68 = arg159_1 = None
        add_72 = torch.ops.aten.add.Tensor(mul_69, arg160_1);  mul_69 = arg160_1 = None
        view_158 = torch.ops.aten.view.default(add_72, [4096, 1024]);  add_72 = None
        permute_98 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg162_1, view_158, permute_98);  arg162_1 = view_158 = permute_98 = None
        view_159 = torch.ops.aten.view.default(addmm_58, [4, 1024, 4096]);  addmm_58 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_159, 0.5)
        mul_71 = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
        erf_9 = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_73 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_70, add_73);  mul_70 = add_73 = None
        view_160 = torch.ops.aten.view.default(mul_72, [4096, 4096]);  mul_72 = None
        permute_99 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg164_1, view_160, permute_99);  arg164_1 = view_160 = permute_99 = None
        view_161 = torch.ops.aten.view.default(addmm_59, [4, 1024, 1024]);  addmm_59 = None
        add_74 = torch.ops.aten.add.Tensor(add_70, view_161);  add_70 = view_161 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_21[0]
        getitem_83 = var_mean_21[1];  var_mean_21 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_74, getitem_83);  getitem_83 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg165_1);  mul_73 = arg165_1 = None
        add_76 = torch.ops.aten.add.Tensor(mul_74, arg166_1);  mul_74 = arg166_1 = None
        view_162 = torch.ops.aten.view.default(add_76, [4096, 1024])
        permute_100 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg168_1, view_162, permute_100);  arg168_1 = view_162 = permute_100 = None
        view_163 = torch.ops.aten.view.default(addmm_60, [4, 1024, 1024]);  addmm_60 = None
        view_164 = torch.ops.aten.view.default(add_76, [4096, 1024])
        permute_101 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg170_1, view_164, permute_101);  arg170_1 = view_164 = permute_101 = None
        view_165 = torch.ops.aten.view.default(addmm_61, [4, 1024, 1024]);  addmm_61 = None
        view_166 = torch.ops.aten.view.default(view_165, [4, -1, 16, 64]);  view_165 = None
        permute_102 = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_61 = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_167 = torch.ops.aten.view.default(add_76, [4096, 1024]);  add_76 = None
        permute_103 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg172_1, view_167, permute_103);  arg172_1 = view_167 = permute_103 = None
        view_168 = torch.ops.aten.view.default(addmm_62, [4, 1024, 1024]);  addmm_62 = None
        view_169 = torch.ops.aten.view.default(view_168, [4, -1, 16, 64]);  view_168 = None
        permute_104 = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_62 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_170 = torch.ops.aten.view.default(view_163, [4, 1024, 16, 64]);  view_163 = None
        permute_105 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_63 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
        expand_33 = torch.ops.aten.expand.default(unsqueeze_23, [4, 1, 1024, 1024]);  unsqueeze_23 = None
        expand_34 = torch.ops.aten.expand.default(expand_33, [4, 16, 1024, 1024]);  expand_33 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_63, clone_61, clone_62, expand_34, False);  clone_63 = expand_34 = None
        getitem_84 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_106 = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None
        view_171 = torch.ops.aten.view.default(permute_106, [4, 1024, 1024]);  permute_106 = None
        view_172 = torch.ops.aten.view.default(view_171, [4096, 1024]);  view_171 = None
        permute_107 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg174_1, view_172, permute_107);  arg174_1 = view_172 = permute_107 = None
        view_173 = torch.ops.aten.view.default(addmm_63, [4, 1024, 1024]);  addmm_63 = None
        add_77 = torch.ops.aten.add.Tensor(add_74, view_173);  add_74 = view_173 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_77, getitem_89);  getitem_89 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, arg175_1);  mul_75 = arg175_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_76, arg176_1);  mul_76 = arg176_1 = None
        view_174 = torch.ops.aten.view.default(add_79, [4096, 1024]);  add_79 = None
        permute_108 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg178_1, view_174, permute_108);  arg178_1 = view_174 = permute_108 = None
        view_175 = torch.ops.aten.view.default(addmm_64, [4, 1024, 4096]);  addmm_64 = None
        mul_77 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_78 = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_10 = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_80 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_77, add_80);  mul_77 = add_80 = None
        view_176 = torch.ops.aten.view.default(mul_79, [4096, 4096]);  mul_79 = None
        permute_109 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg180_1, view_176, permute_109);  arg180_1 = view_176 = permute_109 = None
        view_177 = torch.ops.aten.view.default(addmm_65, [4, 1024, 1024]);  addmm_65 = None
        add_81 = torch.ops.aten.add.Tensor(add_77, view_177);  add_77 = view_177 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_23[0]
        getitem_91 = var_mean_23[1];  var_mean_23 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_81, getitem_91);  getitem_91 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg181_1);  mul_80 = arg181_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_81, arg182_1);  mul_81 = arg182_1 = None
        view_178 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_110 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg184_1, view_178, permute_110);  arg184_1 = view_178 = permute_110 = None
        view_179 = torch.ops.aten.view.default(addmm_66, [4, 1024, 1024]);  addmm_66 = None
        view_180 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_111 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg186_1, view_180, permute_111);  arg186_1 = view_180 = permute_111 = None
        view_181 = torch.ops.aten.view.default(addmm_67, [4, 1024, 1024]);  addmm_67 = None
        view_182 = torch.ops.aten.view.default(view_181, [4, -1, 16, 64]);  view_181 = None
        permute_112 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_67 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_183 = torch.ops.aten.view.default(add_83, [4096, 1024]);  add_83 = None
        permute_113 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg188_1, view_183, permute_113);  arg188_1 = view_183 = permute_113 = None
        view_184 = torch.ops.aten.view.default(addmm_68, [4, 1024, 1024]);  addmm_68 = None
        view_185 = torch.ops.aten.view.default(view_184, [4, -1, 16, 64]);  view_184 = None
        permute_114 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_68 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_186 = torch.ops.aten.view.default(view_179, [4, 1024, 16, 64]);  view_179 = None
        permute_115 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_69 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
        expand_36 = torch.ops.aten.expand.default(unsqueeze_25, [4, 1, 1024, 1024]);  unsqueeze_25 = None
        expand_37 = torch.ops.aten.expand.default(expand_36, [4, 16, 1024, 1024]);  expand_36 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_69, clone_67, clone_68, expand_37, False);  clone_69 = expand_37 = None
        getitem_92 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_116 = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None
        view_187 = torch.ops.aten.view.default(permute_116, [4, 1024, 1024]);  permute_116 = None
        view_188 = torch.ops.aten.view.default(view_187, [4096, 1024]);  view_187 = None
        permute_117 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg190_1, view_188, permute_117);  arg190_1 = view_188 = permute_117 = None
        view_189 = torch.ops.aten.view.default(addmm_69, [4, 1024, 1024]);  addmm_69 = None
        add_84 = torch.ops.aten.add.Tensor(add_81, view_189);  add_81 = view_189 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_84, getitem_97);  getitem_97 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg191_1);  mul_82 = arg191_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_83, arg192_1);  mul_83 = arg192_1 = None
        view_190 = torch.ops.aten.view.default(add_86, [4096, 1024]);  add_86 = None
        permute_118 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg194_1, view_190, permute_118);  arg194_1 = view_190 = permute_118 = None
        view_191 = torch.ops.aten.view.default(addmm_70, [4, 1024, 4096]);  addmm_70 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_191, 0.5)
        mul_85 = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476);  view_191 = None
        erf_11 = torch.ops.aten.erf.default(mul_85);  mul_85 = None
        add_87 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
        view_192 = torch.ops.aten.view.default(mul_86, [4096, 4096]);  mul_86 = None
        permute_119 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg196_1, view_192, permute_119);  arg196_1 = view_192 = permute_119 = None
        view_193 = torch.ops.aten.view.default(addmm_71, [4, 1024, 1024]);  addmm_71 = None
        add_88 = torch.ops.aten.add.Tensor(add_84, view_193);  add_84 = view_193 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_98 = var_mean_25[0]
        getitem_99 = var_mean_25[1];  var_mean_25 = None
        add_89 = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_88, getitem_99);  add_88 = getitem_99 = None
        mul_87 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, arg197_1);  mul_87 = arg197_1 = None
        add_90 = torch.ops.aten.add.Tensor(mul_88, arg198_1);  mul_88 = arg198_1 = None
        permute_120 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_194 = torch.ops.aten.view.default(add_90, [4096, 1024]);  add_90 = None
        full_default_4 = torch.ops.aten.full.default([1024, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_120, full_default_4], 1);  permute_120 = full_default_4 = None
        mm_default = torch.ops.aten.mm.default(view_194, cat_default);  view_194 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_195 = torch.ops.aten.view.default(slice_tensor, [4, 1024, 50265]);  slice_tensor = None
        view_196 = torch.ops.aten.view.default(view_195, [-1, 50265])
        view_197 = torch.ops.aten.view.default(arg199_1, [-1]);  arg199_1 = None
        amax = torch.ops.aten.amax.default(view_196, [1], True)
        sub_26 = torch.ops.aten.sub.Tensor(view_196, amax);  view_196 = amax = None
        exp = torch.ops.aten.exp.default(sub_26)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne = torch.ops.aten.ne.Scalar(view_197, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_197, full_default_2);  ne = full_default_2 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_26);  sub_27 = unsqueeze_26 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_197, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_197, -100);  view_197 = None
        sum_2 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_195, clone_1, clone_2, clone_7, clone_8, clone_13, clone_14, clone_19, clone_20, clone_25, clone_26, clone_31, clone_32, clone_37, clone_38, clone_43, clone_44, clone_49, clone_50, clone_55, clone_56, clone_61, clone_62, clone_67, clone_68)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 205885440, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50265, 1024), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4202496, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1026, 1024), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1024,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1024,), is_leaf=True)  # arg5_1
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
    buf13 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1024, 1024), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1024,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1024,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1024,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (4096, 1024), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf18, (4096,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024, 4096), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1024,), is_leaf=True)  # arg21_1
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
    buf29 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1024, 1024), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1024,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1024,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1024,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf33, (4096, 1024), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf34, (4096,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1024, 4096), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1024,), is_leaf=True)  # arg37_1
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
    buf45 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1024, 1024), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1024,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1024,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1024,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf49, (4096, 1024), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf50, (4096,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024, 4096), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1024,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1024,), is_leaf=True)  # arg53_1
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
    buf61 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1024, 1024), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1024,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1024,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf64, (1024,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf65, (4096, 1024), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf66, (4096,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1024, 4096), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1024,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1024,), is_leaf=True)  # arg69_1
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
    buf77 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1024, 1024), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1024,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1024,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf80, (1024,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf81, (4096, 1024), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf82, (4096,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1024, 4096), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1024,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1024,), is_leaf=True)  # arg85_1
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
    buf93 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1024, 1024), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf94, (1024,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1024,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1024,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf97, (4096, 1024), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf98, (4096,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1024, 4096), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1024,), is_leaf=True)  # arg101_1
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
    buf109 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1024, 1024), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1024,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf111, (1024,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf112, (1024,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf113, (4096, 1024), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf114, (4096,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1024, 4096), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1024,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024,), is_leaf=True)  # arg117_1
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
    buf125 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1024, 1024), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1024,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1024,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf128, (1024,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf129, (4096, 1024), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf130, (4096,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024, 4096), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024,), is_leaf=True)  # arg133_1
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
    buf141 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1024, 1024), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1024,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1024,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1024,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf145, (4096, 1024), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf146, (4096,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1024, 4096), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1024,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1024,), is_leaf=True)  # arg149_1
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
    buf157 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1024, 1024), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1024,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1024,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf160, (1024,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf161, (4096, 1024), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf162, (4096,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1024, 4096), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf164, (1024,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf165, (1024,), is_leaf=True)  # arg165_1
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
    buf173 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1024, 1024), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1024,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1024,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1024,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf177, (4096, 1024), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf178, (4096,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1024, 4096), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1024,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1024,), is_leaf=True)  # arg181_1
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
    buf189 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1024, 1024), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf190, (1024,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1024,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf192, (1024,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf193, (4096, 1024), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf194, (4096,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1024, 4096), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1024,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1024,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1024,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf199, (4, 1024), dtype=torch.int64, is_leaf=True)  # arg199_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)