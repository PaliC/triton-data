
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1):
        full = torch.ops.aten.full.default([32, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand = torch.ops.aten.expand.default(arg1_1, [32, 512]);  arg1_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default = torch.ops.aten.full.default([32, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None
        embedding = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, arg283_1);  arg3_1 = arg283_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
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
        view = torch.ops.aten.view.default(add_3, [16384, 768])
        permute = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [32, 512, 384]);  addmm = None
        view_2 = torch.ops.aten.view.default(add_3, [16384, 768])
        permute_1 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg10_1, view_2, permute_1);  arg10_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [32, 512, 384]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(add_3, [16384, 768])
        permute_2 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg12_1, view_4, permute_2);  arg12_1 = view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(addmm_2, [32, 512, 384]);  addmm_2 = None
        permute_3 = torch.ops.aten.permute.default(add_3, [0, 2, 1])
        convolution = torch.ops.aten.convolution.default(permute_3, arg14_1, None, [1], [4], [1], False, [0], 768);  permute_3 = arg14_1 = None
        convolution_1 = torch.ops.aten.convolution.default(convolution, arg15_1, None, [1], [0], [1], False, [0], 1);  convolution = arg15_1 = None
        add_4 = torch.ops.aten.add.Tensor(convolution_1, arg13_1);  convolution_1 = arg13_1 = None
        view_6 = torch.ops.aten.view.default(view_1, [32, 512, 6, 64])
        view_7 = torch.ops.aten.view.default(view_3, [32, 512, 6, 64]);  view_3 = None
        view_8 = torch.ops.aten.view.default(view_5, [32, 512, 6, 64]);  view_5 = None
        permute_8 = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
        mul_3 = torch.ops.aten.mul.Tensor(permute_8, view_1);  permute_8 = view_1 = None
        permute_9 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        clone_1 = torch.ops.aten.clone.default(mul_3, memory_format = torch.contiguous_format);  mul_3 = None
        view_9 = torch.ops.aten.view.default(clone_1, [16384, 384]);  clone_1 = None
        mm = torch.ops.aten.mm.default(view_9, permute_9);  view_9 = permute_9 = None
        view_10 = torch.ops.aten.view.default(mm, [32, 512, 54]);  mm = None
        add_5 = torch.ops.aten.add.Tensor(view_10, arg17_1);  view_10 = arg17_1 = None
        view_11 = torch.ops.aten.view.default(add_5, [-1, 9, 1]);  add_5 = None
        amax = torch.ops.aten.amax.default(view_11, [1], True)
        sub_2 = torch.ops.aten.sub.Tensor(view_11, amax);  view_11 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        view_12 = torch.ops.aten.view.default(add_3, [16384, 768])
        permute_10 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg19_1, view_12, permute_10);  arg19_1 = view_12 = permute_10 = None
        view_13 = torch.ops.aten.view.default(addmm_3, [32, 512, 384]);  addmm_3 = None
        view_14 = torch.ops.aten.view.default(view_13, [32, -1, 384]);  view_13 = None
        permute_11 = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        clone_2 = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(clone_2, -1);  clone_2 = None
        iota = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add_6 = torch.ops.aten.add.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        iota_2 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        add_7 = torch.ops.aten.add.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 0, 4, 4], 0.0);  unsqueeze_2 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
        index = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_8, add_7]);  constant_pad_nd = unsqueeze_8 = add_7 = None
        permute_12 = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_15 = torch.ops.aten.view.default(permute_12, [32, 3456, 512]);  permute_12 = None
        permute_13 = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        view_16 = torch.ops.aten.view.default(permute_13, [32, 512, 384, 9]);  permute_13 = None
        clone_3 = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format);  view_16 = None
        view_17 = torch.ops.aten.view.default(clone_3, [98304, 64, 9]);  clone_3 = None
        expand_1 = torch.ops.aten.expand.default(view_17, [98304, 64, 9]);  view_17 = None
        expand_2 = torch.ops.aten.expand.default(div, [98304, 9, 1]);  div = None
        bmm = torch.ops.aten.bmm.default(expand_1, expand_2);  expand_1 = expand_2 = None
        view_21 = torch.ops.aten.view.default(bmm, [-1, 384]);  bmm = None
        permute_default_33 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_default_34 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        permute_default_35 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_63 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        permute_15 = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        clone_8 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_28 = torch.ops.aten.view.default(view_21, [32, -1, 6, 64]);  view_21 = None
        cat = torch.ops.aten.cat.default([clone_8, view_28], 2);  clone_8 = view_28 = None
        view_29 = torch.ops.aten.view.default(cat, [32, 512, 768]);  cat = None
        view_30 = torch.ops.aten.view.default(view_29, [16384, 768]);  view_29 = None
        permute_16 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg21_1, view_30, permute_16);  arg21_1 = view_30 = permute_16 = None
        view_31 = torch.ops.aten.view.default(addmm_4, [32, 512, 768]);  addmm_4 = None
        add_9 = torch.ops.aten.add.Tensor(view_31, add_3);  view_31 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_3);  add_9 = getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg22_1);  mul_4 = arg22_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_5, arg23_1);  mul_5 = arg23_1 = None
        view_32 = torch.ops.aten.view.default(add_11, [16384, 768])
        permute_17 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg25_1, view_32, permute_17);  arg25_1 = view_32 = permute_17 = None
        view_33 = torch.ops.aten.view.default(addmm_5, [32, 512, 3072]);  addmm_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_33, 0.5)
        mul_7 = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
        erf = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_12 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None
        view_34 = torch.ops.aten.view.default(mul_8, [16384, 3072]);  mul_8 = None
        permute_18 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg27_1, view_34, permute_18);  arg27_1 = view_34 = permute_18 = None
        view_35 = torch.ops.aten.view.default(addmm_6, [32, 512, 768]);  addmm_6 = None
        add_13 = torch.ops.aten.add.Tensor(view_35, add_11);  view_35 = add_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_13, getitem_5);  add_13 = getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg28_1);  mul_9 = arg28_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
        view_36 = torch.ops.aten.view.default(add_15, [16384, 768])
        permute_19 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg31_1, view_36, permute_19);  arg31_1 = view_36 = permute_19 = None
        view_37 = torch.ops.aten.view.default(addmm_7, [32, 512, 384]);  addmm_7 = None
        view_38 = torch.ops.aten.view.default(add_15, [16384, 768])
        permute_20 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg33_1, view_38, permute_20);  arg33_1 = view_38 = permute_20 = None
        view_39 = torch.ops.aten.view.default(addmm_8, [32, 512, 384]);  addmm_8 = None
        view_40 = torch.ops.aten.view.default(add_15, [16384, 768])
        permute_21 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg35_1, view_40, permute_21);  arg35_1 = view_40 = permute_21 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [32, 512, 384]);  addmm_9 = None
        permute_22 = torch.ops.aten.permute.default(add_15, [0, 2, 1])
        convolution_2 = torch.ops.aten.convolution.default(permute_22, arg37_1, None, [1], [4], [1], False, [0], 768);  permute_22 = arg37_1 = None
        convolution_3 = torch.ops.aten.convolution.default(convolution_2, arg38_1, None, [1], [0], [1], False, [0], 1);  convolution_2 = arg38_1 = None
        add_16 = torch.ops.aten.add.Tensor(convolution_3, arg36_1);  convolution_3 = arg36_1 = None
        view_42 = torch.ops.aten.view.default(view_37, [32, 512, 6, 64])
        view_43 = torch.ops.aten.view.default(view_39, [32, 512, 6, 64]);  view_39 = None
        view_44 = torch.ops.aten.view.default(view_41, [32, 512, 6, 64]);  view_41 = None
        permute_27 = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
        mul_11 = torch.ops.aten.mul.Tensor(permute_27, view_37);  permute_27 = view_37 = None
        permute_28 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        clone_11 = torch.ops.aten.clone.default(mul_11, memory_format = torch.contiguous_format);  mul_11 = None
        view_45 = torch.ops.aten.view.default(clone_11, [16384, 384]);  clone_11 = None
        mm_1 = torch.ops.aten.mm.default(view_45, permute_28);  view_45 = permute_28 = None
        view_46 = torch.ops.aten.view.default(mm_1, [32, 512, 54]);  mm_1 = None
        add_17 = torch.ops.aten.add.Tensor(view_46, arg40_1);  view_46 = arg40_1 = None
        view_47 = torch.ops.aten.view.default(add_17, [-1, 9, 1]);  add_17 = None
        amax_2 = torch.ops.aten.amax.default(view_47, [1], True)
        sub_6 = torch.ops.aten.sub.Tensor(view_47, amax_2);  view_47 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        view_48 = torch.ops.aten.view.default(add_15, [16384, 768])
        permute_29 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg42_1, view_48, permute_29);  arg42_1 = view_48 = permute_29 = None
        view_49 = torch.ops.aten.view.default(addmm_10, [32, 512, 384]);  addmm_10 = None
        view_50 = torch.ops.aten.view.default(view_49, [32, -1, 384]);  view_49 = None
        permute_30 = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        clone_12 = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(clone_12, -1);  clone_12 = None
        iota_4 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        iota_5 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        add_18 = torch.ops.aten.add.Tensor(unsqueeze_10, unsqueeze_11);  unsqueeze_10 = unsqueeze_11 = None
        iota_6 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
        iota_7 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        add_19 = torch.ops.aten.add.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None
        constant_pad_nd_1 = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 0, 4, 4], 0.0);  unsqueeze_9 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(add_18, -1);  add_18 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        index_1 = torch.ops.aten.index.Tensor(constant_pad_nd_1, [None, None, unsqueeze_15, add_19]);  constant_pad_nd_1 = unsqueeze_15 = add_19 = None
        permute_31 = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        view_51 = torch.ops.aten.view.default(permute_31, [32, 3456, 512]);  permute_31 = None
        permute_32 = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        view_52 = torch.ops.aten.view.default(permute_32, [32, 512, 384, 9]);  permute_32 = None
        clone_13 = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
        view_53 = torch.ops.aten.view.default(clone_13, [98304, 64, 9]);  clone_13 = None
        expand_7 = torch.ops.aten.expand.default(view_53, [98304, 64, 9]);  view_53 = None
        expand_8 = torch.ops.aten.expand.default(div_3, [98304, 9, 1]);  div_3 = None
        bmm_3 = torch.ops.aten.bmm.default(expand_7, expand_8);  expand_7 = expand_8 = None
        view_57 = torch.ops.aten.view.default(bmm_3, [-1, 384]);  bmm_3 = None
        permute_default_30 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        permute_default_31 = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        permute_default_32 = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_62 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        permute_34 = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
        clone_18 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_64 = torch.ops.aten.view.default(view_57, [32, -1, 6, 64]);  view_57 = None
        cat_1 = torch.ops.aten.cat.default([clone_18, view_64], 2);  clone_18 = view_64 = None
        view_65 = torch.ops.aten.view.default(cat_1, [32, 512, 768]);  cat_1 = None
        view_66 = torch.ops.aten.view.default(view_65, [16384, 768]);  view_65 = None
        permute_35 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg44_1, view_66, permute_35);  arg44_1 = view_66 = permute_35 = None
        view_67 = torch.ops.aten.view.default(addmm_11, [32, 512, 768]);  addmm_11 = None
        add_21 = torch.ops.aten.add.Tensor(view_67, add_15);  view_67 = add_15 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_21, getitem_7);  add_21 = getitem_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg45_1);  mul_12 = arg45_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_13, arg46_1);  mul_13 = arg46_1 = None
        view_68 = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_36 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg48_1, view_68, permute_36);  arg48_1 = view_68 = permute_36 = None
        view_69 = torch.ops.aten.view.default(addmm_12, [32, 512, 3072]);  addmm_12 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_69, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_24 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_24);  mul_14 = add_24 = None
        view_70 = torch.ops.aten.view.default(mul_16, [16384, 3072]);  mul_16 = None
        permute_37 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg50_1, view_70, permute_37);  arg50_1 = view_70 = permute_37 = None
        view_71 = torch.ops.aten.view.default(addmm_13, [32, 512, 768]);  addmm_13 = None
        add_25 = torch.ops.aten.add.Tensor(view_71, add_23);  view_71 = add_23 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_25, getitem_9);  add_25 = getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg51_1);  mul_17 = arg51_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_18, arg52_1);  mul_18 = arg52_1 = None
        view_72 = torch.ops.aten.view.default(add_27, [16384, 768])
        permute_38 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg54_1, view_72, permute_38);  arg54_1 = view_72 = permute_38 = None
        view_73 = torch.ops.aten.view.default(addmm_14, [32, 512, 384]);  addmm_14 = None
        view_74 = torch.ops.aten.view.default(add_27, [16384, 768])
        permute_39 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg56_1, view_74, permute_39);  arg56_1 = view_74 = permute_39 = None
        view_75 = torch.ops.aten.view.default(addmm_15, [32, 512, 384]);  addmm_15 = None
        view_76 = torch.ops.aten.view.default(add_27, [16384, 768])
        permute_40 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg58_1, view_76, permute_40);  arg58_1 = view_76 = permute_40 = None
        view_77 = torch.ops.aten.view.default(addmm_16, [32, 512, 384]);  addmm_16 = None
        permute_41 = torch.ops.aten.permute.default(add_27, [0, 2, 1])
        convolution_4 = torch.ops.aten.convolution.default(permute_41, arg60_1, None, [1], [4], [1], False, [0], 768);  permute_41 = arg60_1 = None
        convolution_5 = torch.ops.aten.convolution.default(convolution_4, arg61_1, None, [1], [0], [1], False, [0], 1);  convolution_4 = arg61_1 = None
        add_28 = torch.ops.aten.add.Tensor(convolution_5, arg59_1);  convolution_5 = arg59_1 = None
        view_78 = torch.ops.aten.view.default(view_73, [32, 512, 6, 64])
        view_79 = torch.ops.aten.view.default(view_75, [32, 512, 6, 64]);  view_75 = None
        view_80 = torch.ops.aten.view.default(view_77, [32, 512, 6, 64]);  view_77 = None
        permute_46 = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
        mul_19 = torch.ops.aten.mul.Tensor(permute_46, view_73);  permute_46 = view_73 = None
        permute_47 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        clone_21 = torch.ops.aten.clone.default(mul_19, memory_format = torch.contiguous_format);  mul_19 = None
        view_81 = torch.ops.aten.view.default(clone_21, [16384, 384]);  clone_21 = None
        mm_2 = torch.ops.aten.mm.default(view_81, permute_47);  view_81 = permute_47 = None
        view_82 = torch.ops.aten.view.default(mm_2, [32, 512, 54]);  mm_2 = None
        add_29 = torch.ops.aten.add.Tensor(view_82, arg63_1);  view_82 = arg63_1 = None
        view_83 = torch.ops.aten.view.default(add_29, [-1, 9, 1]);  add_29 = None
        amax_4 = torch.ops.aten.amax.default(view_83, [1], True)
        sub_10 = torch.ops.aten.sub.Tensor(view_83, amax_4);  view_83 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        view_84 = torch.ops.aten.view.default(add_27, [16384, 768])
        permute_48 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg65_1, view_84, permute_48);  arg65_1 = view_84 = permute_48 = None
        view_85 = torch.ops.aten.view.default(addmm_17, [32, 512, 384]);  addmm_17 = None
        view_86 = torch.ops.aten.view.default(view_85, [32, -1, 384]);  view_85 = None
        permute_49 = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
        clone_22 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(clone_22, -1);  clone_22 = None
        iota_8 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
        iota_9 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        add_30 = torch.ops.aten.add.Tensor(unsqueeze_17, unsqueeze_18);  unsqueeze_17 = unsqueeze_18 = None
        iota_10 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(iota_10, 0);  iota_10 = None
        iota_11 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        add_31 = torch.ops.aten.add.Tensor(unsqueeze_19, unsqueeze_20);  unsqueeze_19 = unsqueeze_20 = None
        constant_pad_nd_2 = torch.ops.aten.constant_pad_nd.default(unsqueeze_16, [0, 0, 4, 4], 0.0);  unsqueeze_16 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(add_30, -1);  add_30 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
        index_2 = torch.ops.aten.index.Tensor(constant_pad_nd_2, [None, None, unsqueeze_22, add_31]);  constant_pad_nd_2 = unsqueeze_22 = add_31 = None
        permute_50 = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
        view_87 = torch.ops.aten.view.default(permute_50, [32, 3456, 512]);  permute_50 = None
        permute_51 = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        view_88 = torch.ops.aten.view.default(permute_51, [32, 512, 384, 9]);  permute_51 = None
        clone_23 = torch.ops.aten.clone.default(view_88, memory_format = torch.contiguous_format);  view_88 = None
        view_89 = torch.ops.aten.view.default(clone_23, [98304, 64, 9]);  clone_23 = None
        expand_13 = torch.ops.aten.expand.default(view_89, [98304, 64, 9]);  view_89 = None
        expand_14 = torch.ops.aten.expand.default(div_6, [98304, 9, 1]);  div_6 = None
        bmm_6 = torch.ops.aten.bmm.default(expand_13, expand_14);  expand_13 = expand_14 = None
        view_93 = torch.ops.aten.view.default(bmm_6, [-1, 384]);  bmm_6 = None
        permute_default_27 = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        permute_default_28 = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        permute_default_29 = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_61 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        permute_53 = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
        clone_28 = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_100 = torch.ops.aten.view.default(view_93, [32, -1, 6, 64]);  view_93 = None
        cat_2 = torch.ops.aten.cat.default([clone_28, view_100], 2);  clone_28 = view_100 = None
        view_101 = torch.ops.aten.view.default(cat_2, [32, 512, 768]);  cat_2 = None
        view_102 = torch.ops.aten.view.default(view_101, [16384, 768]);  view_101 = None
        permute_54 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg67_1, view_102, permute_54);  arg67_1 = view_102 = permute_54 = None
        view_103 = torch.ops.aten.view.default(addmm_18, [32, 512, 768]);  addmm_18 = None
        add_33 = torch.ops.aten.add.Tensor(view_103, add_27);  view_103 = add_27 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_33, getitem_11);  add_33 = getitem_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg68_1);  mul_20 = arg68_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_21, arg69_1);  mul_21 = arg69_1 = None
        view_104 = torch.ops.aten.view.default(add_35, [16384, 768])
        permute_55 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg71_1, view_104, permute_55);  arg71_1 = view_104 = permute_55 = None
        view_105 = torch.ops.aten.view.default(addmm_19, [32, 512, 3072]);  addmm_19 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_105, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476);  view_105 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_36 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_36);  mul_22 = add_36 = None
        view_106 = torch.ops.aten.view.default(mul_24, [16384, 3072]);  mul_24 = None
        permute_56 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg73_1, view_106, permute_56);  arg73_1 = view_106 = permute_56 = None
        view_107 = torch.ops.aten.view.default(addmm_20, [32, 512, 768]);  addmm_20 = None
        add_37 = torch.ops.aten.add.Tensor(view_107, add_35);  view_107 = add_35 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = rsqrt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg74_1);  mul_25 = arg74_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_26, arg75_1);  mul_26 = arg75_1 = None
        view_108 = torch.ops.aten.view.default(add_39, [16384, 768])
        permute_57 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg77_1, view_108, permute_57);  arg77_1 = view_108 = permute_57 = None
        view_109 = torch.ops.aten.view.default(addmm_21, [32, 512, 384]);  addmm_21 = None
        view_110 = torch.ops.aten.view.default(add_39, [16384, 768])
        permute_58 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg79_1, view_110, permute_58);  arg79_1 = view_110 = permute_58 = None
        view_111 = torch.ops.aten.view.default(addmm_22, [32, 512, 384]);  addmm_22 = None
        view_112 = torch.ops.aten.view.default(add_39, [16384, 768])
        permute_59 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg81_1, view_112, permute_59);  arg81_1 = view_112 = permute_59 = None
        view_113 = torch.ops.aten.view.default(addmm_23, [32, 512, 384]);  addmm_23 = None
        permute_60 = torch.ops.aten.permute.default(add_39, [0, 2, 1])
        convolution_6 = torch.ops.aten.convolution.default(permute_60, arg83_1, None, [1], [4], [1], False, [0], 768);  permute_60 = arg83_1 = None
        convolution_7 = torch.ops.aten.convolution.default(convolution_6, arg84_1, None, [1], [0], [1], False, [0], 1);  convolution_6 = arg84_1 = None
        add_40 = torch.ops.aten.add.Tensor(convolution_7, arg82_1);  convolution_7 = arg82_1 = None
        view_114 = torch.ops.aten.view.default(view_109, [32, 512, 6, 64])
        view_115 = torch.ops.aten.view.default(view_111, [32, 512, 6, 64]);  view_111 = None
        view_116 = torch.ops.aten.view.default(view_113, [32, 512, 6, 64]);  view_113 = None
        permute_65 = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
        mul_27 = torch.ops.aten.mul.Tensor(permute_65, view_109);  permute_65 = view_109 = None
        permute_66 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        clone_31 = torch.ops.aten.clone.default(mul_27, memory_format = torch.contiguous_format);  mul_27 = None
        view_117 = torch.ops.aten.view.default(clone_31, [16384, 384]);  clone_31 = None
        mm_3 = torch.ops.aten.mm.default(view_117, permute_66);  view_117 = permute_66 = None
        view_118 = torch.ops.aten.view.default(mm_3, [32, 512, 54]);  mm_3 = None
        add_41 = torch.ops.aten.add.Tensor(view_118, arg86_1);  view_118 = arg86_1 = None
        view_119 = torch.ops.aten.view.default(add_41, [-1, 9, 1]);  add_41 = None
        amax_6 = torch.ops.aten.amax.default(view_119, [1], True)
        sub_14 = torch.ops.aten.sub.Tensor(view_119, amax_6);  view_119 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        view_120 = torch.ops.aten.view.default(add_39, [16384, 768])
        permute_67 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg88_1, view_120, permute_67);  arg88_1 = view_120 = permute_67 = None
        view_121 = torch.ops.aten.view.default(addmm_24, [32, 512, 384]);  addmm_24 = None
        view_122 = torch.ops.aten.view.default(view_121, [32, -1, 384]);  view_121 = None
        permute_68 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        clone_32 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(clone_32, -1);  clone_32 = None
        iota_12 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(iota_12, 0);  iota_12 = None
        iota_13 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        add_42 = torch.ops.aten.add.Tensor(unsqueeze_24, unsqueeze_25);  unsqueeze_24 = unsqueeze_25 = None
        iota_14 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(iota_14, 0);  iota_14 = None
        iota_15 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        add_43 = torch.ops.aten.add.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
        constant_pad_nd_3 = torch.ops.aten.constant_pad_nd.default(unsqueeze_23, [0, 0, 4, 4], 0.0);  unsqueeze_23 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(add_42, -1);  add_42 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        index_3 = torch.ops.aten.index.Tensor(constant_pad_nd_3, [None, None, unsqueeze_29, add_43]);  constant_pad_nd_3 = unsqueeze_29 = add_43 = None
        permute_69 = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
        view_123 = torch.ops.aten.view.default(permute_69, [32, 3456, 512]);  permute_69 = None
        permute_70 = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        view_124 = torch.ops.aten.view.default(permute_70, [32, 512, 384, 9]);  permute_70 = None
        clone_33 = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
        view_125 = torch.ops.aten.view.default(clone_33, [98304, 64, 9]);  clone_33 = None
        expand_19 = torch.ops.aten.expand.default(view_125, [98304, 64, 9]);  view_125 = None
        expand_20 = torch.ops.aten.expand.default(div_9, [98304, 9, 1]);  div_9 = None
        bmm_9 = torch.ops.aten.bmm.default(expand_19, expand_20);  expand_19 = expand_20 = None
        view_129 = torch.ops.aten.view.default(bmm_9, [-1, 384]);  bmm_9 = None
        permute_default_24 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_default_25 = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        permute_default_26 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_60 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        permute_72 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        clone_38 = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_136 = torch.ops.aten.view.default(view_129, [32, -1, 6, 64]);  view_129 = None
        cat_3 = torch.ops.aten.cat.default([clone_38, view_136], 2);  clone_38 = view_136 = None
        view_137 = torch.ops.aten.view.default(cat_3, [32, 512, 768]);  cat_3 = None
        view_138 = torch.ops.aten.view.default(view_137, [16384, 768]);  view_137 = None
        permute_73 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg90_1, view_138, permute_73);  arg90_1 = view_138 = permute_73 = None
        view_139 = torch.ops.aten.view.default(addmm_25, [32, 512, 768]);  addmm_25 = None
        add_45 = torch.ops.aten.add.Tensor(view_139, add_39);  view_139 = add_39 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_45, getitem_15);  add_45 = getitem_15 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg91_1);  mul_28 = arg91_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_29, arg92_1);  mul_29 = arg92_1 = None
        view_140 = torch.ops.aten.view.default(add_47, [16384, 768])
        permute_74 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg94_1, view_140, permute_74);  arg94_1 = view_140 = permute_74 = None
        view_141 = torch.ops.aten.view.default(addmm_26, [32, 512, 3072]);  addmm_26 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_3 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_48 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_48);  mul_30 = add_48 = None
        view_142 = torch.ops.aten.view.default(mul_32, [16384, 3072]);  mul_32 = None
        permute_75 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg96_1, view_142, permute_75);  arg96_1 = view_142 = permute_75 = None
        view_143 = torch.ops.aten.view.default(addmm_27, [32, 512, 768]);  addmm_27 = None
        add_49 = torch.ops.aten.add.Tensor(view_143, add_47);  view_143 = add_47 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_49, getitem_17);  add_49 = getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = rsqrt_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg97_1);  mul_33 = arg97_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_34, arg98_1);  mul_34 = arg98_1 = None
        view_144 = torch.ops.aten.view.default(add_51, [16384, 768])
        permute_76 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg100_1, view_144, permute_76);  arg100_1 = view_144 = permute_76 = None
        view_145 = torch.ops.aten.view.default(addmm_28, [32, 512, 384]);  addmm_28 = None
        view_146 = torch.ops.aten.view.default(add_51, [16384, 768])
        permute_77 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg102_1, view_146, permute_77);  arg102_1 = view_146 = permute_77 = None
        view_147 = torch.ops.aten.view.default(addmm_29, [32, 512, 384]);  addmm_29 = None
        view_148 = torch.ops.aten.view.default(add_51, [16384, 768])
        permute_78 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg104_1, view_148, permute_78);  arg104_1 = view_148 = permute_78 = None
        view_149 = torch.ops.aten.view.default(addmm_30, [32, 512, 384]);  addmm_30 = None
        permute_79 = torch.ops.aten.permute.default(add_51, [0, 2, 1])
        convolution_8 = torch.ops.aten.convolution.default(permute_79, arg106_1, None, [1], [4], [1], False, [0], 768);  permute_79 = arg106_1 = None
        convolution_9 = torch.ops.aten.convolution.default(convolution_8, arg107_1, None, [1], [0], [1], False, [0], 1);  convolution_8 = arg107_1 = None
        add_52 = torch.ops.aten.add.Tensor(convolution_9, arg105_1);  convolution_9 = arg105_1 = None
        view_150 = torch.ops.aten.view.default(view_145, [32, 512, 6, 64])
        view_151 = torch.ops.aten.view.default(view_147, [32, 512, 6, 64]);  view_147 = None
        view_152 = torch.ops.aten.view.default(view_149, [32, 512, 6, 64]);  view_149 = None
        permute_84 = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
        mul_35 = torch.ops.aten.mul.Tensor(permute_84, view_145);  permute_84 = view_145 = None
        permute_85 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        clone_41 = torch.ops.aten.clone.default(mul_35, memory_format = torch.contiguous_format);  mul_35 = None
        view_153 = torch.ops.aten.view.default(clone_41, [16384, 384]);  clone_41 = None
        mm_4 = torch.ops.aten.mm.default(view_153, permute_85);  view_153 = permute_85 = None
        view_154 = torch.ops.aten.view.default(mm_4, [32, 512, 54]);  mm_4 = None
        add_53 = torch.ops.aten.add.Tensor(view_154, arg109_1);  view_154 = arg109_1 = None
        view_155 = torch.ops.aten.view.default(add_53, [-1, 9, 1]);  add_53 = None
        amax_8 = torch.ops.aten.amax.default(view_155, [1], True)
        sub_18 = torch.ops.aten.sub.Tensor(view_155, amax_8);  view_155 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        view_156 = torch.ops.aten.view.default(add_51, [16384, 768])
        permute_86 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg111_1, view_156, permute_86);  arg111_1 = view_156 = permute_86 = None
        view_157 = torch.ops.aten.view.default(addmm_31, [32, 512, 384]);  addmm_31 = None
        view_158 = torch.ops.aten.view.default(view_157, [32, -1, 384]);  view_157 = None
        permute_87 = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
        clone_42 = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(clone_42, -1);  clone_42 = None
        iota_16 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(iota_16, 0);  iota_16 = None
        iota_17 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        add_54 = torch.ops.aten.add.Tensor(unsqueeze_31, unsqueeze_32);  unsqueeze_31 = unsqueeze_32 = None
        iota_18 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(iota_18, 0);  iota_18 = None
        iota_19 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
        add_55 = torch.ops.aten.add.Tensor(unsqueeze_33, unsqueeze_34);  unsqueeze_33 = unsqueeze_34 = None
        constant_pad_nd_4 = torch.ops.aten.constant_pad_nd.default(unsqueeze_30, [0, 0, 4, 4], 0.0);  unsqueeze_30 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(add_54, -1);  add_54 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
        index_4 = torch.ops.aten.index.Tensor(constant_pad_nd_4, [None, None, unsqueeze_36, add_55]);  constant_pad_nd_4 = unsqueeze_36 = add_55 = None
        permute_88 = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
        view_159 = torch.ops.aten.view.default(permute_88, [32, 3456, 512]);  permute_88 = None
        permute_89 = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
        view_160 = torch.ops.aten.view.default(permute_89, [32, 512, 384, 9]);  permute_89 = None
        clone_43 = torch.ops.aten.clone.default(view_160, memory_format = torch.contiguous_format);  view_160 = None
        view_161 = torch.ops.aten.view.default(clone_43, [98304, 64, 9]);  clone_43 = None
        expand_25 = torch.ops.aten.expand.default(view_161, [98304, 64, 9]);  view_161 = None
        expand_26 = torch.ops.aten.expand.default(div_12, [98304, 9, 1]);  div_12 = None
        bmm_12 = torch.ops.aten.bmm.default(expand_25, expand_26);  expand_25 = expand_26 = None
        view_165 = torch.ops.aten.view.default(bmm_12, [-1, 384]);  bmm_12 = None
        permute_default_21 = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        permute_default_22 = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        permute_default_23 = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_59 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        permute_91 = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        clone_48 = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_172 = torch.ops.aten.view.default(view_165, [32, -1, 6, 64]);  view_165 = None
        cat_4 = torch.ops.aten.cat.default([clone_48, view_172], 2);  clone_48 = view_172 = None
        view_173 = torch.ops.aten.view.default(cat_4, [32, 512, 768]);  cat_4 = None
        view_174 = torch.ops.aten.view.default(view_173, [16384, 768]);  view_173 = None
        permute_92 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg113_1, view_174, permute_92);  arg113_1 = view_174 = permute_92 = None
        view_175 = torch.ops.aten.view.default(addmm_32, [32, 512, 768]);  addmm_32 = None
        add_57 = torch.ops.aten.add.Tensor(view_175, add_51);  view_175 = add_51 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_57, getitem_19);  add_57 = getitem_19 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg114_1);  mul_36 = arg114_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_37, arg115_1);  mul_37 = arg115_1 = None
        view_176 = torch.ops.aten.view.default(add_59, [16384, 768])
        permute_93 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg117_1, view_176, permute_93);  arg117_1 = view_176 = permute_93 = None
        view_177 = torch.ops.aten.view.default(addmm_33, [32, 512, 3072]);  addmm_33 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_177, 0.5)
        mul_39 = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476);  view_177 = None
        erf_4 = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_60 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, add_60);  mul_38 = add_60 = None
        view_178 = torch.ops.aten.view.default(mul_40, [16384, 3072]);  mul_40 = None
        permute_94 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg119_1, view_178, permute_94);  arg119_1 = view_178 = permute_94 = None
        view_179 = torch.ops.aten.view.default(addmm_34, [32, 512, 768]);  addmm_34 = None
        add_61 = torch.ops.aten.add.Tensor(view_179, add_59);  view_179 = add_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_61, getitem_21);  add_61 = getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_10);  sub_21 = rsqrt_10 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg120_1);  mul_41 = arg120_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_42, arg121_1);  mul_42 = arg121_1 = None
        view_180 = torch.ops.aten.view.default(add_63, [16384, 768])
        permute_95 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg123_1, view_180, permute_95);  arg123_1 = view_180 = permute_95 = None
        view_181 = torch.ops.aten.view.default(addmm_35, [32, 512, 384]);  addmm_35 = None
        view_182 = torch.ops.aten.view.default(add_63, [16384, 768])
        permute_96 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg125_1, view_182, permute_96);  arg125_1 = view_182 = permute_96 = None
        view_183 = torch.ops.aten.view.default(addmm_36, [32, 512, 384]);  addmm_36 = None
        view_184 = torch.ops.aten.view.default(add_63, [16384, 768])
        permute_97 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg127_1, view_184, permute_97);  arg127_1 = view_184 = permute_97 = None
        view_185 = torch.ops.aten.view.default(addmm_37, [32, 512, 384]);  addmm_37 = None
        permute_98 = torch.ops.aten.permute.default(add_63, [0, 2, 1])
        convolution_10 = torch.ops.aten.convolution.default(permute_98, arg129_1, None, [1], [4], [1], False, [0], 768);  permute_98 = arg129_1 = None
        convolution_11 = torch.ops.aten.convolution.default(convolution_10, arg130_1, None, [1], [0], [1], False, [0], 1);  convolution_10 = arg130_1 = None
        add_64 = torch.ops.aten.add.Tensor(convolution_11, arg128_1);  convolution_11 = arg128_1 = None
        view_186 = torch.ops.aten.view.default(view_181, [32, 512, 6, 64])
        view_187 = torch.ops.aten.view.default(view_183, [32, 512, 6, 64]);  view_183 = None
        view_188 = torch.ops.aten.view.default(view_185, [32, 512, 6, 64]);  view_185 = None
        permute_103 = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
        mul_43 = torch.ops.aten.mul.Tensor(permute_103, view_181);  permute_103 = view_181 = None
        permute_104 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        clone_51 = torch.ops.aten.clone.default(mul_43, memory_format = torch.contiguous_format);  mul_43 = None
        view_189 = torch.ops.aten.view.default(clone_51, [16384, 384]);  clone_51 = None
        mm_5 = torch.ops.aten.mm.default(view_189, permute_104);  view_189 = permute_104 = None
        view_190 = torch.ops.aten.view.default(mm_5, [32, 512, 54]);  mm_5 = None
        add_65 = torch.ops.aten.add.Tensor(view_190, arg132_1);  view_190 = arg132_1 = None
        view_191 = torch.ops.aten.view.default(add_65, [-1, 9, 1]);  add_65 = None
        amax_10 = torch.ops.aten.amax.default(view_191, [1], True)
        sub_22 = torch.ops.aten.sub.Tensor(view_191, amax_10);  view_191 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        view_192 = torch.ops.aten.view.default(add_63, [16384, 768])
        permute_105 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg134_1, view_192, permute_105);  arg134_1 = view_192 = permute_105 = None
        view_193 = torch.ops.aten.view.default(addmm_38, [32, 512, 384]);  addmm_38 = None
        view_194 = torch.ops.aten.view.default(view_193, [32, -1, 384]);  view_193 = None
        permute_106 = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
        clone_52 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(clone_52, -1);  clone_52 = None
        iota_20 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(iota_20, 0);  iota_20 = None
        iota_21 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        add_66 = torch.ops.aten.add.Tensor(unsqueeze_38, unsqueeze_39);  unsqueeze_38 = unsqueeze_39 = None
        iota_22 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(iota_22, 0);  iota_22 = None
        iota_23 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
        add_67 = torch.ops.aten.add.Tensor(unsqueeze_40, unsqueeze_41);  unsqueeze_40 = unsqueeze_41 = None
        constant_pad_nd_5 = torch.ops.aten.constant_pad_nd.default(unsqueeze_37, [0, 0, 4, 4], 0.0);  unsqueeze_37 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(add_66, -1);  add_66 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        index_5 = torch.ops.aten.index.Tensor(constant_pad_nd_5, [None, None, unsqueeze_43, add_67]);  constant_pad_nd_5 = unsqueeze_43 = add_67 = None
        permute_107 = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
        view_195 = torch.ops.aten.view.default(permute_107, [32, 3456, 512]);  permute_107 = None
        permute_108 = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        view_196 = torch.ops.aten.view.default(permute_108, [32, 512, 384, 9]);  permute_108 = None
        clone_53 = torch.ops.aten.clone.default(view_196, memory_format = torch.contiguous_format);  view_196 = None
        view_197 = torch.ops.aten.view.default(clone_53, [98304, 64, 9]);  clone_53 = None
        expand_31 = torch.ops.aten.expand.default(view_197, [98304, 64, 9]);  view_197 = None
        expand_32 = torch.ops.aten.expand.default(div_15, [98304, 9, 1]);  div_15 = None
        bmm_15 = torch.ops.aten.bmm.default(expand_31, expand_32);  expand_31 = expand_32 = None
        view_201 = torch.ops.aten.view.default(bmm_15, [-1, 384]);  bmm_15 = None
        permute_default_18 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        permute_default_19 = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        permute_default_20 = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_58 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        permute_110 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_58 = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_208 = torch.ops.aten.view.default(view_201, [32, -1, 6, 64]);  view_201 = None
        cat_5 = torch.ops.aten.cat.default([clone_58, view_208], 2);  clone_58 = view_208 = None
        view_209 = torch.ops.aten.view.default(cat_5, [32, 512, 768]);  cat_5 = None
        view_210 = torch.ops.aten.view.default(view_209, [16384, 768]);  view_209 = None
        permute_111 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg136_1, view_210, permute_111);  arg136_1 = view_210 = permute_111 = None
        view_211 = torch.ops.aten.view.default(addmm_39, [32, 512, 768]);  addmm_39 = None
        add_69 = torch.ops.aten.add.Tensor(view_211, add_63);  view_211 = add_63 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_69, getitem_23);  add_69 = getitem_23 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg137_1);  mul_44 = arg137_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_45, arg138_1);  mul_45 = arg138_1 = None
        view_212 = torch.ops.aten.view.default(add_71, [16384, 768])
        permute_112 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg140_1, view_212, permute_112);  arg140_1 = view_212 = permute_112 = None
        view_213 = torch.ops.aten.view.default(addmm_40, [32, 512, 3072]);  addmm_40 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_213, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
        erf_5 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_72 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_72);  mul_46 = add_72 = None
        view_214 = torch.ops.aten.view.default(mul_48, [16384, 3072]);  mul_48 = None
        permute_113 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg142_1, view_214, permute_113);  arg142_1 = view_214 = permute_113 = None
        view_215 = torch.ops.aten.view.default(addmm_41, [32, 512, 768]);  addmm_41 = None
        add_73 = torch.ops.aten.add.Tensor(view_215, add_71);  view_215 = add_71 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_73, getitem_25);  add_73 = getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_12);  sub_25 = rsqrt_12 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg143_1);  mul_49 = arg143_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_50, arg144_1);  mul_50 = arg144_1 = None
        view_216 = torch.ops.aten.view.default(add_75, [16384, 768])
        permute_114 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg146_1, view_216, permute_114);  arg146_1 = view_216 = permute_114 = None
        view_217 = torch.ops.aten.view.default(addmm_42, [32, 512, 384]);  addmm_42 = None
        view_218 = torch.ops.aten.view.default(add_75, [16384, 768])
        permute_115 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg148_1, view_218, permute_115);  arg148_1 = view_218 = permute_115 = None
        view_219 = torch.ops.aten.view.default(addmm_43, [32, 512, 384]);  addmm_43 = None
        view_220 = torch.ops.aten.view.default(add_75, [16384, 768])
        permute_116 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg150_1, view_220, permute_116);  arg150_1 = view_220 = permute_116 = None
        view_221 = torch.ops.aten.view.default(addmm_44, [32, 512, 384]);  addmm_44 = None
        permute_117 = torch.ops.aten.permute.default(add_75, [0, 2, 1])
        convolution_12 = torch.ops.aten.convolution.default(permute_117, arg152_1, None, [1], [4], [1], False, [0], 768);  permute_117 = arg152_1 = None
        convolution_13 = torch.ops.aten.convolution.default(convolution_12, arg153_1, None, [1], [0], [1], False, [0], 1);  convolution_12 = arg153_1 = None
        add_76 = torch.ops.aten.add.Tensor(convolution_13, arg151_1);  convolution_13 = arg151_1 = None
        view_222 = torch.ops.aten.view.default(view_217, [32, 512, 6, 64])
        view_223 = torch.ops.aten.view.default(view_219, [32, 512, 6, 64]);  view_219 = None
        view_224 = torch.ops.aten.view.default(view_221, [32, 512, 6, 64]);  view_221 = None
        permute_122 = torch.ops.aten.permute.default(add_76, [0, 2, 1]);  add_76 = None
        mul_51 = torch.ops.aten.mul.Tensor(permute_122, view_217);  permute_122 = view_217 = None
        permute_123 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        clone_61 = torch.ops.aten.clone.default(mul_51, memory_format = torch.contiguous_format);  mul_51 = None
        view_225 = torch.ops.aten.view.default(clone_61, [16384, 384]);  clone_61 = None
        mm_6 = torch.ops.aten.mm.default(view_225, permute_123);  view_225 = permute_123 = None
        view_226 = torch.ops.aten.view.default(mm_6, [32, 512, 54]);  mm_6 = None
        add_77 = torch.ops.aten.add.Tensor(view_226, arg155_1);  view_226 = arg155_1 = None
        view_227 = torch.ops.aten.view.default(add_77, [-1, 9, 1]);  add_77 = None
        amax_12 = torch.ops.aten.amax.default(view_227, [1], True)
        sub_26 = torch.ops.aten.sub.Tensor(view_227, amax_12);  view_227 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        view_228 = torch.ops.aten.view.default(add_75, [16384, 768])
        permute_124 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg157_1, view_228, permute_124);  arg157_1 = view_228 = permute_124 = None
        view_229 = torch.ops.aten.view.default(addmm_45, [32, 512, 384]);  addmm_45 = None
        view_230 = torch.ops.aten.view.default(view_229, [32, -1, 384]);  view_229 = None
        permute_125 = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        clone_62 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(clone_62, -1);  clone_62 = None
        iota_24 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(iota_24, 0);  iota_24 = None
        iota_25 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        add_78 = torch.ops.aten.add.Tensor(unsqueeze_45, unsqueeze_46);  unsqueeze_45 = unsqueeze_46 = None
        iota_26 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(iota_26, 0);  iota_26 = None
        iota_27 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
        add_79 = torch.ops.aten.add.Tensor(unsqueeze_47, unsqueeze_48);  unsqueeze_47 = unsqueeze_48 = None
        constant_pad_nd_6 = torch.ops.aten.constant_pad_nd.default(unsqueeze_44, [0, 0, 4, 4], 0.0);  unsqueeze_44 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(add_78, -1);  add_78 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
        index_6 = torch.ops.aten.index.Tensor(constant_pad_nd_6, [None, None, unsqueeze_50, add_79]);  constant_pad_nd_6 = unsqueeze_50 = add_79 = None
        permute_126 = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
        view_231 = torch.ops.aten.view.default(permute_126, [32, 3456, 512]);  permute_126 = None
        permute_127 = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        view_232 = torch.ops.aten.view.default(permute_127, [32, 512, 384, 9]);  permute_127 = None
        clone_63 = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
        view_233 = torch.ops.aten.view.default(clone_63, [98304, 64, 9]);  clone_63 = None
        expand_37 = torch.ops.aten.expand.default(view_233, [98304, 64, 9]);  view_233 = None
        expand_38 = torch.ops.aten.expand.default(div_18, [98304, 9, 1]);  div_18 = None
        bmm_18 = torch.ops.aten.bmm.default(expand_37, expand_38);  expand_37 = expand_38 = None
        view_237 = torch.ops.aten.view.default(bmm_18, [-1, 384]);  bmm_18 = None
        permute_default_15 = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        permute_default_16 = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
        permute_default_17 = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_57 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        permute_129 = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_68 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_244 = torch.ops.aten.view.default(view_237, [32, -1, 6, 64]);  view_237 = None
        cat_6 = torch.ops.aten.cat.default([clone_68, view_244], 2);  clone_68 = view_244 = None
        view_245 = torch.ops.aten.view.default(cat_6, [32, 512, 768]);  cat_6 = None
        view_246 = torch.ops.aten.view.default(view_245, [16384, 768]);  view_245 = None
        permute_130 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg159_1, view_246, permute_130);  arg159_1 = view_246 = permute_130 = None
        view_247 = torch.ops.aten.view.default(addmm_46, [32, 512, 768]);  addmm_46 = None
        add_81 = torch.ops.aten.add.Tensor(view_247, add_75);  view_247 = add_75 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_81, getitem_27);  add_81 = getitem_27 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg160_1);  mul_52 = arg160_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_53, arg161_1);  mul_53 = arg161_1 = None
        view_248 = torch.ops.aten.view.default(add_83, [16384, 768])
        permute_131 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg163_1, view_248, permute_131);  arg163_1 = view_248 = permute_131 = None
        view_249 = torch.ops.aten.view.default(addmm_47, [32, 512, 3072]);  addmm_47 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_249, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
        erf_6 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_84 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_84);  mul_54 = add_84 = None
        view_250 = torch.ops.aten.view.default(mul_56, [16384, 3072]);  mul_56 = None
        permute_132 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg165_1, view_250, permute_132);  arg165_1 = view_250 = permute_132 = None
        view_251 = torch.ops.aten.view.default(addmm_48, [32, 512, 768]);  addmm_48 = None
        add_85 = torch.ops.aten.add.Tensor(view_251, add_83);  view_251 = add_83 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_85, getitem_29);  add_85 = getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = rsqrt_14 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg166_1);  mul_57 = arg166_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_58, arg167_1);  mul_58 = arg167_1 = None
        view_252 = torch.ops.aten.view.default(add_87, [16384, 768])
        permute_133 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg169_1, view_252, permute_133);  arg169_1 = view_252 = permute_133 = None
        view_253 = torch.ops.aten.view.default(addmm_49, [32, 512, 384]);  addmm_49 = None
        view_254 = torch.ops.aten.view.default(add_87, [16384, 768])
        permute_134 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg171_1, view_254, permute_134);  arg171_1 = view_254 = permute_134 = None
        view_255 = torch.ops.aten.view.default(addmm_50, [32, 512, 384]);  addmm_50 = None
        view_256 = torch.ops.aten.view.default(add_87, [16384, 768])
        permute_135 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg173_1, view_256, permute_135);  arg173_1 = view_256 = permute_135 = None
        view_257 = torch.ops.aten.view.default(addmm_51, [32, 512, 384]);  addmm_51 = None
        permute_136 = torch.ops.aten.permute.default(add_87, [0, 2, 1])
        convolution_14 = torch.ops.aten.convolution.default(permute_136, arg175_1, None, [1], [4], [1], False, [0], 768);  permute_136 = arg175_1 = None
        convolution_15 = torch.ops.aten.convolution.default(convolution_14, arg176_1, None, [1], [0], [1], False, [0], 1);  convolution_14 = arg176_1 = None
        add_88 = torch.ops.aten.add.Tensor(convolution_15, arg174_1);  convolution_15 = arg174_1 = None
        view_258 = torch.ops.aten.view.default(view_253, [32, 512, 6, 64])
        view_259 = torch.ops.aten.view.default(view_255, [32, 512, 6, 64]);  view_255 = None
        view_260 = torch.ops.aten.view.default(view_257, [32, 512, 6, 64]);  view_257 = None
        permute_141 = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
        mul_59 = torch.ops.aten.mul.Tensor(permute_141, view_253);  permute_141 = view_253 = None
        permute_142 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        clone_71 = torch.ops.aten.clone.default(mul_59, memory_format = torch.contiguous_format);  mul_59 = None
        view_261 = torch.ops.aten.view.default(clone_71, [16384, 384]);  clone_71 = None
        mm_7 = torch.ops.aten.mm.default(view_261, permute_142);  view_261 = permute_142 = None
        view_262 = torch.ops.aten.view.default(mm_7, [32, 512, 54]);  mm_7 = None
        add_89 = torch.ops.aten.add.Tensor(view_262, arg178_1);  view_262 = arg178_1 = None
        view_263 = torch.ops.aten.view.default(add_89, [-1, 9, 1]);  add_89 = None
        amax_14 = torch.ops.aten.amax.default(view_263, [1], True)
        sub_30 = torch.ops.aten.sub.Tensor(view_263, amax_14);  view_263 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        view_264 = torch.ops.aten.view.default(add_87, [16384, 768])
        permute_143 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg180_1, view_264, permute_143);  arg180_1 = view_264 = permute_143 = None
        view_265 = torch.ops.aten.view.default(addmm_52, [32, 512, 384]);  addmm_52 = None
        view_266 = torch.ops.aten.view.default(view_265, [32, -1, 384]);  view_265 = None
        permute_144 = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        clone_72 = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(clone_72, -1);  clone_72 = None
        iota_28 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(iota_28, 0);  iota_28 = None
        iota_29 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        add_90 = torch.ops.aten.add.Tensor(unsqueeze_52, unsqueeze_53);  unsqueeze_52 = unsqueeze_53 = None
        iota_30 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(iota_30, 0);  iota_30 = None
        iota_31 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
        add_91 = torch.ops.aten.add.Tensor(unsqueeze_54, unsqueeze_55);  unsqueeze_54 = unsqueeze_55 = None
        constant_pad_nd_7 = torch.ops.aten.constant_pad_nd.default(unsqueeze_51, [0, 0, 4, 4], 0.0);  unsqueeze_51 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(add_90, -1);  add_90 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        index_7 = torch.ops.aten.index.Tensor(constant_pad_nd_7, [None, None, unsqueeze_57, add_91]);  constant_pad_nd_7 = unsqueeze_57 = add_91 = None
        permute_145 = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
        view_267 = torch.ops.aten.view.default(permute_145, [32, 3456, 512]);  permute_145 = None
        permute_146 = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        view_268 = torch.ops.aten.view.default(permute_146, [32, 512, 384, 9]);  permute_146 = None
        clone_73 = torch.ops.aten.clone.default(view_268, memory_format = torch.contiguous_format);  view_268 = None
        view_269 = torch.ops.aten.view.default(clone_73, [98304, 64, 9]);  clone_73 = None
        expand_43 = torch.ops.aten.expand.default(view_269, [98304, 64, 9]);  view_269 = None
        expand_44 = torch.ops.aten.expand.default(div_21, [98304, 9, 1]);  div_21 = None
        bmm_21 = torch.ops.aten.bmm.default(expand_43, expand_44);  expand_43 = expand_44 = None
        view_273 = torch.ops.aten.view.default(bmm_21, [-1, 384]);  bmm_21 = None
        permute_default_12 = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        permute_default_13 = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        permute_default_14 = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_56 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        permute_148 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_78 = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_280 = torch.ops.aten.view.default(view_273, [32, -1, 6, 64]);  view_273 = None
        cat_7 = torch.ops.aten.cat.default([clone_78, view_280], 2);  clone_78 = view_280 = None
        view_281 = torch.ops.aten.view.default(cat_7, [32, 512, 768]);  cat_7 = None
        view_282 = torch.ops.aten.view.default(view_281, [16384, 768]);  view_281 = None
        permute_149 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg182_1, view_282, permute_149);  arg182_1 = view_282 = permute_149 = None
        view_283 = torch.ops.aten.view.default(addmm_53, [32, 512, 768]);  addmm_53 = None
        add_93 = torch.ops.aten.add.Tensor(view_283, add_87);  view_283 = add_87 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_93, getitem_31);  add_93 = getitem_31 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg183_1);  mul_60 = arg183_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_61, arg184_1);  mul_61 = arg184_1 = None
        view_284 = torch.ops.aten.view.default(add_95, [16384, 768])
        permute_150 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg186_1, view_284, permute_150);  arg186_1 = view_284 = permute_150 = None
        view_285 = torch.ops.aten.view.default(addmm_54, [32, 512, 3072]);  addmm_54 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_285, 0.5)
        mul_63 = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
        erf_7 = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_96 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_62, add_96);  mul_62 = add_96 = None
        view_286 = torch.ops.aten.view.default(mul_64, [16384, 3072]);  mul_64 = None
        permute_151 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg188_1, view_286, permute_151);  arg188_1 = view_286 = permute_151 = None
        view_287 = torch.ops.aten.view.default(addmm_55, [32, 512, 768]);  addmm_55 = None
        add_97 = torch.ops.aten.add.Tensor(view_287, add_95);  view_287 = add_95 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_97, getitem_33);  add_97 = getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_16);  sub_33 = rsqrt_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg189_1);  mul_65 = arg189_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_66, arg190_1);  mul_66 = arg190_1 = None
        view_288 = torch.ops.aten.view.default(add_99, [16384, 768])
        permute_152 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg192_1, view_288, permute_152);  arg192_1 = view_288 = permute_152 = None
        view_289 = torch.ops.aten.view.default(addmm_56, [32, 512, 384]);  addmm_56 = None
        view_290 = torch.ops.aten.view.default(add_99, [16384, 768])
        permute_153 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg194_1, view_290, permute_153);  arg194_1 = view_290 = permute_153 = None
        view_291 = torch.ops.aten.view.default(addmm_57, [32, 512, 384]);  addmm_57 = None
        view_292 = torch.ops.aten.view.default(add_99, [16384, 768])
        permute_154 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg196_1, view_292, permute_154);  arg196_1 = view_292 = permute_154 = None
        view_293 = torch.ops.aten.view.default(addmm_58, [32, 512, 384]);  addmm_58 = None
        permute_155 = torch.ops.aten.permute.default(add_99, [0, 2, 1])
        convolution_16 = torch.ops.aten.convolution.default(permute_155, arg198_1, None, [1], [4], [1], False, [0], 768);  permute_155 = arg198_1 = None
        convolution_17 = torch.ops.aten.convolution.default(convolution_16, arg199_1, None, [1], [0], [1], False, [0], 1);  convolution_16 = arg199_1 = None
        add_100 = torch.ops.aten.add.Tensor(convolution_17, arg197_1);  convolution_17 = arg197_1 = None
        view_294 = torch.ops.aten.view.default(view_289, [32, 512, 6, 64])
        view_295 = torch.ops.aten.view.default(view_291, [32, 512, 6, 64]);  view_291 = None
        view_296 = torch.ops.aten.view.default(view_293, [32, 512, 6, 64]);  view_293 = None
        permute_160 = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
        mul_67 = torch.ops.aten.mul.Tensor(permute_160, view_289);  permute_160 = view_289 = None
        permute_161 = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        clone_81 = torch.ops.aten.clone.default(mul_67, memory_format = torch.contiguous_format);  mul_67 = None
        view_297 = torch.ops.aten.view.default(clone_81, [16384, 384]);  clone_81 = None
        mm_8 = torch.ops.aten.mm.default(view_297, permute_161);  view_297 = permute_161 = None
        view_298 = torch.ops.aten.view.default(mm_8, [32, 512, 54]);  mm_8 = None
        add_101 = torch.ops.aten.add.Tensor(view_298, arg201_1);  view_298 = arg201_1 = None
        view_299 = torch.ops.aten.view.default(add_101, [-1, 9, 1]);  add_101 = None
        amax_16 = torch.ops.aten.amax.default(view_299, [1], True)
        sub_34 = torch.ops.aten.sub.Tensor(view_299, amax_16);  view_299 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        view_300 = torch.ops.aten.view.default(add_99, [16384, 768])
        permute_162 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg203_1, view_300, permute_162);  arg203_1 = view_300 = permute_162 = None
        view_301 = torch.ops.aten.view.default(addmm_59, [32, 512, 384]);  addmm_59 = None
        view_302 = torch.ops.aten.view.default(view_301, [32, -1, 384]);  view_301 = None
        permute_163 = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
        clone_82 = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(clone_82, -1);  clone_82 = None
        iota_32 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
        iota_33 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        add_102 = torch.ops.aten.add.Tensor(unsqueeze_59, unsqueeze_60);  unsqueeze_59 = unsqueeze_60 = None
        iota_34 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = None
        iota_35 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        add_103 = torch.ops.aten.add.Tensor(unsqueeze_61, unsqueeze_62);  unsqueeze_61 = unsqueeze_62 = None
        constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(unsqueeze_58, [0, 0, 4, 4], 0.0);  unsqueeze_58 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(add_102, -1);  add_102 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
        index_8 = torch.ops.aten.index.Tensor(constant_pad_nd_8, [None, None, unsqueeze_64, add_103]);  constant_pad_nd_8 = unsqueeze_64 = add_103 = None
        permute_164 = torch.ops.aten.permute.default(index_8, [0, 1, 2, 4, 3, 5]);  index_8 = None
        view_303 = torch.ops.aten.view.default(permute_164, [32, 3456, 512]);  permute_164 = None
        permute_165 = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
        view_304 = torch.ops.aten.view.default(permute_165, [32, 512, 384, 9]);  permute_165 = None
        clone_83 = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
        view_305 = torch.ops.aten.view.default(clone_83, [98304, 64, 9]);  clone_83 = None
        expand_49 = torch.ops.aten.expand.default(view_305, [98304, 64, 9]);  view_305 = None
        expand_50 = torch.ops.aten.expand.default(div_24, [98304, 9, 1]);  div_24 = None
        bmm_24 = torch.ops.aten.bmm.default(expand_49, expand_50);  expand_49 = expand_50 = None
        view_309 = torch.ops.aten.view.default(bmm_24, [-1, 384]);  bmm_24 = None
        permute_default_9 = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        permute_default_10 = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
        permute_default_11 = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_55 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        permute_167 = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        clone_88 = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_316 = torch.ops.aten.view.default(view_309, [32, -1, 6, 64]);  view_309 = None
        cat_8 = torch.ops.aten.cat.default([clone_88, view_316], 2);  clone_88 = view_316 = None
        view_317 = torch.ops.aten.view.default(cat_8, [32, 512, 768]);  cat_8 = None
        view_318 = torch.ops.aten.view.default(view_317, [16384, 768]);  view_317 = None
        permute_168 = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg205_1, view_318, permute_168);  arg205_1 = view_318 = permute_168 = None
        view_319 = torch.ops.aten.view.default(addmm_60, [32, 512, 768]);  addmm_60 = None
        add_105 = torch.ops.aten.add.Tensor(view_319, add_99);  view_319 = add_99 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_105, getitem_35);  add_105 = getitem_35 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = rsqrt_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg206_1);  mul_68 = arg206_1 = None
        add_107 = torch.ops.aten.add.Tensor(mul_69, arg207_1);  mul_69 = arg207_1 = None
        view_320 = torch.ops.aten.view.default(add_107, [16384, 768])
        permute_169 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg209_1, view_320, permute_169);  arg209_1 = view_320 = permute_169 = None
        view_321 = torch.ops.aten.view.default(addmm_61, [32, 512, 3072]);  addmm_61 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_321, 0.5)
        mul_71 = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476);  view_321 = None
        erf_8 = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_108 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_70, add_108);  mul_70 = add_108 = None
        view_322 = torch.ops.aten.view.default(mul_72, [16384, 3072]);  mul_72 = None
        permute_170 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg211_1, view_322, permute_170);  arg211_1 = view_322 = permute_170 = None
        view_323 = torch.ops.aten.view.default(addmm_62, [32, 512, 768]);  addmm_62 = None
        add_109 = torch.ops.aten.add.Tensor(view_323, add_107);  view_323 = add_107 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_109, getitem_37);  add_109 = getitem_37 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_18);  sub_37 = rsqrt_18 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg212_1);  mul_73 = arg212_1 = None
        add_111 = torch.ops.aten.add.Tensor(mul_74, arg213_1);  mul_74 = arg213_1 = None
        view_324 = torch.ops.aten.view.default(add_111, [16384, 768])
        permute_171 = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg215_1, view_324, permute_171);  arg215_1 = view_324 = permute_171 = None
        view_325 = torch.ops.aten.view.default(addmm_63, [32, 512, 384]);  addmm_63 = None
        view_326 = torch.ops.aten.view.default(add_111, [16384, 768])
        permute_172 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg217_1, view_326, permute_172);  arg217_1 = view_326 = permute_172 = None
        view_327 = torch.ops.aten.view.default(addmm_64, [32, 512, 384]);  addmm_64 = None
        view_328 = torch.ops.aten.view.default(add_111, [16384, 768])
        permute_173 = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg219_1, view_328, permute_173);  arg219_1 = view_328 = permute_173 = None
        view_329 = torch.ops.aten.view.default(addmm_65, [32, 512, 384]);  addmm_65 = None
        permute_174 = torch.ops.aten.permute.default(add_111, [0, 2, 1])
        convolution_18 = torch.ops.aten.convolution.default(permute_174, arg221_1, None, [1], [4], [1], False, [0], 768);  permute_174 = arg221_1 = None
        convolution_19 = torch.ops.aten.convolution.default(convolution_18, arg222_1, None, [1], [0], [1], False, [0], 1);  convolution_18 = arg222_1 = None
        add_112 = torch.ops.aten.add.Tensor(convolution_19, arg220_1);  convolution_19 = arg220_1 = None
        view_330 = torch.ops.aten.view.default(view_325, [32, 512, 6, 64])
        view_331 = torch.ops.aten.view.default(view_327, [32, 512, 6, 64]);  view_327 = None
        view_332 = torch.ops.aten.view.default(view_329, [32, 512, 6, 64]);  view_329 = None
        permute_179 = torch.ops.aten.permute.default(add_112, [0, 2, 1]);  add_112 = None
        mul_75 = torch.ops.aten.mul.Tensor(permute_179, view_325);  permute_179 = view_325 = None
        permute_180 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        clone_91 = torch.ops.aten.clone.default(mul_75, memory_format = torch.contiguous_format);  mul_75 = None
        view_333 = torch.ops.aten.view.default(clone_91, [16384, 384]);  clone_91 = None
        mm_9 = torch.ops.aten.mm.default(view_333, permute_180);  view_333 = permute_180 = None
        view_334 = torch.ops.aten.view.default(mm_9, [32, 512, 54]);  mm_9 = None
        add_113 = torch.ops.aten.add.Tensor(view_334, arg224_1);  view_334 = arg224_1 = None
        view_335 = torch.ops.aten.view.default(add_113, [-1, 9, 1]);  add_113 = None
        amax_18 = torch.ops.aten.amax.default(view_335, [1], True)
        sub_38 = torch.ops.aten.sub.Tensor(view_335, amax_18);  view_335 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        view_336 = torch.ops.aten.view.default(add_111, [16384, 768])
        permute_181 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg226_1, view_336, permute_181);  arg226_1 = view_336 = permute_181 = None
        view_337 = torch.ops.aten.view.default(addmm_66, [32, 512, 384]);  addmm_66 = None
        view_338 = torch.ops.aten.view.default(view_337, [32, -1, 384]);  view_337 = None
        permute_182 = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        clone_92 = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(clone_92, -1);  clone_92 = None
        iota_36 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
        iota_37 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        add_114 = torch.ops.aten.add.Tensor(unsqueeze_66, unsqueeze_67);  unsqueeze_66 = unsqueeze_67 = None
        iota_38 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = None
        iota_39 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        add_115 = torch.ops.aten.add.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
        constant_pad_nd_9 = torch.ops.aten.constant_pad_nd.default(unsqueeze_65, [0, 0, 4, 4], 0.0);  unsqueeze_65 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(add_114, -1);  add_114 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        index_9 = torch.ops.aten.index.Tensor(constant_pad_nd_9, [None, None, unsqueeze_71, add_115]);  constant_pad_nd_9 = unsqueeze_71 = add_115 = None
        permute_183 = torch.ops.aten.permute.default(index_9, [0, 1, 2, 4, 3, 5]);  index_9 = None
        view_339 = torch.ops.aten.view.default(permute_183, [32, 3456, 512]);  permute_183 = None
        permute_184 = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        view_340 = torch.ops.aten.view.default(permute_184, [32, 512, 384, 9]);  permute_184 = None
        clone_93 = torch.ops.aten.clone.default(view_340, memory_format = torch.contiguous_format);  view_340 = None
        view_341 = torch.ops.aten.view.default(clone_93, [98304, 64, 9]);  clone_93 = None
        expand_55 = torch.ops.aten.expand.default(view_341, [98304, 64, 9]);  view_341 = None
        expand_56 = torch.ops.aten.expand.default(div_27, [98304, 9, 1]);  div_27 = None
        bmm_27 = torch.ops.aten.bmm.default(expand_55, expand_56);  expand_55 = expand_56 = None
        view_345 = torch.ops.aten.view.default(bmm_27, [-1, 384]);  bmm_27 = None
        permute_default_6 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        permute_default_7 = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
        permute_default_8 = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_54 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        permute_186 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_98 = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_352 = torch.ops.aten.view.default(view_345, [32, -1, 6, 64]);  view_345 = None
        cat_9 = torch.ops.aten.cat.default([clone_98, view_352], 2);  clone_98 = view_352 = None
        view_353 = torch.ops.aten.view.default(cat_9, [32, 512, 768]);  cat_9 = None
        view_354 = torch.ops.aten.view.default(view_353, [16384, 768]);  view_353 = None
        permute_187 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg228_1, view_354, permute_187);  arg228_1 = view_354 = permute_187 = None
        view_355 = torch.ops.aten.view.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        add_117 = torch.ops.aten.add.Tensor(view_355, add_111);  view_355 = add_111 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_117, getitem_39);  add_117 = getitem_39 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = rsqrt_19 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, arg229_1);  mul_76 = arg229_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_77, arg230_1);  mul_77 = arg230_1 = None
        view_356 = torch.ops.aten.view.default(add_119, [16384, 768])
        permute_188 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg232_1, view_356, permute_188);  arg232_1 = view_356 = permute_188 = None
        view_357 = torch.ops.aten.view.default(addmm_68, [32, 512, 3072]);  addmm_68 = None
        mul_78 = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_79 = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_9 = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_120 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_78, add_120);  mul_78 = add_120 = None
        view_358 = torch.ops.aten.view.default(mul_80, [16384, 3072]);  mul_80 = None
        permute_189 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg234_1, view_358, permute_189);  arg234_1 = view_358 = permute_189 = None
        view_359 = torch.ops.aten.view.default(addmm_69, [32, 512, 768]);  addmm_69 = None
        add_121 = torch.ops.aten.add.Tensor(view_359, add_119);  view_359 = add_119 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_121, getitem_41);  add_121 = getitem_41 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_20);  sub_41 = rsqrt_20 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, arg235_1);  mul_81 = arg235_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_82, arg236_1);  mul_82 = arg236_1 = None
        view_360 = torch.ops.aten.view.default(add_123, [16384, 768])
        permute_190 = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg238_1, view_360, permute_190);  arg238_1 = view_360 = permute_190 = None
        view_361 = torch.ops.aten.view.default(addmm_70, [32, 512, 384]);  addmm_70 = None
        view_362 = torch.ops.aten.view.default(add_123, [16384, 768])
        permute_191 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg240_1, view_362, permute_191);  arg240_1 = view_362 = permute_191 = None
        view_363 = torch.ops.aten.view.default(addmm_71, [32, 512, 384]);  addmm_71 = None
        view_364 = torch.ops.aten.view.default(add_123, [16384, 768])
        permute_192 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg242_1, view_364, permute_192);  arg242_1 = view_364 = permute_192 = None
        view_365 = torch.ops.aten.view.default(addmm_72, [32, 512, 384]);  addmm_72 = None
        permute_193 = torch.ops.aten.permute.default(add_123, [0, 2, 1])
        convolution_20 = torch.ops.aten.convolution.default(permute_193, arg244_1, None, [1], [4], [1], False, [0], 768);  permute_193 = arg244_1 = None
        convolution_21 = torch.ops.aten.convolution.default(convolution_20, arg245_1, None, [1], [0], [1], False, [0], 1);  convolution_20 = arg245_1 = None
        add_124 = torch.ops.aten.add.Tensor(convolution_21, arg243_1);  convolution_21 = arg243_1 = None
        view_366 = torch.ops.aten.view.default(view_361, [32, 512, 6, 64])
        view_367 = torch.ops.aten.view.default(view_363, [32, 512, 6, 64]);  view_363 = None
        view_368 = torch.ops.aten.view.default(view_365, [32, 512, 6, 64]);  view_365 = None
        permute_198 = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
        mul_83 = torch.ops.aten.mul.Tensor(permute_198, view_361);  permute_198 = view_361 = None
        permute_199 = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        clone_101 = torch.ops.aten.clone.default(mul_83, memory_format = torch.contiguous_format);  mul_83 = None
        view_369 = torch.ops.aten.view.default(clone_101, [16384, 384]);  clone_101 = None
        mm_10 = torch.ops.aten.mm.default(view_369, permute_199);  view_369 = permute_199 = None
        view_370 = torch.ops.aten.view.default(mm_10, [32, 512, 54]);  mm_10 = None
        add_125 = torch.ops.aten.add.Tensor(view_370, arg247_1);  view_370 = arg247_1 = None
        view_371 = torch.ops.aten.view.default(add_125, [-1, 9, 1]);  add_125 = None
        amax_20 = torch.ops.aten.amax.default(view_371, [1], True)
        sub_42 = torch.ops.aten.sub.Tensor(view_371, amax_20);  view_371 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
        div_30 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        view_372 = torch.ops.aten.view.default(add_123, [16384, 768])
        permute_200 = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg249_1, view_372, permute_200);  arg249_1 = view_372 = permute_200 = None
        view_373 = torch.ops.aten.view.default(addmm_73, [32, 512, 384]);  addmm_73 = None
        view_374 = torch.ops.aten.view.default(view_373, [32, -1, 384]);  view_373 = None
        permute_201 = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
        clone_102 = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(clone_102, -1);  clone_102 = None
        iota_40 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
        iota_41 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        add_126 = torch.ops.aten.add.Tensor(unsqueeze_73, unsqueeze_74);  unsqueeze_73 = unsqueeze_74 = None
        iota_42 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = None
        iota_43 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        add_127 = torch.ops.aten.add.Tensor(unsqueeze_75, unsqueeze_76);  unsqueeze_75 = unsqueeze_76 = None
        constant_pad_nd_10 = torch.ops.aten.constant_pad_nd.default(unsqueeze_72, [0, 0, 4, 4], 0.0);  unsqueeze_72 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(add_126, -1);  add_126 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
        index_10 = torch.ops.aten.index.Tensor(constant_pad_nd_10, [None, None, unsqueeze_78, add_127]);  constant_pad_nd_10 = unsqueeze_78 = add_127 = None
        permute_202 = torch.ops.aten.permute.default(index_10, [0, 1, 2, 4, 3, 5]);  index_10 = None
        view_375 = torch.ops.aten.view.default(permute_202, [32, 3456, 512]);  permute_202 = None
        permute_203 = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        view_376 = torch.ops.aten.view.default(permute_203, [32, 512, 384, 9]);  permute_203 = None
        clone_103 = torch.ops.aten.clone.default(view_376, memory_format = torch.contiguous_format);  view_376 = None
        view_377 = torch.ops.aten.view.default(clone_103, [98304, 64, 9]);  clone_103 = None
        expand_61 = torch.ops.aten.expand.default(view_377, [98304, 64, 9]);  view_377 = None
        expand_62 = torch.ops.aten.expand.default(div_30, [98304, 9, 1]);  div_30 = None
        bmm_30 = torch.ops.aten.bmm.default(expand_61, expand_62);  expand_61 = expand_62 = None
        view_381 = torch.ops.aten.view.default(bmm_30, [-1, 384]);  bmm_30 = None
        permute_default_3 = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        permute_default_4 = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
        permute_default_5 = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_53 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        permute_205 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        clone_108 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_388 = torch.ops.aten.view.default(view_381, [32, -1, 6, 64]);  view_381 = None
        cat_10 = torch.ops.aten.cat.default([clone_108, view_388], 2);  clone_108 = view_388 = None
        view_389 = torch.ops.aten.view.default(cat_10, [32, 512, 768]);  cat_10 = None
        view_390 = torch.ops.aten.view.default(view_389, [16384, 768]);  view_389 = None
        permute_206 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg251_1, view_390, permute_206);  arg251_1 = view_390 = permute_206 = None
        view_391 = torch.ops.aten.view.default(addmm_74, [32, 512, 768]);  addmm_74 = None
        add_129 = torch.ops.aten.add.Tensor(view_391, add_123);  view_391 = add_123 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_129, getitem_43);  add_129 = getitem_43 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = rsqrt_21 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg252_1);  mul_84 = arg252_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_85, arg253_1);  mul_85 = arg253_1 = None
        view_392 = torch.ops.aten.view.default(add_131, [16384, 768])
        permute_207 = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg255_1, view_392, permute_207);  arg255_1 = view_392 = permute_207 = None
        view_393 = torch.ops.aten.view.default(addmm_75, [32, 512, 3072]);  addmm_75 = None
        mul_86 = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_87 = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
        erf_10 = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_132 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_86, add_132);  mul_86 = add_132 = None
        view_394 = torch.ops.aten.view.default(mul_88, [16384, 3072]);  mul_88 = None
        permute_208 = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg257_1, view_394, permute_208);  arg257_1 = view_394 = permute_208 = None
        view_395 = torch.ops.aten.view.default(addmm_76, [32, 512, 768]);  addmm_76 = None
        add_133 = torch.ops.aten.add.Tensor(view_395, add_131);  view_395 = add_131 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_133, getitem_45);  add_133 = getitem_45 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_22);  sub_45 = rsqrt_22 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, arg258_1);  mul_89 = arg258_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_90, arg259_1);  mul_90 = arg259_1 = None
        view_396 = torch.ops.aten.view.default(add_135, [16384, 768])
        permute_209 = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg261_1, view_396, permute_209);  arg261_1 = view_396 = permute_209 = None
        view_397 = torch.ops.aten.view.default(addmm_77, [32, 512, 384]);  addmm_77 = None
        view_398 = torch.ops.aten.view.default(add_135, [16384, 768])
        permute_210 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg263_1, view_398, permute_210);  arg263_1 = view_398 = permute_210 = None
        view_399 = torch.ops.aten.view.default(addmm_78, [32, 512, 384]);  addmm_78 = None
        view_400 = torch.ops.aten.view.default(add_135, [16384, 768])
        permute_211 = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg265_1, view_400, permute_211);  arg265_1 = view_400 = permute_211 = None
        view_401 = torch.ops.aten.view.default(addmm_79, [32, 512, 384]);  addmm_79 = None
        permute_212 = torch.ops.aten.permute.default(add_135, [0, 2, 1])
        convolution_22 = torch.ops.aten.convolution.default(permute_212, arg267_1, None, [1], [4], [1], False, [0], 768);  permute_212 = arg267_1 = None
        convolution_23 = torch.ops.aten.convolution.default(convolution_22, arg268_1, None, [1], [0], [1], False, [0], 1);  convolution_22 = arg268_1 = None
        add_136 = torch.ops.aten.add.Tensor(convolution_23, arg266_1);  convolution_23 = arg266_1 = None
        view_402 = torch.ops.aten.view.default(view_397, [32, 512, 6, 64])
        view_403 = torch.ops.aten.view.default(view_399, [32, 512, 6, 64]);  view_399 = None
        view_404 = torch.ops.aten.view.default(view_401, [32, 512, 6, 64]);  view_401 = None
        permute_217 = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
        mul_91 = torch.ops.aten.mul.Tensor(permute_217, view_397);  permute_217 = view_397 = None
        permute_218 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        clone_111 = torch.ops.aten.clone.default(mul_91, memory_format = torch.contiguous_format);  mul_91 = None
        view_405 = torch.ops.aten.view.default(clone_111, [16384, 384]);  clone_111 = None
        mm_11 = torch.ops.aten.mm.default(view_405, permute_218);  view_405 = permute_218 = None
        view_406 = torch.ops.aten.view.default(mm_11, [32, 512, 54]);  mm_11 = None
        add_137 = torch.ops.aten.add.Tensor(view_406, arg270_1);  view_406 = arg270_1 = None
        view_407 = torch.ops.aten.view.default(add_137, [-1, 9, 1]);  add_137 = None
        amax_22 = torch.ops.aten.amax.default(view_407, [1], True)
        sub_46 = torch.ops.aten.sub.Tensor(view_407, amax_22);  view_407 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        view_408 = torch.ops.aten.view.default(add_135, [16384, 768])
        permute_219 = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg272_1, view_408, permute_219);  arg272_1 = view_408 = permute_219 = None
        view_409 = torch.ops.aten.view.default(addmm_80, [32, 512, 384]);  addmm_80 = None
        view_410 = torch.ops.aten.view.default(view_409, [32, -1, 384]);  view_409 = None
        permute_220 = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        clone_112 = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(clone_112, -1);  clone_112 = None
        iota_44 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
        iota_45 = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        add_138 = torch.ops.aten.add.Tensor(unsqueeze_80, unsqueeze_81);  unsqueeze_80 = unsqueeze_81 = None
        iota_46 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = None
        iota_47 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        add_139 = torch.ops.aten.add.Tensor(unsqueeze_82, unsqueeze_83);  unsqueeze_82 = unsqueeze_83 = None
        constant_pad_nd_11 = torch.ops.aten.constant_pad_nd.default(unsqueeze_79, [0, 0, 4, 4], 0.0);  unsqueeze_79 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(add_138, -1);  add_138 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        index_11 = torch.ops.aten.index.Tensor(constant_pad_nd_11, [None, None, unsqueeze_85, add_139]);  constant_pad_nd_11 = unsqueeze_85 = add_139 = None
        permute_221 = torch.ops.aten.permute.default(index_11, [0, 1, 2, 4, 3, 5]);  index_11 = None
        view_411 = torch.ops.aten.view.default(permute_221, [32, 3456, 512]);  permute_221 = None
        permute_222 = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
        view_412 = torch.ops.aten.view.default(permute_222, [32, 512, 384, 9]);  permute_222 = None
        clone_113 = torch.ops.aten.clone.default(view_412, memory_format = torch.contiguous_format);  view_412 = None
        view_413 = torch.ops.aten.view.default(clone_113, [98304, 64, 9]);  clone_113 = None
        expand_67 = torch.ops.aten.expand.default(view_413, [98304, 64, 9]);  view_413 = None
        expand_68 = torch.ops.aten.expand.default(div_33, [98304, 9, 1]);  div_33 = None
        bmm_33 = torch.ops.aten.bmm.default(expand_67, expand_68);  expand_67 = expand_68 = None
        view_417 = torch.ops.aten.view.default(bmm_33, [-1, 384]);  bmm_33 = None
        permute_default = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        permute_default_1 = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        permute_default_2 = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_52 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        permute_224 = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_118 = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_424 = torch.ops.aten.view.default(view_417, [32, -1, 6, 64]);  view_417 = None
        cat_11 = torch.ops.aten.cat.default([clone_118, view_424], 2);  clone_118 = view_424 = None
        view_425 = torch.ops.aten.view.default(cat_11, [32, 512, 768]);  cat_11 = None
        view_426 = torch.ops.aten.view.default(view_425, [16384, 768]);  view_425 = None
        permute_225 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg274_1, view_426, permute_225);  arg274_1 = view_426 = permute_225 = None
        view_427 = torch.ops.aten.view.default(addmm_81, [32, 512, 768]);  addmm_81 = None
        add_141 = torch.ops.aten.add.Tensor(view_427, add_135);  view_427 = add_135 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_141, getitem_47);  add_141 = getitem_47 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = rsqrt_23 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, arg275_1);  mul_92 = arg275_1 = None
        add_143 = torch.ops.aten.add.Tensor(mul_93, arg276_1);  mul_93 = arg276_1 = None
        view_428 = torch.ops.aten.view.default(add_143, [16384, 768])
        permute_226 = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg278_1, view_428, permute_226);  arg278_1 = view_428 = permute_226 = None
        view_429 = torch.ops.aten.view.default(addmm_82, [32, 512, 3072]);  addmm_82 = None
        mul_94 = torch.ops.aten.mul.Tensor(view_429, 0.5)
        mul_95 = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476);  view_429 = None
        erf_11 = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_144 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_94, add_144);  mul_94 = add_144 = None
        view_430 = torch.ops.aten.view.default(mul_96, [16384, 3072]);  mul_96 = None
        permute_227 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg280_1, view_430, permute_227);  arg280_1 = view_430 = permute_227 = None
        view_431 = torch.ops.aten.view.default(addmm_83, [32, 512, 768]);  addmm_83 = None
        add_145 = torch.ops.aten.add.Tensor(view_431, add_143);  view_431 = add_143 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_146 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_145, getitem_49);  add_145 = getitem_49 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_24);  sub_49 = rsqrt_24 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, arg281_1);  mul_97 = arg281_1 = None
        add_147 = torch.ops.aten.add.Tensor(mul_98, arg282_1);  mul_98 = arg282_1 = None
        view_432 = torch.ops.aten.view.default(add_147, [16384, 768]);  add_147 = None
        permute_228 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg285_1, view_432, permute_228);  arg285_1 = view_432 = permute_228 = None
        view_433 = torch.ops.aten.view.default(addmm_84, [32, 512, 768]);  addmm_84 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_433, 0.5)
        mul_100 = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476);  view_433 = None
        erf_12 = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_148 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_99, add_148);  mul_99 = add_148 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_101, getitem_51);  mul_101 = getitem_51 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = rsqrt_25 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg286_1);  mul_102 = arg286_1 = None
        add_150 = torch.ops.aten.add.Tensor(mul_103, arg287_1);  mul_103 = arg287_1 = None
        view_434 = torch.ops.aten.view.default(add_150, [16384, 768]);  add_150 = None
        permute_229 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        full_default_3 = torch.ops.aten.full.default([768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_229, full_default_3], 1);  permute_229 = full_default_3 = None
        full_default_4 = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default_1 = torch.ops.aten.cat.default([arg288_1, full_default_4]);  arg288_1 = full_default_4 = None
        addmm_default = torch.ops.aten.addmm.default(cat_default_1, view_434, cat_default);  cat_default_1 = view_434 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_435 = torch.ops.aten.view.default(slice_tensor, [32, 512, 30522]);  slice_tensor = None
        view_436 = torch.ops.aten.view.default(view_435, [-1, 30522])
        view_437 = torch.ops.aten.view.default(arg289_1, [-1]);  arg289_1 = None
        amax_24 = torch.ops.aten.amax.default(view_436, [1], True)
        sub_51 = torch.ops.aten.sub.Tensor(view_436, amax_24);  view_436 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_51)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_52 = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
        ne = torch.ops.aten.ne.Scalar(view_437, -100)
        full_default_1 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, view_437, full_default_1);  ne = full_default_1 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_52, 1, unsqueeze_86);  sub_52 = unsqueeze_86 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_437, -100)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_2);  ne_1 = neg = full_default_2 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_437, -100);  view_437 = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_36 = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
        return (div_36, view_435)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (32, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 93763584, device=device(type='cuda', index=0))
    reader.tensor(buf2, (30522, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf7, (384, 768), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf9, (384, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf10, (384,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf11, (384, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf12, (384,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf13, (384, 1), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768, 1, 9), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf15, (384, 768, 1), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf16, (54, 384), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (54,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 768), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (384,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 768), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf24, (3072, 768), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf25, (3072,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 3072), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf30, (384, 768), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf31, (384,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf32, (384, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (384,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf34, (384, 768), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf35, (384,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf36, (384, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 1, 9), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384, 768, 1), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf39, (54, 384), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf40, (54,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf41, (384, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf42, (384,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (3072,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768, 3072), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf53, (384, 768), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf54, (384,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf55, (384, 768), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (384,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf57, (384, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf58, (384,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf59, (384, 1), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768, 1, 9), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf61, (384, 768, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf62, (54, 384), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf63, (54,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf64, (384, 768), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf65, (384,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 768), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf70, (3072, 768), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf71, (3072,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768, 3072), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf76, (384, 768), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf77, (384,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384, 768), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf79, (384,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf80, (384, 768), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (384,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf82, (384, 1), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768, 1, 9), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf84, (384, 768, 1), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf85, (54, 384), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf86, (54,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384, 768), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768, 768), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf93, (3072, 768), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf94, (3072,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768, 3072), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384, 768), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf100, (384,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf101, (384, 768), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf102, (384,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf103, (384, 768), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf104, (384,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (384, 1), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 1, 9), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384, 768, 1), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf108, (54, 384), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf109, (54,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384, 768), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768, 768), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf116, (3072, 768), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf117, (3072,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768, 3072), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384, 768), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf124, (384, 768), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384, 768), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384, 1), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768, 1, 9), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf130, (384, 768, 1), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf131, (54, 384), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf132, (54,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf133, (384, 768), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768, 768), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf139, (3072, 768), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf140, (3072,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768, 3072), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384, 768), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (384,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf147, (384, 768), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf148, (384,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf149, (384, 768), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf151, (384, 1), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768, 1, 9), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf153, (384, 768, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf154, (54, 384), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf155, (54,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf156, (384, 768), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf157, (384,), is_leaf=True)  # arg157_1
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
    buf168 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf168, (384, 768), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf169, (384,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf170, (384, 768), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf171, (384,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf172, (384, 768), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf173, (384,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf174, (384, 1), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768, 1, 9), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf176, (384, 768, 1), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf177, (54, 384), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf178, (54,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf179, (384, 768), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf180, (384,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768, 768), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf185, (3072, 768), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf186, (3072,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768, 3072), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf191, (384, 768), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf192, (384,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf193, (384, 768), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf194, (384,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf195, (384, 768), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf196, (384,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf197, (384, 1), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768, 1, 9), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf199, (384, 768, 1), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf200, (54, 384), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf201, (54,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf202, (384, 768), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf203, (384,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf204, (768, 768), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf205, (768,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf206, (768,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf207, (768,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf208, (3072, 768), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf209, (3072,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf210, (768, 3072), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf211, (768,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf212, (768,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf213, (768,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf214, (384, 768), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf215, (384,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf216, (384, 768), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf217, (384,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf218, (384, 768), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf219, (384,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf220, (384, 1), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf221, (768, 1, 9), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf222, (384, 768, 1), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf223, (54, 384), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf224, (54,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf225, (384, 768), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf226, (384,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf227, (768, 768), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf228, (768,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf229, (768,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf230, (768,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf231, (3072, 768), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf232, (3072,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf233, (768, 3072), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf234, (768,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf235, (768,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf236, (768,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf237, (384, 768), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf238, (384,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf239, (384, 768), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (384,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf241, (384, 768), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf242, (384,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf243, (384, 1), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf244, (768, 1, 9), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf245, (384, 768, 1), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf246, (54, 384), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf247, (54,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf248, (384, 768), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (384,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf250, (768, 768), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf251, (768,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf252, (768,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf253, (768,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf254, (3072, 768), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf255, (3072,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf256, (768, 3072), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf257, (768,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf258, (768,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf259, (768,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf260, (384, 768), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf261, (384,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf262, (384, 768), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf263, (384,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf264, (384, 768), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf265, (384,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf266, (384, 1), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf267, (768, 1, 9), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf268, (384, 768, 1), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf269, (54, 384), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 216, device=device(type='cuda', index=0))
    reader.tensor(buf270, (54,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf271, (384, 768), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf272, (384,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf273, (768, 768), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf274, (768,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf275, (768,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf276, (768,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf277, (3072, 768), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf278, (3072,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf279, (768, 3072), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf280, (768,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf281, (768,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf282, (768,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf283, (1, 512), dtype=torch.int64, is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf284, (768, 768), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf285, (768,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf286, (768,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf287, (768,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 122088, device=device(type='cuda', index=0))
    reader.tensor(buf288, (30522,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf289, (32, 512), dtype=torch.int64, is_leaf=True)  # arg289_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)