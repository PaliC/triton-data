
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1):
        full = torch.ops.aten.full.default([8, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default = torch.ops.aten.full.default([8, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default_1 = torch.ops.aten.full.default([8, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_1 = None
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, full_default);  arg3_1 = full_default = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg2_1, arg390_1);  arg2_1 = arg390_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, getitem_1);  getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        view = torch.ops.aten.view.default(add_3, [4096, 1024])
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [8, 512, 1024]);  addmm = None
        view_2 = torch.ops.aten.view.default(add_3, [4096, 1024])
        permute_1 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_2, permute_1);  arg9_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [8, 512, 1024]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [8, 512, 16, 64]);  view_3 = None
        view_5 = torch.ops.aten.view.default(add_3, [4096, 1024]);  add_3 = None
        permute_3 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_5, permute_3);  arg11_1 = view_5 = permute_3 = None
        view_6 = torch.ops.aten.view.default(addmm_2, [8, 512, 1024]);  addmm_2 = None
        view_7 = torch.ops.aten.view.default(view_6, [8, 512, 16, 64]);  view_6 = None
        view_8 = torch.ops.aten.view.default(view_1, [8, 512, 16, 64]);  view_1 = None
        permute_default_69 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        permute_default_70 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_default_71 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_69, permute_default_70, permute_default_71, None, False, scale = 0.125);  permute_default_69 = permute_default_70 = permute_default_71 = None
        getitem_123 = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None
        permute_7 = torch.ops.aten.permute.default(getitem_123, [0, 2, 1, 3]);  getitem_123 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15 = torch.ops.aten.view.default(clone_5, [8, 512, 1024]);  clone_5 = None
        view_16 = torch.ops.aten.view.default(view_15, [4096, 1024]);  view_15 = None
        permute_8 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_16, permute_8);  arg13_1 = view_16 = permute_8 = None
        view_17 = torch.ops.aten.view.default(addmm_3, [8, 512, 1024]);  addmm_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_1, view_17);  add_1 = view_17 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg14_1);  mul_3 = arg14_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_4, arg15_1);  mul_4 = arg15_1 = None
        view_18 = torch.ops.aten.view.default(add_7, [4096, 1024]);  add_7 = None
        permute_9 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  arg17_1 = view_18 = permute_9 = None
        view_19 = torch.ops.aten.view.default(addmm_4, [8, 512, 4096]);  addmm_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_6 = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        view_20 = torch.ops.aten.view.default(mul_7, [4096, 4096]);  mul_7 = None
        permute_10 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_20, permute_10);  arg19_1 = view_20 = permute_10 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [8, 512, 1024]);  addmm_5 = None
        add_9 = torch.ops.aten.add.Tensor(add_5, view_21);  add_5 = view_21 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg20_1);  mul_8 = arg20_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg21_1);  mul_9 = arg21_1 = None
        view_22 = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_11 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  arg23_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [8, 512, 1024]);  addmm_6 = None
        view_24 = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_12 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_24, permute_12);  arg25_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [8, 512, 1024]);  addmm_7 = None
        view_26 = torch.ops.aten.view.default(view_25, [8, 512, 16, 64]);  view_25 = None
        view_27 = torch.ops.aten.view.default(add_11, [4096, 1024]);  add_11 = None
        permute_14 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg27_1, view_27, permute_14);  arg27_1 = view_27 = permute_14 = None
        view_28 = torch.ops.aten.view.default(addmm_8, [8, 512, 1024]);  addmm_8 = None
        view_29 = torch.ops.aten.view.default(view_28, [8, 512, 16, 64]);  view_28 = None
        view_30 = torch.ops.aten.view.default(view_23, [8, 512, 16, 64]);  view_23 = None
        permute_default_66 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        permute_default_67 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        permute_default_68 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_66, permute_default_67, permute_default_68, None, False, scale = 0.125);  permute_default_66 = permute_default_67 = permute_default_68 = None
        getitem_122 = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None
        permute_18 = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None
        clone_12 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37 = torch.ops.aten.view.default(clone_12, [8, 512, 1024]);  clone_12 = None
        view_38 = torch.ops.aten.view.default(view_37, [4096, 1024]);  view_37 = None
        permute_19 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg29_1, view_38, permute_19);  arg29_1 = view_38 = permute_19 = None
        view_39 = torch.ops.aten.view.default(addmm_9, [8, 512, 1024]);  addmm_9 = None
        add_13 = torch.ops.aten.add.Tensor(add_9, view_39);  add_9 = view_39 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg30_1);  mul_10 = arg30_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_11, arg31_1);  mul_11 = arg31_1 = None
        view_40 = torch.ops.aten.view.default(add_15, [4096, 1024]);  add_15 = None
        permute_20 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg33_1, view_40, permute_20);  arg33_1 = view_40 = permute_20 = None
        view_41 = torch.ops.aten.view.default(addmm_10, [8, 512, 4096]);  addmm_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_13 = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1 = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_42 = torch.ops.aten.view.default(mul_14, [4096, 4096]);  mul_14 = None
        permute_21 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg35_1, view_42, permute_21);  arg35_1 = view_42 = permute_21 = None
        view_43 = torch.ops.aten.view.default(addmm_11, [8, 512, 1024]);  addmm_11 = None
        add_17 = torch.ops.aten.add.Tensor(add_13, view_43);  add_13 = view_43 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, arg36_1);  mul_15 = arg36_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_16, arg37_1);  mul_16 = arg37_1 = None
        view_44 = torch.ops.aten.view.default(add_19, [4096, 1024])
        permute_22 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg39_1, view_44, permute_22);  arg39_1 = view_44 = permute_22 = None
        view_45 = torch.ops.aten.view.default(addmm_12, [8, 512, 1024]);  addmm_12 = None
        view_46 = torch.ops.aten.view.default(add_19, [4096, 1024])
        permute_23 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg41_1, view_46, permute_23);  arg41_1 = view_46 = permute_23 = None
        view_47 = torch.ops.aten.view.default(addmm_13, [8, 512, 1024]);  addmm_13 = None
        view_48 = torch.ops.aten.view.default(view_47, [8, 512, 16, 64]);  view_47 = None
        view_49 = torch.ops.aten.view.default(add_19, [4096, 1024]);  add_19 = None
        permute_25 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg43_1, view_49, permute_25);  arg43_1 = view_49 = permute_25 = None
        view_50 = torch.ops.aten.view.default(addmm_14, [8, 512, 1024]);  addmm_14 = None
        view_51 = torch.ops.aten.view.default(view_50, [8, 512, 16, 64]);  view_50 = None
        view_52 = torch.ops.aten.view.default(view_45, [8, 512, 16, 64]);  view_45 = None
        permute_default_63 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        permute_default_64 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        permute_default_65 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_63, permute_default_64, permute_default_65, None, False, scale = 0.125);  permute_default_63 = permute_default_64 = permute_default_65 = None
        getitem_121 = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None
        permute_29 = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        clone_19 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59 = torch.ops.aten.view.default(clone_19, [8, 512, 1024]);  clone_19 = None
        view_60 = torch.ops.aten.view.default(view_59, [4096, 1024]);  view_59 = None
        permute_30 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg45_1, view_60, permute_30);  arg45_1 = view_60 = permute_30 = None
        view_61 = torch.ops.aten.view.default(addmm_15, [8, 512, 1024]);  addmm_15 = None
        add_21 = torch.ops.aten.add.Tensor(add_17, view_61);  add_17 = view_61 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_11);  getitem_11 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg46_1);  mul_17 = arg46_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_18, arg47_1);  mul_18 = arg47_1 = None
        view_62 = torch.ops.aten.view.default(add_23, [4096, 1024]);  add_23 = None
        permute_31 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg49_1, view_62, permute_31);  arg49_1 = view_62 = permute_31 = None
        view_63 = torch.ops.aten.view.default(addmm_16, [8, 512, 4096]);  addmm_16 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_20 = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2 = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
        view_64 = torch.ops.aten.view.default(mul_21, [4096, 4096]);  mul_21 = None
        permute_32 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg51_1, view_64, permute_32);  arg51_1 = view_64 = permute_32 = None
        view_65 = torch.ops.aten.view.default(addmm_17, [8, 512, 1024]);  addmm_17 = None
        add_25 = torch.ops.aten.add.Tensor(add_21, view_65);  add_21 = view_65 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  getitem_13 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg52_1);  mul_22 = arg52_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_23, arg53_1);  mul_23 = arg53_1 = None
        view_66 = torch.ops.aten.view.default(add_27, [4096, 1024])
        permute_33 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_66, permute_33);  arg55_1 = view_66 = permute_33 = None
        view_67 = torch.ops.aten.view.default(addmm_18, [8, 512, 1024]);  addmm_18 = None
        view_68 = torch.ops.aten.view.default(add_27, [4096, 1024])
        permute_34 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_68, permute_34);  arg57_1 = view_68 = permute_34 = None
        view_69 = torch.ops.aten.view.default(addmm_19, [8, 512, 1024]);  addmm_19 = None
        view_70 = torch.ops.aten.view.default(view_69, [8, 512, 16, 64]);  view_69 = None
        view_71 = torch.ops.aten.view.default(add_27, [4096, 1024]);  add_27 = None
        permute_36 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_71, permute_36);  arg59_1 = view_71 = permute_36 = None
        view_72 = torch.ops.aten.view.default(addmm_20, [8, 512, 1024]);  addmm_20 = None
        view_73 = torch.ops.aten.view.default(view_72, [8, 512, 16, 64]);  view_72 = None
        view_74 = torch.ops.aten.view.default(view_67, [8, 512, 16, 64]);  view_67 = None
        permute_default_60 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_default_61 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        permute_default_62 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_60, permute_default_61, permute_default_62, None, False, scale = 0.125);  permute_default_60 = permute_default_61 = permute_default_62 = None
        getitem_120 = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None
        permute_40 = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        clone_26 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81 = torch.ops.aten.view.default(clone_26, [8, 512, 1024]);  clone_26 = None
        view_82 = torch.ops.aten.view.default(view_81, [4096, 1024]);  view_81 = None
        permute_41 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_82, permute_41);  arg61_1 = view_82 = permute_41 = None
        view_83 = torch.ops.aten.view.default(addmm_21, [8, 512, 1024]);  addmm_21 = None
        add_29 = torch.ops.aten.add.Tensor(add_25, view_83);  add_25 = view_83 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  getitem_15 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg62_1);  mul_24 = arg62_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_25, arg63_1);  mul_25 = arg63_1 = None
        view_84 = torch.ops.aten.view.default(add_31, [4096, 1024]);  add_31 = None
        permute_42 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg65_1, view_84, permute_42);  arg65_1 = view_84 = permute_42 = None
        view_85 = torch.ops.aten.view.default(addmm_22, [8, 512, 4096]);  addmm_22 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_27 = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3 = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        view_86 = torch.ops.aten.view.default(mul_28, [4096, 4096]);  mul_28 = None
        permute_43 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg67_1, view_86, permute_43);  arg67_1 = view_86 = permute_43 = None
        view_87 = torch.ops.aten.view.default(addmm_23, [8, 512, 1024]);  addmm_23 = None
        add_33 = torch.ops.aten.add.Tensor(add_29, view_87);  add_29 = view_87 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  getitem_17 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, arg68_1);  mul_29 = arg68_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_30, arg69_1);  mul_30 = arg69_1 = None
        view_88 = torch.ops.aten.view.default(add_35, [4096, 1024])
        permute_44 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg71_1, view_88, permute_44);  arg71_1 = view_88 = permute_44 = None
        view_89 = torch.ops.aten.view.default(addmm_24, [8, 512, 1024]);  addmm_24 = None
        view_90 = torch.ops.aten.view.default(add_35, [4096, 1024])
        permute_45 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg73_1, view_90, permute_45);  arg73_1 = view_90 = permute_45 = None
        view_91 = torch.ops.aten.view.default(addmm_25, [8, 512, 1024]);  addmm_25 = None
        view_92 = torch.ops.aten.view.default(view_91, [8, 512, 16, 64]);  view_91 = None
        view_93 = torch.ops.aten.view.default(add_35, [4096, 1024]);  add_35 = None
        permute_47 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg75_1, view_93, permute_47);  arg75_1 = view_93 = permute_47 = None
        view_94 = torch.ops.aten.view.default(addmm_26, [8, 512, 1024]);  addmm_26 = None
        view_95 = torch.ops.aten.view.default(view_94, [8, 512, 16, 64]);  view_94 = None
        view_96 = torch.ops.aten.view.default(view_89, [8, 512, 16, 64]);  view_89 = None
        permute_default_57 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        permute_default_58 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        permute_default_59 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_57, permute_default_58, permute_default_59, None, False, scale = 0.125);  permute_default_57 = permute_default_58 = permute_default_59 = None
        getitem_119 = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None
        permute_51 = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None
        clone_33 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103 = torch.ops.aten.view.default(clone_33, [8, 512, 1024]);  clone_33 = None
        view_104 = torch.ops.aten.view.default(view_103, [4096, 1024]);  view_103 = None
        permute_52 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg77_1, view_104, permute_52);  arg77_1 = view_104 = permute_52 = None
        view_105 = torch.ops.aten.view.default(addmm_27, [8, 512, 1024]);  addmm_27 = None
        add_37 = torch.ops.aten.add.Tensor(add_33, view_105);  add_33 = view_105 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_37, getitem_19);  getitem_19 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, arg78_1);  mul_31 = arg78_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_32, arg79_1);  mul_32 = arg79_1 = None
        view_106 = torch.ops.aten.view.default(add_39, [4096, 1024]);  add_39 = None
        permute_53 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg81_1, view_106, permute_53);  arg81_1 = view_106 = permute_53 = None
        view_107 = torch.ops.aten.view.default(addmm_28, [8, 512, 4096]);  addmm_28 = None
        mul_33 = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_34 = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4 = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
        view_108 = torch.ops.aten.view.default(mul_35, [4096, 4096]);  mul_35 = None
        permute_54 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg83_1, view_108, permute_54);  arg83_1 = view_108 = permute_54 = None
        view_109 = torch.ops.aten.view.default(addmm_29, [8, 512, 1024]);  addmm_29 = None
        add_41 = torch.ops.aten.add.Tensor(add_37, view_109);  add_37 = view_109 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  getitem_21 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg84_1);  mul_36 = arg84_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_37, arg85_1);  mul_37 = arg85_1 = None
        view_110 = torch.ops.aten.view.default(add_43, [4096, 1024])
        permute_55 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg87_1, view_110, permute_55);  arg87_1 = view_110 = permute_55 = None
        view_111 = torch.ops.aten.view.default(addmm_30, [8, 512, 1024]);  addmm_30 = None
        view_112 = torch.ops.aten.view.default(add_43, [4096, 1024])
        permute_56 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg89_1, view_112, permute_56);  arg89_1 = view_112 = permute_56 = None
        view_113 = torch.ops.aten.view.default(addmm_31, [8, 512, 1024]);  addmm_31 = None
        view_114 = torch.ops.aten.view.default(view_113, [8, 512, 16, 64]);  view_113 = None
        view_115 = torch.ops.aten.view.default(add_43, [4096, 1024]);  add_43 = None
        permute_58 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg91_1, view_115, permute_58);  arg91_1 = view_115 = permute_58 = None
        view_116 = torch.ops.aten.view.default(addmm_32, [8, 512, 1024]);  addmm_32 = None
        view_117 = torch.ops.aten.view.default(view_116, [8, 512, 16, 64]);  view_116 = None
        view_118 = torch.ops.aten.view.default(view_111, [8, 512, 16, 64]);  view_111 = None
        permute_default_54 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        permute_default_55 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_default_56 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_54, permute_default_55, permute_default_56, None, False, scale = 0.125);  permute_default_54 = permute_default_55 = permute_default_56 = None
        getitem_118 = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None
        permute_62 = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None
        clone_40 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125 = torch.ops.aten.view.default(clone_40, [8, 512, 1024]);  clone_40 = None
        view_126 = torch.ops.aten.view.default(view_125, [4096, 1024]);  view_125 = None
        permute_63 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg93_1, view_126, permute_63);  arg93_1 = view_126 = permute_63 = None
        view_127 = torch.ops.aten.view.default(addmm_33, [8, 512, 1024]);  addmm_33 = None
        add_45 = torch.ops.aten.add.Tensor(add_41, view_127);  add_41 = view_127 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_45, getitem_23);  getitem_23 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg94_1);  mul_38 = arg94_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_39, arg95_1);  mul_39 = arg95_1 = None
        view_128 = torch.ops.aten.view.default(add_47, [4096, 1024]);  add_47 = None
        permute_64 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg97_1, view_128, permute_64);  arg97_1 = view_128 = permute_64 = None
        view_129 = torch.ops.aten.view.default(addmm_34, [8, 512, 4096]);  addmm_34 = None
        mul_40 = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
        view_130 = torch.ops.aten.view.default(mul_42, [4096, 4096]);  mul_42 = None
        permute_65 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg99_1, view_130, permute_65);  arg99_1 = view_130 = permute_65 = None
        view_131 = torch.ops.aten.view.default(addmm_35, [8, 512, 1024]);  addmm_35 = None
        add_49 = torch.ops.aten.add.Tensor(add_45, view_131);  add_45 = view_131 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  getitem_25 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, arg100_1);  mul_43 = arg100_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_44, arg101_1);  mul_44 = arg101_1 = None
        view_132 = torch.ops.aten.view.default(add_51, [4096, 1024])
        permute_66 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg103_1, view_132, permute_66);  arg103_1 = view_132 = permute_66 = None
        view_133 = torch.ops.aten.view.default(addmm_36, [8, 512, 1024]);  addmm_36 = None
        view_134 = torch.ops.aten.view.default(add_51, [4096, 1024])
        permute_67 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg105_1, view_134, permute_67);  arg105_1 = view_134 = permute_67 = None
        view_135 = torch.ops.aten.view.default(addmm_37, [8, 512, 1024]);  addmm_37 = None
        view_136 = torch.ops.aten.view.default(view_135, [8, 512, 16, 64]);  view_135 = None
        view_137 = torch.ops.aten.view.default(add_51, [4096, 1024]);  add_51 = None
        permute_69 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg107_1, view_137, permute_69);  arg107_1 = view_137 = permute_69 = None
        view_138 = torch.ops.aten.view.default(addmm_38, [8, 512, 1024]);  addmm_38 = None
        view_139 = torch.ops.aten.view.default(view_138, [8, 512, 16, 64]);  view_138 = None
        view_140 = torch.ops.aten.view.default(view_133, [8, 512, 16, 64]);  view_133 = None
        permute_default_51 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        permute_default_52 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_default_53 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_51, permute_default_52, permute_default_53, None, False, scale = 0.125);  permute_default_51 = permute_default_52 = permute_default_53 = None
        getitem_117 = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None
        permute_73 = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        clone_47 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147 = torch.ops.aten.view.default(clone_47, [8, 512, 1024]);  clone_47 = None
        view_148 = torch.ops.aten.view.default(view_147, [4096, 1024]);  view_147 = None
        permute_74 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg109_1, view_148, permute_74);  arg109_1 = view_148 = permute_74 = None
        view_149 = torch.ops.aten.view.default(addmm_39, [8, 512, 1024]);  addmm_39 = None
        add_53 = torch.ops.aten.add.Tensor(add_49, view_149);  add_49 = view_149 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_53, getitem_27);  getitem_27 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, arg110_1);  mul_45 = arg110_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_46, arg111_1);  mul_46 = arg111_1 = None
        view_150 = torch.ops.aten.view.default(add_55, [4096, 1024]);  add_55 = None
        permute_75 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg113_1, view_150, permute_75);  arg113_1 = view_150 = permute_75 = None
        view_151 = torch.ops.aten.view.default(addmm_40, [8, 512, 4096]);  addmm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_48 = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6 = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
        view_152 = torch.ops.aten.view.default(mul_49, [4096, 4096]);  mul_49 = None
        permute_76 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg115_1, view_152, permute_76);  arg115_1 = view_152 = permute_76 = None
        view_153 = torch.ops.aten.view.default(addmm_41, [8, 512, 1024]);  addmm_41 = None
        add_57 = torch.ops.aten.add.Tensor(add_53, view_153);  add_53 = view_153 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  getitem_29 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg116_1);  mul_50 = arg116_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_51, arg117_1);  mul_51 = arg117_1 = None
        view_154 = torch.ops.aten.view.default(add_59, [4096, 1024])
        permute_77 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg119_1, view_154, permute_77);  arg119_1 = view_154 = permute_77 = None
        view_155 = torch.ops.aten.view.default(addmm_42, [8, 512, 1024]);  addmm_42 = None
        view_156 = torch.ops.aten.view.default(add_59, [4096, 1024])
        permute_78 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg121_1, view_156, permute_78);  arg121_1 = view_156 = permute_78 = None
        view_157 = torch.ops.aten.view.default(addmm_43, [8, 512, 1024]);  addmm_43 = None
        view_158 = torch.ops.aten.view.default(view_157, [8, 512, 16, 64]);  view_157 = None
        view_159 = torch.ops.aten.view.default(add_59, [4096, 1024]);  add_59 = None
        permute_80 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg123_1, view_159, permute_80);  arg123_1 = view_159 = permute_80 = None
        view_160 = torch.ops.aten.view.default(addmm_44, [8, 512, 1024]);  addmm_44 = None
        view_161 = torch.ops.aten.view.default(view_160, [8, 512, 16, 64]);  view_160 = None
        view_162 = torch.ops.aten.view.default(view_155, [8, 512, 16, 64]);  view_155 = None
        permute_default_48 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_default_49 = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        permute_default_50 = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_48, permute_default_49, permute_default_50, None, False, scale = 0.125);  permute_default_48 = permute_default_49 = permute_default_50 = None
        getitem_116 = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None
        permute_84 = torch.ops.aten.permute.default(getitem_116, [0, 2, 1, 3]);  getitem_116 = None
        clone_54 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169 = torch.ops.aten.view.default(clone_54, [8, 512, 1024]);  clone_54 = None
        view_170 = torch.ops.aten.view.default(view_169, [4096, 1024]);  view_169 = None
        permute_85 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg125_1, view_170, permute_85);  arg125_1 = view_170 = permute_85 = None
        view_171 = torch.ops.aten.view.default(addmm_45, [8, 512, 1024]);  addmm_45 = None
        add_61 = torch.ops.aten.add.Tensor(add_57, view_171);  add_57 = view_171 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_61, getitem_31);  getitem_31 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg126_1);  mul_52 = arg126_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_53, arg127_1);  mul_53 = arg127_1 = None
        view_172 = torch.ops.aten.view.default(add_63, [4096, 1024]);  add_63 = None
        permute_86 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg129_1, view_172, permute_86);  arg129_1 = view_172 = permute_86 = None
        view_173 = torch.ops.aten.view.default(addmm_46, [8, 512, 4096]);  addmm_46 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
        view_174 = torch.ops.aten.view.default(mul_56, [4096, 4096]);  mul_56 = None
        permute_87 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg131_1, view_174, permute_87);  arg131_1 = view_174 = permute_87 = None
        view_175 = torch.ops.aten.view.default(addmm_47, [8, 512, 1024]);  addmm_47 = None
        add_65 = torch.ops.aten.add.Tensor(add_61, view_175);  add_61 = view_175 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  getitem_33 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg132_1);  mul_57 = arg132_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_58, arg133_1);  mul_58 = arg133_1 = None
        view_176 = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_88 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg135_1, view_176, permute_88);  arg135_1 = view_176 = permute_88 = None
        view_177 = torch.ops.aten.view.default(addmm_48, [8, 512, 1024]);  addmm_48 = None
        view_178 = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_89 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg137_1, view_178, permute_89);  arg137_1 = view_178 = permute_89 = None
        view_179 = torch.ops.aten.view.default(addmm_49, [8, 512, 1024]);  addmm_49 = None
        view_180 = torch.ops.aten.view.default(view_179, [8, 512, 16, 64]);  view_179 = None
        view_181 = torch.ops.aten.view.default(add_67, [4096, 1024]);  add_67 = None
        permute_91 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg139_1, view_181, permute_91);  arg139_1 = view_181 = permute_91 = None
        view_182 = torch.ops.aten.view.default(addmm_50, [8, 512, 1024]);  addmm_50 = None
        view_183 = torch.ops.aten.view.default(view_182, [8, 512, 16, 64]);  view_182 = None
        view_184 = torch.ops.aten.view.default(view_177, [8, 512, 16, 64]);  view_177 = None
        permute_default_45 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        permute_default_46 = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        permute_default_47 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_45, permute_default_46, permute_default_47, None, False, scale = 0.125);  permute_default_45 = permute_default_46 = permute_default_47 = None
        getitem_115 = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
        permute_95 = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        clone_61 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191 = torch.ops.aten.view.default(clone_61, [8, 512, 1024]);  clone_61 = None
        view_192 = torch.ops.aten.view.default(view_191, [4096, 1024]);  view_191 = None
        permute_96 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg141_1, view_192, permute_96);  arg141_1 = view_192 = permute_96 = None
        view_193 = torch.ops.aten.view.default(addmm_51, [8, 512, 1024]);  addmm_51 = None
        add_69 = torch.ops.aten.add.Tensor(add_65, view_193);  add_65 = view_193 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_69, getitem_35);  getitem_35 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg142_1);  mul_59 = arg142_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_60, arg143_1);  mul_60 = arg143_1 = None
        view_194 = torch.ops.aten.view.default(add_71, [4096, 1024]);  add_71 = None
        permute_97 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg145_1, view_194, permute_97);  arg145_1 = view_194 = permute_97 = None
        view_195 = torch.ops.aten.view.default(addmm_52, [8, 512, 4096]);  addmm_52 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_62 = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8 = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
        view_196 = torch.ops.aten.view.default(mul_63, [4096, 4096]);  mul_63 = None
        permute_98 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg147_1, view_196, permute_98);  arg147_1 = view_196 = permute_98 = None
        view_197 = torch.ops.aten.view.default(addmm_53, [8, 512, 1024]);  addmm_53 = None
        add_73 = torch.ops.aten.add.Tensor(add_69, view_197);  add_69 = view_197 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_73, getitem_37);  getitem_37 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg148_1);  mul_64 = arg148_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_65, arg149_1);  mul_65 = arg149_1 = None
        view_198 = torch.ops.aten.view.default(add_75, [4096, 1024])
        permute_99 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg151_1, view_198, permute_99);  arg151_1 = view_198 = permute_99 = None
        view_199 = torch.ops.aten.view.default(addmm_54, [8, 512, 1024]);  addmm_54 = None
        view_200 = torch.ops.aten.view.default(add_75, [4096, 1024])
        permute_100 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg153_1, view_200, permute_100);  arg153_1 = view_200 = permute_100 = None
        view_201 = torch.ops.aten.view.default(addmm_55, [8, 512, 1024]);  addmm_55 = None
        view_202 = torch.ops.aten.view.default(view_201, [8, 512, 16, 64]);  view_201 = None
        view_203 = torch.ops.aten.view.default(add_75, [4096, 1024]);  add_75 = None
        permute_102 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg155_1, view_203, permute_102);  arg155_1 = view_203 = permute_102 = None
        view_204 = torch.ops.aten.view.default(addmm_56, [8, 512, 1024]);  addmm_56 = None
        view_205 = torch.ops.aten.view.default(view_204, [8, 512, 16, 64]);  view_204 = None
        view_206 = torch.ops.aten.view.default(view_199, [8, 512, 16, 64]);  view_199 = None
        permute_default_42 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        permute_default_43 = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        permute_default_44 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_42, permute_default_43, permute_default_44, None, False, scale = 0.125);  permute_default_42 = permute_default_43 = permute_default_44 = None
        getitem_114 = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
        permute_106 = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3]);  getitem_114 = None
        clone_68 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213 = torch.ops.aten.view.default(clone_68, [8, 512, 1024]);  clone_68 = None
        view_214 = torch.ops.aten.view.default(view_213, [4096, 1024]);  view_213 = None
        permute_107 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg157_1, view_214, permute_107);  arg157_1 = view_214 = permute_107 = None
        view_215 = torch.ops.aten.view.default(addmm_57, [8, 512, 1024]);  addmm_57 = None
        add_77 = torch.ops.aten.add.Tensor(add_73, view_215);  add_73 = view_215 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_77, getitem_39);  getitem_39 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg158_1);  mul_66 = arg158_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_67, arg159_1);  mul_67 = arg159_1 = None
        view_216 = torch.ops.aten.view.default(add_79, [4096, 1024]);  add_79 = None
        permute_108 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg161_1, view_216, permute_108);  arg161_1 = view_216 = permute_108 = None
        view_217 = torch.ops.aten.view.default(addmm_58, [8, 512, 4096]);  addmm_58 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_69 = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9 = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
        view_218 = torch.ops.aten.view.default(mul_70, [4096, 4096]);  mul_70 = None
        permute_109 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg163_1, view_218, permute_109);  arg163_1 = view_218 = permute_109 = None
        view_219 = torch.ops.aten.view.default(addmm_59, [8, 512, 1024]);  addmm_59 = None
        add_81 = torch.ops.aten.add.Tensor(add_77, view_219);  add_77 = view_219 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_81, getitem_41);  getitem_41 = None
        mul_71 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_71, arg164_1);  mul_71 = arg164_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_72, arg165_1);  mul_72 = arg165_1 = None
        view_220 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_110 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg167_1, view_220, permute_110);  arg167_1 = view_220 = permute_110 = None
        view_221 = torch.ops.aten.view.default(addmm_60, [8, 512, 1024]);  addmm_60 = None
        view_222 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_111 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg169_1, view_222, permute_111);  arg169_1 = view_222 = permute_111 = None
        view_223 = torch.ops.aten.view.default(addmm_61, [8, 512, 1024]);  addmm_61 = None
        view_224 = torch.ops.aten.view.default(view_223, [8, 512, 16, 64]);  view_223 = None
        view_225 = torch.ops.aten.view.default(add_83, [4096, 1024]);  add_83 = None
        permute_113 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg171_1, view_225, permute_113);  arg171_1 = view_225 = permute_113 = None
        view_226 = torch.ops.aten.view.default(addmm_62, [8, 512, 1024]);  addmm_62 = None
        view_227 = torch.ops.aten.view.default(view_226, [8, 512, 16, 64]);  view_226 = None
        view_228 = torch.ops.aten.view.default(view_221, [8, 512, 16, 64]);  view_221 = None
        permute_default_39 = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        permute_default_40 = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        permute_default_41 = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_39, permute_default_40, permute_default_41, None, False, scale = 0.125);  permute_default_39 = permute_default_40 = permute_default_41 = None
        getitem_113 = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
        permute_117 = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
        clone_75 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235 = torch.ops.aten.view.default(clone_75, [8, 512, 1024]);  clone_75 = None
        view_236 = torch.ops.aten.view.default(view_235, [4096, 1024]);  view_235 = None
        permute_118 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg173_1, view_236, permute_118);  arg173_1 = view_236 = permute_118 = None
        view_237 = torch.ops.aten.view.default(addmm_63, [8, 512, 1024]);  addmm_63 = None
        add_85 = torch.ops.aten.add.Tensor(add_81, view_237);  add_81 = view_237 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_85, getitem_43);  getitem_43 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg174_1);  mul_73 = arg174_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_74, arg175_1);  mul_74 = arg175_1 = None
        view_238 = torch.ops.aten.view.default(add_87, [4096, 1024]);  add_87 = None
        permute_119 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg177_1, view_238, permute_119);  arg177_1 = view_238 = permute_119 = None
        view_239 = torch.ops.aten.view.default(addmm_64, [8, 512, 4096]);  addmm_64 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_76 = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10 = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
        view_240 = torch.ops.aten.view.default(mul_77, [4096, 4096]);  mul_77 = None
        permute_120 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg179_1, view_240, permute_120);  arg179_1 = view_240 = permute_120 = None
        view_241 = torch.ops.aten.view.default(addmm_65, [8, 512, 1024]);  addmm_65 = None
        add_89 = torch.ops.aten.add.Tensor(add_85, view_241);  add_85 = view_241 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_89, getitem_45);  getitem_45 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, arg180_1);  mul_78 = arg180_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_79, arg181_1);  mul_79 = arg181_1 = None
        view_242 = torch.ops.aten.view.default(add_91, [4096, 1024])
        permute_121 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg183_1, view_242, permute_121);  arg183_1 = view_242 = permute_121 = None
        view_243 = torch.ops.aten.view.default(addmm_66, [8, 512, 1024]);  addmm_66 = None
        view_244 = torch.ops.aten.view.default(add_91, [4096, 1024])
        permute_122 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg185_1, view_244, permute_122);  arg185_1 = view_244 = permute_122 = None
        view_245 = torch.ops.aten.view.default(addmm_67, [8, 512, 1024]);  addmm_67 = None
        view_246 = torch.ops.aten.view.default(view_245, [8, 512, 16, 64]);  view_245 = None
        view_247 = torch.ops.aten.view.default(add_91, [4096, 1024]);  add_91 = None
        permute_124 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg187_1, view_247, permute_124);  arg187_1 = view_247 = permute_124 = None
        view_248 = torch.ops.aten.view.default(addmm_68, [8, 512, 1024]);  addmm_68 = None
        view_249 = torch.ops.aten.view.default(view_248, [8, 512, 16, 64]);  view_248 = None
        view_250 = torch.ops.aten.view.default(view_243, [8, 512, 16, 64]);  view_243 = None
        permute_default_36 = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        permute_default_37 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        permute_default_38 = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_36, permute_default_37, permute_default_38, None, False, scale = 0.125);  permute_default_36 = permute_default_37 = permute_default_38 = None
        getitem_112 = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
        permute_128 = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None
        clone_82 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257 = torch.ops.aten.view.default(clone_82, [8, 512, 1024]);  clone_82 = None
        view_258 = torch.ops.aten.view.default(view_257, [4096, 1024]);  view_257 = None
        permute_129 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg189_1, view_258, permute_129);  arg189_1 = view_258 = permute_129 = None
        view_259 = torch.ops.aten.view.default(addmm_69, [8, 512, 1024]);  addmm_69 = None
        add_93 = torch.ops.aten.add.Tensor(add_89, view_259);  add_89 = view_259 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_93, getitem_47);  getitem_47 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg190_1);  mul_80 = arg190_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_81, arg191_1);  mul_81 = arg191_1 = None
        view_260 = torch.ops.aten.view.default(add_95, [4096, 1024]);  add_95 = None
        permute_130 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg193_1, view_260, permute_130);  arg193_1 = view_260 = permute_130 = None
        view_261 = torch.ops.aten.view.default(addmm_70, [8, 512, 4096]);  addmm_70 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_83 = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11 = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
        view_262 = torch.ops.aten.view.default(mul_84, [4096, 4096]);  mul_84 = None
        permute_131 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg195_1, view_262, permute_131);  arg195_1 = view_262 = permute_131 = None
        view_263 = torch.ops.aten.view.default(addmm_71, [8, 512, 1024]);  addmm_71 = None
        add_97 = torch.ops.aten.add.Tensor(add_93, view_263);  add_93 = view_263 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  getitem_49 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, arg196_1);  mul_85 = arg196_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_86, arg197_1);  mul_86 = arg197_1 = None
        view_264 = torch.ops.aten.view.default(add_99, [4096, 1024])
        permute_132 = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg199_1, view_264, permute_132);  arg199_1 = view_264 = permute_132 = None
        view_265 = torch.ops.aten.view.default(addmm_72, [8, 512, 1024]);  addmm_72 = None
        view_266 = torch.ops.aten.view.default(add_99, [4096, 1024])
        permute_133 = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg201_1, view_266, permute_133);  arg201_1 = view_266 = permute_133 = None
        view_267 = torch.ops.aten.view.default(addmm_73, [8, 512, 1024]);  addmm_73 = None
        view_268 = torch.ops.aten.view.default(view_267, [8, 512, 16, 64]);  view_267 = None
        view_269 = torch.ops.aten.view.default(add_99, [4096, 1024]);  add_99 = None
        permute_135 = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg203_1, view_269, permute_135);  arg203_1 = view_269 = permute_135 = None
        view_270 = torch.ops.aten.view.default(addmm_74, [8, 512, 1024]);  addmm_74 = None
        view_271 = torch.ops.aten.view.default(view_270, [8, 512, 16, 64]);  view_270 = None
        view_272 = torch.ops.aten.view.default(view_265, [8, 512, 16, 64]);  view_265 = None
        permute_default_33 = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        permute_default_34 = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        permute_default_35 = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, False, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_111 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        permute_139 = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None
        clone_89 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_279 = torch.ops.aten.view.default(clone_89, [8, 512, 1024]);  clone_89 = None
        view_280 = torch.ops.aten.view.default(view_279, [4096, 1024]);  view_279 = None
        permute_140 = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg205_1, view_280, permute_140);  arg205_1 = view_280 = permute_140 = None
        view_281 = torch.ops.aten.view.default(addmm_75, [8, 512, 1024]);  addmm_75 = None
        add_101 = torch.ops.aten.add.Tensor(add_97, view_281);  add_97 = view_281 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_101, getitem_51);  getitem_51 = None
        mul_87 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = rsqrt_25 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, arg206_1);  mul_87 = arg206_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_88, arg207_1);  mul_88 = arg207_1 = None
        view_282 = torch.ops.aten.view.default(add_103, [4096, 1024]);  add_103 = None
        permute_141 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg209_1, view_282, permute_141);  arg209_1 = view_282 = permute_141 = None
        view_283 = torch.ops.aten.view.default(addmm_76, [8, 512, 4096]);  addmm_76 = None
        mul_89 = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_90 = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476);  view_283 = None
        erf_12 = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_104 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_89, add_104);  mul_89 = add_104 = None
        view_284 = torch.ops.aten.view.default(mul_91, [4096, 4096]);  mul_91 = None
        permute_142 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg211_1, view_284, permute_142);  arg211_1 = view_284 = permute_142 = None
        view_285 = torch.ops.aten.view.default(addmm_77, [8, 512, 1024]);  addmm_77 = None
        add_105 = torch.ops.aten.add.Tensor(add_101, view_285);  add_101 = view_285 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_105, getitem_53);  getitem_53 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, arg212_1);  mul_92 = arg212_1 = None
        add_107 = torch.ops.aten.add.Tensor(mul_93, arg213_1);  mul_93 = arg213_1 = None
        view_286 = torch.ops.aten.view.default(add_107, [4096, 1024])
        permute_143 = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg215_1, view_286, permute_143);  arg215_1 = view_286 = permute_143 = None
        view_287 = torch.ops.aten.view.default(addmm_78, [8, 512, 1024]);  addmm_78 = None
        view_288 = torch.ops.aten.view.default(add_107, [4096, 1024])
        permute_144 = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg217_1, view_288, permute_144);  arg217_1 = view_288 = permute_144 = None
        view_289 = torch.ops.aten.view.default(addmm_79, [8, 512, 1024]);  addmm_79 = None
        view_290 = torch.ops.aten.view.default(view_289, [8, 512, 16, 64]);  view_289 = None
        view_291 = torch.ops.aten.view.default(add_107, [4096, 1024]);  add_107 = None
        permute_146 = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg219_1, view_291, permute_146);  arg219_1 = view_291 = permute_146 = None
        view_292 = torch.ops.aten.view.default(addmm_80, [8, 512, 1024]);  addmm_80 = None
        view_293 = torch.ops.aten.view.default(view_292, [8, 512, 16, 64]);  view_292 = None
        view_294 = torch.ops.aten.view.default(view_287, [8, 512, 16, 64]);  view_287 = None
        permute_default_30 = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        permute_default_31 = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        permute_default_32 = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, False, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_110 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        permute_150 = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
        clone_96 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_301 = torch.ops.aten.view.default(clone_96, [8, 512, 1024]);  clone_96 = None
        view_302 = torch.ops.aten.view.default(view_301, [4096, 1024]);  view_301 = None
        permute_151 = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg221_1, view_302, permute_151);  arg221_1 = view_302 = permute_151 = None
        view_303 = torch.ops.aten.view.default(addmm_81, [8, 512, 1024]);  addmm_81 = None
        add_109 = torch.ops.aten.add.Tensor(add_105, view_303);  add_105 = view_303 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_109, getitem_55);  getitem_55 = None
        mul_94 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = rsqrt_27 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, arg222_1);  mul_94 = arg222_1 = None
        add_111 = torch.ops.aten.add.Tensor(mul_95, arg223_1);  mul_95 = arg223_1 = None
        view_304 = torch.ops.aten.view.default(add_111, [4096, 1024]);  add_111 = None
        permute_152 = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg225_1, view_304, permute_152);  arg225_1 = view_304 = permute_152 = None
        view_305 = torch.ops.aten.view.default(addmm_82, [8, 512, 4096]);  addmm_82 = None
        mul_96 = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_97 = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476);  view_305 = None
        erf_13 = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_112 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_96, add_112);  mul_96 = add_112 = None
        view_306 = torch.ops.aten.view.default(mul_98, [4096, 4096]);  mul_98 = None
        permute_153 = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg227_1, view_306, permute_153);  arg227_1 = view_306 = permute_153 = None
        view_307 = torch.ops.aten.view.default(addmm_83, [8, 512, 1024]);  addmm_83 = None
        add_113 = torch.ops.aten.add.Tensor(add_109, view_307);  add_109 = view_307 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_113, getitem_57);  getitem_57 = None
        mul_99 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = rsqrt_28 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, arg228_1);  mul_99 = arg228_1 = None
        add_115 = torch.ops.aten.add.Tensor(mul_100, arg229_1);  mul_100 = arg229_1 = None
        view_308 = torch.ops.aten.view.default(add_115, [4096, 1024])
        permute_154 = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg231_1, view_308, permute_154);  arg231_1 = view_308 = permute_154 = None
        view_309 = torch.ops.aten.view.default(addmm_84, [8, 512, 1024]);  addmm_84 = None
        view_310 = torch.ops.aten.view.default(add_115, [4096, 1024])
        permute_155 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg233_1, view_310, permute_155);  arg233_1 = view_310 = permute_155 = None
        view_311 = torch.ops.aten.view.default(addmm_85, [8, 512, 1024]);  addmm_85 = None
        view_312 = torch.ops.aten.view.default(view_311, [8, 512, 16, 64]);  view_311 = None
        view_313 = torch.ops.aten.view.default(add_115, [4096, 1024]);  add_115 = None
        permute_157 = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg235_1, view_313, permute_157);  arg235_1 = view_313 = permute_157 = None
        view_314 = torch.ops.aten.view.default(addmm_86, [8, 512, 1024]);  addmm_86 = None
        view_315 = torch.ops.aten.view.default(view_314, [8, 512, 16, 64]);  view_314 = None
        view_316 = torch.ops.aten.view.default(view_309, [8, 512, 16, 64]);  view_309 = None
        permute_default_27 = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        permute_default_28 = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        permute_default_29 = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, False, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_109 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        permute_161 = torch.ops.aten.permute.default(getitem_109, [0, 2, 1, 3]);  getitem_109 = None
        clone_103 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_323 = torch.ops.aten.view.default(clone_103, [8, 512, 1024]);  clone_103 = None
        view_324 = torch.ops.aten.view.default(view_323, [4096, 1024]);  view_323 = None
        permute_162 = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg237_1, view_324, permute_162);  arg237_1 = view_324 = permute_162 = None
        view_325 = torch.ops.aten.view.default(addmm_87, [8, 512, 1024]);  addmm_87 = None
        add_117 = torch.ops.aten.add.Tensor(add_113, view_325);  add_113 = view_325 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_117, getitem_59);  getitem_59 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = rsqrt_29 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, arg238_1);  mul_101 = arg238_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_102, arg239_1);  mul_102 = arg239_1 = None
        view_326 = torch.ops.aten.view.default(add_119, [4096, 1024]);  add_119 = None
        permute_163 = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg241_1, view_326, permute_163);  arg241_1 = view_326 = permute_163 = None
        view_327 = torch.ops.aten.view.default(addmm_88, [8, 512, 4096]);  addmm_88 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_104 = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_14 = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_120 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_103, add_120);  mul_103 = add_120 = None
        view_328 = torch.ops.aten.view.default(mul_105, [4096, 4096]);  mul_105 = None
        permute_164 = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg243_1, view_328, permute_164);  arg243_1 = view_328 = permute_164 = None
        view_329 = torch.ops.aten.view.default(addmm_89, [8, 512, 1024]);  addmm_89 = None
        add_121 = torch.ops.aten.add.Tensor(add_117, view_329);  add_117 = view_329 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_121, getitem_61);  getitem_61 = None
        mul_106 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, arg244_1);  mul_106 = arg244_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_107, arg245_1);  mul_107 = arg245_1 = None
        view_330 = torch.ops.aten.view.default(add_123, [4096, 1024])
        permute_165 = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg247_1, view_330, permute_165);  arg247_1 = view_330 = permute_165 = None
        view_331 = torch.ops.aten.view.default(addmm_90, [8, 512, 1024]);  addmm_90 = None
        view_332 = torch.ops.aten.view.default(add_123, [4096, 1024])
        permute_166 = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg249_1, view_332, permute_166);  arg249_1 = view_332 = permute_166 = None
        view_333 = torch.ops.aten.view.default(addmm_91, [8, 512, 1024]);  addmm_91 = None
        view_334 = torch.ops.aten.view.default(view_333, [8, 512, 16, 64]);  view_333 = None
        view_335 = torch.ops.aten.view.default(add_123, [4096, 1024]);  add_123 = None
        permute_168 = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg251_1, view_335, permute_168);  arg251_1 = view_335 = permute_168 = None
        view_336 = torch.ops.aten.view.default(addmm_92, [8, 512, 1024]);  addmm_92 = None
        view_337 = torch.ops.aten.view.default(view_336, [8, 512, 16, 64]);  view_336 = None
        view_338 = torch.ops.aten.view.default(view_331, [8, 512, 16, 64]);  view_331 = None
        permute_default_24 = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        permute_default_25 = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        permute_default_26 = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, False, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_108 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        permute_172 = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None
        clone_110 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_345 = torch.ops.aten.view.default(clone_110, [8, 512, 1024]);  clone_110 = None
        view_346 = torch.ops.aten.view.default(view_345, [4096, 1024]);  view_345 = None
        permute_173 = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg253_1, view_346, permute_173);  arg253_1 = view_346 = permute_173 = None
        view_347 = torch.ops.aten.view.default(addmm_93, [8, 512, 1024]);  addmm_93 = None
        add_125 = torch.ops.aten.add.Tensor(add_121, view_347);  add_121 = view_347 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_125, getitem_63);  getitem_63 = None
        mul_108 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = rsqrt_31 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, arg254_1);  mul_108 = arg254_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_109, arg255_1);  mul_109 = arg255_1 = None
        view_348 = torch.ops.aten.view.default(add_127, [4096, 1024]);  add_127 = None
        permute_174 = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg257_1, view_348, permute_174);  arg257_1 = view_348 = permute_174 = None
        view_349 = torch.ops.aten.view.default(addmm_94, [8, 512, 4096]);  addmm_94 = None
        mul_110 = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_111 = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
        erf_15 = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_128 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_110, add_128);  mul_110 = add_128 = None
        view_350 = torch.ops.aten.view.default(mul_112, [4096, 4096]);  mul_112 = None
        permute_175 = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg259_1, view_350, permute_175);  arg259_1 = view_350 = permute_175 = None
        view_351 = torch.ops.aten.view.default(addmm_95, [8, 512, 1024]);  addmm_95 = None
        add_129 = torch.ops.aten.add.Tensor(add_125, view_351);  add_125 = view_351 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_129, getitem_65);  getitem_65 = None
        mul_113 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, arg260_1);  mul_113 = arg260_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_114, arg261_1);  mul_114 = arg261_1 = None
        view_352 = torch.ops.aten.view.default(add_131, [4096, 1024])
        permute_176 = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg263_1, view_352, permute_176);  arg263_1 = view_352 = permute_176 = None
        view_353 = torch.ops.aten.view.default(addmm_96, [8, 512, 1024]);  addmm_96 = None
        view_354 = torch.ops.aten.view.default(add_131, [4096, 1024])
        permute_177 = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg265_1, view_354, permute_177);  arg265_1 = view_354 = permute_177 = None
        view_355 = torch.ops.aten.view.default(addmm_97, [8, 512, 1024]);  addmm_97 = None
        view_356 = torch.ops.aten.view.default(view_355, [8, 512, 16, 64]);  view_355 = None
        view_357 = torch.ops.aten.view.default(add_131, [4096, 1024]);  add_131 = None
        permute_179 = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg267_1, view_357, permute_179);  arg267_1 = view_357 = permute_179 = None
        view_358 = torch.ops.aten.view.default(addmm_98, [8, 512, 1024]);  addmm_98 = None
        view_359 = torch.ops.aten.view.default(view_358, [8, 512, 16, 64]);  view_358 = None
        view_360 = torch.ops.aten.view.default(view_353, [8, 512, 16, 64]);  view_353 = None
        permute_default_21 = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        permute_default_22 = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        permute_default_23 = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, False, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_107 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        permute_183 = torch.ops.aten.permute.default(getitem_107, [0, 2, 1, 3]);  getitem_107 = None
        clone_117 = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_367 = torch.ops.aten.view.default(clone_117, [8, 512, 1024]);  clone_117 = None
        view_368 = torch.ops.aten.view.default(view_367, [4096, 1024]);  view_367 = None
        permute_184 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg269_1, view_368, permute_184);  arg269_1 = view_368 = permute_184 = None
        view_369 = torch.ops.aten.view.default(addmm_99, [8, 512, 1024]);  addmm_99 = None
        add_133 = torch.ops.aten.add.Tensor(add_129, view_369);  add_129 = view_369 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_133, getitem_67);  getitem_67 = None
        mul_115 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = rsqrt_33 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, arg270_1);  mul_115 = arg270_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_116, arg271_1);  mul_116 = arg271_1 = None
        view_370 = torch.ops.aten.view.default(add_135, [4096, 1024]);  add_135 = None
        permute_185 = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg273_1, view_370, permute_185);  arg273_1 = view_370 = permute_185 = None
        view_371 = torch.ops.aten.view.default(addmm_100, [8, 512, 4096]);  addmm_100 = None
        mul_117 = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_118 = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
        erf_16 = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_136 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_117, add_136);  mul_117 = add_136 = None
        view_372 = torch.ops.aten.view.default(mul_119, [4096, 4096]);  mul_119 = None
        permute_186 = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg275_1, view_372, permute_186);  arg275_1 = view_372 = permute_186 = None
        view_373 = torch.ops.aten.view.default(addmm_101, [8, 512, 1024]);  addmm_101 = None
        add_137 = torch.ops.aten.add.Tensor(add_133, view_373);  add_133 = view_373 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_137, getitem_69);  getitem_69 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, arg276_1);  mul_120 = arg276_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_121, arg277_1);  mul_121 = arg277_1 = None
        view_374 = torch.ops.aten.view.default(add_139, [4096, 1024])
        permute_187 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg279_1, view_374, permute_187);  arg279_1 = view_374 = permute_187 = None
        view_375 = torch.ops.aten.view.default(addmm_102, [8, 512, 1024]);  addmm_102 = None
        view_376 = torch.ops.aten.view.default(add_139, [4096, 1024])
        permute_188 = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg281_1, view_376, permute_188);  arg281_1 = view_376 = permute_188 = None
        view_377 = torch.ops.aten.view.default(addmm_103, [8, 512, 1024]);  addmm_103 = None
        view_378 = torch.ops.aten.view.default(view_377, [8, 512, 16, 64]);  view_377 = None
        view_379 = torch.ops.aten.view.default(add_139, [4096, 1024]);  add_139 = None
        permute_190 = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg283_1, view_379, permute_190);  arg283_1 = view_379 = permute_190 = None
        view_380 = torch.ops.aten.view.default(addmm_104, [8, 512, 1024]);  addmm_104 = None
        view_381 = torch.ops.aten.view.default(view_380, [8, 512, 16, 64]);  view_380 = None
        view_382 = torch.ops.aten.view.default(view_375, [8, 512, 16, 64]);  view_375 = None
        permute_default_18 = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        permute_default_19 = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        permute_default_20 = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, False, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_106 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        permute_194 = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        clone_124 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_389 = torch.ops.aten.view.default(clone_124, [8, 512, 1024]);  clone_124 = None
        view_390 = torch.ops.aten.view.default(view_389, [4096, 1024]);  view_389 = None
        permute_195 = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg285_1, view_390, permute_195);  arg285_1 = view_390 = permute_195 = None
        view_391 = torch.ops.aten.view.default(addmm_105, [8, 512, 1024]);  addmm_105 = None
        add_141 = torch.ops.aten.add.Tensor(add_137, view_391);  add_137 = view_391 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_141, getitem_71);  getitem_71 = None
        mul_122 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, arg286_1);  mul_122 = arg286_1 = None
        add_143 = torch.ops.aten.add.Tensor(mul_123, arg287_1);  mul_123 = arg287_1 = None
        view_392 = torch.ops.aten.view.default(add_143, [4096, 1024]);  add_143 = None
        permute_196 = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg289_1, view_392, permute_196);  arg289_1 = view_392 = permute_196 = None
        view_393 = torch.ops.aten.view.default(addmm_106, [8, 512, 4096]);  addmm_106 = None
        mul_124 = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_125 = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
        erf_17 = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_144 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_124, add_144);  mul_124 = add_144 = None
        view_394 = torch.ops.aten.view.default(mul_126, [4096, 4096]);  mul_126 = None
        permute_197 = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg291_1, view_394, permute_197);  arg291_1 = view_394 = permute_197 = None
        view_395 = torch.ops.aten.view.default(addmm_107, [8, 512, 1024]);  addmm_107 = None
        add_145 = torch.ops.aten.add.Tensor(add_141, view_395);  add_141 = view_395 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_146 = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_55 = torch.ops.aten.sub.Tensor(add_145, getitem_73);  getitem_73 = None
        mul_127 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, arg292_1);  mul_127 = arg292_1 = None
        add_147 = torch.ops.aten.add.Tensor(mul_128, arg293_1);  mul_128 = arg293_1 = None
        view_396 = torch.ops.aten.view.default(add_147, [4096, 1024])
        permute_198 = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg295_1, view_396, permute_198);  arg295_1 = view_396 = permute_198 = None
        view_397 = torch.ops.aten.view.default(addmm_108, [8, 512, 1024]);  addmm_108 = None
        view_398 = torch.ops.aten.view.default(add_147, [4096, 1024])
        permute_199 = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg297_1, view_398, permute_199);  arg297_1 = view_398 = permute_199 = None
        view_399 = torch.ops.aten.view.default(addmm_109, [8, 512, 1024]);  addmm_109 = None
        view_400 = torch.ops.aten.view.default(view_399, [8, 512, 16, 64]);  view_399 = None
        view_401 = torch.ops.aten.view.default(add_147, [4096, 1024]);  add_147 = None
        permute_201 = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg299_1, view_401, permute_201);  arg299_1 = view_401 = permute_201 = None
        view_402 = torch.ops.aten.view.default(addmm_110, [8, 512, 1024]);  addmm_110 = None
        view_403 = torch.ops.aten.view.default(view_402, [8, 512, 16, 64]);  view_402 = None
        view_404 = torch.ops.aten.view.default(view_397, [8, 512, 16, 64]);  view_397 = None
        permute_default_15 = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        permute_default_16 = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        permute_default_17 = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, False, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_105 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        permute_205 = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None
        clone_131 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_411 = torch.ops.aten.view.default(clone_131, [8, 512, 1024]);  clone_131 = None
        view_412 = torch.ops.aten.view.default(view_411, [4096, 1024]);  view_411 = None
        permute_206 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg301_1, view_412, permute_206);  arg301_1 = view_412 = permute_206 = None
        view_413 = torch.ops.aten.view.default(addmm_111, [8, 512, 1024]);  addmm_111 = None
        add_149 = torch.ops.aten.add.Tensor(add_145, view_413);  add_145 = view_413 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_150 = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_149, getitem_75);  getitem_75 = None
        mul_129 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, arg302_1);  mul_129 = arg302_1 = None
        add_151 = torch.ops.aten.add.Tensor(mul_130, arg303_1);  mul_130 = arg303_1 = None
        view_414 = torch.ops.aten.view.default(add_151, [4096, 1024]);  add_151 = None
        permute_207 = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg305_1, view_414, permute_207);  arg305_1 = view_414 = permute_207 = None
        view_415 = torch.ops.aten.view.default(addmm_112, [8, 512, 4096]);  addmm_112 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_132 = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
        erf_18 = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_152 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_131, add_152);  mul_131 = add_152 = None
        view_416 = torch.ops.aten.view.default(mul_133, [4096, 4096]);  mul_133 = None
        permute_208 = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg307_1, view_416, permute_208);  arg307_1 = view_416 = permute_208 = None
        view_417 = torch.ops.aten.view.default(addmm_113, [8, 512, 1024]);  addmm_113 = None
        add_153 = torch.ops.aten.add.Tensor(add_149, view_417);  add_149 = view_417 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_58 = torch.ops.aten.sub.Tensor(add_153, getitem_77);  getitem_77 = None
        mul_134 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = rsqrt_38 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, arg308_1);  mul_134 = arg308_1 = None
        add_155 = torch.ops.aten.add.Tensor(mul_135, arg309_1);  mul_135 = arg309_1 = None
        view_418 = torch.ops.aten.view.default(add_155, [4096, 1024])
        permute_209 = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg311_1, view_418, permute_209);  arg311_1 = view_418 = permute_209 = None
        view_419 = torch.ops.aten.view.default(addmm_114, [8, 512, 1024]);  addmm_114 = None
        view_420 = torch.ops.aten.view.default(add_155, [4096, 1024])
        permute_210 = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg313_1, view_420, permute_210);  arg313_1 = view_420 = permute_210 = None
        view_421 = torch.ops.aten.view.default(addmm_115, [8, 512, 1024]);  addmm_115 = None
        view_422 = torch.ops.aten.view.default(view_421, [8, 512, 16, 64]);  view_421 = None
        view_423 = torch.ops.aten.view.default(add_155, [4096, 1024]);  add_155 = None
        permute_212 = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg315_1, view_423, permute_212);  arg315_1 = view_423 = permute_212 = None
        view_424 = torch.ops.aten.view.default(addmm_116, [8, 512, 1024]);  addmm_116 = None
        view_425 = torch.ops.aten.view.default(view_424, [8, 512, 16, 64]);  view_424 = None
        view_426 = torch.ops.aten.view.default(view_419, [8, 512, 16, 64]);  view_419 = None
        permute_default_12 = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        permute_default_13 = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        permute_default_14 = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, False, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_104 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        permute_216 = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        clone_138 = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_433 = torch.ops.aten.view.default(clone_138, [8, 512, 1024]);  clone_138 = None
        view_434 = torch.ops.aten.view.default(view_433, [4096, 1024]);  view_433 = None
        permute_217 = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg317_1, view_434, permute_217);  arg317_1 = view_434 = permute_217 = None
        view_435 = torch.ops.aten.view.default(addmm_117, [8, 512, 1024]);  addmm_117 = None
        add_157 = torch.ops.aten.add.Tensor(add_153, view_435);  add_153 = view_435 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_158 = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_60 = torch.ops.aten.sub.Tensor(add_157, getitem_79);  getitem_79 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = rsqrt_39 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, arg318_1);  mul_136 = arg318_1 = None
        add_159 = torch.ops.aten.add.Tensor(mul_137, arg319_1);  mul_137 = arg319_1 = None
        view_436 = torch.ops.aten.view.default(add_159, [4096, 1024]);  add_159 = None
        permute_218 = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg321_1, view_436, permute_218);  arg321_1 = view_436 = permute_218 = None
        view_437 = torch.ops.aten.view.default(addmm_118, [8, 512, 4096]);  addmm_118 = None
        mul_138 = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_139 = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476);  view_437 = None
        erf_19 = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_160 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_138, add_160);  mul_138 = add_160 = None
        view_438 = torch.ops.aten.view.default(mul_140, [4096, 4096]);  mul_140 = None
        permute_219 = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg323_1, view_438, permute_219);  arg323_1 = view_438 = permute_219 = None
        view_439 = torch.ops.aten.view.default(addmm_119, [8, 512, 1024]);  addmm_119 = None
        add_161 = torch.ops.aten.add.Tensor(add_157, view_439);  add_157 = view_439 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_61 = torch.ops.aten.sub.Tensor(add_161, getitem_81);  getitem_81 = None
        mul_141 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = rsqrt_40 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_141, arg324_1);  mul_141 = arg324_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_142, arg325_1);  mul_142 = arg325_1 = None
        view_440 = torch.ops.aten.view.default(add_163, [4096, 1024])
        permute_220 = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg327_1, view_440, permute_220);  arg327_1 = view_440 = permute_220 = None
        view_441 = torch.ops.aten.view.default(addmm_120, [8, 512, 1024]);  addmm_120 = None
        view_442 = torch.ops.aten.view.default(add_163, [4096, 1024])
        permute_221 = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg329_1, view_442, permute_221);  arg329_1 = view_442 = permute_221 = None
        view_443 = torch.ops.aten.view.default(addmm_121, [8, 512, 1024]);  addmm_121 = None
        view_444 = torch.ops.aten.view.default(view_443, [8, 512, 16, 64]);  view_443 = None
        view_445 = torch.ops.aten.view.default(add_163, [4096, 1024]);  add_163 = None
        permute_223 = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg331_1, view_445, permute_223);  arg331_1 = view_445 = permute_223 = None
        view_446 = torch.ops.aten.view.default(addmm_122, [8, 512, 1024]);  addmm_122 = None
        view_447 = torch.ops.aten.view.default(view_446, [8, 512, 16, 64]);  view_446 = None
        view_448 = torch.ops.aten.view.default(view_441, [8, 512, 16, 64]);  view_441 = None
        permute_default_9 = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        permute_default_10 = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        permute_default_11 = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, False, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_103 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        permute_227 = torch.ops.aten.permute.default(getitem_103, [0, 2, 1, 3]);  getitem_103 = None
        clone_145 = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_455 = torch.ops.aten.view.default(clone_145, [8, 512, 1024]);  clone_145 = None
        view_456 = torch.ops.aten.view.default(view_455, [4096, 1024]);  view_455 = None
        permute_228 = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg333_1, view_456, permute_228);  arg333_1 = view_456 = permute_228 = None
        view_457 = torch.ops.aten.view.default(addmm_123, [8, 512, 1024]);  addmm_123 = None
        add_165 = torch.ops.aten.add.Tensor(add_161, view_457);  add_161 = view_457 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_166 = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_63 = torch.ops.aten.sub.Tensor(add_165, getitem_83);  getitem_83 = None
        mul_143 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = rsqrt_41 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, arg334_1);  mul_143 = arg334_1 = None
        add_167 = torch.ops.aten.add.Tensor(mul_144, arg335_1);  mul_144 = arg335_1 = None
        view_458 = torch.ops.aten.view.default(add_167, [4096, 1024]);  add_167 = None
        permute_229 = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg337_1, view_458, permute_229);  arg337_1 = view_458 = permute_229 = None
        view_459 = torch.ops.aten.view.default(addmm_124, [8, 512, 4096]);  addmm_124 = None
        mul_145 = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_146 = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476);  view_459 = None
        erf_20 = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_168 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_145, add_168);  mul_145 = add_168 = None
        view_460 = torch.ops.aten.view.default(mul_147, [4096, 4096]);  mul_147 = None
        permute_230 = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg339_1, view_460, permute_230);  arg339_1 = view_460 = permute_230 = None
        view_461 = torch.ops.aten.view.default(addmm_125, [8, 512, 1024]);  addmm_125 = None
        add_169 = torch.ops.aten.add.Tensor(add_165, view_461);  add_165 = view_461 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_64 = torch.ops.aten.sub.Tensor(add_169, getitem_85);  getitem_85 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = rsqrt_42 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, arg340_1);  mul_148 = arg340_1 = None
        add_171 = torch.ops.aten.add.Tensor(mul_149, arg341_1);  mul_149 = arg341_1 = None
        view_462 = torch.ops.aten.view.default(add_171, [4096, 1024])
        permute_231 = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg343_1, view_462, permute_231);  arg343_1 = view_462 = permute_231 = None
        view_463 = torch.ops.aten.view.default(addmm_126, [8, 512, 1024]);  addmm_126 = None
        view_464 = torch.ops.aten.view.default(add_171, [4096, 1024])
        permute_232 = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg345_1, view_464, permute_232);  arg345_1 = view_464 = permute_232 = None
        view_465 = torch.ops.aten.view.default(addmm_127, [8, 512, 1024]);  addmm_127 = None
        view_466 = torch.ops.aten.view.default(view_465, [8, 512, 16, 64]);  view_465 = None
        view_467 = torch.ops.aten.view.default(add_171, [4096, 1024]);  add_171 = None
        permute_234 = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg347_1, view_467, permute_234);  arg347_1 = view_467 = permute_234 = None
        view_468 = torch.ops.aten.view.default(addmm_128, [8, 512, 1024]);  addmm_128 = None
        view_469 = torch.ops.aten.view.default(view_468, [8, 512, 16, 64]);  view_468 = None
        view_470 = torch.ops.aten.view.default(view_463, [8, 512, 16, 64]);  view_463 = None
        permute_default_6 = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        permute_default_7 = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        permute_default_8 = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, False, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_102 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        permute_238 = torch.ops.aten.permute.default(getitem_102, [0, 2, 1, 3]);  getitem_102 = None
        clone_152 = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_477 = torch.ops.aten.view.default(clone_152, [8, 512, 1024]);  clone_152 = None
        view_478 = torch.ops.aten.view.default(view_477, [4096, 1024]);  view_477 = None
        permute_239 = torch.ops.aten.permute.default(arg348_1, [1, 0]);  arg348_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg349_1, view_478, permute_239);  arg349_1 = view_478 = permute_239 = None
        view_479 = torch.ops.aten.view.default(addmm_129, [8, 512, 1024]);  addmm_129 = None
        add_173 = torch.ops.aten.add.Tensor(add_169, view_479);  add_169 = view_479 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_66 = torch.ops.aten.sub.Tensor(add_173, getitem_87);  getitem_87 = None
        mul_150 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = rsqrt_43 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, arg350_1);  mul_150 = arg350_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_151, arg351_1);  mul_151 = arg351_1 = None
        view_480 = torch.ops.aten.view.default(add_175, [4096, 1024]);  add_175 = None
        permute_240 = torch.ops.aten.permute.default(arg352_1, [1, 0]);  arg352_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg353_1, view_480, permute_240);  arg353_1 = view_480 = permute_240 = None
        view_481 = torch.ops.aten.view.default(addmm_130, [8, 512, 4096]);  addmm_130 = None
        mul_152 = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_153 = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476);  view_481 = None
        erf_21 = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_176 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_152, add_176);  mul_152 = add_176 = None
        view_482 = torch.ops.aten.view.default(mul_154, [4096, 4096]);  mul_154 = None
        permute_241 = torch.ops.aten.permute.default(arg354_1, [1, 0]);  arg354_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg355_1, view_482, permute_241);  arg355_1 = view_482 = permute_241 = None
        view_483 = torch.ops.aten.view.default(addmm_131, [8, 512, 1024]);  addmm_131 = None
        add_177 = torch.ops.aten.add.Tensor(add_173, view_483);  add_173 = view_483 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_178 = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_67 = torch.ops.aten.sub.Tensor(add_177, getitem_89);  getitem_89 = None
        mul_155 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = rsqrt_44 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, arg356_1);  mul_155 = arg356_1 = None
        add_179 = torch.ops.aten.add.Tensor(mul_156, arg357_1);  mul_156 = arg357_1 = None
        view_484 = torch.ops.aten.view.default(add_179, [4096, 1024])
        permute_242 = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg359_1, view_484, permute_242);  arg359_1 = view_484 = permute_242 = None
        view_485 = torch.ops.aten.view.default(addmm_132, [8, 512, 1024]);  addmm_132 = None
        view_486 = torch.ops.aten.view.default(add_179, [4096, 1024])
        permute_243 = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg361_1, view_486, permute_243);  arg361_1 = view_486 = permute_243 = None
        view_487 = torch.ops.aten.view.default(addmm_133, [8, 512, 1024]);  addmm_133 = None
        view_488 = torch.ops.aten.view.default(view_487, [8, 512, 16, 64]);  view_487 = None
        view_489 = torch.ops.aten.view.default(add_179, [4096, 1024]);  add_179 = None
        permute_245 = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg363_1, view_489, permute_245);  arg363_1 = view_489 = permute_245 = None
        view_490 = torch.ops.aten.view.default(addmm_134, [8, 512, 1024]);  addmm_134 = None
        view_491 = torch.ops.aten.view.default(view_490, [8, 512, 16, 64]);  view_490 = None
        view_492 = torch.ops.aten.view.default(view_485, [8, 512, 16, 64]);  view_485 = None
        permute_default_3 = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        permute_default_4 = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        permute_default_5 = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, False, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_101 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        permute_249 = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        clone_159 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_499 = torch.ops.aten.view.default(clone_159, [8, 512, 1024]);  clone_159 = None
        view_500 = torch.ops.aten.view.default(view_499, [4096, 1024]);  view_499 = None
        permute_250 = torch.ops.aten.permute.default(arg364_1, [1, 0]);  arg364_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg365_1, view_500, permute_250);  arg365_1 = view_500 = permute_250 = None
        view_501 = torch.ops.aten.view.default(addmm_135, [8, 512, 1024]);  addmm_135 = None
        add_181 = torch.ops.aten.add.Tensor(add_177, view_501);  add_177 = view_501 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_182 = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_69 = torch.ops.aten.sub.Tensor(add_181, getitem_91);  getitem_91 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = rsqrt_45 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, arg366_1);  mul_157 = arg366_1 = None
        add_183 = torch.ops.aten.add.Tensor(mul_158, arg367_1);  mul_158 = arg367_1 = None
        view_502 = torch.ops.aten.view.default(add_183, [4096, 1024]);  add_183 = None
        permute_251 = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg369_1, view_502, permute_251);  arg369_1 = view_502 = permute_251 = None
        view_503 = torch.ops.aten.view.default(addmm_136, [8, 512, 4096]);  addmm_136 = None
        mul_159 = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_160 = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
        erf_22 = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_184 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_184);  mul_159 = add_184 = None
        view_504 = torch.ops.aten.view.default(mul_161, [4096, 4096]);  mul_161 = None
        permute_252 = torch.ops.aten.permute.default(arg370_1, [1, 0]);  arg370_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg371_1, view_504, permute_252);  arg371_1 = view_504 = permute_252 = None
        view_505 = torch.ops.aten.view.default(addmm_137, [8, 512, 1024]);  addmm_137 = None
        add_185 = torch.ops.aten.add.Tensor(add_181, view_505);  add_181 = view_505 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_70 = torch.ops.aten.sub.Tensor(add_185, getitem_93);  getitem_93 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = rsqrt_46 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, arg372_1);  mul_162 = arg372_1 = None
        add_187 = torch.ops.aten.add.Tensor(mul_163, arg373_1);  mul_163 = arg373_1 = None
        view_506 = torch.ops.aten.view.default(add_187, [4096, 1024])
        permute_253 = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg375_1, view_506, permute_253);  arg375_1 = view_506 = permute_253 = None
        view_507 = torch.ops.aten.view.default(addmm_138, [8, 512, 1024]);  addmm_138 = None
        view_508 = torch.ops.aten.view.default(add_187, [4096, 1024])
        permute_254 = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg377_1, view_508, permute_254);  arg377_1 = view_508 = permute_254 = None
        view_509 = torch.ops.aten.view.default(addmm_139, [8, 512, 1024]);  addmm_139 = None
        view_510 = torch.ops.aten.view.default(view_509, [8, 512, 16, 64]);  view_509 = None
        view_511 = torch.ops.aten.view.default(add_187, [4096, 1024]);  add_187 = None
        permute_256 = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg379_1, view_511, permute_256);  arg379_1 = view_511 = permute_256 = None
        view_512 = torch.ops.aten.view.default(addmm_140, [8, 512, 1024]);  addmm_140 = None
        view_513 = torch.ops.aten.view.default(view_512, [8, 512, 16, 64]);  view_512 = None
        view_514 = torch.ops.aten.view.default(view_507, [8, 512, 16, 64]);  view_507 = None
        permute_default = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        permute_default_1 = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
        permute_default_2 = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, False, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_100 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        permute_260 = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        clone_166 = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_521 = torch.ops.aten.view.default(clone_166, [8, 512, 1024]);  clone_166 = None
        view_522 = torch.ops.aten.view.default(view_521, [4096, 1024]);  view_521 = None
        permute_261 = torch.ops.aten.permute.default(arg380_1, [1, 0]);  arg380_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg381_1, view_522, permute_261);  arg381_1 = view_522 = permute_261 = None
        view_523 = torch.ops.aten.view.default(addmm_141, [8, 512, 1024]);  addmm_141 = None
        add_189 = torch.ops.aten.add.Tensor(add_185, view_523);  add_185 = view_523 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_190 = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_72 = torch.ops.aten.sub.Tensor(add_189, getitem_95);  getitem_95 = None
        mul_164 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = rsqrt_47 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, arg382_1);  mul_164 = arg382_1 = None
        add_191 = torch.ops.aten.add.Tensor(mul_165, arg383_1);  mul_165 = arg383_1 = None
        view_524 = torch.ops.aten.view.default(add_191, [4096, 1024]);  add_191 = None
        permute_262 = torch.ops.aten.permute.default(arg384_1, [1, 0]);  arg384_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg385_1, view_524, permute_262);  arg385_1 = view_524 = permute_262 = None
        view_525 = torch.ops.aten.view.default(addmm_142, [8, 512, 4096]);  addmm_142 = None
        mul_166 = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_167 = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476);  view_525 = None
        erf_23 = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_192 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_166, add_192);  mul_166 = add_192 = None
        view_526 = torch.ops.aten.view.default(mul_168, [4096, 4096]);  mul_168 = None
        permute_263 = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg387_1, view_526, permute_263);  arg387_1 = view_526 = permute_263 = None
        view_527 = torch.ops.aten.view.default(addmm_143, [8, 512, 1024]);  addmm_143 = None
        add_193 = torch.ops.aten.add.Tensor(add_189, view_527);  add_189 = view_527 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_194 = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_73 = torch.ops.aten.sub.Tensor(add_193, getitem_97);  add_193 = getitem_97 = None
        mul_169 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = rsqrt_48 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, arg388_1);  mul_169 = arg388_1 = None
        add_195 = torch.ops.aten.add.Tensor(mul_170, arg389_1);  mul_170 = arg389_1 = None
        view_528 = torch.ops.aten.view.default(add_195, [4096, 1024]);  add_195 = None
        permute_264 = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg392_1, view_528, permute_264);  arg392_1 = view_528 = permute_264 = None
        view_529 = torch.ops.aten.view.default(addmm_144, [8, 512, 2]);  addmm_144 = None
        split = torch.ops.aten.split.Tensor(view_529, 1, -1);  view_529 = None
        getitem_98 = split[0]
        getitem_99 = split[1];  split = None
        squeeze = torch.ops.aten.squeeze.dim(getitem_98, -1);  getitem_98 = None
        clone_169 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1 = torch.ops.aten.squeeze.dim(getitem_99, -1);  getitem_99 = None
        clone_170 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(arg393_1, 0);  arg393_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(arg394_1, 0);  arg394_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        amax_24 = torch.ops.aten.amax.default(clone_169, [1], True)
        sub_74 = torch.ops.aten.sub.Tensor(clone_169, amax_24);  amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_74)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
        ne = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, clamp_max, full_default_2);  ne = full_default_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_75, 1, unsqueeze_2);  sub_75 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        ne_1 = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_48 = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
        amax_25 = torch.ops.aten.amax.default(clone_170, [1], True)
        sub_76 = torch.ops.aten.sub.Tensor(clone_170, amax_25);  amax_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_76)
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
        log_1 = torch.ops.aten.log.default(sum_28);  sum_28 = None
        sub_77 = torch.ops.aten.sub.Tensor(sub_76, log_1);  sub_76 = log_1 = None
        ne_3 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_4 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_4);  ne_3 = full_default_4 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1 = torch.ops.aten.gather.default(sub_77, 1, unsqueeze_3);  sub_77 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_4 = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_5 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(ne_4, neg_1, full_default_5);  ne_4 = neg_1 = full_default_5 = None
        ne_5 = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        sum_29 = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
        sum_30 = torch.ops.aten.sum.default(where_3);  where_3 = None
        div_49 = torch.ops.aten.div.Tensor(sum_30, convert_element_type_1);  sum_30 = convert_element_type_1 = None
        add_196 = torch.ops.aten.add.Tensor(div_48, div_49);  div_48 = div_49 = None
        div_50 = torch.ops.aten.div.Tensor(add_196, 2);  add_196 = None
        return (div_50, clone_169, clone_170)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (8, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 119013376, device=device(type='cuda', index=0))
    reader.tensor(buf1, (29056, 1024), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf2, (512, 1024), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf3, (2, 1024), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1024,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1024,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1024, 1024), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1024,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1024, 1024), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1024, 1024), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1024,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1024, 1024), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1024,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1024,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1024,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf16, (4096, 1024), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf17, (4096,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1024, 4096), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1024,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1024, 1024), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1024,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1024, 1024), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1024,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1024, 1024), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1024,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1024, 1024), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1024,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1024,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1024,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4096, 1024), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf33, (4096,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1024, 4096), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1024,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1024,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1024, 1024), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1024,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1024, 1024), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1024,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1024, 1024), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf43, (1024,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf44, (1024, 1024), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1024,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1024,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1024,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf48, (4096, 1024), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf49, (4096,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1024, 4096), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1024,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf53, (1024,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf54, (1024, 1024), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf55, (1024,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf56, (1024, 1024), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf57, (1024,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf58, (1024, 1024), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1024,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1024, 1024), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1024,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1024,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1024,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf64, (4096, 1024), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf65, (4096,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf66, (1024, 4096), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf67, (1024,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1024,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1024,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1024, 1024), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1024,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1024, 1024), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf73, (1024,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf74, (1024, 1024), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf75, (1024,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1024, 1024), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1024,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1024,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1024,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf80, (4096, 1024), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf81, (4096,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1024, 4096), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1024,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1024,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1024,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1024, 1024), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1024,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1024, 1024), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1024,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf90, (1024, 1024), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1024,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1024, 1024), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1024,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf94, (1024,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1024,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf96, (4096, 1024), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf97, (4096,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf98, (1024, 4096), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1024,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1024,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1024, 1024), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1024,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf104, (1024, 1024), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1024,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1024, 1024), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1024,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1024, 1024), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1024,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1024,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf111, (1024,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf112, (4096, 1024), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf113, (4096,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1024, 4096), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1024,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1024,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1024, 1024), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1024,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1024, 1024), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1024,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1024, 1024), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf123, (1024,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1024, 1024), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1024,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1024,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1024,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf128, (4096, 1024), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf129, (4096,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1024, 4096), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1024, 1024), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024, 1024), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1024, 1024), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1024,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1024, 1024), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1024,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1024,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf143, (1024,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf144, (4096, 1024), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf145, (4096,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1024, 4096), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1024,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1024,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1024,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1024, 1024), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1024,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1024, 1024), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1024,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf154, (1024, 1024), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1024,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1024, 1024), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1024,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1024,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1024,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf160, (4096, 1024), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf161, (4096,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1024, 4096), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1024,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf164, (1024,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf165, (1024,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1024, 1024), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1024,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1024, 1024), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1024,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1024, 1024), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1024,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1024, 1024), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1024,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1024,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1024,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf176, (4096, 1024), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf177, (4096,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1024, 4096), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1024,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1024,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1024,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1024, 1024), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1024,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1024, 1024), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1024,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1024, 1024), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1024,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf188, (1024, 1024), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1024,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf190, (1024,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1024,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf192, (4096, 1024), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf193, (4096,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf194, (1024, 4096), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1024,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1024,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1024,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1024, 1024), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1024,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1024, 1024), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1024,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1024, 1024), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1024,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1024, 1024), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1024,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1024,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1024,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf208, (4096, 1024), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf209, (4096,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf210, (1024, 4096), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf211, (1024,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1024,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1024,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1024, 1024), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1024,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1024, 1024), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1024,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1024, 1024), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1024,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1024, 1024), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf221, (1024,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf222, (1024,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1024,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf224, (4096, 1024), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf225, (4096,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1024, 4096), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1024,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf228, (1024,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf229, (1024,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf230, (1024, 1024), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1024,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1024, 1024), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1024,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf234, (1024, 1024), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf235, (1024,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1024, 1024), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf237, (1024,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf238, (1024,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf239, (1024,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf240, (4096, 1024), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf241, (4096,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf242, (1024, 4096), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf243, (1024,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf244, (1024,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1024,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1024, 1024), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf247, (1024,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf248, (1024, 1024), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1024,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf250, (1024, 1024), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf251, (1024,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf252, (1024, 1024), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf253, (1024,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf254, (1024,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1024,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf256, (4096, 1024), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf257, (4096,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1024, 4096), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1024,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1024,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1024,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1024, 1024), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1024,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1024, 1024), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1024,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1024, 1024), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1024,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1024, 1024), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf269, (1024,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf270, (1024,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1024,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf272, (4096, 1024), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf273, (4096,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf274, (1024, 4096), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1024,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1024,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf277, (1024,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf278, (1024, 1024), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1024,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1024, 1024), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf281, (1024,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf282, (1024, 1024), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf283, (1024,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf284, (1024, 1024), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf285, (1024,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1024,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf287, (1024,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf288, (4096, 1024), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf289, (4096,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1024, 4096), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1024,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1024,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1024,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1024, 1024), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1024,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf296, (1024, 1024), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf297, (1024,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf298, (1024, 1024), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf299, (1024,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf300, (1024, 1024), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf301, (1024,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1024,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1024,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf304, (4096, 1024), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf305, (4096,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1024, 4096), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf307, (1024,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf308, (1024,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1024,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf310, (1024, 1024), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf311, (1024,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf312, (1024, 1024), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf313, (1024,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf314, (1024, 1024), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf315, (1024,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1024, 1024), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1024,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1024,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1024,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf320, (4096, 1024), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf321, (4096,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1024, 4096), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf323, (1024,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1024,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1024,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1024, 1024), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1024,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf328, (1024, 1024), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1024,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1024, 1024), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1024,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1024, 1024), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1024,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf334, (1024,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf335, (1024,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf336, (4096, 1024), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf337, (4096,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf338, (1024, 4096), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1024,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1024,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1024,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1024, 1024), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1024,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1024, 1024), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1024,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1024, 1024), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1024,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1024, 1024), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1024,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1024,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1024,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf352, (4096, 1024), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf353, (4096,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf354, (1024, 4096), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1024,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1024,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf357, (1024,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf358, (1024, 1024), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf359, (1024,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf360, (1024, 1024), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf361, (1024,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf362, (1024, 1024), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf363, (1024,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf364, (1024, 1024), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf365, (1024,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf366, (1024,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf367, (1024,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf368, (4096, 1024), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf369, (4096,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf370, (1024, 4096), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf371, (1024,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf372, (1024,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf373, (1024,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf374, (1024, 1024), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf375, (1024,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf376, (1024, 1024), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf377, (1024,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf378, (1024, 1024), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf379, (1024,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf380, (1024, 1024), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf381, (1024,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf382, (1024,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf383, (1024,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf384, (4096, 1024), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf385, (4096,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf386, (1024, 4096), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf387, (1024,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf388, (1024,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf389, (1024,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf390, (1, 512), dtype=torch.int64, is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf391, (2, 1024), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf392, (2,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 64, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf393, (8,), dtype=torch.int64, is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 64, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf394, (8,), dtype=torch.int64, is_leaf=True)  # arg394_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)