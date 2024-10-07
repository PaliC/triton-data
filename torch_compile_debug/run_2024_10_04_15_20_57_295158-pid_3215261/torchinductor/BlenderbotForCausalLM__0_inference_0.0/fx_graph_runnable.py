
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1):
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
        view_2 = torch.ops.aten.view.default(add_3, [512, 2560])
        permute = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3 = torch.ops.aten.view.default(addmm, [4, 128, 2560]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_3, 0.11180339887498948);  view_3 = None
        view_4 = torch.ops.aten.view.default(add_3, [512, 2560])
        permute_1 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, view_4, permute_1);  arg8_1 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [4, 128, 2560]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [4, -1, 32, 80]);  view_5 = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7 = torch.ops.aten.view.default(add_3, [512, 2560]);  add_3 = None
        permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg10_1, view_7, permute_3);  arg10_1 = view_7 = permute_3 = None
        view_8 = torch.ops.aten.view.default(addmm_2, [4, 128, 2560]);  addmm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [4, -1, 32, 80]);  view_8 = None
        permute_4 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_10 = torch.ops.aten.view.default(mul_3, [4, 128, 32, 80]);  mul_3 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11 = torch.ops.aten.view.default(clone_3, [128, -1, 80]);  clone_3 = None
        view_12 = torch.ops.aten.view.default(clone_1, [128, -1, 80])
        view_13 = torch.ops.aten.view.default(clone_2, [128, -1, 80])
        permute_6 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
        view_14 = torch.ops.aten.view.default(bmm, [4, 32, 128, 128]);  bmm = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [4, 1, 128, 128]);  unsqueeze_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_14, expand_1);  view_14 = None
        view_15 = torch.ops.aten.view.default(add_4, [128, 128, 128]);  add_4 = None
        amax = torch.ops.aten.amax.default(view_15, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm_1 = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
        view_16 = torch.ops.aten.view.default(bmm_1, [4, 32, 128, 80]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17 = torch.ops.aten.view.default(clone_5, [4, 128, 2560]);  clone_5 = None
        view_18 = torch.ops.aten.view.default(view_17, [512, 2560]);  view_17 = None
        permute_8 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg12_1, view_18, permute_8);  arg12_1 = view_18 = permute_8 = None
        view_19 = torch.ops.aten.view.default(addmm_3, [4, 128, 2560]);  addmm_3 = None
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
        view_20 = torch.ops.aten.view.default(add_7, [512, 2560]);  add_7 = None
        permute_9 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg16_1, view_20, permute_9);  arg16_1 = view_20 = permute_9 = None
        view_21 = torch.ops.aten.view.default(addmm_4, [4, 128, 10240]);  addmm_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_7 = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
        view_22 = torch.ops.aten.view.default(mul_8, [512, 10240]);  mul_8 = None
        permute_10 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg18_1, view_22, permute_10);  arg18_1 = view_22 = permute_10 = None
        view_23 = torch.ops.aten.view.default(addmm_5, [4, 128, 2560]);  addmm_5 = None
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
        view_24 = torch.ops.aten.view.default(add_11, [512, 2560])
        permute_11 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg22_1, view_24, permute_11);  arg22_1 = view_24 = permute_11 = None
        view_25 = torch.ops.aten.view.default(addmm_6, [4, 128, 2560]);  addmm_6 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_25, 0.11180339887498948);  view_25 = None
        view_26 = torch.ops.aten.view.default(add_11, [512, 2560])
        permute_12 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg24_1, view_26, permute_12);  arg24_1 = view_26 = permute_12 = None
        view_27 = torch.ops.aten.view.default(addmm_7, [4, 128, 2560]);  addmm_7 = None
        view_28 = torch.ops.aten.view.default(view_27, [4, -1, 32, 80]);  view_27 = None
        permute_13 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_9 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_29 = torch.ops.aten.view.default(add_11, [512, 2560]);  add_11 = None
        permute_14 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg26_1, view_29, permute_14);  arg26_1 = view_29 = permute_14 = None
        view_30 = torch.ops.aten.view.default(addmm_8, [4, 128, 2560]);  addmm_8 = None
        view_31 = torch.ops.aten.view.default(view_30, [4, -1, 32, 80]);  view_30 = None
        permute_15 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        clone_10 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_32 = torch.ops.aten.view.default(mul_11, [4, 128, 32, 80]);  mul_11 = None
        permute_16 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_11 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_33 = torch.ops.aten.view.default(clone_11, [128, -1, 80]);  clone_11 = None
        view_34 = torch.ops.aten.view.default(clone_9, [128, -1, 80])
        view_35 = torch.ops.aten.view.default(clone_10, [128, -1, 80])
        permute_17 = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        bmm_2 = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
        view_36 = torch.ops.aten.view.default(bmm_2, [4, 32, 128, 128]);  bmm_2 = None
        add_12 = torch.ops.aten.add.Tensor(view_36, expand_1);  view_36 = None
        view_37 = torch.ops.aten.view.default(add_12, [128, 128, 128]);  add_12 = None
        amax_1 = torch.ops.aten.amax.default(view_37, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        bmm_3 = torch.ops.aten.bmm.default(div_1, view_35);  div_1 = view_35 = None
        view_38 = torch.ops.aten.view.default(bmm_3, [4, 32, 128, 80]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_13 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39 = torch.ops.aten.view.default(clone_13, [4, 128, 2560]);  clone_13 = None
        view_40 = torch.ops.aten.view.default(view_39, [512, 2560]);  view_39 = None
        permute_19 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg28_1, view_40, permute_19);  arg28_1 = view_40 = permute_19 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [4, 128, 2560]);  addmm_9 = None
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
        view_42 = torch.ops.aten.view.default(add_15, [512, 2560]);  add_15 = None
        permute_20 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg32_1, view_42, permute_20);  arg32_1 = view_42 = permute_20 = None
        view_43 = torch.ops.aten.view.default(addmm_10, [4, 128, 10240]);  addmm_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_16);  mul_14 = add_16 = None
        view_44 = torch.ops.aten.view.default(mul_16, [512, 10240]);  mul_16 = None
        permute_21 = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg34_1, view_44, permute_21);  arg34_1 = view_44 = permute_21 = None
        view_45 = torch.ops.aten.view.default(addmm_11, [4, 128, 2560]);  addmm_11 = None
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
        view_46 = torch.ops.aten.view.default(add_19, [512, 2560])
        permute_22 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg38_1, view_46, permute_22);  arg38_1 = view_46 = permute_22 = None
        view_47 = torch.ops.aten.view.default(addmm_12, [4, 128, 2560]);  addmm_12 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_47, 0.11180339887498948);  view_47 = None
        view_48 = torch.ops.aten.view.default(add_19, [512, 2560])
        permute_23 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg40_1, view_48, permute_23);  arg40_1 = view_48 = permute_23 = None
        view_49 = torch.ops.aten.view.default(addmm_13, [4, 128, 2560]);  addmm_13 = None
        view_50 = torch.ops.aten.view.default(view_49, [4, -1, 32, 80]);  view_49 = None
        permute_24 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_17 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_51 = torch.ops.aten.view.default(add_19, [512, 2560]);  add_19 = None
        permute_25 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg42_1, view_51, permute_25);  arg42_1 = view_51 = permute_25 = None
        view_52 = torch.ops.aten.view.default(addmm_14, [4, 128, 2560]);  addmm_14 = None
        view_53 = torch.ops.aten.view.default(view_52, [4, -1, 32, 80]);  view_52 = None
        permute_26 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_18 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_54 = torch.ops.aten.view.default(mul_19, [4, 128, 32, 80]);  mul_19 = None
        permute_27 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_55 = torch.ops.aten.view.default(clone_19, [128, -1, 80]);  clone_19 = None
        view_56 = torch.ops.aten.view.default(clone_17, [128, -1, 80])
        view_57 = torch.ops.aten.view.default(clone_18, [128, -1, 80])
        permute_28 = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        bmm_4 = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
        view_58 = torch.ops.aten.view.default(bmm_4, [4, 32, 128, 128]);  bmm_4 = None
        add_20 = torch.ops.aten.add.Tensor(view_58, expand_1);  view_58 = None
        view_59 = torch.ops.aten.view.default(add_20, [128, 128, 128]);  add_20 = None
        amax_2 = torch.ops.aten.amax.default(view_59, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        bmm_5 = torch.ops.aten.bmm.default(div_2, view_57);  div_2 = view_57 = None
        view_60 = torch.ops.aten.view.default(bmm_5, [4, 32, 128, 80]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_21 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_61 = torch.ops.aten.view.default(clone_21, [4, 128, 2560]);  clone_21 = None
        view_62 = torch.ops.aten.view.default(view_61, [512, 2560]);  view_61 = None
        permute_30 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg44_1, view_62, permute_30);  arg44_1 = view_62 = permute_30 = None
        view_63 = torch.ops.aten.view.default(addmm_15, [4, 128, 2560]);  addmm_15 = None
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
        view_64 = torch.ops.aten.view.default(add_23, [512, 2560]);  add_23 = None
        permute_31 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg48_1, view_64, permute_31);  arg48_1 = view_64 = permute_31 = None
        view_65 = torch.ops.aten.view.default(addmm_16, [4, 128, 10240]);  addmm_16 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_24);  mul_22 = add_24 = None
        view_66 = torch.ops.aten.view.default(mul_24, [512, 10240]);  mul_24 = None
        permute_32 = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg50_1, view_66, permute_32);  arg50_1 = view_66 = permute_32 = None
        view_67 = torch.ops.aten.view.default(addmm_17, [4, 128, 2560]);  addmm_17 = None
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
        view_68 = torch.ops.aten.view.default(add_27, [512, 2560])
        permute_33 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg54_1, view_68, permute_33);  arg54_1 = view_68 = permute_33 = None
        view_69 = torch.ops.aten.view.default(addmm_18, [4, 128, 2560]);  addmm_18 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_69, 0.11180339887498948);  view_69 = None
        view_70 = torch.ops.aten.view.default(add_27, [512, 2560])
        permute_34 = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg56_1, view_70, permute_34);  arg56_1 = view_70 = permute_34 = None
        view_71 = torch.ops.aten.view.default(addmm_19, [4, 128, 2560]);  addmm_19 = None
        view_72 = torch.ops.aten.view.default(view_71, [4, -1, 32, 80]);  view_71 = None
        permute_35 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_73 = torch.ops.aten.view.default(add_27, [512, 2560]);  add_27 = None
        permute_36 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg58_1, view_73, permute_36);  arg58_1 = view_73 = permute_36 = None
        view_74 = torch.ops.aten.view.default(addmm_20, [4, 128, 2560]);  addmm_20 = None
        view_75 = torch.ops.aten.view.default(view_74, [4, -1, 32, 80]);  view_74 = None
        permute_37 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_26 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_76 = torch.ops.aten.view.default(mul_27, [4, 128, 32, 80]);  mul_27 = None
        permute_38 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_27 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_77 = torch.ops.aten.view.default(clone_27, [128, -1, 80]);  clone_27 = None
        view_78 = torch.ops.aten.view.default(clone_25, [128, -1, 80])
        view_79 = torch.ops.aten.view.default(clone_26, [128, -1, 80])
        permute_39 = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        bmm_6 = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
        view_80 = torch.ops.aten.view.default(bmm_6, [4, 32, 128, 128]);  bmm_6 = None
        add_28 = torch.ops.aten.add.Tensor(view_80, expand_1);  view_80 = None
        view_81 = torch.ops.aten.view.default(add_28, [128, 128, 128]);  add_28 = None
        amax_3 = torch.ops.aten.amax.default(view_81, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        bmm_7 = torch.ops.aten.bmm.default(div_3, view_79);  div_3 = view_79 = None
        view_82 = torch.ops.aten.view.default(bmm_7, [4, 32, 128, 80]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_29 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_83 = torch.ops.aten.view.default(clone_29, [4, 128, 2560]);  clone_29 = None
        view_84 = torch.ops.aten.view.default(view_83, [512, 2560]);  view_83 = None
        permute_41 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg60_1, view_84, permute_41);  arg60_1 = view_84 = permute_41 = None
        view_85 = torch.ops.aten.view.default(addmm_21, [4, 128, 2560]);  addmm_21 = None
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
        view_86 = torch.ops.aten.view.default(add_31, [512, 2560]);  add_31 = None
        permute_42 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg64_1, view_86, permute_42);  arg64_1 = view_86 = permute_42 = None
        view_87 = torch.ops.aten.view.default(addmm_22, [4, 128, 10240]);  addmm_22 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_32);  mul_30 = add_32 = None
        view_88 = torch.ops.aten.view.default(mul_32, [512, 10240]);  mul_32 = None
        permute_43 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg66_1, view_88, permute_43);  arg66_1 = view_88 = permute_43 = None
        view_89 = torch.ops.aten.view.default(addmm_23, [4, 128, 2560]);  addmm_23 = None
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
        view_90 = torch.ops.aten.view.default(add_35, [512, 2560])
        permute_44 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg70_1, view_90, permute_44);  arg70_1 = view_90 = permute_44 = None
        view_91 = torch.ops.aten.view.default(addmm_24, [4, 128, 2560]);  addmm_24 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_91, 0.11180339887498948);  view_91 = None
        view_92 = torch.ops.aten.view.default(add_35, [512, 2560])
        permute_45 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg72_1, view_92, permute_45);  arg72_1 = view_92 = permute_45 = None
        view_93 = torch.ops.aten.view.default(addmm_25, [4, 128, 2560]);  addmm_25 = None
        view_94 = torch.ops.aten.view.default(view_93, [4, -1, 32, 80]);  view_93 = None
        permute_46 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_33 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_95 = torch.ops.aten.view.default(add_35, [512, 2560]);  add_35 = None
        permute_47 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg74_1, view_95, permute_47);  arg74_1 = view_95 = permute_47 = None
        view_96 = torch.ops.aten.view.default(addmm_26, [4, 128, 2560]);  addmm_26 = None
        view_97 = torch.ops.aten.view.default(view_96, [4, -1, 32, 80]);  view_96 = None
        permute_48 = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        clone_34 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_98 = torch.ops.aten.view.default(mul_35, [4, 128, 32, 80]);  mul_35 = None
        permute_49 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_35 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_99 = torch.ops.aten.view.default(clone_35, [128, -1, 80]);  clone_35 = None
        view_100 = torch.ops.aten.view.default(clone_33, [128, -1, 80])
        view_101 = torch.ops.aten.view.default(clone_34, [128, -1, 80])
        permute_50 = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        bmm_8 = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
        view_102 = torch.ops.aten.view.default(bmm_8, [4, 32, 128, 128]);  bmm_8 = None
        add_36 = torch.ops.aten.add.Tensor(view_102, expand_1);  view_102 = None
        view_103 = torch.ops.aten.view.default(add_36, [128, 128, 128]);  add_36 = None
        amax_4 = torch.ops.aten.amax.default(view_103, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        bmm_9 = torch.ops.aten.bmm.default(div_4, view_101);  div_4 = view_101 = None
        view_104 = torch.ops.aten.view.default(bmm_9, [4, 32, 128, 80]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_37 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_105 = torch.ops.aten.view.default(clone_37, [4, 128, 2560]);  clone_37 = None
        view_106 = torch.ops.aten.view.default(view_105, [512, 2560]);  view_105 = None
        permute_52 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg76_1, view_106, permute_52);  arg76_1 = view_106 = permute_52 = None
        view_107 = torch.ops.aten.view.default(addmm_27, [4, 128, 2560]);  addmm_27 = None
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
        view_108 = torch.ops.aten.view.default(add_39, [512, 2560]);  add_39 = None
        permute_53 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg80_1, view_108, permute_53);  arg80_1 = view_108 = permute_53 = None
        view_109 = torch.ops.aten.view.default(addmm_28, [4, 128, 10240]);  addmm_28 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_39 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4 = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, add_40);  mul_38 = add_40 = None
        view_110 = torch.ops.aten.view.default(mul_40, [512, 10240]);  mul_40 = None
        permute_54 = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg82_1, view_110, permute_54);  arg82_1 = view_110 = permute_54 = None
        view_111 = torch.ops.aten.view.default(addmm_29, [4, 128, 2560]);  addmm_29 = None
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
        view_112 = torch.ops.aten.view.default(add_43, [512, 2560])
        permute_55 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg86_1, view_112, permute_55);  arg86_1 = view_112 = permute_55 = None
        view_113 = torch.ops.aten.view.default(addmm_30, [4, 128, 2560]);  addmm_30 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_113, 0.11180339887498948);  view_113 = None
        view_114 = torch.ops.aten.view.default(add_43, [512, 2560])
        permute_56 = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg88_1, view_114, permute_56);  arg88_1 = view_114 = permute_56 = None
        view_115 = torch.ops.aten.view.default(addmm_31, [4, 128, 2560]);  addmm_31 = None
        view_116 = torch.ops.aten.view.default(view_115, [4, -1, 32, 80]);  view_115 = None
        permute_57 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_41 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_117 = torch.ops.aten.view.default(add_43, [512, 2560]);  add_43 = None
        permute_58 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg90_1, view_117, permute_58);  arg90_1 = view_117 = permute_58 = None
        view_118 = torch.ops.aten.view.default(addmm_32, [4, 128, 2560]);  addmm_32 = None
        view_119 = torch.ops.aten.view.default(view_118, [4, -1, 32, 80]);  view_118 = None
        permute_59 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        clone_42 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_120 = torch.ops.aten.view.default(mul_43, [4, 128, 32, 80]);  mul_43 = None
        permute_60 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_43 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_121 = torch.ops.aten.view.default(clone_43, [128, -1, 80]);  clone_43 = None
        view_122 = torch.ops.aten.view.default(clone_41, [128, -1, 80])
        view_123 = torch.ops.aten.view.default(clone_42, [128, -1, 80])
        permute_61 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        bmm_10 = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
        view_124 = torch.ops.aten.view.default(bmm_10, [4, 32, 128, 128]);  bmm_10 = None
        add_44 = torch.ops.aten.add.Tensor(view_124, expand_1);  view_124 = None
        view_125 = torch.ops.aten.view.default(add_44, [128, 128, 128]);  add_44 = None
        amax_5 = torch.ops.aten.amax.default(view_125, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        bmm_11 = torch.ops.aten.bmm.default(div_5, view_123);  div_5 = view_123 = None
        view_126 = torch.ops.aten.view.default(bmm_11, [4, 32, 128, 80]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_45 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_127 = torch.ops.aten.view.default(clone_45, [4, 128, 2560]);  clone_45 = None
        view_128 = torch.ops.aten.view.default(view_127, [512, 2560]);  view_127 = None
        permute_63 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg92_1, view_128, permute_63);  arg92_1 = view_128 = permute_63 = None
        view_129 = torch.ops.aten.view.default(addmm_33, [4, 128, 2560]);  addmm_33 = None
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
        view_130 = torch.ops.aten.view.default(add_47, [512, 2560]);  add_47 = None
        permute_64 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg96_1, view_130, permute_64);  arg96_1 = view_130 = permute_64 = None
        view_131 = torch.ops.aten.view.default(addmm_34, [4, 128, 10240]);  addmm_34 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
        view_132 = torch.ops.aten.view.default(mul_48, [512, 10240]);  mul_48 = None
        permute_65 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg98_1, view_132, permute_65);  arg98_1 = view_132 = permute_65 = None
        view_133 = torch.ops.aten.view.default(addmm_35, [4, 128, 2560]);  addmm_35 = None
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
        view_134 = torch.ops.aten.view.default(add_51, [512, 2560])
        permute_66 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg102_1, view_134, permute_66);  arg102_1 = view_134 = permute_66 = None
        view_135 = torch.ops.aten.view.default(addmm_36, [4, 128, 2560]);  addmm_36 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_135, 0.11180339887498948);  view_135 = None
        view_136 = torch.ops.aten.view.default(add_51, [512, 2560])
        permute_67 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg104_1, view_136, permute_67);  arg104_1 = view_136 = permute_67 = None
        view_137 = torch.ops.aten.view.default(addmm_37, [4, 128, 2560]);  addmm_37 = None
        view_138 = torch.ops.aten.view.default(view_137, [4, -1, 32, 80]);  view_137 = None
        permute_68 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_49 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_139 = torch.ops.aten.view.default(add_51, [512, 2560]);  add_51 = None
        permute_69 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg106_1, view_139, permute_69);  arg106_1 = view_139 = permute_69 = None
        view_140 = torch.ops.aten.view.default(addmm_38, [4, 128, 2560]);  addmm_38 = None
        view_141 = torch.ops.aten.view.default(view_140, [4, -1, 32, 80]);  view_140 = None
        permute_70 = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_50 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_142 = torch.ops.aten.view.default(mul_51, [4, 128, 32, 80]);  mul_51 = None
        permute_71 = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_51 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_143 = torch.ops.aten.view.default(clone_51, [128, -1, 80]);  clone_51 = None
        view_144 = torch.ops.aten.view.default(clone_49, [128, -1, 80])
        view_145 = torch.ops.aten.view.default(clone_50, [128, -1, 80])
        permute_72 = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        bmm_12 = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
        view_146 = torch.ops.aten.view.default(bmm_12, [4, 32, 128, 128]);  bmm_12 = None
        add_52 = torch.ops.aten.add.Tensor(view_146, expand_1);  view_146 = None
        view_147 = torch.ops.aten.view.default(add_52, [128, 128, 128]);  add_52 = None
        amax_6 = torch.ops.aten.amax.default(view_147, [-1], True)
        sub_19 = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        bmm_13 = torch.ops.aten.bmm.default(div_6, view_145);  div_6 = view_145 = None
        view_148 = torch.ops.aten.view.default(bmm_13, [4, 32, 128, 80]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_53 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_149 = torch.ops.aten.view.default(clone_53, [4, 128, 2560]);  clone_53 = None
        view_150 = torch.ops.aten.view.default(view_149, [512, 2560]);  view_149 = None
        permute_74 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg108_1, view_150, permute_74);  arg108_1 = view_150 = permute_74 = None
        view_151 = torch.ops.aten.view.default(addmm_39, [4, 128, 2560]);  addmm_39 = None
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
        view_152 = torch.ops.aten.view.default(add_55, [512, 2560]);  add_55 = None
        permute_75 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg112_1, view_152, permute_75);  arg112_1 = view_152 = permute_75 = None
        view_153 = torch.ops.aten.view.default(addmm_40, [4, 128, 10240]);  addmm_40 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_56);  mul_54 = add_56 = None
        view_154 = torch.ops.aten.view.default(mul_56, [512, 10240]);  mul_56 = None
        permute_76 = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg114_1, view_154, permute_76);  arg114_1 = view_154 = permute_76 = None
        view_155 = torch.ops.aten.view.default(addmm_41, [4, 128, 2560]);  addmm_41 = None
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
        view_156 = torch.ops.aten.view.default(add_59, [512, 2560])
        permute_77 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg118_1, view_156, permute_77);  arg118_1 = view_156 = permute_77 = None
        view_157 = torch.ops.aten.view.default(addmm_42, [4, 128, 2560]);  addmm_42 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_157, 0.11180339887498948);  view_157 = None
        view_158 = torch.ops.aten.view.default(add_59, [512, 2560])
        permute_78 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg120_1, view_158, permute_78);  arg120_1 = view_158 = permute_78 = None
        view_159 = torch.ops.aten.view.default(addmm_43, [4, 128, 2560]);  addmm_43 = None
        view_160 = torch.ops.aten.view.default(view_159, [4, -1, 32, 80]);  view_159 = None
        permute_79 = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_57 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_161 = torch.ops.aten.view.default(add_59, [512, 2560]);  add_59 = None
        permute_80 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg122_1, view_161, permute_80);  arg122_1 = view_161 = permute_80 = None
        view_162 = torch.ops.aten.view.default(addmm_44, [4, 128, 2560]);  addmm_44 = None
        view_163 = torch.ops.aten.view.default(view_162, [4, -1, 32, 80]);  view_162 = None
        permute_81 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_58 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_164 = torch.ops.aten.view.default(mul_59, [4, 128, 32, 80]);  mul_59 = None
        permute_82 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_59 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_165 = torch.ops.aten.view.default(clone_59, [128, -1, 80]);  clone_59 = None
        view_166 = torch.ops.aten.view.default(clone_57, [128, -1, 80])
        view_167 = torch.ops.aten.view.default(clone_58, [128, -1, 80])
        permute_83 = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        bmm_14 = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
        view_168 = torch.ops.aten.view.default(bmm_14, [4, 32, 128, 128]);  bmm_14 = None
        add_60 = torch.ops.aten.add.Tensor(view_168, expand_1);  view_168 = None
        view_169 = torch.ops.aten.view.default(add_60, [128, 128, 128]);  add_60 = None
        amax_7 = torch.ops.aten.amax.default(view_169, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        bmm_15 = torch.ops.aten.bmm.default(div_7, view_167);  div_7 = view_167 = None
        view_170 = torch.ops.aten.view.default(bmm_15, [4, 32, 128, 80]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_61 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_171 = torch.ops.aten.view.default(clone_61, [4, 128, 2560]);  clone_61 = None
        view_172 = torch.ops.aten.view.default(view_171, [512, 2560]);  view_171 = None
        permute_85 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg124_1, view_172, permute_85);  arg124_1 = view_172 = permute_85 = None
        view_173 = torch.ops.aten.view.default(addmm_45, [4, 128, 2560]);  addmm_45 = None
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
        view_174 = torch.ops.aten.view.default(add_63, [512, 2560]);  add_63 = None
        permute_86 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg128_1, view_174, permute_86);  arg128_1 = view_174 = permute_86 = None
        view_175 = torch.ops.aten.view.default(addmm_46, [4, 128, 10240]);  addmm_46 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_63 = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7 = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_62, add_64);  mul_62 = add_64 = None
        view_176 = torch.ops.aten.view.default(mul_64, [512, 10240]);  mul_64 = None
        permute_87 = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg130_1, view_176, permute_87);  arg130_1 = view_176 = permute_87 = None
        view_177 = torch.ops.aten.view.default(addmm_47, [4, 128, 2560]);  addmm_47 = None
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
        view_178 = torch.ops.aten.view.default(add_67, [512, 2560])
        permute_88 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg134_1, view_178, permute_88);  arg134_1 = view_178 = permute_88 = None
        view_179 = torch.ops.aten.view.default(addmm_48, [4, 128, 2560]);  addmm_48 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_179, 0.11180339887498948);  view_179 = None
        view_180 = torch.ops.aten.view.default(add_67, [512, 2560])
        permute_89 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg136_1, view_180, permute_89);  arg136_1 = view_180 = permute_89 = None
        view_181 = torch.ops.aten.view.default(addmm_49, [4, 128, 2560]);  addmm_49 = None
        view_182 = torch.ops.aten.view.default(view_181, [4, -1, 32, 80]);  view_181 = None
        permute_90 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_65 = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_183 = torch.ops.aten.view.default(add_67, [512, 2560]);  add_67 = None
        permute_91 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg138_1, view_183, permute_91);  arg138_1 = view_183 = permute_91 = None
        view_184 = torch.ops.aten.view.default(addmm_50, [4, 128, 2560]);  addmm_50 = None
        view_185 = torch.ops.aten.view.default(view_184, [4, -1, 32, 80]);  view_184 = None
        permute_92 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_66 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_186 = torch.ops.aten.view.default(mul_67, [4, 128, 32, 80]);  mul_67 = None
        permute_93 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_67 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_187 = torch.ops.aten.view.default(clone_67, [128, -1, 80]);  clone_67 = None
        view_188 = torch.ops.aten.view.default(clone_65, [128, -1, 80])
        view_189 = torch.ops.aten.view.default(clone_66, [128, -1, 80])
        permute_94 = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        bmm_16 = torch.ops.aten.bmm.default(view_187, permute_94);  view_187 = permute_94 = None
        view_190 = torch.ops.aten.view.default(bmm_16, [4, 32, 128, 128]);  bmm_16 = None
        add_68 = torch.ops.aten.add.Tensor(view_190, expand_1);  view_190 = None
        view_191 = torch.ops.aten.view.default(add_68, [128, 128, 128]);  add_68 = None
        amax_8 = torch.ops.aten.amax.default(view_191, [-1], True)
        sub_25 = torch.ops.aten.sub.Tensor(view_191, amax_8);  view_191 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        bmm_17 = torch.ops.aten.bmm.default(div_8, view_189);  div_8 = view_189 = None
        view_192 = torch.ops.aten.view.default(bmm_17, [4, 32, 128, 80]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_69 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_193 = torch.ops.aten.view.default(clone_69, [4, 128, 2560]);  clone_69 = None
        view_194 = torch.ops.aten.view.default(view_193, [512, 2560]);  view_193 = None
        permute_96 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg140_1, view_194, permute_96);  arg140_1 = view_194 = permute_96 = None
        view_195 = torch.ops.aten.view.default(addmm_51, [4, 128, 2560]);  addmm_51 = None
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
        view_196 = torch.ops.aten.view.default(add_71, [512, 2560]);  add_71 = None
        permute_97 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg144_1, view_196, permute_97);  arg144_1 = view_196 = permute_97 = None
        view_197 = torch.ops.aten.view.default(addmm_52, [4, 128, 10240]);  addmm_52 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_71 = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_8 = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_70, add_72);  mul_70 = add_72 = None
        view_198 = torch.ops.aten.view.default(mul_72, [512, 10240]);  mul_72 = None
        permute_98 = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg146_1, view_198, permute_98);  arg146_1 = view_198 = permute_98 = None
        view_199 = torch.ops.aten.view.default(addmm_53, [4, 128, 2560]);  addmm_53 = None
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
        view_200 = torch.ops.aten.view.default(add_75, [512, 2560])
        permute_99 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg150_1, view_200, permute_99);  arg150_1 = view_200 = permute_99 = None
        view_201 = torch.ops.aten.view.default(addmm_54, [4, 128, 2560]);  addmm_54 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_201, 0.11180339887498948);  view_201 = None
        view_202 = torch.ops.aten.view.default(add_75, [512, 2560])
        permute_100 = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg152_1, view_202, permute_100);  arg152_1 = view_202 = permute_100 = None
        view_203 = torch.ops.aten.view.default(addmm_55, [4, 128, 2560]);  addmm_55 = None
        view_204 = torch.ops.aten.view.default(view_203, [4, -1, 32, 80]);  view_203 = None
        permute_101 = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        clone_73 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_205 = torch.ops.aten.view.default(add_75, [512, 2560]);  add_75 = None
        permute_102 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg154_1, view_205, permute_102);  arg154_1 = view_205 = permute_102 = None
        view_206 = torch.ops.aten.view.default(addmm_56, [4, 128, 2560]);  addmm_56 = None
        view_207 = torch.ops.aten.view.default(view_206, [4, -1, 32, 80]);  view_206 = None
        permute_103 = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        clone_74 = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_208 = torch.ops.aten.view.default(mul_75, [4, 128, 32, 80]);  mul_75 = None
        permute_104 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_75 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_209 = torch.ops.aten.view.default(clone_75, [128, -1, 80]);  clone_75 = None
        view_210 = torch.ops.aten.view.default(clone_73, [128, -1, 80])
        view_211 = torch.ops.aten.view.default(clone_74, [128, -1, 80])
        permute_105 = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        bmm_18 = torch.ops.aten.bmm.default(view_209, permute_105);  view_209 = permute_105 = None
        view_212 = torch.ops.aten.view.default(bmm_18, [4, 32, 128, 128]);  bmm_18 = None
        add_76 = torch.ops.aten.add.Tensor(view_212, expand_1);  view_212 = None
        view_213 = torch.ops.aten.view.default(add_76, [128, 128, 128]);  add_76 = None
        amax_9 = torch.ops.aten.amax.default(view_213, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(view_213, amax_9);  view_213 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        bmm_19 = torch.ops.aten.bmm.default(div_9, view_211);  div_9 = view_211 = None
        view_214 = torch.ops.aten.view.default(bmm_19, [4, 32, 128, 80]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_77 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_215 = torch.ops.aten.view.default(clone_77, [4, 128, 2560]);  clone_77 = None
        view_216 = torch.ops.aten.view.default(view_215, [512, 2560]);  view_215 = None
        permute_107 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg156_1, view_216, permute_107);  arg156_1 = view_216 = permute_107 = None
        view_217 = torch.ops.aten.view.default(addmm_57, [4, 128, 2560]);  addmm_57 = None
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
        view_218 = torch.ops.aten.view.default(add_79, [512, 2560]);  add_79 = None
        permute_108 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg160_1, view_218, permute_108);  arg160_1 = view_218 = permute_108 = None
        view_219 = torch.ops.aten.view.default(addmm_58, [4, 128, 10240]);  addmm_58 = None
        mul_78 = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_79 = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_9 = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_78, add_80);  mul_78 = add_80 = None
        view_220 = torch.ops.aten.view.default(mul_80, [512, 10240]);  mul_80 = None
        permute_109 = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg162_1, view_220, permute_109);  arg162_1 = view_220 = permute_109 = None
        view_221 = torch.ops.aten.view.default(addmm_59, [4, 128, 2560]);  addmm_59 = None
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
        view_222 = torch.ops.aten.view.default(add_83, [512, 2560])
        permute_110 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg166_1, view_222, permute_110);  arg166_1 = view_222 = permute_110 = None
        view_223 = torch.ops.aten.view.default(addmm_60, [4, 128, 2560]);  addmm_60 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_223, 0.11180339887498948);  view_223 = None
        view_224 = torch.ops.aten.view.default(add_83, [512, 2560])
        permute_111 = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg168_1, view_224, permute_111);  arg168_1 = view_224 = permute_111 = None
        view_225 = torch.ops.aten.view.default(addmm_61, [4, 128, 2560]);  addmm_61 = None
        view_226 = torch.ops.aten.view.default(view_225, [4, -1, 32, 80]);  view_225 = None
        permute_112 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_81 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_227 = torch.ops.aten.view.default(add_83, [512, 2560]);  add_83 = None
        permute_113 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg170_1, view_227, permute_113);  arg170_1 = view_227 = permute_113 = None
        view_228 = torch.ops.aten.view.default(addmm_62, [4, 128, 2560]);  addmm_62 = None
        view_229 = torch.ops.aten.view.default(view_228, [4, -1, 32, 80]);  view_228 = None
        permute_114 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_82 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_230 = torch.ops.aten.view.default(mul_83, [4, 128, 32, 80]);  mul_83 = None
        permute_115 = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_83 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_231 = torch.ops.aten.view.default(clone_83, [128, -1, 80]);  clone_83 = None
        view_232 = torch.ops.aten.view.default(clone_81, [128, -1, 80])
        view_233 = torch.ops.aten.view.default(clone_82, [128, -1, 80])
        permute_116 = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        bmm_20 = torch.ops.aten.bmm.default(view_231, permute_116);  view_231 = permute_116 = None
        view_234 = torch.ops.aten.view.default(bmm_20, [4, 32, 128, 128]);  bmm_20 = None
        add_84 = torch.ops.aten.add.Tensor(view_234, expand_1);  view_234 = None
        view_235 = torch.ops.aten.view.default(add_84, [128, 128, 128]);  add_84 = None
        amax_10 = torch.ops.aten.amax.default(view_235, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(view_235, amax_10);  view_235 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        bmm_21 = torch.ops.aten.bmm.default(div_10, view_233);  div_10 = view_233 = None
        view_236 = torch.ops.aten.view.default(bmm_21, [4, 32, 128, 80]);  bmm_21 = None
        permute_117 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_85 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_237 = torch.ops.aten.view.default(clone_85, [4, 128, 2560]);  clone_85 = None
        view_238 = torch.ops.aten.view.default(view_237, [512, 2560]);  view_237 = None
        permute_118 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg172_1, view_238, permute_118);  arg172_1 = view_238 = permute_118 = None
        view_239 = torch.ops.aten.view.default(addmm_63, [4, 128, 2560]);  addmm_63 = None
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
        view_240 = torch.ops.aten.view.default(add_87, [512, 2560]);  add_87 = None
        permute_119 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg176_1, view_240, permute_119);  arg176_1 = view_240 = permute_119 = None
        view_241 = torch.ops.aten.view.default(addmm_64, [4, 128, 10240]);  addmm_64 = None
        mul_86 = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_87 = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
        erf_10 = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_86, add_88);  mul_86 = add_88 = None
        view_242 = torch.ops.aten.view.default(mul_88, [512, 10240]);  mul_88 = None
        permute_120 = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg178_1, view_242, permute_120);  arg178_1 = view_242 = permute_120 = None
        view_243 = torch.ops.aten.view.default(addmm_65, [4, 128, 2560]);  addmm_65 = None
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
        view_244 = torch.ops.aten.view.default(add_91, [512, 2560])
        permute_121 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg182_1, view_244, permute_121);  arg182_1 = view_244 = permute_121 = None
        view_245 = torch.ops.aten.view.default(addmm_66, [4, 128, 2560]);  addmm_66 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_245, 0.11180339887498948);  view_245 = None
        view_246 = torch.ops.aten.view.default(add_91, [512, 2560])
        permute_122 = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg184_1, view_246, permute_122);  arg184_1 = view_246 = permute_122 = None
        view_247 = torch.ops.aten.view.default(addmm_67, [4, 128, 2560]);  addmm_67 = None
        view_248 = torch.ops.aten.view.default(view_247, [4, -1, 32, 80]);  view_247 = None
        permute_123 = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_89 = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        view_249 = torch.ops.aten.view.default(add_91, [512, 2560]);  add_91 = None
        permute_124 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg186_1, view_249, permute_124);  arg186_1 = view_249 = permute_124 = None
        view_250 = torch.ops.aten.view.default(addmm_68, [4, 128, 2560]);  addmm_68 = None
        view_251 = torch.ops.aten.view.default(view_250, [4, -1, 32, 80]);  view_250 = None
        permute_125 = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        clone_90 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_252 = torch.ops.aten.view.default(mul_91, [4, 128, 32, 80]);  mul_91 = None
        permute_126 = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_91 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_253 = torch.ops.aten.view.default(clone_91, [128, -1, 80]);  clone_91 = None
        view_254 = torch.ops.aten.view.default(clone_89, [128, -1, 80])
        view_255 = torch.ops.aten.view.default(clone_90, [128, -1, 80])
        permute_127 = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        bmm_22 = torch.ops.aten.bmm.default(view_253, permute_127);  view_253 = permute_127 = None
        view_256 = torch.ops.aten.view.default(bmm_22, [4, 32, 128, 128]);  bmm_22 = None
        add_92 = torch.ops.aten.add.Tensor(view_256, expand_1);  view_256 = None
        view_257 = torch.ops.aten.view.default(add_92, [128, 128, 128]);  add_92 = None
        amax_11 = torch.ops.aten.amax.default(view_257, [-1], True)
        sub_34 = torch.ops.aten.sub.Tensor(view_257, amax_11);  view_257 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        bmm_23 = torch.ops.aten.bmm.default(div_11, view_255);  div_11 = view_255 = None
        view_258 = torch.ops.aten.view.default(bmm_23, [4, 32, 128, 80]);  bmm_23 = None
        permute_128 = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_93 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_259 = torch.ops.aten.view.default(clone_93, [4, 128, 2560]);  clone_93 = None
        view_260 = torch.ops.aten.view.default(view_259, [512, 2560]);  view_259 = None
        permute_129 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg188_1, view_260, permute_129);  arg188_1 = view_260 = permute_129 = None
        view_261 = torch.ops.aten.view.default(addmm_69, [4, 128, 2560]);  addmm_69 = None
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
        view_262 = torch.ops.aten.view.default(add_95, [512, 2560]);  add_95 = None
        permute_130 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg192_1, view_262, permute_130);  arg192_1 = view_262 = permute_130 = None
        view_263 = torch.ops.aten.view.default(addmm_70, [4, 128, 10240]);  addmm_70 = None
        mul_94 = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_95 = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_11 = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_94, add_96);  mul_94 = add_96 = None
        view_264 = torch.ops.aten.view.default(mul_96, [512, 10240]);  mul_96 = None
        permute_131 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg194_1, view_264, permute_131);  arg194_1 = view_264 = permute_131 = None
        view_265 = torch.ops.aten.view.default(addmm_71, [4, 128, 2560]);  addmm_71 = None
        add_97 = torch.ops.aten.add.Tensor(add_93, view_265);  add_93 = view_265 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  getitem_49 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        view_266 = torch.ops.aten.view.default(add_99, [512, 2560])
        permute_132 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg198_1, view_266, permute_132);  arg198_1 = view_266 = permute_132 = None
        view_267 = torch.ops.aten.view.default(addmm_72, [4, 128, 2560]);  addmm_72 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_267, 0.11180339887498948);  view_267 = None
        view_268 = torch.ops.aten.view.default(add_99, [512, 2560])
        permute_133 = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg200_1, view_268, permute_133);  arg200_1 = view_268 = permute_133 = None
        view_269 = torch.ops.aten.view.default(addmm_73, [4, 128, 2560]);  addmm_73 = None
        view_270 = torch.ops.aten.view.default(view_269, [4, -1, 32, 80]);  view_269 = None
        permute_134 = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_97 = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
        view_271 = torch.ops.aten.view.default(add_99, [512, 2560]);  add_99 = None
        permute_135 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg202_1, view_271, permute_135);  arg202_1 = view_271 = permute_135 = None
        view_272 = torch.ops.aten.view.default(addmm_74, [4, 128, 2560]);  addmm_74 = None
        view_273 = torch.ops.aten.view.default(view_272, [4, -1, 32, 80]);  view_272 = None
        permute_136 = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
        clone_98 = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_274 = torch.ops.aten.view.default(mul_99, [4, 128, 32, 80]);  mul_99 = None
        permute_137 = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_99 = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_275 = torch.ops.aten.view.default(clone_99, [128, -1, 80]);  clone_99 = None
        view_276 = torch.ops.aten.view.default(clone_97, [128, -1, 80])
        view_277 = torch.ops.aten.view.default(clone_98, [128, -1, 80])
        permute_138 = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
        bmm_24 = torch.ops.aten.bmm.default(view_275, permute_138);  view_275 = permute_138 = None
        view_278 = torch.ops.aten.view.default(bmm_24, [4, 32, 128, 128]);  bmm_24 = None
        add_100 = torch.ops.aten.add.Tensor(view_278, expand_1);  view_278 = None
        view_279 = torch.ops.aten.view.default(add_100, [128, 128, 128]);  add_100 = None
        amax_12 = torch.ops.aten.amax.default(view_279, [-1], True)
        sub_37 = torch.ops.aten.sub.Tensor(view_279, amax_12);  view_279 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        bmm_25 = torch.ops.aten.bmm.default(div_12, view_277);  div_12 = view_277 = None
        view_280 = torch.ops.aten.view.default(bmm_25, [4, 32, 128, 80]);  bmm_25 = None
        permute_139 = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        clone_101 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_281 = torch.ops.aten.view.default(clone_101, [4, 128, 2560]);  clone_101 = None
        view_282 = torch.ops.aten.view.default(view_281, [512, 2560]);  view_281 = None
        permute_140 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg204_1, view_282, permute_140);  arg204_1 = view_282 = permute_140 = None
        view_283 = torch.ops.aten.view.default(addmm_75, [4, 128, 2560]);  addmm_75 = None
        add_101 = torch.ops.aten.add.Tensor(add_97, view_283);  add_97 = view_283 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_101, getitem_51);  getitem_51 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, arg205_1);  mul_100 = arg205_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_101, arg206_1);  mul_101 = arg206_1 = None
        view_284 = torch.ops.aten.view.default(add_103, [512, 2560]);  add_103 = None
        permute_141 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg208_1, view_284, permute_141);  arg208_1 = view_284 = permute_141 = None
        view_285 = torch.ops.aten.view.default(addmm_76, [4, 128, 10240]);  addmm_76 = None
        mul_102 = torch.ops.aten.mul.Tensor(view_285, 0.5)
        mul_103 = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
        erf_12 = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_104 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_102, add_104);  mul_102 = add_104 = None
        view_286 = torch.ops.aten.view.default(mul_104, [512, 10240]);  mul_104 = None
        permute_142 = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg210_1, view_286, permute_142);  arg210_1 = view_286 = permute_142 = None
        view_287 = torch.ops.aten.view.default(addmm_77, [4, 128, 2560]);  addmm_77 = None
        add_105 = torch.ops.aten.add.Tensor(add_101, view_287);  add_101 = view_287 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_105, getitem_53);  getitem_53 = None
        mul_105 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_105, arg211_1);  mul_105 = arg211_1 = None
        add_107 = torch.ops.aten.add.Tensor(mul_106, arg212_1);  mul_106 = arg212_1 = None
        view_288 = torch.ops.aten.view.default(add_107, [512, 2560])
        permute_143 = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg214_1, view_288, permute_143);  arg214_1 = view_288 = permute_143 = None
        view_289 = torch.ops.aten.view.default(addmm_78, [4, 128, 2560]);  addmm_78 = None
        mul_107 = torch.ops.aten.mul.Tensor(view_289, 0.11180339887498948);  view_289 = None
        view_290 = torch.ops.aten.view.default(add_107, [512, 2560])
        permute_144 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg216_1, view_290, permute_144);  arg216_1 = view_290 = permute_144 = None
        view_291 = torch.ops.aten.view.default(addmm_79, [4, 128, 2560]);  addmm_79 = None
        view_292 = torch.ops.aten.view.default(view_291, [4, -1, 32, 80]);  view_291 = None
        permute_145 = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_105 = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_293 = torch.ops.aten.view.default(add_107, [512, 2560]);  add_107 = None
        permute_146 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg218_1, view_293, permute_146);  arg218_1 = view_293 = permute_146 = None
        view_294 = torch.ops.aten.view.default(addmm_80, [4, 128, 2560]);  addmm_80 = None
        view_295 = torch.ops.aten.view.default(view_294, [4, -1, 32, 80]);  view_294 = None
        permute_147 = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
        clone_106 = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_296 = torch.ops.aten.view.default(mul_107, [4, 128, 32, 80]);  mul_107 = None
        permute_148 = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_107 = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_297 = torch.ops.aten.view.default(clone_107, [128, -1, 80]);  clone_107 = None
        view_298 = torch.ops.aten.view.default(clone_105, [128, -1, 80])
        view_299 = torch.ops.aten.view.default(clone_106, [128, -1, 80])
        permute_149 = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        bmm_26 = torch.ops.aten.bmm.default(view_297, permute_149);  view_297 = permute_149 = None
        view_300 = torch.ops.aten.view.default(bmm_26, [4, 32, 128, 128]);  bmm_26 = None
        add_108 = torch.ops.aten.add.Tensor(view_300, expand_1);  view_300 = None
        view_301 = torch.ops.aten.view.default(add_108, [128, 128, 128]);  add_108 = None
        amax_13 = torch.ops.aten.amax.default(view_301, [-1], True)
        sub_40 = torch.ops.aten.sub.Tensor(view_301, amax_13);  view_301 = amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        bmm_27 = torch.ops.aten.bmm.default(div_13, view_299);  div_13 = view_299 = None
        view_302 = torch.ops.aten.view.default(bmm_27, [4, 32, 128, 80]);  bmm_27 = None
        permute_150 = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_109 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_303 = torch.ops.aten.view.default(clone_109, [4, 128, 2560]);  clone_109 = None
        view_304 = torch.ops.aten.view.default(view_303, [512, 2560]);  view_303 = None
        permute_151 = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg220_1, view_304, permute_151);  arg220_1 = view_304 = permute_151 = None
        view_305 = torch.ops.aten.view.default(addmm_81, [4, 128, 2560]);  addmm_81 = None
        add_109 = torch.ops.aten.add.Tensor(add_105, view_305);  add_105 = view_305 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_109, getitem_55);  getitem_55 = None
        mul_108 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, arg221_1);  mul_108 = arg221_1 = None
        add_111 = torch.ops.aten.add.Tensor(mul_109, arg222_1);  mul_109 = arg222_1 = None
        view_306 = torch.ops.aten.view.default(add_111, [512, 2560]);  add_111 = None
        permute_152 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg224_1, view_306, permute_152);  arg224_1 = view_306 = permute_152 = None
        view_307 = torch.ops.aten.view.default(addmm_82, [4, 128, 10240]);  addmm_82 = None
        mul_110 = torch.ops.aten.mul.Tensor(view_307, 0.5)
        mul_111 = torch.ops.aten.mul.Tensor(view_307, 0.7071067811865476);  view_307 = None
        erf_13 = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_112 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_110, add_112);  mul_110 = add_112 = None
        view_308 = torch.ops.aten.view.default(mul_112, [512, 10240]);  mul_112 = None
        permute_153 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg226_1, view_308, permute_153);  arg226_1 = view_308 = permute_153 = None
        view_309 = torch.ops.aten.view.default(addmm_83, [4, 128, 2560]);  addmm_83 = None
        add_113 = torch.ops.aten.add.Tensor(add_109, view_309);  add_109 = view_309 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_113, getitem_57);  getitem_57 = None
        mul_113 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, arg227_1);  mul_113 = arg227_1 = None
        add_115 = torch.ops.aten.add.Tensor(mul_114, arg228_1);  mul_114 = arg228_1 = None
        view_310 = torch.ops.aten.view.default(add_115, [512, 2560])
        permute_154 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg230_1, view_310, permute_154);  arg230_1 = view_310 = permute_154 = None
        view_311 = torch.ops.aten.view.default(addmm_84, [4, 128, 2560]);  addmm_84 = None
        mul_115 = torch.ops.aten.mul.Tensor(view_311, 0.11180339887498948);  view_311 = None
        view_312 = torch.ops.aten.view.default(add_115, [512, 2560])
        permute_155 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg232_1, view_312, permute_155);  arg232_1 = view_312 = permute_155 = None
        view_313 = torch.ops.aten.view.default(addmm_85, [4, 128, 2560]);  addmm_85 = None
        view_314 = torch.ops.aten.view.default(view_313, [4, -1, 32, 80]);  view_313 = None
        permute_156 = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_113 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_315 = torch.ops.aten.view.default(add_115, [512, 2560]);  add_115 = None
        permute_157 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg234_1, view_315, permute_157);  arg234_1 = view_315 = permute_157 = None
        view_316 = torch.ops.aten.view.default(addmm_86, [4, 128, 2560]);  addmm_86 = None
        view_317 = torch.ops.aten.view.default(view_316, [4, -1, 32, 80]);  view_316 = None
        permute_158 = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
        clone_114 = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_318 = torch.ops.aten.view.default(mul_115, [4, 128, 32, 80]);  mul_115 = None
        permute_159 = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_115 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_319 = torch.ops.aten.view.default(clone_115, [128, -1, 80]);  clone_115 = None
        view_320 = torch.ops.aten.view.default(clone_113, [128, -1, 80])
        view_321 = torch.ops.aten.view.default(clone_114, [128, -1, 80])
        permute_160 = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        bmm_28 = torch.ops.aten.bmm.default(view_319, permute_160);  view_319 = permute_160 = None
        view_322 = torch.ops.aten.view.default(bmm_28, [4, 32, 128, 128]);  bmm_28 = None
        add_116 = torch.ops.aten.add.Tensor(view_322, expand_1);  view_322 = None
        view_323 = torch.ops.aten.view.default(add_116, [128, 128, 128]);  add_116 = None
        amax_14 = torch.ops.aten.amax.default(view_323, [-1], True)
        sub_43 = torch.ops.aten.sub.Tensor(view_323, amax_14);  view_323 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        bmm_29 = torch.ops.aten.bmm.default(div_14, view_321);  div_14 = view_321 = None
        view_324 = torch.ops.aten.view.default(bmm_29, [4, 32, 128, 80]);  bmm_29 = None
        permute_161 = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        clone_117 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_325 = torch.ops.aten.view.default(clone_117, [4, 128, 2560]);  clone_117 = None
        view_326 = torch.ops.aten.view.default(view_325, [512, 2560]);  view_325 = None
        permute_162 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg236_1, view_326, permute_162);  arg236_1 = view_326 = permute_162 = None
        view_327 = torch.ops.aten.view.default(addmm_87, [4, 128, 2560]);  addmm_87 = None
        add_117 = torch.ops.aten.add.Tensor(add_113, view_327);  add_113 = view_327 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_117, getitem_59);  getitem_59 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, arg237_1);  mul_116 = arg237_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_117, arg238_1);  mul_117 = arg238_1 = None
        view_328 = torch.ops.aten.view.default(add_119, [512, 2560]);  add_119 = None
        permute_163 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg240_1, view_328, permute_163);  arg240_1 = view_328 = permute_163 = None
        view_329 = torch.ops.aten.view.default(addmm_88, [4, 128, 10240]);  addmm_88 = None
        mul_118 = torch.ops.aten.mul.Tensor(view_329, 0.5)
        mul_119 = torch.ops.aten.mul.Tensor(view_329, 0.7071067811865476);  view_329 = None
        erf_14 = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_120 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_120 = torch.ops.aten.mul.Tensor(mul_118, add_120);  mul_118 = add_120 = None
        view_330 = torch.ops.aten.view.default(mul_120, [512, 10240]);  mul_120 = None
        permute_164 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg242_1, view_330, permute_164);  arg242_1 = view_330 = permute_164 = None
        view_331 = torch.ops.aten.view.default(addmm_89, [4, 128, 2560]);  addmm_89 = None
        add_121 = torch.ops.aten.add.Tensor(add_117, view_331);  add_117 = view_331 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_121, getitem_61);  getitem_61 = None
        mul_121 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, arg243_1);  mul_121 = arg243_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_122, arg244_1);  mul_122 = arg244_1 = None
        view_332 = torch.ops.aten.view.default(add_123, [512, 2560])
        permute_165 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg246_1, view_332, permute_165);  arg246_1 = view_332 = permute_165 = None
        view_333 = torch.ops.aten.view.default(addmm_90, [4, 128, 2560]);  addmm_90 = None
        mul_123 = torch.ops.aten.mul.Tensor(view_333, 0.11180339887498948);  view_333 = None
        view_334 = torch.ops.aten.view.default(add_123, [512, 2560])
        permute_166 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg248_1, view_334, permute_166);  arg248_1 = view_334 = permute_166 = None
        view_335 = torch.ops.aten.view.default(addmm_91, [4, 128, 2560]);  addmm_91 = None
        view_336 = torch.ops.aten.view.default(view_335, [4, -1, 32, 80]);  view_335 = None
        permute_167 = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        clone_121 = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_337 = torch.ops.aten.view.default(add_123, [512, 2560]);  add_123 = None
        permute_168 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg250_1, view_337, permute_168);  arg250_1 = view_337 = permute_168 = None
        view_338 = torch.ops.aten.view.default(addmm_92, [4, 128, 2560]);  addmm_92 = None
        view_339 = torch.ops.aten.view.default(view_338, [4, -1, 32, 80]);  view_338 = None
        permute_169 = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        clone_122 = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_340 = torch.ops.aten.view.default(mul_123, [4, 128, 32, 80]);  mul_123 = None
        permute_170 = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_123 = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_341 = torch.ops.aten.view.default(clone_123, [128, -1, 80]);  clone_123 = None
        view_342 = torch.ops.aten.view.default(clone_121, [128, -1, 80])
        view_343 = torch.ops.aten.view.default(clone_122, [128, -1, 80])
        permute_171 = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
        bmm_30 = torch.ops.aten.bmm.default(view_341, permute_171);  view_341 = permute_171 = None
        view_344 = torch.ops.aten.view.default(bmm_30, [4, 32, 128, 128]);  bmm_30 = None
        add_124 = torch.ops.aten.add.Tensor(view_344, expand_1);  view_344 = None
        view_345 = torch.ops.aten.view.default(add_124, [128, 128, 128]);  add_124 = None
        amax_15 = torch.ops.aten.amax.default(view_345, [-1], True)
        sub_46 = torch.ops.aten.sub.Tensor(view_345, amax_15);  view_345 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        bmm_31 = torch.ops.aten.bmm.default(div_15, view_343);  div_15 = view_343 = None
        view_346 = torch.ops.aten.view.default(bmm_31, [4, 32, 128, 80]);  bmm_31 = None
        permute_172 = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_125 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_347 = torch.ops.aten.view.default(clone_125, [4, 128, 2560]);  clone_125 = None
        view_348 = torch.ops.aten.view.default(view_347, [512, 2560]);  view_347 = None
        permute_173 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg252_1, view_348, permute_173);  arg252_1 = view_348 = permute_173 = None
        view_349 = torch.ops.aten.view.default(addmm_93, [4, 128, 2560]);  addmm_93 = None
        add_125 = torch.ops.aten.add.Tensor(add_121, view_349);  add_121 = view_349 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_125, getitem_63);  getitem_63 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, arg253_1);  mul_124 = arg253_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_125, arg254_1);  mul_125 = arg254_1 = None
        view_350 = torch.ops.aten.view.default(add_127, [512, 2560]);  add_127 = None
        permute_174 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg256_1, view_350, permute_174);  arg256_1 = view_350 = permute_174 = None
        view_351 = torch.ops.aten.view.default(addmm_94, [4, 128, 10240]);  addmm_94 = None
        mul_126 = torch.ops.aten.mul.Tensor(view_351, 0.5)
        mul_127 = torch.ops.aten.mul.Tensor(view_351, 0.7071067811865476);  view_351 = None
        erf_15 = torch.ops.aten.erf.default(mul_127);  mul_127 = None
        add_128 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_126, add_128);  mul_126 = add_128 = None
        view_352 = torch.ops.aten.view.default(mul_128, [512, 10240]);  mul_128 = None
        permute_175 = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg258_1, view_352, permute_175);  arg258_1 = view_352 = permute_175 = None
        view_353 = torch.ops.aten.view.default(addmm_95, [4, 128, 2560]);  addmm_95 = None
        add_129 = torch.ops.aten.add.Tensor(add_125, view_353);  add_125 = view_353 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_129, getitem_65);  getitem_65 = None
        mul_129 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, arg259_1);  mul_129 = arg259_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_130, arg260_1);  mul_130 = arg260_1 = None
        view_354 = torch.ops.aten.view.default(add_131, [512, 2560])
        permute_176 = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg262_1, view_354, permute_176);  arg262_1 = view_354 = permute_176 = None
        view_355 = torch.ops.aten.view.default(addmm_96, [4, 128, 2560]);  addmm_96 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_355, 0.11180339887498948);  view_355 = None
        view_356 = torch.ops.aten.view.default(add_131, [512, 2560])
        permute_177 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg264_1, view_356, permute_177);  arg264_1 = view_356 = permute_177 = None
        view_357 = torch.ops.aten.view.default(addmm_97, [4, 128, 2560]);  addmm_97 = None
        view_358 = torch.ops.aten.view.default(view_357, [4, -1, 32, 80]);  view_357 = None
        permute_178 = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_129 = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_359 = torch.ops.aten.view.default(add_131, [512, 2560]);  add_131 = None
        permute_179 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg266_1, view_359, permute_179);  arg266_1 = view_359 = permute_179 = None
        view_360 = torch.ops.aten.view.default(addmm_98, [4, 128, 2560]);  addmm_98 = None
        view_361 = torch.ops.aten.view.default(view_360, [4, -1, 32, 80]);  view_360 = None
        permute_180 = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
        clone_130 = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_362 = torch.ops.aten.view.default(mul_131, [4, 128, 32, 80]);  mul_131 = None
        permute_181 = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_131 = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_363 = torch.ops.aten.view.default(clone_131, [128, -1, 80]);  clone_131 = None
        view_364 = torch.ops.aten.view.default(clone_129, [128, -1, 80])
        view_365 = torch.ops.aten.view.default(clone_130, [128, -1, 80])
        permute_182 = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        bmm_32 = torch.ops.aten.bmm.default(view_363, permute_182);  view_363 = permute_182 = None
        view_366 = torch.ops.aten.view.default(bmm_32, [4, 32, 128, 128]);  bmm_32 = None
        add_132 = torch.ops.aten.add.Tensor(view_366, expand_1);  view_366 = None
        view_367 = torch.ops.aten.view.default(add_132, [128, 128, 128]);  add_132 = None
        amax_16 = torch.ops.aten.amax.default(view_367, [-1], True)
        sub_49 = torch.ops.aten.sub.Tensor(view_367, amax_16);  view_367 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        bmm_33 = torch.ops.aten.bmm.default(div_16, view_365);  div_16 = view_365 = None
        view_368 = torch.ops.aten.view.default(bmm_33, [4, 32, 128, 80]);  bmm_33 = None
        permute_183 = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        clone_133 = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_369 = torch.ops.aten.view.default(clone_133, [4, 128, 2560]);  clone_133 = None
        view_370 = torch.ops.aten.view.default(view_369, [512, 2560]);  view_369 = None
        permute_184 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg268_1, view_370, permute_184);  arg268_1 = view_370 = permute_184 = None
        view_371 = torch.ops.aten.view.default(addmm_99, [4, 128, 2560]);  addmm_99 = None
        add_133 = torch.ops.aten.add.Tensor(add_129, view_371);  add_129 = view_371 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_50 = torch.ops.aten.sub.Tensor(add_133, getitem_67);  getitem_67 = None
        mul_132 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, arg269_1);  mul_132 = arg269_1 = None
        add_135 = torch.ops.aten.add.Tensor(mul_133, arg270_1);  mul_133 = arg270_1 = None
        view_372 = torch.ops.aten.view.default(add_135, [512, 2560]);  add_135 = None
        permute_185 = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg272_1, view_372, permute_185);  arg272_1 = view_372 = permute_185 = None
        view_373 = torch.ops.aten.view.default(addmm_100, [4, 128, 10240]);  addmm_100 = None
        mul_134 = torch.ops.aten.mul.Tensor(view_373, 0.5)
        mul_135 = torch.ops.aten.mul.Tensor(view_373, 0.7071067811865476);  view_373 = None
        erf_16 = torch.ops.aten.erf.default(mul_135);  mul_135 = None
        add_136 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_134, add_136);  mul_134 = add_136 = None
        view_374 = torch.ops.aten.view.default(mul_136, [512, 10240]);  mul_136 = None
        permute_186 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg274_1, view_374, permute_186);  arg274_1 = view_374 = permute_186 = None
        view_375 = torch.ops.aten.view.default(addmm_101, [4, 128, 2560]);  addmm_101 = None
        add_137 = torch.ops.aten.add.Tensor(add_133, view_375);  add_133 = view_375 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_137, getitem_69);  getitem_69 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, arg275_1);  mul_137 = arg275_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_138, arg276_1);  mul_138 = arg276_1 = None
        view_376 = torch.ops.aten.view.default(add_139, [512, 2560])
        permute_187 = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg278_1, view_376, permute_187);  arg278_1 = view_376 = permute_187 = None
        view_377 = torch.ops.aten.view.default(addmm_102, [4, 128, 2560]);  addmm_102 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_377, 0.11180339887498948);  view_377 = None
        view_378 = torch.ops.aten.view.default(add_139, [512, 2560])
        permute_188 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg280_1, view_378, permute_188);  arg280_1 = view_378 = permute_188 = None
        view_379 = torch.ops.aten.view.default(addmm_103, [4, 128, 2560]);  addmm_103 = None
        view_380 = torch.ops.aten.view.default(view_379, [4, -1, 32, 80]);  view_379 = None
        permute_189 = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_137 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_381 = torch.ops.aten.view.default(add_139, [512, 2560]);  add_139 = None
        permute_190 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg282_1, view_381, permute_190);  arg282_1 = view_381 = permute_190 = None
        view_382 = torch.ops.aten.view.default(addmm_104, [4, 128, 2560]);  addmm_104 = None
        view_383 = torch.ops.aten.view.default(view_382, [4, -1, 32, 80]);  view_382 = None
        permute_191 = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_138 = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_384 = torch.ops.aten.view.default(mul_139, [4, 128, 32, 80]);  mul_139 = None
        permute_192 = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_139 = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_385 = torch.ops.aten.view.default(clone_139, [128, -1, 80]);  clone_139 = None
        view_386 = torch.ops.aten.view.default(clone_137, [128, -1, 80])
        view_387 = torch.ops.aten.view.default(clone_138, [128, -1, 80])
        permute_193 = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
        bmm_34 = torch.ops.aten.bmm.default(view_385, permute_193);  view_385 = permute_193 = None
        view_388 = torch.ops.aten.view.default(bmm_34, [4, 32, 128, 128]);  bmm_34 = None
        add_140 = torch.ops.aten.add.Tensor(view_388, expand_1);  view_388 = None
        view_389 = torch.ops.aten.view.default(add_140, [128, 128, 128]);  add_140 = None
        amax_17 = torch.ops.aten.amax.default(view_389, [-1], True)
        sub_52 = torch.ops.aten.sub.Tensor(view_389, amax_17);  view_389 = amax_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        bmm_35 = torch.ops.aten.bmm.default(div_17, view_387);  div_17 = view_387 = None
        view_390 = torch.ops.aten.view.default(bmm_35, [4, 32, 128, 80]);  bmm_35 = None
        permute_194 = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
        clone_141 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_391 = torch.ops.aten.view.default(clone_141, [4, 128, 2560]);  clone_141 = None
        view_392 = torch.ops.aten.view.default(view_391, [512, 2560]);  view_391 = None
        permute_195 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg284_1, view_392, permute_195);  arg284_1 = view_392 = permute_195 = None
        view_393 = torch.ops.aten.view.default(addmm_105, [4, 128, 2560]);  addmm_105 = None
        add_141 = torch.ops.aten.add.Tensor(add_137, view_393);  add_137 = view_393 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_53 = torch.ops.aten.sub.Tensor(add_141, getitem_71);  getitem_71 = None
        mul_140 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, arg285_1);  mul_140 = arg285_1 = None
        add_143 = torch.ops.aten.add.Tensor(mul_141, arg286_1);  mul_141 = arg286_1 = None
        view_394 = torch.ops.aten.view.default(add_143, [512, 2560]);  add_143 = None
        permute_196 = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg288_1, view_394, permute_196);  arg288_1 = view_394 = permute_196 = None
        view_395 = torch.ops.aten.view.default(addmm_106, [4, 128, 10240]);  addmm_106 = None
        mul_142 = torch.ops.aten.mul.Tensor(view_395, 0.5)
        mul_143 = torch.ops.aten.mul.Tensor(view_395, 0.7071067811865476);  view_395 = None
        erf_17 = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_144 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_142, add_144);  mul_142 = add_144 = None
        view_396 = torch.ops.aten.view.default(mul_144, [512, 10240]);  mul_144 = None
        permute_197 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg290_1, view_396, permute_197);  arg290_1 = view_396 = permute_197 = None
        view_397 = torch.ops.aten.view.default(addmm_107, [4, 128, 2560]);  addmm_107 = None
        add_145 = torch.ops.aten.add.Tensor(add_141, view_397);  add_141 = view_397 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_146 = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_145, getitem_73);  getitem_73 = None
        mul_145 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, arg291_1);  mul_145 = arg291_1 = None
        add_147 = torch.ops.aten.add.Tensor(mul_146, arg292_1);  mul_146 = arg292_1 = None
        view_398 = torch.ops.aten.view.default(add_147, [512, 2560])
        permute_198 = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg294_1, view_398, permute_198);  arg294_1 = view_398 = permute_198 = None
        view_399 = torch.ops.aten.view.default(addmm_108, [4, 128, 2560]);  addmm_108 = None
        mul_147 = torch.ops.aten.mul.Tensor(view_399, 0.11180339887498948);  view_399 = None
        view_400 = torch.ops.aten.view.default(add_147, [512, 2560])
        permute_199 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg296_1, view_400, permute_199);  arg296_1 = view_400 = permute_199 = None
        view_401 = torch.ops.aten.view.default(addmm_109, [4, 128, 2560]);  addmm_109 = None
        view_402 = torch.ops.aten.view.default(view_401, [4, -1, 32, 80]);  view_401 = None
        permute_200 = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_145 = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_403 = torch.ops.aten.view.default(add_147, [512, 2560]);  add_147 = None
        permute_201 = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg298_1, view_403, permute_201);  arg298_1 = view_403 = permute_201 = None
        view_404 = torch.ops.aten.view.default(addmm_110, [4, 128, 2560]);  addmm_110 = None
        view_405 = torch.ops.aten.view.default(view_404, [4, -1, 32, 80]);  view_404 = None
        permute_202 = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        clone_146 = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_406 = torch.ops.aten.view.default(mul_147, [4, 128, 32, 80]);  mul_147 = None
        permute_203 = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        clone_147 = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_407 = torch.ops.aten.view.default(clone_147, [128, -1, 80]);  clone_147 = None
        view_408 = torch.ops.aten.view.default(clone_145, [128, -1, 80])
        view_409 = torch.ops.aten.view.default(clone_146, [128, -1, 80])
        permute_204 = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
        bmm_36 = torch.ops.aten.bmm.default(view_407, permute_204);  view_407 = permute_204 = None
        view_410 = torch.ops.aten.view.default(bmm_36, [4, 32, 128, 128]);  bmm_36 = None
        add_148 = torch.ops.aten.add.Tensor(view_410, expand_1);  view_410 = None
        view_411 = torch.ops.aten.view.default(add_148, [128, 128, 128]);  add_148 = None
        amax_18 = torch.ops.aten.amax.default(view_411, [-1], True)
        sub_55 = torch.ops.aten.sub.Tensor(view_411, amax_18);  view_411 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        bmm_37 = torch.ops.aten.bmm.default(div_18, view_409);  div_18 = view_409 = None
        view_412 = torch.ops.aten.view.default(bmm_37, [4, 32, 128, 80]);  bmm_37 = None
        permute_205 = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_149 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_413 = torch.ops.aten.view.default(clone_149, [4, 128, 2560]);  clone_149 = None
        view_414 = torch.ops.aten.view.default(view_413, [512, 2560]);  view_413 = None
        permute_206 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg300_1, view_414, permute_206);  arg300_1 = view_414 = permute_206 = None
        view_415 = torch.ops.aten.view.default(addmm_111, [4, 128, 2560]);  addmm_111 = None
        add_149 = torch.ops.aten.add.Tensor(add_145, view_415);  add_145 = view_415 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_150 = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_149, getitem_75);  getitem_75 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, arg301_1);  mul_148 = arg301_1 = None
        add_151 = torch.ops.aten.add.Tensor(mul_149, arg302_1);  mul_149 = arg302_1 = None
        view_416 = torch.ops.aten.view.default(add_151, [512, 2560]);  add_151 = None
        permute_207 = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg304_1, view_416, permute_207);  arg304_1 = view_416 = permute_207 = None
        view_417 = torch.ops.aten.view.default(addmm_112, [4, 128, 10240]);  addmm_112 = None
        mul_150 = torch.ops.aten.mul.Tensor(view_417, 0.5)
        mul_151 = torch.ops.aten.mul.Tensor(view_417, 0.7071067811865476);  view_417 = None
        erf_18 = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_152 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_150, add_152);  mul_150 = add_152 = None
        view_418 = torch.ops.aten.view.default(mul_152, [512, 10240]);  mul_152 = None
        permute_208 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg306_1, view_418, permute_208);  arg306_1 = view_418 = permute_208 = None
        view_419 = torch.ops.aten.view.default(addmm_113, [4, 128, 2560]);  addmm_113 = None
        add_153 = torch.ops.aten.add.Tensor(add_149, view_419);  add_149 = view_419 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_153, getitem_77);  getitem_77 = None
        mul_153 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_153, arg307_1);  mul_153 = arg307_1 = None
        add_155 = torch.ops.aten.add.Tensor(mul_154, arg308_1);  mul_154 = arg308_1 = None
        view_420 = torch.ops.aten.view.default(add_155, [512, 2560])
        permute_209 = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg310_1, view_420, permute_209);  arg310_1 = view_420 = permute_209 = None
        view_421 = torch.ops.aten.view.default(addmm_114, [4, 128, 2560]);  addmm_114 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_421, 0.11180339887498948);  view_421 = None
        view_422 = torch.ops.aten.view.default(add_155, [512, 2560])
        permute_210 = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg312_1, view_422, permute_210);  arg312_1 = view_422 = permute_210 = None
        view_423 = torch.ops.aten.view.default(addmm_115, [4, 128, 2560]);  addmm_115 = None
        view_424 = torch.ops.aten.view.default(view_423, [4, -1, 32, 80]);  view_423 = None
        permute_211 = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        clone_153 = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_425 = torch.ops.aten.view.default(add_155, [512, 2560]);  add_155 = None
        permute_212 = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg314_1, view_425, permute_212);  arg314_1 = view_425 = permute_212 = None
        view_426 = torch.ops.aten.view.default(addmm_116, [4, 128, 2560]);  addmm_116 = None
        view_427 = torch.ops.aten.view.default(view_426, [4, -1, 32, 80]);  view_426 = None
        permute_213 = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_154 = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_428 = torch.ops.aten.view.default(mul_155, [4, 128, 32, 80]);  mul_155 = None
        permute_214 = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_155 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_429 = torch.ops.aten.view.default(clone_155, [128, -1, 80]);  clone_155 = None
        view_430 = torch.ops.aten.view.default(clone_153, [128, -1, 80])
        view_431 = torch.ops.aten.view.default(clone_154, [128, -1, 80])
        permute_215 = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
        bmm_38 = torch.ops.aten.bmm.default(view_429, permute_215);  view_429 = permute_215 = None
        view_432 = torch.ops.aten.view.default(bmm_38, [4, 32, 128, 128]);  bmm_38 = None
        add_156 = torch.ops.aten.add.Tensor(view_432, expand_1);  view_432 = None
        view_433 = torch.ops.aten.view.default(add_156, [128, 128, 128]);  add_156 = None
        amax_19 = torch.ops.aten.amax.default(view_433, [-1], True)
        sub_58 = torch.ops.aten.sub.Tensor(view_433, amax_19);  view_433 = amax_19 = None
        exp_19 = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        bmm_39 = torch.ops.aten.bmm.default(div_19, view_431);  div_19 = view_431 = None
        view_434 = torch.ops.aten.view.default(bmm_39, [4, 32, 128, 80]);  bmm_39 = None
        permute_216 = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_157 = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_435 = torch.ops.aten.view.default(clone_157, [4, 128, 2560]);  clone_157 = None
        view_436 = torch.ops.aten.view.default(view_435, [512, 2560]);  view_435 = None
        permute_217 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg316_1, view_436, permute_217);  arg316_1 = view_436 = permute_217 = None
        view_437 = torch.ops.aten.view.default(addmm_117, [4, 128, 2560]);  addmm_117 = None
        add_157 = torch.ops.aten.add.Tensor(add_153, view_437);  add_153 = view_437 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_158 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_157, getitem_79);  getitem_79 = None
        mul_156 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, arg317_1);  mul_156 = arg317_1 = None
        add_159 = torch.ops.aten.add.Tensor(mul_157, arg318_1);  mul_157 = arg318_1 = None
        view_438 = torch.ops.aten.view.default(add_159, [512, 2560]);  add_159 = None
        permute_218 = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg320_1, view_438, permute_218);  arg320_1 = view_438 = permute_218 = None
        view_439 = torch.ops.aten.view.default(addmm_118, [4, 128, 10240]);  addmm_118 = None
        mul_158 = torch.ops.aten.mul.Tensor(view_439, 0.5)
        mul_159 = torch.ops.aten.mul.Tensor(view_439, 0.7071067811865476);  view_439 = None
        erf_19 = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_160 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_158, add_160);  mul_158 = add_160 = None
        view_440 = torch.ops.aten.view.default(mul_160, [512, 10240]);  mul_160 = None
        permute_219 = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg322_1, view_440, permute_219);  arg322_1 = view_440 = permute_219 = None
        view_441 = torch.ops.aten.view.default(addmm_119, [4, 128, 2560]);  addmm_119 = None
        add_161 = torch.ops.aten.add.Tensor(add_157, view_441);  add_157 = view_441 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_60 = torch.ops.aten.sub.Tensor(add_161, getitem_81);  getitem_81 = None
        mul_161 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_161, arg323_1);  mul_161 = arg323_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_162, arg324_1);  mul_162 = arg324_1 = None
        view_442 = torch.ops.aten.view.default(add_163, [512, 2560])
        permute_220 = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg326_1, view_442, permute_220);  arg326_1 = view_442 = permute_220 = None
        view_443 = torch.ops.aten.view.default(addmm_120, [4, 128, 2560]);  addmm_120 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_443, 0.11180339887498948);  view_443 = None
        view_444 = torch.ops.aten.view.default(add_163, [512, 2560])
        permute_221 = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg328_1, view_444, permute_221);  arg328_1 = view_444 = permute_221 = None
        view_445 = torch.ops.aten.view.default(addmm_121, [4, 128, 2560]);  addmm_121 = None
        view_446 = torch.ops.aten.view.default(view_445, [4, -1, 32, 80]);  view_445 = None
        permute_222 = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        clone_161 = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_447 = torch.ops.aten.view.default(add_163, [512, 2560]);  add_163 = None
        permute_223 = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg330_1, view_447, permute_223);  arg330_1 = view_447 = permute_223 = None
        view_448 = torch.ops.aten.view.default(addmm_122, [4, 128, 2560]);  addmm_122 = None
        view_449 = torch.ops.aten.view.default(view_448, [4, -1, 32, 80]);  view_448 = None
        permute_224 = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
        clone_162 = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_450 = torch.ops.aten.view.default(mul_163, [4, 128, 32, 80]);  mul_163 = None
        permute_225 = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        clone_163 = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_451 = torch.ops.aten.view.default(clone_163, [128, -1, 80]);  clone_163 = None
        view_452 = torch.ops.aten.view.default(clone_161, [128, -1, 80])
        view_453 = torch.ops.aten.view.default(clone_162, [128, -1, 80])
        permute_226 = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
        bmm_40 = torch.ops.aten.bmm.default(view_451, permute_226);  view_451 = permute_226 = None
        view_454 = torch.ops.aten.view.default(bmm_40, [4, 32, 128, 128]);  bmm_40 = None
        add_164 = torch.ops.aten.add.Tensor(view_454, expand_1);  view_454 = None
        view_455 = torch.ops.aten.view.default(add_164, [128, 128, 128]);  add_164 = None
        amax_20 = torch.ops.aten.amax.default(view_455, [-1], True)
        sub_61 = torch.ops.aten.sub.Tensor(view_455, amax_20);  view_455 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        bmm_41 = torch.ops.aten.bmm.default(div_20, view_453);  div_20 = view_453 = None
        view_456 = torch.ops.aten.view.default(bmm_41, [4, 32, 128, 80]);  bmm_41 = None
        permute_227 = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_165 = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_457 = torch.ops.aten.view.default(clone_165, [4, 128, 2560]);  clone_165 = None
        view_458 = torch.ops.aten.view.default(view_457, [512, 2560]);  view_457 = None
        permute_228 = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg332_1, view_458, permute_228);  arg332_1 = view_458 = permute_228 = None
        view_459 = torch.ops.aten.view.default(addmm_123, [4, 128, 2560]);  addmm_123 = None
        add_165 = torch.ops.aten.add.Tensor(add_161, view_459);  add_161 = view_459 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_166 = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_165, getitem_83);  getitem_83 = None
        mul_164 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, arg333_1);  mul_164 = arg333_1 = None
        add_167 = torch.ops.aten.add.Tensor(mul_165, arg334_1);  mul_165 = arg334_1 = None
        view_460 = torch.ops.aten.view.default(add_167, [512, 2560]);  add_167 = None
        permute_229 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg336_1, view_460, permute_229);  arg336_1 = view_460 = permute_229 = None
        view_461 = torch.ops.aten.view.default(addmm_124, [4, 128, 10240]);  addmm_124 = None
        mul_166 = torch.ops.aten.mul.Tensor(view_461, 0.5)
        mul_167 = torch.ops.aten.mul.Tensor(view_461, 0.7071067811865476);  view_461 = None
        erf_20 = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_168 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_166, add_168);  mul_166 = add_168 = None
        view_462 = torch.ops.aten.view.default(mul_168, [512, 10240]);  mul_168 = None
        permute_230 = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg338_1, view_462, permute_230);  arg338_1 = view_462 = permute_230 = None
        view_463 = torch.ops.aten.view.default(addmm_125, [4, 128, 2560]);  addmm_125 = None
        add_169 = torch.ops.aten.add.Tensor(add_165, view_463);  add_165 = view_463 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_63 = torch.ops.aten.sub.Tensor(add_169, getitem_85);  getitem_85 = None
        mul_169 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, arg339_1);  mul_169 = arg339_1 = None
        add_171 = torch.ops.aten.add.Tensor(mul_170, arg340_1);  mul_170 = arg340_1 = None
        view_464 = torch.ops.aten.view.default(add_171, [512, 2560])
        permute_231 = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg342_1, view_464, permute_231);  arg342_1 = view_464 = permute_231 = None
        view_465 = torch.ops.aten.view.default(addmm_126, [4, 128, 2560]);  addmm_126 = None
        mul_171 = torch.ops.aten.mul.Tensor(view_465, 0.11180339887498948);  view_465 = None
        view_466 = torch.ops.aten.view.default(add_171, [512, 2560])
        permute_232 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg344_1, view_466, permute_232);  arg344_1 = view_466 = permute_232 = None
        view_467 = torch.ops.aten.view.default(addmm_127, [4, 128, 2560]);  addmm_127 = None
        view_468 = torch.ops.aten.view.default(view_467, [4, -1, 32, 80]);  view_467 = None
        permute_233 = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_169 = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        view_469 = torch.ops.aten.view.default(add_171, [512, 2560]);  add_171 = None
        permute_234 = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg346_1, view_469, permute_234);  arg346_1 = view_469 = permute_234 = None
        view_470 = torch.ops.aten.view.default(addmm_128, [4, 128, 2560]);  addmm_128 = None
        view_471 = torch.ops.aten.view.default(view_470, [4, -1, 32, 80]);  view_470 = None
        permute_235 = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
        clone_170 = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_472 = torch.ops.aten.view.default(mul_171, [4, 128, 32, 80]);  mul_171 = None
        permute_236 = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_171 = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_473 = torch.ops.aten.view.default(clone_171, [128, -1, 80]);  clone_171 = None
        view_474 = torch.ops.aten.view.default(clone_169, [128, -1, 80])
        view_475 = torch.ops.aten.view.default(clone_170, [128, -1, 80])
        permute_237 = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
        bmm_42 = torch.ops.aten.bmm.default(view_473, permute_237);  view_473 = permute_237 = None
        view_476 = torch.ops.aten.view.default(bmm_42, [4, 32, 128, 128]);  bmm_42 = None
        add_172 = torch.ops.aten.add.Tensor(view_476, expand_1);  view_476 = None
        view_477 = torch.ops.aten.view.default(add_172, [128, 128, 128]);  add_172 = None
        amax_21 = torch.ops.aten.amax.default(view_477, [-1], True)
        sub_64 = torch.ops.aten.sub.Tensor(view_477, amax_21);  view_477 = amax_21 = None
        exp_21 = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        bmm_43 = torch.ops.aten.bmm.default(div_21, view_475);  div_21 = view_475 = None
        view_478 = torch.ops.aten.view.default(bmm_43, [4, 32, 128, 80]);  bmm_43 = None
        permute_238 = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        clone_173 = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_479 = torch.ops.aten.view.default(clone_173, [4, 128, 2560]);  clone_173 = None
        view_480 = torch.ops.aten.view.default(view_479, [512, 2560]);  view_479 = None
        permute_239 = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg348_1, view_480, permute_239);  arg348_1 = view_480 = permute_239 = None
        view_481 = torch.ops.aten.view.default(addmm_129, [4, 128, 2560]);  addmm_129 = None
        add_173 = torch.ops.aten.add.Tensor(add_169, view_481);  add_169 = view_481 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_65 = torch.ops.aten.sub.Tensor(add_173, getitem_87);  getitem_87 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, arg349_1);  mul_172 = arg349_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_173, arg350_1);  mul_173 = arg350_1 = None
        view_482 = torch.ops.aten.view.default(add_175, [512, 2560]);  add_175 = None
        permute_240 = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg352_1, view_482, permute_240);  arg352_1 = view_482 = permute_240 = None
        view_483 = torch.ops.aten.view.default(addmm_130, [4, 128, 10240]);  addmm_130 = None
        mul_174 = torch.ops.aten.mul.Tensor(view_483, 0.5)
        mul_175 = torch.ops.aten.mul.Tensor(view_483, 0.7071067811865476);  view_483 = None
        erf_21 = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_176 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_174, add_176);  mul_174 = add_176 = None
        view_484 = torch.ops.aten.view.default(mul_176, [512, 10240]);  mul_176 = None
        permute_241 = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg354_1, view_484, permute_241);  arg354_1 = view_484 = permute_241 = None
        view_485 = torch.ops.aten.view.default(addmm_131, [4, 128, 2560]);  addmm_131 = None
        add_177 = torch.ops.aten.add.Tensor(add_173, view_485);  add_173 = view_485 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_178 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_66 = torch.ops.aten.sub.Tensor(add_177, getitem_89);  getitem_89 = None
        mul_177 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, arg355_1);  mul_177 = arg355_1 = None
        add_179 = torch.ops.aten.add.Tensor(mul_178, arg356_1);  mul_178 = arg356_1 = None
        view_486 = torch.ops.aten.view.default(add_179, [512, 2560])
        permute_242 = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg358_1, view_486, permute_242);  arg358_1 = view_486 = permute_242 = None
        view_487 = torch.ops.aten.view.default(addmm_132, [4, 128, 2560]);  addmm_132 = None
        mul_179 = torch.ops.aten.mul.Tensor(view_487, 0.11180339887498948);  view_487 = None
        view_488 = torch.ops.aten.view.default(add_179, [512, 2560])
        permute_243 = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg360_1, view_488, permute_243);  arg360_1 = view_488 = permute_243 = None
        view_489 = torch.ops.aten.view.default(addmm_133, [4, 128, 2560]);  addmm_133 = None
        view_490 = torch.ops.aten.view.default(view_489, [4, -1, 32, 80]);  view_489 = None
        permute_244 = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        clone_177 = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_491 = torch.ops.aten.view.default(add_179, [512, 2560]);  add_179 = None
        permute_245 = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg362_1, view_491, permute_245);  arg362_1 = view_491 = permute_245 = None
        view_492 = torch.ops.aten.view.default(addmm_134, [4, 128, 2560]);  addmm_134 = None
        view_493 = torch.ops.aten.view.default(view_492, [4, -1, 32, 80]);  view_492 = None
        permute_246 = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
        clone_178 = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_494 = torch.ops.aten.view.default(mul_179, [4, 128, 32, 80]);  mul_179 = None
        permute_247 = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        clone_179 = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_495 = torch.ops.aten.view.default(clone_179, [128, -1, 80]);  clone_179 = None
        view_496 = torch.ops.aten.view.default(clone_177, [128, -1, 80])
        view_497 = torch.ops.aten.view.default(clone_178, [128, -1, 80])
        permute_248 = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
        bmm_44 = torch.ops.aten.bmm.default(view_495, permute_248);  view_495 = permute_248 = None
        view_498 = torch.ops.aten.view.default(bmm_44, [4, 32, 128, 128]);  bmm_44 = None
        add_180 = torch.ops.aten.add.Tensor(view_498, expand_1);  view_498 = None
        view_499 = torch.ops.aten.view.default(add_180, [128, 128, 128]);  add_180 = None
        amax_22 = torch.ops.aten.amax.default(view_499, [-1], True)
        sub_67 = torch.ops.aten.sub.Tensor(view_499, amax_22);  view_499 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        bmm_45 = torch.ops.aten.bmm.default(div_22, view_497);  div_22 = view_497 = None
        view_500 = torch.ops.aten.view.default(bmm_45, [4, 32, 128, 80]);  bmm_45 = None
        permute_249 = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        clone_181 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_501 = torch.ops.aten.view.default(clone_181, [4, 128, 2560]);  clone_181 = None
        view_502 = torch.ops.aten.view.default(view_501, [512, 2560]);  view_501 = None
        permute_250 = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg364_1, view_502, permute_250);  arg364_1 = view_502 = permute_250 = None
        view_503 = torch.ops.aten.view.default(addmm_135, [4, 128, 2560]);  addmm_135 = None
        add_181 = torch.ops.aten.add.Tensor(add_177, view_503);  add_177 = view_503 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_182 = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_68 = torch.ops.aten.sub.Tensor(add_181, getitem_91);  getitem_91 = None
        mul_180 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, arg365_1);  mul_180 = arg365_1 = None
        add_183 = torch.ops.aten.add.Tensor(mul_181, arg366_1);  mul_181 = arg366_1 = None
        view_504 = torch.ops.aten.view.default(add_183, [512, 2560]);  add_183 = None
        permute_251 = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg368_1, view_504, permute_251);  arg368_1 = view_504 = permute_251 = None
        view_505 = torch.ops.aten.view.default(addmm_136, [4, 128, 10240]);  addmm_136 = None
        mul_182 = torch.ops.aten.mul.Tensor(view_505, 0.5)
        mul_183 = torch.ops.aten.mul.Tensor(view_505, 0.7071067811865476);  view_505 = None
        erf_22 = torch.ops.aten.erf.default(mul_183);  mul_183 = None
        add_184 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_182, add_184);  mul_182 = add_184 = None
        view_506 = torch.ops.aten.view.default(mul_184, [512, 10240]);  mul_184 = None
        permute_252 = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg370_1, view_506, permute_252);  arg370_1 = view_506 = permute_252 = None
        view_507 = torch.ops.aten.view.default(addmm_137, [4, 128, 2560]);  addmm_137 = None
        add_185 = torch.ops.aten.add.Tensor(add_181, view_507);  add_181 = view_507 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_69 = torch.ops.aten.sub.Tensor(add_185, getitem_93);  getitem_93 = None
        mul_185 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, arg371_1);  mul_185 = arg371_1 = None
        add_187 = torch.ops.aten.add.Tensor(mul_186, arg372_1);  mul_186 = arg372_1 = None
        view_508 = torch.ops.aten.view.default(add_187, [512, 2560])
        permute_253 = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg374_1, view_508, permute_253);  arg374_1 = view_508 = permute_253 = None
        view_509 = torch.ops.aten.view.default(addmm_138, [4, 128, 2560]);  addmm_138 = None
        mul_187 = torch.ops.aten.mul.Tensor(view_509, 0.11180339887498948);  view_509 = None
        view_510 = torch.ops.aten.view.default(add_187, [512, 2560])
        permute_254 = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg376_1, view_510, permute_254);  arg376_1 = view_510 = permute_254 = None
        view_511 = torch.ops.aten.view.default(addmm_139, [4, 128, 2560]);  addmm_139 = None
        view_512 = torch.ops.aten.view.default(view_511, [4, -1, 32, 80]);  view_511 = None
        permute_255 = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_185 = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        view_513 = torch.ops.aten.view.default(add_187, [512, 2560]);  add_187 = None
        permute_256 = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg378_1, view_513, permute_256);  arg378_1 = view_513 = permute_256 = None
        view_514 = torch.ops.aten.view.default(addmm_140, [4, 128, 2560]);  addmm_140 = None
        view_515 = torch.ops.aten.view.default(view_514, [4, -1, 32, 80]);  view_514 = None
        permute_257 = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
        clone_186 = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_516 = torch.ops.aten.view.default(mul_187, [4, 128, 32, 80]);  mul_187 = None
        permute_258 = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_187 = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_517 = torch.ops.aten.view.default(clone_187, [128, -1, 80]);  clone_187 = None
        view_518 = torch.ops.aten.view.default(clone_185, [128, -1, 80])
        view_519 = torch.ops.aten.view.default(clone_186, [128, -1, 80])
        permute_259 = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
        bmm_46 = torch.ops.aten.bmm.default(view_517, permute_259);  view_517 = permute_259 = None
        view_520 = torch.ops.aten.view.default(bmm_46, [4, 32, 128, 128]);  bmm_46 = None
        add_188 = torch.ops.aten.add.Tensor(view_520, expand_1);  view_520 = expand_1 = None
        view_521 = torch.ops.aten.view.default(add_188, [128, 128, 128]);  add_188 = None
        amax_23 = torch.ops.aten.amax.default(view_521, [-1], True)
        sub_70 = torch.ops.aten.sub.Tensor(view_521, amax_23);  view_521 = amax_23 = None
        exp_23 = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        bmm_47 = torch.ops.aten.bmm.default(div_23, view_519);  div_23 = view_519 = None
        view_522 = torch.ops.aten.view.default(bmm_47, [4, 32, 128, 80]);  bmm_47 = None
        permute_260 = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        clone_189 = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_523 = torch.ops.aten.view.default(clone_189, [4, 128, 2560]);  clone_189 = None
        view_524 = torch.ops.aten.view.default(view_523, [512, 2560]);  view_523 = None
        permute_261 = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg380_1, view_524, permute_261);  arg380_1 = view_524 = permute_261 = None
        view_525 = torch.ops.aten.view.default(addmm_141, [4, 128, 2560]);  addmm_141 = None
        add_189 = torch.ops.aten.add.Tensor(add_185, view_525);  add_185 = view_525 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_190 = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_189, getitem_95);  getitem_95 = None
        mul_188 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, arg381_1);  mul_188 = arg381_1 = None
        add_191 = torch.ops.aten.add.Tensor(mul_189, arg382_1);  mul_189 = arg382_1 = None
        view_526 = torch.ops.aten.view.default(add_191, [512, 2560]);  add_191 = None
        permute_262 = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg384_1, view_526, permute_262);  arg384_1 = view_526 = permute_262 = None
        view_527 = torch.ops.aten.view.default(addmm_142, [4, 128, 10240]);  addmm_142 = None
        mul_190 = torch.ops.aten.mul.Tensor(view_527, 0.5)
        mul_191 = torch.ops.aten.mul.Tensor(view_527, 0.7071067811865476);  view_527 = None
        erf_23 = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_192 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_190, add_192);  mul_190 = add_192 = None
        view_528 = torch.ops.aten.view.default(mul_192, [512, 10240]);  mul_192 = None
        permute_263 = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg386_1, view_528, permute_263);  arg386_1 = view_528 = permute_263 = None
        view_529 = torch.ops.aten.view.default(addmm_143, [4, 128, 2560]);  addmm_143 = None
        add_193 = torch.ops.aten.add.Tensor(add_189, view_529);  add_189 = view_529 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_194 = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_72 = torch.ops.aten.sub.Tensor(add_193, getitem_97);  add_193 = getitem_97 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, arg387_1);  mul_193 = arg387_1 = None
        add_195 = torch.ops.aten.add.Tensor(mul_194, arg388_1);  mul_194 = arg388_1 = None
        permute_264 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_530 = torch.ops.aten.view.default(add_195, [512, 2560]);  add_195 = None
        mm = torch.ops.aten.mm.default(view_530, permute_264);  view_530 = permute_264 = None
        view_531 = torch.ops.aten.view.default(mm, [4, 128, 8008]);  mm = None
        view_532 = torch.ops.aten.view.default(view_531, [-1, 8008])
        view_533 = torch.ops.aten.view.default(arg389_1, [-1]);  arg389_1 = None
        amax_24 = torch.ops.aten.amax.default(view_532, [1], True)
        sub_73 = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_73)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_74 = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
        ne = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_533, full_default_2);  ne = full_default_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_74, 1, unsqueeze_4);  sub_74 = unsqueeze_4 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_533, -100);  view_533 = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_24 = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
        return (div_24, view_531, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58, clone_65, clone_66, clone_73, clone_74, clone_81, clone_82, clone_89, clone_90, clone_97, clone_98, clone_105, clone_106, clone_113, clone_114, clone_121, clone_122, clone_129, clone_130, clone_137, clone_138, clone_145, clone_146, clone_153, clone_154, clone_161, clone_162, clone_169, clone_170, clone_177, clone_178, clone_185, clone_186)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 82001920, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8008, 2560), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128, 2560), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf3, (2560,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2560,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2560, 2560), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2560,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2560, 2560), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2560,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf9, (2560, 2560), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf10, (2560,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf11, (2560, 2560), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf12, (2560,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf13, (2560,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf14, (2560,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf15, (10240, 2560), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf16, (10240,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf17, (2560, 10240), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf18, (2560,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2560,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf20, (2560,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf21, (2560, 2560), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf22, (2560,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf23, (2560, 2560), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf24, (2560,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf25, (2560, 2560), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf26, (2560,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf27, (2560, 2560), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf28, (2560,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf29, (2560,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf30, (2560,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf31, (10240, 2560), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf32, (10240,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf33, (2560, 10240), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf34, (2560,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf35, (2560,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf36, (2560,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf37, (2560, 2560), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf38, (2560,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf39, (2560, 2560), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf40, (2560,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf41, (2560, 2560), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf42, (2560,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf43, (2560, 2560), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf44, (2560,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2560,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2560,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf47, (10240, 2560), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf48, (10240,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf49, (2560, 10240), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf50, (2560,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf51, (2560,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf52, (2560,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf53, (2560, 2560), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2560,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf55, (2560, 2560), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf56, (2560,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf57, (2560, 2560), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf58, (2560,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf59, (2560, 2560), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf60, (2560,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf61, (2560,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf62, (2560,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf63, (10240, 2560), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf64, (10240,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf65, (2560, 10240), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf66, (2560,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf67, (2560,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf68, (2560,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf69, (2560, 2560), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf70, (2560,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf71, (2560, 2560), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf72, (2560,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf73, (2560, 2560), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf74, (2560,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf75, (2560, 2560), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf76, (2560,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf77, (2560,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf78, (2560,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf79, (10240, 2560), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf80, (10240,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf81, (2560, 10240), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf82, (2560,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf83, (2560,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf84, (2560,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf85, (2560, 2560), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf86, (2560,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf87, (2560, 2560), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf88, (2560,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf89, (2560, 2560), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2560,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf91, (2560, 2560), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf92, (2560,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf93, (2560,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf94, (2560,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf95, (10240, 2560), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf96, (10240,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf97, (2560, 10240), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf98, (2560,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf99, (2560,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf100, (2560,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf101, (2560, 2560), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf102, (2560,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf103, (2560, 2560), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf104, (2560,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf105, (2560, 2560), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf106, (2560,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf107, (2560, 2560), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf108, (2560,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf109, (2560,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf110, (2560,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf111, (10240, 2560), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf112, (10240,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf113, (2560, 10240), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf114, (2560,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf115, (2560,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf116, (2560,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf117, (2560, 2560), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf118, (2560,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf119, (2560, 2560), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf120, (2560,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf121, (2560, 2560), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf122, (2560,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf123, (2560, 2560), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf124, (2560,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf125, (2560,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf126, (2560,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf127, (10240, 2560), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf128, (10240,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf129, (2560, 10240), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf130, (2560,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf131, (2560,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf132, (2560,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf133, (2560, 2560), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf134, (2560,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf135, (2560, 2560), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf136, (2560,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf137, (2560, 2560), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf138, (2560,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf139, (2560, 2560), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf140, (2560,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf141, (2560,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf142, (2560,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf143, (10240, 2560), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf144, (10240,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf145, (2560, 10240), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf146, (2560,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf147, (2560,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf148, (2560,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf149, (2560, 2560), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf150, (2560,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf151, (2560, 2560), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf152, (2560,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf153, (2560, 2560), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf154, (2560,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf155, (2560, 2560), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf156, (2560,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf157, (2560,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf158, (2560,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf159, (10240, 2560), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf160, (10240,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf161, (2560, 10240), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf162, (2560,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf163, (2560,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf164, (2560,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf165, (2560, 2560), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf166, (2560,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf167, (2560, 2560), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf168, (2560,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf169, (2560, 2560), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf170, (2560,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf171, (2560, 2560), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf172, (2560,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf173, (2560,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf174, (2560,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf175, (10240, 2560), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf176, (10240,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf177, (2560, 10240), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf178, (2560,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf179, (2560,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf180, (2560,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf181, (2560, 2560), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf182, (2560,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf183, (2560, 2560), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf184, (2560,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf185, (2560, 2560), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf186, (2560,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf187, (2560, 2560), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf188, (2560,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf189, (2560,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf190, (2560,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf191, (10240, 2560), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf192, (10240,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf193, (2560, 10240), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf194, (2560,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf195, (2560,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf196, (2560,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf197, (2560, 2560), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf198, (2560,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf199, (2560, 2560), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf200, (2560,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf201, (2560, 2560), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf202, (2560,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf203, (2560, 2560), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf204, (2560,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf205, (2560,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf206, (2560,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf207, (10240, 2560), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf208, (10240,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf209, (2560, 10240), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf210, (2560,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf211, (2560,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf212, (2560,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf213, (2560, 2560), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf214, (2560,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf215, (2560, 2560), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf216, (2560,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf217, (2560, 2560), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf218, (2560,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf219, (2560, 2560), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf220, (2560,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf221, (2560,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf222, (2560,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf223, (10240, 2560), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf224, (10240,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf225, (2560, 10240), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf226, (2560,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf227, (2560,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf228, (2560,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf229, (2560, 2560), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf230, (2560,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf231, (2560, 2560), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf232, (2560,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf233, (2560, 2560), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf234, (2560,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf235, (2560, 2560), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf236, (2560,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf237, (2560,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf238, (2560,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf239, (10240, 2560), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf240, (10240,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf241, (2560, 10240), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf242, (2560,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf243, (2560,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf244, (2560,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf245, (2560, 2560), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf246, (2560,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf247, (2560, 2560), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf248, (2560,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf249, (2560, 2560), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf250, (2560,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf251, (2560, 2560), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf252, (2560,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf253, (2560,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf254, (2560,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf255, (10240, 2560), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf256, (10240,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf257, (2560, 10240), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf258, (2560,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf259, (2560,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf260, (2560,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf261, (2560, 2560), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf262, (2560,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf263, (2560, 2560), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf264, (2560,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf265, (2560, 2560), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf266, (2560,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf267, (2560, 2560), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf268, (2560,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf269, (2560,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf270, (2560,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf271, (10240, 2560), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf272, (10240,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf273, (2560, 10240), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf274, (2560,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf275, (2560,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf276, (2560,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf277, (2560, 2560), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf278, (2560,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf279, (2560, 2560), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf280, (2560,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf281, (2560, 2560), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf282, (2560,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf283, (2560, 2560), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf284, (2560,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf285, (2560,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf286, (2560,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf287, (10240, 2560), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf288, (10240,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf289, (2560, 10240), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf290, (2560,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf291, (2560,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf292, (2560,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf293, (2560, 2560), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf294, (2560,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf295, (2560, 2560), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf296, (2560,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf297, (2560, 2560), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf298, (2560,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf299, (2560, 2560), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf300, (2560,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf301, (2560,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf302, (2560,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf303, (10240, 2560), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf304, (10240,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf305, (2560, 10240), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf306, (2560,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf307, (2560,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf308, (2560,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf309, (2560, 2560), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf310, (2560,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf311, (2560, 2560), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf312, (2560,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf313, (2560, 2560), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf314, (2560,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf315, (2560, 2560), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf316, (2560,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf317, (2560,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf318, (2560,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf319, (10240, 2560), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf320, (10240,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf321, (2560, 10240), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf322, (2560,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf323, (2560,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf324, (2560,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf325, (2560, 2560), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf326, (2560,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf327, (2560, 2560), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf328, (2560,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf329, (2560, 2560), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf330, (2560,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf331, (2560, 2560), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf332, (2560,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf333, (2560,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf334, (2560,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf335, (10240, 2560), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf336, (10240,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf337, (2560, 10240), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf338, (2560,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf339, (2560,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf340, (2560,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf341, (2560, 2560), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf342, (2560,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf343, (2560, 2560), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf344, (2560,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf345, (2560, 2560), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf346, (2560,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf347, (2560, 2560), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf348, (2560,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf349, (2560,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf350, (2560,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf351, (10240, 2560), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf352, (10240,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf353, (2560, 10240), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf354, (2560,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf355, (2560,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf356, (2560,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf357, (2560, 2560), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf358, (2560,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf359, (2560, 2560), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf360, (2560,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf361, (2560, 2560), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf362, (2560,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf363, (2560, 2560), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf364, (2560,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf365, (2560,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf366, (2560,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf367, (10240, 2560), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf368, (10240,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf369, (2560, 10240), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf370, (2560,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf371, (2560,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf372, (2560,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf373, (2560, 2560), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf374, (2560,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf375, (2560, 2560), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf376, (2560,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf377, (2560, 2560), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf378, (2560,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 26214400, device=device(type='cuda', index=0))
    reader.tensor(buf379, (2560, 2560), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf380, (2560,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf381, (2560,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf382, (2560,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf383, (10240, 2560), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 40960, device=device(type='cuda', index=0))
    reader.tensor(buf384, (10240,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 104857600, device=device(type='cuda', index=0))
    reader.tensor(buf385, (2560, 10240), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf386, (2560,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf387, (2560,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf388, (2560,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf389, (4, 128), dtype=torch.int64, is_leaf=True)  # arg389_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)