
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding = torch.ops.aten.embedding.default(arg1_1, clone);  clone = None
        iota = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        div = torch.ops.aten.div.Tensor(convert_element_type, 1024);  convert_element_type = None
        pow_1 = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        iota_1 = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        permute_2 = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None
        sin = torch.ops.aten.sin.default(mul_1)
        cos = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand = torch.ops.aten.expand.default(unsqueeze_2, [-1, 8, -1]);  unsqueeze_2 = None
        device_put = torch.ops.prims.device_put.default(expand, device(type='cuda', index=0));  expand = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(device_put, torch.float32);  device_put = None
        slice_3 = torch.ops.aten.slice.Tensor(embedding, 0, -512, 9223372036854775807)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        permute_3 = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 4, 2]);  unsqueeze_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        permute_4 = torch.ops.aten.permute.default(unsqueeze_6, [3, 4, 1, 2, 0]);  unsqueeze_6 = None
        permute_5 = torch.ops.aten.permute.default(permute_3, [0, 1, 4, 2, 3]);  permute_3 = None
        view = torch.ops.aten.view.default(permute_5, [1, 4096, 1024]);  permute_5 = None
        permute_6 = torch.ops.aten.permute.default(permute_4, [4, 2, 3, 0, 1]);  permute_4 = None
        view_1 = torch.ops.aten.view.default(permute_6, [1, 1024, 1024]);  permute_6 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2 = torch.ops.aten.view.default(bmm, [512, 8, 1, 16, 64]);  bmm = None
        permute_7 = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute_7, [512, 8, 16, 64]);  permute_7 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
        permute_8 = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 3, 4, 2]);  unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(arg3_1, 3);  arg3_1 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
        permute_9 = torch.ops.aten.permute.default(unsqueeze_10, [3, 4, 1, 2, 0]);  unsqueeze_10 = None
        permute_10 = torch.ops.aten.permute.default(permute_8, [0, 1, 4, 2, 3]);  permute_8 = None
        view_4 = torch.ops.aten.view.default(permute_10, [1, 4096, 1024]);  permute_10 = None
        permute_11 = torch.ops.aten.permute.default(permute_9, [4, 2, 3, 0, 1]);  permute_9 = None
        view_5 = torch.ops.aten.view.default(permute_11, [1, 1024, 1024]);  permute_11 = None
        bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
        view_6 = torch.ops.aten.view.default(bmm_1, [512, 8, 1, 16, 64]);  bmm_1 = None
        permute_12 = torch.ops.aten.permute.default(view_6, [0, 1, 3, 4, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(permute_12, [512, 8, 16, 64]);  permute_12 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        permute_13 = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 3, 4, 2]);  unsqueeze_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(arg4_1, 3);  arg4_1 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
        permute_14 = torch.ops.aten.permute.default(unsqueeze_14, [3, 4, 1, 2, 0]);  unsqueeze_14 = None
        permute_15 = torch.ops.aten.permute.default(permute_13, [0, 1, 4, 2, 3]);  permute_13 = None
        view_8 = torch.ops.aten.view.default(permute_15, [1, 4096, 1024]);  permute_15 = None
        permute_16 = torch.ops.aten.permute.default(permute_14, [4, 2, 3, 0, 1]);  permute_14 = None
        view_9 = torch.ops.aten.view.default(permute_16, [1, 1024, 1024]);  permute_16 = None
        bmm_2 = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
        view_10 = torch.ops.aten.view.default(bmm_2, [512, 8, 1, 16, 64]);  bmm_2 = None
        permute_17 = torch.ops.aten.permute.default(view_10, [0, 1, 3, 4, 2]);  view_10 = None
        view_11 = torch.ops.aten.view.default(permute_17, [512, 8, 16, 64]);  permute_17 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
        permute_18 = torch.ops.aten.permute.default(unsqueeze_16, [0, 1, 3, 4, 2]);  unsqueeze_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(arg6_1, 3);  arg6_1 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        permute_19 = torch.ops.aten.permute.default(unsqueeze_18, [3, 4, 1, 2, 0]);  unsqueeze_18 = None
        permute_20 = torch.ops.aten.permute.default(permute_18, [0, 1, 4, 2, 3]);  permute_18 = None
        view_12 = torch.ops.aten.view.default(permute_20, [1, 8192, 1024]);  permute_20 = None
        permute_21 = torch.ops.aten.permute.default(permute_19, [4, 2, 3, 0, 1]);  permute_19 = None
        view_13 = torch.ops.aten.view.default(permute_21, [1, 1024, 1024]);  permute_21 = None
        bmm_3 = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14 = torch.ops.aten.view.default(bmm_3, [1024, 8, 1, 16, 64]);  bmm_3 = None
        permute_22 = torch.ops.aten.permute.default(view_14, [0, 1, 3, 4, 2]);  view_14 = None
        view_15 = torch.ops.aten.view.default(permute_22, [1024, 8, 16, 64]);  permute_22 = None
        add = torch.ops.aten.add.Tensor(view_3, arg8_1);  arg8_1 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(add, 4);  add = None
        permute_23 = torch.ops.aten.permute.default(unsqueeze_19, [1, 2, 0, 4, 3]);  unsqueeze_19 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(view_7, 4);  view_7 = None
        permute_24 = torch.ops.aten.permute.default(unsqueeze_20, [1, 2, 4, 0, 3]);  unsqueeze_20 = None
        permute_25 = torch.ops.aten.permute.default(permute_23, [0, 1, 2, 4, 3]);  permute_23 = None
        view_16 = torch.ops.aten.view.default(permute_25, [128, 512, 64]);  permute_25 = None
        permute_26 = torch.ops.aten.permute.default(permute_24, [0, 1, 4, 3, 2]);  permute_24 = None
        view_17 = torch.ops.aten.view.default(permute_26, [128, 64, 512]);  permute_26 = None
        bmm_4 = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
        view_18 = torch.ops.aten.view.default(bmm_4, [8, 16, 512, 1, 512]);  bmm_4 = None
        permute_27 = torch.ops.aten.permute.default(view_18, [0, 1, 2, 4, 3]);  view_18 = None
        view_19 = torch.ops.aten.view.default(permute_27, [8, 16, 512, 512]);  permute_27 = None
        add_1 = torch.ops.aten.add.Tensor(view_3, arg7_1);  view_3 = arg7_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(add_1, 4);  add_1 = None
        permute_28 = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 0, 4, 3]);  unsqueeze_21 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(view_15, 4);  view_15 = None
        permute_29 = torch.ops.aten.permute.default(unsqueeze_22, [1, 2, 4, 0, 3]);  unsqueeze_22 = None
        permute_30 = torch.ops.aten.permute.default(permute_28, [0, 1, 2, 4, 3]);  permute_28 = None
        view_20 = torch.ops.aten.view.default(permute_30, [128, 512, 64]);  permute_30 = None
        permute_31 = torch.ops.aten.permute.default(permute_29, [0, 1, 4, 3, 2]);  permute_29 = None
        view_21 = torch.ops.aten.view.default(permute_31, [128, 64, 1024]);  permute_31 = None
        bmm_5 = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = view_21 = None
        view_22 = torch.ops.aten.view.default(bmm_5, [8, 16, 512, 1, 1024]);  bmm_5 = None
        permute_32 = torch.ops.aten.permute.default(view_22, [0, 1, 2, 4, 3]);  view_22 = None
        view_23 = torch.ops.aten.view.default(permute_32, [8, 16, 512, 1024]);  permute_32 = None
        view_24 = torch.ops.aten.view.default(view_23, [8, 16, 1024, 512]);  view_23 = None
        slice_6 = torch.ops.aten.slice.Tensor(view_24, 2, 1, 9223372036854775807);  view_24 = None
        view_25 = torch.ops.aten.view.default(slice_6, [8, 16, 512, 1023]);  slice_6 = None
        iota_2 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index = torch.ops.aten.index.Tensor(view_25, [None, None, None, iota_2]);  view_25 = iota_2 = None
        add_2 = torch.ops.aten.add.Tensor(view_19, index);  view_19 = index = None
        add_3 = torch.ops.aten.add.Tensor(add_2, 0);  add_2 = None
        mul_tensor_46 = torch.ops.aten.mul.Tensor(add_3, 1);  add_3 = None
        amax_default_23 = torch.ops.aten.amax.default(mul_tensor_46, [3], True)
        sub_tensor_23 = torch.ops.aten.sub.Tensor(mul_tensor_46, amax_default_23);  mul_tensor_46 = amax_default_23 = None
        mul_tensor_47 = torch.ops.aten.mul.Tensor(sub_tensor_23, 0.125);  sub_tensor_23 = None
        exp = torch.ops.aten.exp.default(mul_tensor_47);  mul_tensor_47 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(div_1, 4);  div_1 = None
        permute_33 = torch.ops.aten.permute.default(unsqueeze_23, [2, 0, 1, 4, 3]);  unsqueeze_23 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(view_11, 4);  view_11 = None
        permute_34 = torch.ops.aten.permute.default(unsqueeze_24, [4, 1, 2, 3, 0]);  unsqueeze_24 = None
        permute_35 = torch.ops.aten.permute.default(permute_33, [1, 2, 0, 4, 3]);  permute_33 = None
        view_26 = torch.ops.aten.view.default(permute_35, [128, 512, 512]);  permute_35 = None
        permute_36 = torch.ops.aten.permute.default(permute_34, [1, 2, 4, 3, 0]);  permute_34 = None
        view_27 = torch.ops.aten.view.default(permute_36, [128, 512, 64]);  permute_36 = None
        bmm_6 = torch.ops.aten.bmm.default(view_26, view_27);  view_26 = view_27 = None
        view_28 = torch.ops.aten.view.default(bmm_6, [8, 16, 512, 1, 64]);  bmm_6 = None
        permute_37 = torch.ops.aten.permute.default(view_28, [2, 0, 1, 4, 3]);  view_28 = None
        view_29 = torch.ops.aten.view.default(permute_37, [512, 8, 16, 64]);  permute_37 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(view_29, 4);  view_29 = None
        permute_38 = torch.ops.aten.permute.default(unsqueeze_25, [0, 1, 4, 3, 2]);  unsqueeze_25 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(arg5_1, 3);  arg5_1 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
        permute_39 = torch.ops.aten.permute.default(unsqueeze_27, [3, 4, 0, 2, 1]);  unsqueeze_27 = None
        permute_40 = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 4, 2]);  permute_38 = None
        clone_4 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_30 = torch.ops.aten.view.default(clone_4, [1, 4096, 1024]);  clone_4 = None
        permute_41 = torch.ops.aten.permute.default(permute_39, [3, 4, 2, 0, 1]);  permute_39 = None
        clone_5 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_31 = torch.ops.aten.view.default(clone_5, [1, 1024, 1024]);  clone_5 = None
        bmm_7 = torch.ops.aten.bmm.default(view_30, view_31);  view_30 = view_31 = None
        view_32 = torch.ops.aten.view.default(bmm_7, [512, 8, 1, 1, 1024]);  bmm_7 = None
        permute_42 = torch.ops.aten.permute.default(view_32, [0, 1, 4, 2, 3]);  view_32 = None
        view_33 = torch.ops.aten.view.default(permute_42, [512, 8, 1024]);  permute_42 = None
        add_4 = torch.ops.aten.add.Tensor(view_33, embedding);  view_33 = embedding = None
        var_mean = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_5 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_4, getitem_1);  add_4 = getitem_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg9_1);  mul_3 = arg9_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_4, arg10_1);  mul_4 = arg10_1 = None
        view_34 = torch.ops.aten.view.default(add_6, [4096, 1024])
        permute_43 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm = torch.ops.aten.addmm.default(arg14_1, view_34, permute_43);  arg14_1 = view_34 = permute_43 = None
        view_35 = torch.ops.aten.view.default(addmm, [512, 8, 4096]);  addmm = None
        mul_5 = torch.ops.aten.mul.Tensor(view_35, 0.5)
        mul_6 = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
        erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_7);  mul_5 = add_7 = None
        view_36 = torch.ops.aten.view.default(mul_7, [4096, 4096]);  mul_7 = None
        permute_44 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg16_1, view_36, permute_44);  arg16_1 = view_36 = permute_44 = None
        view_37 = torch.ops.aten.view.default(addmm_1, [512, 8, 1024]);  addmm_1 = None
        add_8 = torch.ops.aten.add.Tensor(view_37, add_6);  view_37 = add_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_3);  add_8 = getitem_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg11_1);  mul_8 = arg11_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_9, arg12_1);  mul_9 = arg12_1 = None
        slice_11 = torch.ops.aten.slice.Tensor(add_10, 0, -512, 9223372036854775807)
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(add_10, 3)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, 4);  unsqueeze_28 = None
        permute_45 = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 3, 4, 2]);  unsqueeze_29 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(arg17_1, 3);  arg17_1 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
        permute_46 = torch.ops.aten.permute.default(unsqueeze_31, [3, 4, 1, 2, 0]);  unsqueeze_31 = None
        permute_47 = torch.ops.aten.permute.default(permute_45, [0, 1, 4, 2, 3]);  permute_45 = None
        view_38 = torch.ops.aten.view.default(permute_47, [1, 4096, 1024]);  permute_47 = None
        permute_48 = torch.ops.aten.permute.default(permute_46, [4, 2, 3, 0, 1]);  permute_46 = None
        view_39 = torch.ops.aten.view.default(permute_48, [1, 1024, 1024]);  permute_48 = None
        bmm_8 = torch.ops.aten.bmm.default(view_38, view_39);  view_38 = view_39 = None
        view_40 = torch.ops.aten.view.default(bmm_8, [512, 8, 1, 16, 64]);  bmm_8 = None
        permute_49 = torch.ops.aten.permute.default(view_40, [0, 1, 3, 4, 2]);  view_40 = None
        view_41 = torch.ops.aten.view.default(permute_49, [512, 8, 16, 64]);  permute_49 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(add_10, 3)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, 4);  unsqueeze_32 = None
        permute_50 = torch.ops.aten.permute.default(unsqueeze_33, [0, 1, 3, 4, 2]);  unsqueeze_33 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(arg18_1, 3);  arg18_1 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
        permute_51 = torch.ops.aten.permute.default(unsqueeze_35, [3, 4, 1, 2, 0]);  unsqueeze_35 = None
        permute_52 = torch.ops.aten.permute.default(permute_50, [0, 1, 4, 2, 3]);  permute_50 = None
        view_42 = torch.ops.aten.view.default(permute_52, [1, 4096, 1024]);  permute_52 = None
        permute_53 = torch.ops.aten.permute.default(permute_51, [4, 2, 3, 0, 1]);  permute_51 = None
        view_43 = torch.ops.aten.view.default(permute_53, [1, 1024, 1024]);  permute_53 = None
        bmm_9 = torch.ops.aten.bmm.default(view_42, view_43);  view_42 = view_43 = None
        view_44 = torch.ops.aten.view.default(bmm_9, [512, 8, 1, 16, 64]);  bmm_9 = None
        permute_54 = torch.ops.aten.permute.default(view_44, [0, 1, 3, 4, 2]);  view_44 = None
        view_45 = torch.ops.aten.view.default(permute_54, [512, 8, 16, 64]);  permute_54 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(add_10, 3)
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
        permute_55 = torch.ops.aten.permute.default(unsqueeze_37, [0, 1, 3, 4, 2]);  unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg19_1, 3);  arg19_1 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, 4);  unsqueeze_38 = None
        permute_56 = torch.ops.aten.permute.default(unsqueeze_39, [3, 4, 1, 2, 0]);  unsqueeze_39 = None
        permute_57 = torch.ops.aten.permute.default(permute_55, [0, 1, 4, 2, 3]);  permute_55 = None
        view_46 = torch.ops.aten.view.default(permute_57, [1, 4096, 1024]);  permute_57 = None
        permute_58 = torch.ops.aten.permute.default(permute_56, [4, 2, 3, 0, 1]);  permute_56 = None
        view_47 = torch.ops.aten.view.default(permute_58, [1, 1024, 1024]);  permute_58 = None
        bmm_10 = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
        view_48 = torch.ops.aten.view.default(bmm_10, [512, 8, 1, 16, 64]);  bmm_10 = None
        permute_59 = torch.ops.aten.permute.default(view_48, [0, 1, 3, 4, 2]);  view_48 = None
        view_49 = torch.ops.aten.view.default(permute_59, [512, 8, 16, 64]);  permute_59 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, 4);  unsqueeze_40 = None
        permute_60 = torch.ops.aten.permute.default(unsqueeze_41, [0, 1, 3, 4, 2]);  unsqueeze_41 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(arg21_1, 3);  arg21_1 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, 4);  unsqueeze_42 = None
        permute_61 = torch.ops.aten.permute.default(unsqueeze_43, [3, 4, 1, 2, 0]);  unsqueeze_43 = None
        permute_62 = torch.ops.aten.permute.default(permute_60, [0, 1, 4, 2, 3]);  permute_60 = None
        view_50 = torch.ops.aten.view.default(permute_62, [1, 8192, 1024]);  permute_62 = None
        permute_63 = torch.ops.aten.permute.default(permute_61, [4, 2, 3, 0, 1]);  permute_61 = None
        view_51 = torch.ops.aten.view.default(permute_63, [1, 1024, 1024]);  permute_63 = None
        bmm_11 = torch.ops.aten.bmm.default(view_50, view_51);  view_50 = view_51 = None
        view_52 = torch.ops.aten.view.default(bmm_11, [1024, 8, 1, 16, 64]);  bmm_11 = None
        permute_64 = torch.ops.aten.permute.default(view_52, [0, 1, 3, 4, 2]);  view_52 = None
        view_53 = torch.ops.aten.view.default(permute_64, [1024, 8, 16, 64]);  permute_64 = None
        add_11 = torch.ops.aten.add.Tensor(view_41, arg23_1);  arg23_1 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(add_11, 4);  add_11 = None
        permute_65 = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 0, 4, 3]);  unsqueeze_44 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(view_45, 4);  view_45 = None
        permute_66 = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 4, 0, 3]);  unsqueeze_45 = None
        permute_67 = torch.ops.aten.permute.default(permute_65, [0, 1, 2, 4, 3]);  permute_65 = None
        view_54 = torch.ops.aten.view.default(permute_67, [128, 512, 64]);  permute_67 = None
        permute_68 = torch.ops.aten.permute.default(permute_66, [0, 1, 4, 3, 2]);  permute_66 = None
        view_55 = torch.ops.aten.view.default(permute_68, [128, 64, 512]);  permute_68 = None
        bmm_12 = torch.ops.aten.bmm.default(view_54, view_55);  view_54 = view_55 = None
        view_56 = torch.ops.aten.view.default(bmm_12, [8, 16, 512, 1, 512]);  bmm_12 = None
        permute_69 = torch.ops.aten.permute.default(view_56, [0, 1, 2, 4, 3]);  view_56 = None
        view_57 = torch.ops.aten.view.default(permute_69, [8, 16, 512, 512]);  permute_69 = None
        add_12 = torch.ops.aten.add.Tensor(view_41, arg22_1);  view_41 = arg22_1 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(add_12, 4);  add_12 = None
        permute_70 = torch.ops.aten.permute.default(unsqueeze_46, [1, 2, 0, 4, 3]);  unsqueeze_46 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(view_53, 4);  view_53 = None
        permute_71 = torch.ops.aten.permute.default(unsqueeze_47, [1, 2, 4, 0, 3]);  unsqueeze_47 = None
        permute_72 = torch.ops.aten.permute.default(permute_70, [0, 1, 2, 4, 3]);  permute_70 = None
        view_58 = torch.ops.aten.view.default(permute_72, [128, 512, 64]);  permute_72 = None
        permute_73 = torch.ops.aten.permute.default(permute_71, [0, 1, 4, 3, 2]);  permute_71 = None
        view_59 = torch.ops.aten.view.default(permute_73, [128, 64, 1024]);  permute_73 = None
        bmm_13 = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60 = torch.ops.aten.view.default(bmm_13, [8, 16, 512, 1, 1024]);  bmm_13 = None
        permute_74 = torch.ops.aten.permute.default(view_60, [0, 1, 2, 4, 3]);  view_60 = None
        view_61 = torch.ops.aten.view.default(permute_74, [8, 16, 512, 1024]);  permute_74 = None
        view_62 = torch.ops.aten.view.default(view_61, [8, 16, 1024, 512]);  view_61 = None
        slice_14 = torch.ops.aten.slice.Tensor(view_62, 2, 1, 9223372036854775807);  view_62 = None
        view_63 = torch.ops.aten.view.default(slice_14, [8, 16, 512, 1023]);  slice_14 = None
        iota_3 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_1 = torch.ops.aten.index.Tensor(view_63, [None, None, None, iota_3]);  view_63 = iota_3 = None
        add_13 = torch.ops.aten.add.Tensor(view_57, index_1);  view_57 = index_1 = None
        add_14 = torch.ops.aten.add.Tensor(add_13, 0);  add_13 = None
        mul_tensor_44 = torch.ops.aten.mul.Tensor(add_14, 1);  add_14 = None
        amax_default_22 = torch.ops.aten.amax.default(mul_tensor_44, [3], True)
        sub_tensor_22 = torch.ops.aten.sub.Tensor(mul_tensor_44, amax_default_22);  mul_tensor_44 = amax_default_22 = None
        mul_tensor_45 = torch.ops.aten.mul.Tensor(sub_tensor_22, 0.125);  sub_tensor_22 = None
        exp_1 = torch.ops.aten.exp.default(mul_tensor_45);  mul_tensor_45 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
        div_2 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(div_2, 4);  div_2 = None
        permute_75 = torch.ops.aten.permute.default(unsqueeze_48, [2, 0, 1, 4, 3]);  unsqueeze_48 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(view_49, 4);  view_49 = None
        permute_76 = torch.ops.aten.permute.default(unsqueeze_49, [4, 1, 2, 3, 0]);  unsqueeze_49 = None
        permute_77 = torch.ops.aten.permute.default(permute_75, [1, 2, 0, 4, 3]);  permute_75 = None
        view_64 = torch.ops.aten.view.default(permute_77, [128, 512, 512]);  permute_77 = None
        permute_78 = torch.ops.aten.permute.default(permute_76, [1, 2, 4, 3, 0]);  permute_76 = None
        view_65 = torch.ops.aten.view.default(permute_78, [128, 512, 64]);  permute_78 = None
        bmm_14 = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
        view_66 = torch.ops.aten.view.default(bmm_14, [8, 16, 512, 1, 64]);  bmm_14 = None
        permute_79 = torch.ops.aten.permute.default(view_66, [2, 0, 1, 4, 3]);  view_66 = None
        view_67 = torch.ops.aten.view.default(permute_79, [512, 8, 16, 64]);  permute_79 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(view_67, 4);  view_67 = None
        permute_80 = torch.ops.aten.permute.default(unsqueeze_50, [0, 1, 4, 3, 2]);  unsqueeze_50 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(arg20_1, 3);  arg20_1 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(unsqueeze_51, 4);  unsqueeze_51 = None
        permute_81 = torch.ops.aten.permute.default(unsqueeze_52, [3, 4, 0, 2, 1]);  unsqueeze_52 = None
        permute_82 = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 4, 2]);  permute_80 = None
        clone_10 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_68 = torch.ops.aten.view.default(clone_10, [1, 4096, 1024]);  clone_10 = None
        permute_83 = torch.ops.aten.permute.default(permute_81, [3, 4, 2, 0, 1]);  permute_81 = None
        clone_11 = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_69 = torch.ops.aten.view.default(clone_11, [1, 1024, 1024]);  clone_11 = None
        bmm_15 = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None
        view_70 = torch.ops.aten.view.default(bmm_15, [512, 8, 1, 1, 1024]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(view_70, [0, 1, 4, 2, 3]);  view_70 = None
        view_71 = torch.ops.aten.view.default(permute_84, [512, 8, 1024]);  permute_84 = None
        add_15 = torch.ops.aten.add.Tensor(view_71, add_10);  view_71 = add_10 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, arg24_1);  mul_11 = arg24_1 = None
        add_17 = torch.ops.aten.add.Tensor(mul_12, arg25_1);  mul_12 = arg25_1 = None
        view_72 = torch.ops.aten.view.default(add_17, [4096, 1024])
        permute_85 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg29_1, view_72, permute_85);  arg29_1 = view_72 = permute_85 = None
        view_73 = torch.ops.aten.view.default(addmm_2, [512, 8, 4096]);  addmm_2 = None
        mul_13 = torch.ops.aten.mul.Tensor(view_73, 0.5)
        mul_14 = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
        erf_1 = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_18 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
        view_74 = torch.ops.aten.view.default(mul_15, [4096, 4096]);  mul_15 = None
        permute_86 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg31_1, view_74, permute_86);  arg31_1 = view_74 = permute_86 = None
        view_75 = torch.ops.aten.view.default(addmm_3, [512, 8, 1024]);  addmm_3 = None
        add_19 = torch.ops.aten.add.Tensor(view_75, add_17);  view_75 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg26_1);  mul_16 = arg26_1 = None
        add_21 = torch.ops.aten.add.Tensor(mul_17, arg27_1);  mul_17 = arg27_1 = None
        slice_19 = torch.ops.aten.slice.Tensor(add_21, 0, -512, 9223372036854775807)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(add_21, 3)
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
        permute_87 = torch.ops.aten.permute.default(unsqueeze_54, [0, 1, 3, 4, 2]);  unsqueeze_54 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(arg32_1, 3);  arg32_1 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        permute_88 = torch.ops.aten.permute.default(unsqueeze_56, [3, 4, 1, 2, 0]);  unsqueeze_56 = None
        permute_89 = torch.ops.aten.permute.default(permute_87, [0, 1, 4, 2, 3]);  permute_87 = None
        view_76 = torch.ops.aten.view.default(permute_89, [1, 4096, 1024]);  permute_89 = None
        permute_90 = torch.ops.aten.permute.default(permute_88, [4, 2, 3, 0, 1]);  permute_88 = None
        view_77 = torch.ops.aten.view.default(permute_90, [1, 1024, 1024]);  permute_90 = None
        bmm_16 = torch.ops.aten.bmm.default(view_76, view_77);  view_76 = view_77 = None
        view_78 = torch.ops.aten.view.default(bmm_16, [512, 8, 1, 16, 64]);  bmm_16 = None
        permute_91 = torch.ops.aten.permute.default(view_78, [0, 1, 3, 4, 2]);  view_78 = None
        view_79 = torch.ops.aten.view.default(permute_91, [512, 8, 16, 64]);  permute_91 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(add_21, 3)
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(unsqueeze_57, 4);  unsqueeze_57 = None
        permute_92 = torch.ops.aten.permute.default(unsqueeze_58, [0, 1, 3, 4, 2]);  unsqueeze_58 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(arg33_1, 3);  arg33_1 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
        permute_93 = torch.ops.aten.permute.default(unsqueeze_60, [3, 4, 1, 2, 0]);  unsqueeze_60 = None
        permute_94 = torch.ops.aten.permute.default(permute_92, [0, 1, 4, 2, 3]);  permute_92 = None
        view_80 = torch.ops.aten.view.default(permute_94, [1, 4096, 1024]);  permute_94 = None
        permute_95 = torch.ops.aten.permute.default(permute_93, [4, 2, 3, 0, 1]);  permute_93 = None
        view_81 = torch.ops.aten.view.default(permute_95, [1, 1024, 1024]);  permute_95 = None
        bmm_17 = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
        view_82 = torch.ops.aten.view.default(bmm_17, [512, 8, 1, 16, 64]);  bmm_17 = None
        permute_96 = torch.ops.aten.permute.default(view_82, [0, 1, 3, 4, 2]);  view_82 = None
        view_83 = torch.ops.aten.view.default(permute_96, [512, 8, 16, 64]);  permute_96 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(add_21, 3)
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(unsqueeze_61, 4);  unsqueeze_61 = None
        permute_97 = torch.ops.aten.permute.default(unsqueeze_62, [0, 1, 3, 4, 2]);  unsqueeze_62 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(arg34_1, 3);  arg34_1 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        permute_98 = torch.ops.aten.permute.default(unsqueeze_64, [3, 4, 1, 2, 0]);  unsqueeze_64 = None
        permute_99 = torch.ops.aten.permute.default(permute_97, [0, 1, 4, 2, 3]);  permute_97 = None
        view_84 = torch.ops.aten.view.default(permute_99, [1, 4096, 1024]);  permute_99 = None
        permute_100 = torch.ops.aten.permute.default(permute_98, [4, 2, 3, 0, 1]);  permute_98 = None
        view_85 = torch.ops.aten.view.default(permute_100, [1, 1024, 1024]);  permute_100 = None
        bmm_18 = torch.ops.aten.bmm.default(view_84, view_85);  view_84 = view_85 = None
        view_86 = torch.ops.aten.view.default(bmm_18, [512, 8, 1, 16, 64]);  bmm_18 = None
        permute_101 = torch.ops.aten.permute.default(view_86, [0, 1, 3, 4, 2]);  view_86 = None
        view_87 = torch.ops.aten.view.default(permute_101, [512, 8, 16, 64]);  permute_101 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
        permute_102 = torch.ops.aten.permute.default(unsqueeze_66, [0, 1, 3, 4, 2]);  unsqueeze_66 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(arg36_1, 3);  arg36_1 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(unsqueeze_67, 4);  unsqueeze_67 = None
        permute_103 = torch.ops.aten.permute.default(unsqueeze_68, [3, 4, 1, 2, 0]);  unsqueeze_68 = None
        permute_104 = torch.ops.aten.permute.default(permute_102, [0, 1, 4, 2, 3]);  permute_102 = None
        view_88 = torch.ops.aten.view.default(permute_104, [1, 8192, 1024]);  permute_104 = None
        permute_105 = torch.ops.aten.permute.default(permute_103, [4, 2, 3, 0, 1]);  permute_103 = None
        view_89 = torch.ops.aten.view.default(permute_105, [1, 1024, 1024]);  permute_105 = None
        bmm_19 = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
        view_90 = torch.ops.aten.view.default(bmm_19, [1024, 8, 1, 16, 64]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(view_90, [0, 1, 3, 4, 2]);  view_90 = None
        view_91 = torch.ops.aten.view.default(permute_106, [1024, 8, 16, 64]);  permute_106 = None
        add_22 = torch.ops.aten.add.Tensor(view_79, arg38_1);  arg38_1 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(add_22, 4);  add_22 = None
        permute_107 = torch.ops.aten.permute.default(unsqueeze_69, [1, 2, 0, 4, 3]);  unsqueeze_69 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(view_83, 4);  view_83 = None
        permute_108 = torch.ops.aten.permute.default(unsqueeze_70, [1, 2, 4, 0, 3]);  unsqueeze_70 = None
        permute_109 = torch.ops.aten.permute.default(permute_107, [0, 1, 2, 4, 3]);  permute_107 = None
        view_92 = torch.ops.aten.view.default(permute_109, [128, 512, 64]);  permute_109 = None
        permute_110 = torch.ops.aten.permute.default(permute_108, [0, 1, 4, 3, 2]);  permute_108 = None
        view_93 = torch.ops.aten.view.default(permute_110, [128, 64, 512]);  permute_110 = None
        bmm_20 = torch.ops.aten.bmm.default(view_92, view_93);  view_92 = view_93 = None
        view_94 = torch.ops.aten.view.default(bmm_20, [8, 16, 512, 1, 512]);  bmm_20 = None
        permute_111 = torch.ops.aten.permute.default(view_94, [0, 1, 2, 4, 3]);  view_94 = None
        view_95 = torch.ops.aten.view.default(permute_111, [8, 16, 512, 512]);  permute_111 = None
        add_23 = torch.ops.aten.add.Tensor(view_79, arg37_1);  view_79 = arg37_1 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(add_23, 4);  add_23 = None
        permute_112 = torch.ops.aten.permute.default(unsqueeze_71, [1, 2, 0, 4, 3]);  unsqueeze_71 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(view_91, 4);  view_91 = None
        permute_113 = torch.ops.aten.permute.default(unsqueeze_72, [1, 2, 4, 0, 3]);  unsqueeze_72 = None
        permute_114 = torch.ops.aten.permute.default(permute_112, [0, 1, 2, 4, 3]);  permute_112 = None
        view_96 = torch.ops.aten.view.default(permute_114, [128, 512, 64]);  permute_114 = None
        permute_115 = torch.ops.aten.permute.default(permute_113, [0, 1, 4, 3, 2]);  permute_113 = None
        view_97 = torch.ops.aten.view.default(permute_115, [128, 64, 1024]);  permute_115 = None
        bmm_21 = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
        view_98 = torch.ops.aten.view.default(bmm_21, [8, 16, 512, 1, 1024]);  bmm_21 = None
        permute_116 = torch.ops.aten.permute.default(view_98, [0, 1, 2, 4, 3]);  view_98 = None
        view_99 = torch.ops.aten.view.default(permute_116, [8, 16, 512, 1024]);  permute_116 = None
        view_100 = torch.ops.aten.view.default(view_99, [8, 16, 1024, 512]);  view_99 = None
        slice_22 = torch.ops.aten.slice.Tensor(view_100, 2, 1, 9223372036854775807);  view_100 = None
        view_101 = torch.ops.aten.view.default(slice_22, [8, 16, 512, 1023]);  slice_22 = None
        iota_4 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_2 = torch.ops.aten.index.Tensor(view_101, [None, None, None, iota_4]);  view_101 = iota_4 = None
        add_24 = torch.ops.aten.add.Tensor(view_95, index_2);  view_95 = index_2 = None
        add_25 = torch.ops.aten.add.Tensor(add_24, 0);  add_24 = None
        mul_tensor_42 = torch.ops.aten.mul.Tensor(add_25, 1);  add_25 = None
        amax_default_21 = torch.ops.aten.amax.default(mul_tensor_42, [3], True)
        sub_tensor_21 = torch.ops.aten.sub.Tensor(mul_tensor_42, amax_default_21);  mul_tensor_42 = amax_default_21 = None
        mul_tensor_43 = torch.ops.aten.mul.Tensor(sub_tensor_21, 0.125);  sub_tensor_21 = None
        exp_2 = torch.ops.aten.exp.default(mul_tensor_43);  mul_tensor_43 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
        div_3 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(div_3, 4);  div_3 = None
        permute_117 = torch.ops.aten.permute.default(unsqueeze_73, [2, 0, 1, 4, 3]);  unsqueeze_73 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(view_87, 4);  view_87 = None
        permute_118 = torch.ops.aten.permute.default(unsqueeze_74, [4, 1, 2, 3, 0]);  unsqueeze_74 = None
        permute_119 = torch.ops.aten.permute.default(permute_117, [1, 2, 0, 4, 3]);  permute_117 = None
        view_102 = torch.ops.aten.view.default(permute_119, [128, 512, 512]);  permute_119 = None
        permute_120 = torch.ops.aten.permute.default(permute_118, [1, 2, 4, 3, 0]);  permute_118 = None
        view_103 = torch.ops.aten.view.default(permute_120, [128, 512, 64]);  permute_120 = None
        bmm_22 = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104 = torch.ops.aten.view.default(bmm_22, [8, 16, 512, 1, 64]);  bmm_22 = None
        permute_121 = torch.ops.aten.permute.default(view_104, [2, 0, 1, 4, 3]);  view_104 = None
        view_105 = torch.ops.aten.view.default(permute_121, [512, 8, 16, 64]);  permute_121 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(view_105, 4);  view_105 = None
        permute_122 = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg35_1, 3);  arg35_1 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
        permute_123 = torch.ops.aten.permute.default(unsqueeze_77, [3, 4, 0, 2, 1]);  unsqueeze_77 = None
        permute_124 = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 4, 2]);  permute_122 = None
        clone_16 = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_106 = torch.ops.aten.view.default(clone_16, [1, 4096, 1024]);  clone_16 = None
        permute_125 = torch.ops.aten.permute.default(permute_123, [3, 4, 2, 0, 1]);  permute_123 = None
        clone_17 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_107 = torch.ops.aten.view.default(clone_17, [1, 1024, 1024]);  clone_17 = None
        bmm_23 = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108 = torch.ops.aten.view.default(bmm_23, [512, 8, 1, 1, 1024]);  bmm_23 = None
        permute_126 = torch.ops.aten.permute.default(view_108, [0, 1, 4, 2, 3]);  view_108 = None
        view_109 = torch.ops.aten.view.default(permute_126, [512, 8, 1024]);  permute_126 = None
        add_26 = torch.ops.aten.add.Tensor(view_109, add_21);  view_109 = add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_26, getitem_9);  add_26 = getitem_9 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, arg39_1);  mul_19 = arg39_1 = None
        add_28 = torch.ops.aten.add.Tensor(mul_20, arg40_1);  mul_20 = arg40_1 = None
        view_110 = torch.ops.aten.view.default(add_28, [4096, 1024])
        permute_127 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg44_1, view_110, permute_127);  arg44_1 = view_110 = permute_127 = None
        view_111 = torch.ops.aten.view.default(addmm_4, [512, 8, 4096]);  addmm_4 = None
        mul_21 = torch.ops.aten.mul.Tensor(view_111, 0.5)
        mul_22 = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
        erf_2 = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_29 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_21, add_29);  mul_21 = add_29 = None
        view_112 = torch.ops.aten.view.default(mul_23, [4096, 4096]);  mul_23 = None
        permute_128 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg46_1, view_112, permute_128);  arg46_1 = view_112 = permute_128 = None
        view_113 = torch.ops.aten.view.default(addmm_5, [512, 8, 1024]);  addmm_5 = None
        add_30 = torch.ops.aten.add.Tensor(view_113, add_28);  view_113 = add_28 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_31 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_30, getitem_11);  add_30 = getitem_11 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg41_1);  mul_24 = arg41_1 = None
        add_32 = torch.ops.aten.add.Tensor(mul_25, arg42_1);  mul_25 = arg42_1 = None
        slice_27 = torch.ops.aten.slice.Tensor(add_32, 0, -512, 9223372036854775807)
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(add_32, 3)
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
        permute_129 = torch.ops.aten.permute.default(unsqueeze_79, [0, 1, 3, 4, 2]);  unsqueeze_79 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg47_1, 3);  arg47_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
        permute_130 = torch.ops.aten.permute.default(unsqueeze_81, [3, 4, 1, 2, 0]);  unsqueeze_81 = None
        permute_131 = torch.ops.aten.permute.default(permute_129, [0, 1, 4, 2, 3]);  permute_129 = None
        view_114 = torch.ops.aten.view.default(permute_131, [1, 4096, 1024]);  permute_131 = None
        permute_132 = torch.ops.aten.permute.default(permute_130, [4, 2, 3, 0, 1]);  permute_130 = None
        view_115 = torch.ops.aten.view.default(permute_132, [1, 1024, 1024]);  permute_132 = None
        bmm_24 = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
        view_116 = torch.ops.aten.view.default(bmm_24, [512, 8, 1, 16, 64]);  bmm_24 = None
        permute_133 = torch.ops.aten.permute.default(view_116, [0, 1, 3, 4, 2]);  view_116 = None
        view_117 = torch.ops.aten.view.default(permute_133, [512, 8, 16, 64]);  permute_133 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(add_32, 3)
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, 4);  unsqueeze_82 = None
        permute_134 = torch.ops.aten.permute.default(unsqueeze_83, [0, 1, 3, 4, 2]);  unsqueeze_83 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg48_1, 3);  arg48_1 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 4);  unsqueeze_84 = None
        permute_135 = torch.ops.aten.permute.default(unsqueeze_85, [3, 4, 1, 2, 0]);  unsqueeze_85 = None
        permute_136 = torch.ops.aten.permute.default(permute_134, [0, 1, 4, 2, 3]);  permute_134 = None
        view_118 = torch.ops.aten.view.default(permute_136, [1, 4096, 1024]);  permute_136 = None
        permute_137 = torch.ops.aten.permute.default(permute_135, [4, 2, 3, 0, 1]);  permute_135 = None
        view_119 = torch.ops.aten.view.default(permute_137, [1, 1024, 1024]);  permute_137 = None
        bmm_25 = torch.ops.aten.bmm.default(view_118, view_119);  view_118 = view_119 = None
        view_120 = torch.ops.aten.view.default(bmm_25, [512, 8, 1, 16, 64]);  bmm_25 = None
        permute_138 = torch.ops.aten.permute.default(view_120, [0, 1, 3, 4, 2]);  view_120 = None
        view_121 = torch.ops.aten.view.default(permute_138, [512, 8, 16, 64]);  permute_138 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(add_32, 3)
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, 4);  unsqueeze_86 = None
        permute_139 = torch.ops.aten.permute.default(unsqueeze_87, [0, 1, 3, 4, 2]);  unsqueeze_87 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg49_1, 3);  arg49_1 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
        permute_140 = torch.ops.aten.permute.default(unsqueeze_89, [3, 4, 1, 2, 0]);  unsqueeze_89 = None
        permute_141 = torch.ops.aten.permute.default(permute_139, [0, 1, 4, 2, 3]);  permute_139 = None
        view_122 = torch.ops.aten.view.default(permute_141, [1, 4096, 1024]);  permute_141 = None
        permute_142 = torch.ops.aten.permute.default(permute_140, [4, 2, 3, 0, 1]);  permute_140 = None
        view_123 = torch.ops.aten.view.default(permute_142, [1, 1024, 1024]);  permute_142 = None
        bmm_26 = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124 = torch.ops.aten.view.default(bmm_26, [512, 8, 1, 16, 64]);  bmm_26 = None
        permute_143 = torch.ops.aten.permute.default(view_124, [0, 1, 3, 4, 2]);  view_124 = None
        view_125 = torch.ops.aten.view.default(permute_143, [512, 8, 16, 64]);  permute_143 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, 4);  unsqueeze_90 = None
        permute_144 = torch.ops.aten.permute.default(unsqueeze_91, [0, 1, 3, 4, 2]);  unsqueeze_91 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg51_1, 3);  arg51_1 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, 4);  unsqueeze_92 = None
        permute_145 = torch.ops.aten.permute.default(unsqueeze_93, [3, 4, 1, 2, 0]);  unsqueeze_93 = None
        permute_146 = torch.ops.aten.permute.default(permute_144, [0, 1, 4, 2, 3]);  permute_144 = None
        view_126 = torch.ops.aten.view.default(permute_146, [1, 8192, 1024]);  permute_146 = None
        permute_147 = torch.ops.aten.permute.default(permute_145, [4, 2, 3, 0, 1]);  permute_145 = None
        view_127 = torch.ops.aten.view.default(permute_147, [1, 1024, 1024]);  permute_147 = None
        bmm_27 = torch.ops.aten.bmm.default(view_126, view_127);  view_126 = view_127 = None
        view_128 = torch.ops.aten.view.default(bmm_27, [1024, 8, 1, 16, 64]);  bmm_27 = None
        permute_148 = torch.ops.aten.permute.default(view_128, [0, 1, 3, 4, 2]);  view_128 = None
        view_129 = torch.ops.aten.view.default(permute_148, [1024, 8, 16, 64]);  permute_148 = None
        add_33 = torch.ops.aten.add.Tensor(view_117, arg53_1);  arg53_1 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(add_33, 4);  add_33 = None
        permute_149 = torch.ops.aten.permute.default(unsqueeze_94, [1, 2, 0, 4, 3]);  unsqueeze_94 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(view_121, 4);  view_121 = None
        permute_150 = torch.ops.aten.permute.default(unsqueeze_95, [1, 2, 4, 0, 3]);  unsqueeze_95 = None
        permute_151 = torch.ops.aten.permute.default(permute_149, [0, 1, 2, 4, 3]);  permute_149 = None
        view_130 = torch.ops.aten.view.default(permute_151, [128, 512, 64]);  permute_151 = None
        permute_152 = torch.ops.aten.permute.default(permute_150, [0, 1, 4, 3, 2]);  permute_150 = None
        view_131 = torch.ops.aten.view.default(permute_152, [128, 64, 512]);  permute_152 = None
        bmm_28 = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132 = torch.ops.aten.view.default(bmm_28, [8, 16, 512, 1, 512]);  bmm_28 = None
        permute_153 = torch.ops.aten.permute.default(view_132, [0, 1, 2, 4, 3]);  view_132 = None
        view_133 = torch.ops.aten.view.default(permute_153, [8, 16, 512, 512]);  permute_153 = None
        add_34 = torch.ops.aten.add.Tensor(view_117, arg52_1);  view_117 = arg52_1 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(add_34, 4);  add_34 = None
        permute_154 = torch.ops.aten.permute.default(unsqueeze_96, [1, 2, 0, 4, 3]);  unsqueeze_96 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(view_129, 4);  view_129 = None
        permute_155 = torch.ops.aten.permute.default(unsqueeze_97, [1, 2, 4, 0, 3]);  unsqueeze_97 = None
        permute_156 = torch.ops.aten.permute.default(permute_154, [0, 1, 2, 4, 3]);  permute_154 = None
        view_134 = torch.ops.aten.view.default(permute_156, [128, 512, 64]);  permute_156 = None
        permute_157 = torch.ops.aten.permute.default(permute_155, [0, 1, 4, 3, 2]);  permute_155 = None
        view_135 = torch.ops.aten.view.default(permute_157, [128, 64, 1024]);  permute_157 = None
        bmm_29 = torch.ops.aten.bmm.default(view_134, view_135);  view_134 = view_135 = None
        view_136 = torch.ops.aten.view.default(bmm_29, [8, 16, 512, 1, 1024]);  bmm_29 = None
        permute_158 = torch.ops.aten.permute.default(view_136, [0, 1, 2, 4, 3]);  view_136 = None
        view_137 = torch.ops.aten.view.default(permute_158, [8, 16, 512, 1024]);  permute_158 = None
        view_138 = torch.ops.aten.view.default(view_137, [8, 16, 1024, 512]);  view_137 = None
        slice_30 = torch.ops.aten.slice.Tensor(view_138, 2, 1, 9223372036854775807);  view_138 = None
        view_139 = torch.ops.aten.view.default(slice_30, [8, 16, 512, 1023]);  slice_30 = None
        iota_5 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_3 = torch.ops.aten.index.Tensor(view_139, [None, None, None, iota_5]);  view_139 = iota_5 = None
        add_35 = torch.ops.aten.add.Tensor(view_133, index_3);  view_133 = index_3 = None
        add_36 = torch.ops.aten.add.Tensor(add_35, 0);  add_35 = None
        mul_tensor_40 = torch.ops.aten.mul.Tensor(add_36, 1);  add_36 = None
        amax_default_20 = torch.ops.aten.amax.default(mul_tensor_40, [3], True)
        sub_tensor_20 = torch.ops.aten.sub.Tensor(mul_tensor_40, amax_default_20);  mul_tensor_40 = amax_default_20 = None
        mul_tensor_41 = torch.ops.aten.mul.Tensor(sub_tensor_20, 0.125);  sub_tensor_20 = None
        exp_3 = torch.ops.aten.exp.default(mul_tensor_41);  mul_tensor_41 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
        div_4 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(div_4, 4);  div_4 = None
        permute_159 = torch.ops.aten.permute.default(unsqueeze_98, [2, 0, 1, 4, 3]);  unsqueeze_98 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(view_125, 4);  view_125 = None
        permute_160 = torch.ops.aten.permute.default(unsqueeze_99, [4, 1, 2, 3, 0]);  unsqueeze_99 = None
        permute_161 = torch.ops.aten.permute.default(permute_159, [1, 2, 0, 4, 3]);  permute_159 = None
        view_140 = torch.ops.aten.view.default(permute_161, [128, 512, 512]);  permute_161 = None
        permute_162 = torch.ops.aten.permute.default(permute_160, [1, 2, 4, 3, 0]);  permute_160 = None
        view_141 = torch.ops.aten.view.default(permute_162, [128, 512, 64]);  permute_162 = None
        bmm_30 = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
        view_142 = torch.ops.aten.view.default(bmm_30, [8, 16, 512, 1, 64]);  bmm_30 = None
        permute_163 = torch.ops.aten.permute.default(view_142, [2, 0, 1, 4, 3]);  view_142 = None
        view_143 = torch.ops.aten.view.default(permute_163, [512, 8, 16, 64]);  permute_163 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(view_143, 4);  view_143 = None
        permute_164 = torch.ops.aten.permute.default(unsqueeze_100, [0, 1, 4, 3, 2]);  unsqueeze_100 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(arg50_1, 3);  arg50_1 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
        permute_165 = torch.ops.aten.permute.default(unsqueeze_102, [3, 4, 0, 2, 1]);  unsqueeze_102 = None
        permute_166 = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 4, 2]);  permute_164 = None
        clone_22 = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_144 = torch.ops.aten.view.default(clone_22, [1, 4096, 1024]);  clone_22 = None
        permute_167 = torch.ops.aten.permute.default(permute_165, [3, 4, 2, 0, 1]);  permute_165 = None
        clone_23 = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_145 = torch.ops.aten.view.default(clone_23, [1, 1024, 1024]);  clone_23 = None
        bmm_31 = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
        view_146 = torch.ops.aten.view.default(bmm_31, [512, 8, 1, 1, 1024]);  bmm_31 = None
        permute_168 = torch.ops.aten.permute.default(view_146, [0, 1, 4, 2, 3]);  view_146 = None
        view_147 = torch.ops.aten.view.default(permute_168, [512, 8, 1024]);  permute_168 = None
        add_37 = torch.ops.aten.add.Tensor(view_147, add_32);  view_147 = add_32 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
        mul_27 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, arg54_1);  mul_27 = arg54_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_28, arg55_1);  mul_28 = arg55_1 = None
        view_148 = torch.ops.aten.view.default(add_39, [4096, 1024])
        permute_169 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg59_1, view_148, permute_169);  arg59_1 = view_148 = permute_169 = None
        view_149 = torch.ops.aten.view.default(addmm_6, [512, 8, 4096]);  addmm_6 = None
        mul_29 = torch.ops.aten.mul.Tensor(view_149, 0.5)
        mul_30 = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
        erf_3 = torch.ops.aten.erf.default(mul_30);  mul_30 = None
        add_40 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_29, add_40);  mul_29 = add_40 = None
        view_150 = torch.ops.aten.view.default(mul_31, [4096, 4096]);  mul_31 = None
        permute_170 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg61_1, view_150, permute_170);  arg61_1 = view_150 = permute_170 = None
        view_151 = torch.ops.aten.view.default(addmm_7, [512, 8, 1024]);  addmm_7 = None
        add_41 = torch.ops.aten.add.Tensor(view_151, add_39);  view_151 = add_39 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_41, getitem_15);  add_41 = getitem_15 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg56_1);  mul_32 = arg56_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_33, arg57_1);  mul_33 = arg57_1 = None
        slice_35 = torch.ops.aten.slice.Tensor(add_43, 0, -512, 9223372036854775807)
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(add_43, 3)
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
        permute_171 = torch.ops.aten.permute.default(unsqueeze_104, [0, 1, 3, 4, 2]);  unsqueeze_104 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(arg62_1, 3);  arg62_1 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
        permute_172 = torch.ops.aten.permute.default(unsqueeze_106, [3, 4, 1, 2, 0]);  unsqueeze_106 = None
        permute_173 = torch.ops.aten.permute.default(permute_171, [0, 1, 4, 2, 3]);  permute_171 = None
        view_152 = torch.ops.aten.view.default(permute_173, [1, 4096, 1024]);  permute_173 = None
        permute_174 = torch.ops.aten.permute.default(permute_172, [4, 2, 3, 0, 1]);  permute_172 = None
        view_153 = torch.ops.aten.view.default(permute_174, [1, 1024, 1024]);  permute_174 = None
        bmm_32 = torch.ops.aten.bmm.default(view_152, view_153);  view_152 = view_153 = None
        view_154 = torch.ops.aten.view.default(bmm_32, [512, 8, 1, 16, 64]);  bmm_32 = None
        permute_175 = torch.ops.aten.permute.default(view_154, [0, 1, 3, 4, 2]);  view_154 = None
        view_155 = torch.ops.aten.view.default(permute_175, [512, 8, 16, 64]);  permute_175 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(add_43, 3)
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(unsqueeze_107, 4);  unsqueeze_107 = None
        permute_176 = torch.ops.aten.permute.default(unsqueeze_108, [0, 1, 3, 4, 2]);  unsqueeze_108 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(arg63_1, 3);  arg63_1 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(unsqueeze_109, 4);  unsqueeze_109 = None
        permute_177 = torch.ops.aten.permute.default(unsqueeze_110, [3, 4, 1, 2, 0]);  unsqueeze_110 = None
        permute_178 = torch.ops.aten.permute.default(permute_176, [0, 1, 4, 2, 3]);  permute_176 = None
        view_156 = torch.ops.aten.view.default(permute_178, [1, 4096, 1024]);  permute_178 = None
        permute_179 = torch.ops.aten.permute.default(permute_177, [4, 2, 3, 0, 1]);  permute_177 = None
        view_157 = torch.ops.aten.view.default(permute_179, [1, 1024, 1024]);  permute_179 = None
        bmm_33 = torch.ops.aten.bmm.default(view_156, view_157);  view_156 = view_157 = None
        view_158 = torch.ops.aten.view.default(bmm_33, [512, 8, 1, 16, 64]);  bmm_33 = None
        permute_180 = torch.ops.aten.permute.default(view_158, [0, 1, 3, 4, 2]);  view_158 = None
        view_159 = torch.ops.aten.view.default(permute_180, [512, 8, 16, 64]);  permute_180 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(add_43, 3)
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(unsqueeze_111, 4);  unsqueeze_111 = None
        permute_181 = torch.ops.aten.permute.default(unsqueeze_112, [0, 1, 3, 4, 2]);  unsqueeze_112 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(arg64_1, 3);  arg64_1 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
        permute_182 = torch.ops.aten.permute.default(unsqueeze_114, [3, 4, 1, 2, 0]);  unsqueeze_114 = None
        permute_183 = torch.ops.aten.permute.default(permute_181, [0, 1, 4, 2, 3]);  permute_181 = None
        view_160 = torch.ops.aten.view.default(permute_183, [1, 4096, 1024]);  permute_183 = None
        permute_184 = torch.ops.aten.permute.default(permute_182, [4, 2, 3, 0, 1]);  permute_182 = None
        view_161 = torch.ops.aten.view.default(permute_184, [1, 1024, 1024]);  permute_184 = None
        bmm_34 = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
        view_162 = torch.ops.aten.view.default(bmm_34, [512, 8, 1, 16, 64]);  bmm_34 = None
        permute_185 = torch.ops.aten.permute.default(view_162, [0, 1, 3, 4, 2]);  view_162 = None
        view_163 = torch.ops.aten.view.default(permute_185, [512, 8, 16, 64]);  permute_185 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(unsqueeze_115, 4);  unsqueeze_115 = None
        permute_186 = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 3, 4, 2]);  unsqueeze_116 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(arg66_1, 3);  arg66_1 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
        permute_187 = torch.ops.aten.permute.default(unsqueeze_118, [3, 4, 1, 2, 0]);  unsqueeze_118 = None
        permute_188 = torch.ops.aten.permute.default(permute_186, [0, 1, 4, 2, 3]);  permute_186 = None
        view_164 = torch.ops.aten.view.default(permute_188, [1, 8192, 1024]);  permute_188 = None
        permute_189 = torch.ops.aten.permute.default(permute_187, [4, 2, 3, 0, 1]);  permute_187 = None
        view_165 = torch.ops.aten.view.default(permute_189, [1, 1024, 1024]);  permute_189 = None
        bmm_35 = torch.ops.aten.bmm.default(view_164, view_165);  view_164 = view_165 = None
        view_166 = torch.ops.aten.view.default(bmm_35, [1024, 8, 1, 16, 64]);  bmm_35 = None
        permute_190 = torch.ops.aten.permute.default(view_166, [0, 1, 3, 4, 2]);  view_166 = None
        view_167 = torch.ops.aten.view.default(permute_190, [1024, 8, 16, 64]);  permute_190 = None
        add_44 = torch.ops.aten.add.Tensor(view_155, arg68_1);  arg68_1 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(add_44, 4);  add_44 = None
        permute_191 = torch.ops.aten.permute.default(unsqueeze_119, [1, 2, 0, 4, 3]);  unsqueeze_119 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(view_159, 4);  view_159 = None
        permute_192 = torch.ops.aten.permute.default(unsqueeze_120, [1, 2, 4, 0, 3]);  unsqueeze_120 = None
        permute_193 = torch.ops.aten.permute.default(permute_191, [0, 1, 2, 4, 3]);  permute_191 = None
        view_168 = torch.ops.aten.view.default(permute_193, [128, 512, 64]);  permute_193 = None
        permute_194 = torch.ops.aten.permute.default(permute_192, [0, 1, 4, 3, 2]);  permute_192 = None
        view_169 = torch.ops.aten.view.default(permute_194, [128, 64, 512]);  permute_194 = None
        bmm_36 = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
        view_170 = torch.ops.aten.view.default(bmm_36, [8, 16, 512, 1, 512]);  bmm_36 = None
        permute_195 = torch.ops.aten.permute.default(view_170, [0, 1, 2, 4, 3]);  view_170 = None
        view_171 = torch.ops.aten.view.default(permute_195, [8, 16, 512, 512]);  permute_195 = None
        add_45 = torch.ops.aten.add.Tensor(view_155, arg67_1);  view_155 = arg67_1 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(add_45, 4);  add_45 = None
        permute_196 = torch.ops.aten.permute.default(unsqueeze_121, [1, 2, 0, 4, 3]);  unsqueeze_121 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(view_167, 4);  view_167 = None
        permute_197 = torch.ops.aten.permute.default(unsqueeze_122, [1, 2, 4, 0, 3]);  unsqueeze_122 = None
        permute_198 = torch.ops.aten.permute.default(permute_196, [0, 1, 2, 4, 3]);  permute_196 = None
        view_172 = torch.ops.aten.view.default(permute_198, [128, 512, 64]);  permute_198 = None
        permute_199 = torch.ops.aten.permute.default(permute_197, [0, 1, 4, 3, 2]);  permute_197 = None
        view_173 = torch.ops.aten.view.default(permute_199, [128, 64, 1024]);  permute_199 = None
        bmm_37 = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = view_173 = None
        view_174 = torch.ops.aten.view.default(bmm_37, [8, 16, 512, 1, 1024]);  bmm_37 = None
        permute_200 = torch.ops.aten.permute.default(view_174, [0, 1, 2, 4, 3]);  view_174 = None
        view_175 = torch.ops.aten.view.default(permute_200, [8, 16, 512, 1024]);  permute_200 = None
        view_176 = torch.ops.aten.view.default(view_175, [8, 16, 1024, 512]);  view_175 = None
        slice_38 = torch.ops.aten.slice.Tensor(view_176, 2, 1, 9223372036854775807);  view_176 = None
        view_177 = torch.ops.aten.view.default(slice_38, [8, 16, 512, 1023]);  slice_38 = None
        iota_6 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_4 = torch.ops.aten.index.Tensor(view_177, [None, None, None, iota_6]);  view_177 = iota_6 = None
        add_46 = torch.ops.aten.add.Tensor(view_171, index_4);  view_171 = index_4 = None
        add_47 = torch.ops.aten.add.Tensor(add_46, 0);  add_46 = None
        mul_tensor_38 = torch.ops.aten.mul.Tensor(add_47, 1);  add_47 = None
        amax_default_19 = torch.ops.aten.amax.default(mul_tensor_38, [3], True)
        sub_tensor_19 = torch.ops.aten.sub.Tensor(mul_tensor_38, amax_default_19);  mul_tensor_38 = amax_default_19 = None
        mul_tensor_39 = torch.ops.aten.mul.Tensor(sub_tensor_19, 0.125);  sub_tensor_19 = None
        exp_4 = torch.ops.aten.exp.default(mul_tensor_39);  mul_tensor_39 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
        div_5 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(div_5, 4);  div_5 = None
        permute_201 = torch.ops.aten.permute.default(unsqueeze_123, [2, 0, 1, 4, 3]);  unsqueeze_123 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(view_163, 4);  view_163 = None
        permute_202 = torch.ops.aten.permute.default(unsqueeze_124, [4, 1, 2, 3, 0]);  unsqueeze_124 = None
        permute_203 = torch.ops.aten.permute.default(permute_201, [1, 2, 0, 4, 3]);  permute_201 = None
        view_178 = torch.ops.aten.view.default(permute_203, [128, 512, 512]);  permute_203 = None
        permute_204 = torch.ops.aten.permute.default(permute_202, [1, 2, 4, 3, 0]);  permute_202 = None
        view_179 = torch.ops.aten.view.default(permute_204, [128, 512, 64]);  permute_204 = None
        bmm_38 = torch.ops.aten.bmm.default(view_178, view_179);  view_178 = view_179 = None
        view_180 = torch.ops.aten.view.default(bmm_38, [8, 16, 512, 1, 64]);  bmm_38 = None
        permute_205 = torch.ops.aten.permute.default(view_180, [2, 0, 1, 4, 3]);  view_180 = None
        view_181 = torch.ops.aten.view.default(permute_205, [512, 8, 16, 64]);  permute_205 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(view_181, 4);  view_181 = None
        permute_206 = torch.ops.aten.permute.default(unsqueeze_125, [0, 1, 4, 3, 2]);  unsqueeze_125 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(arg65_1, 3);  arg65_1 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, 4);  unsqueeze_126 = None
        permute_207 = torch.ops.aten.permute.default(unsqueeze_127, [3, 4, 0, 2, 1]);  unsqueeze_127 = None
        permute_208 = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 4, 2]);  permute_206 = None
        clone_28 = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_182 = torch.ops.aten.view.default(clone_28, [1, 4096, 1024]);  clone_28 = None
        permute_209 = torch.ops.aten.permute.default(permute_207, [3, 4, 2, 0, 1]);  permute_207 = None
        clone_29 = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_183 = torch.ops.aten.view.default(clone_29, [1, 1024, 1024]);  clone_29 = None
        bmm_39 = torch.ops.aten.bmm.default(view_182, view_183);  view_182 = view_183 = None
        view_184 = torch.ops.aten.view.default(bmm_39, [512, 8, 1, 1, 1024]);  bmm_39 = None
        permute_210 = torch.ops.aten.permute.default(view_184, [0, 1, 4, 2, 3]);  view_184 = None
        view_185 = torch.ops.aten.view.default(permute_210, [512, 8, 1024]);  permute_210 = None
        add_48 = torch.ops.aten.add.Tensor(view_185, add_43);  view_185 = add_43 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_48, getitem_17);  add_48 = getitem_17 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
        add_50 = torch.ops.aten.add.Tensor(mul_36, arg70_1);  mul_36 = arg70_1 = None
        view_186 = torch.ops.aten.view.default(add_50, [4096, 1024])
        permute_211 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg74_1, view_186, permute_211);  arg74_1 = view_186 = permute_211 = None
        view_187 = torch.ops.aten.view.default(addmm_8, [512, 8, 4096]);  addmm_8 = None
        mul_37 = torch.ops.aten.mul.Tensor(view_187, 0.5)
        mul_38 = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
        erf_4 = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_51 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_37, add_51);  mul_37 = add_51 = None
        view_188 = torch.ops.aten.view.default(mul_39, [4096, 4096]);  mul_39 = None
        permute_212 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg76_1, view_188, permute_212);  arg76_1 = view_188 = permute_212 = None
        view_189 = torch.ops.aten.view.default(addmm_9, [512, 8, 1024]);  addmm_9 = None
        add_52 = torch.ops.aten.add.Tensor(view_189, add_50);  view_189 = add_50 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_52, getitem_19);  add_52 = getitem_19 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg71_1);  mul_40 = arg71_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_41, arg72_1);  mul_41 = arg72_1 = None
        slice_43 = torch.ops.aten.slice.Tensor(add_54, 0, -512, 9223372036854775807)
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(add_54, 3)
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
        permute_213 = torch.ops.aten.permute.default(unsqueeze_129, [0, 1, 3, 4, 2]);  unsqueeze_129 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(arg77_1, 3);  arg77_1 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
        permute_214 = torch.ops.aten.permute.default(unsqueeze_131, [3, 4, 1, 2, 0]);  unsqueeze_131 = None
        permute_215 = torch.ops.aten.permute.default(permute_213, [0, 1, 4, 2, 3]);  permute_213 = None
        view_190 = torch.ops.aten.view.default(permute_215, [1, 4096, 1024]);  permute_215 = None
        permute_216 = torch.ops.aten.permute.default(permute_214, [4, 2, 3, 0, 1]);  permute_214 = None
        view_191 = torch.ops.aten.view.default(permute_216, [1, 1024, 1024]);  permute_216 = None
        bmm_40 = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
        view_192 = torch.ops.aten.view.default(bmm_40, [512, 8, 1, 16, 64]);  bmm_40 = None
        permute_217 = torch.ops.aten.permute.default(view_192, [0, 1, 3, 4, 2]);  view_192 = None
        view_193 = torch.ops.aten.view.default(permute_217, [512, 8, 16, 64]);  permute_217 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(add_54, 3)
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, 4);  unsqueeze_132 = None
        permute_218 = torch.ops.aten.permute.default(unsqueeze_133, [0, 1, 3, 4, 2]);  unsqueeze_133 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(arg78_1, 3);  arg78_1 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
        permute_219 = torch.ops.aten.permute.default(unsqueeze_135, [3, 4, 1, 2, 0]);  unsqueeze_135 = None
        permute_220 = torch.ops.aten.permute.default(permute_218, [0, 1, 4, 2, 3]);  permute_218 = None
        view_194 = torch.ops.aten.view.default(permute_220, [1, 4096, 1024]);  permute_220 = None
        permute_221 = torch.ops.aten.permute.default(permute_219, [4, 2, 3, 0, 1]);  permute_219 = None
        view_195 = torch.ops.aten.view.default(permute_221, [1, 1024, 1024]);  permute_221 = None
        bmm_41 = torch.ops.aten.bmm.default(view_194, view_195);  view_194 = view_195 = None
        view_196 = torch.ops.aten.view.default(bmm_41, [512, 8, 1, 16, 64]);  bmm_41 = None
        permute_222 = torch.ops.aten.permute.default(view_196, [0, 1, 3, 4, 2]);  view_196 = None
        view_197 = torch.ops.aten.view.default(permute_222, [512, 8, 16, 64]);  permute_222 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(add_54, 3)
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, 4);  unsqueeze_136 = None
        permute_223 = torch.ops.aten.permute.default(unsqueeze_137, [0, 1, 3, 4, 2]);  unsqueeze_137 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(arg79_1, 3);  arg79_1 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
        permute_224 = torch.ops.aten.permute.default(unsqueeze_139, [3, 4, 1, 2, 0]);  unsqueeze_139 = None
        permute_225 = torch.ops.aten.permute.default(permute_223, [0, 1, 4, 2, 3]);  permute_223 = None
        view_198 = torch.ops.aten.view.default(permute_225, [1, 4096, 1024]);  permute_225 = None
        permute_226 = torch.ops.aten.permute.default(permute_224, [4, 2, 3, 0, 1]);  permute_224 = None
        view_199 = torch.ops.aten.view.default(permute_226, [1, 1024, 1024]);  permute_226 = None
        bmm_42 = torch.ops.aten.bmm.default(view_198, view_199);  view_198 = view_199 = None
        view_200 = torch.ops.aten.view.default(bmm_42, [512, 8, 1, 16, 64]);  bmm_42 = None
        permute_227 = torch.ops.aten.permute.default(view_200, [0, 1, 3, 4, 2]);  view_200 = None
        view_201 = torch.ops.aten.view.default(permute_227, [512, 8, 16, 64]);  permute_227 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, 4);  unsqueeze_140 = None
        permute_228 = torch.ops.aten.permute.default(unsqueeze_141, [0, 1, 3, 4, 2]);  unsqueeze_141 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(arg81_1, 3);  arg81_1 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, 4);  unsqueeze_142 = None
        permute_229 = torch.ops.aten.permute.default(unsqueeze_143, [3, 4, 1, 2, 0]);  unsqueeze_143 = None
        permute_230 = torch.ops.aten.permute.default(permute_228, [0, 1, 4, 2, 3]);  permute_228 = None
        view_202 = torch.ops.aten.view.default(permute_230, [1, 8192, 1024]);  permute_230 = None
        permute_231 = torch.ops.aten.permute.default(permute_229, [4, 2, 3, 0, 1]);  permute_229 = None
        view_203 = torch.ops.aten.view.default(permute_231, [1, 1024, 1024]);  permute_231 = None
        bmm_43 = torch.ops.aten.bmm.default(view_202, view_203);  view_202 = view_203 = None
        view_204 = torch.ops.aten.view.default(bmm_43, [1024, 8, 1, 16, 64]);  bmm_43 = None
        permute_232 = torch.ops.aten.permute.default(view_204, [0, 1, 3, 4, 2]);  view_204 = None
        view_205 = torch.ops.aten.view.default(permute_232, [1024, 8, 16, 64]);  permute_232 = None
        add_55 = torch.ops.aten.add.Tensor(view_193, arg83_1);  arg83_1 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(add_55, 4);  add_55 = None
        permute_233 = torch.ops.aten.permute.default(unsqueeze_144, [1, 2, 0, 4, 3]);  unsqueeze_144 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(view_197, 4);  view_197 = None
        permute_234 = torch.ops.aten.permute.default(unsqueeze_145, [1, 2, 4, 0, 3]);  unsqueeze_145 = None
        permute_235 = torch.ops.aten.permute.default(permute_233, [0, 1, 2, 4, 3]);  permute_233 = None
        view_206 = torch.ops.aten.view.default(permute_235, [128, 512, 64]);  permute_235 = None
        permute_236 = torch.ops.aten.permute.default(permute_234, [0, 1, 4, 3, 2]);  permute_234 = None
        view_207 = torch.ops.aten.view.default(permute_236, [128, 64, 512]);  permute_236 = None
        bmm_44 = torch.ops.aten.bmm.default(view_206, view_207);  view_206 = view_207 = None
        view_208 = torch.ops.aten.view.default(bmm_44, [8, 16, 512, 1, 512]);  bmm_44 = None
        permute_237 = torch.ops.aten.permute.default(view_208, [0, 1, 2, 4, 3]);  view_208 = None
        view_209 = torch.ops.aten.view.default(permute_237, [8, 16, 512, 512]);  permute_237 = None
        add_56 = torch.ops.aten.add.Tensor(view_193, arg82_1);  view_193 = arg82_1 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(add_56, 4);  add_56 = None
        permute_238 = torch.ops.aten.permute.default(unsqueeze_146, [1, 2, 0, 4, 3]);  unsqueeze_146 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(view_205, 4);  view_205 = None
        permute_239 = torch.ops.aten.permute.default(unsqueeze_147, [1, 2, 4, 0, 3]);  unsqueeze_147 = None
        permute_240 = torch.ops.aten.permute.default(permute_238, [0, 1, 2, 4, 3]);  permute_238 = None
        view_210 = torch.ops.aten.view.default(permute_240, [128, 512, 64]);  permute_240 = None
        permute_241 = torch.ops.aten.permute.default(permute_239, [0, 1, 4, 3, 2]);  permute_239 = None
        view_211 = torch.ops.aten.view.default(permute_241, [128, 64, 1024]);  permute_241 = None
        bmm_45 = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212 = torch.ops.aten.view.default(bmm_45, [8, 16, 512, 1, 1024]);  bmm_45 = None
        permute_242 = torch.ops.aten.permute.default(view_212, [0, 1, 2, 4, 3]);  view_212 = None
        view_213 = torch.ops.aten.view.default(permute_242, [8, 16, 512, 1024]);  permute_242 = None
        view_214 = torch.ops.aten.view.default(view_213, [8, 16, 1024, 512]);  view_213 = None
        slice_46 = torch.ops.aten.slice.Tensor(view_214, 2, 1, 9223372036854775807);  view_214 = None
        view_215 = torch.ops.aten.view.default(slice_46, [8, 16, 512, 1023]);  slice_46 = None
        iota_7 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_5 = torch.ops.aten.index.Tensor(view_215, [None, None, None, iota_7]);  view_215 = iota_7 = None
        add_57 = torch.ops.aten.add.Tensor(view_209, index_5);  view_209 = index_5 = None
        add_58 = torch.ops.aten.add.Tensor(add_57, 0);  add_57 = None
        mul_tensor_36 = torch.ops.aten.mul.Tensor(add_58, 1);  add_58 = None
        amax_default_18 = torch.ops.aten.amax.default(mul_tensor_36, [3], True)
        sub_tensor_18 = torch.ops.aten.sub.Tensor(mul_tensor_36, amax_default_18);  mul_tensor_36 = amax_default_18 = None
        mul_tensor_37 = torch.ops.aten.mul.Tensor(sub_tensor_18, 0.125);  sub_tensor_18 = None
        exp_5 = torch.ops.aten.exp.default(mul_tensor_37);  mul_tensor_37 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
        div_6 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(div_6, 4);  div_6 = None
        permute_243 = torch.ops.aten.permute.default(unsqueeze_148, [2, 0, 1, 4, 3]);  unsqueeze_148 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(view_201, 4);  view_201 = None
        permute_244 = torch.ops.aten.permute.default(unsqueeze_149, [4, 1, 2, 3, 0]);  unsqueeze_149 = None
        permute_245 = torch.ops.aten.permute.default(permute_243, [1, 2, 0, 4, 3]);  permute_243 = None
        view_216 = torch.ops.aten.view.default(permute_245, [128, 512, 512]);  permute_245 = None
        permute_246 = torch.ops.aten.permute.default(permute_244, [1, 2, 4, 3, 0]);  permute_244 = None
        view_217 = torch.ops.aten.view.default(permute_246, [128, 512, 64]);  permute_246 = None
        bmm_46 = torch.ops.aten.bmm.default(view_216, view_217);  view_216 = view_217 = None
        view_218 = torch.ops.aten.view.default(bmm_46, [8, 16, 512, 1, 64]);  bmm_46 = None
        permute_247 = torch.ops.aten.permute.default(view_218, [2, 0, 1, 4, 3]);  view_218 = None
        view_219 = torch.ops.aten.view.default(permute_247, [512, 8, 16, 64]);  permute_247 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(view_219, 4);  view_219 = None
        permute_248 = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 4, 3, 2]);  unsqueeze_150 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(arg80_1, 3);  arg80_1 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
        permute_249 = torch.ops.aten.permute.default(unsqueeze_152, [3, 4, 0, 2, 1]);  unsqueeze_152 = None
        permute_250 = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 4, 2]);  permute_248 = None
        clone_34 = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_220 = torch.ops.aten.view.default(clone_34, [1, 4096, 1024]);  clone_34 = None
        permute_251 = torch.ops.aten.permute.default(permute_249, [3, 4, 2, 0, 1]);  permute_249 = None
        clone_35 = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
        view_221 = torch.ops.aten.view.default(clone_35, [1, 1024, 1024]);  clone_35 = None
        bmm_47 = torch.ops.aten.bmm.default(view_220, view_221);  view_220 = view_221 = None
        view_222 = torch.ops.aten.view.default(bmm_47, [512, 8, 1, 1, 1024]);  bmm_47 = None
        permute_252 = torch.ops.aten.permute.default(view_222, [0, 1, 4, 2, 3]);  view_222 = None
        view_223 = torch.ops.aten.view.default(permute_252, [512, 8, 1024]);  permute_252 = None
        add_59 = torch.ops.aten.add.Tensor(view_223, add_54);  view_223 = add_54 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_59, getitem_21);  add_59 = getitem_21 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, arg84_1);  mul_43 = arg84_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_44, arg85_1);  mul_44 = arg85_1 = None
        view_224 = torch.ops.aten.view.default(add_61, [4096, 1024])
        permute_253 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg89_1, view_224, permute_253);  arg89_1 = view_224 = permute_253 = None
        view_225 = torch.ops.aten.view.default(addmm_10, [512, 8, 4096]);  addmm_10 = None
        mul_45 = torch.ops.aten.mul.Tensor(view_225, 0.5)
        mul_46 = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
        erf_5 = torch.ops.aten.erf.default(mul_46);  mul_46 = None
        add_62 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_45, add_62);  mul_45 = add_62 = None
        view_226 = torch.ops.aten.view.default(mul_47, [4096, 4096]);  mul_47 = None
        permute_254 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg91_1, view_226, permute_254);  arg91_1 = view_226 = permute_254 = None
        view_227 = torch.ops.aten.view.default(addmm_11, [512, 8, 1024]);  addmm_11 = None
        add_63 = torch.ops.aten.add.Tensor(view_227, add_61);  view_227 = add_61 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_63, getitem_23);  add_63 = getitem_23 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg86_1);  mul_48 = arg86_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_49, arg87_1);  mul_49 = arg87_1 = None
        slice_51 = torch.ops.aten.slice.Tensor(add_65, 0, -512, 9223372036854775807)
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(add_65, 3)
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
        permute_255 = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 3, 4, 2]);  unsqueeze_154 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(arg92_1, 3);  arg92_1 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(unsqueeze_155, 4);  unsqueeze_155 = None
        permute_256 = torch.ops.aten.permute.default(unsqueeze_156, [3, 4, 1, 2, 0]);  unsqueeze_156 = None
        permute_257 = torch.ops.aten.permute.default(permute_255, [0, 1, 4, 2, 3]);  permute_255 = None
        view_228 = torch.ops.aten.view.default(permute_257, [1, 4096, 1024]);  permute_257 = None
        permute_258 = torch.ops.aten.permute.default(permute_256, [4, 2, 3, 0, 1]);  permute_256 = None
        view_229 = torch.ops.aten.view.default(permute_258, [1, 1024, 1024]);  permute_258 = None
        bmm_48 = torch.ops.aten.bmm.default(view_228, view_229);  view_228 = view_229 = None
        view_230 = torch.ops.aten.view.default(bmm_48, [512, 8, 1, 16, 64]);  bmm_48 = None
        permute_259 = torch.ops.aten.permute.default(view_230, [0, 1, 3, 4, 2]);  view_230 = None
        view_231 = torch.ops.aten.view.default(permute_259, [512, 8, 16, 64]);  permute_259 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(add_65, 3)
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(unsqueeze_157, 4);  unsqueeze_157 = None
        permute_260 = torch.ops.aten.permute.default(unsqueeze_158, [0, 1, 3, 4, 2]);  unsqueeze_158 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(arg93_1, 3);  arg93_1 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
        permute_261 = torch.ops.aten.permute.default(unsqueeze_160, [3, 4, 1, 2, 0]);  unsqueeze_160 = None
        permute_262 = torch.ops.aten.permute.default(permute_260, [0, 1, 4, 2, 3]);  permute_260 = None
        view_232 = torch.ops.aten.view.default(permute_262, [1, 4096, 1024]);  permute_262 = None
        permute_263 = torch.ops.aten.permute.default(permute_261, [4, 2, 3, 0, 1]);  permute_261 = None
        view_233 = torch.ops.aten.view.default(permute_263, [1, 1024, 1024]);  permute_263 = None
        bmm_49 = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
        view_234 = torch.ops.aten.view.default(bmm_49, [512, 8, 1, 16, 64]);  bmm_49 = None
        permute_264 = torch.ops.aten.permute.default(view_234, [0, 1, 3, 4, 2]);  view_234 = None
        view_235 = torch.ops.aten.view.default(permute_264, [512, 8, 16, 64]);  permute_264 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(add_65, 3)
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(unsqueeze_161, 4);  unsqueeze_161 = None
        permute_265 = torch.ops.aten.permute.default(unsqueeze_162, [0, 1, 3, 4, 2]);  unsqueeze_162 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(arg94_1, 3);  arg94_1 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(unsqueeze_163, 4);  unsqueeze_163 = None
        permute_266 = torch.ops.aten.permute.default(unsqueeze_164, [3, 4, 1, 2, 0]);  unsqueeze_164 = None
        permute_267 = torch.ops.aten.permute.default(permute_265, [0, 1, 4, 2, 3]);  permute_265 = None
        view_236 = torch.ops.aten.view.default(permute_267, [1, 4096, 1024]);  permute_267 = None
        permute_268 = torch.ops.aten.permute.default(permute_266, [4, 2, 3, 0, 1]);  permute_266 = None
        view_237 = torch.ops.aten.view.default(permute_268, [1, 1024, 1024]);  permute_268 = None
        bmm_50 = torch.ops.aten.bmm.default(view_236, view_237);  view_236 = view_237 = None
        view_238 = torch.ops.aten.view.default(bmm_50, [512, 8, 1, 16, 64]);  bmm_50 = None
        permute_269 = torch.ops.aten.permute.default(view_238, [0, 1, 3, 4, 2]);  view_238 = None
        view_239 = torch.ops.aten.view.default(permute_269, [512, 8, 16, 64]);  permute_269 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(unsqueeze_165, 4);  unsqueeze_165 = None
        permute_270 = torch.ops.aten.permute.default(unsqueeze_166, [0, 1, 3, 4, 2]);  unsqueeze_166 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(arg96_1, 3);  arg96_1 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
        permute_271 = torch.ops.aten.permute.default(unsqueeze_168, [3, 4, 1, 2, 0]);  unsqueeze_168 = None
        permute_272 = torch.ops.aten.permute.default(permute_270, [0, 1, 4, 2, 3]);  permute_270 = None
        view_240 = torch.ops.aten.view.default(permute_272, [1, 8192, 1024]);  permute_272 = None
        permute_273 = torch.ops.aten.permute.default(permute_271, [4, 2, 3, 0, 1]);  permute_271 = None
        view_241 = torch.ops.aten.view.default(permute_273, [1, 1024, 1024]);  permute_273 = None
        bmm_51 = torch.ops.aten.bmm.default(view_240, view_241);  view_240 = view_241 = None
        view_242 = torch.ops.aten.view.default(bmm_51, [1024, 8, 1, 16, 64]);  bmm_51 = None
        permute_274 = torch.ops.aten.permute.default(view_242, [0, 1, 3, 4, 2]);  view_242 = None
        view_243 = torch.ops.aten.view.default(permute_274, [1024, 8, 16, 64]);  permute_274 = None
        add_66 = torch.ops.aten.add.Tensor(view_231, arg98_1);  arg98_1 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(add_66, 4);  add_66 = None
        permute_275 = torch.ops.aten.permute.default(unsqueeze_169, [1, 2, 0, 4, 3]);  unsqueeze_169 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(view_235, 4);  view_235 = None
        permute_276 = torch.ops.aten.permute.default(unsqueeze_170, [1, 2, 4, 0, 3]);  unsqueeze_170 = None
        permute_277 = torch.ops.aten.permute.default(permute_275, [0, 1, 2, 4, 3]);  permute_275 = None
        view_244 = torch.ops.aten.view.default(permute_277, [128, 512, 64]);  permute_277 = None
        permute_278 = torch.ops.aten.permute.default(permute_276, [0, 1, 4, 3, 2]);  permute_276 = None
        view_245 = torch.ops.aten.view.default(permute_278, [128, 64, 512]);  permute_278 = None
        bmm_52 = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
        view_246 = torch.ops.aten.view.default(bmm_52, [8, 16, 512, 1, 512]);  bmm_52 = None
        permute_279 = torch.ops.aten.permute.default(view_246, [0, 1, 2, 4, 3]);  view_246 = None
        view_247 = torch.ops.aten.view.default(permute_279, [8, 16, 512, 512]);  permute_279 = None
        add_67 = torch.ops.aten.add.Tensor(view_231, arg97_1);  view_231 = arg97_1 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(add_67, 4);  add_67 = None
        permute_280 = torch.ops.aten.permute.default(unsqueeze_171, [1, 2, 0, 4, 3]);  unsqueeze_171 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(view_243, 4);  view_243 = None
        permute_281 = torch.ops.aten.permute.default(unsqueeze_172, [1, 2, 4, 0, 3]);  unsqueeze_172 = None
        permute_282 = torch.ops.aten.permute.default(permute_280, [0, 1, 2, 4, 3]);  permute_280 = None
        view_248 = torch.ops.aten.view.default(permute_282, [128, 512, 64]);  permute_282 = None
        permute_283 = torch.ops.aten.permute.default(permute_281, [0, 1, 4, 3, 2]);  permute_281 = None
        view_249 = torch.ops.aten.view.default(permute_283, [128, 64, 1024]);  permute_283 = None
        bmm_53 = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
        view_250 = torch.ops.aten.view.default(bmm_53, [8, 16, 512, 1, 1024]);  bmm_53 = None
        permute_284 = torch.ops.aten.permute.default(view_250, [0, 1, 2, 4, 3]);  view_250 = None
        view_251 = torch.ops.aten.view.default(permute_284, [8, 16, 512, 1024]);  permute_284 = None
        view_252 = torch.ops.aten.view.default(view_251, [8, 16, 1024, 512]);  view_251 = None
        slice_54 = torch.ops.aten.slice.Tensor(view_252, 2, 1, 9223372036854775807);  view_252 = None
        view_253 = torch.ops.aten.view.default(slice_54, [8, 16, 512, 1023]);  slice_54 = None
        iota_8 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_6 = torch.ops.aten.index.Tensor(view_253, [None, None, None, iota_8]);  view_253 = iota_8 = None
        add_68 = torch.ops.aten.add.Tensor(view_247, index_6);  view_247 = index_6 = None
        add_69 = torch.ops.aten.add.Tensor(add_68, 0);  add_68 = None
        mul_tensor_34 = torch.ops.aten.mul.Tensor(add_69, 1);  add_69 = None
        amax_default_17 = torch.ops.aten.amax.default(mul_tensor_34, [3], True)
        sub_tensor_17 = torch.ops.aten.sub.Tensor(mul_tensor_34, amax_default_17);  mul_tensor_34 = amax_default_17 = None
        mul_tensor_35 = torch.ops.aten.mul.Tensor(sub_tensor_17, 0.125);  sub_tensor_17 = None
        exp_6 = torch.ops.aten.exp.default(mul_tensor_35);  mul_tensor_35 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
        div_7 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(div_7, 4);  div_7 = None
        permute_285 = torch.ops.aten.permute.default(unsqueeze_173, [2, 0, 1, 4, 3]);  unsqueeze_173 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(view_239, 4);  view_239 = None
        permute_286 = torch.ops.aten.permute.default(unsqueeze_174, [4, 1, 2, 3, 0]);  unsqueeze_174 = None
        permute_287 = torch.ops.aten.permute.default(permute_285, [1, 2, 0, 4, 3]);  permute_285 = None
        view_254 = torch.ops.aten.view.default(permute_287, [128, 512, 512]);  permute_287 = None
        permute_288 = torch.ops.aten.permute.default(permute_286, [1, 2, 4, 3, 0]);  permute_286 = None
        view_255 = torch.ops.aten.view.default(permute_288, [128, 512, 64]);  permute_288 = None
        bmm_54 = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
        view_256 = torch.ops.aten.view.default(bmm_54, [8, 16, 512, 1, 64]);  bmm_54 = None
        permute_289 = torch.ops.aten.permute.default(view_256, [2, 0, 1, 4, 3]);  view_256 = None
        view_257 = torch.ops.aten.view.default(permute_289, [512, 8, 16, 64]);  permute_289 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(view_257, 4);  view_257 = None
        permute_290 = torch.ops.aten.permute.default(unsqueeze_175, [0, 1, 4, 3, 2]);  unsqueeze_175 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(arg95_1, 3);  arg95_1 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, 4);  unsqueeze_176 = None
        permute_291 = torch.ops.aten.permute.default(unsqueeze_177, [3, 4, 0, 2, 1]);  unsqueeze_177 = None
        permute_292 = torch.ops.aten.permute.default(permute_290, [0, 1, 3, 4, 2]);  permute_290 = None
        clone_40 = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_258 = torch.ops.aten.view.default(clone_40, [1, 4096, 1024]);  clone_40 = None
        permute_293 = torch.ops.aten.permute.default(permute_291, [3, 4, 2, 0, 1]);  permute_291 = None
        clone_41 = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_259 = torch.ops.aten.view.default(clone_41, [1, 1024, 1024]);  clone_41 = None
        bmm_55 = torch.ops.aten.bmm.default(view_258, view_259);  view_258 = view_259 = None
        view_260 = torch.ops.aten.view.default(bmm_55, [512, 8, 1, 1, 1024]);  bmm_55 = None
        permute_294 = torch.ops.aten.permute.default(view_260, [0, 1, 4, 2, 3]);  view_260 = None
        view_261 = torch.ops.aten.view.default(permute_294, [512, 8, 1024]);  permute_294 = None
        add_70 = torch.ops.aten.add.Tensor(view_261, add_65);  view_261 = add_65 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_70, getitem_25);  add_70 = getitem_25 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, arg99_1);  mul_51 = arg99_1 = None
        add_72 = torch.ops.aten.add.Tensor(mul_52, arg100_1);  mul_52 = arg100_1 = None
        view_262 = torch.ops.aten.view.default(add_72, [4096, 1024])
        permute_295 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg104_1, view_262, permute_295);  arg104_1 = view_262 = permute_295 = None
        view_263 = torch.ops.aten.view.default(addmm_12, [512, 8, 4096]);  addmm_12 = None
        mul_53 = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_54 = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_6 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_73 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_53, add_73);  mul_53 = add_73 = None
        view_264 = torch.ops.aten.view.default(mul_55, [4096, 4096]);  mul_55 = None
        permute_296 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg106_1, view_264, permute_296);  arg106_1 = view_264 = permute_296 = None
        view_265 = torch.ops.aten.view.default(addmm_13, [512, 8, 1024]);  addmm_13 = None
        add_74 = torch.ops.aten.add.Tensor(view_265, add_72);  view_265 = add_72 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_74, getitem_27);  add_74 = getitem_27 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg101_1);  mul_56 = arg101_1 = None
        add_76 = torch.ops.aten.add.Tensor(mul_57, arg102_1);  mul_57 = arg102_1 = None
        slice_59 = torch.ops.aten.slice.Tensor(add_76, 0, -512, 9223372036854775807)
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(add_76, 3)
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, 4);  unsqueeze_178 = None
        permute_297 = torch.ops.aten.permute.default(unsqueeze_179, [0, 1, 3, 4, 2]);  unsqueeze_179 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(arg107_1, 3);  arg107_1 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
        permute_298 = torch.ops.aten.permute.default(unsqueeze_181, [3, 4, 1, 2, 0]);  unsqueeze_181 = None
        permute_299 = torch.ops.aten.permute.default(permute_297, [0, 1, 4, 2, 3]);  permute_297 = None
        view_266 = torch.ops.aten.view.default(permute_299, [1, 4096, 1024]);  permute_299 = None
        permute_300 = torch.ops.aten.permute.default(permute_298, [4, 2, 3, 0, 1]);  permute_298 = None
        view_267 = torch.ops.aten.view.default(permute_300, [1, 1024, 1024]);  permute_300 = None
        bmm_56 = torch.ops.aten.bmm.default(view_266, view_267);  view_266 = view_267 = None
        view_268 = torch.ops.aten.view.default(bmm_56, [512, 8, 1, 16, 64]);  bmm_56 = None
        permute_301 = torch.ops.aten.permute.default(view_268, [0, 1, 3, 4, 2]);  view_268 = None
        view_269 = torch.ops.aten.view.default(permute_301, [512, 8, 16, 64]);  permute_301 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(add_76, 3)
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, 4);  unsqueeze_182 = None
        permute_302 = torch.ops.aten.permute.default(unsqueeze_183, [0, 1, 3, 4, 2]);  unsqueeze_183 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(arg108_1, 3);  arg108_1 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
        permute_303 = torch.ops.aten.permute.default(unsqueeze_185, [3, 4, 1, 2, 0]);  unsqueeze_185 = None
        permute_304 = torch.ops.aten.permute.default(permute_302, [0, 1, 4, 2, 3]);  permute_302 = None
        view_270 = torch.ops.aten.view.default(permute_304, [1, 4096, 1024]);  permute_304 = None
        permute_305 = torch.ops.aten.permute.default(permute_303, [4, 2, 3, 0, 1]);  permute_303 = None
        view_271 = torch.ops.aten.view.default(permute_305, [1, 1024, 1024]);  permute_305 = None
        bmm_57 = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
        view_272 = torch.ops.aten.view.default(bmm_57, [512, 8, 1, 16, 64]);  bmm_57 = None
        permute_306 = torch.ops.aten.permute.default(view_272, [0, 1, 3, 4, 2]);  view_272 = None
        view_273 = torch.ops.aten.view.default(permute_306, [512, 8, 16, 64]);  permute_306 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(add_76, 3)
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, 4);  unsqueeze_186 = None
        permute_307 = torch.ops.aten.permute.default(unsqueeze_187, [0, 1, 3, 4, 2]);  unsqueeze_187 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(arg109_1, 3);  arg109_1 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, 4);  unsqueeze_188 = None
        permute_308 = torch.ops.aten.permute.default(unsqueeze_189, [3, 4, 1, 2, 0]);  unsqueeze_189 = None
        permute_309 = torch.ops.aten.permute.default(permute_307, [0, 1, 4, 2, 3]);  permute_307 = None
        view_274 = torch.ops.aten.view.default(permute_309, [1, 4096, 1024]);  permute_309 = None
        permute_310 = torch.ops.aten.permute.default(permute_308, [4, 2, 3, 0, 1]);  permute_308 = None
        view_275 = torch.ops.aten.view.default(permute_310, [1, 1024, 1024]);  permute_310 = None
        bmm_58 = torch.ops.aten.bmm.default(view_274, view_275);  view_274 = view_275 = None
        view_276 = torch.ops.aten.view.default(bmm_58, [512, 8, 1, 16, 64]);  bmm_58 = None
        permute_311 = torch.ops.aten.permute.default(view_276, [0, 1, 3, 4, 2]);  view_276 = None
        view_277 = torch.ops.aten.view.default(permute_311, [512, 8, 16, 64]);  permute_311 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, 4);  unsqueeze_190 = None
        permute_312 = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 3, 4, 2]);  unsqueeze_191 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(arg111_1, 3);  arg111_1 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
        permute_313 = torch.ops.aten.permute.default(unsqueeze_193, [3, 4, 1, 2, 0]);  unsqueeze_193 = None
        permute_314 = torch.ops.aten.permute.default(permute_312, [0, 1, 4, 2, 3]);  permute_312 = None
        view_278 = torch.ops.aten.view.default(permute_314, [1, 8192, 1024]);  permute_314 = None
        permute_315 = torch.ops.aten.permute.default(permute_313, [4, 2, 3, 0, 1]);  permute_313 = None
        view_279 = torch.ops.aten.view.default(permute_315, [1, 1024, 1024]);  permute_315 = None
        bmm_59 = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
        view_280 = torch.ops.aten.view.default(bmm_59, [1024, 8, 1, 16, 64]);  bmm_59 = None
        permute_316 = torch.ops.aten.permute.default(view_280, [0, 1, 3, 4, 2]);  view_280 = None
        view_281 = torch.ops.aten.view.default(permute_316, [1024, 8, 16, 64]);  permute_316 = None
        add_77 = torch.ops.aten.add.Tensor(view_269, arg113_1);  arg113_1 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(add_77, 4);  add_77 = None
        permute_317 = torch.ops.aten.permute.default(unsqueeze_194, [1, 2, 0, 4, 3]);  unsqueeze_194 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(view_273, 4);  view_273 = None
        permute_318 = torch.ops.aten.permute.default(unsqueeze_195, [1, 2, 4, 0, 3]);  unsqueeze_195 = None
        permute_319 = torch.ops.aten.permute.default(permute_317, [0, 1, 2, 4, 3]);  permute_317 = None
        view_282 = torch.ops.aten.view.default(permute_319, [128, 512, 64]);  permute_319 = None
        permute_320 = torch.ops.aten.permute.default(permute_318, [0, 1, 4, 3, 2]);  permute_318 = None
        view_283 = torch.ops.aten.view.default(permute_320, [128, 64, 512]);  permute_320 = None
        bmm_60 = torch.ops.aten.bmm.default(view_282, view_283);  view_282 = view_283 = None
        view_284 = torch.ops.aten.view.default(bmm_60, [8, 16, 512, 1, 512]);  bmm_60 = None
        permute_321 = torch.ops.aten.permute.default(view_284, [0, 1, 2, 4, 3]);  view_284 = None
        view_285 = torch.ops.aten.view.default(permute_321, [8, 16, 512, 512]);  permute_321 = None
        add_78 = torch.ops.aten.add.Tensor(view_269, arg112_1);  view_269 = arg112_1 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(add_78, 4);  add_78 = None
        permute_322 = torch.ops.aten.permute.default(unsqueeze_196, [1, 2, 0, 4, 3]);  unsqueeze_196 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(view_281, 4);  view_281 = None
        permute_323 = torch.ops.aten.permute.default(unsqueeze_197, [1, 2, 4, 0, 3]);  unsqueeze_197 = None
        permute_324 = torch.ops.aten.permute.default(permute_322, [0, 1, 2, 4, 3]);  permute_322 = None
        view_286 = torch.ops.aten.view.default(permute_324, [128, 512, 64]);  permute_324 = None
        permute_325 = torch.ops.aten.permute.default(permute_323, [0, 1, 4, 3, 2]);  permute_323 = None
        view_287 = torch.ops.aten.view.default(permute_325, [128, 64, 1024]);  permute_325 = None
        bmm_61 = torch.ops.aten.bmm.default(view_286, view_287);  view_286 = view_287 = None
        view_288 = torch.ops.aten.view.default(bmm_61, [8, 16, 512, 1, 1024]);  bmm_61 = None
        permute_326 = torch.ops.aten.permute.default(view_288, [0, 1, 2, 4, 3]);  view_288 = None
        view_289 = torch.ops.aten.view.default(permute_326, [8, 16, 512, 1024]);  permute_326 = None
        view_290 = torch.ops.aten.view.default(view_289, [8, 16, 1024, 512]);  view_289 = None
        slice_62 = torch.ops.aten.slice.Tensor(view_290, 2, 1, 9223372036854775807);  view_290 = None
        view_291 = torch.ops.aten.view.default(slice_62, [8, 16, 512, 1023]);  slice_62 = None
        iota_9 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_7 = torch.ops.aten.index.Tensor(view_291, [None, None, None, iota_9]);  view_291 = iota_9 = None
        add_79 = torch.ops.aten.add.Tensor(view_285, index_7);  view_285 = index_7 = None
        add_80 = torch.ops.aten.add.Tensor(add_79, 0);  add_79 = None
        mul_tensor_32 = torch.ops.aten.mul.Tensor(add_80, 1);  add_80 = None
        amax_default_16 = torch.ops.aten.amax.default(mul_tensor_32, [3], True)
        sub_tensor_16 = torch.ops.aten.sub.Tensor(mul_tensor_32, amax_default_16);  mul_tensor_32 = amax_default_16 = None
        mul_tensor_33 = torch.ops.aten.mul.Tensor(sub_tensor_16, 0.125);  sub_tensor_16 = None
        exp_7 = torch.ops.aten.exp.default(mul_tensor_33);  mul_tensor_33 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
        div_8 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(div_8, 4);  div_8 = None
        permute_327 = torch.ops.aten.permute.default(unsqueeze_198, [2, 0, 1, 4, 3]);  unsqueeze_198 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(view_277, 4);  view_277 = None
        permute_328 = torch.ops.aten.permute.default(unsqueeze_199, [4, 1, 2, 3, 0]);  unsqueeze_199 = None
        permute_329 = torch.ops.aten.permute.default(permute_327, [1, 2, 0, 4, 3]);  permute_327 = None
        view_292 = torch.ops.aten.view.default(permute_329, [128, 512, 512]);  permute_329 = None
        permute_330 = torch.ops.aten.permute.default(permute_328, [1, 2, 4, 3, 0]);  permute_328 = None
        view_293 = torch.ops.aten.view.default(permute_330, [128, 512, 64]);  permute_330 = None
        bmm_62 = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
        view_294 = torch.ops.aten.view.default(bmm_62, [8, 16, 512, 1, 64]);  bmm_62 = None
        permute_331 = torch.ops.aten.permute.default(view_294, [2, 0, 1, 4, 3]);  view_294 = None
        view_295 = torch.ops.aten.view.default(permute_331, [512, 8, 16, 64]);  permute_331 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(view_295, 4);  view_295 = None
        permute_332 = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 3, 2]);  unsqueeze_200 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(arg110_1, 3);  arg110_1 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
        permute_333 = torch.ops.aten.permute.default(unsqueeze_202, [3, 4, 0, 2, 1]);  unsqueeze_202 = None
        permute_334 = torch.ops.aten.permute.default(permute_332, [0, 1, 3, 4, 2]);  permute_332 = None
        clone_46 = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_296 = torch.ops.aten.view.default(clone_46, [1, 4096, 1024]);  clone_46 = None
        permute_335 = torch.ops.aten.permute.default(permute_333, [3, 4, 2, 0, 1]);  permute_333 = None
        clone_47 = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_297 = torch.ops.aten.view.default(clone_47, [1, 1024, 1024]);  clone_47 = None
        bmm_63 = torch.ops.aten.bmm.default(view_296, view_297);  view_296 = view_297 = None
        view_298 = torch.ops.aten.view.default(bmm_63, [512, 8, 1, 1, 1024]);  bmm_63 = None
        permute_336 = torch.ops.aten.permute.default(view_298, [0, 1, 4, 2, 3]);  view_298 = None
        view_299 = torch.ops.aten.view.default(permute_336, [512, 8, 1024]);  permute_336 = None
        add_81 = torch.ops.aten.add.Tensor(view_299, add_76);  view_299 = add_76 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_81, getitem_29);  add_81 = getitem_29 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg114_1);  mul_59 = arg114_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_60, arg115_1);  mul_60 = arg115_1 = None
        view_300 = torch.ops.aten.view.default(add_83, [4096, 1024])
        permute_337 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg119_1, view_300, permute_337);  arg119_1 = view_300 = permute_337 = None
        view_301 = torch.ops.aten.view.default(addmm_14, [512, 8, 4096]);  addmm_14 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_301, 0.5)
        mul_62 = torch.ops.aten.mul.Tensor(view_301, 0.7071067811865476);  view_301 = None
        erf_7 = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_84 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_61, add_84);  mul_61 = add_84 = None
        view_302 = torch.ops.aten.view.default(mul_63, [4096, 4096]);  mul_63 = None
        permute_338 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg121_1, view_302, permute_338);  arg121_1 = view_302 = permute_338 = None
        view_303 = torch.ops.aten.view.default(addmm_15, [512, 8, 1024]);  addmm_15 = None
        add_85 = torch.ops.aten.add.Tensor(view_303, add_83);  view_303 = add_83 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_85, getitem_31);  add_85 = getitem_31 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg116_1);  mul_64 = arg116_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_65, arg117_1);  mul_65 = arg117_1 = None
        slice_67 = torch.ops.aten.slice.Tensor(add_87, 0, -512, 9223372036854775807)
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(add_87, 3)
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
        permute_339 = torch.ops.aten.permute.default(unsqueeze_204, [0, 1, 3, 4, 2]);  unsqueeze_204 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(arg122_1, 3);  arg122_1 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
        permute_340 = torch.ops.aten.permute.default(unsqueeze_206, [3, 4, 1, 2, 0]);  unsqueeze_206 = None
        permute_341 = torch.ops.aten.permute.default(permute_339, [0, 1, 4, 2, 3]);  permute_339 = None
        view_304 = torch.ops.aten.view.default(permute_341, [1, 4096, 1024]);  permute_341 = None
        permute_342 = torch.ops.aten.permute.default(permute_340, [4, 2, 3, 0, 1]);  permute_340 = None
        view_305 = torch.ops.aten.view.default(permute_342, [1, 1024, 1024]);  permute_342 = None
        bmm_64 = torch.ops.aten.bmm.default(view_304, view_305);  view_304 = view_305 = None
        view_306 = torch.ops.aten.view.default(bmm_64, [512, 8, 1, 16, 64]);  bmm_64 = None
        permute_343 = torch.ops.aten.permute.default(view_306, [0, 1, 3, 4, 2]);  view_306 = None
        view_307 = torch.ops.aten.view.default(permute_343, [512, 8, 16, 64]);  permute_343 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(add_87, 3)
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(unsqueeze_207, 4);  unsqueeze_207 = None
        permute_344 = torch.ops.aten.permute.default(unsqueeze_208, [0, 1, 3, 4, 2]);  unsqueeze_208 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(arg123_1, 3);  arg123_1 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
        permute_345 = torch.ops.aten.permute.default(unsqueeze_210, [3, 4, 1, 2, 0]);  unsqueeze_210 = None
        permute_346 = torch.ops.aten.permute.default(permute_344, [0, 1, 4, 2, 3]);  permute_344 = None
        view_308 = torch.ops.aten.view.default(permute_346, [1, 4096, 1024]);  permute_346 = None
        permute_347 = torch.ops.aten.permute.default(permute_345, [4, 2, 3, 0, 1]);  permute_345 = None
        view_309 = torch.ops.aten.view.default(permute_347, [1, 1024, 1024]);  permute_347 = None
        bmm_65 = torch.ops.aten.bmm.default(view_308, view_309);  view_308 = view_309 = None
        view_310 = torch.ops.aten.view.default(bmm_65, [512, 8, 1, 16, 64]);  bmm_65 = None
        permute_348 = torch.ops.aten.permute.default(view_310, [0, 1, 3, 4, 2]);  view_310 = None
        view_311 = torch.ops.aten.view.default(permute_348, [512, 8, 16, 64]);  permute_348 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(add_87, 3)
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(unsqueeze_211, 4);  unsqueeze_211 = None
        permute_349 = torch.ops.aten.permute.default(unsqueeze_212, [0, 1, 3, 4, 2]);  unsqueeze_212 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(arg124_1, 3);  arg124_1 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(unsqueeze_213, 4);  unsqueeze_213 = None
        permute_350 = torch.ops.aten.permute.default(unsqueeze_214, [3, 4, 1, 2, 0]);  unsqueeze_214 = None
        permute_351 = torch.ops.aten.permute.default(permute_349, [0, 1, 4, 2, 3]);  permute_349 = None
        view_312 = torch.ops.aten.view.default(permute_351, [1, 4096, 1024]);  permute_351 = None
        permute_352 = torch.ops.aten.permute.default(permute_350, [4, 2, 3, 0, 1]);  permute_350 = None
        view_313 = torch.ops.aten.view.default(permute_352, [1, 1024, 1024]);  permute_352 = None
        bmm_66 = torch.ops.aten.bmm.default(view_312, view_313);  view_312 = view_313 = None
        view_314 = torch.ops.aten.view.default(bmm_66, [512, 8, 1, 16, 64]);  bmm_66 = None
        permute_353 = torch.ops.aten.permute.default(view_314, [0, 1, 3, 4, 2]);  view_314 = None
        view_315 = torch.ops.aten.view.default(permute_353, [512, 8, 16, 64]);  permute_353 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(unsqueeze_215, 4);  unsqueeze_215 = None
        permute_354 = torch.ops.aten.permute.default(unsqueeze_216, [0, 1, 3, 4, 2]);  unsqueeze_216 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(arg126_1, 3);  arg126_1 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(unsqueeze_217, 4);  unsqueeze_217 = None
        permute_355 = torch.ops.aten.permute.default(unsqueeze_218, [3, 4, 1, 2, 0]);  unsqueeze_218 = None
        permute_356 = torch.ops.aten.permute.default(permute_354, [0, 1, 4, 2, 3]);  permute_354 = None
        view_316 = torch.ops.aten.view.default(permute_356, [1, 8192, 1024]);  permute_356 = None
        permute_357 = torch.ops.aten.permute.default(permute_355, [4, 2, 3, 0, 1]);  permute_355 = None
        view_317 = torch.ops.aten.view.default(permute_357, [1, 1024, 1024]);  permute_357 = None
        bmm_67 = torch.ops.aten.bmm.default(view_316, view_317);  view_316 = view_317 = None
        view_318 = torch.ops.aten.view.default(bmm_67, [1024, 8, 1, 16, 64]);  bmm_67 = None
        permute_358 = torch.ops.aten.permute.default(view_318, [0, 1, 3, 4, 2]);  view_318 = None
        view_319 = torch.ops.aten.view.default(permute_358, [1024, 8, 16, 64]);  permute_358 = None
        add_88 = torch.ops.aten.add.Tensor(view_307, arg128_1);  arg128_1 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(add_88, 4);  add_88 = None
        permute_359 = torch.ops.aten.permute.default(unsqueeze_219, [1, 2, 0, 4, 3]);  unsqueeze_219 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(view_311, 4);  view_311 = None
        permute_360 = torch.ops.aten.permute.default(unsqueeze_220, [1, 2, 4, 0, 3]);  unsqueeze_220 = None
        permute_361 = torch.ops.aten.permute.default(permute_359, [0, 1, 2, 4, 3]);  permute_359 = None
        view_320 = torch.ops.aten.view.default(permute_361, [128, 512, 64]);  permute_361 = None
        permute_362 = torch.ops.aten.permute.default(permute_360, [0, 1, 4, 3, 2]);  permute_360 = None
        view_321 = torch.ops.aten.view.default(permute_362, [128, 64, 512]);  permute_362 = None
        bmm_68 = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
        view_322 = torch.ops.aten.view.default(bmm_68, [8, 16, 512, 1, 512]);  bmm_68 = None
        permute_363 = torch.ops.aten.permute.default(view_322, [0, 1, 2, 4, 3]);  view_322 = None
        view_323 = torch.ops.aten.view.default(permute_363, [8, 16, 512, 512]);  permute_363 = None
        add_89 = torch.ops.aten.add.Tensor(view_307, arg127_1);  view_307 = arg127_1 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(add_89, 4);  add_89 = None
        permute_364 = torch.ops.aten.permute.default(unsqueeze_221, [1, 2, 0, 4, 3]);  unsqueeze_221 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(view_319, 4);  view_319 = None
        permute_365 = torch.ops.aten.permute.default(unsqueeze_222, [1, 2, 4, 0, 3]);  unsqueeze_222 = None
        permute_366 = torch.ops.aten.permute.default(permute_364, [0, 1, 2, 4, 3]);  permute_364 = None
        view_324 = torch.ops.aten.view.default(permute_366, [128, 512, 64]);  permute_366 = None
        permute_367 = torch.ops.aten.permute.default(permute_365, [0, 1, 4, 3, 2]);  permute_365 = None
        view_325 = torch.ops.aten.view.default(permute_367, [128, 64, 1024]);  permute_367 = None
        bmm_69 = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
        view_326 = torch.ops.aten.view.default(bmm_69, [8, 16, 512, 1, 1024]);  bmm_69 = None
        permute_368 = torch.ops.aten.permute.default(view_326, [0, 1, 2, 4, 3]);  view_326 = None
        view_327 = torch.ops.aten.view.default(permute_368, [8, 16, 512, 1024]);  permute_368 = None
        view_328 = torch.ops.aten.view.default(view_327, [8, 16, 1024, 512]);  view_327 = None
        slice_70 = torch.ops.aten.slice.Tensor(view_328, 2, 1, 9223372036854775807);  view_328 = None
        view_329 = torch.ops.aten.view.default(slice_70, [8, 16, 512, 1023]);  slice_70 = None
        iota_10 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_8 = torch.ops.aten.index.Tensor(view_329, [None, None, None, iota_10]);  view_329 = iota_10 = None
        add_90 = torch.ops.aten.add.Tensor(view_323, index_8);  view_323 = index_8 = None
        add_91 = torch.ops.aten.add.Tensor(add_90, 0);  add_90 = None
        mul_tensor_30 = torch.ops.aten.mul.Tensor(add_91, 1);  add_91 = None
        amax_default_15 = torch.ops.aten.amax.default(mul_tensor_30, [3], True)
        sub_tensor_15 = torch.ops.aten.sub.Tensor(mul_tensor_30, amax_default_15);  mul_tensor_30 = amax_default_15 = None
        mul_tensor_31 = torch.ops.aten.mul.Tensor(sub_tensor_15, 0.125);  sub_tensor_15 = None
        exp_8 = torch.ops.aten.exp.default(mul_tensor_31);  mul_tensor_31 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
        div_9 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(div_9, 4);  div_9 = None
        permute_369 = torch.ops.aten.permute.default(unsqueeze_223, [2, 0, 1, 4, 3]);  unsqueeze_223 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(view_315, 4);  view_315 = None
        permute_370 = torch.ops.aten.permute.default(unsqueeze_224, [4, 1, 2, 3, 0]);  unsqueeze_224 = None
        permute_371 = torch.ops.aten.permute.default(permute_369, [1, 2, 0, 4, 3]);  permute_369 = None
        view_330 = torch.ops.aten.view.default(permute_371, [128, 512, 512]);  permute_371 = None
        permute_372 = torch.ops.aten.permute.default(permute_370, [1, 2, 4, 3, 0]);  permute_370 = None
        view_331 = torch.ops.aten.view.default(permute_372, [128, 512, 64]);  permute_372 = None
        bmm_70 = torch.ops.aten.bmm.default(view_330, view_331);  view_330 = view_331 = None
        view_332 = torch.ops.aten.view.default(bmm_70, [8, 16, 512, 1, 64]);  bmm_70 = None
        permute_373 = torch.ops.aten.permute.default(view_332, [2, 0, 1, 4, 3]);  view_332 = None
        view_333 = torch.ops.aten.view.default(permute_373, [512, 8, 16, 64]);  permute_373 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(view_333, 4);  view_333 = None
        permute_374 = torch.ops.aten.permute.default(unsqueeze_225, [0, 1, 4, 3, 2]);  unsqueeze_225 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(arg125_1, 3);  arg125_1 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
        permute_375 = torch.ops.aten.permute.default(unsqueeze_227, [3, 4, 0, 2, 1]);  unsqueeze_227 = None
        permute_376 = torch.ops.aten.permute.default(permute_374, [0, 1, 3, 4, 2]);  permute_374 = None
        clone_52 = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
        view_334 = torch.ops.aten.view.default(clone_52, [1, 4096, 1024]);  clone_52 = None
        permute_377 = torch.ops.aten.permute.default(permute_375, [3, 4, 2, 0, 1]);  permute_375 = None
        clone_53 = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
        view_335 = torch.ops.aten.view.default(clone_53, [1, 1024, 1024]);  clone_53 = None
        bmm_71 = torch.ops.aten.bmm.default(view_334, view_335);  view_334 = view_335 = None
        view_336 = torch.ops.aten.view.default(bmm_71, [512, 8, 1, 1, 1024]);  bmm_71 = None
        permute_378 = torch.ops.aten.permute.default(view_336, [0, 1, 4, 2, 3]);  view_336 = None
        view_337 = torch.ops.aten.view.default(permute_378, [512, 8, 1024]);  permute_378 = None
        add_92 = torch.ops.aten.add.Tensor(view_337, add_87);  view_337 = add_87 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_92, getitem_33);  add_92 = getitem_33 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_68 = torch.ops.aten.mul.Tensor(mul_67, arg129_1);  mul_67 = arg129_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_68, arg130_1);  mul_68 = arg130_1 = None
        view_338 = torch.ops.aten.view.default(add_94, [4096, 1024])
        permute_379 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg134_1, view_338, permute_379);  arg134_1 = view_338 = permute_379 = None
        view_339 = torch.ops.aten.view.default(addmm_16, [512, 8, 4096]);  addmm_16 = None
        mul_69 = torch.ops.aten.mul.Tensor(view_339, 0.5)
        mul_70 = torch.ops.aten.mul.Tensor(view_339, 0.7071067811865476);  view_339 = None
        erf_8 = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_95 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_69, add_95);  mul_69 = add_95 = None
        view_340 = torch.ops.aten.view.default(mul_71, [4096, 4096]);  mul_71 = None
        permute_380 = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg136_1, view_340, permute_380);  arg136_1 = view_340 = permute_380 = None
        view_341 = torch.ops.aten.view.default(addmm_17, [512, 8, 1024]);  addmm_17 = None
        add_96 = torch.ops.aten.add.Tensor(view_341, add_94);  view_341 = add_94 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_97 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_96, getitem_35);  add_96 = getitem_35 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg131_1);  mul_72 = arg131_1 = None
        add_98 = torch.ops.aten.add.Tensor(mul_73, arg132_1);  mul_73 = arg132_1 = None
        slice_75 = torch.ops.aten.slice.Tensor(add_98, 0, -512, 9223372036854775807)
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(add_98, 3)
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
        permute_381 = torch.ops.aten.permute.default(unsqueeze_229, [0, 1, 3, 4, 2]);  unsqueeze_229 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(arg137_1, 3);  arg137_1 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, 4);  unsqueeze_230 = None
        permute_382 = torch.ops.aten.permute.default(unsqueeze_231, [3, 4, 1, 2, 0]);  unsqueeze_231 = None
        permute_383 = torch.ops.aten.permute.default(permute_381, [0, 1, 4, 2, 3]);  permute_381 = None
        view_342 = torch.ops.aten.view.default(permute_383, [1, 4096, 1024]);  permute_383 = None
        permute_384 = torch.ops.aten.permute.default(permute_382, [4, 2, 3, 0, 1]);  permute_382 = None
        view_343 = torch.ops.aten.view.default(permute_384, [1, 1024, 1024]);  permute_384 = None
        bmm_72 = torch.ops.aten.bmm.default(view_342, view_343);  view_342 = view_343 = None
        view_344 = torch.ops.aten.view.default(bmm_72, [512, 8, 1, 16, 64]);  bmm_72 = None
        permute_385 = torch.ops.aten.permute.default(view_344, [0, 1, 3, 4, 2]);  view_344 = None
        view_345 = torch.ops.aten.view.default(permute_385, [512, 8, 16, 64]);  permute_385 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(add_98, 3)
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(unsqueeze_232, 4);  unsqueeze_232 = None
        permute_386 = torch.ops.aten.permute.default(unsqueeze_233, [0, 1, 3, 4, 2]);  unsqueeze_233 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(arg138_1, 3);  arg138_1 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
        permute_387 = torch.ops.aten.permute.default(unsqueeze_235, [3, 4, 1, 2, 0]);  unsqueeze_235 = None
        permute_388 = torch.ops.aten.permute.default(permute_386, [0, 1, 4, 2, 3]);  permute_386 = None
        view_346 = torch.ops.aten.view.default(permute_388, [1, 4096, 1024]);  permute_388 = None
        permute_389 = torch.ops.aten.permute.default(permute_387, [4, 2, 3, 0, 1]);  permute_387 = None
        view_347 = torch.ops.aten.view.default(permute_389, [1, 1024, 1024]);  permute_389 = None
        bmm_73 = torch.ops.aten.bmm.default(view_346, view_347);  view_346 = view_347 = None
        view_348 = torch.ops.aten.view.default(bmm_73, [512, 8, 1, 16, 64]);  bmm_73 = None
        permute_390 = torch.ops.aten.permute.default(view_348, [0, 1, 3, 4, 2]);  view_348 = None
        view_349 = torch.ops.aten.view.default(permute_390, [512, 8, 16, 64]);  permute_390 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(add_98, 3)
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, 4);  unsqueeze_236 = None
        permute_391 = torch.ops.aten.permute.default(unsqueeze_237, [0, 1, 3, 4, 2]);  unsqueeze_237 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(arg139_1, 3);  arg139_1 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
        permute_392 = torch.ops.aten.permute.default(unsqueeze_239, [3, 4, 1, 2, 0]);  unsqueeze_239 = None
        permute_393 = torch.ops.aten.permute.default(permute_391, [0, 1, 4, 2, 3]);  permute_391 = None
        view_350 = torch.ops.aten.view.default(permute_393, [1, 4096, 1024]);  permute_393 = None
        permute_394 = torch.ops.aten.permute.default(permute_392, [4, 2, 3, 0, 1]);  permute_392 = None
        view_351 = torch.ops.aten.view.default(permute_394, [1, 1024, 1024]);  permute_394 = None
        bmm_74 = torch.ops.aten.bmm.default(view_350, view_351);  view_350 = view_351 = None
        view_352 = torch.ops.aten.view.default(bmm_74, [512, 8, 1, 16, 64]);  bmm_74 = None
        permute_395 = torch.ops.aten.permute.default(view_352, [0, 1, 3, 4, 2]);  view_352 = None
        view_353 = torch.ops.aten.view.default(permute_395, [512, 8, 16, 64]);  permute_395 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, 4);  unsqueeze_240 = None
        permute_396 = torch.ops.aten.permute.default(unsqueeze_241, [0, 1, 3, 4, 2]);  unsqueeze_241 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(arg141_1, 3);  arg141_1 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
        permute_397 = torch.ops.aten.permute.default(unsqueeze_243, [3, 4, 1, 2, 0]);  unsqueeze_243 = None
        permute_398 = torch.ops.aten.permute.default(permute_396, [0, 1, 4, 2, 3]);  permute_396 = None
        view_354 = torch.ops.aten.view.default(permute_398, [1, 8192, 1024]);  permute_398 = None
        permute_399 = torch.ops.aten.permute.default(permute_397, [4, 2, 3, 0, 1]);  permute_397 = None
        view_355 = torch.ops.aten.view.default(permute_399, [1, 1024, 1024]);  permute_399 = None
        bmm_75 = torch.ops.aten.bmm.default(view_354, view_355);  view_354 = view_355 = None
        view_356 = torch.ops.aten.view.default(bmm_75, [1024, 8, 1, 16, 64]);  bmm_75 = None
        permute_400 = torch.ops.aten.permute.default(view_356, [0, 1, 3, 4, 2]);  view_356 = None
        view_357 = torch.ops.aten.view.default(permute_400, [1024, 8, 16, 64]);  permute_400 = None
        add_99 = torch.ops.aten.add.Tensor(view_345, arg143_1);  arg143_1 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(add_99, 4);  add_99 = None
        permute_401 = torch.ops.aten.permute.default(unsqueeze_244, [1, 2, 0, 4, 3]);  unsqueeze_244 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(view_349, 4);  view_349 = None
        permute_402 = torch.ops.aten.permute.default(unsqueeze_245, [1, 2, 4, 0, 3]);  unsqueeze_245 = None
        permute_403 = torch.ops.aten.permute.default(permute_401, [0, 1, 2, 4, 3]);  permute_401 = None
        view_358 = torch.ops.aten.view.default(permute_403, [128, 512, 64]);  permute_403 = None
        permute_404 = torch.ops.aten.permute.default(permute_402, [0, 1, 4, 3, 2]);  permute_402 = None
        view_359 = torch.ops.aten.view.default(permute_404, [128, 64, 512]);  permute_404 = None
        bmm_76 = torch.ops.aten.bmm.default(view_358, view_359);  view_358 = view_359 = None
        view_360 = torch.ops.aten.view.default(bmm_76, [8, 16, 512, 1, 512]);  bmm_76 = None
        permute_405 = torch.ops.aten.permute.default(view_360, [0, 1, 2, 4, 3]);  view_360 = None
        view_361 = torch.ops.aten.view.default(permute_405, [8, 16, 512, 512]);  permute_405 = None
        add_100 = torch.ops.aten.add.Tensor(view_345, arg142_1);  view_345 = arg142_1 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(add_100, 4);  add_100 = None
        permute_406 = torch.ops.aten.permute.default(unsqueeze_246, [1, 2, 0, 4, 3]);  unsqueeze_246 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(view_357, 4);  view_357 = None
        permute_407 = torch.ops.aten.permute.default(unsqueeze_247, [1, 2, 4, 0, 3]);  unsqueeze_247 = None
        permute_408 = torch.ops.aten.permute.default(permute_406, [0, 1, 2, 4, 3]);  permute_406 = None
        view_362 = torch.ops.aten.view.default(permute_408, [128, 512, 64]);  permute_408 = None
        permute_409 = torch.ops.aten.permute.default(permute_407, [0, 1, 4, 3, 2]);  permute_407 = None
        view_363 = torch.ops.aten.view.default(permute_409, [128, 64, 1024]);  permute_409 = None
        bmm_77 = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
        view_364 = torch.ops.aten.view.default(bmm_77, [8, 16, 512, 1, 1024]);  bmm_77 = None
        permute_410 = torch.ops.aten.permute.default(view_364, [0, 1, 2, 4, 3]);  view_364 = None
        view_365 = torch.ops.aten.view.default(permute_410, [8, 16, 512, 1024]);  permute_410 = None
        view_366 = torch.ops.aten.view.default(view_365, [8, 16, 1024, 512]);  view_365 = None
        slice_78 = torch.ops.aten.slice.Tensor(view_366, 2, 1, 9223372036854775807);  view_366 = None
        view_367 = torch.ops.aten.view.default(slice_78, [8, 16, 512, 1023]);  slice_78 = None
        iota_11 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_9 = torch.ops.aten.index.Tensor(view_367, [None, None, None, iota_11]);  view_367 = iota_11 = None
        add_101 = torch.ops.aten.add.Tensor(view_361, index_9);  view_361 = index_9 = None
        add_102 = torch.ops.aten.add.Tensor(add_101, 0);  add_101 = None
        mul_tensor_28 = torch.ops.aten.mul.Tensor(add_102, 1);  add_102 = None
        amax_default_14 = torch.ops.aten.amax.default(mul_tensor_28, [3], True)
        sub_tensor_14 = torch.ops.aten.sub.Tensor(mul_tensor_28, amax_default_14);  mul_tensor_28 = amax_default_14 = None
        mul_tensor_29 = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        exp_9 = torch.ops.aten.exp.default(mul_tensor_29);  mul_tensor_29 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
        div_10 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(div_10, 4);  div_10 = None
        permute_411 = torch.ops.aten.permute.default(unsqueeze_248, [2, 0, 1, 4, 3]);  unsqueeze_248 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(view_353, 4);  view_353 = None
        permute_412 = torch.ops.aten.permute.default(unsqueeze_249, [4, 1, 2, 3, 0]);  unsqueeze_249 = None
        permute_413 = torch.ops.aten.permute.default(permute_411, [1, 2, 0, 4, 3]);  permute_411 = None
        view_368 = torch.ops.aten.view.default(permute_413, [128, 512, 512]);  permute_413 = None
        permute_414 = torch.ops.aten.permute.default(permute_412, [1, 2, 4, 3, 0]);  permute_412 = None
        view_369 = torch.ops.aten.view.default(permute_414, [128, 512, 64]);  permute_414 = None
        bmm_78 = torch.ops.aten.bmm.default(view_368, view_369);  view_368 = view_369 = None
        view_370 = torch.ops.aten.view.default(bmm_78, [8, 16, 512, 1, 64]);  bmm_78 = None
        permute_415 = torch.ops.aten.permute.default(view_370, [2, 0, 1, 4, 3]);  view_370 = None
        view_371 = torch.ops.aten.view.default(permute_415, [512, 8, 16, 64]);  permute_415 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(view_371, 4);  view_371 = None
        permute_416 = torch.ops.aten.permute.default(unsqueeze_250, [0, 1, 4, 3, 2]);  unsqueeze_250 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(arg140_1, 3);  arg140_1 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
        permute_417 = torch.ops.aten.permute.default(unsqueeze_252, [3, 4, 0, 2, 1]);  unsqueeze_252 = None
        permute_418 = torch.ops.aten.permute.default(permute_416, [0, 1, 3, 4, 2]);  permute_416 = None
        clone_58 = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_372 = torch.ops.aten.view.default(clone_58, [1, 4096, 1024]);  clone_58 = None
        permute_419 = torch.ops.aten.permute.default(permute_417, [3, 4, 2, 0, 1]);  permute_417 = None
        clone_59 = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_373 = torch.ops.aten.view.default(clone_59, [1, 1024, 1024]);  clone_59 = None
        bmm_79 = torch.ops.aten.bmm.default(view_372, view_373);  view_372 = view_373 = None
        view_374 = torch.ops.aten.view.default(bmm_79, [512, 8, 1, 1, 1024]);  bmm_79 = None
        permute_420 = torch.ops.aten.permute.default(view_374, [0, 1, 4, 2, 3]);  view_374 = None
        view_375 = torch.ops.aten.view.default(permute_420, [512, 8, 1024]);  permute_420 = None
        add_103 = torch.ops.aten.add.Tensor(view_375, add_98);  view_375 = add_98 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_104 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_103, getitem_37);  add_103 = getitem_37 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, arg144_1);  mul_75 = arg144_1 = None
        add_105 = torch.ops.aten.add.Tensor(mul_76, arg145_1);  mul_76 = arg145_1 = None
        view_376 = torch.ops.aten.view.default(add_105, [4096, 1024])
        permute_421 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg149_1, view_376, permute_421);  arg149_1 = view_376 = permute_421 = None
        view_377 = torch.ops.aten.view.default(addmm_18, [512, 8, 4096]);  addmm_18 = None
        mul_77 = torch.ops.aten.mul.Tensor(view_377, 0.5)
        mul_78 = torch.ops.aten.mul.Tensor(view_377, 0.7071067811865476);  view_377 = None
        erf_9 = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_106 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_77, add_106);  mul_77 = add_106 = None
        view_378 = torch.ops.aten.view.default(mul_79, [4096, 4096]);  mul_79 = None
        permute_422 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg151_1, view_378, permute_422);  arg151_1 = view_378 = permute_422 = None
        view_379 = torch.ops.aten.view.default(addmm_19, [512, 8, 1024]);  addmm_19 = None
        add_107 = torch.ops.aten.add.Tensor(view_379, add_105);  view_379 = add_105 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_108 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_107, getitem_39);  add_107 = getitem_39 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg146_1);  mul_80 = arg146_1 = None
        add_109 = torch.ops.aten.add.Tensor(mul_81, arg147_1);  mul_81 = arg147_1 = None
        slice_83 = torch.ops.aten.slice.Tensor(add_109, 0, -512, 9223372036854775807)
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(add_109, 3)
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
        permute_423 = torch.ops.aten.permute.default(unsqueeze_254, [0, 1, 3, 4, 2]);  unsqueeze_254 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(arg152_1, 3);  arg152_1 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(unsqueeze_255, 4);  unsqueeze_255 = None
        permute_424 = torch.ops.aten.permute.default(unsqueeze_256, [3, 4, 1, 2, 0]);  unsqueeze_256 = None
        permute_425 = torch.ops.aten.permute.default(permute_423, [0, 1, 4, 2, 3]);  permute_423 = None
        view_380 = torch.ops.aten.view.default(permute_425, [1, 4096, 1024]);  permute_425 = None
        permute_426 = torch.ops.aten.permute.default(permute_424, [4, 2, 3, 0, 1]);  permute_424 = None
        view_381 = torch.ops.aten.view.default(permute_426, [1, 1024, 1024]);  permute_426 = None
        bmm_80 = torch.ops.aten.bmm.default(view_380, view_381);  view_380 = view_381 = None
        view_382 = torch.ops.aten.view.default(bmm_80, [512, 8, 1, 16, 64]);  bmm_80 = None
        permute_427 = torch.ops.aten.permute.default(view_382, [0, 1, 3, 4, 2]);  view_382 = None
        view_383 = torch.ops.aten.view.default(permute_427, [512, 8, 16, 64]);  permute_427 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(add_109, 3)
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(unsqueeze_257, 4);  unsqueeze_257 = None
        permute_428 = torch.ops.aten.permute.default(unsqueeze_258, [0, 1, 3, 4, 2]);  unsqueeze_258 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(arg153_1, 3);  arg153_1 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(unsqueeze_259, 4);  unsqueeze_259 = None
        permute_429 = torch.ops.aten.permute.default(unsqueeze_260, [3, 4, 1, 2, 0]);  unsqueeze_260 = None
        permute_430 = torch.ops.aten.permute.default(permute_428, [0, 1, 4, 2, 3]);  permute_428 = None
        view_384 = torch.ops.aten.view.default(permute_430, [1, 4096, 1024]);  permute_430 = None
        permute_431 = torch.ops.aten.permute.default(permute_429, [4, 2, 3, 0, 1]);  permute_429 = None
        view_385 = torch.ops.aten.view.default(permute_431, [1, 1024, 1024]);  permute_431 = None
        bmm_81 = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
        view_386 = torch.ops.aten.view.default(bmm_81, [512, 8, 1, 16, 64]);  bmm_81 = None
        permute_432 = torch.ops.aten.permute.default(view_386, [0, 1, 3, 4, 2]);  view_386 = None
        view_387 = torch.ops.aten.view.default(permute_432, [512, 8, 16, 64]);  permute_432 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(add_109, 3)
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(unsqueeze_261, 4);  unsqueeze_261 = None
        permute_433 = torch.ops.aten.permute.default(unsqueeze_262, [0, 1, 3, 4, 2]);  unsqueeze_262 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(arg154_1, 3);  arg154_1 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
        permute_434 = torch.ops.aten.permute.default(unsqueeze_264, [3, 4, 1, 2, 0]);  unsqueeze_264 = None
        permute_435 = torch.ops.aten.permute.default(permute_433, [0, 1, 4, 2, 3]);  permute_433 = None
        view_388 = torch.ops.aten.view.default(permute_435, [1, 4096, 1024]);  permute_435 = None
        permute_436 = torch.ops.aten.permute.default(permute_434, [4, 2, 3, 0, 1]);  permute_434 = None
        view_389 = torch.ops.aten.view.default(permute_436, [1, 1024, 1024]);  permute_436 = None
        bmm_82 = torch.ops.aten.bmm.default(view_388, view_389);  view_388 = view_389 = None
        view_390 = torch.ops.aten.view.default(bmm_82, [512, 8, 1, 16, 64]);  bmm_82 = None
        permute_437 = torch.ops.aten.permute.default(view_390, [0, 1, 3, 4, 2]);  view_390 = None
        view_391 = torch.ops.aten.view.default(permute_437, [512, 8, 16, 64]);  permute_437 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(unsqueeze_265, 4);  unsqueeze_265 = None
        permute_438 = torch.ops.aten.permute.default(unsqueeze_266, [0, 1, 3, 4, 2]);  unsqueeze_266 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(arg156_1, 3);  arg156_1 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(unsqueeze_267, 4);  unsqueeze_267 = None
        permute_439 = torch.ops.aten.permute.default(unsqueeze_268, [3, 4, 1, 2, 0]);  unsqueeze_268 = None
        permute_440 = torch.ops.aten.permute.default(permute_438, [0, 1, 4, 2, 3]);  permute_438 = None
        view_392 = torch.ops.aten.view.default(permute_440, [1, 8192, 1024]);  permute_440 = None
        permute_441 = torch.ops.aten.permute.default(permute_439, [4, 2, 3, 0, 1]);  permute_439 = None
        view_393 = torch.ops.aten.view.default(permute_441, [1, 1024, 1024]);  permute_441 = None
        bmm_83 = torch.ops.aten.bmm.default(view_392, view_393);  view_392 = view_393 = None
        view_394 = torch.ops.aten.view.default(bmm_83, [1024, 8, 1, 16, 64]);  bmm_83 = None
        permute_442 = torch.ops.aten.permute.default(view_394, [0, 1, 3, 4, 2]);  view_394 = None
        view_395 = torch.ops.aten.view.default(permute_442, [1024, 8, 16, 64]);  permute_442 = None
        add_110 = torch.ops.aten.add.Tensor(view_383, arg158_1);  arg158_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(add_110, 4);  add_110 = None
        permute_443 = torch.ops.aten.permute.default(unsqueeze_269, [1, 2, 0, 4, 3]);  unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(view_387, 4);  view_387 = None
        permute_444 = torch.ops.aten.permute.default(unsqueeze_270, [1, 2, 4, 0, 3]);  unsqueeze_270 = None
        permute_445 = torch.ops.aten.permute.default(permute_443, [0, 1, 2, 4, 3]);  permute_443 = None
        view_396 = torch.ops.aten.view.default(permute_445, [128, 512, 64]);  permute_445 = None
        permute_446 = torch.ops.aten.permute.default(permute_444, [0, 1, 4, 3, 2]);  permute_444 = None
        view_397 = torch.ops.aten.view.default(permute_446, [128, 64, 512]);  permute_446 = None
        bmm_84 = torch.ops.aten.bmm.default(view_396, view_397);  view_396 = view_397 = None
        view_398 = torch.ops.aten.view.default(bmm_84, [8, 16, 512, 1, 512]);  bmm_84 = None
        permute_447 = torch.ops.aten.permute.default(view_398, [0, 1, 2, 4, 3]);  view_398 = None
        view_399 = torch.ops.aten.view.default(permute_447, [8, 16, 512, 512]);  permute_447 = None
        add_111 = torch.ops.aten.add.Tensor(view_383, arg157_1);  view_383 = arg157_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(add_111, 4);  add_111 = None
        permute_448 = torch.ops.aten.permute.default(unsqueeze_271, [1, 2, 0, 4, 3]);  unsqueeze_271 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(view_395, 4);  view_395 = None
        permute_449 = torch.ops.aten.permute.default(unsqueeze_272, [1, 2, 4, 0, 3]);  unsqueeze_272 = None
        permute_450 = torch.ops.aten.permute.default(permute_448, [0, 1, 2, 4, 3]);  permute_448 = None
        view_400 = torch.ops.aten.view.default(permute_450, [128, 512, 64]);  permute_450 = None
        permute_451 = torch.ops.aten.permute.default(permute_449, [0, 1, 4, 3, 2]);  permute_449 = None
        view_401 = torch.ops.aten.view.default(permute_451, [128, 64, 1024]);  permute_451 = None
        bmm_85 = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
        view_402 = torch.ops.aten.view.default(bmm_85, [8, 16, 512, 1, 1024]);  bmm_85 = None
        permute_452 = torch.ops.aten.permute.default(view_402, [0, 1, 2, 4, 3]);  view_402 = None
        view_403 = torch.ops.aten.view.default(permute_452, [8, 16, 512, 1024]);  permute_452 = None
        view_404 = torch.ops.aten.view.default(view_403, [8, 16, 1024, 512]);  view_403 = None
        slice_86 = torch.ops.aten.slice.Tensor(view_404, 2, 1, 9223372036854775807);  view_404 = None
        view_405 = torch.ops.aten.view.default(slice_86, [8, 16, 512, 1023]);  slice_86 = None
        iota_12 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_10 = torch.ops.aten.index.Tensor(view_405, [None, None, None, iota_12]);  view_405 = iota_12 = None
        add_112 = torch.ops.aten.add.Tensor(view_399, index_10);  view_399 = index_10 = None
        add_113 = torch.ops.aten.add.Tensor(add_112, 0);  add_112 = None
        mul_tensor_26 = torch.ops.aten.mul.Tensor(add_113, 1);  add_113 = None
        amax_default_13 = torch.ops.aten.amax.default(mul_tensor_26, [3], True)
        sub_tensor_13 = torch.ops.aten.sub.Tensor(mul_tensor_26, amax_default_13);  mul_tensor_26 = amax_default_13 = None
        mul_tensor_27 = torch.ops.aten.mul.Tensor(sub_tensor_13, 0.125);  sub_tensor_13 = None
        exp_10 = torch.ops.aten.exp.default(mul_tensor_27);  mul_tensor_27 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
        div_11 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(div_11, 4);  div_11 = None
        permute_453 = torch.ops.aten.permute.default(unsqueeze_273, [2, 0, 1, 4, 3]);  unsqueeze_273 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(view_391, 4);  view_391 = None
        permute_454 = torch.ops.aten.permute.default(unsqueeze_274, [4, 1, 2, 3, 0]);  unsqueeze_274 = None
        permute_455 = torch.ops.aten.permute.default(permute_453, [1, 2, 0, 4, 3]);  permute_453 = None
        view_406 = torch.ops.aten.view.default(permute_455, [128, 512, 512]);  permute_455 = None
        permute_456 = torch.ops.aten.permute.default(permute_454, [1, 2, 4, 3, 0]);  permute_454 = None
        view_407 = torch.ops.aten.view.default(permute_456, [128, 512, 64]);  permute_456 = None
        bmm_86 = torch.ops.aten.bmm.default(view_406, view_407);  view_406 = view_407 = None
        view_408 = torch.ops.aten.view.default(bmm_86, [8, 16, 512, 1, 64]);  bmm_86 = None
        permute_457 = torch.ops.aten.permute.default(view_408, [2, 0, 1, 4, 3]);  view_408 = None
        view_409 = torch.ops.aten.view.default(permute_457, [512, 8, 16, 64]);  permute_457 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(view_409, 4);  view_409 = None
        permute_458 = torch.ops.aten.permute.default(unsqueeze_275, [0, 1, 4, 3, 2]);  unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg155_1, 3);  arg155_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, 4);  unsqueeze_276 = None
        permute_459 = torch.ops.aten.permute.default(unsqueeze_277, [3, 4, 0, 2, 1]);  unsqueeze_277 = None
        permute_460 = torch.ops.aten.permute.default(permute_458, [0, 1, 3, 4, 2]);  permute_458 = None
        clone_64 = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
        view_410 = torch.ops.aten.view.default(clone_64, [1, 4096, 1024]);  clone_64 = None
        permute_461 = torch.ops.aten.permute.default(permute_459, [3, 4, 2, 0, 1]);  permute_459 = None
        clone_65 = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
        view_411 = torch.ops.aten.view.default(clone_65, [1, 1024, 1024]);  clone_65 = None
        bmm_87 = torch.ops.aten.bmm.default(view_410, view_411);  view_410 = view_411 = None
        view_412 = torch.ops.aten.view.default(bmm_87, [512, 8, 1, 1, 1024]);  bmm_87 = None
        permute_462 = torch.ops.aten.permute.default(view_412, [0, 1, 4, 2, 3]);  view_412 = None
        view_413 = torch.ops.aten.view.default(permute_462, [512, 8, 1024]);  permute_462 = None
        add_114 = torch.ops.aten.add.Tensor(view_413, add_109);  view_413 = add_109 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_115 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_114, getitem_41);  add_114 = getitem_41 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, arg159_1);  mul_83 = arg159_1 = None
        add_116 = torch.ops.aten.add.Tensor(mul_84, arg160_1);  mul_84 = arg160_1 = None
        view_414 = torch.ops.aten.view.default(add_116, [4096, 1024])
        permute_463 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg164_1, view_414, permute_463);  arg164_1 = view_414 = permute_463 = None
        view_415 = torch.ops.aten.view.default(addmm_20, [512, 8, 4096]);  addmm_20 = None
        mul_85 = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_86 = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
        erf_10 = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_117 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_85, add_117);  mul_85 = add_117 = None
        view_416 = torch.ops.aten.view.default(mul_87, [4096, 4096]);  mul_87 = None
        permute_464 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg166_1, view_416, permute_464);  arg166_1 = view_416 = permute_464 = None
        view_417 = torch.ops.aten.view.default(addmm_21, [512, 8, 1024]);  addmm_21 = None
        add_118 = torch.ops.aten.add.Tensor(view_417, add_116);  view_417 = add_116 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_119 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_118, getitem_43);  add_118 = getitem_43 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg161_1);  mul_88 = arg161_1 = None
        add_120 = torch.ops.aten.add.Tensor(mul_89, arg162_1);  mul_89 = arg162_1 = None
        slice_91 = torch.ops.aten.slice.Tensor(add_120, 0, -512, 9223372036854775807)
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(add_120, 3)
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, 4);  unsqueeze_278 = None
        permute_465 = torch.ops.aten.permute.default(unsqueeze_279, [0, 1, 3, 4, 2]);  unsqueeze_279 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg167_1, 3);  arg167_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
        permute_466 = torch.ops.aten.permute.default(unsqueeze_281, [3, 4, 1, 2, 0]);  unsqueeze_281 = None
        permute_467 = torch.ops.aten.permute.default(permute_465, [0, 1, 4, 2, 3]);  permute_465 = None
        view_418 = torch.ops.aten.view.default(permute_467, [1, 4096, 1024]);  permute_467 = None
        permute_468 = torch.ops.aten.permute.default(permute_466, [4, 2, 3, 0, 1]);  permute_466 = None
        view_419 = torch.ops.aten.view.default(permute_468, [1, 1024, 1024]);  permute_468 = None
        bmm_88 = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
        view_420 = torch.ops.aten.view.default(bmm_88, [512, 8, 1, 16, 64]);  bmm_88 = None
        permute_469 = torch.ops.aten.permute.default(view_420, [0, 1, 3, 4, 2]);  view_420 = None
        view_421 = torch.ops.aten.view.default(permute_469, [512, 8, 16, 64]);  permute_469 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(add_120, 3)
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, 4);  unsqueeze_282 = None
        permute_470 = torch.ops.aten.permute.default(unsqueeze_283, [0, 1, 3, 4, 2]);  unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg168_1, 3);  arg168_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
        permute_471 = torch.ops.aten.permute.default(unsqueeze_285, [3, 4, 1, 2, 0]);  unsqueeze_285 = None
        permute_472 = torch.ops.aten.permute.default(permute_470, [0, 1, 4, 2, 3]);  permute_470 = None
        view_422 = torch.ops.aten.view.default(permute_472, [1, 4096, 1024]);  permute_472 = None
        permute_473 = torch.ops.aten.permute.default(permute_471, [4, 2, 3, 0, 1]);  permute_471 = None
        view_423 = torch.ops.aten.view.default(permute_473, [1, 1024, 1024]);  permute_473 = None
        bmm_89 = torch.ops.aten.bmm.default(view_422, view_423);  view_422 = view_423 = None
        view_424 = torch.ops.aten.view.default(bmm_89, [512, 8, 1, 16, 64]);  bmm_89 = None
        permute_474 = torch.ops.aten.permute.default(view_424, [0, 1, 3, 4, 2]);  view_424 = None
        view_425 = torch.ops.aten.view.default(permute_474, [512, 8, 16, 64]);  permute_474 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(add_120, 3)
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, 4);  unsqueeze_286 = None
        permute_475 = torch.ops.aten.permute.default(unsqueeze_287, [0, 1, 3, 4, 2]);  unsqueeze_287 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg169_1, 3);  arg169_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
        permute_476 = torch.ops.aten.permute.default(unsqueeze_289, [3, 4, 1, 2, 0]);  unsqueeze_289 = None
        permute_477 = torch.ops.aten.permute.default(permute_475, [0, 1, 4, 2, 3]);  permute_475 = None
        view_426 = torch.ops.aten.view.default(permute_477, [1, 4096, 1024]);  permute_477 = None
        permute_478 = torch.ops.aten.permute.default(permute_476, [4, 2, 3, 0, 1]);  permute_476 = None
        view_427 = torch.ops.aten.view.default(permute_478, [1, 1024, 1024]);  permute_478 = None
        bmm_90 = torch.ops.aten.bmm.default(view_426, view_427);  view_426 = view_427 = None
        view_428 = torch.ops.aten.view.default(bmm_90, [512, 8, 1, 16, 64]);  bmm_90 = None
        permute_479 = torch.ops.aten.permute.default(view_428, [0, 1, 3, 4, 2]);  view_428 = None
        view_429 = torch.ops.aten.view.default(permute_479, [512, 8, 16, 64]);  permute_479 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, 4);  unsqueeze_290 = None
        permute_480 = torch.ops.aten.permute.default(unsqueeze_291, [0, 1, 3, 4, 2]);  unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg171_1, 3);  arg171_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, 4);  unsqueeze_292 = None
        permute_481 = torch.ops.aten.permute.default(unsqueeze_293, [3, 4, 1, 2, 0]);  unsqueeze_293 = None
        permute_482 = torch.ops.aten.permute.default(permute_480, [0, 1, 4, 2, 3]);  permute_480 = None
        view_430 = torch.ops.aten.view.default(permute_482, [1, 8192, 1024]);  permute_482 = None
        permute_483 = torch.ops.aten.permute.default(permute_481, [4, 2, 3, 0, 1]);  permute_481 = None
        view_431 = torch.ops.aten.view.default(permute_483, [1, 1024, 1024]);  permute_483 = None
        bmm_91 = torch.ops.aten.bmm.default(view_430, view_431);  view_430 = view_431 = None
        view_432 = torch.ops.aten.view.default(bmm_91, [1024, 8, 1, 16, 64]);  bmm_91 = None
        permute_484 = torch.ops.aten.permute.default(view_432, [0, 1, 3, 4, 2]);  view_432 = None
        view_433 = torch.ops.aten.view.default(permute_484, [1024, 8, 16, 64]);  permute_484 = None
        add_121 = torch.ops.aten.add.Tensor(view_421, arg173_1);  arg173_1 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(add_121, 4);  add_121 = None
        permute_485 = torch.ops.aten.permute.default(unsqueeze_294, [1, 2, 0, 4, 3]);  unsqueeze_294 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(view_425, 4);  view_425 = None
        permute_486 = torch.ops.aten.permute.default(unsqueeze_295, [1, 2, 4, 0, 3]);  unsqueeze_295 = None
        permute_487 = torch.ops.aten.permute.default(permute_485, [0, 1, 2, 4, 3]);  permute_485 = None
        view_434 = torch.ops.aten.view.default(permute_487, [128, 512, 64]);  permute_487 = None
        permute_488 = torch.ops.aten.permute.default(permute_486, [0, 1, 4, 3, 2]);  permute_486 = None
        view_435 = torch.ops.aten.view.default(permute_488, [128, 64, 512]);  permute_488 = None
        bmm_92 = torch.ops.aten.bmm.default(view_434, view_435);  view_434 = view_435 = None
        view_436 = torch.ops.aten.view.default(bmm_92, [8, 16, 512, 1, 512]);  bmm_92 = None
        permute_489 = torch.ops.aten.permute.default(view_436, [0, 1, 2, 4, 3]);  view_436 = None
        view_437 = torch.ops.aten.view.default(permute_489, [8, 16, 512, 512]);  permute_489 = None
        add_122 = torch.ops.aten.add.Tensor(view_421, arg172_1);  view_421 = arg172_1 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(add_122, 4);  add_122 = None
        permute_490 = torch.ops.aten.permute.default(unsqueeze_296, [1, 2, 0, 4, 3]);  unsqueeze_296 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(view_433, 4);  view_433 = None
        permute_491 = torch.ops.aten.permute.default(unsqueeze_297, [1, 2, 4, 0, 3]);  unsqueeze_297 = None
        permute_492 = torch.ops.aten.permute.default(permute_490, [0, 1, 2, 4, 3]);  permute_490 = None
        view_438 = torch.ops.aten.view.default(permute_492, [128, 512, 64]);  permute_492 = None
        permute_493 = torch.ops.aten.permute.default(permute_491, [0, 1, 4, 3, 2]);  permute_491 = None
        view_439 = torch.ops.aten.view.default(permute_493, [128, 64, 1024]);  permute_493 = None
        bmm_93 = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
        view_440 = torch.ops.aten.view.default(bmm_93, [8, 16, 512, 1, 1024]);  bmm_93 = None
        permute_494 = torch.ops.aten.permute.default(view_440, [0, 1, 2, 4, 3]);  view_440 = None
        view_441 = torch.ops.aten.view.default(permute_494, [8, 16, 512, 1024]);  permute_494 = None
        view_442 = torch.ops.aten.view.default(view_441, [8, 16, 1024, 512]);  view_441 = None
        slice_94 = torch.ops.aten.slice.Tensor(view_442, 2, 1, 9223372036854775807);  view_442 = None
        view_443 = torch.ops.aten.view.default(slice_94, [8, 16, 512, 1023]);  slice_94 = None
        iota_13 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_11 = torch.ops.aten.index.Tensor(view_443, [None, None, None, iota_13]);  view_443 = iota_13 = None
        add_123 = torch.ops.aten.add.Tensor(view_437, index_11);  view_437 = index_11 = None
        add_124 = torch.ops.aten.add.Tensor(add_123, 0);  add_123 = None
        mul_tensor_24 = torch.ops.aten.mul.Tensor(add_124, 1);  add_124 = None
        amax_default_12 = torch.ops.aten.amax.default(mul_tensor_24, [3], True)
        sub_tensor_12 = torch.ops.aten.sub.Tensor(mul_tensor_24, amax_default_12);  mul_tensor_24 = amax_default_12 = None
        mul_tensor_25 = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        exp_11 = torch.ops.aten.exp.default(mul_tensor_25);  mul_tensor_25 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
        div_12 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(div_12, 4);  div_12 = None
        permute_495 = torch.ops.aten.permute.default(unsqueeze_298, [2, 0, 1, 4, 3]);  unsqueeze_298 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(view_429, 4);  view_429 = None
        permute_496 = torch.ops.aten.permute.default(unsqueeze_299, [4, 1, 2, 3, 0]);  unsqueeze_299 = None
        permute_497 = torch.ops.aten.permute.default(permute_495, [1, 2, 0, 4, 3]);  permute_495 = None
        view_444 = torch.ops.aten.view.default(permute_497, [128, 512, 512]);  permute_497 = None
        permute_498 = torch.ops.aten.permute.default(permute_496, [1, 2, 4, 3, 0]);  permute_496 = None
        view_445 = torch.ops.aten.view.default(permute_498, [128, 512, 64]);  permute_498 = None
        bmm_94 = torch.ops.aten.bmm.default(view_444, view_445);  view_444 = view_445 = None
        view_446 = torch.ops.aten.view.default(bmm_94, [8, 16, 512, 1, 64]);  bmm_94 = None
        permute_499 = torch.ops.aten.permute.default(view_446, [2, 0, 1, 4, 3]);  view_446 = None
        view_447 = torch.ops.aten.view.default(permute_499, [512, 8, 16, 64]);  permute_499 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
        permute_500 = torch.ops.aten.permute.default(unsqueeze_300, [0, 1, 4, 3, 2]);  unsqueeze_300 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(arg170_1, 3);  arg170_1 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
        permute_501 = torch.ops.aten.permute.default(unsqueeze_302, [3, 4, 0, 2, 1]);  unsqueeze_302 = None
        permute_502 = torch.ops.aten.permute.default(permute_500, [0, 1, 3, 4, 2]);  permute_500 = None
        clone_70 = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
        view_448 = torch.ops.aten.view.default(clone_70, [1, 4096, 1024]);  clone_70 = None
        permute_503 = torch.ops.aten.permute.default(permute_501, [3, 4, 2, 0, 1]);  permute_501 = None
        clone_71 = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
        view_449 = torch.ops.aten.view.default(clone_71, [1, 1024, 1024]);  clone_71 = None
        bmm_95 = torch.ops.aten.bmm.default(view_448, view_449);  view_448 = view_449 = None
        view_450 = torch.ops.aten.view.default(bmm_95, [512, 8, 1, 1, 1024]);  bmm_95 = None
        permute_504 = torch.ops.aten.permute.default(view_450, [0, 1, 4, 2, 3]);  view_450 = None
        view_451 = torch.ops.aten.view.default(permute_504, [512, 8, 1024]);  permute_504 = None
        add_125 = torch.ops.aten.add.Tensor(view_451, add_120);  view_451 = add_120 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_125, getitem_45);  add_125 = getitem_45 = None
        mul_91 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_91, arg174_1);  mul_91 = arg174_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_92, arg175_1);  mul_92 = arg175_1 = None
        view_452 = torch.ops.aten.view.default(add_127, [4096, 1024])
        permute_505 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg179_1, view_452, permute_505);  arg179_1 = view_452 = permute_505 = None
        view_453 = torch.ops.aten.view.default(addmm_22, [512, 8, 4096]);  addmm_22 = None
        mul_93 = torch.ops.aten.mul.Tensor(view_453, 0.5)
        mul_94 = torch.ops.aten.mul.Tensor(view_453, 0.7071067811865476);  view_453 = None
        erf_11 = torch.ops.aten.erf.default(mul_94);  mul_94 = None
        add_128 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_93, add_128);  mul_93 = add_128 = None
        view_454 = torch.ops.aten.view.default(mul_95, [4096, 4096]);  mul_95 = None
        permute_506 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg181_1, view_454, permute_506);  arg181_1 = view_454 = permute_506 = None
        view_455 = torch.ops.aten.view.default(addmm_23, [512, 8, 1024]);  addmm_23 = None
        add_129 = torch.ops.aten.add.Tensor(view_455, add_127);  view_455 = add_127 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_129, getitem_47);  add_129 = getitem_47 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg176_1);  mul_96 = arg176_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_97, arg177_1);  mul_97 = arg177_1 = None
        slice_99 = torch.ops.aten.slice.Tensor(add_131, 0, -512, 9223372036854775807)
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(add_131, 3)
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
        permute_507 = torch.ops.aten.permute.default(unsqueeze_304, [0, 1, 3, 4, 2]);  unsqueeze_304 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(arg182_1, 3);  arg182_1 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
        permute_508 = torch.ops.aten.permute.default(unsqueeze_306, [3, 4, 1, 2, 0]);  unsqueeze_306 = None
        permute_509 = torch.ops.aten.permute.default(permute_507, [0, 1, 4, 2, 3]);  permute_507 = None
        view_456 = torch.ops.aten.view.default(permute_509, [1, 4096, 1024]);  permute_509 = None
        permute_510 = torch.ops.aten.permute.default(permute_508, [4, 2, 3, 0, 1]);  permute_508 = None
        view_457 = torch.ops.aten.view.default(permute_510, [1, 1024, 1024]);  permute_510 = None
        bmm_96 = torch.ops.aten.bmm.default(view_456, view_457);  view_456 = view_457 = None
        view_458 = torch.ops.aten.view.default(bmm_96, [512, 8, 1, 16, 64]);  bmm_96 = None
        permute_511 = torch.ops.aten.permute.default(view_458, [0, 1, 3, 4, 2]);  view_458 = None
        view_459 = torch.ops.aten.view.default(permute_511, [512, 8, 16, 64]);  permute_511 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(add_131, 3)
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(unsqueeze_307, 4);  unsqueeze_307 = None
        permute_512 = torch.ops.aten.permute.default(unsqueeze_308, [0, 1, 3, 4, 2]);  unsqueeze_308 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(arg183_1, 3);  arg183_1 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
        permute_513 = torch.ops.aten.permute.default(unsqueeze_310, [3, 4, 1, 2, 0]);  unsqueeze_310 = None
        permute_514 = torch.ops.aten.permute.default(permute_512, [0, 1, 4, 2, 3]);  permute_512 = None
        view_460 = torch.ops.aten.view.default(permute_514, [1, 4096, 1024]);  permute_514 = None
        permute_515 = torch.ops.aten.permute.default(permute_513, [4, 2, 3, 0, 1]);  permute_513 = None
        view_461 = torch.ops.aten.view.default(permute_515, [1, 1024, 1024]);  permute_515 = None
        bmm_97 = torch.ops.aten.bmm.default(view_460, view_461);  view_460 = view_461 = None
        view_462 = torch.ops.aten.view.default(bmm_97, [512, 8, 1, 16, 64]);  bmm_97 = None
        permute_516 = torch.ops.aten.permute.default(view_462, [0, 1, 3, 4, 2]);  view_462 = None
        view_463 = torch.ops.aten.view.default(permute_516, [512, 8, 16, 64]);  permute_516 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(add_131, 3)
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(unsqueeze_311, 4);  unsqueeze_311 = None
        permute_517 = torch.ops.aten.permute.default(unsqueeze_312, [0, 1, 3, 4, 2]);  unsqueeze_312 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(arg184_1, 3);  arg184_1 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(unsqueeze_313, 4);  unsqueeze_313 = None
        permute_518 = torch.ops.aten.permute.default(unsqueeze_314, [3, 4, 1, 2, 0]);  unsqueeze_314 = None
        permute_519 = torch.ops.aten.permute.default(permute_517, [0, 1, 4, 2, 3]);  permute_517 = None
        view_464 = torch.ops.aten.view.default(permute_519, [1, 4096, 1024]);  permute_519 = None
        permute_520 = torch.ops.aten.permute.default(permute_518, [4, 2, 3, 0, 1]);  permute_518 = None
        view_465 = torch.ops.aten.view.default(permute_520, [1, 1024, 1024]);  permute_520 = None
        bmm_98 = torch.ops.aten.bmm.default(view_464, view_465);  view_464 = view_465 = None
        view_466 = torch.ops.aten.view.default(bmm_98, [512, 8, 1, 16, 64]);  bmm_98 = None
        permute_521 = torch.ops.aten.permute.default(view_466, [0, 1, 3, 4, 2]);  view_466 = None
        view_467 = torch.ops.aten.view.default(permute_521, [512, 8, 16, 64]);  permute_521 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(unsqueeze_315, 4);  unsqueeze_315 = None
        permute_522 = torch.ops.aten.permute.default(unsqueeze_316, [0, 1, 3, 4, 2]);  unsqueeze_316 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(arg186_1, 3);  arg186_1 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(unsqueeze_317, 4);  unsqueeze_317 = None
        permute_523 = torch.ops.aten.permute.default(unsqueeze_318, [3, 4, 1, 2, 0]);  unsqueeze_318 = None
        permute_524 = torch.ops.aten.permute.default(permute_522, [0, 1, 4, 2, 3]);  permute_522 = None
        view_468 = torch.ops.aten.view.default(permute_524, [1, 8192, 1024]);  permute_524 = None
        permute_525 = torch.ops.aten.permute.default(permute_523, [4, 2, 3, 0, 1]);  permute_523 = None
        view_469 = torch.ops.aten.view.default(permute_525, [1, 1024, 1024]);  permute_525 = None
        bmm_99 = torch.ops.aten.bmm.default(view_468, view_469);  view_468 = view_469 = None
        view_470 = torch.ops.aten.view.default(bmm_99, [1024, 8, 1, 16, 64]);  bmm_99 = None
        permute_526 = torch.ops.aten.permute.default(view_470, [0, 1, 3, 4, 2]);  view_470 = None
        view_471 = torch.ops.aten.view.default(permute_526, [1024, 8, 16, 64]);  permute_526 = None
        add_132 = torch.ops.aten.add.Tensor(view_459, arg188_1);  arg188_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(add_132, 4);  add_132 = None
        permute_527 = torch.ops.aten.permute.default(unsqueeze_319, [1, 2, 0, 4, 3]);  unsqueeze_319 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(view_463, 4);  view_463 = None
        permute_528 = torch.ops.aten.permute.default(unsqueeze_320, [1, 2, 4, 0, 3]);  unsqueeze_320 = None
        permute_529 = torch.ops.aten.permute.default(permute_527, [0, 1, 2, 4, 3]);  permute_527 = None
        view_472 = torch.ops.aten.view.default(permute_529, [128, 512, 64]);  permute_529 = None
        permute_530 = torch.ops.aten.permute.default(permute_528, [0, 1, 4, 3, 2]);  permute_528 = None
        view_473 = torch.ops.aten.view.default(permute_530, [128, 64, 512]);  permute_530 = None
        bmm_100 = torch.ops.aten.bmm.default(view_472, view_473);  view_472 = view_473 = None
        view_474 = torch.ops.aten.view.default(bmm_100, [8, 16, 512, 1, 512]);  bmm_100 = None
        permute_531 = torch.ops.aten.permute.default(view_474, [0, 1, 2, 4, 3]);  view_474 = None
        view_475 = torch.ops.aten.view.default(permute_531, [8, 16, 512, 512]);  permute_531 = None
        add_133 = torch.ops.aten.add.Tensor(view_459, arg187_1);  view_459 = arg187_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(add_133, 4);  add_133 = None
        permute_532 = torch.ops.aten.permute.default(unsqueeze_321, [1, 2, 0, 4, 3]);  unsqueeze_321 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(view_471, 4);  view_471 = None
        permute_533 = torch.ops.aten.permute.default(unsqueeze_322, [1, 2, 4, 0, 3]);  unsqueeze_322 = None
        permute_534 = torch.ops.aten.permute.default(permute_532, [0, 1, 2, 4, 3]);  permute_532 = None
        view_476 = torch.ops.aten.view.default(permute_534, [128, 512, 64]);  permute_534 = None
        permute_535 = torch.ops.aten.permute.default(permute_533, [0, 1, 4, 3, 2]);  permute_533 = None
        view_477 = torch.ops.aten.view.default(permute_535, [128, 64, 1024]);  permute_535 = None
        bmm_101 = torch.ops.aten.bmm.default(view_476, view_477);  view_476 = view_477 = None
        view_478 = torch.ops.aten.view.default(bmm_101, [8, 16, 512, 1, 1024]);  bmm_101 = None
        permute_536 = torch.ops.aten.permute.default(view_478, [0, 1, 2, 4, 3]);  view_478 = None
        view_479 = torch.ops.aten.view.default(permute_536, [8, 16, 512, 1024]);  permute_536 = None
        view_480 = torch.ops.aten.view.default(view_479, [8, 16, 1024, 512]);  view_479 = None
        slice_102 = torch.ops.aten.slice.Tensor(view_480, 2, 1, 9223372036854775807);  view_480 = None
        view_481 = torch.ops.aten.view.default(slice_102, [8, 16, 512, 1023]);  slice_102 = None
        iota_14 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_12 = torch.ops.aten.index.Tensor(view_481, [None, None, None, iota_14]);  view_481 = iota_14 = None
        add_134 = torch.ops.aten.add.Tensor(view_475, index_12);  view_475 = index_12 = None
        add_135 = torch.ops.aten.add.Tensor(add_134, 0);  add_134 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(add_135, 1);  add_135 = None
        amax_default_11 = torch.ops.aten.amax.default(mul_tensor_22, [3], True)
        sub_tensor_11 = torch.ops.aten.sub.Tensor(mul_tensor_22, amax_default_11);  mul_tensor_22 = amax_default_11 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(sub_tensor_11, 0.125);  sub_tensor_11 = None
        exp_12 = torch.ops.aten.exp.default(mul_tensor_23);  mul_tensor_23 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
        div_13 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(div_13, 4);  div_13 = None
        permute_537 = torch.ops.aten.permute.default(unsqueeze_323, [2, 0, 1, 4, 3]);  unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(view_467, 4);  view_467 = None
        permute_538 = torch.ops.aten.permute.default(unsqueeze_324, [4, 1, 2, 3, 0]);  unsqueeze_324 = None
        permute_539 = torch.ops.aten.permute.default(permute_537, [1, 2, 0, 4, 3]);  permute_537 = None
        view_482 = torch.ops.aten.view.default(permute_539, [128, 512, 512]);  permute_539 = None
        permute_540 = torch.ops.aten.permute.default(permute_538, [1, 2, 4, 3, 0]);  permute_538 = None
        view_483 = torch.ops.aten.view.default(permute_540, [128, 512, 64]);  permute_540 = None
        bmm_102 = torch.ops.aten.bmm.default(view_482, view_483);  view_482 = view_483 = None
        view_484 = torch.ops.aten.view.default(bmm_102, [8, 16, 512, 1, 64]);  bmm_102 = None
        permute_541 = torch.ops.aten.permute.default(view_484, [2, 0, 1, 4, 3]);  view_484 = None
        view_485 = torch.ops.aten.view.default(permute_541, [512, 8, 16, 64]);  permute_541 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(view_485, 4);  view_485 = None
        permute_542 = torch.ops.aten.permute.default(unsqueeze_325, [0, 1, 4, 3, 2]);  unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg185_1, 3);  arg185_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, 4);  unsqueeze_326 = None
        permute_543 = torch.ops.aten.permute.default(unsqueeze_327, [3, 4, 0, 2, 1]);  unsqueeze_327 = None
        permute_544 = torch.ops.aten.permute.default(permute_542, [0, 1, 3, 4, 2]);  permute_542 = None
        clone_76 = torch.ops.aten.clone.default(permute_544, memory_format = torch.contiguous_format);  permute_544 = None
        view_486 = torch.ops.aten.view.default(clone_76, [1, 4096, 1024]);  clone_76 = None
        permute_545 = torch.ops.aten.permute.default(permute_543, [3, 4, 2, 0, 1]);  permute_543 = None
        clone_77 = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
        view_487 = torch.ops.aten.view.default(clone_77, [1, 1024, 1024]);  clone_77 = None
        bmm_103 = torch.ops.aten.bmm.default(view_486, view_487);  view_486 = view_487 = None
        view_488 = torch.ops.aten.view.default(bmm_103, [512, 8, 1, 1, 1024]);  bmm_103 = None
        permute_546 = torch.ops.aten.permute.default(view_488, [0, 1, 4, 2, 3]);  view_488 = None
        view_489 = torch.ops.aten.view.default(permute_546, [512, 8, 1024]);  permute_546 = None
        add_136 = torch.ops.aten.add.Tensor(view_489, add_131);  view_489 = add_131 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_137 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_136, getitem_49);  add_136 = getitem_49 = None
        mul_99 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, arg189_1);  mul_99 = arg189_1 = None
        add_138 = torch.ops.aten.add.Tensor(mul_100, arg190_1);  mul_100 = arg190_1 = None
        view_490 = torch.ops.aten.view.default(add_138, [4096, 1024])
        permute_547 = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg194_1, view_490, permute_547);  arg194_1 = view_490 = permute_547 = None
        view_491 = torch.ops.aten.view.default(addmm_24, [512, 8, 4096]);  addmm_24 = None
        mul_101 = torch.ops.aten.mul.Tensor(view_491, 0.5)
        mul_102 = torch.ops.aten.mul.Tensor(view_491, 0.7071067811865476);  view_491 = None
        erf_12 = torch.ops.aten.erf.default(mul_102);  mul_102 = None
        add_139 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_101, add_139);  mul_101 = add_139 = None
        view_492 = torch.ops.aten.view.default(mul_103, [4096, 4096]);  mul_103 = None
        permute_548 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg196_1, view_492, permute_548);  arg196_1 = view_492 = permute_548 = None
        view_493 = torch.ops.aten.view.default(addmm_25, [512, 8, 1024]);  addmm_25 = None
        add_140 = torch.ops.aten.add.Tensor(view_493, add_138);  view_493 = add_138 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_141 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_140, getitem_51);  add_140 = getitem_51 = None
        mul_104 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, arg191_1);  mul_104 = arg191_1 = None
        add_142 = torch.ops.aten.add.Tensor(mul_105, arg192_1);  mul_105 = arg192_1 = None
        slice_107 = torch.ops.aten.slice.Tensor(add_142, 0, -512, 9223372036854775807)
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(add_142, 3)
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
        permute_549 = torch.ops.aten.permute.default(unsqueeze_329, [0, 1, 3, 4, 2]);  unsqueeze_329 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(arg197_1, 3);  arg197_1 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, 4);  unsqueeze_330 = None
        permute_550 = torch.ops.aten.permute.default(unsqueeze_331, [3, 4, 1, 2, 0]);  unsqueeze_331 = None
        permute_551 = torch.ops.aten.permute.default(permute_549, [0, 1, 4, 2, 3]);  permute_549 = None
        view_494 = torch.ops.aten.view.default(permute_551, [1, 4096, 1024]);  permute_551 = None
        permute_552 = torch.ops.aten.permute.default(permute_550, [4, 2, 3, 0, 1]);  permute_550 = None
        view_495 = torch.ops.aten.view.default(permute_552, [1, 1024, 1024]);  permute_552 = None
        bmm_104 = torch.ops.aten.bmm.default(view_494, view_495);  view_494 = view_495 = None
        view_496 = torch.ops.aten.view.default(bmm_104, [512, 8, 1, 16, 64]);  bmm_104 = None
        permute_553 = torch.ops.aten.permute.default(view_496, [0, 1, 3, 4, 2]);  view_496 = None
        view_497 = torch.ops.aten.view.default(permute_553, [512, 8, 16, 64]);  permute_553 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(add_142, 3)
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, 4);  unsqueeze_332 = None
        permute_554 = torch.ops.aten.permute.default(unsqueeze_333, [0, 1, 3, 4, 2]);  unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg198_1, 3);  arg198_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, 4);  unsqueeze_334 = None
        permute_555 = torch.ops.aten.permute.default(unsqueeze_335, [3, 4, 1, 2, 0]);  unsqueeze_335 = None
        permute_556 = torch.ops.aten.permute.default(permute_554, [0, 1, 4, 2, 3]);  permute_554 = None
        view_498 = torch.ops.aten.view.default(permute_556, [1, 4096, 1024]);  permute_556 = None
        permute_557 = torch.ops.aten.permute.default(permute_555, [4, 2, 3, 0, 1]);  permute_555 = None
        view_499 = torch.ops.aten.view.default(permute_557, [1, 1024, 1024]);  permute_557 = None
        bmm_105 = torch.ops.aten.bmm.default(view_498, view_499);  view_498 = view_499 = None
        view_500 = torch.ops.aten.view.default(bmm_105, [512, 8, 1, 16, 64]);  bmm_105 = None
        permute_558 = torch.ops.aten.permute.default(view_500, [0, 1, 3, 4, 2]);  view_500 = None
        view_501 = torch.ops.aten.view.default(permute_558, [512, 8, 16, 64]);  permute_558 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(add_142, 3)
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, 4);  unsqueeze_336 = None
        permute_559 = torch.ops.aten.permute.default(unsqueeze_337, [0, 1, 3, 4, 2]);  unsqueeze_337 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(arg199_1, 3);  arg199_1 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
        permute_560 = torch.ops.aten.permute.default(unsqueeze_339, [3, 4, 1, 2, 0]);  unsqueeze_339 = None
        permute_561 = torch.ops.aten.permute.default(permute_559, [0, 1, 4, 2, 3]);  permute_559 = None
        view_502 = torch.ops.aten.view.default(permute_561, [1, 4096, 1024]);  permute_561 = None
        permute_562 = torch.ops.aten.permute.default(permute_560, [4, 2, 3, 0, 1]);  permute_560 = None
        view_503 = torch.ops.aten.view.default(permute_562, [1, 1024, 1024]);  permute_562 = None
        bmm_106 = torch.ops.aten.bmm.default(view_502, view_503);  view_502 = view_503 = None
        view_504 = torch.ops.aten.view.default(bmm_106, [512, 8, 1, 16, 64]);  bmm_106 = None
        permute_563 = torch.ops.aten.permute.default(view_504, [0, 1, 3, 4, 2]);  view_504 = None
        view_505 = torch.ops.aten.view.default(permute_563, [512, 8, 16, 64]);  permute_563 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, 4);  unsqueeze_340 = None
        permute_564 = torch.ops.aten.permute.default(unsqueeze_341, [0, 1, 3, 4, 2]);  unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg201_1, 3);  arg201_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
        permute_565 = torch.ops.aten.permute.default(unsqueeze_343, [3, 4, 1, 2, 0]);  unsqueeze_343 = None
        permute_566 = torch.ops.aten.permute.default(permute_564, [0, 1, 4, 2, 3]);  permute_564 = None
        view_506 = torch.ops.aten.view.default(permute_566, [1, 8192, 1024]);  permute_566 = None
        permute_567 = torch.ops.aten.permute.default(permute_565, [4, 2, 3, 0, 1]);  permute_565 = None
        view_507 = torch.ops.aten.view.default(permute_567, [1, 1024, 1024]);  permute_567 = None
        bmm_107 = torch.ops.aten.bmm.default(view_506, view_507);  view_506 = view_507 = None
        view_508 = torch.ops.aten.view.default(bmm_107, [1024, 8, 1, 16, 64]);  bmm_107 = None
        permute_568 = torch.ops.aten.permute.default(view_508, [0, 1, 3, 4, 2]);  view_508 = None
        view_509 = torch.ops.aten.view.default(permute_568, [1024, 8, 16, 64]);  permute_568 = None
        add_143 = torch.ops.aten.add.Tensor(view_497, arg203_1);  arg203_1 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(add_143, 4);  add_143 = None
        permute_569 = torch.ops.aten.permute.default(unsqueeze_344, [1, 2, 0, 4, 3]);  unsqueeze_344 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(view_501, 4);  view_501 = None
        permute_570 = torch.ops.aten.permute.default(unsqueeze_345, [1, 2, 4, 0, 3]);  unsqueeze_345 = None
        permute_571 = torch.ops.aten.permute.default(permute_569, [0, 1, 2, 4, 3]);  permute_569 = None
        view_510 = torch.ops.aten.view.default(permute_571, [128, 512, 64]);  permute_571 = None
        permute_572 = torch.ops.aten.permute.default(permute_570, [0, 1, 4, 3, 2]);  permute_570 = None
        view_511 = torch.ops.aten.view.default(permute_572, [128, 64, 512]);  permute_572 = None
        bmm_108 = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512 = torch.ops.aten.view.default(bmm_108, [8, 16, 512, 1, 512]);  bmm_108 = None
        permute_573 = torch.ops.aten.permute.default(view_512, [0, 1, 2, 4, 3]);  view_512 = None
        view_513 = torch.ops.aten.view.default(permute_573, [8, 16, 512, 512]);  permute_573 = None
        add_144 = torch.ops.aten.add.Tensor(view_497, arg202_1);  view_497 = arg202_1 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(add_144, 4);  add_144 = None
        permute_574 = torch.ops.aten.permute.default(unsqueeze_346, [1, 2, 0, 4, 3]);  unsqueeze_346 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(view_509, 4);  view_509 = None
        permute_575 = torch.ops.aten.permute.default(unsqueeze_347, [1, 2, 4, 0, 3]);  unsqueeze_347 = None
        permute_576 = torch.ops.aten.permute.default(permute_574, [0, 1, 2, 4, 3]);  permute_574 = None
        view_514 = torch.ops.aten.view.default(permute_576, [128, 512, 64]);  permute_576 = None
        permute_577 = torch.ops.aten.permute.default(permute_575, [0, 1, 4, 3, 2]);  permute_575 = None
        view_515 = torch.ops.aten.view.default(permute_577, [128, 64, 1024]);  permute_577 = None
        bmm_109 = torch.ops.aten.bmm.default(view_514, view_515);  view_514 = view_515 = None
        view_516 = torch.ops.aten.view.default(bmm_109, [8, 16, 512, 1, 1024]);  bmm_109 = None
        permute_578 = torch.ops.aten.permute.default(view_516, [0, 1, 2, 4, 3]);  view_516 = None
        view_517 = torch.ops.aten.view.default(permute_578, [8, 16, 512, 1024]);  permute_578 = None
        view_518 = torch.ops.aten.view.default(view_517, [8, 16, 1024, 512]);  view_517 = None
        slice_110 = torch.ops.aten.slice.Tensor(view_518, 2, 1, 9223372036854775807);  view_518 = None
        view_519 = torch.ops.aten.view.default(slice_110, [8, 16, 512, 1023]);  slice_110 = None
        iota_15 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_13 = torch.ops.aten.index.Tensor(view_519, [None, None, None, iota_15]);  view_519 = iota_15 = None
        add_145 = torch.ops.aten.add.Tensor(view_513, index_13);  view_513 = index_13 = None
        add_146 = torch.ops.aten.add.Tensor(add_145, 0);  add_145 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(add_146, 1);  add_146 = None
        amax_default_10 = torch.ops.aten.amax.default(mul_tensor_20, [3], True)
        sub_tensor_10 = torch.ops.aten.sub.Tensor(mul_tensor_20, amax_default_10);  mul_tensor_20 = amax_default_10 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        exp_13 = torch.ops.aten.exp.default(mul_tensor_21);  mul_tensor_21 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
        div_14 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(div_14, 4);  div_14 = None
        permute_579 = torch.ops.aten.permute.default(unsqueeze_348, [2, 0, 1, 4, 3]);  unsqueeze_348 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(view_505, 4);  view_505 = None
        permute_580 = torch.ops.aten.permute.default(unsqueeze_349, [4, 1, 2, 3, 0]);  unsqueeze_349 = None
        permute_581 = torch.ops.aten.permute.default(permute_579, [1, 2, 0, 4, 3]);  permute_579 = None
        view_520 = torch.ops.aten.view.default(permute_581, [128, 512, 512]);  permute_581 = None
        permute_582 = torch.ops.aten.permute.default(permute_580, [1, 2, 4, 3, 0]);  permute_580 = None
        view_521 = torch.ops.aten.view.default(permute_582, [128, 512, 64]);  permute_582 = None
        bmm_110 = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
        view_522 = torch.ops.aten.view.default(bmm_110, [8, 16, 512, 1, 64]);  bmm_110 = None
        permute_583 = torch.ops.aten.permute.default(view_522, [2, 0, 1, 4, 3]);  view_522 = None
        view_523 = torch.ops.aten.view.default(permute_583, [512, 8, 16, 64]);  permute_583 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(view_523, 4);  view_523 = None
        permute_584 = torch.ops.aten.permute.default(unsqueeze_350, [0, 1, 4, 3, 2]);  unsqueeze_350 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(arg200_1, 3);  arg200_1 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
        permute_585 = torch.ops.aten.permute.default(unsqueeze_352, [3, 4, 0, 2, 1]);  unsqueeze_352 = None
        permute_586 = torch.ops.aten.permute.default(permute_584, [0, 1, 3, 4, 2]);  permute_584 = None
        clone_82 = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
        view_524 = torch.ops.aten.view.default(clone_82, [1, 4096, 1024]);  clone_82 = None
        permute_587 = torch.ops.aten.permute.default(permute_585, [3, 4, 2, 0, 1]);  permute_585 = None
        clone_83 = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        view_525 = torch.ops.aten.view.default(clone_83, [1, 1024, 1024]);  clone_83 = None
        bmm_111 = torch.ops.aten.bmm.default(view_524, view_525);  view_524 = view_525 = None
        view_526 = torch.ops.aten.view.default(bmm_111, [512, 8, 1, 1, 1024]);  bmm_111 = None
        permute_588 = torch.ops.aten.permute.default(view_526, [0, 1, 4, 2, 3]);  view_526 = None
        view_527 = torch.ops.aten.view.default(permute_588, [512, 8, 1024]);  permute_588 = None
        add_147 = torch.ops.aten.add.Tensor(view_527, add_142);  view_527 = add_142 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_148 = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_147, getitem_53);  add_147 = getitem_53 = None
        mul_107 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, arg204_1);  mul_107 = arg204_1 = None
        add_149 = torch.ops.aten.add.Tensor(mul_108, arg205_1);  mul_108 = arg205_1 = None
        view_528 = torch.ops.aten.view.default(add_149, [4096, 1024])
        permute_589 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg209_1, view_528, permute_589);  arg209_1 = view_528 = permute_589 = None
        view_529 = torch.ops.aten.view.default(addmm_26, [512, 8, 4096]);  addmm_26 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_110 = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
        erf_13 = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_150 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_109, add_150);  mul_109 = add_150 = None
        view_530 = torch.ops.aten.view.default(mul_111, [4096, 4096]);  mul_111 = None
        permute_590 = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg211_1, view_530, permute_590);  arg211_1 = view_530 = permute_590 = None
        view_531 = torch.ops.aten.view.default(addmm_27, [512, 8, 1024]);  addmm_27 = None
        add_151 = torch.ops.aten.add.Tensor(view_531, add_149);  view_531 = add_149 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_152 = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_151, getitem_55);  add_151 = getitem_55 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, arg206_1);  mul_112 = arg206_1 = None
        add_153 = torch.ops.aten.add.Tensor(mul_113, arg207_1);  mul_113 = arg207_1 = None
        slice_115 = torch.ops.aten.slice.Tensor(add_153, 0, -512, 9223372036854775807)
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(add_153, 3)
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
        permute_591 = torch.ops.aten.permute.default(unsqueeze_354, [0, 1, 3, 4, 2]);  unsqueeze_354 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(arg212_1, 3);  arg212_1 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
        permute_592 = torch.ops.aten.permute.default(unsqueeze_356, [3, 4, 1, 2, 0]);  unsqueeze_356 = None
        permute_593 = torch.ops.aten.permute.default(permute_591, [0, 1, 4, 2, 3]);  permute_591 = None
        view_532 = torch.ops.aten.view.default(permute_593, [1, 4096, 1024]);  permute_593 = None
        permute_594 = torch.ops.aten.permute.default(permute_592, [4, 2, 3, 0, 1]);  permute_592 = None
        view_533 = torch.ops.aten.view.default(permute_594, [1, 1024, 1024]);  permute_594 = None
        bmm_112 = torch.ops.aten.bmm.default(view_532, view_533);  view_532 = view_533 = None
        view_534 = torch.ops.aten.view.default(bmm_112, [512, 8, 1, 16, 64]);  bmm_112 = None
        permute_595 = torch.ops.aten.permute.default(view_534, [0, 1, 3, 4, 2]);  view_534 = None
        view_535 = torch.ops.aten.view.default(permute_595, [512, 8, 16, 64]);  permute_595 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(add_153, 3)
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(unsqueeze_357, 4);  unsqueeze_357 = None
        permute_596 = torch.ops.aten.permute.default(unsqueeze_358, [0, 1, 3, 4, 2]);  unsqueeze_358 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(arg213_1, 3);  arg213_1 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(unsqueeze_359, 4);  unsqueeze_359 = None
        permute_597 = torch.ops.aten.permute.default(unsqueeze_360, [3, 4, 1, 2, 0]);  unsqueeze_360 = None
        permute_598 = torch.ops.aten.permute.default(permute_596, [0, 1, 4, 2, 3]);  permute_596 = None
        view_536 = torch.ops.aten.view.default(permute_598, [1, 4096, 1024]);  permute_598 = None
        permute_599 = torch.ops.aten.permute.default(permute_597, [4, 2, 3, 0, 1]);  permute_597 = None
        view_537 = torch.ops.aten.view.default(permute_599, [1, 1024, 1024]);  permute_599 = None
        bmm_113 = torch.ops.aten.bmm.default(view_536, view_537);  view_536 = view_537 = None
        view_538 = torch.ops.aten.view.default(bmm_113, [512, 8, 1, 16, 64]);  bmm_113 = None
        permute_600 = torch.ops.aten.permute.default(view_538, [0, 1, 3, 4, 2]);  view_538 = None
        view_539 = torch.ops.aten.view.default(permute_600, [512, 8, 16, 64]);  permute_600 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(add_153, 3)
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(unsqueeze_361, 4);  unsqueeze_361 = None
        permute_601 = torch.ops.aten.permute.default(unsqueeze_362, [0, 1, 3, 4, 2]);  unsqueeze_362 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(arg214_1, 3);  arg214_1 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(unsqueeze_363, 4);  unsqueeze_363 = None
        permute_602 = torch.ops.aten.permute.default(unsqueeze_364, [3, 4, 1, 2, 0]);  unsqueeze_364 = None
        permute_603 = torch.ops.aten.permute.default(permute_601, [0, 1, 4, 2, 3]);  permute_601 = None
        view_540 = torch.ops.aten.view.default(permute_603, [1, 4096, 1024]);  permute_603 = None
        permute_604 = torch.ops.aten.permute.default(permute_602, [4, 2, 3, 0, 1]);  permute_602 = None
        view_541 = torch.ops.aten.view.default(permute_604, [1, 1024, 1024]);  permute_604 = None
        bmm_114 = torch.ops.aten.bmm.default(view_540, view_541);  view_540 = view_541 = None
        view_542 = torch.ops.aten.view.default(bmm_114, [512, 8, 1, 16, 64]);  bmm_114 = None
        permute_605 = torch.ops.aten.permute.default(view_542, [0, 1, 3, 4, 2]);  view_542 = None
        view_543 = torch.ops.aten.view.default(permute_605, [512, 8, 16, 64]);  permute_605 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(unsqueeze_365, 4);  unsqueeze_365 = None
        permute_606 = torch.ops.aten.permute.default(unsqueeze_366, [0, 1, 3, 4, 2]);  unsqueeze_366 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(arg216_1, 3);  arg216_1 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
        permute_607 = torch.ops.aten.permute.default(unsqueeze_368, [3, 4, 1, 2, 0]);  unsqueeze_368 = None
        permute_608 = torch.ops.aten.permute.default(permute_606, [0, 1, 4, 2, 3]);  permute_606 = None
        view_544 = torch.ops.aten.view.default(permute_608, [1, 8192, 1024]);  permute_608 = None
        permute_609 = torch.ops.aten.permute.default(permute_607, [4, 2, 3, 0, 1]);  permute_607 = None
        view_545 = torch.ops.aten.view.default(permute_609, [1, 1024, 1024]);  permute_609 = None
        bmm_115 = torch.ops.aten.bmm.default(view_544, view_545);  view_544 = view_545 = None
        view_546 = torch.ops.aten.view.default(bmm_115, [1024, 8, 1, 16, 64]);  bmm_115 = None
        permute_610 = torch.ops.aten.permute.default(view_546, [0, 1, 3, 4, 2]);  view_546 = None
        view_547 = torch.ops.aten.view.default(permute_610, [1024, 8, 16, 64]);  permute_610 = None
        add_154 = torch.ops.aten.add.Tensor(view_535, arg218_1);  arg218_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(add_154, 4);  add_154 = None
        permute_611 = torch.ops.aten.permute.default(unsqueeze_369, [1, 2, 0, 4, 3]);  unsqueeze_369 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(view_539, 4);  view_539 = None
        permute_612 = torch.ops.aten.permute.default(unsqueeze_370, [1, 2, 4, 0, 3]);  unsqueeze_370 = None
        permute_613 = torch.ops.aten.permute.default(permute_611, [0, 1, 2, 4, 3]);  permute_611 = None
        view_548 = torch.ops.aten.view.default(permute_613, [128, 512, 64]);  permute_613 = None
        permute_614 = torch.ops.aten.permute.default(permute_612, [0, 1, 4, 3, 2]);  permute_612 = None
        view_549 = torch.ops.aten.view.default(permute_614, [128, 64, 512]);  permute_614 = None
        bmm_116 = torch.ops.aten.bmm.default(view_548, view_549);  view_548 = view_549 = None
        view_550 = torch.ops.aten.view.default(bmm_116, [8, 16, 512, 1, 512]);  bmm_116 = None
        permute_615 = torch.ops.aten.permute.default(view_550, [0, 1, 2, 4, 3]);  view_550 = None
        view_551 = torch.ops.aten.view.default(permute_615, [8, 16, 512, 512]);  permute_615 = None
        add_155 = torch.ops.aten.add.Tensor(view_535, arg217_1);  view_535 = arg217_1 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(add_155, 4);  add_155 = None
        permute_616 = torch.ops.aten.permute.default(unsqueeze_371, [1, 2, 0, 4, 3]);  unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(view_547, 4);  view_547 = None
        permute_617 = torch.ops.aten.permute.default(unsqueeze_372, [1, 2, 4, 0, 3]);  unsqueeze_372 = None
        permute_618 = torch.ops.aten.permute.default(permute_616, [0, 1, 2, 4, 3]);  permute_616 = None
        view_552 = torch.ops.aten.view.default(permute_618, [128, 512, 64]);  permute_618 = None
        permute_619 = torch.ops.aten.permute.default(permute_617, [0, 1, 4, 3, 2]);  permute_617 = None
        view_553 = torch.ops.aten.view.default(permute_619, [128, 64, 1024]);  permute_619 = None
        bmm_117 = torch.ops.aten.bmm.default(view_552, view_553);  view_552 = view_553 = None
        view_554 = torch.ops.aten.view.default(bmm_117, [8, 16, 512, 1, 1024]);  bmm_117 = None
        permute_620 = torch.ops.aten.permute.default(view_554, [0, 1, 2, 4, 3]);  view_554 = None
        view_555 = torch.ops.aten.view.default(permute_620, [8, 16, 512, 1024]);  permute_620 = None
        view_556 = torch.ops.aten.view.default(view_555, [8, 16, 1024, 512]);  view_555 = None
        slice_118 = torch.ops.aten.slice.Tensor(view_556, 2, 1, 9223372036854775807);  view_556 = None
        view_557 = torch.ops.aten.view.default(slice_118, [8, 16, 512, 1023]);  slice_118 = None
        iota_16 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_14 = torch.ops.aten.index.Tensor(view_557, [None, None, None, iota_16]);  view_557 = iota_16 = None
        add_156 = torch.ops.aten.add.Tensor(view_551, index_14);  view_551 = index_14 = None
        add_157 = torch.ops.aten.add.Tensor(add_156, 0);  add_156 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(add_157, 1);  add_157 = None
        amax_default_9 = torch.ops.aten.amax.default(mul_tensor_18, [3], True)
        sub_tensor_9 = torch.ops.aten.sub.Tensor(mul_tensor_18, amax_default_9);  mul_tensor_18 = amax_default_9 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(sub_tensor_9, 0.125);  sub_tensor_9 = None
        exp_14 = torch.ops.aten.exp.default(mul_tensor_19);  mul_tensor_19 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
        div_15 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(div_15, 4);  div_15 = None
        permute_621 = torch.ops.aten.permute.default(unsqueeze_373, [2, 0, 1, 4, 3]);  unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(view_543, 4);  view_543 = None
        permute_622 = torch.ops.aten.permute.default(unsqueeze_374, [4, 1, 2, 3, 0]);  unsqueeze_374 = None
        permute_623 = torch.ops.aten.permute.default(permute_621, [1, 2, 0, 4, 3]);  permute_621 = None
        view_558 = torch.ops.aten.view.default(permute_623, [128, 512, 512]);  permute_623 = None
        permute_624 = torch.ops.aten.permute.default(permute_622, [1, 2, 4, 3, 0]);  permute_622 = None
        view_559 = torch.ops.aten.view.default(permute_624, [128, 512, 64]);  permute_624 = None
        bmm_118 = torch.ops.aten.bmm.default(view_558, view_559);  view_558 = view_559 = None
        view_560 = torch.ops.aten.view.default(bmm_118, [8, 16, 512, 1, 64]);  bmm_118 = None
        permute_625 = torch.ops.aten.permute.default(view_560, [2, 0, 1, 4, 3]);  view_560 = None
        view_561 = torch.ops.aten.view.default(permute_625, [512, 8, 16, 64]);  permute_625 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(view_561, 4);  view_561 = None
        permute_626 = torch.ops.aten.permute.default(unsqueeze_375, [0, 1, 4, 3, 2]);  unsqueeze_375 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg215_1, 3);  arg215_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
        permute_627 = torch.ops.aten.permute.default(unsqueeze_377, [3, 4, 0, 2, 1]);  unsqueeze_377 = None
        permute_628 = torch.ops.aten.permute.default(permute_626, [0, 1, 3, 4, 2]);  permute_626 = None
        clone_88 = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
        view_562 = torch.ops.aten.view.default(clone_88, [1, 4096, 1024]);  clone_88 = None
        permute_629 = torch.ops.aten.permute.default(permute_627, [3, 4, 2, 0, 1]);  permute_627 = None
        clone_89 = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_563 = torch.ops.aten.view.default(clone_89, [1, 1024, 1024]);  clone_89 = None
        bmm_119 = torch.ops.aten.bmm.default(view_562, view_563);  view_562 = view_563 = None
        view_564 = torch.ops.aten.view.default(bmm_119, [512, 8, 1, 1, 1024]);  bmm_119 = None
        permute_630 = torch.ops.aten.permute.default(view_564, [0, 1, 4, 2, 3]);  view_564 = None
        view_565 = torch.ops.aten.view.default(permute_630, [512, 8, 1024]);  permute_630 = None
        add_158 = torch.ops.aten.add.Tensor(view_565, add_153);  view_565 = add_153 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_158, getitem_57);  add_158 = getitem_57 = None
        mul_115 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = rsqrt_28 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, arg219_1);  mul_115 = arg219_1 = None
        add_160 = torch.ops.aten.add.Tensor(mul_116, arg220_1);  mul_116 = arg220_1 = None
        view_566 = torch.ops.aten.view.default(add_160, [4096, 1024])
        permute_631 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg224_1, view_566, permute_631);  arg224_1 = view_566 = permute_631 = None
        view_567 = torch.ops.aten.view.default(addmm_28, [512, 8, 4096]);  addmm_28 = None
        mul_117 = torch.ops.aten.mul.Tensor(view_567, 0.5)
        mul_118 = torch.ops.aten.mul.Tensor(view_567, 0.7071067811865476);  view_567 = None
        erf_14 = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_161 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_117, add_161);  mul_117 = add_161 = None
        view_568 = torch.ops.aten.view.default(mul_119, [4096, 4096]);  mul_119 = None
        permute_632 = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg226_1, view_568, permute_632);  arg226_1 = view_568 = permute_632 = None
        view_569 = torch.ops.aten.view.default(addmm_29, [512, 8, 1024]);  addmm_29 = None
        add_162 = torch.ops.aten.add.Tensor(view_569, add_160);  view_569 = add_160 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_163 = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_162, getitem_59);  add_162 = getitem_59 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, arg221_1);  mul_120 = arg221_1 = None
        add_164 = torch.ops.aten.add.Tensor(mul_121, arg222_1);  mul_121 = arg222_1 = None
        slice_123 = torch.ops.aten.slice.Tensor(add_164, 0, -512, 9223372036854775807)
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(add_164, 3)
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, 4);  unsqueeze_378 = None
        permute_633 = torch.ops.aten.permute.default(unsqueeze_379, [0, 1, 3, 4, 2]);  unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg227_1, 3);  arg227_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, 4);  unsqueeze_380 = None
        permute_634 = torch.ops.aten.permute.default(unsqueeze_381, [3, 4, 1, 2, 0]);  unsqueeze_381 = None
        permute_635 = torch.ops.aten.permute.default(permute_633, [0, 1, 4, 2, 3]);  permute_633 = None
        view_570 = torch.ops.aten.view.default(permute_635, [1, 4096, 1024]);  permute_635 = None
        permute_636 = torch.ops.aten.permute.default(permute_634, [4, 2, 3, 0, 1]);  permute_634 = None
        view_571 = torch.ops.aten.view.default(permute_636, [1, 1024, 1024]);  permute_636 = None
        bmm_120 = torch.ops.aten.bmm.default(view_570, view_571);  view_570 = view_571 = None
        view_572 = torch.ops.aten.view.default(bmm_120, [512, 8, 1, 16, 64]);  bmm_120 = None
        permute_637 = torch.ops.aten.permute.default(view_572, [0, 1, 3, 4, 2]);  view_572 = None
        view_573 = torch.ops.aten.view.default(permute_637, [512, 8, 16, 64]);  permute_637 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(add_164, 3)
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, 4);  unsqueeze_382 = None
        permute_638 = torch.ops.aten.permute.default(unsqueeze_383, [0, 1, 3, 4, 2]);  unsqueeze_383 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg228_1, 3);  arg228_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, 4);  unsqueeze_384 = None
        permute_639 = torch.ops.aten.permute.default(unsqueeze_385, [3, 4, 1, 2, 0]);  unsqueeze_385 = None
        permute_640 = torch.ops.aten.permute.default(permute_638, [0, 1, 4, 2, 3]);  permute_638 = None
        view_574 = torch.ops.aten.view.default(permute_640, [1, 4096, 1024]);  permute_640 = None
        permute_641 = torch.ops.aten.permute.default(permute_639, [4, 2, 3, 0, 1]);  permute_639 = None
        view_575 = torch.ops.aten.view.default(permute_641, [1, 1024, 1024]);  permute_641 = None
        bmm_121 = torch.ops.aten.bmm.default(view_574, view_575);  view_574 = view_575 = None
        view_576 = torch.ops.aten.view.default(bmm_121, [512, 8, 1, 16, 64]);  bmm_121 = None
        permute_642 = torch.ops.aten.permute.default(view_576, [0, 1, 3, 4, 2]);  view_576 = None
        view_577 = torch.ops.aten.view.default(permute_642, [512, 8, 16, 64]);  permute_642 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(add_164, 3)
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, 4);  unsqueeze_386 = None
        permute_643 = torch.ops.aten.permute.default(unsqueeze_387, [0, 1, 3, 4, 2]);  unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg229_1, 3);  arg229_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, 4);  unsqueeze_388 = None
        permute_644 = torch.ops.aten.permute.default(unsqueeze_389, [3, 4, 1, 2, 0]);  unsqueeze_389 = None
        permute_645 = torch.ops.aten.permute.default(permute_643, [0, 1, 4, 2, 3]);  permute_643 = None
        view_578 = torch.ops.aten.view.default(permute_645, [1, 4096, 1024]);  permute_645 = None
        permute_646 = torch.ops.aten.permute.default(permute_644, [4, 2, 3, 0, 1]);  permute_644 = None
        view_579 = torch.ops.aten.view.default(permute_646, [1, 1024, 1024]);  permute_646 = None
        bmm_122 = torch.ops.aten.bmm.default(view_578, view_579);  view_578 = view_579 = None
        view_580 = torch.ops.aten.view.default(bmm_122, [512, 8, 1, 16, 64]);  bmm_122 = None
        permute_647 = torch.ops.aten.permute.default(view_580, [0, 1, 3, 4, 2]);  view_580 = None
        view_581 = torch.ops.aten.view.default(permute_647, [512, 8, 16, 64]);  permute_647 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, 4);  unsqueeze_390 = None
        permute_648 = torch.ops.aten.permute.default(unsqueeze_391, [0, 1, 3, 4, 2]);  unsqueeze_391 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg231_1, 3);  arg231_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, 4);  unsqueeze_392 = None
        permute_649 = torch.ops.aten.permute.default(unsqueeze_393, [3, 4, 1, 2, 0]);  unsqueeze_393 = None
        permute_650 = torch.ops.aten.permute.default(permute_648, [0, 1, 4, 2, 3]);  permute_648 = None
        view_582 = torch.ops.aten.view.default(permute_650, [1, 8192, 1024]);  permute_650 = None
        permute_651 = torch.ops.aten.permute.default(permute_649, [4, 2, 3, 0, 1]);  permute_649 = None
        view_583 = torch.ops.aten.view.default(permute_651, [1, 1024, 1024]);  permute_651 = None
        bmm_123 = torch.ops.aten.bmm.default(view_582, view_583);  view_582 = view_583 = None
        view_584 = torch.ops.aten.view.default(bmm_123, [1024, 8, 1, 16, 64]);  bmm_123 = None
        permute_652 = torch.ops.aten.permute.default(view_584, [0, 1, 3, 4, 2]);  view_584 = None
        view_585 = torch.ops.aten.view.default(permute_652, [1024, 8, 16, 64]);  permute_652 = None
        add_165 = torch.ops.aten.add.Tensor(view_573, arg233_1);  arg233_1 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(add_165, 4);  add_165 = None
        permute_653 = torch.ops.aten.permute.default(unsqueeze_394, [1, 2, 0, 4, 3]);  unsqueeze_394 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(view_577, 4);  view_577 = None
        permute_654 = torch.ops.aten.permute.default(unsqueeze_395, [1, 2, 4, 0, 3]);  unsqueeze_395 = None
        permute_655 = torch.ops.aten.permute.default(permute_653, [0, 1, 2, 4, 3]);  permute_653 = None
        view_586 = torch.ops.aten.view.default(permute_655, [128, 512, 64]);  permute_655 = None
        permute_656 = torch.ops.aten.permute.default(permute_654, [0, 1, 4, 3, 2]);  permute_654 = None
        view_587 = torch.ops.aten.view.default(permute_656, [128, 64, 512]);  permute_656 = None
        bmm_124 = torch.ops.aten.bmm.default(view_586, view_587);  view_586 = view_587 = None
        view_588 = torch.ops.aten.view.default(bmm_124, [8, 16, 512, 1, 512]);  bmm_124 = None
        permute_657 = torch.ops.aten.permute.default(view_588, [0, 1, 2, 4, 3]);  view_588 = None
        view_589 = torch.ops.aten.view.default(permute_657, [8, 16, 512, 512]);  permute_657 = None
        add_166 = torch.ops.aten.add.Tensor(view_573, arg232_1);  view_573 = arg232_1 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(add_166, 4);  add_166 = None
        permute_658 = torch.ops.aten.permute.default(unsqueeze_396, [1, 2, 0, 4, 3]);  unsqueeze_396 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(view_585, 4);  view_585 = None
        permute_659 = torch.ops.aten.permute.default(unsqueeze_397, [1, 2, 4, 0, 3]);  unsqueeze_397 = None
        permute_660 = torch.ops.aten.permute.default(permute_658, [0, 1, 2, 4, 3]);  permute_658 = None
        view_590 = torch.ops.aten.view.default(permute_660, [128, 512, 64]);  permute_660 = None
        permute_661 = torch.ops.aten.permute.default(permute_659, [0, 1, 4, 3, 2]);  permute_659 = None
        view_591 = torch.ops.aten.view.default(permute_661, [128, 64, 1024]);  permute_661 = None
        bmm_125 = torch.ops.aten.bmm.default(view_590, view_591);  view_590 = view_591 = None
        view_592 = torch.ops.aten.view.default(bmm_125, [8, 16, 512, 1, 1024]);  bmm_125 = None
        permute_662 = torch.ops.aten.permute.default(view_592, [0, 1, 2, 4, 3]);  view_592 = None
        view_593 = torch.ops.aten.view.default(permute_662, [8, 16, 512, 1024]);  permute_662 = None
        view_594 = torch.ops.aten.view.default(view_593, [8, 16, 1024, 512]);  view_593 = None
        slice_126 = torch.ops.aten.slice.Tensor(view_594, 2, 1, 9223372036854775807);  view_594 = None
        view_595 = torch.ops.aten.view.default(slice_126, [8, 16, 512, 1023]);  slice_126 = None
        iota_17 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_15 = torch.ops.aten.index.Tensor(view_595, [None, None, None, iota_17]);  view_595 = iota_17 = None
        add_167 = torch.ops.aten.add.Tensor(view_589, index_15);  view_589 = index_15 = None
        add_168 = torch.ops.aten.add.Tensor(add_167, 0);  add_167 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(add_168, 1);  add_168 = None
        amax_default_8 = torch.ops.aten.amax.default(mul_tensor_16, [3], True)
        sub_tensor_8 = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = amax_default_8 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        exp_15 = torch.ops.aten.exp.default(mul_tensor_17);  mul_tensor_17 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
        div_16 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(div_16, 4);  div_16 = None
        permute_663 = torch.ops.aten.permute.default(unsqueeze_398, [2, 0, 1, 4, 3]);  unsqueeze_398 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(view_581, 4);  view_581 = None
        permute_664 = torch.ops.aten.permute.default(unsqueeze_399, [4, 1, 2, 3, 0]);  unsqueeze_399 = None
        permute_665 = torch.ops.aten.permute.default(permute_663, [1, 2, 0, 4, 3]);  permute_663 = None
        view_596 = torch.ops.aten.view.default(permute_665, [128, 512, 512]);  permute_665 = None
        permute_666 = torch.ops.aten.permute.default(permute_664, [1, 2, 4, 3, 0]);  permute_664 = None
        view_597 = torch.ops.aten.view.default(permute_666, [128, 512, 64]);  permute_666 = None
        bmm_126 = torch.ops.aten.bmm.default(view_596, view_597);  view_596 = view_597 = None
        view_598 = torch.ops.aten.view.default(bmm_126, [8, 16, 512, 1, 64]);  bmm_126 = None
        permute_667 = torch.ops.aten.permute.default(view_598, [2, 0, 1, 4, 3]);  view_598 = None
        view_599 = torch.ops.aten.view.default(permute_667, [512, 8, 16, 64]);  permute_667 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(view_599, 4);  view_599 = None
        permute_668 = torch.ops.aten.permute.default(unsqueeze_400, [0, 1, 4, 3, 2]);  unsqueeze_400 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(arg230_1, 3);  arg230_1 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(unsqueeze_401, 4);  unsqueeze_401 = None
        permute_669 = torch.ops.aten.permute.default(unsqueeze_402, [3, 4, 0, 2, 1]);  unsqueeze_402 = None
        permute_670 = torch.ops.aten.permute.default(permute_668, [0, 1, 3, 4, 2]);  permute_668 = None
        clone_94 = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
        view_600 = torch.ops.aten.view.default(clone_94, [1, 4096, 1024]);  clone_94 = None
        permute_671 = torch.ops.aten.permute.default(permute_669, [3, 4, 2, 0, 1]);  permute_669 = None
        clone_95 = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
        view_601 = torch.ops.aten.view.default(clone_95, [1, 1024, 1024]);  clone_95 = None
        bmm_127 = torch.ops.aten.bmm.default(view_600, view_601);  view_600 = view_601 = None
        view_602 = torch.ops.aten.view.default(bmm_127, [512, 8, 1, 1, 1024]);  bmm_127 = None
        permute_672 = torch.ops.aten.permute.default(view_602, [0, 1, 4, 2, 3]);  view_602 = None
        view_603 = torch.ops.aten.view.default(permute_672, [512, 8, 1024]);  permute_672 = None
        add_169 = torch.ops.aten.add.Tensor(view_603, add_164);  view_603 = add_164 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_169, getitem_61);  add_169 = getitem_61 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, arg234_1);  mul_123 = arg234_1 = None
        add_171 = torch.ops.aten.add.Tensor(mul_124, arg235_1);  mul_124 = arg235_1 = None
        view_604 = torch.ops.aten.view.default(add_171, [4096, 1024])
        permute_673 = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg239_1, view_604, permute_673);  arg239_1 = view_604 = permute_673 = None
        view_605 = torch.ops.aten.view.default(addmm_30, [512, 8, 4096]);  addmm_30 = None
        mul_125 = torch.ops.aten.mul.Tensor(view_605, 0.5)
        mul_126 = torch.ops.aten.mul.Tensor(view_605, 0.7071067811865476);  view_605 = None
        erf_15 = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_172 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_125, add_172);  mul_125 = add_172 = None
        view_606 = torch.ops.aten.view.default(mul_127, [4096, 4096]);  mul_127 = None
        permute_674 = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg241_1, view_606, permute_674);  arg241_1 = view_606 = permute_674 = None
        view_607 = torch.ops.aten.view.default(addmm_31, [512, 8, 1024]);  addmm_31 = None
        add_173 = torch.ops.aten.add.Tensor(view_607, add_171);  view_607 = add_171 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_173, getitem_63);  add_173 = getitem_63 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, arg236_1);  mul_128 = arg236_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_129, arg237_1);  mul_129 = arg237_1 = None
        slice_131 = torch.ops.aten.slice.Tensor(add_175, 0, -512, 9223372036854775807)
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(add_175, 3)
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(unsqueeze_403, 4);  unsqueeze_403 = None
        permute_675 = torch.ops.aten.permute.default(unsqueeze_404, [0, 1, 3, 4, 2]);  unsqueeze_404 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(arg242_1, 3);  arg242_1 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(unsqueeze_405, 4);  unsqueeze_405 = None
        permute_676 = torch.ops.aten.permute.default(unsqueeze_406, [3, 4, 1, 2, 0]);  unsqueeze_406 = None
        permute_677 = torch.ops.aten.permute.default(permute_675, [0, 1, 4, 2, 3]);  permute_675 = None
        view_608 = torch.ops.aten.view.default(permute_677, [1, 4096, 1024]);  permute_677 = None
        permute_678 = torch.ops.aten.permute.default(permute_676, [4, 2, 3, 0, 1]);  permute_676 = None
        view_609 = torch.ops.aten.view.default(permute_678, [1, 1024, 1024]);  permute_678 = None
        bmm_128 = torch.ops.aten.bmm.default(view_608, view_609);  view_608 = view_609 = None
        view_610 = torch.ops.aten.view.default(bmm_128, [512, 8, 1, 16, 64]);  bmm_128 = None
        permute_679 = torch.ops.aten.permute.default(view_610, [0, 1, 3, 4, 2]);  view_610 = None
        view_611 = torch.ops.aten.view.default(permute_679, [512, 8, 16, 64]);  permute_679 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(add_175, 3)
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(unsqueeze_407, 4);  unsqueeze_407 = None
        permute_680 = torch.ops.aten.permute.default(unsqueeze_408, [0, 1, 3, 4, 2]);  unsqueeze_408 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(arg243_1, 3);  arg243_1 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(unsqueeze_409, 4);  unsqueeze_409 = None
        permute_681 = torch.ops.aten.permute.default(unsqueeze_410, [3, 4, 1, 2, 0]);  unsqueeze_410 = None
        permute_682 = torch.ops.aten.permute.default(permute_680, [0, 1, 4, 2, 3]);  permute_680 = None
        view_612 = torch.ops.aten.view.default(permute_682, [1, 4096, 1024]);  permute_682 = None
        permute_683 = torch.ops.aten.permute.default(permute_681, [4, 2, 3, 0, 1]);  permute_681 = None
        view_613 = torch.ops.aten.view.default(permute_683, [1, 1024, 1024]);  permute_683 = None
        bmm_129 = torch.ops.aten.bmm.default(view_612, view_613);  view_612 = view_613 = None
        view_614 = torch.ops.aten.view.default(bmm_129, [512, 8, 1, 16, 64]);  bmm_129 = None
        permute_684 = torch.ops.aten.permute.default(view_614, [0, 1, 3, 4, 2]);  view_614 = None
        view_615 = torch.ops.aten.view.default(permute_684, [512, 8, 16, 64]);  permute_684 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(add_175, 3)
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(unsqueeze_411, 4);  unsqueeze_411 = None
        permute_685 = torch.ops.aten.permute.default(unsqueeze_412, [0, 1, 3, 4, 2]);  unsqueeze_412 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(arg244_1, 3);  arg244_1 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
        permute_686 = torch.ops.aten.permute.default(unsqueeze_414, [3, 4, 1, 2, 0]);  unsqueeze_414 = None
        permute_687 = torch.ops.aten.permute.default(permute_685, [0, 1, 4, 2, 3]);  permute_685 = None
        view_616 = torch.ops.aten.view.default(permute_687, [1, 4096, 1024]);  permute_687 = None
        permute_688 = torch.ops.aten.permute.default(permute_686, [4, 2, 3, 0, 1]);  permute_686 = None
        view_617 = torch.ops.aten.view.default(permute_688, [1, 1024, 1024]);  permute_688 = None
        bmm_130 = torch.ops.aten.bmm.default(view_616, view_617);  view_616 = view_617 = None
        view_618 = torch.ops.aten.view.default(bmm_130, [512, 8, 1, 16, 64]);  bmm_130 = None
        permute_689 = torch.ops.aten.permute.default(view_618, [0, 1, 3, 4, 2]);  view_618 = None
        view_619 = torch.ops.aten.view.default(permute_689, [512, 8, 16, 64]);  permute_689 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(unsqueeze_415, 4);  unsqueeze_415 = None
        permute_690 = torch.ops.aten.permute.default(unsqueeze_416, [0, 1, 3, 4, 2]);  unsqueeze_416 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(arg246_1, 3);  arg246_1 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(unsqueeze_417, 4);  unsqueeze_417 = None
        permute_691 = torch.ops.aten.permute.default(unsqueeze_418, [3, 4, 1, 2, 0]);  unsqueeze_418 = None
        permute_692 = torch.ops.aten.permute.default(permute_690, [0, 1, 4, 2, 3]);  permute_690 = None
        view_620 = torch.ops.aten.view.default(permute_692, [1, 8192, 1024]);  permute_692 = None
        permute_693 = torch.ops.aten.permute.default(permute_691, [4, 2, 3, 0, 1]);  permute_691 = None
        view_621 = torch.ops.aten.view.default(permute_693, [1, 1024, 1024]);  permute_693 = None
        bmm_131 = torch.ops.aten.bmm.default(view_620, view_621);  view_620 = view_621 = None
        view_622 = torch.ops.aten.view.default(bmm_131, [1024, 8, 1, 16, 64]);  bmm_131 = None
        permute_694 = torch.ops.aten.permute.default(view_622, [0, 1, 3, 4, 2]);  view_622 = None
        view_623 = torch.ops.aten.view.default(permute_694, [1024, 8, 16, 64]);  permute_694 = None
        add_176 = torch.ops.aten.add.Tensor(view_611, arg248_1);  arg248_1 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(add_176, 4);  add_176 = None
        permute_695 = torch.ops.aten.permute.default(unsqueeze_419, [1, 2, 0, 4, 3]);  unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(view_615, 4);  view_615 = None
        permute_696 = torch.ops.aten.permute.default(unsqueeze_420, [1, 2, 4, 0, 3]);  unsqueeze_420 = None
        permute_697 = torch.ops.aten.permute.default(permute_695, [0, 1, 2, 4, 3]);  permute_695 = None
        view_624 = torch.ops.aten.view.default(permute_697, [128, 512, 64]);  permute_697 = None
        permute_698 = torch.ops.aten.permute.default(permute_696, [0, 1, 4, 3, 2]);  permute_696 = None
        view_625 = torch.ops.aten.view.default(permute_698, [128, 64, 512]);  permute_698 = None
        bmm_132 = torch.ops.aten.bmm.default(view_624, view_625);  view_624 = view_625 = None
        view_626 = torch.ops.aten.view.default(bmm_132, [8, 16, 512, 1, 512]);  bmm_132 = None
        permute_699 = torch.ops.aten.permute.default(view_626, [0, 1, 2, 4, 3]);  view_626 = None
        view_627 = torch.ops.aten.view.default(permute_699, [8, 16, 512, 512]);  permute_699 = None
        add_177 = torch.ops.aten.add.Tensor(view_611, arg247_1);  view_611 = arg247_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(add_177, 4);  add_177 = None
        permute_700 = torch.ops.aten.permute.default(unsqueeze_421, [1, 2, 0, 4, 3]);  unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(view_623, 4);  view_623 = None
        permute_701 = torch.ops.aten.permute.default(unsqueeze_422, [1, 2, 4, 0, 3]);  unsqueeze_422 = None
        permute_702 = torch.ops.aten.permute.default(permute_700, [0, 1, 2, 4, 3]);  permute_700 = None
        view_628 = torch.ops.aten.view.default(permute_702, [128, 512, 64]);  permute_702 = None
        permute_703 = torch.ops.aten.permute.default(permute_701, [0, 1, 4, 3, 2]);  permute_701 = None
        view_629 = torch.ops.aten.view.default(permute_703, [128, 64, 1024]);  permute_703 = None
        bmm_133 = torch.ops.aten.bmm.default(view_628, view_629);  view_628 = view_629 = None
        view_630 = torch.ops.aten.view.default(bmm_133, [8, 16, 512, 1, 1024]);  bmm_133 = None
        permute_704 = torch.ops.aten.permute.default(view_630, [0, 1, 2, 4, 3]);  view_630 = None
        view_631 = torch.ops.aten.view.default(permute_704, [8, 16, 512, 1024]);  permute_704 = None
        view_632 = torch.ops.aten.view.default(view_631, [8, 16, 1024, 512]);  view_631 = None
        slice_134 = torch.ops.aten.slice.Tensor(view_632, 2, 1, 9223372036854775807);  view_632 = None
        view_633 = torch.ops.aten.view.default(slice_134, [8, 16, 512, 1023]);  slice_134 = None
        iota_18 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_16 = torch.ops.aten.index.Tensor(view_633, [None, None, None, iota_18]);  view_633 = iota_18 = None
        add_178 = torch.ops.aten.add.Tensor(view_627, index_16);  view_627 = index_16 = None
        add_179 = torch.ops.aten.add.Tensor(add_178, 0);  add_178 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(add_179, 1);  add_179 = None
        amax_default_7 = torch.ops.aten.amax.default(mul_tensor_14, [3], True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, 0.125);  sub_tensor_7 = None
        exp_16 = torch.ops.aten.exp.default(mul_tensor_15);  mul_tensor_15 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
        div_17 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(div_17, 4);  div_17 = None
        permute_705 = torch.ops.aten.permute.default(unsqueeze_423, [2, 0, 1, 4, 3]);  unsqueeze_423 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(view_619, 4);  view_619 = None
        permute_706 = torch.ops.aten.permute.default(unsqueeze_424, [4, 1, 2, 3, 0]);  unsqueeze_424 = None
        permute_707 = torch.ops.aten.permute.default(permute_705, [1, 2, 0, 4, 3]);  permute_705 = None
        view_634 = torch.ops.aten.view.default(permute_707, [128, 512, 512]);  permute_707 = None
        permute_708 = torch.ops.aten.permute.default(permute_706, [1, 2, 4, 3, 0]);  permute_706 = None
        view_635 = torch.ops.aten.view.default(permute_708, [128, 512, 64]);  permute_708 = None
        bmm_134 = torch.ops.aten.bmm.default(view_634, view_635);  view_634 = view_635 = None
        view_636 = torch.ops.aten.view.default(bmm_134, [8, 16, 512, 1, 64]);  bmm_134 = None
        permute_709 = torch.ops.aten.permute.default(view_636, [2, 0, 1, 4, 3]);  view_636 = None
        view_637 = torch.ops.aten.view.default(permute_709, [512, 8, 16, 64]);  permute_709 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
        permute_710 = torch.ops.aten.permute.default(unsqueeze_425, [0, 1, 4, 3, 2]);  unsqueeze_425 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(arg245_1, 3);  arg245_1 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, 4);  unsqueeze_426 = None
        permute_711 = torch.ops.aten.permute.default(unsqueeze_427, [3, 4, 0, 2, 1]);  unsqueeze_427 = None
        permute_712 = torch.ops.aten.permute.default(permute_710, [0, 1, 3, 4, 2]);  permute_710 = None
        clone_100 = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
        view_638 = torch.ops.aten.view.default(clone_100, [1, 4096, 1024]);  clone_100 = None
        permute_713 = torch.ops.aten.permute.default(permute_711, [3, 4, 2, 0, 1]);  permute_711 = None
        clone_101 = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
        view_639 = torch.ops.aten.view.default(clone_101, [1, 1024, 1024]);  clone_101 = None
        bmm_135 = torch.ops.aten.bmm.default(view_638, view_639);  view_638 = view_639 = None
        view_640 = torch.ops.aten.view.default(bmm_135, [512, 8, 1, 1, 1024]);  bmm_135 = None
        permute_714 = torch.ops.aten.permute.default(view_640, [0, 1, 4, 2, 3]);  view_640 = None
        view_641 = torch.ops.aten.view.default(permute_714, [512, 8, 1024]);  permute_714 = None
        add_180 = torch.ops.aten.add.Tensor(view_641, add_175);  view_641 = add_175 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_180, getitem_65);  add_180 = getitem_65 = None
        mul_131 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_131, arg249_1);  mul_131 = arg249_1 = None
        add_182 = torch.ops.aten.add.Tensor(mul_132, arg250_1);  mul_132 = arg250_1 = None
        view_642 = torch.ops.aten.view.default(add_182, [4096, 1024])
        permute_715 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg254_1, view_642, permute_715);  arg254_1 = view_642 = permute_715 = None
        view_643 = torch.ops.aten.view.default(addmm_32, [512, 8, 4096]);  addmm_32 = None
        mul_133 = torch.ops.aten.mul.Tensor(view_643, 0.5)
        mul_134 = torch.ops.aten.mul.Tensor(view_643, 0.7071067811865476);  view_643 = None
        erf_16 = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_183 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_133, add_183);  mul_133 = add_183 = None
        view_644 = torch.ops.aten.view.default(mul_135, [4096, 4096]);  mul_135 = None
        permute_716 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg256_1, view_644, permute_716);  arg256_1 = view_644 = permute_716 = None
        view_645 = torch.ops.aten.view.default(addmm_33, [512, 8, 1024]);  addmm_33 = None
        add_184 = torch.ops.aten.add.Tensor(view_645, add_182);  view_645 = add_182 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_185 = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_50 = torch.ops.aten.sub.Tensor(add_184, getitem_67);  add_184 = getitem_67 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, arg251_1);  mul_136 = arg251_1 = None
        add_186 = torch.ops.aten.add.Tensor(mul_137, arg252_1);  mul_137 = arg252_1 = None
        slice_139 = torch.ops.aten.slice.Tensor(add_186, 0, -512, 9223372036854775807)
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(add_186, 3)
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, 4);  unsqueeze_428 = None
        permute_717 = torch.ops.aten.permute.default(unsqueeze_429, [0, 1, 3, 4, 2]);  unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg257_1, 3);  arg257_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, 4);  unsqueeze_430 = None
        permute_718 = torch.ops.aten.permute.default(unsqueeze_431, [3, 4, 1, 2, 0]);  unsqueeze_431 = None
        permute_719 = torch.ops.aten.permute.default(permute_717, [0, 1, 4, 2, 3]);  permute_717 = None
        view_646 = torch.ops.aten.view.default(permute_719, [1, 4096, 1024]);  permute_719 = None
        permute_720 = torch.ops.aten.permute.default(permute_718, [4, 2, 3, 0, 1]);  permute_718 = None
        view_647 = torch.ops.aten.view.default(permute_720, [1, 1024, 1024]);  permute_720 = None
        bmm_136 = torch.ops.aten.bmm.default(view_646, view_647);  view_646 = view_647 = None
        view_648 = torch.ops.aten.view.default(bmm_136, [512, 8, 1, 16, 64]);  bmm_136 = None
        permute_721 = torch.ops.aten.permute.default(view_648, [0, 1, 3, 4, 2]);  view_648 = None
        view_649 = torch.ops.aten.view.default(permute_721, [512, 8, 16, 64]);  permute_721 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(add_186, 3)
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, 4);  unsqueeze_432 = None
        permute_722 = torch.ops.aten.permute.default(unsqueeze_433, [0, 1, 3, 4, 2]);  unsqueeze_433 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(arg258_1, 3);  arg258_1 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, 4);  unsqueeze_434 = None
        permute_723 = torch.ops.aten.permute.default(unsqueeze_435, [3, 4, 1, 2, 0]);  unsqueeze_435 = None
        permute_724 = torch.ops.aten.permute.default(permute_722, [0, 1, 4, 2, 3]);  permute_722 = None
        view_650 = torch.ops.aten.view.default(permute_724, [1, 4096, 1024]);  permute_724 = None
        permute_725 = torch.ops.aten.permute.default(permute_723, [4, 2, 3, 0, 1]);  permute_723 = None
        view_651 = torch.ops.aten.view.default(permute_725, [1, 1024, 1024]);  permute_725 = None
        bmm_137 = torch.ops.aten.bmm.default(view_650, view_651);  view_650 = view_651 = None
        view_652 = torch.ops.aten.view.default(bmm_137, [512, 8, 1, 16, 64]);  bmm_137 = None
        permute_726 = torch.ops.aten.permute.default(view_652, [0, 1, 3, 4, 2]);  view_652 = None
        view_653 = torch.ops.aten.view.default(permute_726, [512, 8, 16, 64]);  permute_726 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(add_186, 3)
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, 4);  unsqueeze_436 = None
        permute_727 = torch.ops.aten.permute.default(unsqueeze_437, [0, 1, 3, 4, 2]);  unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg259_1, 3);  arg259_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, 4);  unsqueeze_438 = None
        permute_728 = torch.ops.aten.permute.default(unsqueeze_439, [3, 4, 1, 2, 0]);  unsqueeze_439 = None
        permute_729 = torch.ops.aten.permute.default(permute_727, [0, 1, 4, 2, 3]);  permute_727 = None
        view_654 = torch.ops.aten.view.default(permute_729, [1, 4096, 1024]);  permute_729 = None
        permute_730 = torch.ops.aten.permute.default(permute_728, [4, 2, 3, 0, 1]);  permute_728 = None
        view_655 = torch.ops.aten.view.default(permute_730, [1, 1024, 1024]);  permute_730 = None
        bmm_138 = torch.ops.aten.bmm.default(view_654, view_655);  view_654 = view_655 = None
        view_656 = torch.ops.aten.view.default(bmm_138, [512, 8, 1, 16, 64]);  bmm_138 = None
        permute_731 = torch.ops.aten.permute.default(view_656, [0, 1, 3, 4, 2]);  view_656 = None
        view_657 = torch.ops.aten.view.default(permute_731, [512, 8, 16, 64]);  permute_731 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, 4);  unsqueeze_440 = None
        permute_732 = torch.ops.aten.permute.default(unsqueeze_441, [0, 1, 3, 4, 2]);  unsqueeze_441 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(arg261_1, 3);  arg261_1 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, 4);  unsqueeze_442 = None
        permute_733 = torch.ops.aten.permute.default(unsqueeze_443, [3, 4, 1, 2, 0]);  unsqueeze_443 = None
        permute_734 = torch.ops.aten.permute.default(permute_732, [0, 1, 4, 2, 3]);  permute_732 = None
        view_658 = torch.ops.aten.view.default(permute_734, [1, 8192, 1024]);  permute_734 = None
        permute_735 = torch.ops.aten.permute.default(permute_733, [4, 2, 3, 0, 1]);  permute_733 = None
        view_659 = torch.ops.aten.view.default(permute_735, [1, 1024, 1024]);  permute_735 = None
        bmm_139 = torch.ops.aten.bmm.default(view_658, view_659);  view_658 = view_659 = None
        view_660 = torch.ops.aten.view.default(bmm_139, [1024, 8, 1, 16, 64]);  bmm_139 = None
        permute_736 = torch.ops.aten.permute.default(view_660, [0, 1, 3, 4, 2]);  view_660 = None
        view_661 = torch.ops.aten.view.default(permute_736, [1024, 8, 16, 64]);  permute_736 = None
        add_187 = torch.ops.aten.add.Tensor(view_649, arg263_1);  arg263_1 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(add_187, 4);  add_187 = None
        permute_737 = torch.ops.aten.permute.default(unsqueeze_444, [1, 2, 0, 4, 3]);  unsqueeze_444 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(view_653, 4);  view_653 = None
        permute_738 = torch.ops.aten.permute.default(unsqueeze_445, [1, 2, 4, 0, 3]);  unsqueeze_445 = None
        permute_739 = torch.ops.aten.permute.default(permute_737, [0, 1, 2, 4, 3]);  permute_737 = None
        view_662 = torch.ops.aten.view.default(permute_739, [128, 512, 64]);  permute_739 = None
        permute_740 = torch.ops.aten.permute.default(permute_738, [0, 1, 4, 3, 2]);  permute_738 = None
        view_663 = torch.ops.aten.view.default(permute_740, [128, 64, 512]);  permute_740 = None
        bmm_140 = torch.ops.aten.bmm.default(view_662, view_663);  view_662 = view_663 = None
        view_664 = torch.ops.aten.view.default(bmm_140, [8, 16, 512, 1, 512]);  bmm_140 = None
        permute_741 = torch.ops.aten.permute.default(view_664, [0, 1, 2, 4, 3]);  view_664 = None
        view_665 = torch.ops.aten.view.default(permute_741, [8, 16, 512, 512]);  permute_741 = None
        add_188 = torch.ops.aten.add.Tensor(view_649, arg262_1);  view_649 = arg262_1 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(add_188, 4);  add_188 = None
        permute_742 = torch.ops.aten.permute.default(unsqueeze_446, [1, 2, 0, 4, 3]);  unsqueeze_446 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(view_661, 4);  view_661 = None
        permute_743 = torch.ops.aten.permute.default(unsqueeze_447, [1, 2, 4, 0, 3]);  unsqueeze_447 = None
        permute_744 = torch.ops.aten.permute.default(permute_742, [0, 1, 2, 4, 3]);  permute_742 = None
        view_666 = torch.ops.aten.view.default(permute_744, [128, 512, 64]);  permute_744 = None
        permute_745 = torch.ops.aten.permute.default(permute_743, [0, 1, 4, 3, 2]);  permute_743 = None
        view_667 = torch.ops.aten.view.default(permute_745, [128, 64, 1024]);  permute_745 = None
        bmm_141 = torch.ops.aten.bmm.default(view_666, view_667);  view_666 = view_667 = None
        view_668 = torch.ops.aten.view.default(bmm_141, [8, 16, 512, 1, 1024]);  bmm_141 = None
        permute_746 = torch.ops.aten.permute.default(view_668, [0, 1, 2, 4, 3]);  view_668 = None
        view_669 = torch.ops.aten.view.default(permute_746, [8, 16, 512, 1024]);  permute_746 = None
        view_670 = torch.ops.aten.view.default(view_669, [8, 16, 1024, 512]);  view_669 = None
        slice_142 = torch.ops.aten.slice.Tensor(view_670, 2, 1, 9223372036854775807);  view_670 = None
        view_671 = torch.ops.aten.view.default(slice_142, [8, 16, 512, 1023]);  slice_142 = None
        iota_19 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_17 = torch.ops.aten.index.Tensor(view_671, [None, None, None, iota_19]);  view_671 = iota_19 = None
        add_189 = torch.ops.aten.add.Tensor(view_665, index_17);  view_665 = index_17 = None
        add_190 = torch.ops.aten.add.Tensor(add_189, 0);  add_189 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(add_190, 1);  add_190 = None
        amax_default_6 = torch.ops.aten.amax.default(mul_tensor_12, [3], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        exp_17 = torch.ops.aten.exp.default(mul_tensor_13);  mul_tensor_13 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
        div_18 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(div_18, 4);  div_18 = None
        permute_747 = torch.ops.aten.permute.default(unsqueeze_448, [2, 0, 1, 4, 3]);  unsqueeze_448 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(view_657, 4);  view_657 = None
        permute_748 = torch.ops.aten.permute.default(unsqueeze_449, [4, 1, 2, 3, 0]);  unsqueeze_449 = None
        permute_749 = torch.ops.aten.permute.default(permute_747, [1, 2, 0, 4, 3]);  permute_747 = None
        view_672 = torch.ops.aten.view.default(permute_749, [128, 512, 512]);  permute_749 = None
        permute_750 = torch.ops.aten.permute.default(permute_748, [1, 2, 4, 3, 0]);  permute_748 = None
        view_673 = torch.ops.aten.view.default(permute_750, [128, 512, 64]);  permute_750 = None
        bmm_142 = torch.ops.aten.bmm.default(view_672, view_673);  view_672 = view_673 = None
        view_674 = torch.ops.aten.view.default(bmm_142, [8, 16, 512, 1, 64]);  bmm_142 = None
        permute_751 = torch.ops.aten.permute.default(view_674, [2, 0, 1, 4, 3]);  view_674 = None
        view_675 = torch.ops.aten.view.default(permute_751, [512, 8, 16, 64]);  permute_751 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
        permute_752 = torch.ops.aten.permute.default(unsqueeze_450, [0, 1, 4, 3, 2]);  unsqueeze_450 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(arg260_1, 3);  arg260_1 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(unsqueeze_451, 4);  unsqueeze_451 = None
        permute_753 = torch.ops.aten.permute.default(unsqueeze_452, [3, 4, 0, 2, 1]);  unsqueeze_452 = None
        permute_754 = torch.ops.aten.permute.default(permute_752, [0, 1, 3, 4, 2]);  permute_752 = None
        clone_106 = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_676 = torch.ops.aten.view.default(clone_106, [1, 4096, 1024]);  clone_106 = None
        permute_755 = torch.ops.aten.permute.default(permute_753, [3, 4, 2, 0, 1]);  permute_753 = None
        clone_107 = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
        view_677 = torch.ops.aten.view.default(clone_107, [1, 1024, 1024]);  clone_107 = None
        bmm_143 = torch.ops.aten.bmm.default(view_676, view_677);  view_676 = view_677 = None
        view_678 = torch.ops.aten.view.default(bmm_143, [512, 8, 1, 1, 1024]);  bmm_143 = None
        permute_756 = torch.ops.aten.permute.default(view_678, [0, 1, 4, 2, 3]);  view_678 = None
        view_679 = torch.ops.aten.view.default(permute_756, [512, 8, 1024]);  permute_756 = None
        add_191 = torch.ops.aten.add.Tensor(view_679, add_186);  view_679 = add_186 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_191, getitem_69);  add_191 = getitem_69 = None
        mul_139 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, arg264_1);  mul_139 = arg264_1 = None
        add_193 = torch.ops.aten.add.Tensor(mul_140, arg265_1);  mul_140 = arg265_1 = None
        view_680 = torch.ops.aten.view.default(add_193, [4096, 1024])
        permute_757 = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg269_1, view_680, permute_757);  arg269_1 = view_680 = permute_757 = None
        view_681 = torch.ops.aten.view.default(addmm_34, [512, 8, 4096]);  addmm_34 = None
        mul_141 = torch.ops.aten.mul.Tensor(view_681, 0.5)
        mul_142 = torch.ops.aten.mul.Tensor(view_681, 0.7071067811865476);  view_681 = None
        erf_17 = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_194 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_141, add_194);  mul_141 = add_194 = None
        view_682 = torch.ops.aten.view.default(mul_143, [4096, 4096]);  mul_143 = None
        permute_758 = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg271_1, view_682, permute_758);  arg271_1 = view_682 = permute_758 = None
        view_683 = torch.ops.aten.view.default(addmm_35, [512, 8, 1024]);  addmm_35 = None
        add_195 = torch.ops.aten.add.Tensor(view_683, add_193);  view_683 = add_193 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_195, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_196 = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_53 = torch.ops.aten.sub.Tensor(add_195, getitem_71);  add_195 = getitem_71 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, arg266_1);  mul_144 = arg266_1 = None
        add_197 = torch.ops.aten.add.Tensor(mul_145, arg267_1);  mul_145 = arg267_1 = None
        slice_147 = torch.ops.aten.slice.Tensor(add_197, 0, -512, 9223372036854775807)
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(add_197, 3)
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(unsqueeze_453, 4);  unsqueeze_453 = None
        permute_759 = torch.ops.aten.permute.default(unsqueeze_454, [0, 1, 3, 4, 2]);  unsqueeze_454 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(arg272_1, 3);  arg272_1 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(unsqueeze_455, 4);  unsqueeze_455 = None
        permute_760 = torch.ops.aten.permute.default(unsqueeze_456, [3, 4, 1, 2, 0]);  unsqueeze_456 = None
        permute_761 = torch.ops.aten.permute.default(permute_759, [0, 1, 4, 2, 3]);  permute_759 = None
        view_684 = torch.ops.aten.view.default(permute_761, [1, 4096, 1024]);  permute_761 = None
        permute_762 = torch.ops.aten.permute.default(permute_760, [4, 2, 3, 0, 1]);  permute_760 = None
        view_685 = torch.ops.aten.view.default(permute_762, [1, 1024, 1024]);  permute_762 = None
        bmm_144 = torch.ops.aten.bmm.default(view_684, view_685);  view_684 = view_685 = None
        view_686 = torch.ops.aten.view.default(bmm_144, [512, 8, 1, 16, 64]);  bmm_144 = None
        permute_763 = torch.ops.aten.permute.default(view_686, [0, 1, 3, 4, 2]);  view_686 = None
        view_687 = torch.ops.aten.view.default(permute_763, [512, 8, 16, 64]);  permute_763 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(add_197, 3)
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(unsqueeze_457, 4);  unsqueeze_457 = None
        permute_764 = torch.ops.aten.permute.default(unsqueeze_458, [0, 1, 3, 4, 2]);  unsqueeze_458 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(arg273_1, 3);  arg273_1 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(unsqueeze_459, 4);  unsqueeze_459 = None
        permute_765 = torch.ops.aten.permute.default(unsqueeze_460, [3, 4, 1, 2, 0]);  unsqueeze_460 = None
        permute_766 = torch.ops.aten.permute.default(permute_764, [0, 1, 4, 2, 3]);  permute_764 = None
        view_688 = torch.ops.aten.view.default(permute_766, [1, 4096, 1024]);  permute_766 = None
        permute_767 = torch.ops.aten.permute.default(permute_765, [4, 2, 3, 0, 1]);  permute_765 = None
        view_689 = torch.ops.aten.view.default(permute_767, [1, 1024, 1024]);  permute_767 = None
        bmm_145 = torch.ops.aten.bmm.default(view_688, view_689);  view_688 = view_689 = None
        view_690 = torch.ops.aten.view.default(bmm_145, [512, 8, 1, 16, 64]);  bmm_145 = None
        permute_768 = torch.ops.aten.permute.default(view_690, [0, 1, 3, 4, 2]);  view_690 = None
        view_691 = torch.ops.aten.view.default(permute_768, [512, 8, 16, 64]);  permute_768 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(add_197, 3)
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(unsqueeze_461, 4);  unsqueeze_461 = None
        permute_769 = torch.ops.aten.permute.default(unsqueeze_462, [0, 1, 3, 4, 2]);  unsqueeze_462 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(arg274_1, 3);  arg274_1 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(unsqueeze_463, 4);  unsqueeze_463 = None
        permute_770 = torch.ops.aten.permute.default(unsqueeze_464, [3, 4, 1, 2, 0]);  unsqueeze_464 = None
        permute_771 = torch.ops.aten.permute.default(permute_769, [0, 1, 4, 2, 3]);  permute_769 = None
        view_692 = torch.ops.aten.view.default(permute_771, [1, 4096, 1024]);  permute_771 = None
        permute_772 = torch.ops.aten.permute.default(permute_770, [4, 2, 3, 0, 1]);  permute_770 = None
        view_693 = torch.ops.aten.view.default(permute_772, [1, 1024, 1024]);  permute_772 = None
        bmm_146 = torch.ops.aten.bmm.default(view_692, view_693);  view_692 = view_693 = None
        view_694 = torch.ops.aten.view.default(bmm_146, [512, 8, 1, 16, 64]);  bmm_146 = None
        permute_773 = torch.ops.aten.permute.default(view_694, [0, 1, 3, 4, 2]);  view_694 = None
        view_695 = torch.ops.aten.view.default(permute_773, [512, 8, 16, 64]);  permute_773 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(unsqueeze_465, 4);  unsqueeze_465 = None
        permute_774 = torch.ops.aten.permute.default(unsqueeze_466, [0, 1, 3, 4, 2]);  unsqueeze_466 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(arg276_1, 3);  arg276_1 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(unsqueeze_467, 4);  unsqueeze_467 = None
        permute_775 = torch.ops.aten.permute.default(unsqueeze_468, [3, 4, 1, 2, 0]);  unsqueeze_468 = None
        permute_776 = torch.ops.aten.permute.default(permute_774, [0, 1, 4, 2, 3]);  permute_774 = None
        view_696 = torch.ops.aten.view.default(permute_776, [1, 8192, 1024]);  permute_776 = None
        permute_777 = torch.ops.aten.permute.default(permute_775, [4, 2, 3, 0, 1]);  permute_775 = None
        view_697 = torch.ops.aten.view.default(permute_777, [1, 1024, 1024]);  permute_777 = None
        bmm_147 = torch.ops.aten.bmm.default(view_696, view_697);  view_696 = view_697 = None
        view_698 = torch.ops.aten.view.default(bmm_147, [1024, 8, 1, 16, 64]);  bmm_147 = None
        permute_778 = torch.ops.aten.permute.default(view_698, [0, 1, 3, 4, 2]);  view_698 = None
        view_699 = torch.ops.aten.view.default(permute_778, [1024, 8, 16, 64]);  permute_778 = None
        add_198 = torch.ops.aten.add.Tensor(view_687, arg278_1);  arg278_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(add_198, 4);  add_198 = None
        permute_779 = torch.ops.aten.permute.default(unsqueeze_469, [1, 2, 0, 4, 3]);  unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(view_691, 4);  view_691 = None
        permute_780 = torch.ops.aten.permute.default(unsqueeze_470, [1, 2, 4, 0, 3]);  unsqueeze_470 = None
        permute_781 = torch.ops.aten.permute.default(permute_779, [0, 1, 2, 4, 3]);  permute_779 = None
        view_700 = torch.ops.aten.view.default(permute_781, [128, 512, 64]);  permute_781 = None
        permute_782 = torch.ops.aten.permute.default(permute_780, [0, 1, 4, 3, 2]);  permute_780 = None
        view_701 = torch.ops.aten.view.default(permute_782, [128, 64, 512]);  permute_782 = None
        bmm_148 = torch.ops.aten.bmm.default(view_700, view_701);  view_700 = view_701 = None
        view_702 = torch.ops.aten.view.default(bmm_148, [8, 16, 512, 1, 512]);  bmm_148 = None
        permute_783 = torch.ops.aten.permute.default(view_702, [0, 1, 2, 4, 3]);  view_702 = None
        view_703 = torch.ops.aten.view.default(permute_783, [8, 16, 512, 512]);  permute_783 = None
        add_199 = torch.ops.aten.add.Tensor(view_687, arg277_1);  view_687 = arg277_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(add_199, 4);  add_199 = None
        permute_784 = torch.ops.aten.permute.default(unsqueeze_471, [1, 2, 0, 4, 3]);  unsqueeze_471 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(view_699, 4);  view_699 = None
        permute_785 = torch.ops.aten.permute.default(unsqueeze_472, [1, 2, 4, 0, 3]);  unsqueeze_472 = None
        permute_786 = torch.ops.aten.permute.default(permute_784, [0, 1, 2, 4, 3]);  permute_784 = None
        view_704 = torch.ops.aten.view.default(permute_786, [128, 512, 64]);  permute_786 = None
        permute_787 = torch.ops.aten.permute.default(permute_785, [0, 1, 4, 3, 2]);  permute_785 = None
        view_705 = torch.ops.aten.view.default(permute_787, [128, 64, 1024]);  permute_787 = None
        bmm_149 = torch.ops.aten.bmm.default(view_704, view_705);  view_704 = view_705 = None
        view_706 = torch.ops.aten.view.default(bmm_149, [8, 16, 512, 1, 1024]);  bmm_149 = None
        permute_788 = torch.ops.aten.permute.default(view_706, [0, 1, 2, 4, 3]);  view_706 = None
        view_707 = torch.ops.aten.view.default(permute_788, [8, 16, 512, 1024]);  permute_788 = None
        view_708 = torch.ops.aten.view.default(view_707, [8, 16, 1024, 512]);  view_707 = None
        slice_150 = torch.ops.aten.slice.Tensor(view_708, 2, 1, 9223372036854775807);  view_708 = None
        view_709 = torch.ops.aten.view.default(slice_150, [8, 16, 512, 1023]);  slice_150 = None
        iota_20 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_18 = torch.ops.aten.index.Tensor(view_709, [None, None, None, iota_20]);  view_709 = iota_20 = None
        add_200 = torch.ops.aten.add.Tensor(view_703, index_18);  view_703 = index_18 = None
        add_201 = torch.ops.aten.add.Tensor(add_200, 0);  add_200 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(add_201, 1);  add_201 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_10, [3], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.125);  sub_tensor_5 = None
        exp_18 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
        div_19 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(div_19, 4);  div_19 = None
        permute_789 = torch.ops.aten.permute.default(unsqueeze_473, [2, 0, 1, 4, 3]);  unsqueeze_473 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(view_695, 4);  view_695 = None
        permute_790 = torch.ops.aten.permute.default(unsqueeze_474, [4, 1, 2, 3, 0]);  unsqueeze_474 = None
        permute_791 = torch.ops.aten.permute.default(permute_789, [1, 2, 0, 4, 3]);  permute_789 = None
        view_710 = torch.ops.aten.view.default(permute_791, [128, 512, 512]);  permute_791 = None
        permute_792 = torch.ops.aten.permute.default(permute_790, [1, 2, 4, 3, 0]);  permute_790 = None
        view_711 = torch.ops.aten.view.default(permute_792, [128, 512, 64]);  permute_792 = None
        bmm_150 = torch.ops.aten.bmm.default(view_710, view_711);  view_710 = view_711 = None
        view_712 = torch.ops.aten.view.default(bmm_150, [8, 16, 512, 1, 64]);  bmm_150 = None
        permute_793 = torch.ops.aten.permute.default(view_712, [2, 0, 1, 4, 3]);  view_712 = None
        view_713 = torch.ops.aten.view.default(permute_793, [512, 8, 16, 64]);  permute_793 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
        permute_794 = torch.ops.aten.permute.default(unsqueeze_475, [0, 1, 4, 3, 2]);  unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg275_1, 3);  arg275_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, 4);  unsqueeze_476 = None
        permute_795 = torch.ops.aten.permute.default(unsqueeze_477, [3, 4, 0, 2, 1]);  unsqueeze_477 = None
        permute_796 = torch.ops.aten.permute.default(permute_794, [0, 1, 3, 4, 2]);  permute_794 = None
        clone_112 = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
        view_714 = torch.ops.aten.view.default(clone_112, [1, 4096, 1024]);  clone_112 = None
        permute_797 = torch.ops.aten.permute.default(permute_795, [3, 4, 2, 0, 1]);  permute_795 = None
        clone_113 = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_715 = torch.ops.aten.view.default(clone_113, [1, 1024, 1024]);  clone_113 = None
        bmm_151 = torch.ops.aten.bmm.default(view_714, view_715);  view_714 = view_715 = None
        view_716 = torch.ops.aten.view.default(bmm_151, [512, 8, 1, 1, 1024]);  bmm_151 = None
        permute_798 = torch.ops.aten.permute.default(view_716, [0, 1, 4, 2, 3]);  view_716 = None
        view_717 = torch.ops.aten.view.default(permute_798, [512, 8, 1024]);  permute_798 = None
        add_202 = torch.ops.aten.add.Tensor(view_717, add_197);  view_717 = add_197 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_203 = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_55 = torch.ops.aten.sub.Tensor(add_202, getitem_73);  add_202 = getitem_73 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_147, arg279_1);  mul_147 = arg279_1 = None
        add_204 = torch.ops.aten.add.Tensor(mul_148, arg280_1);  mul_148 = arg280_1 = None
        view_718 = torch.ops.aten.view.default(add_204, [4096, 1024])
        permute_799 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg284_1, view_718, permute_799);  arg284_1 = view_718 = permute_799 = None
        view_719 = torch.ops.aten.view.default(addmm_36, [512, 8, 4096]);  addmm_36 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_719, 0.5)
        mul_150 = torch.ops.aten.mul.Tensor(view_719, 0.7071067811865476);  view_719 = None
        erf_18 = torch.ops.aten.erf.default(mul_150);  mul_150 = None
        add_205 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_149, add_205);  mul_149 = add_205 = None
        view_720 = torch.ops.aten.view.default(mul_151, [4096, 4096]);  mul_151 = None
        permute_800 = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg286_1, view_720, permute_800);  arg286_1 = view_720 = permute_800 = None
        view_721 = torch.ops.aten.view.default(addmm_37, [512, 8, 1024]);  addmm_37 = None
        add_206 = torch.ops.aten.add.Tensor(view_721, add_204);  view_721 = add_204 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_206, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_207 = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_206, getitem_75);  add_206 = getitem_75 = None
        mul_152 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, arg281_1);  mul_152 = arg281_1 = None
        add_208 = torch.ops.aten.add.Tensor(mul_153, arg282_1);  mul_153 = arg282_1 = None
        slice_155 = torch.ops.aten.slice.Tensor(add_208, 0, -512, 9223372036854775807)
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(add_208, 3)
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, 4);  unsqueeze_478 = None
        permute_801 = torch.ops.aten.permute.default(unsqueeze_479, [0, 1, 3, 4, 2]);  unsqueeze_479 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg287_1, 3);  arg287_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, 4);  unsqueeze_480 = None
        permute_802 = torch.ops.aten.permute.default(unsqueeze_481, [3, 4, 1, 2, 0]);  unsqueeze_481 = None
        permute_803 = torch.ops.aten.permute.default(permute_801, [0, 1, 4, 2, 3]);  permute_801 = None
        view_722 = torch.ops.aten.view.default(permute_803, [1, 4096, 1024]);  permute_803 = None
        permute_804 = torch.ops.aten.permute.default(permute_802, [4, 2, 3, 0, 1]);  permute_802 = None
        view_723 = torch.ops.aten.view.default(permute_804, [1, 1024, 1024]);  permute_804 = None
        bmm_152 = torch.ops.aten.bmm.default(view_722, view_723);  view_722 = view_723 = None
        view_724 = torch.ops.aten.view.default(bmm_152, [512, 8, 1, 16, 64]);  bmm_152 = None
        permute_805 = torch.ops.aten.permute.default(view_724, [0, 1, 3, 4, 2]);  view_724 = None
        view_725 = torch.ops.aten.view.default(permute_805, [512, 8, 16, 64]);  permute_805 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(add_208, 3)
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, 4);  unsqueeze_482 = None
        permute_806 = torch.ops.aten.permute.default(unsqueeze_483, [0, 1, 3, 4, 2]);  unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg288_1, 3);  arg288_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, 4);  unsqueeze_484 = None
        permute_807 = torch.ops.aten.permute.default(unsqueeze_485, [3, 4, 1, 2, 0]);  unsqueeze_485 = None
        permute_808 = torch.ops.aten.permute.default(permute_806, [0, 1, 4, 2, 3]);  permute_806 = None
        view_726 = torch.ops.aten.view.default(permute_808, [1, 4096, 1024]);  permute_808 = None
        permute_809 = torch.ops.aten.permute.default(permute_807, [4, 2, 3, 0, 1]);  permute_807 = None
        view_727 = torch.ops.aten.view.default(permute_809, [1, 1024, 1024]);  permute_809 = None
        bmm_153 = torch.ops.aten.bmm.default(view_726, view_727);  view_726 = view_727 = None
        view_728 = torch.ops.aten.view.default(bmm_153, [512, 8, 1, 16, 64]);  bmm_153 = None
        permute_810 = torch.ops.aten.permute.default(view_728, [0, 1, 3, 4, 2]);  view_728 = None
        view_729 = torch.ops.aten.view.default(permute_810, [512, 8, 16, 64]);  permute_810 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(add_208, 3)
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, 4);  unsqueeze_486 = None
        permute_811 = torch.ops.aten.permute.default(unsqueeze_487, [0, 1, 3, 4, 2]);  unsqueeze_487 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg289_1, 3);  arg289_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, 4);  unsqueeze_488 = None
        permute_812 = torch.ops.aten.permute.default(unsqueeze_489, [3, 4, 1, 2, 0]);  unsqueeze_489 = None
        permute_813 = torch.ops.aten.permute.default(permute_811, [0, 1, 4, 2, 3]);  permute_811 = None
        view_730 = torch.ops.aten.view.default(permute_813, [1, 4096, 1024]);  permute_813 = None
        permute_814 = torch.ops.aten.permute.default(permute_812, [4, 2, 3, 0, 1]);  permute_812 = None
        view_731 = torch.ops.aten.view.default(permute_814, [1, 1024, 1024]);  permute_814 = None
        bmm_154 = torch.ops.aten.bmm.default(view_730, view_731);  view_730 = view_731 = None
        view_732 = torch.ops.aten.view.default(bmm_154, [512, 8, 1, 16, 64]);  bmm_154 = None
        permute_815 = torch.ops.aten.permute.default(view_732, [0, 1, 3, 4, 2]);  view_732 = None
        view_733 = torch.ops.aten.view.default(permute_815, [512, 8, 16, 64]);  permute_815 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, 4);  unsqueeze_490 = None
        permute_816 = torch.ops.aten.permute.default(unsqueeze_491, [0, 1, 3, 4, 2]);  unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg291_1, 3);  arg291_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, 4);  unsqueeze_492 = None
        permute_817 = torch.ops.aten.permute.default(unsqueeze_493, [3, 4, 1, 2, 0]);  unsqueeze_493 = None
        permute_818 = torch.ops.aten.permute.default(permute_816, [0, 1, 4, 2, 3]);  permute_816 = None
        view_734 = torch.ops.aten.view.default(permute_818, [1, 8192, 1024]);  permute_818 = None
        permute_819 = torch.ops.aten.permute.default(permute_817, [4, 2, 3, 0, 1]);  permute_817 = None
        view_735 = torch.ops.aten.view.default(permute_819, [1, 1024, 1024]);  permute_819 = None
        bmm_155 = torch.ops.aten.bmm.default(view_734, view_735);  view_734 = view_735 = None
        view_736 = torch.ops.aten.view.default(bmm_155, [1024, 8, 1, 16, 64]);  bmm_155 = None
        permute_820 = torch.ops.aten.permute.default(view_736, [0, 1, 3, 4, 2]);  view_736 = None
        view_737 = torch.ops.aten.view.default(permute_820, [1024, 8, 16, 64]);  permute_820 = None
        add_209 = torch.ops.aten.add.Tensor(view_725, arg293_1);  arg293_1 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(add_209, 4);  add_209 = None
        permute_821 = torch.ops.aten.permute.default(unsqueeze_494, [1, 2, 0, 4, 3]);  unsqueeze_494 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(view_729, 4);  view_729 = None
        permute_822 = torch.ops.aten.permute.default(unsqueeze_495, [1, 2, 4, 0, 3]);  unsqueeze_495 = None
        permute_823 = torch.ops.aten.permute.default(permute_821, [0, 1, 2, 4, 3]);  permute_821 = None
        view_738 = torch.ops.aten.view.default(permute_823, [128, 512, 64]);  permute_823 = None
        permute_824 = torch.ops.aten.permute.default(permute_822, [0, 1, 4, 3, 2]);  permute_822 = None
        view_739 = torch.ops.aten.view.default(permute_824, [128, 64, 512]);  permute_824 = None
        bmm_156 = torch.ops.aten.bmm.default(view_738, view_739);  view_738 = view_739 = None
        view_740 = torch.ops.aten.view.default(bmm_156, [8, 16, 512, 1, 512]);  bmm_156 = None
        permute_825 = torch.ops.aten.permute.default(view_740, [0, 1, 2, 4, 3]);  view_740 = None
        view_741 = torch.ops.aten.view.default(permute_825, [8, 16, 512, 512]);  permute_825 = None
        add_210 = torch.ops.aten.add.Tensor(view_725, arg292_1);  view_725 = arg292_1 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(add_210, 4);  add_210 = None
        permute_826 = torch.ops.aten.permute.default(unsqueeze_496, [1, 2, 0, 4, 3]);  unsqueeze_496 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(view_737, 4);  view_737 = None
        permute_827 = torch.ops.aten.permute.default(unsqueeze_497, [1, 2, 4, 0, 3]);  unsqueeze_497 = None
        permute_828 = torch.ops.aten.permute.default(permute_826, [0, 1, 2, 4, 3]);  permute_826 = None
        view_742 = torch.ops.aten.view.default(permute_828, [128, 512, 64]);  permute_828 = None
        permute_829 = torch.ops.aten.permute.default(permute_827, [0, 1, 4, 3, 2]);  permute_827 = None
        view_743 = torch.ops.aten.view.default(permute_829, [128, 64, 1024]);  permute_829 = None
        bmm_157 = torch.ops.aten.bmm.default(view_742, view_743);  view_742 = view_743 = None
        view_744 = torch.ops.aten.view.default(bmm_157, [8, 16, 512, 1, 1024]);  bmm_157 = None
        permute_830 = torch.ops.aten.permute.default(view_744, [0, 1, 2, 4, 3]);  view_744 = None
        view_745 = torch.ops.aten.view.default(permute_830, [8, 16, 512, 1024]);  permute_830 = None
        view_746 = torch.ops.aten.view.default(view_745, [8, 16, 1024, 512]);  view_745 = None
        slice_158 = torch.ops.aten.slice.Tensor(view_746, 2, 1, 9223372036854775807);  view_746 = None
        view_747 = torch.ops.aten.view.default(slice_158, [8, 16, 512, 1023]);  slice_158 = None
        iota_21 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_19 = torch.ops.aten.index.Tensor(view_747, [None, None, None, iota_21]);  view_747 = iota_21 = None
        add_211 = torch.ops.aten.add.Tensor(view_741, index_19);  view_741 = index_19 = None
        add_212 = torch.ops.aten.add.Tensor(add_211, 0);  add_211 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_212, 1);  add_212 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_8, [3], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        exp_19 = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
        div_20 = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(div_20, 4);  div_20 = None
        permute_831 = torch.ops.aten.permute.default(unsqueeze_498, [2, 0, 1, 4, 3]);  unsqueeze_498 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(view_733, 4);  view_733 = None
        permute_832 = torch.ops.aten.permute.default(unsqueeze_499, [4, 1, 2, 3, 0]);  unsqueeze_499 = None
        permute_833 = torch.ops.aten.permute.default(permute_831, [1, 2, 0, 4, 3]);  permute_831 = None
        view_748 = torch.ops.aten.view.default(permute_833, [128, 512, 512]);  permute_833 = None
        permute_834 = torch.ops.aten.permute.default(permute_832, [1, 2, 4, 3, 0]);  permute_832 = None
        view_749 = torch.ops.aten.view.default(permute_834, [128, 512, 64]);  permute_834 = None
        bmm_158 = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
        view_750 = torch.ops.aten.view.default(bmm_158, [8, 16, 512, 1, 64]);  bmm_158 = None
        permute_835 = torch.ops.aten.permute.default(view_750, [2, 0, 1, 4, 3]);  view_750 = None
        view_751 = torch.ops.aten.view.default(permute_835, [512, 8, 16, 64]);  permute_835 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(view_751, 4);  view_751 = None
        permute_836 = torch.ops.aten.permute.default(unsqueeze_500, [0, 1, 4, 3, 2]);  unsqueeze_500 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(arg290_1, 3);  arg290_1 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(unsqueeze_501, 4);  unsqueeze_501 = None
        permute_837 = torch.ops.aten.permute.default(unsqueeze_502, [3, 4, 0, 2, 1]);  unsqueeze_502 = None
        permute_838 = torch.ops.aten.permute.default(permute_836, [0, 1, 3, 4, 2]);  permute_836 = None
        clone_118 = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_752 = torch.ops.aten.view.default(clone_118, [1, 4096, 1024]);  clone_118 = None
        permute_839 = torch.ops.aten.permute.default(permute_837, [3, 4, 2, 0, 1]);  permute_837 = None
        clone_119 = torch.ops.aten.clone.default(permute_839, memory_format = torch.contiguous_format);  permute_839 = None
        view_753 = torch.ops.aten.view.default(clone_119, [1, 1024, 1024]);  clone_119 = None
        bmm_159 = torch.ops.aten.bmm.default(view_752, view_753);  view_752 = view_753 = None
        view_754 = torch.ops.aten.view.default(bmm_159, [512, 8, 1, 1, 1024]);  bmm_159 = None
        permute_840 = torch.ops.aten.permute.default(view_754, [0, 1, 4, 2, 3]);  view_754 = None
        view_755 = torch.ops.aten.view.default(permute_840, [512, 8, 1024]);  permute_840 = None
        add_213 = torch.ops.aten.add.Tensor(view_755, add_208);  view_755 = add_208 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_214 = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_58 = torch.ops.aten.sub.Tensor(add_213, getitem_77);  add_213 = getitem_77 = None
        mul_155 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = rsqrt_38 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, arg294_1);  mul_155 = arg294_1 = None
        add_215 = torch.ops.aten.add.Tensor(mul_156, arg295_1);  mul_156 = arg295_1 = None
        view_756 = torch.ops.aten.view.default(add_215, [4096, 1024])
        permute_841 = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg299_1, view_756, permute_841);  arg299_1 = view_756 = permute_841 = None
        view_757 = torch.ops.aten.view.default(addmm_38, [512, 8, 4096]);  addmm_38 = None
        mul_157 = torch.ops.aten.mul.Tensor(view_757, 0.5)
        mul_158 = torch.ops.aten.mul.Tensor(view_757, 0.7071067811865476);  view_757 = None
        erf_19 = torch.ops.aten.erf.default(mul_158);  mul_158 = None
        add_216 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_157, add_216);  mul_157 = add_216 = None
        view_758 = torch.ops.aten.view.default(mul_159, [4096, 4096]);  mul_159 = None
        permute_842 = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg301_1, view_758, permute_842);  arg301_1 = view_758 = permute_842 = None
        view_759 = torch.ops.aten.view.default(addmm_39, [512, 8, 1024]);  addmm_39 = None
        add_217 = torch.ops.aten.add.Tensor(view_759, add_215);  view_759 = add_215 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_217, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_217, getitem_79);  add_217 = getitem_79 = None
        mul_160 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, arg296_1);  mul_160 = arg296_1 = None
        add_219 = torch.ops.aten.add.Tensor(mul_161, arg297_1);  mul_161 = arg297_1 = None
        slice_163 = torch.ops.aten.slice.Tensor(add_219, 0, -512, 9223372036854775807)
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(add_219, 3)
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(unsqueeze_503, 4);  unsqueeze_503 = None
        permute_843 = torch.ops.aten.permute.default(unsqueeze_504, [0, 1, 3, 4, 2]);  unsqueeze_504 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(arg302_1, 3);  arg302_1 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(unsqueeze_505, 4);  unsqueeze_505 = None
        permute_844 = torch.ops.aten.permute.default(unsqueeze_506, [3, 4, 1, 2, 0]);  unsqueeze_506 = None
        permute_845 = torch.ops.aten.permute.default(permute_843, [0, 1, 4, 2, 3]);  permute_843 = None
        view_760 = torch.ops.aten.view.default(permute_845, [1, 4096, 1024]);  permute_845 = None
        permute_846 = torch.ops.aten.permute.default(permute_844, [4, 2, 3, 0, 1]);  permute_844 = None
        view_761 = torch.ops.aten.view.default(permute_846, [1, 1024, 1024]);  permute_846 = None
        bmm_160 = torch.ops.aten.bmm.default(view_760, view_761);  view_760 = view_761 = None
        view_762 = torch.ops.aten.view.default(bmm_160, [512, 8, 1, 16, 64]);  bmm_160 = None
        permute_847 = torch.ops.aten.permute.default(view_762, [0, 1, 3, 4, 2]);  view_762 = None
        view_763 = torch.ops.aten.view.default(permute_847, [512, 8, 16, 64]);  permute_847 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(add_219, 3)
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(unsqueeze_507, 4);  unsqueeze_507 = None
        permute_848 = torch.ops.aten.permute.default(unsqueeze_508, [0, 1, 3, 4, 2]);  unsqueeze_508 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(arg303_1, 3);  arg303_1 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(unsqueeze_509, 4);  unsqueeze_509 = None
        permute_849 = torch.ops.aten.permute.default(unsqueeze_510, [3, 4, 1, 2, 0]);  unsqueeze_510 = None
        permute_850 = torch.ops.aten.permute.default(permute_848, [0, 1, 4, 2, 3]);  permute_848 = None
        view_764 = torch.ops.aten.view.default(permute_850, [1, 4096, 1024]);  permute_850 = None
        permute_851 = torch.ops.aten.permute.default(permute_849, [4, 2, 3, 0, 1]);  permute_849 = None
        view_765 = torch.ops.aten.view.default(permute_851, [1, 1024, 1024]);  permute_851 = None
        bmm_161 = torch.ops.aten.bmm.default(view_764, view_765);  view_764 = view_765 = None
        view_766 = torch.ops.aten.view.default(bmm_161, [512, 8, 1, 16, 64]);  bmm_161 = None
        permute_852 = torch.ops.aten.permute.default(view_766, [0, 1, 3, 4, 2]);  view_766 = None
        view_767 = torch.ops.aten.view.default(permute_852, [512, 8, 16, 64]);  permute_852 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(add_219, 3)
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(unsqueeze_511, 4);  unsqueeze_511 = None
        permute_853 = torch.ops.aten.permute.default(unsqueeze_512, [0, 1, 3, 4, 2]);  unsqueeze_512 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(arg304_1, 3);  arg304_1 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(unsqueeze_513, 4);  unsqueeze_513 = None
        permute_854 = torch.ops.aten.permute.default(unsqueeze_514, [3, 4, 1, 2, 0]);  unsqueeze_514 = None
        permute_855 = torch.ops.aten.permute.default(permute_853, [0, 1, 4, 2, 3]);  permute_853 = None
        view_768 = torch.ops.aten.view.default(permute_855, [1, 4096, 1024]);  permute_855 = None
        permute_856 = torch.ops.aten.permute.default(permute_854, [4, 2, 3, 0, 1]);  permute_854 = None
        view_769 = torch.ops.aten.view.default(permute_856, [1, 1024, 1024]);  permute_856 = None
        bmm_162 = torch.ops.aten.bmm.default(view_768, view_769);  view_768 = view_769 = None
        view_770 = torch.ops.aten.view.default(bmm_162, [512, 8, 1, 16, 64]);  bmm_162 = None
        permute_857 = torch.ops.aten.permute.default(view_770, [0, 1, 3, 4, 2]);  view_770 = None
        view_771 = torch.ops.aten.view.default(permute_857, [512, 8, 16, 64]);  permute_857 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(unsqueeze_515, 4);  unsqueeze_515 = None
        permute_858 = torch.ops.aten.permute.default(unsqueeze_516, [0, 1, 3, 4, 2]);  unsqueeze_516 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(arg306_1, 3);  arg306_1 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(unsqueeze_517, 4);  unsqueeze_517 = None
        permute_859 = torch.ops.aten.permute.default(unsqueeze_518, [3, 4, 1, 2, 0]);  unsqueeze_518 = None
        permute_860 = torch.ops.aten.permute.default(permute_858, [0, 1, 4, 2, 3]);  permute_858 = None
        view_772 = torch.ops.aten.view.default(permute_860, [1, 8192, 1024]);  permute_860 = None
        permute_861 = torch.ops.aten.permute.default(permute_859, [4, 2, 3, 0, 1]);  permute_859 = None
        view_773 = torch.ops.aten.view.default(permute_861, [1, 1024, 1024]);  permute_861 = None
        bmm_163 = torch.ops.aten.bmm.default(view_772, view_773);  view_772 = view_773 = None
        view_774 = torch.ops.aten.view.default(bmm_163, [1024, 8, 1, 16, 64]);  bmm_163 = None
        permute_862 = torch.ops.aten.permute.default(view_774, [0, 1, 3, 4, 2]);  view_774 = None
        view_775 = torch.ops.aten.view.default(permute_862, [1024, 8, 16, 64]);  permute_862 = None
        add_220 = torch.ops.aten.add.Tensor(view_763, arg308_1);  arg308_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(add_220, 4);  add_220 = None
        permute_863 = torch.ops.aten.permute.default(unsqueeze_519, [1, 2, 0, 4, 3]);  unsqueeze_519 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(view_767, 4);  view_767 = None
        permute_864 = torch.ops.aten.permute.default(unsqueeze_520, [1, 2, 4, 0, 3]);  unsqueeze_520 = None
        permute_865 = torch.ops.aten.permute.default(permute_863, [0, 1, 2, 4, 3]);  permute_863 = None
        view_776 = torch.ops.aten.view.default(permute_865, [128, 512, 64]);  permute_865 = None
        permute_866 = torch.ops.aten.permute.default(permute_864, [0, 1, 4, 3, 2]);  permute_864 = None
        view_777 = torch.ops.aten.view.default(permute_866, [128, 64, 512]);  permute_866 = None
        bmm_164 = torch.ops.aten.bmm.default(view_776, view_777);  view_776 = view_777 = None
        view_778 = torch.ops.aten.view.default(bmm_164, [8, 16, 512, 1, 512]);  bmm_164 = None
        permute_867 = torch.ops.aten.permute.default(view_778, [0, 1, 2, 4, 3]);  view_778 = None
        view_779 = torch.ops.aten.view.default(permute_867, [8, 16, 512, 512]);  permute_867 = None
        add_221 = torch.ops.aten.add.Tensor(view_763, arg307_1);  view_763 = arg307_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(add_221, 4);  add_221 = None
        permute_868 = torch.ops.aten.permute.default(unsqueeze_521, [1, 2, 0, 4, 3]);  unsqueeze_521 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(view_775, 4);  view_775 = None
        permute_869 = torch.ops.aten.permute.default(unsqueeze_522, [1, 2, 4, 0, 3]);  unsqueeze_522 = None
        permute_870 = torch.ops.aten.permute.default(permute_868, [0, 1, 2, 4, 3]);  permute_868 = None
        view_780 = torch.ops.aten.view.default(permute_870, [128, 512, 64]);  permute_870 = None
        permute_871 = torch.ops.aten.permute.default(permute_869, [0, 1, 4, 3, 2]);  permute_869 = None
        view_781 = torch.ops.aten.view.default(permute_871, [128, 64, 1024]);  permute_871 = None
        bmm_165 = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782 = torch.ops.aten.view.default(bmm_165, [8, 16, 512, 1, 1024]);  bmm_165 = None
        permute_872 = torch.ops.aten.permute.default(view_782, [0, 1, 2, 4, 3]);  view_782 = None
        view_783 = torch.ops.aten.view.default(permute_872, [8, 16, 512, 1024]);  permute_872 = None
        view_784 = torch.ops.aten.view.default(view_783, [8, 16, 1024, 512]);  view_783 = None
        slice_166 = torch.ops.aten.slice.Tensor(view_784, 2, 1, 9223372036854775807);  view_784 = None
        view_785 = torch.ops.aten.view.default(slice_166, [8, 16, 512, 1023]);  slice_166 = None
        iota_22 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_20 = torch.ops.aten.index.Tensor(view_785, [None, None, None, iota_22]);  view_785 = iota_22 = None
        add_222 = torch.ops.aten.add.Tensor(view_779, index_20);  view_779 = index_20 = None
        add_223 = torch.ops.aten.add.Tensor(add_222, 0);  add_222 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_223, 1);  add_223 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_6, [3], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.125);  sub_tensor_3 = None
        exp_20 = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
        div_21 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(div_21, 4);  div_21 = None
        permute_873 = torch.ops.aten.permute.default(unsqueeze_523, [2, 0, 1, 4, 3]);  unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(view_771, 4);  view_771 = None
        permute_874 = torch.ops.aten.permute.default(unsqueeze_524, [4, 1, 2, 3, 0]);  unsqueeze_524 = None
        permute_875 = torch.ops.aten.permute.default(permute_873, [1, 2, 0, 4, 3]);  permute_873 = None
        view_786 = torch.ops.aten.view.default(permute_875, [128, 512, 512]);  permute_875 = None
        permute_876 = torch.ops.aten.permute.default(permute_874, [1, 2, 4, 3, 0]);  permute_874 = None
        view_787 = torch.ops.aten.view.default(permute_876, [128, 512, 64]);  permute_876 = None
        bmm_166 = torch.ops.aten.bmm.default(view_786, view_787);  view_786 = view_787 = None
        view_788 = torch.ops.aten.view.default(bmm_166, [8, 16, 512, 1, 64]);  bmm_166 = None
        permute_877 = torch.ops.aten.permute.default(view_788, [2, 0, 1, 4, 3]);  view_788 = None
        view_789 = torch.ops.aten.view.default(permute_877, [512, 8, 16, 64]);  permute_877 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(view_789, 4);  view_789 = None
        permute_878 = torch.ops.aten.permute.default(unsqueeze_525, [0, 1, 4, 3, 2]);  unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg305_1, 3);  arg305_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, 4);  unsqueeze_526 = None
        permute_879 = torch.ops.aten.permute.default(unsqueeze_527, [3, 4, 0, 2, 1]);  unsqueeze_527 = None
        permute_880 = torch.ops.aten.permute.default(permute_878, [0, 1, 3, 4, 2]);  permute_878 = None
        clone_124 = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_790 = torch.ops.aten.view.default(clone_124, [1, 4096, 1024]);  clone_124 = None
        permute_881 = torch.ops.aten.permute.default(permute_879, [3, 4, 2, 0, 1]);  permute_879 = None
        clone_125 = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
        view_791 = torch.ops.aten.view.default(clone_125, [1, 1024, 1024]);  clone_125 = None
        bmm_167 = torch.ops.aten.bmm.default(view_790, view_791);  view_790 = view_791 = None
        view_792 = torch.ops.aten.view.default(bmm_167, [512, 8, 1, 1, 1024]);  bmm_167 = None
        permute_882 = torch.ops.aten.permute.default(view_792, [0, 1, 4, 2, 3]);  view_792 = None
        view_793 = torch.ops.aten.view.default(permute_882, [512, 8, 1024]);  permute_882 = None
        add_224 = torch.ops.aten.add.Tensor(view_793, add_219);  view_793 = add_219 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_225 = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_61 = torch.ops.aten.sub.Tensor(add_224, getitem_81);  add_224 = getitem_81 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = rsqrt_40 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, arg309_1);  mul_163 = arg309_1 = None
        add_226 = torch.ops.aten.add.Tensor(mul_164, arg310_1);  mul_164 = arg310_1 = None
        view_794 = torch.ops.aten.view.default(add_226, [4096, 1024])
        permute_883 = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg314_1, view_794, permute_883);  arg314_1 = view_794 = permute_883 = None
        view_795 = torch.ops.aten.view.default(addmm_40, [512, 8, 4096]);  addmm_40 = None
        mul_165 = torch.ops.aten.mul.Tensor(view_795, 0.5)
        mul_166 = torch.ops.aten.mul.Tensor(view_795, 0.7071067811865476);  view_795 = None
        erf_20 = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_227 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_165, add_227);  mul_165 = add_227 = None
        view_796 = torch.ops.aten.view.default(mul_167, [4096, 4096]);  mul_167 = None
        permute_884 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg316_1, view_796, permute_884);  arg316_1 = view_796 = permute_884 = None
        view_797 = torch.ops.aten.view.default(addmm_41, [512, 8, 1024]);  addmm_41 = None
        add_228 = torch.ops.aten.add.Tensor(view_797, add_226);  view_797 = add_226 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_228, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_228, getitem_83);  add_228 = getitem_83 = None
        mul_168 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, arg311_1);  mul_168 = arg311_1 = None
        add_230 = torch.ops.aten.add.Tensor(mul_169, arg312_1);  mul_169 = arg312_1 = None
        slice_171 = torch.ops.aten.slice.Tensor(add_230, 0, -512, 9223372036854775807)
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(add_230, 3)
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, 4);  unsqueeze_528 = None
        permute_885 = torch.ops.aten.permute.default(unsqueeze_529, [0, 1, 3, 4, 2]);  unsqueeze_529 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(arg317_1, 3);  arg317_1 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, 4);  unsqueeze_530 = None
        permute_886 = torch.ops.aten.permute.default(unsqueeze_531, [3, 4, 1, 2, 0]);  unsqueeze_531 = None
        permute_887 = torch.ops.aten.permute.default(permute_885, [0, 1, 4, 2, 3]);  permute_885 = None
        view_798 = torch.ops.aten.view.default(permute_887, [1, 4096, 1024]);  permute_887 = None
        permute_888 = torch.ops.aten.permute.default(permute_886, [4, 2, 3, 0, 1]);  permute_886 = None
        view_799 = torch.ops.aten.view.default(permute_888, [1, 1024, 1024]);  permute_888 = None
        bmm_168 = torch.ops.aten.bmm.default(view_798, view_799);  view_798 = view_799 = None
        view_800 = torch.ops.aten.view.default(bmm_168, [512, 8, 1, 16, 64]);  bmm_168 = None
        permute_889 = torch.ops.aten.permute.default(view_800, [0, 1, 3, 4, 2]);  view_800 = None
        view_801 = torch.ops.aten.view.default(permute_889, [512, 8, 16, 64]);  permute_889 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(add_230, 3)
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, 4);  unsqueeze_532 = None
        permute_890 = torch.ops.aten.permute.default(unsqueeze_533, [0, 1, 3, 4, 2]);  unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg318_1, 3);  arg318_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, 4);  unsqueeze_534 = None
        permute_891 = torch.ops.aten.permute.default(unsqueeze_535, [3, 4, 1, 2, 0]);  unsqueeze_535 = None
        permute_892 = torch.ops.aten.permute.default(permute_890, [0, 1, 4, 2, 3]);  permute_890 = None
        view_802 = torch.ops.aten.view.default(permute_892, [1, 4096, 1024]);  permute_892 = None
        permute_893 = torch.ops.aten.permute.default(permute_891, [4, 2, 3, 0, 1]);  permute_891 = None
        view_803 = torch.ops.aten.view.default(permute_893, [1, 1024, 1024]);  permute_893 = None
        bmm_169 = torch.ops.aten.bmm.default(view_802, view_803);  view_802 = view_803 = None
        view_804 = torch.ops.aten.view.default(bmm_169, [512, 8, 1, 16, 64]);  bmm_169 = None
        permute_894 = torch.ops.aten.permute.default(view_804, [0, 1, 3, 4, 2]);  view_804 = None
        view_805 = torch.ops.aten.view.default(permute_894, [512, 8, 16, 64]);  permute_894 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(add_230, 3)
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, 4);  unsqueeze_536 = None
        permute_895 = torch.ops.aten.permute.default(unsqueeze_537, [0, 1, 3, 4, 2]);  unsqueeze_537 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(arg319_1, 3);  arg319_1 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, 4);  unsqueeze_538 = None
        permute_896 = torch.ops.aten.permute.default(unsqueeze_539, [3, 4, 1, 2, 0]);  unsqueeze_539 = None
        permute_897 = torch.ops.aten.permute.default(permute_895, [0, 1, 4, 2, 3]);  permute_895 = None
        view_806 = torch.ops.aten.view.default(permute_897, [1, 4096, 1024]);  permute_897 = None
        permute_898 = torch.ops.aten.permute.default(permute_896, [4, 2, 3, 0, 1]);  permute_896 = None
        view_807 = torch.ops.aten.view.default(permute_898, [1, 1024, 1024]);  permute_898 = None
        bmm_170 = torch.ops.aten.bmm.default(view_806, view_807);  view_806 = view_807 = None
        view_808 = torch.ops.aten.view.default(bmm_170, [512, 8, 1, 16, 64]);  bmm_170 = None
        permute_899 = torch.ops.aten.permute.default(view_808, [0, 1, 3, 4, 2]);  view_808 = None
        view_809 = torch.ops.aten.view.default(permute_899, [512, 8, 16, 64]);  permute_899 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, 4);  unsqueeze_540 = None
        permute_900 = torch.ops.aten.permute.default(unsqueeze_541, [0, 1, 3, 4, 2]);  unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg321_1, 3);  arg321_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, 4);  unsqueeze_542 = None
        permute_901 = torch.ops.aten.permute.default(unsqueeze_543, [3, 4, 1, 2, 0]);  unsqueeze_543 = None
        permute_902 = torch.ops.aten.permute.default(permute_900, [0, 1, 4, 2, 3]);  permute_900 = None
        view_810 = torch.ops.aten.view.default(permute_902, [1, 8192, 1024]);  permute_902 = None
        permute_903 = torch.ops.aten.permute.default(permute_901, [4, 2, 3, 0, 1]);  permute_901 = None
        view_811 = torch.ops.aten.view.default(permute_903, [1, 1024, 1024]);  permute_903 = None
        bmm_171 = torch.ops.aten.bmm.default(view_810, view_811);  view_810 = view_811 = None
        view_812 = torch.ops.aten.view.default(bmm_171, [1024, 8, 1, 16, 64]);  bmm_171 = None
        permute_904 = torch.ops.aten.permute.default(view_812, [0, 1, 3, 4, 2]);  view_812 = None
        view_813 = torch.ops.aten.view.default(permute_904, [1024, 8, 16, 64]);  permute_904 = None
        add_231 = torch.ops.aten.add.Tensor(view_801, arg323_1);  arg323_1 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(add_231, 4);  add_231 = None
        permute_905 = torch.ops.aten.permute.default(unsqueeze_544, [1, 2, 0, 4, 3]);  unsqueeze_544 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(view_805, 4);  view_805 = None
        permute_906 = torch.ops.aten.permute.default(unsqueeze_545, [1, 2, 4, 0, 3]);  unsqueeze_545 = None
        permute_907 = torch.ops.aten.permute.default(permute_905, [0, 1, 2, 4, 3]);  permute_905 = None
        view_814 = torch.ops.aten.view.default(permute_907, [128, 512, 64]);  permute_907 = None
        permute_908 = torch.ops.aten.permute.default(permute_906, [0, 1, 4, 3, 2]);  permute_906 = None
        view_815 = torch.ops.aten.view.default(permute_908, [128, 64, 512]);  permute_908 = None
        bmm_172 = torch.ops.aten.bmm.default(view_814, view_815);  view_814 = view_815 = None
        view_816 = torch.ops.aten.view.default(bmm_172, [8, 16, 512, 1, 512]);  bmm_172 = None
        permute_909 = torch.ops.aten.permute.default(view_816, [0, 1, 2, 4, 3]);  view_816 = None
        view_817 = torch.ops.aten.view.default(permute_909, [8, 16, 512, 512]);  permute_909 = None
        add_232 = torch.ops.aten.add.Tensor(view_801, arg322_1);  view_801 = arg322_1 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(add_232, 4);  add_232 = None
        permute_910 = torch.ops.aten.permute.default(unsqueeze_546, [1, 2, 0, 4, 3]);  unsqueeze_546 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(view_813, 4);  view_813 = None
        permute_911 = torch.ops.aten.permute.default(unsqueeze_547, [1, 2, 4, 0, 3]);  unsqueeze_547 = None
        permute_912 = torch.ops.aten.permute.default(permute_910, [0, 1, 2, 4, 3]);  permute_910 = None
        view_818 = torch.ops.aten.view.default(permute_912, [128, 512, 64]);  permute_912 = None
        permute_913 = torch.ops.aten.permute.default(permute_911, [0, 1, 4, 3, 2]);  permute_911 = None
        view_819 = torch.ops.aten.view.default(permute_913, [128, 64, 1024]);  permute_913 = None
        bmm_173 = torch.ops.aten.bmm.default(view_818, view_819);  view_818 = view_819 = None
        view_820 = torch.ops.aten.view.default(bmm_173, [8, 16, 512, 1, 1024]);  bmm_173 = None
        permute_914 = torch.ops.aten.permute.default(view_820, [0, 1, 2, 4, 3]);  view_820 = None
        view_821 = torch.ops.aten.view.default(permute_914, [8, 16, 512, 1024]);  permute_914 = None
        view_822 = torch.ops.aten.view.default(view_821, [8, 16, 1024, 512]);  view_821 = None
        slice_174 = torch.ops.aten.slice.Tensor(view_822, 2, 1, 9223372036854775807);  view_822 = None
        view_823 = torch.ops.aten.view.default(slice_174, [8, 16, 512, 1023]);  slice_174 = None
        iota_23 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_21 = torch.ops.aten.index.Tensor(view_823, [None, None, None, iota_23]);  view_823 = iota_23 = None
        add_233 = torch.ops.aten.add.Tensor(view_817, index_21);  view_817 = index_21 = None
        add_234 = torch.ops.aten.add.Tensor(add_233, 0);  add_233 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_234, 1);  add_234 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_4, [3], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.125);  sub_tensor_2 = None
        exp_21 = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
        div_22 = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(div_22, 4);  div_22 = None
        permute_915 = torch.ops.aten.permute.default(unsqueeze_548, [2, 0, 1, 4, 3]);  unsqueeze_548 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(view_809, 4);  view_809 = None
        permute_916 = torch.ops.aten.permute.default(unsqueeze_549, [4, 1, 2, 3, 0]);  unsqueeze_549 = None
        permute_917 = torch.ops.aten.permute.default(permute_915, [1, 2, 0, 4, 3]);  permute_915 = None
        view_824 = torch.ops.aten.view.default(permute_917, [128, 512, 512]);  permute_917 = None
        permute_918 = torch.ops.aten.permute.default(permute_916, [1, 2, 4, 3, 0]);  permute_916 = None
        view_825 = torch.ops.aten.view.default(permute_918, [128, 512, 64]);  permute_918 = None
        bmm_174 = torch.ops.aten.bmm.default(view_824, view_825);  view_824 = view_825 = None
        view_826 = torch.ops.aten.view.default(bmm_174, [8, 16, 512, 1, 64]);  bmm_174 = None
        permute_919 = torch.ops.aten.permute.default(view_826, [2, 0, 1, 4, 3]);  view_826 = None
        view_827 = torch.ops.aten.view.default(permute_919, [512, 8, 16, 64]);  permute_919 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(view_827, 4);  view_827 = None
        permute_920 = torch.ops.aten.permute.default(unsqueeze_550, [0, 1, 4, 3, 2]);  unsqueeze_550 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(arg320_1, 3);  arg320_1 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(unsqueeze_551, 4);  unsqueeze_551 = None
        permute_921 = torch.ops.aten.permute.default(unsqueeze_552, [3, 4, 0, 2, 1]);  unsqueeze_552 = None
        permute_922 = torch.ops.aten.permute.default(permute_920, [0, 1, 3, 4, 2]);  permute_920 = None
        clone_130 = torch.ops.aten.clone.default(permute_922, memory_format = torch.contiguous_format);  permute_922 = None
        view_828 = torch.ops.aten.view.default(clone_130, [1, 4096, 1024]);  clone_130 = None
        permute_923 = torch.ops.aten.permute.default(permute_921, [3, 4, 2, 0, 1]);  permute_921 = None
        clone_131 = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
        view_829 = torch.ops.aten.view.default(clone_131, [1, 1024, 1024]);  clone_131 = None
        bmm_175 = torch.ops.aten.bmm.default(view_828, view_829);  view_828 = view_829 = None
        view_830 = torch.ops.aten.view.default(bmm_175, [512, 8, 1, 1, 1024]);  bmm_175 = None
        permute_924 = torch.ops.aten.permute.default(view_830, [0, 1, 4, 2, 3]);  view_830 = None
        view_831 = torch.ops.aten.view.default(permute_924, [512, 8, 1024]);  permute_924 = None
        add_235 = torch.ops.aten.add.Tensor(view_831, add_230);  view_831 = add_230 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_235, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_64 = torch.ops.aten.sub.Tensor(add_235, getitem_85);  add_235 = getitem_85 = None
        mul_171 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = rsqrt_42 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, arg324_1);  mul_171 = arg324_1 = None
        add_237 = torch.ops.aten.add.Tensor(mul_172, arg325_1);  mul_172 = arg325_1 = None
        view_832 = torch.ops.aten.view.default(add_237, [4096, 1024])
        permute_925 = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg329_1, view_832, permute_925);  arg329_1 = view_832 = permute_925 = None
        view_833 = torch.ops.aten.view.default(addmm_42, [512, 8, 4096]);  addmm_42 = None
        mul_173 = torch.ops.aten.mul.Tensor(view_833, 0.5)
        mul_174 = torch.ops.aten.mul.Tensor(view_833, 0.7071067811865476);  view_833 = None
        erf_21 = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_238 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_175 = torch.ops.aten.mul.Tensor(mul_173, add_238);  mul_173 = add_238 = None
        view_834 = torch.ops.aten.view.default(mul_175, [4096, 4096]);  mul_175 = None
        permute_926 = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg331_1, view_834, permute_926);  arg331_1 = view_834 = permute_926 = None
        view_835 = torch.ops.aten.view.default(addmm_43, [512, 8, 1024]);  addmm_43 = None
        add_239 = torch.ops.aten.add.Tensor(view_835, add_237);  view_835 = add_237 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_239, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_65 = torch.ops.aten.sub.Tensor(add_239, getitem_87);  add_239 = getitem_87 = None
        mul_176 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, arg326_1);  mul_176 = arg326_1 = None
        add_241 = torch.ops.aten.add.Tensor(mul_177, arg327_1);  mul_177 = arg327_1 = None
        slice_179 = torch.ops.aten.slice.Tensor(add_241, 0, -512, 9223372036854775807)
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(add_241, 3)
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(unsqueeze_553, 4);  unsqueeze_553 = None
        permute_927 = torch.ops.aten.permute.default(unsqueeze_554, [0, 1, 3, 4, 2]);  unsqueeze_554 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(arg332_1, 3);  arg332_1 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(unsqueeze_555, 4);  unsqueeze_555 = None
        permute_928 = torch.ops.aten.permute.default(unsqueeze_556, [3, 4, 1, 2, 0]);  unsqueeze_556 = None
        permute_929 = torch.ops.aten.permute.default(permute_927, [0, 1, 4, 2, 3]);  permute_927 = None
        view_836 = torch.ops.aten.view.default(permute_929, [1, 4096, 1024]);  permute_929 = None
        permute_930 = torch.ops.aten.permute.default(permute_928, [4, 2, 3, 0, 1]);  permute_928 = None
        view_837 = torch.ops.aten.view.default(permute_930, [1, 1024, 1024]);  permute_930 = None
        bmm_176 = torch.ops.aten.bmm.default(view_836, view_837);  view_836 = view_837 = None
        view_838 = torch.ops.aten.view.default(bmm_176, [512, 8, 1, 16, 64]);  bmm_176 = None
        permute_931 = torch.ops.aten.permute.default(view_838, [0, 1, 3, 4, 2]);  view_838 = None
        view_839 = torch.ops.aten.view.default(permute_931, [512, 8, 16, 64]);  permute_931 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(add_241, 3)
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(unsqueeze_557, 4);  unsqueeze_557 = None
        permute_932 = torch.ops.aten.permute.default(unsqueeze_558, [0, 1, 3, 4, 2]);  unsqueeze_558 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(arg333_1, 3);  arg333_1 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(unsqueeze_559, 4);  unsqueeze_559 = None
        permute_933 = torch.ops.aten.permute.default(unsqueeze_560, [3, 4, 1, 2, 0]);  unsqueeze_560 = None
        permute_934 = torch.ops.aten.permute.default(permute_932, [0, 1, 4, 2, 3]);  permute_932 = None
        view_840 = torch.ops.aten.view.default(permute_934, [1, 4096, 1024]);  permute_934 = None
        permute_935 = torch.ops.aten.permute.default(permute_933, [4, 2, 3, 0, 1]);  permute_933 = None
        view_841 = torch.ops.aten.view.default(permute_935, [1, 1024, 1024]);  permute_935 = None
        bmm_177 = torch.ops.aten.bmm.default(view_840, view_841);  view_840 = view_841 = None
        view_842 = torch.ops.aten.view.default(bmm_177, [512, 8, 1, 16, 64]);  bmm_177 = None
        permute_936 = torch.ops.aten.permute.default(view_842, [0, 1, 3, 4, 2]);  view_842 = None
        view_843 = torch.ops.aten.view.default(permute_936, [512, 8, 16, 64]);  permute_936 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(add_241, 3)
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(unsqueeze_561, 4);  unsqueeze_561 = None
        permute_937 = torch.ops.aten.permute.default(unsqueeze_562, [0, 1, 3, 4, 2]);  unsqueeze_562 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(arg334_1, 3);  arg334_1 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(unsqueeze_563, 4);  unsqueeze_563 = None
        permute_938 = torch.ops.aten.permute.default(unsqueeze_564, [3, 4, 1, 2, 0]);  unsqueeze_564 = None
        permute_939 = torch.ops.aten.permute.default(permute_937, [0, 1, 4, 2, 3]);  permute_937 = None
        view_844 = torch.ops.aten.view.default(permute_939, [1, 4096, 1024]);  permute_939 = None
        permute_940 = torch.ops.aten.permute.default(permute_938, [4, 2, 3, 0, 1]);  permute_938 = None
        view_845 = torch.ops.aten.view.default(permute_940, [1, 1024, 1024]);  permute_940 = None
        bmm_178 = torch.ops.aten.bmm.default(view_844, view_845);  view_844 = view_845 = None
        view_846 = torch.ops.aten.view.default(bmm_178, [512, 8, 1, 16, 64]);  bmm_178 = None
        permute_941 = torch.ops.aten.permute.default(view_846, [0, 1, 3, 4, 2]);  view_846 = None
        view_847 = torch.ops.aten.view.default(permute_941, [512, 8, 16, 64]);  permute_941 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3)
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(unsqueeze_565, 4);  unsqueeze_565 = None
        permute_942 = torch.ops.aten.permute.default(unsqueeze_566, [0, 1, 3, 4, 2]);  unsqueeze_566 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(arg336_1, 3);  arg336_1 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(unsqueeze_567, 4);  unsqueeze_567 = None
        permute_943 = torch.ops.aten.permute.default(unsqueeze_568, [3, 4, 1, 2, 0]);  unsqueeze_568 = None
        permute_944 = torch.ops.aten.permute.default(permute_942, [0, 1, 4, 2, 3]);  permute_942 = None
        view_848 = torch.ops.aten.view.default(permute_944, [1, 8192, 1024]);  permute_944 = None
        permute_945 = torch.ops.aten.permute.default(permute_943, [4, 2, 3, 0, 1]);  permute_943 = None
        view_849 = torch.ops.aten.view.default(permute_945, [1, 1024, 1024]);  permute_945 = None
        bmm_179 = torch.ops.aten.bmm.default(view_848, view_849);  view_848 = view_849 = None
        view_850 = torch.ops.aten.view.default(bmm_179, [1024, 8, 1, 16, 64]);  bmm_179 = None
        permute_946 = torch.ops.aten.permute.default(view_850, [0, 1, 3, 4, 2]);  view_850 = None
        view_851 = torch.ops.aten.view.default(permute_946, [1024, 8, 16, 64]);  permute_946 = None
        add_242 = torch.ops.aten.add.Tensor(view_839, arg338_1);  arg338_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(add_242, 4);  add_242 = None
        permute_947 = torch.ops.aten.permute.default(unsqueeze_569, [1, 2, 0, 4, 3]);  unsqueeze_569 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(view_843, 4);  view_843 = None
        permute_948 = torch.ops.aten.permute.default(unsqueeze_570, [1, 2, 4, 0, 3]);  unsqueeze_570 = None
        permute_949 = torch.ops.aten.permute.default(permute_947, [0, 1, 2, 4, 3]);  permute_947 = None
        view_852 = torch.ops.aten.view.default(permute_949, [128, 512, 64]);  permute_949 = None
        permute_950 = torch.ops.aten.permute.default(permute_948, [0, 1, 4, 3, 2]);  permute_948 = None
        view_853 = torch.ops.aten.view.default(permute_950, [128, 64, 512]);  permute_950 = None
        bmm_180 = torch.ops.aten.bmm.default(view_852, view_853);  view_852 = view_853 = None
        view_854 = torch.ops.aten.view.default(bmm_180, [8, 16, 512, 1, 512]);  bmm_180 = None
        permute_951 = torch.ops.aten.permute.default(view_854, [0, 1, 2, 4, 3]);  view_854 = None
        view_855 = torch.ops.aten.view.default(permute_951, [8, 16, 512, 512]);  permute_951 = None
        add_243 = torch.ops.aten.add.Tensor(view_839, arg337_1);  view_839 = arg337_1 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(add_243, 4);  add_243 = None
        permute_952 = torch.ops.aten.permute.default(unsqueeze_571, [1, 2, 0, 4, 3]);  unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(view_851, 4);  view_851 = None
        permute_953 = torch.ops.aten.permute.default(unsqueeze_572, [1, 2, 4, 0, 3]);  unsqueeze_572 = None
        permute_954 = torch.ops.aten.permute.default(permute_952, [0, 1, 2, 4, 3]);  permute_952 = None
        view_856 = torch.ops.aten.view.default(permute_954, [128, 512, 64]);  permute_954 = None
        permute_955 = torch.ops.aten.permute.default(permute_953, [0, 1, 4, 3, 2]);  permute_953 = None
        view_857 = torch.ops.aten.view.default(permute_955, [128, 64, 1024]);  permute_955 = None
        bmm_181 = torch.ops.aten.bmm.default(view_856, view_857);  view_856 = view_857 = None
        view_858 = torch.ops.aten.view.default(bmm_181, [8, 16, 512, 1, 1024]);  bmm_181 = None
        permute_956 = torch.ops.aten.permute.default(view_858, [0, 1, 2, 4, 3]);  view_858 = None
        view_859 = torch.ops.aten.view.default(permute_956, [8, 16, 512, 1024]);  permute_956 = None
        view_860 = torch.ops.aten.view.default(view_859, [8, 16, 1024, 512]);  view_859 = None
        slice_182 = torch.ops.aten.slice.Tensor(view_860, 2, 1, 9223372036854775807);  view_860 = None
        view_861 = torch.ops.aten.view.default(slice_182, [8, 16, 512, 1023]);  slice_182 = None
        iota_24 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_22 = torch.ops.aten.index.Tensor(view_861, [None, None, None, iota_24]);  view_861 = iota_24 = None
        add_244 = torch.ops.aten.add.Tensor(view_855, index_22);  view_855 = index_22 = None
        add_245 = torch.ops.aten.add.Tensor(add_244, 0);  add_244 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_245, 1);  add_245 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_2, [3], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.125);  sub_tensor_1 = None
        exp_22 = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
        div_23 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(div_23, 4);  div_23 = None
        permute_957 = torch.ops.aten.permute.default(unsqueeze_573, [2, 0, 1, 4, 3]);  unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(view_847, 4);  view_847 = None
        permute_958 = torch.ops.aten.permute.default(unsqueeze_574, [4, 1, 2, 3, 0]);  unsqueeze_574 = None
        permute_959 = torch.ops.aten.permute.default(permute_957, [1, 2, 0, 4, 3]);  permute_957 = None
        view_862 = torch.ops.aten.view.default(permute_959, [128, 512, 512]);  permute_959 = None
        permute_960 = torch.ops.aten.permute.default(permute_958, [1, 2, 4, 3, 0]);  permute_958 = None
        view_863 = torch.ops.aten.view.default(permute_960, [128, 512, 64]);  permute_960 = None
        bmm_182 = torch.ops.aten.bmm.default(view_862, view_863);  view_862 = view_863 = None
        view_864 = torch.ops.aten.view.default(bmm_182, [8, 16, 512, 1, 64]);  bmm_182 = None
        permute_961 = torch.ops.aten.permute.default(view_864, [2, 0, 1, 4, 3]);  view_864 = None
        view_865 = torch.ops.aten.view.default(permute_961, [512, 8, 16, 64]);  permute_961 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(view_865, 4);  view_865 = None
        permute_962 = torch.ops.aten.permute.default(unsqueeze_575, [0, 1, 4, 3, 2]);  unsqueeze_575 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg335_1, 3);  arg335_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, 4);  unsqueeze_576 = None
        permute_963 = torch.ops.aten.permute.default(unsqueeze_577, [3, 4, 0, 2, 1]);  unsqueeze_577 = None
        permute_964 = torch.ops.aten.permute.default(permute_962, [0, 1, 3, 4, 2]);  permute_962 = None
        clone_136 = torch.ops.aten.clone.default(permute_964, memory_format = torch.contiguous_format);  permute_964 = None
        view_866 = torch.ops.aten.view.default(clone_136, [1, 4096, 1024]);  clone_136 = None
        permute_965 = torch.ops.aten.permute.default(permute_963, [3, 4, 2, 0, 1]);  permute_963 = None
        clone_137 = torch.ops.aten.clone.default(permute_965, memory_format = torch.contiguous_format);  permute_965 = None
        view_867 = torch.ops.aten.view.default(clone_137, [1, 1024, 1024]);  clone_137 = None
        bmm_183 = torch.ops.aten.bmm.default(view_866, view_867);  view_866 = view_867 = None
        view_868 = torch.ops.aten.view.default(bmm_183, [512, 8, 1, 1, 1024]);  bmm_183 = None
        permute_966 = torch.ops.aten.permute.default(view_868, [0, 1, 4, 2, 3]);  view_868 = None
        view_869 = torch.ops.aten.view.default(permute_966, [512, 8, 1024]);  permute_966 = None
        add_246 = torch.ops.aten.add.Tensor(view_869, add_241);  view_869 = add_241 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_246, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_247 = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_67 = torch.ops.aten.sub.Tensor(add_246, getitem_89);  add_246 = getitem_89 = None
        mul_179 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = rsqrt_44 = None
        mul_180 = torch.ops.aten.mul.Tensor(mul_179, arg339_1);  mul_179 = arg339_1 = None
        add_248 = torch.ops.aten.add.Tensor(mul_180, arg340_1);  mul_180 = arg340_1 = None
        view_870 = torch.ops.aten.view.default(add_248, [4096, 1024])
        permute_967 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg344_1, view_870, permute_967);  arg344_1 = view_870 = permute_967 = None
        view_871 = torch.ops.aten.view.default(addmm_44, [512, 8, 4096]);  addmm_44 = None
        mul_181 = torch.ops.aten.mul.Tensor(view_871, 0.5)
        mul_182 = torch.ops.aten.mul.Tensor(view_871, 0.7071067811865476);  view_871 = None
        erf_22 = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_249 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_181, add_249);  mul_181 = add_249 = None
        view_872 = torch.ops.aten.view.default(mul_183, [4096, 4096]);  mul_183 = None
        permute_968 = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg346_1, view_872, permute_968);  arg346_1 = view_872 = permute_968 = None
        view_873 = torch.ops.aten.view.default(addmm_45, [512, 8, 1024]);  addmm_45 = None
        add_250 = torch.ops.aten.add.Tensor(view_873, add_248);  view_873 = add_248 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_250, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_251 = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_68 = torch.ops.aten.sub.Tensor(add_250, getitem_91);  add_250 = getitem_91 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, arg341_1);  mul_184 = arg341_1 = None
        add_252 = torch.ops.aten.add.Tensor(mul_185, arg342_1);  mul_185 = arg342_1 = None
        slice_187 = torch.ops.aten.slice.Tensor(add_252, 0, -512, 9223372036854775807)
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(add_252, 3)
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, 4);  unsqueeze_578 = None
        permute_969 = torch.ops.aten.permute.default(unsqueeze_579, [0, 1, 3, 4, 2]);  unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg347_1, 3);  arg347_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, 4);  unsqueeze_580 = None
        permute_970 = torch.ops.aten.permute.default(unsqueeze_581, [3, 4, 1, 2, 0]);  unsqueeze_581 = None
        permute_971 = torch.ops.aten.permute.default(permute_969, [0, 1, 4, 2, 3]);  permute_969 = None
        view_874 = torch.ops.aten.view.default(permute_971, [1, 4096, 1024]);  permute_971 = None
        permute_972 = torch.ops.aten.permute.default(permute_970, [4, 2, 3, 0, 1]);  permute_970 = None
        view_875 = torch.ops.aten.view.default(permute_972, [1, 1024, 1024]);  permute_972 = None
        bmm_184 = torch.ops.aten.bmm.default(view_874, view_875);  view_874 = view_875 = None
        view_876 = torch.ops.aten.view.default(bmm_184, [512, 8, 1, 16, 64]);  bmm_184 = None
        permute_973 = torch.ops.aten.permute.default(view_876, [0, 1, 3, 4, 2]);  view_876 = None
        view_877 = torch.ops.aten.view.default(permute_973, [512, 8, 16, 64]);  permute_973 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(add_252, 3)
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, 4);  unsqueeze_582 = None
        permute_974 = torch.ops.aten.permute.default(unsqueeze_583, [0, 1, 3, 4, 2]);  unsqueeze_583 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg348_1, 3);  arg348_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, 4);  unsqueeze_584 = None
        permute_975 = torch.ops.aten.permute.default(unsqueeze_585, [3, 4, 1, 2, 0]);  unsqueeze_585 = None
        permute_976 = torch.ops.aten.permute.default(permute_974, [0, 1, 4, 2, 3]);  permute_974 = None
        view_878 = torch.ops.aten.view.default(permute_976, [1, 4096, 1024]);  permute_976 = None
        permute_977 = torch.ops.aten.permute.default(permute_975, [4, 2, 3, 0, 1]);  permute_975 = None
        view_879 = torch.ops.aten.view.default(permute_977, [1, 1024, 1024]);  permute_977 = None
        bmm_185 = torch.ops.aten.bmm.default(view_878, view_879);  view_878 = view_879 = None
        view_880 = torch.ops.aten.view.default(bmm_185, [512, 8, 1, 16, 64]);  bmm_185 = None
        permute_978 = torch.ops.aten.permute.default(view_880, [0, 1, 3, 4, 2]);  view_880 = None
        view_881 = torch.ops.aten.view.default(permute_978, [512, 8, 16, 64]);  permute_978 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(add_252, 3)
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, 4);  unsqueeze_586 = None
        permute_979 = torch.ops.aten.permute.default(unsqueeze_587, [0, 1, 3, 4, 2]);  unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg349_1, 3);  arg349_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, 4);  unsqueeze_588 = None
        permute_980 = torch.ops.aten.permute.default(unsqueeze_589, [3, 4, 1, 2, 0]);  unsqueeze_589 = None
        permute_981 = torch.ops.aten.permute.default(permute_979, [0, 1, 4, 2, 3]);  permute_979 = None
        view_882 = torch.ops.aten.view.default(permute_981, [1, 4096, 1024]);  permute_981 = None
        permute_982 = torch.ops.aten.permute.default(permute_980, [4, 2, 3, 0, 1]);  permute_980 = None
        view_883 = torch.ops.aten.view.default(permute_982, [1, 1024, 1024]);  permute_982 = None
        bmm_186 = torch.ops.aten.bmm.default(view_882, view_883);  view_882 = view_883 = None
        view_884 = torch.ops.aten.view.default(bmm_186, [512, 8, 1, 16, 64]);  bmm_186 = None
        permute_983 = torch.ops.aten.permute.default(view_884, [0, 1, 3, 4, 2]);  view_884 = None
        view_885 = torch.ops.aten.view.default(permute_983, [512, 8, 16, 64]);  permute_983 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, 4);  unsqueeze_590 = None
        permute_984 = torch.ops.aten.permute.default(unsqueeze_591, [0, 1, 3, 4, 2]);  unsqueeze_591 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg351_1, 3);  arg351_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, 4);  unsqueeze_592 = None
        permute_985 = torch.ops.aten.permute.default(unsqueeze_593, [3, 4, 1, 2, 0]);  unsqueeze_593 = None
        permute_986 = torch.ops.aten.permute.default(permute_984, [0, 1, 4, 2, 3]);  permute_984 = None
        view_886 = torch.ops.aten.view.default(permute_986, [1, 8192, 1024]);  permute_986 = None
        permute_987 = torch.ops.aten.permute.default(permute_985, [4, 2, 3, 0, 1]);  permute_985 = None
        view_887 = torch.ops.aten.view.default(permute_987, [1, 1024, 1024]);  permute_987 = None
        bmm_187 = torch.ops.aten.bmm.default(view_886, view_887);  view_886 = view_887 = None
        view_888 = torch.ops.aten.view.default(bmm_187, [1024, 8, 1, 16, 64]);  bmm_187 = None
        permute_988 = torch.ops.aten.permute.default(view_888, [0, 1, 3, 4, 2]);  view_888 = None
        view_889 = torch.ops.aten.view.default(permute_988, [1024, 8, 16, 64]);  permute_988 = None
        add_253 = torch.ops.aten.add.Tensor(view_877, arg353_1);  arg353_1 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(add_253, 4);  add_253 = None
        permute_989 = torch.ops.aten.permute.default(unsqueeze_594, [1, 2, 0, 4, 3]);  unsqueeze_594 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(view_881, 4);  view_881 = None
        permute_990 = torch.ops.aten.permute.default(unsqueeze_595, [1, 2, 4, 0, 3]);  unsqueeze_595 = None
        permute_991 = torch.ops.aten.permute.default(permute_989, [0, 1, 2, 4, 3]);  permute_989 = None
        view_890 = torch.ops.aten.view.default(permute_991, [128, 512, 64]);  permute_991 = None
        permute_992 = torch.ops.aten.permute.default(permute_990, [0, 1, 4, 3, 2]);  permute_990 = None
        view_891 = torch.ops.aten.view.default(permute_992, [128, 64, 512]);  permute_992 = None
        bmm_188 = torch.ops.aten.bmm.default(view_890, view_891);  view_890 = view_891 = None
        view_892 = torch.ops.aten.view.default(bmm_188, [8, 16, 512, 1, 512]);  bmm_188 = None
        permute_993 = torch.ops.aten.permute.default(view_892, [0, 1, 2, 4, 3]);  view_892 = None
        view_893 = torch.ops.aten.view.default(permute_993, [8, 16, 512, 512]);  permute_993 = None
        add_254 = torch.ops.aten.add.Tensor(view_877, arg352_1);  view_877 = arg352_1 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(add_254, 4);  add_254 = None
        permute_994 = torch.ops.aten.permute.default(unsqueeze_596, [1, 2, 0, 4, 3]);  unsqueeze_596 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(view_889, 4);  view_889 = None
        permute_995 = torch.ops.aten.permute.default(unsqueeze_597, [1, 2, 4, 0, 3]);  unsqueeze_597 = None
        permute_996 = torch.ops.aten.permute.default(permute_994, [0, 1, 2, 4, 3]);  permute_994 = None
        view_894 = torch.ops.aten.view.default(permute_996, [128, 512, 64]);  permute_996 = None
        permute_997 = torch.ops.aten.permute.default(permute_995, [0, 1, 4, 3, 2]);  permute_995 = None
        view_895 = torch.ops.aten.view.default(permute_997, [128, 64, 1024]);  permute_997 = None
        bmm_189 = torch.ops.aten.bmm.default(view_894, view_895);  view_894 = view_895 = None
        view_896 = torch.ops.aten.view.default(bmm_189, [8, 16, 512, 1, 1024]);  bmm_189 = None
        permute_998 = torch.ops.aten.permute.default(view_896, [0, 1, 2, 4, 3]);  view_896 = None
        view_897 = torch.ops.aten.view.default(permute_998, [8, 16, 512, 1024]);  permute_998 = None
        view_898 = torch.ops.aten.view.default(view_897, [8, 16, 1024, 512]);  view_897 = None
        slice_190 = torch.ops.aten.slice.Tensor(view_898, 2, 1, 9223372036854775807);  view_898 = None
        view_899 = torch.ops.aten.view.default(slice_190, [8, 16, 512, 1023]);  slice_190 = None
        iota_25 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_23 = torch.ops.aten.index.Tensor(view_899, [None, None, None, iota_25]);  view_899 = iota_25 = None
        add_255 = torch.ops.aten.add.Tensor(view_893, index_23);  view_893 = index_23 = None
        add_256 = torch.ops.aten.add.Tensor(add_255, 0);  add_255 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_256, 1);  add_256 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [3], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_23 = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
        div_24 = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(div_24, 4);  div_24 = None
        permute_999 = torch.ops.aten.permute.default(unsqueeze_598, [2, 0, 1, 4, 3]);  unsqueeze_598 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(view_885, 4);  view_885 = None
        permute_1000 = torch.ops.aten.permute.default(unsqueeze_599, [4, 1, 2, 3, 0]);  unsqueeze_599 = None
        permute_1001 = torch.ops.aten.permute.default(permute_999, [1, 2, 0, 4, 3]);  permute_999 = None
        view_900 = torch.ops.aten.view.default(permute_1001, [128, 512, 512]);  permute_1001 = None
        permute_1002 = torch.ops.aten.permute.default(permute_1000, [1, 2, 4, 3, 0]);  permute_1000 = None
        view_901 = torch.ops.aten.view.default(permute_1002, [128, 512, 64]);  permute_1002 = None
        bmm_190 = torch.ops.aten.bmm.default(view_900, view_901);  view_900 = view_901 = None
        view_902 = torch.ops.aten.view.default(bmm_190, [8, 16, 512, 1, 64]);  bmm_190 = None
        permute_1003 = torch.ops.aten.permute.default(view_902, [2, 0, 1, 4, 3]);  view_902 = None
        view_903 = torch.ops.aten.view.default(permute_1003, [512, 8, 16, 64]);  permute_1003 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(view_903, 4);  view_903 = None
        permute_1004 = torch.ops.aten.permute.default(unsqueeze_600, [0, 1, 4, 3, 2]);  unsqueeze_600 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(arg350_1, 3);  arg350_1 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(unsqueeze_601, 4);  unsqueeze_601 = None
        permute_1005 = torch.ops.aten.permute.default(unsqueeze_602, [3, 4, 0, 2, 1]);  unsqueeze_602 = None
        permute_1006 = torch.ops.aten.permute.default(permute_1004, [0, 1, 3, 4, 2]);  permute_1004 = None
        clone_142 = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
        view_904 = torch.ops.aten.view.default(clone_142, [1, 4096, 1024]);  clone_142 = None
        permute_1007 = torch.ops.aten.permute.default(permute_1005, [3, 4, 2, 0, 1]);  permute_1005 = None
        clone_143 = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
        view_905 = torch.ops.aten.view.default(clone_143, [1, 1024, 1024]);  clone_143 = None
        bmm_191 = torch.ops.aten.bmm.default(view_904, view_905);  view_904 = view_905 = None
        view_906 = torch.ops.aten.view.default(bmm_191, [512, 8, 1, 1, 1024]);  bmm_191 = None
        permute_1008 = torch.ops.aten.permute.default(view_906, [0, 1, 4, 2, 3]);  view_906 = None
        view_907 = torch.ops.aten.view.default(permute_1008, [512, 8, 1024]);  permute_1008 = None
        add_257 = torch.ops.aten.add.Tensor(view_907, add_252);  view_907 = add_252 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_257, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_258 = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        sub_70 = torch.ops.aten.sub.Tensor(add_257, getitem_93);  add_257 = getitem_93 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = rsqrt_46 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, arg354_1);  mul_187 = arg354_1 = None
        add_259 = torch.ops.aten.add.Tensor(mul_188, arg355_1);  mul_188 = arg355_1 = None
        view_908 = torch.ops.aten.view.default(add_259, [4096, 1024])
        permute_1009 = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg359_1, view_908, permute_1009);  arg359_1 = view_908 = permute_1009 = None
        view_909 = torch.ops.aten.view.default(addmm_46, [512, 8, 4096]);  addmm_46 = None
        mul_189 = torch.ops.aten.mul.Tensor(view_909, 0.5)
        mul_190 = torch.ops.aten.mul.Tensor(view_909, 0.7071067811865476);  view_909 = None
        erf_23 = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_260 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_189, add_260);  mul_189 = add_260 = None
        view_910 = torch.ops.aten.view.default(mul_191, [4096, 4096]);  mul_191 = None
        permute_1010 = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg361_1, view_910, permute_1010);  arg361_1 = view_910 = permute_1010 = None
        view_911 = torch.ops.aten.view.default(addmm_47, [512, 8, 1024]);  addmm_47 = None
        add_261 = torch.ops.aten.add.Tensor(view_911, add_259);  view_911 = add_259 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_261, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_262 = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_261, getitem_95);  add_261 = getitem_95 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, arg356_1);  mul_192 = arg356_1 = None
        add_263 = torch.ops.aten.add.Tensor(mul_193, arg357_1);  mul_193 = arg357_1 = None
        permute_1011 = torch.ops.aten.permute.default(add_263, [1, 0, 2]);  add_263 = None
        clone_148 = torch.ops.aten.clone.default(permute_1011, memory_format = torch.contiguous_format);  permute_1011 = None
        view_912 = torch.ops.aten.view.default(clone_148, [4096, 1024]);  clone_148 = None
        permute_1012 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg362_1, view_912, permute_1012);  arg362_1 = view_912 = permute_1012 = None
        view_913 = torch.ops.aten.view.default(addmm_48, [8, 512, 32000]);  addmm_48 = None
        view_914 = torch.ops.aten.view.default(view_913, [-1, 32000])
        view_915 = torch.ops.aten.view.default(arg363_1, [-1]);  arg363_1 = None
        amax_24 = torch.ops.aten.amax.default(view_914, [1], True)
        sub_72 = torch.ops.aten.sub.Tensor(view_914, amax_24);  view_914 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_72)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_73 = torch.ops.aten.sub.Tensor(sub_72, log);  sub_72 = log = None
        ne = torch.ops.aten.ne.Scalar(view_915, -100)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, view_915, full_default);  ne = full_default = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_73, 1, unsqueeze_603);  sub_73 = unsqueeze_603 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_915, -100)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_915, -100);  view_915 = None
        sum_26 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_25 = torch.ops.aten.div.Tensor(sum_27, convert_element_type_3);  sum_27 = convert_element_type_3 = None
        return (div_25, view_913, slice_3, slice_11, slice_19, slice_27, slice_35, slice_43, slice_51, slice_59, slice_67, slice_75, slice_83, slice_91, slice_99, slice_107, slice_115, slice_123, slice_131, slice_139, slice_147, slice_155, slice_163, slice_171, slice_179, slice_187)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (8, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32000, 1024), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1024, 16, 64), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024, 16, 64), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1024, 16, 64), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1024, 16, 64), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1024, 16, 64), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf7, (16, 64), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf8, (16, 64), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1024,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1024,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1024,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf13, (4096, 1024), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4096,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1024, 4096), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1024,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1024, 16, 64), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1024, 16, 64), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024, 16, 64), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024, 16, 64), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1024, 16, 64), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf22, (16, 64), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf23, (16, 64), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1024,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1024,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1024,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1024,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf28, (4096, 1024), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf29, (4096,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1024, 4096), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1024,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1024, 16, 64), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1024, 16, 64), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1024, 16, 64), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1024, 16, 64), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024, 16, 64), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf37, (16, 64), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf38, (16, 64), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1024,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1024,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1024,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1024,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf43, (4096, 1024), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf44, (4096,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1024, 4096), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1024,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1024, 16, 64), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1024, 16, 64), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf49, (1024, 16, 64), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1024, 16, 64), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024, 16, 64), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf52, (16, 64), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf53, (16, 64), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf54, (1024,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf55, (1024,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf56, (1024,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf57, (1024,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf58, (4096, 1024), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf59, (4096,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1024, 4096), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1024,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf62, (1024, 16, 64), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1024, 16, 64), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf64, (1024, 16, 64), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1024, 16, 64), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf66, (1024, 16, 64), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf67, (16, 64), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf68, (16, 64), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1024,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1024,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1024,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1024,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf73, (4096, 1024), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf74, (4096,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf75, (1024, 4096), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf76, (1024,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf77, (1024, 16, 64), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1024, 16, 64), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1024, 16, 64), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf80, (1024, 16, 64), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf81, (1024, 16, 64), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf82, (16, 64), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf83, (16, 64), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1024,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1024,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf86, (1024,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1024,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf88, (4096, 1024), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf89, (4096,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf90, (1024, 4096), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1024,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1024, 16, 64), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1024, 16, 64), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf94, (1024, 16, 64), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf95, (1024, 16, 64), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1024, 16, 64), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf97, (16, 64), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf98, (16, 64), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1024,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1024,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1024,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf103, (4096, 1024), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf104, (4096,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1024, 4096), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1024,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1024, 16, 64), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1024, 16, 64), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1024, 16, 64), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1024, 16, 64), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf111, (1024, 16, 64), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf112, (16, 64), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf113, (16, 64), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1024,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1024,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1024,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1024,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf118, (4096, 1024), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf119, (4096,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1024, 4096), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1024,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1024, 16, 64), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf123, (1024, 16, 64), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1024, 16, 64), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1024, 16, 64), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1024, 16, 64), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf127, (16, 64), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf128, (16, 64), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf129, (1024,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf130, (1024,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf133, (4096, 1024), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf134, (4096,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024, 4096), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024, 16, 64), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1024, 16, 64), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1024, 16, 64), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1024, 16, 64), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1024, 16, 64), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf142, (16, 64), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf143, (16, 64), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1024,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1024,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1024,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1024,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf148, (4096, 1024), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf149, (4096,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1024, 4096), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1024,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1024, 16, 64), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1024, 16, 64), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf154, (1024, 16, 64), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1024, 16, 64), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1024, 16, 64), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf157, (16, 64), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf158, (16, 64), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1024,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf160, (1024,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1024,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1024,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf163, (4096, 1024), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf164, (4096,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf165, (1024, 4096), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1024,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1024, 16, 64), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1024, 16, 64), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1024, 16, 64), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1024, 16, 64), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1024, 16, 64), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf172, (16, 64), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf173, (16, 64), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1024,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1024,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1024,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1024,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf178, (4096, 1024), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf179, (4096,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1024, 4096), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1024,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1024, 16, 64), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1024, 16, 64), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1024, 16, 64), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1024, 16, 64), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1024, 16, 64), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf187, (16, 64), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf188, (16, 64), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1024,), is_leaf=True)  # arg189_1
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
    buf197 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1024, 16, 64), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1024, 16, 64), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1024, 16, 64), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1024, 16, 64), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1024, 16, 64), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf202, (16, 64), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf203, (16, 64), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1024,), is_leaf=True)  # arg204_1
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
    buf212 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1024, 16, 64), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1024, 16, 64), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1024, 16, 64), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1024, 16, 64), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1024, 16, 64), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf217, (16, 64), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf218, (16, 64), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1024,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1024,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf221, (1024,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf222, (1024,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf223, (4096, 1024), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf224, (4096,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf225, (1024, 4096), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1024,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1024, 16, 64), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf228, (1024, 16, 64), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf229, (1024, 16, 64), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf230, (1024, 16, 64), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1024, 16, 64), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf232, (16, 64), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf233, (16, 64), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf234, (1024,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf235, (1024,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1024,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf237, (1024,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf238, (4096, 1024), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf239, (4096,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf240, (1024, 4096), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf241, (1024,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf242, (1024, 16, 64), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf243, (1024, 16, 64), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf244, (1024, 16, 64), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1024, 16, 64), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1024, 16, 64), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf247, (16, 64), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf248, (16, 64), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1024,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf250, (1024,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf251, (1024,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf252, (1024,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf253, (4096, 1024), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf254, (4096,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1024, 4096), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1024,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1024, 16, 64), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1024, 16, 64), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1024, 16, 64), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1024, 16, 64), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1024, 16, 64), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf262, (16, 64), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf263, (16, 64), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1024,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1024,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1024,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1024,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf268, (4096, 1024), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf269, (4096,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf270, (1024, 4096), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1024,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf272, (1024, 16, 64), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf273, (1024, 16, 64), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf274, (1024, 16, 64), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1024, 16, 64), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1024, 16, 64), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf277, (16, 64), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf278, (16, 64), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1024,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1024,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf281, (1024,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf282, (1024,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf283, (4096, 1024), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf284, (4096,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf285, (1024, 4096), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1024,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf287, (1024, 16, 64), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf288, (1024, 16, 64), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf289, (1024, 16, 64), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1024, 16, 64), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1024, 16, 64), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf292, (16, 64), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf293, (16, 64), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1024,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1024,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf296, (1024,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf297, (1024,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf298, (4096, 1024), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf299, (4096,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf300, (1024, 4096), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf301, (1024,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1024, 16, 64), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1024, 16, 64), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1024, 16, 64), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1024, 16, 64), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1024, 16, 64), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf307, (16, 64), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf308, (16, 64), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1024,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf310, (1024,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf311, (1024,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf312, (1024,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf313, (4096, 1024), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf314, (4096,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf315, (1024, 4096), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1024,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1024, 16, 64), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1024, 16, 64), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1024, 16, 64), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf320, (1024, 16, 64), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf321, (1024, 16, 64), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf322, (16, 64), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf323, (16, 64), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1024,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1024,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1024,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1024,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf328, (4096, 1024), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf329, (4096,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1024, 4096), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1024,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1024, 16, 64), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1024, 16, 64), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf334, (1024, 16, 64), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf335, (1024, 16, 64), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf336, (1024, 16, 64), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf337, (16, 64), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf338, (16, 64), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1024,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1024,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1024,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1024,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf343, (4096, 1024), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf344, (4096,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1024, 4096), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1024,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1024, 16, 64), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1024, 16, 64), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1024, 16, 64), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1024, 16, 64), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1024, 16, 64), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf352, (16, 64), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf353, (16, 64), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf354, (1024,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1024,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1024,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf357, (1024,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf358, (4096, 1024), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf359, (4096,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf360, (1024, 4096), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf361, (1024,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 128000, device=device(type='cuda', index=0))
    reader.tensor(buf362, (32000,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf363, (8, 512), dtype=torch.int64, is_leaf=True)  # arg363_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)