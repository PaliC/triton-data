
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1):
        view = torch.ops.aten.view.default(arg1_1, [-1, 128])
        embedding = torch.ops.aten.embedding.default(arg2_1, view, 0);  view = None
        mul = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, iota);  arg3_1 = iota = None
        add = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        view_1 = torch.ops.aten.view.default(add_2, [4096, 1024])
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view_1, permute);  arg7_1 = view_1 = permute = None
        view_2 = torch.ops.aten.view.default(addmm, [32, 128, 1024]);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None
        view_3 = torch.ops.aten.view.default(add_2, [4096, 1024])
        permute_1 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_3, permute_1);  arg9_1 = view_3 = permute_1 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [32, 128, 1024]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [32, -1, 16, 64]);  view_4 = None
        permute_2 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_6 = torch.ops.aten.view.default(add_2, [4096, 1024]);  add_2 = None
        permute_3 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_6, permute_3);  arg11_1 = view_6 = permute_3 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [32, 128, 1024]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [32, -1, 16, 64]);  view_7 = None
        permute_4 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_9 = torch.ops.aten.view.default(mul_3, [32, 128, 16, 64]);  mul_3 = None
        permute_5 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_10 = torch.ops.aten.view.default(clone_3, [512, -1, 64]);  clone_3 = None
        view_11 = torch.ops.aten.view.default(clone_1, [512, -1, 64]);  clone_1 = None
        view_12 = torch.ops.aten.view.default(clone_2, [512, -1, 64]);  clone_2 = None
        unsqueeze_default_69 = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
        unsqueeze_default_70 = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
        unsqueeze_default_71 = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
        _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_69, unsqueeze_default_70, unsqueeze_default_71, None, False, scale = 1.0);  unsqueeze_default_69 = unsqueeze_default_70 = unsqueeze_default_71 = None
        getitem_147 = _scaled_dot_product_efficient_attention_default_23[0];  _scaled_dot_product_efficient_attention_default_23 = None
        squeeze_dim_23 = torch.ops.aten.squeeze.dim(getitem_147, 0);  getitem_147 = None
        view_13 = torch.ops.aten.view.default(squeeze_dim_23, [32, 16, 128, 64]);  squeeze_dim_23 = None
        permute_7 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_5 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_14 = torch.ops.aten.view.default(clone_5, [32, 128, 1024]);  clone_5 = None
        view_15 = torch.ops.aten.view.default(view_14, [4096, 1024]);  view_14 = None
        permute_8 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_15, permute_8);  arg13_1 = view_15 = permute_8 = None
        view_16 = torch.ops.aten.view.default(addmm_3, [32, 128, 1024]);  addmm_3 = None
        add_3 = torch.ops.aten.add.Tensor(add, view_16);  add = view_16 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_3, getitem_3);  getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        view_17 = torch.ops.aten.view.default(add_5, [4096, 1024]);  add_5 = None
        permute_9 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_17, permute_9);  arg17_1 = view_17 = permute_9 = None
        view_18 = torch.ops.aten.view.default(addmm_4, [32, 128, 4096]);  addmm_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_18, 0.5)
        mul_7 = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
        erf = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
        view_19 = torch.ops.aten.view.default(mul_8, [4096, 4096]);  mul_8 = None
        permute_10 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_19, permute_10);  arg19_1 = view_19 = permute_10 = None
        view_20 = torch.ops.aten.view.default(addmm_5, [32, 128, 1024]);  addmm_5 = None
        add_7 = torch.ops.aten.add.Tensor(add_3, view_20);  add_3 = view_20 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        add_9 = torch.ops.aten.add.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        view_21 = torch.ops.aten.view.default(add_9, [4096, 1024])
        permute_11 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_21, permute_11);  arg23_1 = view_21 = permute_11 = None
        view_22 = torch.ops.aten.view.default(addmm_6, [32, 128, 1024]);  addmm_6 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_22, 0.125);  view_22 = None
        view_23 = torch.ops.aten.view.default(add_9, [4096, 1024])
        permute_12 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_23, permute_12);  arg25_1 = view_23 = permute_12 = None
        view_24 = torch.ops.aten.view.default(addmm_7, [32, 128, 1024]);  addmm_7 = None
        view_25 = torch.ops.aten.view.default(view_24, [32, -1, 16, 64]);  view_24 = None
        permute_13 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_9 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_26 = torch.ops.aten.view.default(add_9, [4096, 1024]);  add_9 = None
        permute_14 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg27_1, view_26, permute_14);  arg27_1 = view_26 = permute_14 = None
        view_27 = torch.ops.aten.view.default(addmm_8, [32, 128, 1024]);  addmm_8 = None
        view_28 = torch.ops.aten.view.default(view_27, [32, -1, 16, 64]);  view_27 = None
        permute_15 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_10 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_29 = torch.ops.aten.view.default(mul_11, [32, 128, 16, 64]);  mul_11 = None
        permute_16 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        clone_11 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_30 = torch.ops.aten.view.default(clone_11, [512, -1, 64]);  clone_11 = None
        view_31 = torch.ops.aten.view.default(clone_9, [512, -1, 64]);  clone_9 = None
        view_32 = torch.ops.aten.view.default(clone_10, [512, -1, 64]);  clone_10 = None
        unsqueeze_default_66 = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
        unsqueeze_default_67 = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
        unsqueeze_default_68 = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
        _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_66, unsqueeze_default_67, unsqueeze_default_68, None, False, scale = 1.0);  unsqueeze_default_66 = unsqueeze_default_67 = unsqueeze_default_68 = None
        getitem_146 = _scaled_dot_product_efficient_attention_default_22[0];  _scaled_dot_product_efficient_attention_default_22 = None
        squeeze_dim_22 = torch.ops.aten.squeeze.dim(getitem_146, 0);  getitem_146 = None
        view_33 = torch.ops.aten.view.default(squeeze_dim_22, [32, 16, 128, 64]);  squeeze_dim_22 = None
        permute_18 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_13 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_34 = torch.ops.aten.view.default(clone_13, [32, 128, 1024]);  clone_13 = None
        view_35 = torch.ops.aten.view.default(view_34, [4096, 1024]);  view_34 = None
        permute_19 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg29_1, view_35, permute_19);  arg29_1 = view_35 = permute_19 = None
        view_36 = torch.ops.aten.view.default(addmm_9, [32, 128, 1024]);  addmm_9 = None
        add_10 = torch.ops.aten.add.Tensor(add_7, view_36);  add_7 = view_36 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_10, getitem_7);  getitem_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg30_1);  mul_12 = arg30_1 = None
        add_12 = torch.ops.aten.add.Tensor(mul_13, arg31_1);  mul_13 = arg31_1 = None
        view_37 = torch.ops.aten.view.default(add_12, [4096, 1024]);  add_12 = None
        permute_20 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg33_1, view_37, permute_20);  arg33_1 = view_37 = permute_20 = None
        view_38 = torch.ops.aten.view.default(addmm_10, [32, 128, 4096]);  addmm_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_38, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
        erf_1 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_13 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_13);  mul_14 = add_13 = None
        view_39 = torch.ops.aten.view.default(mul_16, [4096, 4096]);  mul_16 = None
        permute_21 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg35_1, view_39, permute_21);  arg35_1 = view_39 = permute_21 = None
        view_40 = torch.ops.aten.view.default(addmm_11, [32, 128, 1024]);  addmm_11 = None
        add_14 = torch.ops.aten.add.Tensor(add_10, view_40);  add_10 = view_40 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_14, getitem_9);  getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg36_1);  mul_17 = arg36_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_18, arg37_1);  mul_18 = arg37_1 = None
        view_41 = torch.ops.aten.view.default(add_16, [4096, 1024])
        permute_22 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg39_1, view_41, permute_22);  arg39_1 = view_41 = permute_22 = None
        view_42 = torch.ops.aten.view.default(addmm_12, [32, 128, 1024]);  addmm_12 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
        view_43 = torch.ops.aten.view.default(add_16, [4096, 1024])
        permute_23 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg41_1, view_43, permute_23);  arg41_1 = view_43 = permute_23 = None
        view_44 = torch.ops.aten.view.default(addmm_13, [32, 128, 1024]);  addmm_13 = None
        view_45 = torch.ops.aten.view.default(view_44, [32, -1, 16, 64]);  view_44 = None
        permute_24 = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_17 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_46 = torch.ops.aten.view.default(add_16, [4096, 1024]);  add_16 = None
        permute_25 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg43_1, view_46, permute_25);  arg43_1 = view_46 = permute_25 = None
        view_47 = torch.ops.aten.view.default(addmm_14, [32, 128, 1024]);  addmm_14 = None
        view_48 = torch.ops.aten.view.default(view_47, [32, -1, 16, 64]);  view_47 = None
        permute_26 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_18 = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_49 = torch.ops.aten.view.default(mul_19, [32, 128, 16, 64]);  mul_19 = None
        permute_27 = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_19 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_50 = torch.ops.aten.view.default(clone_19, [512, -1, 64]);  clone_19 = None
        view_51 = torch.ops.aten.view.default(clone_17, [512, -1, 64]);  clone_17 = None
        view_52 = torch.ops.aten.view.default(clone_18, [512, -1, 64]);  clone_18 = None
        unsqueeze_default_63 = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        unsqueeze_default_64 = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
        unsqueeze_default_65 = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
        _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_63, unsqueeze_default_64, unsqueeze_default_65, None, False, scale = 1.0);  unsqueeze_default_63 = unsqueeze_default_64 = unsqueeze_default_65 = None
        getitem_145 = _scaled_dot_product_efficient_attention_default_21[0];  _scaled_dot_product_efficient_attention_default_21 = None
        squeeze_dim_21 = torch.ops.aten.squeeze.dim(getitem_145, 0);  getitem_145 = None
        view_53 = torch.ops.aten.view.default(squeeze_dim_21, [32, 16, 128, 64]);  squeeze_dim_21 = None
        permute_29 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_21 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_54 = torch.ops.aten.view.default(clone_21, [32, 128, 1024]);  clone_21 = None
        view_55 = torch.ops.aten.view.default(view_54, [4096, 1024]);  view_54 = None
        permute_30 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg45_1, view_55, permute_30);  arg45_1 = view_55 = permute_30 = None
        view_56 = torch.ops.aten.view.default(addmm_15, [32, 128, 1024]);  addmm_15 = None
        add_17 = torch.ops.aten.add.Tensor(add_14, view_56);  add_14 = view_56 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_17, getitem_11);  getitem_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg46_1);  mul_20 = arg46_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_21, arg47_1);  mul_21 = arg47_1 = None
        view_57 = torch.ops.aten.view.default(add_19, [4096, 1024]);  add_19 = None
        permute_31 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg49_1, view_57, permute_31);  arg49_1 = view_57 = permute_31 = None
        view_58 = torch.ops.aten.view.default(addmm_16, [32, 128, 4096]);  addmm_16 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_20 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_20);  mul_22 = add_20 = None
        view_59 = torch.ops.aten.view.default(mul_24, [4096, 4096]);  mul_24 = None
        permute_32 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg51_1, view_59, permute_32);  arg51_1 = view_59 = permute_32 = None
        view_60 = torch.ops.aten.view.default(addmm_17, [32, 128, 1024]);  addmm_17 = None
        add_21 = torch.ops.aten.add.Tensor(add_17, view_60);  add_17 = view_60 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_13);  getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, arg52_1);  mul_25 = arg52_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_26, arg53_1);  mul_26 = arg53_1 = None
        view_61 = torch.ops.aten.view.default(add_23, [4096, 1024])
        permute_33 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_61, permute_33);  arg55_1 = view_61 = permute_33 = None
        view_62 = torch.ops.aten.view.default(addmm_18, [32, 128, 1024]);  addmm_18 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_62, 0.125);  view_62 = None
        view_63 = torch.ops.aten.view.default(add_23, [4096, 1024])
        permute_34 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_63, permute_34);  arg57_1 = view_63 = permute_34 = None
        view_64 = torch.ops.aten.view.default(addmm_19, [32, 128, 1024]);  addmm_19 = None
        view_65 = torch.ops.aten.view.default(view_64, [32, -1, 16, 64]);  view_64 = None
        permute_35 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_66 = torch.ops.aten.view.default(add_23, [4096, 1024]);  add_23 = None
        permute_36 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_66, permute_36);  arg59_1 = view_66 = permute_36 = None
        view_67 = torch.ops.aten.view.default(addmm_20, [32, 128, 1024]);  addmm_20 = None
        view_68 = torch.ops.aten.view.default(view_67, [32, -1, 16, 64]);  view_67 = None
        permute_37 = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_26 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_69 = torch.ops.aten.view.default(mul_27, [32, 128, 16, 64]);  mul_27 = None
        permute_38 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_27 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_70 = torch.ops.aten.view.default(clone_27, [512, -1, 64]);  clone_27 = None
        view_71 = torch.ops.aten.view.default(clone_25, [512, -1, 64]);  clone_25 = None
        view_72 = torch.ops.aten.view.default(clone_26, [512, -1, 64]);  clone_26 = None
        unsqueeze_default_60 = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
        unsqueeze_default_61 = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
        unsqueeze_default_62 = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
        _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_60, unsqueeze_default_61, unsqueeze_default_62, None, False, scale = 1.0);  unsqueeze_default_60 = unsqueeze_default_61 = unsqueeze_default_62 = None
        getitem_144 = _scaled_dot_product_efficient_attention_default_20[0];  _scaled_dot_product_efficient_attention_default_20 = None
        squeeze_dim_20 = torch.ops.aten.squeeze.dim(getitem_144, 0);  getitem_144 = None
        view_73 = torch.ops.aten.view.default(squeeze_dim_20, [32, 16, 128, 64]);  squeeze_dim_20 = None
        permute_40 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        clone_29 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_74 = torch.ops.aten.view.default(clone_29, [32, 128, 1024]);  clone_29 = None
        view_75 = torch.ops.aten.view.default(view_74, [4096, 1024]);  view_74 = None
        permute_41 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_75, permute_41);  arg61_1 = view_75 = permute_41 = None
        view_76 = torch.ops.aten.view.default(addmm_21, [32, 128, 1024]);  addmm_21 = None
        add_24 = torch.ops.aten.add.Tensor(add_21, view_76);  add_21 = view_76 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_24, getitem_15);  getitem_15 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg62_1);  mul_28 = arg62_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_29, arg63_1);  mul_29 = arg63_1 = None
        view_77 = torch.ops.aten.view.default(add_26, [4096, 1024]);  add_26 = None
        permute_42 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg65_1, view_77, permute_42);  arg65_1 = view_77 = permute_42 = None
        view_78 = torch.ops.aten.view.default(addmm_22, [32, 128, 4096]);  addmm_22 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_78, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
        erf_3 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_27 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_27);  mul_30 = add_27 = None
        view_79 = torch.ops.aten.view.default(mul_32, [4096, 4096]);  mul_32 = None
        permute_43 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg67_1, view_79, permute_43);  arg67_1 = view_79 = permute_43 = None
        view_80 = torch.ops.aten.view.default(addmm_23, [32, 128, 1024]);  addmm_23 = None
        add_28 = torch.ops.aten.add.Tensor(add_24, view_80);  add_24 = view_80 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_28, getitem_17);  getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg68_1);  mul_33 = arg68_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_34, arg69_1);  mul_34 = arg69_1 = None
        view_81 = torch.ops.aten.view.default(add_30, [4096, 1024])
        permute_44 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg71_1, view_81, permute_44);  arg71_1 = view_81 = permute_44 = None
        view_82 = torch.ops.aten.view.default(addmm_24, [32, 128, 1024]);  addmm_24 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_82, 0.125);  view_82 = None
        view_83 = torch.ops.aten.view.default(add_30, [4096, 1024])
        permute_45 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg73_1, view_83, permute_45);  arg73_1 = view_83 = permute_45 = None
        view_84 = torch.ops.aten.view.default(addmm_25, [32, 128, 1024]);  addmm_25 = None
        view_85 = torch.ops.aten.view.default(view_84, [32, -1, 16, 64]);  view_84 = None
        permute_46 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        clone_33 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_86 = torch.ops.aten.view.default(add_30, [4096, 1024]);  add_30 = None
        permute_47 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg75_1, view_86, permute_47);  arg75_1 = view_86 = permute_47 = None
        view_87 = torch.ops.aten.view.default(addmm_26, [32, 128, 1024]);  addmm_26 = None
        view_88 = torch.ops.aten.view.default(view_87, [32, -1, 16, 64]);  view_87 = None
        permute_48 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_34 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_89 = torch.ops.aten.view.default(mul_35, [32, 128, 16, 64]);  mul_35 = None
        permute_49 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_35 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_90 = torch.ops.aten.view.default(clone_35, [512, -1, 64]);  clone_35 = None
        view_91 = torch.ops.aten.view.default(clone_33, [512, -1, 64]);  clone_33 = None
        view_92 = torch.ops.aten.view.default(clone_34, [512, -1, 64]);  clone_34 = None
        unsqueeze_default_57 = torch.ops.aten.unsqueeze.default(view_90, 0);  view_90 = None
        unsqueeze_default_58 = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
        unsqueeze_default_59 = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
        _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_57, unsqueeze_default_58, unsqueeze_default_59, None, False, scale = 1.0);  unsqueeze_default_57 = unsqueeze_default_58 = unsqueeze_default_59 = None
        getitem_143 = _scaled_dot_product_efficient_attention_default_19[0];  _scaled_dot_product_efficient_attention_default_19 = None
        squeeze_dim_19 = torch.ops.aten.squeeze.dim(getitem_143, 0);  getitem_143 = None
        view_93 = torch.ops.aten.view.default(squeeze_dim_19, [32, 16, 128, 64]);  squeeze_dim_19 = None
        permute_51 = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        clone_37 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_94 = torch.ops.aten.view.default(clone_37, [32, 128, 1024]);  clone_37 = None
        view_95 = torch.ops.aten.view.default(view_94, [4096, 1024]);  view_94 = None
        permute_52 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg77_1, view_95, permute_52);  arg77_1 = view_95 = permute_52 = None
        view_96 = torch.ops.aten.view.default(addmm_27, [32, 128, 1024]);  addmm_27 = None
        add_31 = torch.ops.aten.add.Tensor(add_28, view_96);  add_28 = view_96 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_31, getitem_19);  getitem_19 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg78_1);  mul_36 = arg78_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_37, arg79_1);  mul_37 = arg79_1 = None
        view_97 = torch.ops.aten.view.default(add_33, [4096, 1024]);  add_33 = None
        permute_53 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg81_1, view_97, permute_53);  arg81_1 = view_97 = permute_53 = None
        view_98 = torch.ops.aten.view.default(addmm_28, [32, 128, 4096]);  addmm_28 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_98, 0.5)
        mul_39 = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
        erf_4 = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_34 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, add_34);  mul_38 = add_34 = None
        view_99 = torch.ops.aten.view.default(mul_40, [4096, 4096]);  mul_40 = None
        permute_54 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg83_1, view_99, permute_54);  arg83_1 = view_99 = permute_54 = None
        view_100 = torch.ops.aten.view.default(addmm_29, [32, 128, 1024]);  addmm_29 = None
        add_35 = torch.ops.aten.add.Tensor(add_31, view_100);  add_31 = view_100 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_35, getitem_21);  getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, arg84_1);  mul_41 = arg84_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_42, arg85_1);  mul_42 = arg85_1 = None
        view_101 = torch.ops.aten.view.default(add_37, [4096, 1024])
        permute_55 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg87_1, view_101, permute_55);  arg87_1 = view_101 = permute_55 = None
        view_102 = torch.ops.aten.view.default(addmm_30, [32, 128, 1024]);  addmm_30 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_102, 0.125);  view_102 = None
        view_103 = torch.ops.aten.view.default(add_37, [4096, 1024])
        permute_56 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg89_1, view_103, permute_56);  arg89_1 = view_103 = permute_56 = None
        view_104 = torch.ops.aten.view.default(addmm_31, [32, 128, 1024]);  addmm_31 = None
        view_105 = torch.ops.aten.view.default(view_104, [32, -1, 16, 64]);  view_104 = None
        permute_57 = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        clone_41 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_106 = torch.ops.aten.view.default(add_37, [4096, 1024]);  add_37 = None
        permute_58 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg91_1, view_106, permute_58);  arg91_1 = view_106 = permute_58 = None
        view_107 = torch.ops.aten.view.default(addmm_32, [32, 128, 1024]);  addmm_32 = None
        view_108 = torch.ops.aten.view.default(view_107, [32, -1, 16, 64]);  view_107 = None
        permute_59 = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_42 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_109 = torch.ops.aten.view.default(mul_43, [32, 128, 16, 64]);  mul_43 = None
        permute_60 = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_43 = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_110 = torch.ops.aten.view.default(clone_43, [512, -1, 64]);  clone_43 = None
        view_111 = torch.ops.aten.view.default(clone_41, [512, -1, 64]);  clone_41 = None
        view_112 = torch.ops.aten.view.default(clone_42, [512, -1, 64]);  clone_42 = None
        unsqueeze_default_54 = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
        unsqueeze_default_55 = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
        unsqueeze_default_56 = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
        _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_54, unsqueeze_default_55, unsqueeze_default_56, None, False, scale = 1.0);  unsqueeze_default_54 = unsqueeze_default_55 = unsqueeze_default_56 = None
        getitem_142 = _scaled_dot_product_efficient_attention_default_18[0];  _scaled_dot_product_efficient_attention_default_18 = None
        squeeze_dim_18 = torch.ops.aten.squeeze.dim(getitem_142, 0);  getitem_142 = None
        view_113 = torch.ops.aten.view.default(squeeze_dim_18, [32, 16, 128, 64]);  squeeze_dim_18 = None
        permute_62 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_45 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_114 = torch.ops.aten.view.default(clone_45, [32, 128, 1024]);  clone_45 = None
        view_115 = torch.ops.aten.view.default(view_114, [4096, 1024]);  view_114 = None
        permute_63 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg93_1, view_115, permute_63);  arg93_1 = view_115 = permute_63 = None
        view_116 = torch.ops.aten.view.default(addmm_33, [32, 128, 1024]);  addmm_33 = None
        add_38 = torch.ops.aten.add.Tensor(add_35, view_116);  add_35 = view_116 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_38, getitem_23);  getitem_23 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg94_1);  mul_44 = arg94_1 = None
        add_40 = torch.ops.aten.add.Tensor(mul_45, arg95_1);  mul_45 = arg95_1 = None
        view_117 = torch.ops.aten.view.default(add_40, [4096, 1024]);  add_40 = None
        permute_64 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg97_1, view_117, permute_64);  arg97_1 = view_117 = permute_64 = None
        view_118 = torch.ops.aten.view.default(addmm_34, [32, 128, 4096]);  addmm_34 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
        erf_5 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_41 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_41);  mul_46 = add_41 = None
        view_119 = torch.ops.aten.view.default(mul_48, [4096, 4096]);  mul_48 = None
        permute_65 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg99_1, view_119, permute_65);  arg99_1 = view_119 = permute_65 = None
        view_120 = torch.ops.aten.view.default(addmm_35, [32, 128, 1024]);  addmm_35 = None
        add_42 = torch.ops.aten.add.Tensor(add_38, view_120);  add_38 = view_120 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_42, getitem_25);  getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg100_1);  mul_49 = arg100_1 = None
        add_44 = torch.ops.aten.add.Tensor(mul_50, arg101_1);  mul_50 = arg101_1 = None
        view_121 = torch.ops.aten.view.default(add_44, [4096, 1024])
        permute_66 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg103_1, view_121, permute_66);  arg103_1 = view_121 = permute_66 = None
        view_122 = torch.ops.aten.view.default(addmm_36, [32, 128, 1024]);  addmm_36 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_122, 0.125);  view_122 = None
        view_123 = torch.ops.aten.view.default(add_44, [4096, 1024])
        permute_67 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg105_1, view_123, permute_67);  arg105_1 = view_123 = permute_67 = None
        view_124 = torch.ops.aten.view.default(addmm_37, [32, 128, 1024]);  addmm_37 = None
        view_125 = torch.ops.aten.view.default(view_124, [32, -1, 16, 64]);  view_124 = None
        permute_68 = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_49 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_126 = torch.ops.aten.view.default(add_44, [4096, 1024]);  add_44 = None
        permute_69 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg107_1, view_126, permute_69);  arg107_1 = view_126 = permute_69 = None
        view_127 = torch.ops.aten.view.default(addmm_38, [32, 128, 1024]);  addmm_38 = None
        view_128 = torch.ops.aten.view.default(view_127, [32, -1, 16, 64]);  view_127 = None
        permute_70 = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        clone_50 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_129 = torch.ops.aten.view.default(mul_51, [32, 128, 16, 64]);  mul_51 = None
        permute_71 = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_51 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_130 = torch.ops.aten.view.default(clone_51, [512, -1, 64]);  clone_51 = None
        view_131 = torch.ops.aten.view.default(clone_49, [512, -1, 64]);  clone_49 = None
        view_132 = torch.ops.aten.view.default(clone_50, [512, -1, 64]);  clone_50 = None
        unsqueeze_default_51 = torch.ops.aten.unsqueeze.default(view_130, 0);  view_130 = None
        unsqueeze_default_52 = torch.ops.aten.unsqueeze.default(view_131, 0);  view_131 = None
        unsqueeze_default_53 = torch.ops.aten.unsqueeze.default(view_132, 0);  view_132 = None
        _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_51, unsqueeze_default_52, unsqueeze_default_53, None, False, scale = 1.0);  unsqueeze_default_51 = unsqueeze_default_52 = unsqueeze_default_53 = None
        getitem_141 = _scaled_dot_product_efficient_attention_default_17[0];  _scaled_dot_product_efficient_attention_default_17 = None
        squeeze_dim_17 = torch.ops.aten.squeeze.dim(getitem_141, 0);  getitem_141 = None
        view_133 = torch.ops.aten.view.default(squeeze_dim_17, [32, 16, 128, 64]);  squeeze_dim_17 = None
        permute_73 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        clone_53 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_134 = torch.ops.aten.view.default(clone_53, [32, 128, 1024]);  clone_53 = None
        view_135 = torch.ops.aten.view.default(view_134, [4096, 1024]);  view_134 = None
        permute_74 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg109_1, view_135, permute_74);  arg109_1 = view_135 = permute_74 = None
        view_136 = torch.ops.aten.view.default(addmm_39, [32, 128, 1024]);  addmm_39 = None
        add_45 = torch.ops.aten.add.Tensor(add_42, view_136);  add_42 = view_136 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_45, getitem_27);  getitem_27 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg110_1);  mul_52 = arg110_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_53, arg111_1);  mul_53 = arg111_1 = None
        view_137 = torch.ops.aten.view.default(add_47, [4096, 1024]);  add_47 = None
        permute_75 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg113_1, view_137, permute_75);  arg113_1 = view_137 = permute_75 = None
        view_138 = torch.ops.aten.view.default(addmm_40, [32, 128, 4096]);  addmm_40 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_138, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
        erf_6 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_48 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_48);  mul_54 = add_48 = None
        view_139 = torch.ops.aten.view.default(mul_56, [4096, 4096]);  mul_56 = None
        permute_76 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg115_1, view_139, permute_76);  arg115_1 = view_139 = permute_76 = None
        view_140 = torch.ops.aten.view.default(addmm_41, [32, 128, 1024]);  addmm_41 = None
        add_49 = torch.ops.aten.add.Tensor(add_45, view_140);  add_45 = view_140 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_49, getitem_29);  getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg116_1);  mul_57 = arg116_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_58, arg117_1);  mul_58 = arg117_1 = None
        view_141 = torch.ops.aten.view.default(add_51, [4096, 1024])
        permute_77 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg119_1, view_141, permute_77);  arg119_1 = view_141 = permute_77 = None
        view_142 = torch.ops.aten.view.default(addmm_42, [32, 128, 1024]);  addmm_42 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
        view_143 = torch.ops.aten.view.default(add_51, [4096, 1024])
        permute_78 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg121_1, view_143, permute_78);  arg121_1 = view_143 = permute_78 = None
        view_144 = torch.ops.aten.view.default(addmm_43, [32, 128, 1024]);  addmm_43 = None
        view_145 = torch.ops.aten.view.default(view_144, [32, -1, 16, 64]);  view_144 = None
        permute_79 = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        clone_57 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_146 = torch.ops.aten.view.default(add_51, [4096, 1024]);  add_51 = None
        permute_80 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg123_1, view_146, permute_80);  arg123_1 = view_146 = permute_80 = None
        view_147 = torch.ops.aten.view.default(addmm_44, [32, 128, 1024]);  addmm_44 = None
        view_148 = torch.ops.aten.view.default(view_147, [32, -1, 16, 64]);  view_147 = None
        permute_81 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_58 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_149 = torch.ops.aten.view.default(mul_59, [32, 128, 16, 64]);  mul_59 = None
        permute_82 = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        clone_59 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_150 = torch.ops.aten.view.default(clone_59, [512, -1, 64]);  clone_59 = None
        view_151 = torch.ops.aten.view.default(clone_57, [512, -1, 64]);  clone_57 = None
        view_152 = torch.ops.aten.view.default(clone_58, [512, -1, 64]);  clone_58 = None
        unsqueeze_default_48 = torch.ops.aten.unsqueeze.default(view_150, 0);  view_150 = None
        unsqueeze_default_49 = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
        unsqueeze_default_50 = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
        _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_48, unsqueeze_default_49, unsqueeze_default_50, None, False, scale = 1.0);  unsqueeze_default_48 = unsqueeze_default_49 = unsqueeze_default_50 = None
        getitem_140 = _scaled_dot_product_efficient_attention_default_16[0];  _scaled_dot_product_efficient_attention_default_16 = None
        squeeze_dim_16 = torch.ops.aten.squeeze.dim(getitem_140, 0);  getitem_140 = None
        view_153 = torch.ops.aten.view.default(squeeze_dim_16, [32, 16, 128, 64]);  squeeze_dim_16 = None
        permute_84 = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        clone_61 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_154 = torch.ops.aten.view.default(clone_61, [32, 128, 1024]);  clone_61 = None
        view_155 = torch.ops.aten.view.default(view_154, [4096, 1024]);  view_154 = None
        permute_85 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg125_1, view_155, permute_85);  arg125_1 = view_155 = permute_85 = None
        view_156 = torch.ops.aten.view.default(addmm_45, [32, 128, 1024]);  addmm_45 = None
        add_52 = torch.ops.aten.add.Tensor(add_49, view_156);  add_49 = view_156 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_52, getitem_31);  getitem_31 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg126_1);  mul_60 = arg126_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_61, arg127_1);  mul_61 = arg127_1 = None
        view_157 = torch.ops.aten.view.default(add_54, [4096, 1024]);  add_54 = None
        permute_86 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg129_1, view_157, permute_86);  arg129_1 = view_157 = permute_86 = None
        view_158 = torch.ops.aten.view.default(addmm_46, [32, 128, 4096]);  addmm_46 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_158, 0.5)
        mul_63 = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
        erf_7 = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_55 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_62, add_55);  mul_62 = add_55 = None
        view_159 = torch.ops.aten.view.default(mul_64, [4096, 4096]);  mul_64 = None
        permute_87 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg131_1, view_159, permute_87);  arg131_1 = view_159 = permute_87 = None
        view_160 = torch.ops.aten.view.default(addmm_47, [32, 128, 1024]);  addmm_47 = None
        add_56 = torch.ops.aten.add.Tensor(add_52, view_160);  add_52 = view_160 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_56, getitem_33);  getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg132_1);  mul_65 = arg132_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_66, arg133_1);  mul_66 = arg133_1 = None
        view_161 = torch.ops.aten.view.default(add_58, [4096, 1024])
        permute_88 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg135_1, view_161, permute_88);  arg135_1 = view_161 = permute_88 = None
        view_162 = torch.ops.aten.view.default(addmm_48, [32, 128, 1024]);  addmm_48 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_162, 0.125);  view_162 = None
        view_163 = torch.ops.aten.view.default(add_58, [4096, 1024])
        permute_89 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg137_1, view_163, permute_89);  arg137_1 = view_163 = permute_89 = None
        view_164 = torch.ops.aten.view.default(addmm_49, [32, 128, 1024]);  addmm_49 = None
        view_165 = torch.ops.aten.view.default(view_164, [32, -1, 16, 64]);  view_164 = None
        permute_90 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        clone_65 = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_166 = torch.ops.aten.view.default(add_58, [4096, 1024]);  add_58 = None
        permute_91 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg139_1, view_166, permute_91);  arg139_1 = view_166 = permute_91 = None
        view_167 = torch.ops.aten.view.default(addmm_50, [32, 128, 1024]);  addmm_50 = None
        view_168 = torch.ops.aten.view.default(view_167, [32, -1, 16, 64]);  view_167 = None
        permute_92 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_66 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_169 = torch.ops.aten.view.default(mul_67, [32, 128, 16, 64]);  mul_67 = None
        permute_93 = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_67 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_170 = torch.ops.aten.view.default(clone_67, [512, -1, 64]);  clone_67 = None
        view_171 = torch.ops.aten.view.default(clone_65, [512, -1, 64]);  clone_65 = None
        view_172 = torch.ops.aten.view.default(clone_66, [512, -1, 64]);  clone_66 = None
        unsqueeze_default_45 = torch.ops.aten.unsqueeze.default(view_170, 0);  view_170 = None
        unsqueeze_default_46 = torch.ops.aten.unsqueeze.default(view_171, 0);  view_171 = None
        unsqueeze_default_47 = torch.ops.aten.unsqueeze.default(view_172, 0);  view_172 = None
        _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_45, unsqueeze_default_46, unsqueeze_default_47, None, False, scale = 1.0);  unsqueeze_default_45 = unsqueeze_default_46 = unsqueeze_default_47 = None
        getitem_139 = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
        squeeze_dim_15 = torch.ops.aten.squeeze.dim(getitem_139, 0);  getitem_139 = None
        view_173 = torch.ops.aten.view.default(squeeze_dim_15, [32, 16, 128, 64]);  squeeze_dim_15 = None
        permute_95 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        clone_69 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_174 = torch.ops.aten.view.default(clone_69, [32, 128, 1024]);  clone_69 = None
        view_175 = torch.ops.aten.view.default(view_174, [4096, 1024]);  view_174 = None
        permute_96 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg141_1, view_175, permute_96);  arg141_1 = view_175 = permute_96 = None
        view_176 = torch.ops.aten.view.default(addmm_51, [32, 128, 1024]);  addmm_51 = None
        add_59 = torch.ops.aten.add.Tensor(add_56, view_176);  add_56 = view_176 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_59, getitem_35);  getitem_35 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg142_1);  mul_68 = arg142_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_69, arg143_1);  mul_69 = arg143_1 = None
        view_177 = torch.ops.aten.view.default(add_61, [4096, 1024]);  add_61 = None
        permute_97 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg145_1, view_177, permute_97);  arg145_1 = view_177 = permute_97 = None
        view_178 = torch.ops.aten.view.default(addmm_52, [32, 128, 4096]);  addmm_52 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_178, 0.5)
        mul_71 = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
        erf_8 = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_62 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_70, add_62);  mul_70 = add_62 = None
        view_179 = torch.ops.aten.view.default(mul_72, [4096, 4096]);  mul_72 = None
        permute_98 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg147_1, view_179, permute_98);  arg147_1 = view_179 = permute_98 = None
        view_180 = torch.ops.aten.view.default(addmm_53, [32, 128, 1024]);  addmm_53 = None
        add_63 = torch.ops.aten.add.Tensor(add_59, view_180);  add_59 = view_180 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_63, getitem_37);  getitem_37 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg148_1);  mul_73 = arg148_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_74, arg149_1);  mul_74 = arg149_1 = None
        view_181 = torch.ops.aten.view.default(add_65, [4096, 1024])
        permute_99 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg151_1, view_181, permute_99);  arg151_1 = view_181 = permute_99 = None
        view_182 = torch.ops.aten.view.default(addmm_54, [32, 128, 1024]);  addmm_54 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_182, 0.125);  view_182 = None
        view_183 = torch.ops.aten.view.default(add_65, [4096, 1024])
        permute_100 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg153_1, view_183, permute_100);  arg153_1 = view_183 = permute_100 = None
        view_184 = torch.ops.aten.view.default(addmm_55, [32, 128, 1024]);  addmm_55 = None
        view_185 = torch.ops.aten.view.default(view_184, [32, -1, 16, 64]);  view_184 = None
        permute_101 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_73 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_186 = torch.ops.aten.view.default(add_65, [4096, 1024]);  add_65 = None
        permute_102 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg155_1, view_186, permute_102);  arg155_1 = view_186 = permute_102 = None
        view_187 = torch.ops.aten.view.default(addmm_56, [32, 128, 1024]);  addmm_56 = None
        view_188 = torch.ops.aten.view.default(view_187, [32, -1, 16, 64]);  view_187 = None
        permute_103 = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        clone_74 = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_189 = torch.ops.aten.view.default(mul_75, [32, 128, 16, 64]);  mul_75 = None
        permute_104 = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        clone_75 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_190 = torch.ops.aten.view.default(clone_75, [512, -1, 64]);  clone_75 = None
        view_191 = torch.ops.aten.view.default(clone_73, [512, -1, 64]);  clone_73 = None
        view_192 = torch.ops.aten.view.default(clone_74, [512, -1, 64]);  clone_74 = None
        unsqueeze_default_42 = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
        unsqueeze_default_43 = torch.ops.aten.unsqueeze.default(view_191, 0);  view_191 = None
        unsqueeze_default_44 = torch.ops.aten.unsqueeze.default(view_192, 0);  view_192 = None
        _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_42, unsqueeze_default_43, unsqueeze_default_44, None, False, scale = 1.0);  unsqueeze_default_42 = unsqueeze_default_43 = unsqueeze_default_44 = None
        getitem_138 = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
        squeeze_dim_14 = torch.ops.aten.squeeze.dim(getitem_138, 0);  getitem_138 = None
        view_193 = torch.ops.aten.view.default(squeeze_dim_14, [32, 16, 128, 64]);  squeeze_dim_14 = None
        permute_106 = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        clone_77 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_194 = torch.ops.aten.view.default(clone_77, [32, 128, 1024]);  clone_77 = None
        view_195 = torch.ops.aten.view.default(view_194, [4096, 1024]);  view_194 = None
        permute_107 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg157_1, view_195, permute_107);  arg157_1 = view_195 = permute_107 = None
        view_196 = torch.ops.aten.view.default(addmm_57, [32, 128, 1024]);  addmm_57 = None
        add_66 = torch.ops.aten.add.Tensor(add_63, view_196);  add_63 = view_196 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_67 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_66, getitem_39);  getitem_39 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, arg158_1);  mul_76 = arg158_1 = None
        add_68 = torch.ops.aten.add.Tensor(mul_77, arg159_1);  mul_77 = arg159_1 = None
        view_197 = torch.ops.aten.view.default(add_68, [4096, 1024]);  add_68 = None
        permute_108 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg161_1, view_197, permute_108);  arg161_1 = view_197 = permute_108 = None
        view_198 = torch.ops.aten.view.default(addmm_58, [32, 128, 4096]);  addmm_58 = None
        mul_78 = torch.ops.aten.mul.Tensor(view_198, 0.5)
        mul_79 = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
        erf_9 = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_69 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_78, add_69);  mul_78 = add_69 = None
        view_199 = torch.ops.aten.view.default(mul_80, [4096, 4096]);  mul_80 = None
        permute_109 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg163_1, view_199, permute_109);  arg163_1 = view_199 = permute_109 = None
        view_200 = torch.ops.aten.view.default(addmm_59, [32, 128, 1024]);  addmm_59 = None
        add_70 = torch.ops.aten.add.Tensor(add_66, view_200);  add_66 = view_200 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_70, getitem_41);  getitem_41 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, arg164_1);  mul_81 = arg164_1 = None
        add_72 = torch.ops.aten.add.Tensor(mul_82, arg165_1);  mul_82 = arg165_1 = None
        view_201 = torch.ops.aten.view.default(add_72, [4096, 1024])
        permute_110 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg167_1, view_201, permute_110);  arg167_1 = view_201 = permute_110 = None
        view_202 = torch.ops.aten.view.default(addmm_60, [32, 128, 1024]);  addmm_60 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_202, 0.125);  view_202 = None
        view_203 = torch.ops.aten.view.default(add_72, [4096, 1024])
        permute_111 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg169_1, view_203, permute_111);  arg169_1 = view_203 = permute_111 = None
        view_204 = torch.ops.aten.view.default(addmm_61, [32, 128, 1024]);  addmm_61 = None
        view_205 = torch.ops.aten.view.default(view_204, [32, -1, 16, 64]);  view_204 = None
        permute_112 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_81 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_206 = torch.ops.aten.view.default(add_72, [4096, 1024]);  add_72 = None
        permute_113 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg171_1, view_206, permute_113);  arg171_1 = view_206 = permute_113 = None
        view_207 = torch.ops.aten.view.default(addmm_62, [32, 128, 1024]);  addmm_62 = None
        view_208 = torch.ops.aten.view.default(view_207, [32, -1, 16, 64]);  view_207 = None
        permute_114 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_82 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_209 = torch.ops.aten.view.default(mul_83, [32, 128, 16, 64]);  mul_83 = None
        permute_115 = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_83 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_210 = torch.ops.aten.view.default(clone_83, [512, -1, 64]);  clone_83 = None
        view_211 = torch.ops.aten.view.default(clone_81, [512, -1, 64]);  clone_81 = None
        view_212 = torch.ops.aten.view.default(clone_82, [512, -1, 64]);  clone_82 = None
        unsqueeze_default_39 = torch.ops.aten.unsqueeze.default(view_210, 0);  view_210 = None
        unsqueeze_default_40 = torch.ops.aten.unsqueeze.default(view_211, 0);  view_211 = None
        unsqueeze_default_41 = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
        _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_39, unsqueeze_default_40, unsqueeze_default_41, None, False, scale = 1.0);  unsqueeze_default_39 = unsqueeze_default_40 = unsqueeze_default_41 = None
        getitem_137 = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
        squeeze_dim_13 = torch.ops.aten.squeeze.dim(getitem_137, 0);  getitem_137 = None
        view_213 = torch.ops.aten.view.default(squeeze_dim_13, [32, 16, 128, 64]);  squeeze_dim_13 = None
        permute_117 = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        clone_85 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_214 = torch.ops.aten.view.default(clone_85, [32, 128, 1024]);  clone_85 = None
        view_215 = torch.ops.aten.view.default(view_214, [4096, 1024]);  view_214 = None
        permute_118 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg173_1, view_215, permute_118);  arg173_1 = view_215 = permute_118 = None
        view_216 = torch.ops.aten.view.default(addmm_63, [32, 128, 1024]);  addmm_63 = None
        add_73 = torch.ops.aten.add.Tensor(add_70, view_216);  add_70 = view_216 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_73, getitem_43);  getitem_43 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg174_1);  mul_84 = arg174_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_85, arg175_1);  mul_85 = arg175_1 = None
        view_217 = torch.ops.aten.view.default(add_75, [4096, 1024]);  add_75 = None
        permute_119 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg177_1, view_217, permute_119);  arg177_1 = view_217 = permute_119 = None
        view_218 = torch.ops.aten.view.default(addmm_64, [32, 128, 4096]);  addmm_64 = None
        mul_86 = torch.ops.aten.mul.Tensor(view_218, 0.5)
        mul_87 = torch.ops.aten.mul.Tensor(view_218, 0.7071067811865476);  view_218 = None
        erf_10 = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_76 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_86, add_76);  mul_86 = add_76 = None
        view_219 = torch.ops.aten.view.default(mul_88, [4096, 4096]);  mul_88 = None
        permute_120 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg179_1, view_219, permute_120);  arg179_1 = view_219 = permute_120 = None
        view_220 = torch.ops.aten.view.default(addmm_65, [32, 128, 1024]);  addmm_65 = None
        add_77 = torch.ops.aten.add.Tensor(add_73, view_220);  add_73 = view_220 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_77, getitem_45);  getitem_45 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, arg180_1);  mul_89 = arg180_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_90, arg181_1);  mul_90 = arg181_1 = None
        view_221 = torch.ops.aten.view.default(add_79, [4096, 1024])
        permute_121 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg183_1, view_221, permute_121);  arg183_1 = view_221 = permute_121 = None
        view_222 = torch.ops.aten.view.default(addmm_66, [32, 128, 1024]);  addmm_66 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_222, 0.125);  view_222 = None
        view_223 = torch.ops.aten.view.default(add_79, [4096, 1024])
        permute_122 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg185_1, view_223, permute_122);  arg185_1 = view_223 = permute_122 = None
        view_224 = torch.ops.aten.view.default(addmm_67, [32, 128, 1024]);  addmm_67 = None
        view_225 = torch.ops.aten.view.default(view_224, [32, -1, 16, 64]);  view_224 = None
        permute_123 = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        clone_89 = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        view_226 = torch.ops.aten.view.default(add_79, [4096, 1024]);  add_79 = None
        permute_124 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg187_1, view_226, permute_124);  arg187_1 = view_226 = permute_124 = None
        view_227 = torch.ops.aten.view.default(addmm_68, [32, 128, 1024]);  addmm_68 = None
        view_228 = torch.ops.aten.view.default(view_227, [32, -1, 16, 64]);  view_227 = None
        permute_125 = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        clone_90 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_229 = torch.ops.aten.view.default(mul_91, [32, 128, 16, 64]);  mul_91 = None
        permute_126 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_91 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_230 = torch.ops.aten.view.default(clone_91, [512, -1, 64]);  clone_91 = None
        view_231 = torch.ops.aten.view.default(clone_89, [512, -1, 64]);  clone_89 = None
        view_232 = torch.ops.aten.view.default(clone_90, [512, -1, 64]);  clone_90 = None
        unsqueeze_default_36 = torch.ops.aten.unsqueeze.default(view_230, 0);  view_230 = None
        unsqueeze_default_37 = torch.ops.aten.unsqueeze.default(view_231, 0);  view_231 = None
        unsqueeze_default_38 = torch.ops.aten.unsqueeze.default(view_232, 0);  view_232 = None
        _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_36, unsqueeze_default_37, unsqueeze_default_38, None, False, scale = 1.0);  unsqueeze_default_36 = unsqueeze_default_37 = unsqueeze_default_38 = None
        getitem_136 = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
        squeeze_dim_12 = torch.ops.aten.squeeze.dim(getitem_136, 0);  getitem_136 = None
        view_233 = torch.ops.aten.view.default(squeeze_dim_12, [32, 16, 128, 64]);  squeeze_dim_12 = None
        permute_128 = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        clone_93 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_234 = torch.ops.aten.view.default(clone_93, [32, 128, 1024]);  clone_93 = None
        view_235 = torch.ops.aten.view.default(view_234, [4096, 1024]);  view_234 = None
        permute_129 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg189_1, view_235, permute_129);  arg189_1 = view_235 = permute_129 = None
        view_236 = torch.ops.aten.view.default(addmm_69, [32, 128, 1024]);  addmm_69 = None
        add_80 = torch.ops.aten.add.Tensor(add_77, view_236);  add_77 = view_236 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_80, getitem_47);  getitem_47 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, arg190_1);  mul_92 = arg190_1 = None
        add_82 = torch.ops.aten.add.Tensor(mul_93, arg191_1);  mul_93 = arg191_1 = None
        view_237 = torch.ops.aten.view.default(add_82, [4096, 1024]);  add_82 = None
        permute_130 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg193_1, view_237, permute_130);  arg193_1 = view_237 = permute_130 = None
        view_238 = torch.ops.aten.view.default(addmm_70, [32, 128, 4096]);  addmm_70 = None
        mul_94 = torch.ops.aten.mul.Tensor(view_238, 0.5)
        mul_95 = torch.ops.aten.mul.Tensor(view_238, 0.7071067811865476);  view_238 = None
        erf_11 = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_83 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_94, add_83);  mul_94 = add_83 = None
        view_239 = torch.ops.aten.view.default(mul_96, [4096, 4096]);  mul_96 = None
        permute_131 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg195_1, view_239, permute_131);  arg195_1 = view_239 = permute_131 = None
        view_240 = torch.ops.aten.view.default(addmm_71, [32, 128, 1024]);  addmm_71 = None
        add_84 = torch.ops.aten.add.Tensor(add_80, view_240);  add_80 = view_240 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, arg196_1);  mul_97 = arg196_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_98, arg197_1);  mul_98 = arg197_1 = None
        view_241 = torch.ops.aten.view.default(arg1_1, [-1, 128]);  arg1_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg2_1, view_241, 0);  view_241 = None
        mul_99 = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
        full_default = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_1 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_87 = torch.ops.aten.add.Tensor(iota_1, 1)
        view_242 = torch.ops.aten.view.default(add_87, [128, 1]);  add_87 = None
        lt = torch.ops.aten.lt.Tensor(iota_1, view_242);  iota_1 = view_242 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        iota_2 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_3 = torch.ops.aten.embedding.default(arg198_1, iota_2);  arg198_1 = iota_2 = None
        add_88 = torch.ops.aten.add.Tensor(mul_99, embedding_3);  mul_99 = embedding_3 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_89 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_88, getitem_51);  getitem_51 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, arg199_1);  mul_100 = arg199_1 = None
        add_90 = torch.ops.aten.add.Tensor(mul_101, arg200_1);  mul_101 = arg200_1 = None
        view_243 = torch.ops.aten.view.default(add_90, [4096, 1024])
        permute_132 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg202_1, view_243, permute_132);  arg202_1 = view_243 = permute_132 = None
        view_244 = torch.ops.aten.view.default(addmm_72, [32, 128, 1024]);  addmm_72 = None
        mul_102 = torch.ops.aten.mul.Tensor(view_244, 0.125);  view_244 = None
        view_245 = torch.ops.aten.view.default(add_90, [4096, 1024])
        permute_133 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg204_1, view_245, permute_133);  arg204_1 = view_245 = permute_133 = None
        view_246 = torch.ops.aten.view.default(addmm_73, [32, 128, 1024]);  addmm_73 = None
        view_247 = torch.ops.aten.view.default(view_246, [32, -1, 16, 64]);  view_246 = None
        permute_134 = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        clone_98 = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
        view_248 = torch.ops.aten.view.default(add_90, [4096, 1024]);  add_90 = None
        permute_135 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg206_1, view_248, permute_135);  arg206_1 = view_248 = permute_135 = None
        view_249 = torch.ops.aten.view.default(addmm_74, [32, 128, 1024]);  addmm_74 = None
        view_250 = torch.ops.aten.view.default(view_249, [32, -1, 16, 64]);  view_249 = None
        permute_136 = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        clone_99 = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_251 = torch.ops.aten.view.default(mul_102, [32, 128, 16, 64]);  mul_102 = None
        permute_137 = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        clone_100 = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_252 = torch.ops.aten.view.default(clone_100, [512, -1, 64]);  clone_100 = None
        view_253 = torch.ops.aten.view.default(clone_98, [512, -1, 64]);  clone_98 = None
        view_254 = torch.ops.aten.view.default(clone_99, [512, -1, 64]);  clone_99 = None
        permute_138 = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        bmm_24 = torch.ops.aten.bmm.default(view_252, permute_138);  view_252 = permute_138 = None
        view_255 = torch.ops.aten.view.default(bmm_24, [32, 16, 128, 128]);  bmm_24 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [32, 1, 128, 128]);  unsqueeze_3 = None
        add_91 = torch.ops.aten.add.Tensor(view_255, expand_1);  view_255 = None
        view_256 = torch.ops.aten.view.default(add_91, [512, 128, 128]);  add_91 = None
        amax_12 = torch.ops.aten.amax.default(view_256, [-1], True)
        sub_38 = torch.ops.aten.sub.Tensor(view_256, amax_12);  view_256 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        bmm_25 = torch.ops.aten.bmm.default(div_12, view_254);  div_12 = view_254 = None
        view_257 = torch.ops.aten.view.default(bmm_25, [32, 16, 128, 64]);  bmm_25 = None
        permute_139 = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        clone_102 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_258 = torch.ops.aten.view.default(clone_102, [32, 128, 1024]);  clone_102 = None
        view_259 = torch.ops.aten.view.default(view_258, [4096, 1024]);  view_258 = None
        permute_140 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg208_1, view_259, permute_140);  arg208_1 = view_259 = permute_140 = None
        view_260 = torch.ops.aten.view.default(addmm_75, [32, 128, 1024]);  addmm_75 = None
        add_92 = torch.ops.aten.add.Tensor(add_88, view_260);  add_88 = view_260 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_92, getitem_53);  getitem_53 = None
        mul_103 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_103, arg209_1);  mul_103 = arg209_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_104, arg210_1);  mul_104 = arg210_1 = None
        view_261 = torch.ops.aten.view.default(add_94, [4096, 1024]);  add_94 = None
        permute_141 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg212_1, view_261, permute_141);  arg212_1 = view_261 = permute_141 = None
        view_262 = torch.ops.aten.view.default(addmm_76, [32, 128, 1024]);  addmm_76 = None
        mul_105 = torch.ops.aten.mul.Tensor(view_262, 0.125);  view_262 = None
        view_263 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_142 = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg214_1, view_263, permute_142);  arg214_1 = view_263 = permute_142 = None
        view_264 = torch.ops.aten.view.default(addmm_77, [32, 128, 1024]);  addmm_77 = None
        view_265 = torch.ops.aten.view.default(view_264, [32, -1, 16, 64]);  view_264 = None
        permute_143 = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_104 = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_266 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_144 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg216_1, view_266, permute_144);  arg216_1 = view_266 = permute_144 = None
        view_267 = torch.ops.aten.view.default(addmm_78, [32, 128, 1024]);  addmm_78 = None
        view_268 = torch.ops.aten.view.default(view_267, [32, -1, 16, 64]);  view_267 = None
        permute_145 = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_105 = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_269 = torch.ops.aten.view.default(mul_105, [32, 128, 16, 64]);  mul_105 = None
        permute_146 = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        clone_106 = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_270 = torch.ops.aten.view.default(clone_106, [512, -1, 64]);  clone_106 = None
        view_271 = torch.ops.aten.view.default(clone_104, [512, -1, 64]);  clone_104 = None
        view_272 = torch.ops.aten.view.default(clone_105, [512, -1, 64]);  clone_105 = None
        unsqueeze_default_33 = torch.ops.aten.unsqueeze.default(view_270, 0);  view_270 = None
        unsqueeze_default_34 = torch.ops.aten.unsqueeze.default(view_271, 0);  view_271 = None
        unsqueeze_default_35 = torch.ops.aten.unsqueeze.default(view_272, 0);  view_272 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, None, False, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
        getitem_135 = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
        squeeze_dim_11 = torch.ops.aten.squeeze.dim(getitem_135, 0);  getitem_135 = None
        view_273 = torch.ops.aten.view.default(squeeze_dim_11, [32, 16, 128, 64]);  squeeze_dim_11 = None
        permute_148 = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
        clone_108 = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_274 = torch.ops.aten.view.default(clone_108, [32, 128, 1024]);  clone_108 = None
        view_275 = torch.ops.aten.view.default(view_274, [4096, 1024]);  view_274 = None
        permute_149 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg218_1, view_275, permute_149);  arg218_1 = view_275 = permute_149 = None
        view_276 = torch.ops.aten.view.default(addmm_79, [32, 128, 1024]);  addmm_79 = None
        add_95 = torch.ops.aten.add.Tensor(add_92, view_276);  add_92 = view_276 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_95, getitem_55);  getitem_55 = None
        mul_106 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, arg219_1);  mul_106 = arg219_1 = None
        add_97 = torch.ops.aten.add.Tensor(mul_107, arg220_1);  mul_107 = arg220_1 = None
        view_277 = torch.ops.aten.view.default(add_97, [4096, 1024]);  add_97 = None
        permute_150 = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg222_1, view_277, permute_150);  arg222_1 = view_277 = permute_150 = None
        view_278 = torch.ops.aten.view.default(addmm_80, [32, 128, 4096]);  addmm_80 = None
        mul_108 = torch.ops.aten.mul.Tensor(view_278, 0.5)
        mul_109 = torch.ops.aten.mul.Tensor(view_278, 0.7071067811865476);  view_278 = None
        erf_12 = torch.ops.aten.erf.default(mul_109);  mul_109 = None
        add_98 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_108, add_98);  mul_108 = add_98 = None
        view_279 = torch.ops.aten.view.default(mul_110, [4096, 4096]);  mul_110 = None
        permute_151 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg224_1, view_279, permute_151);  arg224_1 = view_279 = permute_151 = None
        view_280 = torch.ops.aten.view.default(addmm_81, [32, 128, 1024]);  addmm_81 = None
        add_99 = torch.ops.aten.add.Tensor(add_95, view_280);  add_95 = view_280 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_100 = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_99, getitem_57);  getitem_57 = None
        mul_111 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_111, arg225_1);  mul_111 = arg225_1 = None
        add_101 = torch.ops.aten.add.Tensor(mul_112, arg226_1);  mul_112 = arg226_1 = None
        view_281 = torch.ops.aten.view.default(add_101, [4096, 1024])
        permute_152 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg228_1, view_281, permute_152);  arg228_1 = view_281 = permute_152 = None
        view_282 = torch.ops.aten.view.default(addmm_82, [32, 128, 1024]);  addmm_82 = None
        mul_113 = torch.ops.aten.mul.Tensor(view_282, 0.125);  view_282 = None
        view_283 = torch.ops.aten.view.default(add_101, [4096, 1024])
        permute_153 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg230_1, view_283, permute_153);  arg230_1 = view_283 = permute_153 = None
        view_284 = torch.ops.aten.view.default(addmm_83, [32, 128, 1024]);  addmm_83 = None
        view_285 = torch.ops.aten.view.default(view_284, [32, -1, 16, 64]);  view_284 = None
        permute_154 = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        clone_112 = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_286 = torch.ops.aten.view.default(add_101, [4096, 1024]);  add_101 = None
        permute_155 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg232_1, view_286, permute_155);  arg232_1 = view_286 = permute_155 = None
        view_287 = torch.ops.aten.view.default(addmm_84, [32, 128, 1024]);  addmm_84 = None
        view_288 = torch.ops.aten.view.default(view_287, [32, -1, 16, 64]);  view_287 = None
        permute_156 = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_113 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_289 = torch.ops.aten.view.default(mul_113, [32, 128, 16, 64]);  mul_113 = None
        permute_157 = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
        clone_114 = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_290 = torch.ops.aten.view.default(clone_114, [512, -1, 64]);  clone_114 = None
        view_291 = torch.ops.aten.view.default(clone_112, [512, -1, 64]);  clone_112 = None
        view_292 = torch.ops.aten.view.default(clone_113, [512, -1, 64]);  clone_113 = None
        permute_158 = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        bmm_28 = torch.ops.aten.bmm.default(view_290, permute_158);  view_290 = permute_158 = None
        view_293 = torch.ops.aten.view.default(bmm_28, [32, 16, 128, 128]);  bmm_28 = None
        add_102 = torch.ops.aten.add.Tensor(view_293, expand_1);  view_293 = None
        view_294 = torch.ops.aten.view.default(add_102, [512, 128, 128]);  add_102 = None
        amax_14 = torch.ops.aten.amax.default(view_294, [-1], True)
        sub_43 = torch.ops.aten.sub.Tensor(view_294, amax_14);  view_294 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        bmm_29 = torch.ops.aten.bmm.default(div_14, view_292);  div_14 = view_292 = None
        view_295 = torch.ops.aten.view.default(bmm_29, [32, 16, 128, 64]);  bmm_29 = None
        permute_159 = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
        clone_116 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_296 = torch.ops.aten.view.default(clone_116, [32, 128, 1024]);  clone_116 = None
        view_297 = torch.ops.aten.view.default(view_296, [4096, 1024]);  view_296 = None
        permute_160 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg234_1, view_297, permute_160);  arg234_1 = view_297 = permute_160 = None
        view_298 = torch.ops.aten.view.default(addmm_85, [32, 128, 1024]);  addmm_85 = None
        add_103 = torch.ops.aten.add.Tensor(add_99, view_298);  add_99 = view_298 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_104 = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_103, getitem_59);  getitem_59 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, arg235_1);  mul_114 = arg235_1 = None
        add_105 = torch.ops.aten.add.Tensor(mul_115, arg236_1);  mul_115 = arg236_1 = None
        view_299 = torch.ops.aten.view.default(add_105, [4096, 1024]);  add_105 = None
        permute_161 = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg238_1, view_299, permute_161);  arg238_1 = view_299 = permute_161 = None
        view_300 = torch.ops.aten.view.default(addmm_86, [32, 128, 1024]);  addmm_86 = None
        mul_116 = torch.ops.aten.mul.Tensor(view_300, 0.125);  view_300 = None
        view_301 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_162 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg240_1, view_301, permute_162);  arg240_1 = view_301 = permute_162 = None
        view_302 = torch.ops.aten.view.default(addmm_87, [32, 128, 1024]);  addmm_87 = None
        view_303 = torch.ops.aten.view.default(view_302, [32, -1, 16, 64]);  view_302 = None
        permute_163 = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        clone_118 = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        view_304 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_164 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg242_1, view_304, permute_164);  arg242_1 = view_304 = permute_164 = None
        view_305 = torch.ops.aten.view.default(addmm_88, [32, 128, 1024]);  addmm_88 = None
        view_306 = torch.ops.aten.view.default(view_305, [32, -1, 16, 64]);  view_305 = None
        permute_165 = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        clone_119 = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_307 = torch.ops.aten.view.default(mul_116, [32, 128, 16, 64]);  mul_116 = None
        permute_166 = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        clone_120 = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_308 = torch.ops.aten.view.default(clone_120, [512, -1, 64]);  clone_120 = None
        view_309 = torch.ops.aten.view.default(clone_118, [512, -1, 64]);  clone_118 = None
        view_310 = torch.ops.aten.view.default(clone_119, [512, -1, 64]);  clone_119 = None
        unsqueeze_default_30 = torch.ops.aten.unsqueeze.default(view_308, 0);  view_308 = None
        unsqueeze_default_31 = torch.ops.aten.unsqueeze.default(view_309, 0);  view_309 = None
        unsqueeze_default_32 = torch.ops.aten.unsqueeze.default(view_310, 0);  view_310 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, None, False, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
        getitem_134 = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
        squeeze_dim_10 = torch.ops.aten.squeeze.dim(getitem_134, 0);  getitem_134 = None
        view_311 = torch.ops.aten.view.default(squeeze_dim_10, [32, 16, 128, 64]);  squeeze_dim_10 = None
        permute_168 = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
        clone_122 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_312 = torch.ops.aten.view.default(clone_122, [32, 128, 1024]);  clone_122 = None
        view_313 = torch.ops.aten.view.default(view_312, [4096, 1024]);  view_312 = None
        permute_169 = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg244_1, view_313, permute_169);  arg244_1 = view_313 = permute_169 = None
        view_314 = torch.ops.aten.view.default(addmm_89, [32, 128, 1024]);  addmm_89 = None
        add_106 = torch.ops.aten.add.Tensor(add_103, view_314);  add_103 = view_314 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_107 = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_106, getitem_61);  getitem_61 = None
        mul_117 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_117, arg245_1);  mul_117 = arg245_1 = None
        add_108 = torch.ops.aten.add.Tensor(mul_118, arg246_1);  mul_118 = arg246_1 = None
        view_315 = torch.ops.aten.view.default(add_108, [4096, 1024]);  add_108 = None
        permute_170 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg248_1, view_315, permute_170);  arg248_1 = view_315 = permute_170 = None
        view_316 = torch.ops.aten.view.default(addmm_90, [32, 128, 4096]);  addmm_90 = None
        mul_119 = torch.ops.aten.mul.Tensor(view_316, 0.5)
        mul_120 = torch.ops.aten.mul.Tensor(view_316, 0.7071067811865476);  view_316 = None
        erf_13 = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_109 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_119, add_109);  mul_119 = add_109 = None
        view_317 = torch.ops.aten.view.default(mul_121, [4096, 4096]);  mul_121 = None
        permute_171 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg250_1, view_317, permute_171);  arg250_1 = view_317 = permute_171 = None
        view_318 = torch.ops.aten.view.default(addmm_91, [32, 128, 1024]);  addmm_91 = None
        add_110 = torch.ops.aten.add.Tensor(add_106, view_318);  add_106 = view_318 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_111 = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_110, getitem_63);  getitem_63 = None
        mul_122 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, arg251_1);  mul_122 = arg251_1 = None
        add_112 = torch.ops.aten.add.Tensor(mul_123, arg252_1);  mul_123 = arg252_1 = None
        view_319 = torch.ops.aten.view.default(add_112, [4096, 1024])
        permute_172 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg254_1, view_319, permute_172);  arg254_1 = view_319 = permute_172 = None
        view_320 = torch.ops.aten.view.default(addmm_92, [32, 128, 1024]);  addmm_92 = None
        mul_124 = torch.ops.aten.mul.Tensor(view_320, 0.125);  view_320 = None
        view_321 = torch.ops.aten.view.default(add_112, [4096, 1024])
        permute_173 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg256_1, view_321, permute_173);  arg256_1 = view_321 = permute_173 = None
        view_322 = torch.ops.aten.view.default(addmm_93, [32, 128, 1024]);  addmm_93 = None
        view_323 = torch.ops.aten.view.default(view_322, [32, -1, 16, 64]);  view_322 = None
        permute_174 = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        clone_126 = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        view_324 = torch.ops.aten.view.default(add_112, [4096, 1024]);  add_112 = None
        permute_175 = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg258_1, view_324, permute_175);  arg258_1 = view_324 = permute_175 = None
        view_325 = torch.ops.aten.view.default(addmm_94, [32, 128, 1024]);  addmm_94 = None
        view_326 = torch.ops.aten.view.default(view_325, [32, -1, 16, 64]);  view_325 = None
        permute_176 = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        clone_127 = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_327 = torch.ops.aten.view.default(mul_124, [32, 128, 16, 64]);  mul_124 = None
        permute_177 = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        clone_128 = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_328 = torch.ops.aten.view.default(clone_128, [512, -1, 64]);  clone_128 = None
        view_329 = torch.ops.aten.view.default(clone_126, [512, -1, 64]);  clone_126 = None
        view_330 = torch.ops.aten.view.default(clone_127, [512, -1, 64]);  clone_127 = None
        permute_178 = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
        bmm_32 = torch.ops.aten.bmm.default(view_328, permute_178);  view_328 = permute_178 = None
        view_331 = torch.ops.aten.view.default(bmm_32, [32, 16, 128, 128]);  bmm_32 = None
        add_113 = torch.ops.aten.add.Tensor(view_331, expand_1);  view_331 = None
        view_332 = torch.ops.aten.view.default(add_113, [512, 128, 128]);  add_113 = None
        amax_16 = torch.ops.aten.amax.default(view_332, [-1], True)
        sub_48 = torch.ops.aten.sub.Tensor(view_332, amax_16);  view_332 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        bmm_33 = torch.ops.aten.bmm.default(div_16, view_330);  div_16 = view_330 = None
        view_333 = torch.ops.aten.view.default(bmm_33, [32, 16, 128, 64]);  bmm_33 = None
        permute_179 = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        clone_130 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_334 = torch.ops.aten.view.default(clone_130, [32, 128, 1024]);  clone_130 = None
        view_335 = torch.ops.aten.view.default(view_334, [4096, 1024]);  view_334 = None
        permute_180 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg260_1, view_335, permute_180);  arg260_1 = view_335 = permute_180 = None
        view_336 = torch.ops.aten.view.default(addmm_95, [32, 128, 1024]);  addmm_95 = None
        add_114 = torch.ops.aten.add.Tensor(add_110, view_336);  add_110 = view_336 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_115 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_114, getitem_65);  getitem_65 = None
        mul_125 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_125, arg261_1);  mul_125 = arg261_1 = None
        add_116 = torch.ops.aten.add.Tensor(mul_126, arg262_1);  mul_126 = arg262_1 = None
        view_337 = torch.ops.aten.view.default(add_116, [4096, 1024]);  add_116 = None
        permute_181 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg264_1, view_337, permute_181);  arg264_1 = view_337 = permute_181 = None
        view_338 = torch.ops.aten.view.default(addmm_96, [32, 128, 1024]);  addmm_96 = None
        mul_127 = torch.ops.aten.mul.Tensor(view_338, 0.125);  view_338 = None
        view_339 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_182 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg266_1, view_339, permute_182);  arg266_1 = view_339 = permute_182 = None
        view_340 = torch.ops.aten.view.default(addmm_97, [32, 128, 1024]);  addmm_97 = None
        view_341 = torch.ops.aten.view.default(view_340, [32, -1, 16, 64]);  view_340 = None
        permute_183 = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
        clone_132 = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_342 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_184 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg268_1, view_342, permute_184);  arg268_1 = view_342 = permute_184 = None
        view_343 = torch.ops.aten.view.default(addmm_98, [32, 128, 1024]);  addmm_98 = None
        view_344 = torch.ops.aten.view.default(view_343, [32, -1, 16, 64]);  view_343 = None
        permute_185 = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_133 = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_345 = torch.ops.aten.view.default(mul_127, [32, 128, 16, 64]);  mul_127 = None
        permute_186 = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_134 = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_346 = torch.ops.aten.view.default(clone_134, [512, -1, 64]);  clone_134 = None
        view_347 = torch.ops.aten.view.default(clone_132, [512, -1, 64]);  clone_132 = None
        view_348 = torch.ops.aten.view.default(clone_133, [512, -1, 64]);  clone_133 = None
        unsqueeze_default_27 = torch.ops.aten.unsqueeze.default(view_346, 0);  view_346 = None
        unsqueeze_default_28 = torch.ops.aten.unsqueeze.default(view_347, 0);  view_347 = None
        unsqueeze_default_29 = torch.ops.aten.unsqueeze.default(view_348, 0);  view_348 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, None, False, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
        getitem_133 = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
        squeeze_dim_9 = torch.ops.aten.squeeze.dim(getitem_133, 0);  getitem_133 = None
        view_349 = torch.ops.aten.view.default(squeeze_dim_9, [32, 16, 128, 64]);  squeeze_dim_9 = None
        permute_188 = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
        clone_136 = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_350 = torch.ops.aten.view.default(clone_136, [32, 128, 1024]);  clone_136 = None
        view_351 = torch.ops.aten.view.default(view_350, [4096, 1024]);  view_350 = None
        permute_189 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg270_1, view_351, permute_189);  arg270_1 = view_351 = permute_189 = None
        view_352 = torch.ops.aten.view.default(addmm_99, [32, 128, 1024]);  addmm_99 = None
        add_117 = torch.ops.aten.add.Tensor(add_114, view_352);  add_114 = view_352 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_117, getitem_67);  getitem_67 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = rsqrt_33 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, arg271_1);  mul_128 = arg271_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_129, arg272_1);  mul_129 = arg272_1 = None
        view_353 = torch.ops.aten.view.default(add_119, [4096, 1024]);  add_119 = None
        permute_190 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg274_1, view_353, permute_190);  arg274_1 = view_353 = permute_190 = None
        view_354 = torch.ops.aten.view.default(addmm_100, [32, 128, 4096]);  addmm_100 = None
        mul_130 = torch.ops.aten.mul.Tensor(view_354, 0.5)
        mul_131 = torch.ops.aten.mul.Tensor(view_354, 0.7071067811865476);  view_354 = None
        erf_14 = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_120 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_130, add_120);  mul_130 = add_120 = None
        view_355 = torch.ops.aten.view.default(mul_132, [4096, 4096]);  mul_132 = None
        permute_191 = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg276_1, view_355, permute_191);  arg276_1 = view_355 = permute_191 = None
        view_356 = torch.ops.aten.view.default(addmm_101, [32, 128, 1024]);  addmm_101 = None
        add_121 = torch.ops.aten.add.Tensor(add_117, view_356);  add_117 = view_356 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_121, getitem_69);  getitem_69 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, arg277_1);  mul_133 = arg277_1 = None
        add_123 = torch.ops.aten.add.Tensor(mul_134, arg278_1);  mul_134 = arg278_1 = None
        view_357 = torch.ops.aten.view.default(add_123, [4096, 1024])
        permute_192 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg280_1, view_357, permute_192);  arg280_1 = view_357 = permute_192 = None
        view_358 = torch.ops.aten.view.default(addmm_102, [32, 128, 1024]);  addmm_102 = None
        mul_135 = torch.ops.aten.mul.Tensor(view_358, 0.125);  view_358 = None
        view_359 = torch.ops.aten.view.default(add_123, [4096, 1024])
        permute_193 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg282_1, view_359, permute_193);  arg282_1 = view_359 = permute_193 = None
        view_360 = torch.ops.aten.view.default(addmm_103, [32, 128, 1024]);  addmm_103 = None
        view_361 = torch.ops.aten.view.default(view_360, [32, -1, 16, 64]);  view_360 = None
        permute_194 = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
        clone_140 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_362 = torch.ops.aten.view.default(add_123, [4096, 1024]);  add_123 = None
        permute_195 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg284_1, view_362, permute_195);  arg284_1 = view_362 = permute_195 = None
        view_363 = torch.ops.aten.view.default(addmm_104, [32, 128, 1024]);  addmm_104 = None
        view_364 = torch.ops.aten.view.default(view_363, [32, -1, 16, 64]);  view_363 = None
        permute_196 = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        clone_141 = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_365 = torch.ops.aten.view.default(mul_135, [32, 128, 16, 64]);  mul_135 = None
        permute_197 = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
        clone_142 = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_366 = torch.ops.aten.view.default(clone_142, [512, -1, 64]);  clone_142 = None
        view_367 = torch.ops.aten.view.default(clone_140, [512, -1, 64]);  clone_140 = None
        view_368 = torch.ops.aten.view.default(clone_141, [512, -1, 64]);  clone_141 = None
        permute_198 = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
        bmm_36 = torch.ops.aten.bmm.default(view_366, permute_198);  view_366 = permute_198 = None
        view_369 = torch.ops.aten.view.default(bmm_36, [32, 16, 128, 128]);  bmm_36 = None
        add_124 = torch.ops.aten.add.Tensor(view_369, expand_1);  view_369 = None
        view_370 = torch.ops.aten.view.default(add_124, [512, 128, 128]);  add_124 = None
        amax_18 = torch.ops.aten.amax.default(view_370, [-1], True)
        sub_53 = torch.ops.aten.sub.Tensor(view_370, amax_18);  view_370 = amax_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        bmm_37 = torch.ops.aten.bmm.default(div_18, view_368);  div_18 = view_368 = None
        view_371 = torch.ops.aten.view.default(bmm_37, [32, 16, 128, 64]);  bmm_37 = None
        permute_199 = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        clone_144 = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_372 = torch.ops.aten.view.default(clone_144, [32, 128, 1024]);  clone_144 = None
        view_373 = torch.ops.aten.view.default(view_372, [4096, 1024]);  view_372 = None
        permute_200 = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg286_1, view_373, permute_200);  arg286_1 = view_373 = permute_200 = None
        view_374 = torch.ops.aten.view.default(addmm_105, [32, 128, 1024]);  addmm_105 = None
        add_125 = torch.ops.aten.add.Tensor(add_121, view_374);  add_121 = view_374 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_125, getitem_71);  getitem_71 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = rsqrt_35 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, arg287_1);  mul_136 = arg287_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_137, arg288_1);  mul_137 = arg288_1 = None
        view_375 = torch.ops.aten.view.default(add_127, [4096, 1024]);  add_127 = None
        permute_201 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg290_1, view_375, permute_201);  arg290_1 = view_375 = permute_201 = None
        view_376 = torch.ops.aten.view.default(addmm_106, [32, 128, 1024]);  addmm_106 = None
        mul_138 = torch.ops.aten.mul.Tensor(view_376, 0.125);  view_376 = None
        view_377 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_202 = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg292_1, view_377, permute_202);  arg292_1 = view_377 = permute_202 = None
        view_378 = torch.ops.aten.view.default(addmm_107, [32, 128, 1024]);  addmm_107 = None
        view_379 = torch.ops.aten.view.default(view_378, [32, -1, 16, 64]);  view_378 = None
        permute_203 = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        clone_146 = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_380 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_204 = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg294_1, view_380, permute_204);  arg294_1 = view_380 = permute_204 = None
        view_381 = torch.ops.aten.view.default(addmm_108, [32, 128, 1024]);  addmm_108 = None
        view_382 = torch.ops.aten.view.default(view_381, [32, -1, 16, 64]);  view_381 = None
        permute_205 = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_147 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_383 = torch.ops.aten.view.default(mul_138, [32, 128, 16, 64]);  mul_138 = None
        permute_206 = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_148 = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_384 = torch.ops.aten.view.default(clone_148, [512, -1, 64]);  clone_148 = None
        view_385 = torch.ops.aten.view.default(clone_146, [512, -1, 64]);  clone_146 = None
        view_386 = torch.ops.aten.view.default(clone_147, [512, -1, 64]);  clone_147 = None
        unsqueeze_default_24 = torch.ops.aten.unsqueeze.default(view_384, 0);  view_384 = None
        unsqueeze_default_25 = torch.ops.aten.unsqueeze.default(view_385, 0);  view_385 = None
        unsqueeze_default_26 = torch.ops.aten.unsqueeze.default(view_386, 0);  view_386 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, None, False, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
        getitem_132 = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
        squeeze_dim_8 = torch.ops.aten.squeeze.dim(getitem_132, 0);  getitem_132 = None
        view_387 = torch.ops.aten.view.default(squeeze_dim_8, [32, 16, 128, 64]);  squeeze_dim_8 = None
        permute_208 = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        clone_150 = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_388 = torch.ops.aten.view.default(clone_150, [32, 128, 1024]);  clone_150 = None
        view_389 = torch.ops.aten.view.default(view_388, [4096, 1024]);  view_388 = None
        permute_209 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg296_1, view_389, permute_209);  arg296_1 = view_389 = permute_209 = None
        view_390 = torch.ops.aten.view.default(addmm_109, [32, 128, 1024]);  addmm_109 = None
        add_128 = torch.ops.aten.add.Tensor(add_125, view_390);  add_125 = view_390 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_129 = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_128, getitem_73);  getitem_73 = None
        mul_139 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_36);  sub_56 = rsqrt_36 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, arg297_1);  mul_139 = arg297_1 = None
        add_130 = torch.ops.aten.add.Tensor(mul_140, arg298_1);  mul_140 = arg298_1 = None
        view_391 = torch.ops.aten.view.default(add_130, [4096, 1024]);  add_130 = None
        permute_210 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg300_1, view_391, permute_210);  arg300_1 = view_391 = permute_210 = None
        view_392 = torch.ops.aten.view.default(addmm_110, [32, 128, 4096]);  addmm_110 = None
        mul_141 = torch.ops.aten.mul.Tensor(view_392, 0.5)
        mul_142 = torch.ops.aten.mul.Tensor(view_392, 0.7071067811865476);  view_392 = None
        erf_15 = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_131 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_141, add_131);  mul_141 = add_131 = None
        view_393 = torch.ops.aten.view.default(mul_143, [4096, 4096]);  mul_143 = None
        permute_211 = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg302_1, view_393, permute_211);  arg302_1 = view_393 = permute_211 = None
        view_394 = torch.ops.aten.view.default(addmm_111, [32, 128, 1024]);  addmm_111 = None
        add_132 = torch.ops.aten.add.Tensor(add_128, view_394);  add_128 = view_394 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_133 = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_132, getitem_75);  getitem_75 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = rsqrt_37 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, arg303_1);  mul_144 = arg303_1 = None
        add_134 = torch.ops.aten.add.Tensor(mul_145, arg304_1);  mul_145 = arg304_1 = None
        view_395 = torch.ops.aten.view.default(add_134, [4096, 1024])
        permute_212 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg306_1, view_395, permute_212);  arg306_1 = view_395 = permute_212 = None
        view_396 = torch.ops.aten.view.default(addmm_112, [32, 128, 1024]);  addmm_112 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_396, 0.125);  view_396 = None
        view_397 = torch.ops.aten.view.default(add_134, [4096, 1024])
        permute_213 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg308_1, view_397, permute_213);  arg308_1 = view_397 = permute_213 = None
        view_398 = torch.ops.aten.view.default(addmm_113, [32, 128, 1024]);  addmm_113 = None
        view_399 = torch.ops.aten.view.default(view_398, [32, -1, 16, 64]);  view_398 = None
        permute_214 = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_154 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_400 = torch.ops.aten.view.default(add_134, [4096, 1024]);  add_134 = None
        permute_215 = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg310_1, view_400, permute_215);  arg310_1 = view_400 = permute_215 = None
        view_401 = torch.ops.aten.view.default(addmm_114, [32, 128, 1024]);  addmm_114 = None
        view_402 = torch.ops.aten.view.default(view_401, [32, -1, 16, 64]);  view_401 = None
        permute_216 = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_155 = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_403 = torch.ops.aten.view.default(mul_146, [32, 128, 16, 64]);  mul_146 = None
        permute_217 = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        clone_156 = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_404 = torch.ops.aten.view.default(clone_156, [512, -1, 64]);  clone_156 = None
        view_405 = torch.ops.aten.view.default(clone_154, [512, -1, 64]);  clone_154 = None
        view_406 = torch.ops.aten.view.default(clone_155, [512, -1, 64]);  clone_155 = None
        permute_218 = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        bmm_40 = torch.ops.aten.bmm.default(view_404, permute_218);  view_404 = permute_218 = None
        view_407 = torch.ops.aten.view.default(bmm_40, [32, 16, 128, 128]);  bmm_40 = None
        add_135 = torch.ops.aten.add.Tensor(view_407, expand_1);  view_407 = None
        view_408 = torch.ops.aten.view.default(add_135, [512, 128, 128]);  add_135 = None
        amax_20 = torch.ops.aten.amax.default(view_408, [-1], True)
        sub_58 = torch.ops.aten.sub.Tensor(view_408, amax_20);  view_408 = amax_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20 = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        bmm_41 = torch.ops.aten.bmm.default(div_20, view_406);  div_20 = view_406 = None
        view_409 = torch.ops.aten.view.default(bmm_41, [32, 16, 128, 64]);  bmm_41 = None
        permute_219 = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
        clone_158 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_410 = torch.ops.aten.view.default(clone_158, [32, 128, 1024]);  clone_158 = None
        view_411 = torch.ops.aten.view.default(view_410, [4096, 1024]);  view_410 = None
        permute_220 = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg312_1, view_411, permute_220);  arg312_1 = view_411 = permute_220 = None
        view_412 = torch.ops.aten.view.default(addmm_115, [32, 128, 1024]);  addmm_115 = None
        add_136 = torch.ops.aten.add.Tensor(add_132, view_412);  add_132 = view_412 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_137 = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_136, getitem_77);  getitem_77 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = rsqrt_38 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_147, arg313_1);  mul_147 = arg313_1 = None
        add_138 = torch.ops.aten.add.Tensor(mul_148, arg314_1);  mul_148 = arg314_1 = None
        view_413 = torch.ops.aten.view.default(add_138, [4096, 1024]);  add_138 = None
        permute_221 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg316_1, view_413, permute_221);  arg316_1 = view_413 = permute_221 = None
        view_414 = torch.ops.aten.view.default(addmm_116, [32, 128, 1024]);  addmm_116 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_414, 0.125);  view_414 = None
        view_415 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_222 = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg318_1, view_415, permute_222);  arg318_1 = view_415 = permute_222 = None
        view_416 = torch.ops.aten.view.default(addmm_117, [32, 128, 1024]);  addmm_117 = None
        view_417 = torch.ops.aten.view.default(view_416, [32, -1, 16, 64]);  view_416 = None
        permute_223 = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
        clone_160 = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_418 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_224 = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg320_1, view_418, permute_224);  arg320_1 = view_418 = permute_224 = None
        view_419 = torch.ops.aten.view.default(addmm_118, [32, 128, 1024]);  addmm_118 = None
        view_420 = torch.ops.aten.view.default(view_419, [32, -1, 16, 64]);  view_419 = None
        permute_225 = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_161 = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_421 = torch.ops.aten.view.default(mul_149, [32, 128, 16, 64]);  mul_149 = None
        permute_226 = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
        clone_162 = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_422 = torch.ops.aten.view.default(clone_162, [512, -1, 64]);  clone_162 = None
        view_423 = torch.ops.aten.view.default(clone_160, [512, -1, 64]);  clone_160 = None
        view_424 = torch.ops.aten.view.default(clone_161, [512, -1, 64]);  clone_161 = None
        unsqueeze_default_21 = torch.ops.aten.unsqueeze.default(view_422, 0);  view_422 = None
        unsqueeze_default_22 = torch.ops.aten.unsqueeze.default(view_423, 0);  view_423 = None
        unsqueeze_default_23 = torch.ops.aten.unsqueeze.default(view_424, 0);  view_424 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, None, False, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
        getitem_131 = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
        squeeze_dim_7 = torch.ops.aten.squeeze.dim(getitem_131, 0);  getitem_131 = None
        view_425 = torch.ops.aten.view.default(squeeze_dim_7, [32, 16, 128, 64]);  squeeze_dim_7 = None
        permute_228 = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
        clone_164 = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_426 = torch.ops.aten.view.default(clone_164, [32, 128, 1024]);  clone_164 = None
        view_427 = torch.ops.aten.view.default(view_426, [4096, 1024]);  view_426 = None
        permute_229 = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg322_1, view_427, permute_229);  arg322_1 = view_427 = permute_229 = None
        view_428 = torch.ops.aten.view.default(addmm_119, [32, 128, 1024]);  addmm_119 = None
        add_139 = torch.ops.aten.add.Tensor(add_136, view_428);  add_136 = view_428 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_140 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_61 = torch.ops.aten.sub.Tensor(add_139, getitem_79);  getitem_79 = None
        mul_150 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_39);  sub_61 = rsqrt_39 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, arg323_1);  mul_150 = arg323_1 = None
        add_141 = torch.ops.aten.add.Tensor(mul_151, arg324_1);  mul_151 = arg324_1 = None
        view_429 = torch.ops.aten.view.default(add_141, [4096, 1024]);  add_141 = None
        permute_230 = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg326_1, view_429, permute_230);  arg326_1 = view_429 = permute_230 = None
        view_430 = torch.ops.aten.view.default(addmm_120, [32, 128, 4096]);  addmm_120 = None
        mul_152 = torch.ops.aten.mul.Tensor(view_430, 0.5)
        mul_153 = torch.ops.aten.mul.Tensor(view_430, 0.7071067811865476);  view_430 = None
        erf_16 = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_142 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_152, add_142);  mul_152 = add_142 = None
        view_431 = torch.ops.aten.view.default(mul_154, [4096, 4096]);  mul_154 = None
        permute_231 = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg328_1, view_431, permute_231);  arg328_1 = view_431 = permute_231 = None
        view_432 = torch.ops.aten.view.default(addmm_121, [32, 128, 1024]);  addmm_121 = None
        add_143 = torch.ops.aten.add.Tensor(add_139, view_432);  add_139 = view_432 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_144 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_143, getitem_81);  getitem_81 = None
        mul_155 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = rsqrt_40 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, arg329_1);  mul_155 = arg329_1 = None
        add_145 = torch.ops.aten.add.Tensor(mul_156, arg330_1);  mul_156 = arg330_1 = None
        view_433 = torch.ops.aten.view.default(add_145, [4096, 1024])
        permute_232 = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg332_1, view_433, permute_232);  arg332_1 = view_433 = permute_232 = None
        view_434 = torch.ops.aten.view.default(addmm_122, [32, 128, 1024]);  addmm_122 = None
        mul_157 = torch.ops.aten.mul.Tensor(view_434, 0.125);  view_434 = None
        view_435 = torch.ops.aten.view.default(add_145, [4096, 1024])
        permute_233 = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg334_1, view_435, permute_233);  arg334_1 = view_435 = permute_233 = None
        view_436 = torch.ops.aten.view.default(addmm_123, [32, 128, 1024]);  addmm_123 = None
        view_437 = torch.ops.aten.view.default(view_436, [32, -1, 16, 64]);  view_436 = None
        permute_234 = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
        clone_168 = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_438 = torch.ops.aten.view.default(add_145, [4096, 1024]);  add_145 = None
        permute_235 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg336_1, view_438, permute_235);  arg336_1 = view_438 = permute_235 = None
        view_439 = torch.ops.aten.view.default(addmm_124, [32, 128, 1024]);  addmm_124 = None
        view_440 = torch.ops.aten.view.default(view_439, [32, -1, 16, 64]);  view_439 = None
        permute_236 = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
        clone_169 = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_441 = torch.ops.aten.view.default(mul_157, [32, 128, 16, 64]);  mul_157 = None
        permute_237 = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
        clone_170 = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_442 = torch.ops.aten.view.default(clone_170, [512, -1, 64]);  clone_170 = None
        view_443 = torch.ops.aten.view.default(clone_168, [512, -1, 64]);  clone_168 = None
        view_444 = torch.ops.aten.view.default(clone_169, [512, -1, 64]);  clone_169 = None
        permute_238 = torch.ops.aten.permute.default(view_443, [0, 2, 1]);  view_443 = None
        bmm_44 = torch.ops.aten.bmm.default(view_442, permute_238);  view_442 = permute_238 = None
        view_445 = torch.ops.aten.view.default(bmm_44, [32, 16, 128, 128]);  bmm_44 = None
        add_146 = torch.ops.aten.add.Tensor(view_445, expand_1);  view_445 = None
        view_446 = torch.ops.aten.view.default(add_146, [512, 128, 128]);  add_146 = None
        amax_22 = torch.ops.aten.amax.default(view_446, [-1], True)
        sub_63 = torch.ops.aten.sub.Tensor(view_446, amax_22);  view_446 = amax_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22 = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        bmm_45 = torch.ops.aten.bmm.default(div_22, view_444);  div_22 = view_444 = None
        view_447 = torch.ops.aten.view.default(bmm_45, [32, 16, 128, 64]);  bmm_45 = None
        permute_239 = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        clone_172 = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_448 = torch.ops.aten.view.default(clone_172, [32, 128, 1024]);  clone_172 = None
        view_449 = torch.ops.aten.view.default(view_448, [4096, 1024]);  view_448 = None
        permute_240 = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg338_1, view_449, permute_240);  arg338_1 = view_449 = permute_240 = None
        view_450 = torch.ops.aten.view.default(addmm_125, [32, 128, 1024]);  addmm_125 = None
        add_147 = torch.ops.aten.add.Tensor(add_143, view_450);  add_143 = view_450 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_148 = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_64 = torch.ops.aten.sub.Tensor(add_147, getitem_83);  getitem_83 = None
        mul_158 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = rsqrt_41 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_158, arg339_1);  mul_158 = arg339_1 = None
        add_149 = torch.ops.aten.add.Tensor(mul_159, arg340_1);  mul_159 = arg340_1 = None
        view_451 = torch.ops.aten.view.default(add_149, [4096, 1024]);  add_149 = None
        permute_241 = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg342_1, view_451, permute_241);  arg342_1 = view_451 = permute_241 = None
        view_452 = torch.ops.aten.view.default(addmm_126, [32, 128, 1024]);  addmm_126 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_452, 0.125);  view_452 = None
        view_453 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_242 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg344_1, view_453, permute_242);  arg344_1 = view_453 = permute_242 = None
        view_454 = torch.ops.aten.view.default(addmm_127, [32, 128, 1024]);  addmm_127 = None
        view_455 = torch.ops.aten.view.default(view_454, [32, -1, 16, 64]);  view_454 = None
        permute_243 = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        clone_174 = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_456 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_244 = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg346_1, view_456, permute_244);  arg346_1 = view_456 = permute_244 = None
        view_457 = torch.ops.aten.view.default(addmm_128, [32, 128, 1024]);  addmm_128 = None
        view_458 = torch.ops.aten.view.default(view_457, [32, -1, 16, 64]);  view_457 = None
        permute_245 = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_175 = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_459 = torch.ops.aten.view.default(mul_160, [32, 128, 16, 64]);  mul_160 = None
        permute_246 = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        clone_176 = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_460 = torch.ops.aten.view.default(clone_176, [512, -1, 64]);  clone_176 = None
        view_461 = torch.ops.aten.view.default(clone_174, [512, -1, 64]);  clone_174 = None
        view_462 = torch.ops.aten.view.default(clone_175, [512, -1, 64]);  clone_175 = None
        unsqueeze_default_18 = torch.ops.aten.unsqueeze.default(view_460, 0);  view_460 = None
        unsqueeze_default_19 = torch.ops.aten.unsqueeze.default(view_461, 0);  view_461 = None
        unsqueeze_default_20 = torch.ops.aten.unsqueeze.default(view_462, 0);  view_462 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, None, False, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
        getitem_130 = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
        squeeze_dim_6 = torch.ops.aten.squeeze.dim(getitem_130, 0);  getitem_130 = None
        view_463 = torch.ops.aten.view.default(squeeze_dim_6, [32, 16, 128, 64]);  squeeze_dim_6 = None
        permute_248 = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
        clone_178 = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_464 = torch.ops.aten.view.default(clone_178, [32, 128, 1024]);  clone_178 = None
        view_465 = torch.ops.aten.view.default(view_464, [4096, 1024]);  view_464 = None
        permute_249 = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg348_1, view_465, permute_249);  arg348_1 = view_465 = permute_249 = None
        view_466 = torch.ops.aten.view.default(addmm_129, [32, 128, 1024]);  addmm_129 = None
        add_150 = torch.ops.aten.add.Tensor(add_147, view_466);  add_147 = view_466 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_151 = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_66 = torch.ops.aten.sub.Tensor(add_150, getitem_85);  getitem_85 = None
        mul_161 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_42);  sub_66 = rsqrt_42 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_161, arg349_1);  mul_161 = arg349_1 = None
        add_152 = torch.ops.aten.add.Tensor(mul_162, arg350_1);  mul_162 = arg350_1 = None
        view_467 = torch.ops.aten.view.default(add_152, [4096, 1024]);  add_152 = None
        permute_250 = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg352_1, view_467, permute_250);  arg352_1 = view_467 = permute_250 = None
        view_468 = torch.ops.aten.view.default(addmm_130, [32, 128, 4096]);  addmm_130 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_468, 0.5)
        mul_164 = torch.ops.aten.mul.Tensor(view_468, 0.7071067811865476);  view_468 = None
        erf_17 = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_153 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_163, add_153);  mul_163 = add_153 = None
        view_469 = torch.ops.aten.view.default(mul_165, [4096, 4096]);  mul_165 = None
        permute_251 = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg354_1, view_469, permute_251);  arg354_1 = view_469 = permute_251 = None
        view_470 = torch.ops.aten.view.default(addmm_131, [32, 128, 1024]);  addmm_131 = None
        add_154 = torch.ops.aten.add.Tensor(add_150, view_470);  add_150 = view_470 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_155 = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_67 = torch.ops.aten.sub.Tensor(add_154, getitem_87);  getitem_87 = None
        mul_166 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = rsqrt_43 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, arg355_1);  mul_166 = arg355_1 = None
        add_156 = torch.ops.aten.add.Tensor(mul_167, arg356_1);  mul_167 = arg356_1 = None
        view_471 = torch.ops.aten.view.default(add_156, [4096, 1024])
        permute_252 = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg358_1, view_471, permute_252);  arg358_1 = view_471 = permute_252 = None
        view_472 = torch.ops.aten.view.default(addmm_132, [32, 128, 1024]);  addmm_132 = None
        mul_168 = torch.ops.aten.mul.Tensor(view_472, 0.125);  view_472 = None
        view_473 = torch.ops.aten.view.default(add_156, [4096, 1024])
        permute_253 = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg360_1, view_473, permute_253);  arg360_1 = view_473 = permute_253 = None
        view_474 = torch.ops.aten.view.default(addmm_133, [32, 128, 1024]);  addmm_133 = None
        view_475 = torch.ops.aten.view.default(view_474, [32, -1, 16, 64]);  view_474 = None
        permute_254 = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
        clone_182 = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_476 = torch.ops.aten.view.default(add_156, [4096, 1024]);  add_156 = None
        permute_255 = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg362_1, view_476, permute_255);  arg362_1 = view_476 = permute_255 = None
        view_477 = torch.ops.aten.view.default(addmm_134, [32, 128, 1024]);  addmm_134 = None
        view_478 = torch.ops.aten.view.default(view_477, [32, -1, 16, 64]);  view_477 = None
        permute_256 = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        clone_183 = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_479 = torch.ops.aten.view.default(mul_168, [32, 128, 16, 64]);  mul_168 = None
        permute_257 = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        clone_184 = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_480 = torch.ops.aten.view.default(clone_184, [512, -1, 64]);  clone_184 = None
        view_481 = torch.ops.aten.view.default(clone_182, [512, -1, 64]);  clone_182 = None
        view_482 = torch.ops.aten.view.default(clone_183, [512, -1, 64]);  clone_183 = None
        permute_258 = torch.ops.aten.permute.default(view_481, [0, 2, 1]);  view_481 = None
        bmm_48 = torch.ops.aten.bmm.default(view_480, permute_258);  view_480 = permute_258 = None
        view_483 = torch.ops.aten.view.default(bmm_48, [32, 16, 128, 128]);  bmm_48 = None
        add_157 = torch.ops.aten.add.Tensor(view_483, expand_1);  view_483 = None
        view_484 = torch.ops.aten.view.default(add_157, [512, 128, 128]);  add_157 = None
        amax_24 = torch.ops.aten.amax.default(view_484, [-1], True)
        sub_68 = torch.ops.aten.sub.Tensor(view_484, amax_24);  view_484 = amax_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        bmm_49 = torch.ops.aten.bmm.default(div_24, view_482);  div_24 = view_482 = None
        view_485 = torch.ops.aten.view.default(bmm_49, [32, 16, 128, 64]);  bmm_49 = None
        permute_259 = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
        clone_186 = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_486 = torch.ops.aten.view.default(clone_186, [32, 128, 1024]);  clone_186 = None
        view_487 = torch.ops.aten.view.default(view_486, [4096, 1024]);  view_486 = None
        permute_260 = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg364_1, view_487, permute_260);  arg364_1 = view_487 = permute_260 = None
        view_488 = torch.ops.aten.view.default(addmm_135, [32, 128, 1024]);  addmm_135 = None
        add_158 = torch.ops.aten.add.Tensor(add_154, view_488);  add_154 = view_488 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_69 = torch.ops.aten.sub.Tensor(add_158, getitem_89);  getitem_89 = None
        mul_169 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_44);  sub_69 = rsqrt_44 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, arg365_1);  mul_169 = arg365_1 = None
        add_160 = torch.ops.aten.add.Tensor(mul_170, arg366_1);  mul_170 = arg366_1 = None
        view_489 = torch.ops.aten.view.default(add_160, [4096, 1024]);  add_160 = None
        permute_261 = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg368_1, view_489, permute_261);  arg368_1 = view_489 = permute_261 = None
        view_490 = torch.ops.aten.view.default(addmm_136, [32, 128, 1024]);  addmm_136 = None
        mul_171 = torch.ops.aten.mul.Tensor(view_490, 0.125);  view_490 = None
        view_491 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_262 = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg370_1, view_491, permute_262);  arg370_1 = view_491 = permute_262 = None
        view_492 = torch.ops.aten.view.default(addmm_137, [32, 128, 1024]);  addmm_137 = None
        view_493 = torch.ops.aten.view.default(view_492, [32, -1, 16, 64]);  view_492 = None
        permute_263 = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
        clone_188 = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_494 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_264 = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg372_1, view_494, permute_264);  arg372_1 = view_494 = permute_264 = None
        view_495 = torch.ops.aten.view.default(addmm_138, [32, 128, 1024]);  addmm_138 = None
        view_496 = torch.ops.aten.view.default(view_495, [32, -1, 16, 64]);  view_495 = None
        permute_265 = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        clone_189 = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
        view_497 = torch.ops.aten.view.default(mul_171, [32, 128, 16, 64]);  mul_171 = None
        permute_266 = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
        clone_190 = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_498 = torch.ops.aten.view.default(clone_190, [512, -1, 64]);  clone_190 = None
        view_499 = torch.ops.aten.view.default(clone_188, [512, -1, 64]);  clone_188 = None
        view_500 = torch.ops.aten.view.default(clone_189, [512, -1, 64]);  clone_189 = None
        unsqueeze_default_15 = torch.ops.aten.unsqueeze.default(view_498, 0);  view_498 = None
        unsqueeze_default_16 = torch.ops.aten.unsqueeze.default(view_499, 0);  view_499 = None
        unsqueeze_default_17 = torch.ops.aten.unsqueeze.default(view_500, 0);  view_500 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, None, False, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
        getitem_129 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
        squeeze_dim_5 = torch.ops.aten.squeeze.dim(getitem_129, 0);  getitem_129 = None
        view_501 = torch.ops.aten.view.default(squeeze_dim_5, [32, 16, 128, 64]);  squeeze_dim_5 = None
        permute_268 = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
        clone_192 = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
        view_502 = torch.ops.aten.view.default(clone_192, [32, 128, 1024]);  clone_192 = None
        view_503 = torch.ops.aten.view.default(view_502, [4096, 1024]);  view_502 = None
        permute_269 = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg374_1, view_503, permute_269);  arg374_1 = view_503 = permute_269 = None
        view_504 = torch.ops.aten.view.default(addmm_139, [32, 128, 1024]);  addmm_139 = None
        add_161 = torch.ops.aten.add.Tensor(add_158, view_504);  add_158 = view_504 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_161, getitem_91);  getitem_91 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_45);  sub_71 = rsqrt_45 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, arg375_1);  mul_172 = arg375_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_173, arg376_1);  mul_173 = arg376_1 = None
        view_505 = torch.ops.aten.view.default(add_163, [4096, 1024]);  add_163 = None
        permute_270 = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg378_1, view_505, permute_270);  arg378_1 = view_505 = permute_270 = None
        view_506 = torch.ops.aten.view.default(addmm_140, [32, 128, 4096]);  addmm_140 = None
        mul_174 = torch.ops.aten.mul.Tensor(view_506, 0.5)
        mul_175 = torch.ops.aten.mul.Tensor(view_506, 0.7071067811865476);  view_506 = None
        erf_18 = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_164 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_174, add_164);  mul_174 = add_164 = None
        view_507 = torch.ops.aten.view.default(mul_176, [4096, 4096]);  mul_176 = None
        permute_271 = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg380_1, view_507, permute_271);  arg380_1 = view_507 = permute_271 = None
        view_508 = torch.ops.aten.view.default(addmm_141, [32, 128, 1024]);  addmm_141 = None
        add_165 = torch.ops.aten.add.Tensor(add_161, view_508);  add_161 = view_508 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_166 = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_72 = torch.ops.aten.sub.Tensor(add_165, getitem_93);  getitem_93 = None
        mul_177 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_46);  sub_72 = rsqrt_46 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, arg381_1);  mul_177 = arg381_1 = None
        add_167 = torch.ops.aten.add.Tensor(mul_178, arg382_1);  mul_178 = arg382_1 = None
        view_509 = torch.ops.aten.view.default(add_167, [4096, 1024])
        permute_272 = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg384_1, view_509, permute_272);  arg384_1 = view_509 = permute_272 = None
        view_510 = torch.ops.aten.view.default(addmm_142, [32, 128, 1024]);  addmm_142 = None
        mul_179 = torch.ops.aten.mul.Tensor(view_510, 0.125);  view_510 = None
        view_511 = torch.ops.aten.view.default(add_167, [4096, 1024])
        permute_273 = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg386_1, view_511, permute_273);  arg386_1 = view_511 = permute_273 = None
        view_512 = torch.ops.aten.view.default(addmm_143, [32, 128, 1024]);  addmm_143 = None
        view_513 = torch.ops.aten.view.default(view_512, [32, -1, 16, 64]);  view_512 = None
        permute_274 = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
        clone_196 = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
        view_514 = torch.ops.aten.view.default(add_167, [4096, 1024]);  add_167 = None
        permute_275 = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg388_1, view_514, permute_275);  arg388_1 = view_514 = permute_275 = None
        view_515 = torch.ops.aten.view.default(addmm_144, [32, 128, 1024]);  addmm_144 = None
        view_516 = torch.ops.aten.view.default(view_515, [32, -1, 16, 64]);  view_515 = None
        permute_276 = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_197 = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
        view_517 = torch.ops.aten.view.default(mul_179, [32, 128, 16, 64]);  mul_179 = None
        permute_277 = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
        clone_198 = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
        view_518 = torch.ops.aten.view.default(clone_198, [512, -1, 64]);  clone_198 = None
        view_519 = torch.ops.aten.view.default(clone_196, [512, -1, 64]);  clone_196 = None
        view_520 = torch.ops.aten.view.default(clone_197, [512, -1, 64]);  clone_197 = None
        permute_278 = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
        bmm_52 = torch.ops.aten.bmm.default(view_518, permute_278);  view_518 = permute_278 = None
        view_521 = torch.ops.aten.view.default(bmm_52, [32, 16, 128, 128]);  bmm_52 = None
        add_168 = torch.ops.aten.add.Tensor(view_521, expand_1);  view_521 = None
        view_522 = torch.ops.aten.view.default(add_168, [512, 128, 128]);  add_168 = None
        amax_26 = torch.ops.aten.amax.default(view_522, [-1], True)
        sub_73 = torch.ops.aten.sub.Tensor(view_522, amax_26);  view_522 = amax_26 = None
        exp_26 = torch.ops.aten.exp.default(sub_73);  sub_73 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        bmm_53 = torch.ops.aten.bmm.default(div_26, view_520);  div_26 = view_520 = None
        view_523 = torch.ops.aten.view.default(bmm_53, [32, 16, 128, 64]);  bmm_53 = None
        permute_279 = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
        clone_200 = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_524 = torch.ops.aten.view.default(clone_200, [32, 128, 1024]);  clone_200 = None
        view_525 = torch.ops.aten.view.default(view_524, [4096, 1024]);  view_524 = None
        permute_280 = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg390_1, view_525, permute_280);  arg390_1 = view_525 = permute_280 = None
        view_526 = torch.ops.aten.view.default(addmm_145, [32, 128, 1024]);  addmm_145 = None
        add_169 = torch.ops.aten.add.Tensor(add_165, view_526);  add_165 = view_526 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_74 = torch.ops.aten.sub.Tensor(add_169, getitem_95);  getitem_95 = None
        mul_180 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_47);  sub_74 = rsqrt_47 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, arg391_1);  mul_180 = arg391_1 = None
        add_171 = torch.ops.aten.add.Tensor(mul_181, arg392_1);  mul_181 = arg392_1 = None
        view_527 = torch.ops.aten.view.default(add_171, [4096, 1024]);  add_171 = None
        permute_281 = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg394_1, view_527, permute_281);  arg394_1 = view_527 = permute_281 = None
        view_528 = torch.ops.aten.view.default(addmm_146, [32, 128, 1024]);  addmm_146 = None
        mul_182 = torch.ops.aten.mul.Tensor(view_528, 0.125);  view_528 = None
        view_529 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_282 = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg396_1, view_529, permute_282);  arg396_1 = view_529 = permute_282 = None
        view_530 = torch.ops.aten.view.default(addmm_147, [32, 128, 1024]);  addmm_147 = None
        view_531 = torch.ops.aten.view.default(view_530, [32, -1, 16, 64]);  view_530 = None
        permute_283 = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
        clone_202 = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
        view_532 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_284 = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg398_1, view_532, permute_284);  arg398_1 = view_532 = permute_284 = None
        view_533 = torch.ops.aten.view.default(addmm_148, [32, 128, 1024]);  addmm_148 = None
        view_534 = torch.ops.aten.view.default(view_533, [32, -1, 16, 64]);  view_533 = None
        permute_285 = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        clone_203 = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        view_535 = torch.ops.aten.view.default(mul_182, [32, 128, 16, 64]);  mul_182 = None
        permute_286 = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_204 = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_536 = torch.ops.aten.view.default(clone_204, [512, -1, 64]);  clone_204 = None
        view_537 = torch.ops.aten.view.default(clone_202, [512, -1, 64]);  clone_202 = None
        view_538 = torch.ops.aten.view.default(clone_203, [512, -1, 64]);  clone_203 = None
        unsqueeze_default_12 = torch.ops.aten.unsqueeze.default(view_536, 0);  view_536 = None
        unsqueeze_default_13 = torch.ops.aten.unsqueeze.default(view_537, 0);  view_537 = None
        unsqueeze_default_14 = torch.ops.aten.unsqueeze.default(view_538, 0);  view_538 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, None, False, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
        getitem_128 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
        squeeze_dim_4 = torch.ops.aten.squeeze.dim(getitem_128, 0);  getitem_128 = None
        view_539 = torch.ops.aten.view.default(squeeze_dim_4, [32, 16, 128, 64]);  squeeze_dim_4 = None
        permute_288 = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
        clone_206 = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_540 = torch.ops.aten.view.default(clone_206, [32, 128, 1024]);  clone_206 = None
        view_541 = torch.ops.aten.view.default(view_540, [4096, 1024]);  view_540 = None
        permute_289 = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg400_1, view_541, permute_289);  arg400_1 = view_541 = permute_289 = None
        view_542 = torch.ops.aten.view.default(addmm_149, [32, 128, 1024]);  addmm_149 = None
        add_172 = torch.ops.aten.add.Tensor(add_169, view_542);  add_169 = view_542 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_172, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_173 = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        sub_76 = torch.ops.aten.sub.Tensor(add_172, getitem_97);  getitem_97 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_48);  sub_76 = rsqrt_48 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, arg401_1);  mul_183 = arg401_1 = None
        add_174 = torch.ops.aten.add.Tensor(mul_184, arg402_1);  mul_184 = arg402_1 = None
        view_543 = torch.ops.aten.view.default(add_174, [4096, 1024]);  add_174 = None
        permute_290 = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg404_1, view_543, permute_290);  arg404_1 = view_543 = permute_290 = None
        view_544 = torch.ops.aten.view.default(addmm_150, [32, 128, 4096]);  addmm_150 = None
        mul_185 = torch.ops.aten.mul.Tensor(view_544, 0.5)
        mul_186 = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476);  view_544 = None
        erf_19 = torch.ops.aten.erf.default(mul_186);  mul_186 = None
        add_175 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_185, add_175);  mul_185 = add_175 = None
        view_545 = torch.ops.aten.view.default(mul_187, [4096, 4096]);  mul_187 = None
        permute_291 = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg406_1, view_545, permute_291);  arg406_1 = view_545 = permute_291 = None
        view_546 = torch.ops.aten.view.default(addmm_151, [32, 128, 1024]);  addmm_151 = None
        add_176 = torch.ops.aten.add.Tensor(add_172, view_546);  add_172 = view_546 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
        getitem_98 = var_mean_49[0]
        getitem_99 = var_mean_49[1];  var_mean_49 = None
        add_177 = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_77 = torch.ops.aten.sub.Tensor(add_176, getitem_99);  getitem_99 = None
        mul_188 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_49);  sub_77 = rsqrt_49 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, arg407_1);  mul_188 = arg407_1 = None
        add_178 = torch.ops.aten.add.Tensor(mul_189, arg408_1);  mul_189 = arg408_1 = None
        view_547 = torch.ops.aten.view.default(add_178, [4096, 1024])
        permute_292 = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg410_1, view_547, permute_292);  arg410_1 = view_547 = permute_292 = None
        view_548 = torch.ops.aten.view.default(addmm_152, [32, 128, 1024]);  addmm_152 = None
        mul_190 = torch.ops.aten.mul.Tensor(view_548, 0.125);  view_548 = None
        view_549 = torch.ops.aten.view.default(add_178, [4096, 1024])
        permute_293 = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg412_1, view_549, permute_293);  arg412_1 = view_549 = permute_293 = None
        view_550 = torch.ops.aten.view.default(addmm_153, [32, 128, 1024]);  addmm_153 = None
        view_551 = torch.ops.aten.view.default(view_550, [32, -1, 16, 64]);  view_550 = None
        permute_294 = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
        clone_210 = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_552 = torch.ops.aten.view.default(add_178, [4096, 1024]);  add_178 = None
        permute_295 = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg414_1, view_552, permute_295);  arg414_1 = view_552 = permute_295 = None
        view_553 = torch.ops.aten.view.default(addmm_154, [32, 128, 1024]);  addmm_154 = None
        view_554 = torch.ops.aten.view.default(view_553, [32, -1, 16, 64]);  view_553 = None
        permute_296 = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        clone_211 = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
        view_555 = torch.ops.aten.view.default(mul_190, [32, 128, 16, 64]);  mul_190 = None
        permute_297 = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
        clone_212 = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
        view_556 = torch.ops.aten.view.default(clone_212, [512, -1, 64]);  clone_212 = None
        view_557 = torch.ops.aten.view.default(clone_210, [512, -1, 64]);  clone_210 = None
        view_558 = torch.ops.aten.view.default(clone_211, [512, -1, 64]);  clone_211 = None
        permute_298 = torch.ops.aten.permute.default(view_557, [0, 2, 1]);  view_557 = None
        bmm_56 = torch.ops.aten.bmm.default(view_556, permute_298);  view_556 = permute_298 = None
        view_559 = torch.ops.aten.view.default(bmm_56, [32, 16, 128, 128]);  bmm_56 = None
        add_179 = torch.ops.aten.add.Tensor(view_559, expand_1);  view_559 = None
        view_560 = torch.ops.aten.view.default(add_179, [512, 128, 128]);  add_179 = None
        amax_28 = torch.ops.aten.amax.default(view_560, [-1], True)
        sub_78 = torch.ops.aten.sub.Tensor(view_560, amax_28);  view_560 = amax_28 = None
        exp_28 = torch.ops.aten.exp.default(sub_78);  sub_78 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28 = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        bmm_57 = torch.ops.aten.bmm.default(div_28, view_558);  div_28 = view_558 = None
        view_561 = torch.ops.aten.view.default(bmm_57, [32, 16, 128, 64]);  bmm_57 = None
        permute_299 = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_214 = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
        view_562 = torch.ops.aten.view.default(clone_214, [32, 128, 1024]);  clone_214 = None
        view_563 = torch.ops.aten.view.default(view_562, [4096, 1024]);  view_562 = None
        permute_300 = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg416_1, view_563, permute_300);  arg416_1 = view_563 = permute_300 = None
        view_564 = torch.ops.aten.view.default(addmm_155, [32, 128, 1024]);  addmm_155 = None
        add_180 = torch.ops.aten.add.Tensor(add_176, view_564);  add_176 = view_564 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_100 = var_mean_50[0]
        getitem_101 = var_mean_50[1];  var_mean_50 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_79 = torch.ops.aten.sub.Tensor(add_180, getitem_101);  getitem_101 = None
        mul_191 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_50);  sub_79 = rsqrt_50 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_191, arg417_1);  mul_191 = arg417_1 = None
        add_182 = torch.ops.aten.add.Tensor(mul_192, arg418_1);  mul_192 = arg418_1 = None
        view_565 = torch.ops.aten.view.default(add_182, [4096, 1024]);  add_182 = None
        permute_301 = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg420_1, view_565, permute_301);  arg420_1 = view_565 = permute_301 = None
        view_566 = torch.ops.aten.view.default(addmm_156, [32, 128, 1024]);  addmm_156 = None
        mul_193 = torch.ops.aten.mul.Tensor(view_566, 0.125);  view_566 = None
        view_567 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_302 = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg422_1, view_567, permute_302);  arg422_1 = view_567 = permute_302 = None
        view_568 = torch.ops.aten.view.default(addmm_157, [32, 128, 1024]);  addmm_157 = None
        view_569 = torch.ops.aten.view.default(view_568, [32, -1, 16, 64]);  view_568 = None
        permute_303 = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
        clone_216 = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_570 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_304 = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg424_1, view_570, permute_304);  arg424_1 = view_570 = permute_304 = None
        view_571 = torch.ops.aten.view.default(addmm_158, [32, 128, 1024]);  addmm_158 = None
        view_572 = torch.ops.aten.view.default(view_571, [32, -1, 16, 64]);  view_571 = None
        permute_305 = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        clone_217 = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
        view_573 = torch.ops.aten.view.default(mul_193, [32, 128, 16, 64]);  mul_193 = None
        permute_306 = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
        clone_218 = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
        view_574 = torch.ops.aten.view.default(clone_218, [512, -1, 64]);  clone_218 = None
        view_575 = torch.ops.aten.view.default(clone_216, [512, -1, 64]);  clone_216 = None
        view_576 = torch.ops.aten.view.default(clone_217, [512, -1, 64]);  clone_217 = None
        unsqueeze_default_9 = torch.ops.aten.unsqueeze.default(view_574, 0);  view_574 = None
        unsqueeze_default_10 = torch.ops.aten.unsqueeze.default(view_575, 0);  view_575 = None
        unsqueeze_default_11 = torch.ops.aten.unsqueeze.default(view_576, 0);  view_576 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, None, False, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
        getitem_127 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(getitem_127, 0);  getitem_127 = None
        view_577 = torch.ops.aten.view.default(squeeze_dim_3, [32, 16, 128, 64]);  squeeze_dim_3 = None
        permute_308 = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
        clone_220 = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_578 = torch.ops.aten.view.default(clone_220, [32, 128, 1024]);  clone_220 = None
        view_579 = torch.ops.aten.view.default(view_578, [4096, 1024]);  view_578 = None
        permute_309 = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg426_1, view_579, permute_309);  arg426_1 = view_579 = permute_309 = None
        view_580 = torch.ops.aten.view.default(addmm_159, [32, 128, 1024]);  addmm_159 = None
        add_183 = torch.ops.aten.add.Tensor(add_180, view_580);  add_180 = view_580 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
        getitem_102 = var_mean_51[0]
        getitem_103 = var_mean_51[1];  var_mean_51 = None
        add_184 = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_81 = torch.ops.aten.sub.Tensor(add_183, getitem_103);  getitem_103 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_51);  sub_81 = rsqrt_51 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, arg427_1);  mul_194 = arg427_1 = None
        add_185 = torch.ops.aten.add.Tensor(mul_195, arg428_1);  mul_195 = arg428_1 = None
        view_581 = torch.ops.aten.view.default(add_185, [4096, 1024]);  add_185 = None
        permute_310 = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg430_1, view_581, permute_310);  arg430_1 = view_581 = permute_310 = None
        view_582 = torch.ops.aten.view.default(addmm_160, [32, 128, 4096]);  addmm_160 = None
        mul_196 = torch.ops.aten.mul.Tensor(view_582, 0.5)
        mul_197 = torch.ops.aten.mul.Tensor(view_582, 0.7071067811865476);  view_582 = None
        erf_20 = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_186 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_196, add_186);  mul_196 = add_186 = None
        view_583 = torch.ops.aten.view.default(mul_198, [4096, 4096]);  mul_198 = None
        permute_311 = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg432_1, view_583, permute_311);  arg432_1 = view_583 = permute_311 = None
        view_584 = torch.ops.aten.view.default(addmm_161, [32, 128, 1024]);  addmm_161 = None
        add_187 = torch.ops.aten.add.Tensor(add_183, view_584);  add_183 = view_584 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
        getitem_104 = var_mean_52[0]
        getitem_105 = var_mean_52[1];  var_mean_52 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_82 = torch.ops.aten.sub.Tensor(add_187, getitem_105);  getitem_105 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_52);  sub_82 = rsqrt_52 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, arg433_1);  mul_199 = arg433_1 = None
        add_189 = torch.ops.aten.add.Tensor(mul_200, arg434_1);  mul_200 = arg434_1 = None
        view_585 = torch.ops.aten.view.default(add_189, [4096, 1024])
        permute_312 = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg436_1, view_585, permute_312);  arg436_1 = view_585 = permute_312 = None
        view_586 = torch.ops.aten.view.default(addmm_162, [32, 128, 1024]);  addmm_162 = None
        mul_201 = torch.ops.aten.mul.Tensor(view_586, 0.125);  view_586 = None
        view_587 = torch.ops.aten.view.default(add_189, [4096, 1024])
        permute_313 = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg438_1, view_587, permute_313);  arg438_1 = view_587 = permute_313 = None
        view_588 = torch.ops.aten.view.default(addmm_163, [32, 128, 1024]);  addmm_163 = None
        view_589 = torch.ops.aten.view.default(view_588, [32, -1, 16, 64]);  view_588 = None
        permute_314 = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
        clone_224 = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_590 = torch.ops.aten.view.default(add_189, [4096, 1024]);  add_189 = None
        permute_315 = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg440_1, view_590, permute_315);  arg440_1 = view_590 = permute_315 = None
        view_591 = torch.ops.aten.view.default(addmm_164, [32, 128, 1024]);  addmm_164 = None
        view_592 = torch.ops.aten.view.default(view_591, [32, -1, 16, 64]);  view_591 = None
        permute_316 = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
        clone_225 = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
        view_593 = torch.ops.aten.view.default(mul_201, [32, 128, 16, 64]);  mul_201 = None
        permute_317 = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
        clone_226 = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
        view_594 = torch.ops.aten.view.default(clone_226, [512, -1, 64]);  clone_226 = None
        view_595 = torch.ops.aten.view.default(clone_224, [512, -1, 64]);  clone_224 = None
        view_596 = torch.ops.aten.view.default(clone_225, [512, -1, 64]);  clone_225 = None
        permute_318 = torch.ops.aten.permute.default(view_595, [0, 2, 1]);  view_595 = None
        bmm_60 = torch.ops.aten.bmm.default(view_594, permute_318);  view_594 = permute_318 = None
        view_597 = torch.ops.aten.view.default(bmm_60, [32, 16, 128, 128]);  bmm_60 = None
        add_190 = torch.ops.aten.add.Tensor(view_597, expand_1);  view_597 = None
        view_598 = torch.ops.aten.view.default(add_190, [512, 128, 128]);  add_190 = None
        amax_30 = torch.ops.aten.amax.default(view_598, [-1], True)
        sub_83 = torch.ops.aten.sub.Tensor(view_598, amax_30);  view_598 = amax_30 = None
        exp_30 = torch.ops.aten.exp.default(sub_83);  sub_83 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30 = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        bmm_61 = torch.ops.aten.bmm.default(div_30, view_596);  div_30 = view_596 = None
        view_599 = torch.ops.aten.view.default(bmm_61, [32, 16, 128, 64]);  bmm_61 = None
        permute_319 = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        clone_228 = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_600 = torch.ops.aten.view.default(clone_228, [32, 128, 1024]);  clone_228 = None
        view_601 = torch.ops.aten.view.default(view_600, [4096, 1024]);  view_600 = None
        permute_320 = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg442_1, view_601, permute_320);  arg442_1 = view_601 = permute_320 = None
        view_602 = torch.ops.aten.view.default(addmm_165, [32, 128, 1024]);  addmm_165 = None
        add_191 = torch.ops.aten.add.Tensor(add_187, view_602);  add_187 = view_602 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_106 = var_mean_53[0]
        getitem_107 = var_mean_53[1];  var_mean_53 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_84 = torch.ops.aten.sub.Tensor(add_191, getitem_107);  getitem_107 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_53);  sub_84 = rsqrt_53 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, arg443_1);  mul_202 = arg443_1 = None
        add_193 = torch.ops.aten.add.Tensor(mul_203, arg444_1);  mul_203 = arg444_1 = None
        view_603 = torch.ops.aten.view.default(add_193, [4096, 1024]);  add_193 = None
        permute_321 = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg446_1, view_603, permute_321);  arg446_1 = view_603 = permute_321 = None
        view_604 = torch.ops.aten.view.default(addmm_166, [32, 128, 1024]);  addmm_166 = None
        mul_204 = torch.ops.aten.mul.Tensor(view_604, 0.125);  view_604 = None
        view_605 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_322 = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg448_1, view_605, permute_322);  arg448_1 = view_605 = permute_322 = None
        view_606 = torch.ops.aten.view.default(addmm_167, [32, 128, 1024]);  addmm_167 = None
        view_607 = torch.ops.aten.view.default(view_606, [32, -1, 16, 64]);  view_606 = None
        permute_323 = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
        clone_230 = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        view_608 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_324 = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg450_1, view_608, permute_324);  arg450_1 = view_608 = permute_324 = None
        view_609 = torch.ops.aten.view.default(addmm_168, [32, 128, 1024]);  addmm_168 = None
        view_610 = torch.ops.aten.view.default(view_609, [32, -1, 16, 64]);  view_609 = None
        permute_325 = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
        clone_231 = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_611 = torch.ops.aten.view.default(mul_204, [32, 128, 16, 64]);  mul_204 = None
        permute_326 = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None
        clone_232 = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        view_612 = torch.ops.aten.view.default(clone_232, [512, -1, 64]);  clone_232 = None
        view_613 = torch.ops.aten.view.default(clone_230, [512, -1, 64]);  clone_230 = None
        view_614 = torch.ops.aten.view.default(clone_231, [512, -1, 64]);  clone_231 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(view_612, 0);  view_612 = None
        unsqueeze_default_7 = torch.ops.aten.unsqueeze.default(view_613, 0);  view_613 = None
        unsqueeze_default_8 = torch.ops.aten.unsqueeze.default(view_614, 0);  view_614 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, None, False, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
        getitem_126 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(getitem_126, 0);  getitem_126 = None
        view_615 = torch.ops.aten.view.default(squeeze_dim_2, [32, 16, 128, 64]);  squeeze_dim_2 = None
        permute_328 = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
        clone_234 = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_616 = torch.ops.aten.view.default(clone_234, [32, 128, 1024]);  clone_234 = None
        view_617 = torch.ops.aten.view.default(view_616, [4096, 1024]);  view_616 = None
        permute_329 = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg452_1, view_617, permute_329);  arg452_1 = view_617 = permute_329 = None
        view_618 = torch.ops.aten.view.default(addmm_169, [32, 128, 1024]);  addmm_169 = None
        add_194 = torch.ops.aten.add.Tensor(add_191, view_618);  add_191 = view_618 = None
        var_mean_54 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
        getitem_108 = var_mean_54[0]
        getitem_109 = var_mean_54[1];  var_mean_54 = None
        add_195 = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_86 = torch.ops.aten.sub.Tensor(add_194, getitem_109);  getitem_109 = None
        mul_205 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_54);  sub_86 = rsqrt_54 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, arg453_1);  mul_205 = arg453_1 = None
        add_196 = torch.ops.aten.add.Tensor(mul_206, arg454_1);  mul_206 = arg454_1 = None
        view_619 = torch.ops.aten.view.default(add_196, [4096, 1024]);  add_196 = None
        permute_330 = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg456_1, view_619, permute_330);  arg456_1 = view_619 = permute_330 = None
        view_620 = torch.ops.aten.view.default(addmm_170, [32, 128, 4096]);  addmm_170 = None
        mul_207 = torch.ops.aten.mul.Tensor(view_620, 0.5)
        mul_208 = torch.ops.aten.mul.Tensor(view_620, 0.7071067811865476);  view_620 = None
        erf_21 = torch.ops.aten.erf.default(mul_208);  mul_208 = None
        add_197 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_207, add_197);  mul_207 = add_197 = None
        view_621 = torch.ops.aten.view.default(mul_209, [4096, 4096]);  mul_209 = None
        permute_331 = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg458_1, view_621, permute_331);  arg458_1 = view_621 = permute_331 = None
        view_622 = torch.ops.aten.view.default(addmm_171, [32, 128, 1024]);  addmm_171 = None
        add_198 = torch.ops.aten.add.Tensor(add_194, view_622);  add_194 = view_622 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_55[0]
        getitem_111 = var_mean_55[1];  var_mean_55 = None
        add_199 = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_87 = torch.ops.aten.sub.Tensor(add_198, getitem_111);  getitem_111 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_55);  sub_87 = rsqrt_55 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_210, arg459_1);  mul_210 = arg459_1 = None
        add_200 = torch.ops.aten.add.Tensor(mul_211, arg460_1);  mul_211 = arg460_1 = None
        view_623 = torch.ops.aten.view.default(add_200, [4096, 1024])
        permute_332 = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg462_1, view_623, permute_332);  arg462_1 = view_623 = permute_332 = None
        view_624 = torch.ops.aten.view.default(addmm_172, [32, 128, 1024]);  addmm_172 = None
        mul_212 = torch.ops.aten.mul.Tensor(view_624, 0.125);  view_624 = None
        view_625 = torch.ops.aten.view.default(add_200, [4096, 1024])
        permute_333 = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg464_1, view_625, permute_333);  arg464_1 = view_625 = permute_333 = None
        view_626 = torch.ops.aten.view.default(addmm_173, [32, 128, 1024]);  addmm_173 = None
        view_627 = torch.ops.aten.view.default(view_626, [32, -1, 16, 64]);  view_626 = None
        permute_334 = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
        clone_238 = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_628 = torch.ops.aten.view.default(add_200, [4096, 1024]);  add_200 = None
        permute_335 = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg466_1, view_628, permute_335);  arg466_1 = view_628 = permute_335 = None
        view_629 = torch.ops.aten.view.default(addmm_174, [32, 128, 1024]);  addmm_174 = None
        view_630 = torch.ops.aten.view.default(view_629, [32, -1, 16, 64]);  view_629 = None
        permute_336 = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        clone_239 = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_631 = torch.ops.aten.view.default(mul_212, [32, 128, 16, 64]);  mul_212 = None
        permute_337 = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
        clone_240 = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_632 = torch.ops.aten.view.default(clone_240, [512, -1, 64]);  clone_240 = None
        view_633 = torch.ops.aten.view.default(clone_238, [512, -1, 64]);  clone_238 = None
        view_634 = torch.ops.aten.view.default(clone_239, [512, -1, 64]);  clone_239 = None
        permute_338 = torch.ops.aten.permute.default(view_633, [0, 2, 1]);  view_633 = None
        bmm_64 = torch.ops.aten.bmm.default(view_632, permute_338);  view_632 = permute_338 = None
        view_635 = torch.ops.aten.view.default(bmm_64, [32, 16, 128, 128]);  bmm_64 = None
        add_201 = torch.ops.aten.add.Tensor(view_635, expand_1);  view_635 = None
        view_636 = torch.ops.aten.view.default(add_201, [512, 128, 128]);  add_201 = None
        amax_32 = torch.ops.aten.amax.default(view_636, [-1], True)
        sub_88 = torch.ops.aten.sub.Tensor(view_636, amax_32);  view_636 = amax_32 = None
        exp_32 = torch.ops.aten.exp.default(sub_88);  sub_88 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32 = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        bmm_65 = torch.ops.aten.bmm.default(div_32, view_634);  div_32 = view_634 = None
        view_637 = torch.ops.aten.view.default(bmm_65, [32, 16, 128, 64]);  bmm_65 = None
        permute_339 = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
        clone_242 = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
        view_638 = torch.ops.aten.view.default(clone_242, [32, 128, 1024]);  clone_242 = None
        view_639 = torch.ops.aten.view.default(view_638, [4096, 1024]);  view_638 = None
        permute_340 = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg468_1, view_639, permute_340);  arg468_1 = view_639 = permute_340 = None
        view_640 = torch.ops.aten.view.default(addmm_175, [32, 128, 1024]);  addmm_175 = None
        add_202 = torch.ops.aten.add.Tensor(add_198, view_640);  add_198 = view_640 = None
        var_mean_56 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_112 = var_mean_56[0]
        getitem_113 = var_mean_56[1];  var_mean_56 = None
        add_203 = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_89 = torch.ops.aten.sub.Tensor(add_202, getitem_113);  getitem_113 = None
        mul_213 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_56);  sub_89 = rsqrt_56 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, arg469_1);  mul_213 = arg469_1 = None
        add_204 = torch.ops.aten.add.Tensor(mul_214, arg470_1);  mul_214 = arg470_1 = None
        view_641 = torch.ops.aten.view.default(add_204, [4096, 1024]);  add_204 = None
        permute_341 = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg472_1, view_641, permute_341);  arg472_1 = view_641 = permute_341 = None
        view_642 = torch.ops.aten.view.default(addmm_176, [32, 128, 1024]);  addmm_176 = None
        mul_215 = torch.ops.aten.mul.Tensor(view_642, 0.125);  view_642 = None
        view_643 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_342 = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg474_1, view_643, permute_342);  arg474_1 = view_643 = permute_342 = None
        view_644 = torch.ops.aten.view.default(addmm_177, [32, 128, 1024]);  addmm_177 = None
        view_645 = torch.ops.aten.view.default(view_644, [32, -1, 16, 64]);  view_644 = None
        permute_343 = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
        clone_244 = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
        view_646 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_344 = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg476_1, view_646, permute_344);  arg476_1 = view_646 = permute_344 = None
        view_647 = torch.ops.aten.view.default(addmm_178, [32, 128, 1024]);  addmm_178 = None
        view_648 = torch.ops.aten.view.default(view_647, [32, -1, 16, 64]);  view_647 = None
        permute_345 = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
        clone_245 = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
        view_649 = torch.ops.aten.view.default(mul_215, [32, 128, 16, 64]);  mul_215 = None
        permute_346 = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
        clone_246 = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
        view_650 = torch.ops.aten.view.default(clone_246, [512, -1, 64]);  clone_246 = None
        view_651 = torch.ops.aten.view.default(clone_244, [512, -1, 64]);  clone_244 = None
        view_652 = torch.ops.aten.view.default(clone_245, [512, -1, 64]);  clone_245 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(view_650, 0);  view_650 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(view_651, 0);  view_651 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(view_652, 0);  view_652 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, None, False, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
        getitem_125 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(getitem_125, 0);  getitem_125 = None
        view_653 = torch.ops.aten.view.default(squeeze_dim_1, [32, 16, 128, 64]);  squeeze_dim_1 = None
        permute_348 = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
        clone_248 = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
        view_654 = torch.ops.aten.view.default(clone_248, [32, 128, 1024]);  clone_248 = None
        view_655 = torch.ops.aten.view.default(view_654, [4096, 1024]);  view_654 = None
        permute_349 = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg478_1, view_655, permute_349);  arg478_1 = view_655 = permute_349 = None
        view_656 = torch.ops.aten.view.default(addmm_179, [32, 128, 1024]);  addmm_179 = None
        add_205 = torch.ops.aten.add.Tensor(add_202, view_656);  add_202 = view_656 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
        getitem_114 = var_mean_57[0]
        getitem_115 = var_mean_57[1];  var_mean_57 = None
        add_206 = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_91 = torch.ops.aten.sub.Tensor(add_205, getitem_115);  getitem_115 = None
        mul_216 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_57);  sub_91 = rsqrt_57 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, arg479_1);  mul_216 = arg479_1 = None
        add_207 = torch.ops.aten.add.Tensor(mul_217, arg480_1);  mul_217 = arg480_1 = None
        view_657 = torch.ops.aten.view.default(add_207, [4096, 1024]);  add_207 = None
        permute_350 = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg482_1, view_657, permute_350);  arg482_1 = view_657 = permute_350 = None
        view_658 = torch.ops.aten.view.default(addmm_180, [32, 128, 4096]);  addmm_180 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_658, 0.5)
        mul_219 = torch.ops.aten.mul.Tensor(view_658, 0.7071067811865476);  view_658 = None
        erf_22 = torch.ops.aten.erf.default(mul_219);  mul_219 = None
        add_208 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_220 = torch.ops.aten.mul.Tensor(mul_218, add_208);  mul_218 = add_208 = None
        view_659 = torch.ops.aten.view.default(mul_220, [4096, 4096]);  mul_220 = None
        permute_351 = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg484_1, view_659, permute_351);  arg484_1 = view_659 = permute_351 = None
        view_660 = torch.ops.aten.view.default(addmm_181, [32, 128, 1024]);  addmm_181 = None
        add_209 = torch.ops.aten.add.Tensor(add_205, view_660);  add_205 = view_660 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(add_209, [2], correction = 0, keepdim = True)
        getitem_116 = var_mean_58[0]
        getitem_117 = var_mean_58[1];  var_mean_58 = None
        add_210 = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        sub_92 = torch.ops.aten.sub.Tensor(add_209, getitem_117);  getitem_117 = None
        mul_221 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_58);  sub_92 = rsqrt_58 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, arg485_1);  mul_221 = arg485_1 = None
        add_211 = torch.ops.aten.add.Tensor(mul_222, arg486_1);  mul_222 = arg486_1 = None
        view_661 = torch.ops.aten.view.default(add_211, [4096, 1024])
        permute_352 = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg488_1, view_661, permute_352);  arg488_1 = view_661 = permute_352 = None
        view_662 = torch.ops.aten.view.default(addmm_182, [32, 128, 1024]);  addmm_182 = None
        mul_223 = torch.ops.aten.mul.Tensor(view_662, 0.125);  view_662 = None
        view_663 = torch.ops.aten.view.default(add_211, [4096, 1024])
        permute_353 = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg490_1, view_663, permute_353);  arg490_1 = view_663 = permute_353 = None
        view_664 = torch.ops.aten.view.default(addmm_183, [32, 128, 1024]);  addmm_183 = None
        view_665 = torch.ops.aten.view.default(view_664, [32, -1, 16, 64]);  view_664 = None
        permute_354 = torch.ops.aten.permute.default(view_665, [0, 2, 1, 3]);  view_665 = None
        clone_252 = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
        view_666 = torch.ops.aten.view.default(add_211, [4096, 1024]);  add_211 = None
        permute_355 = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg492_1, view_666, permute_355);  arg492_1 = view_666 = permute_355 = None
        view_667 = torch.ops.aten.view.default(addmm_184, [32, 128, 1024]);  addmm_184 = None
        view_668 = torch.ops.aten.view.default(view_667, [32, -1, 16, 64]);  view_667 = None
        permute_356 = torch.ops.aten.permute.default(view_668, [0, 2, 1, 3]);  view_668 = None
        clone_253 = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
        view_669 = torch.ops.aten.view.default(mul_223, [32, 128, 16, 64]);  mul_223 = None
        permute_357 = torch.ops.aten.permute.default(view_669, [0, 2, 1, 3]);  view_669 = None
        clone_254 = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
        view_670 = torch.ops.aten.view.default(clone_254, [512, -1, 64]);  clone_254 = None
        view_671 = torch.ops.aten.view.default(clone_252, [512, -1, 64]);  clone_252 = None
        view_672 = torch.ops.aten.view.default(clone_253, [512, -1, 64]);  clone_253 = None
        permute_358 = torch.ops.aten.permute.default(view_671, [0, 2, 1]);  view_671 = None
        bmm_68 = torch.ops.aten.bmm.default(view_670, permute_358);  view_670 = permute_358 = None
        view_673 = torch.ops.aten.view.default(bmm_68, [32, 16, 128, 128]);  bmm_68 = None
        add_212 = torch.ops.aten.add.Tensor(view_673, expand_1);  view_673 = expand_1 = None
        view_674 = torch.ops.aten.view.default(add_212, [512, 128, 128]);  add_212 = None
        amax_34 = torch.ops.aten.amax.default(view_674, [-1], True)
        sub_93 = torch.ops.aten.sub.Tensor(view_674, amax_34);  view_674 = amax_34 = None
        exp_34 = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34 = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        bmm_69 = torch.ops.aten.bmm.default(div_34, view_672);  div_34 = view_672 = None
        view_675 = torch.ops.aten.view.default(bmm_69, [32, 16, 128, 64]);  bmm_69 = None
        permute_359 = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None
        clone_256 = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_676 = torch.ops.aten.view.default(clone_256, [32, 128, 1024]);  clone_256 = None
        view_677 = torch.ops.aten.view.default(view_676, [4096, 1024]);  view_676 = None
        permute_360 = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg494_1, view_677, permute_360);  arg494_1 = view_677 = permute_360 = None
        view_678 = torch.ops.aten.view.default(addmm_185, [32, 128, 1024]);  addmm_185 = None
        add_213 = torch.ops.aten.add.Tensor(add_209, view_678);  add_209 = view_678 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
        getitem_118 = var_mean_59[0]
        getitem_119 = var_mean_59[1];  var_mean_59 = None
        add_214 = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_94 = torch.ops.aten.sub.Tensor(add_213, getitem_119);  getitem_119 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_59);  sub_94 = rsqrt_59 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, arg495_1);  mul_224 = arg495_1 = None
        add_215 = torch.ops.aten.add.Tensor(mul_225, arg496_1);  mul_225 = arg496_1 = None
        view_679 = torch.ops.aten.view.default(add_215, [4096, 1024]);  add_215 = None
        permute_361 = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg498_1, view_679, permute_361);  arg498_1 = view_679 = permute_361 = None
        view_680 = torch.ops.aten.view.default(addmm_186, [32, 128, 1024]);  addmm_186 = None
        mul_226 = torch.ops.aten.mul.Tensor(view_680, 0.125);  view_680 = None
        view_681 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_362 = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg500_1, view_681, permute_362);  arg500_1 = view_681 = permute_362 = None
        view_682 = torch.ops.aten.view.default(addmm_187, [32, 128, 1024]);  addmm_187 = None
        view_683 = torch.ops.aten.view.default(view_682, [32, -1, 16, 64]);  view_682 = None
        permute_363 = torch.ops.aten.permute.default(view_683, [0, 2, 1, 3]);  view_683 = None
        clone_258 = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
        view_684 = torch.ops.aten.view.default(add_86, [4096, 1024])
        permute_364 = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg502_1, view_684, permute_364);  arg502_1 = view_684 = permute_364 = None
        view_685 = torch.ops.aten.view.default(addmm_188, [32, 128, 1024]);  addmm_188 = None
        view_686 = torch.ops.aten.view.default(view_685, [32, -1, 16, 64]);  view_685 = None
        permute_365 = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
        clone_259 = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_687 = torch.ops.aten.view.default(mul_226, [32, 128, 16, 64]);  mul_226 = None
        permute_366 = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
        clone_260 = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
        view_688 = torch.ops.aten.view.default(clone_260, [512, -1, 64]);  clone_260 = None
        view_689 = torch.ops.aten.view.default(clone_258, [512, -1, 64]);  clone_258 = None
        view_690 = torch.ops.aten.view.default(clone_259, [512, -1, 64]);  clone_259 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(view_688, 0);  view_688 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(view_689, 0);  view_689 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(view_690, 0);  view_690 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, False, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
        getitem_124 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem_124, 0);  getitem_124 = None
        view_691 = torch.ops.aten.view.default(squeeze_dim, [32, 16, 128, 64]);  squeeze_dim = None
        permute_368 = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None
        clone_262 = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_692 = torch.ops.aten.view.default(clone_262, [32, 128, 1024]);  clone_262 = None
        view_693 = torch.ops.aten.view.default(view_692, [4096, 1024]);  view_692 = None
        permute_369 = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg504_1, view_693, permute_369);  arg504_1 = view_693 = permute_369 = None
        view_694 = torch.ops.aten.view.default(addmm_189, [32, 128, 1024]);  addmm_189 = None
        add_216 = torch.ops.aten.add.Tensor(add_213, view_694);  add_213 = view_694 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(add_216, [2], correction = 0, keepdim = True)
        getitem_120 = var_mean_60[0]
        getitem_121 = var_mean_60[1];  var_mean_60 = None
        add_217 = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        sub_96 = torch.ops.aten.sub.Tensor(add_216, getitem_121);  getitem_121 = None
        mul_227 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_60);  sub_96 = rsqrt_60 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, arg505_1);  mul_227 = arg505_1 = None
        add_218 = torch.ops.aten.add.Tensor(mul_228, arg506_1);  mul_228 = arg506_1 = None
        view_695 = torch.ops.aten.view.default(add_218, [4096, 1024]);  add_218 = None
        permute_370 = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg508_1, view_695, permute_370);  arg508_1 = view_695 = permute_370 = None
        view_696 = torch.ops.aten.view.default(addmm_190, [32, 128, 4096]);  addmm_190 = None
        mul_229 = torch.ops.aten.mul.Tensor(view_696, 0.5)
        mul_230 = torch.ops.aten.mul.Tensor(view_696, 0.7071067811865476);  view_696 = None
        erf_23 = torch.ops.aten.erf.default(mul_230);  mul_230 = None
        add_219 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_229, add_219);  mul_229 = add_219 = None
        view_697 = torch.ops.aten.view.default(mul_231, [4096, 4096]);  mul_231 = None
        permute_371 = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg510_1, view_697, permute_371);  arg510_1 = view_697 = permute_371 = None
        view_698 = torch.ops.aten.view.default(addmm_191, [32, 128, 1024]);  addmm_191 = None
        add_220 = torch.ops.aten.add.Tensor(add_216, view_698);  add_216 = view_698 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(add_220, [2], correction = 0, keepdim = True)
        getitem_122 = var_mean_61[0]
        getitem_123 = var_mean_61[1];  var_mean_61 = None
        add_221 = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_97 = torch.ops.aten.sub.Tensor(add_220, getitem_123);  add_220 = getitem_123 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_61);  sub_97 = rsqrt_61 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, arg511_1);  mul_232 = arg511_1 = None
        add_222 = torch.ops.aten.add.Tensor(mul_233, arg512_1);  mul_233 = arg512_1 = None
        permute_372 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_699 = torch.ops.aten.view.default(add_222, [4096, 1024]);  add_222 = None
        full_default_4 = torch.ops.aten.full.default([1024, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default = torch.ops.aten.cat.default([permute_372, full_default_4], 1);  permute_372 = full_default_4 = None
        mm_default = torch.ops.aten.mm.default(view_699, cat_default);  view_699 = cat_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        view_700 = torch.ops.aten.view.default(slice_tensor, [32, 128, 50265]);  slice_tensor = None
        add_223 = torch.ops.aten.add.Tensor(view_700, arg513_1);  view_700 = arg513_1 = None
        view_701 = torch.ops.aten.view.default(add_223, [-1, 50265])
        view_702 = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        amax_36 = torch.ops.aten.amax.default(view_701, [1], True)
        sub_98 = torch.ops.aten.sub.Tensor(view_701, amax_36);  view_701 = amax_36 = None
        exp_36 = torch.ops.aten.exp.default(sub_98)
        sum_37 = torch.ops.aten.sum.dim_IntList(exp_36, [1], True);  exp_36 = None
        log = torch.ops.aten.log.default(sum_37);  sum_37 = None
        sub_99 = torch.ops.aten.sub.Tensor(sub_98, log);  sub_98 = log = None
        ne = torch.ops.aten.ne.Scalar(view_702, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, view_702, full_default_2);  ne = full_default_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_99, 1, unsqueeze_4);  sub_99 = unsqueeze_4 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1 = torch.ops.aten.ne.Scalar(view_702, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_702, -100);  view_702 = None
        sum_38 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
        sum_39 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_36 = torch.ops.aten.div.Tensor(sum_39, convert_element_type);  sum_39 = convert_element_type = None
        return (div_36, add_223, add_86)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (32, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (32, 128), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 205885440, device=device(type='cuda', index=0))
    reader.tensor(buf2, (50265, 1024), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024, 1024), is_leaf=True)  # arg3_1
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
    buf200 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1024,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1024, 1024), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1024,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1024, 1024), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1024,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1024, 1024), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1024,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1024, 1024), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf208, (1024,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf209, (1024,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf210, (1024,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf211, (1024, 1024), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1024,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1024, 1024), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1024,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1024, 1024), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1024,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1024, 1024), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1024,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1024,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1024,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf221, (4096, 1024), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf222, (4096,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1024, 4096), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf224, (1024,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf225, (1024,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1024,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1024, 1024), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf228, (1024,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf229, (1024, 1024), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf230, (1024,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1024, 1024), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1024,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1024, 1024), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf234, (1024,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf235, (1024,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf236, (1024,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf237, (1024, 1024), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf238, (1024,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf239, (1024, 1024), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf240, (1024,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf241, (1024, 1024), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf242, (1024,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf243, (1024, 1024), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf244, (1024,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1024,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1024,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf247, (4096, 1024), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf248, (4096,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1024, 4096), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf250, (1024,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf251, (1024,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf252, (1024,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf253, (1024, 1024), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf254, (1024,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1024, 1024), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1024,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1024, 1024), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1024,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1024, 1024), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1024,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1024,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1024,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1024, 1024), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1024,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1024, 1024), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf266, (1024,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1024, 1024), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1024,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf269, (1024, 1024), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf270, (1024,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1024,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf272, (1024,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf273, (4096, 1024), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf274, (4096,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1024, 4096), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1024,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf277, (1024,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf278, (1024,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1024, 1024), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1024,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf281, (1024, 1024), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf282, (1024,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf283, (1024, 1024), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf284, (1024,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf285, (1024, 1024), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1024,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf287, (1024,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf288, (1024,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf289, (1024, 1024), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1024,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1024, 1024), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1024,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1024, 1024), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1024,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1024, 1024), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf296, (1024,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf297, (1024,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf298, (1024,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf299, (4096, 1024), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf300, (4096,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf301, (1024, 4096), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1024,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1024,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1024,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1024, 1024), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1024,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf307, (1024, 1024), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf308, (1024,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1024, 1024), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf310, (1024,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf311, (1024, 1024), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf312, (1024,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf313, (1024,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf314, (1024,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf315, (1024, 1024), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1024,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1024, 1024), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1024,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1024, 1024), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf320, (1024,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf321, (1024, 1024), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1024,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf323, (1024,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1024,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf325, (4096, 1024), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf326, (4096,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1024, 4096), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf328, (1024,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1024,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1024,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1024, 1024), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1024,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1024, 1024), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf334, (1024,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf335, (1024, 1024), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf336, (1024,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf337, (1024, 1024), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf338, (1024,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1024,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf340, (1024,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf341, (1024, 1024), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf342, (1024,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1024, 1024), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1024,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1024, 1024), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1024,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1024, 1024), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1024,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1024,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1024,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf351, (4096, 1024), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf352, (4096,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf353, (1024, 4096), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf354, (1024,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1024,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1024,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf357, (1024, 1024), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf358, (1024,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf359, (1024, 1024), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf360, (1024,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf361, (1024, 1024), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf362, (1024,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf363, (1024, 1024), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf364, (1024,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf365, (1024,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf366, (1024,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf367, (1024, 1024), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf368, (1024,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf369, (1024, 1024), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf370, (1024,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf371, (1024, 1024), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf372, (1024,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf373, (1024, 1024), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf374, (1024,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf375, (1024,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf376, (1024,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf377, (4096, 1024), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf378, (4096,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf379, (1024, 4096), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf380, (1024,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf381, (1024,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf382, (1024,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf383, (1024, 1024), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf384, (1024,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf385, (1024, 1024), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf386, (1024,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf387, (1024, 1024), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf388, (1024,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf389, (1024, 1024), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf390, (1024,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf391, (1024,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf392, (1024,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf393, (1024, 1024), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf394, (1024,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf395, (1024, 1024), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf396, (1024,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf397, (1024, 1024), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf398, (1024,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf399, (1024, 1024), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf400, (1024,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf401, (1024,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf402, (1024,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf403, (4096, 1024), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf404, (4096,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf405, (1024, 4096), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf406, (1024,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf407, (1024,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf408, (1024,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf409, (1024, 1024), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf410, (1024,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf411, (1024, 1024), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf412, (1024,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf413, (1024, 1024), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf414, (1024,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf415, (1024, 1024), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf416, (1024,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf417, (1024,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf418, (1024,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf419, (1024, 1024), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf420, (1024,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf421, (1024, 1024), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf422, (1024,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf423, (1024, 1024), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf424, (1024,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf425, (1024, 1024), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf426, (1024,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf427, (1024,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf428, (1024,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf429, (4096, 1024), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf430, (4096,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf431, (1024, 4096), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf432, (1024,), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf433, (1024,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf434, (1024,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf435, (1024, 1024), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf436, (1024,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf437, (1024, 1024), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf438, (1024,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf439, (1024, 1024), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf440, (1024,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf441, (1024, 1024), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf442, (1024,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf443, (1024,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf444, (1024,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf445, (1024, 1024), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf446, (1024,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf447, (1024, 1024), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf448, (1024,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf449, (1024, 1024), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf450, (1024,), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf451, (1024, 1024), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf452, (1024,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf453, (1024,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf454, (1024,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf455, (4096, 1024), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf456, (4096,), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf457, (1024, 4096), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf458, (1024,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf459, (1024,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf460, (1024,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf461, (1024, 1024), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf462, (1024,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf463, (1024, 1024), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf464, (1024,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf465, (1024, 1024), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf466, (1024,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf467, (1024, 1024), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf468, (1024,), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf469, (1024,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf470, (1024,), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf471, (1024, 1024), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf472, (1024,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf473, (1024, 1024), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf474, (1024,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf475, (1024, 1024), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf476, (1024,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf477, (1024, 1024), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf478, (1024,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf479, (1024,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf480, (1024,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf481, (4096, 1024), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf482, (4096,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf483, (1024, 4096), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf484, (1024,), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf485, (1024,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf486, (1024,), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf487, (1024, 1024), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf488, (1024,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf489, (1024, 1024), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf490, (1024,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf491, (1024, 1024), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf492, (1024,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf493, (1024, 1024), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf494, (1024,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf495, (1024,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf496, (1024,), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf497, (1024, 1024), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf498, (1024,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf499, (1024, 1024), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf500, (1024,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf501, (1024, 1024), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf502, (1024,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf503, (1024, 1024), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf504, (1024,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf505, (1024,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf506, (1024,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf507, (4096, 1024), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf508, (4096,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf509, (1024, 4096), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf510, (1024,), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf511, (1024,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf512, (1024,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 201060, device=device(type='cuda', index=0))
    reader.tensor(buf513, (1, 50265), is_leaf=True)  # arg513_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)