
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1):
        view = torch.ops.aten.view.default(arg1_1, [-1, 128])
        embedding = torch.ops.aten.embedding.default(arg2_1, view, 1)
        mul = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
        ne = torch.ops.aten.ne.Scalar(view, 1);  view = None
        convert_element_type = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        cumsum = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_1 = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        view_1 = torch.ops.aten.view.default(add_1, [-1]);  add_1 = None
        index = torch.ops.aten.index.Tensor(arg3_1, [view_1]);  arg3_1 = view_1 = None
        view_2 = torch.ops.aten.view.default(index, [16, 128, 1024]);  index = None
        add_2 = torch.ops.aten.add.Tensor(mul, view_2);  mul = view_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub = torch.ops.aten.sub.Tensor(add_2, getitem_1);  getitem_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        view_3 = torch.ops.aten.view.default(add_4, [2048, 1024])
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view_3, permute);  arg7_1 = view_3 = permute = None
        view_4 = torch.ops.aten.view.default(addmm, [16, 128, 1024]);  addmm = None
        view_5 = torch.ops.aten.view.default(add_4, [2048, 1024])
        permute_1 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_5, permute_1);  arg9_1 = view_5 = permute_1 = None
        view_6 = torch.ops.aten.view.default(addmm_1, [16, 128, 1024]);  addmm_1 = None
        view_7 = torch.ops.aten.view.default(view_6, [16, -1, 16, 64]);  view_6 = None
        permute_2 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_8 = torch.ops.aten.view.default(add_4, [2048, 1024]);  add_4 = None
        permute_3 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_8, permute_3);  arg11_1 = view_8 = permute_3 = None
        view_9 = torch.ops.aten.view.default(addmm_2, [16, 128, 1024]);  addmm_2 = None
        view_10 = torch.ops.aten.view.default(view_9, [16, -1, 16, 64]);  view_9 = None
        permute_4 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_11 = torch.ops.aten.view.default(view_4, [16, 128, 16, 64]);  view_4 = None
        permute_5 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_3, clone_1, clone_2, None, False);  clone_3 = clone_1 = clone_2 = None
        getitem_2 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_12 = torch.ops.aten.view.default(permute_6, [16, 128, 1024]);  permute_6 = None
        view_13 = torch.ops.aten.view.default(view_12, [2048, 1024]);  view_12 = None
        permute_7 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_13, permute_7);  arg13_1 = view_13 = permute_7 = None
        view_14 = torch.ops.aten.view.default(addmm_3, [16, 128, 1024]);  addmm_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_2, view_14);  add_2 = view_14 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_5, getitem_7);  getitem_7 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        view_15 = torch.ops.aten.view.default(add_7, [2048, 1024]);  add_7 = None
        permute_8 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_15, permute_8);  arg17_1 = view_15 = permute_8 = None
        view_16 = torch.ops.aten.view.default(addmm_4, [16, 128, 4096]);  addmm_4 = None
        relu = torch.ops.aten.relu.default(view_16);  view_16 = None
        view_17 = torch.ops.aten.view.default(relu, [2048, 4096]);  relu = None
        permute_9 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_17, permute_9);  arg19_1 = view_17 = permute_9 = None
        view_18 = torch.ops.aten.view.default(addmm_5, [16, 128, 1024]);  addmm_5 = None
        add_8 = torch.ops.aten.add.Tensor(add_5, view_18);  add_5 = view_18 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_9);  getitem_9 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg20_1);  mul_6 = arg20_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_7, arg21_1);  mul_7 = arg21_1 = None
        view_19 = torch.ops.aten.view.default(add_10, [2048, 1024])
        permute_10 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_19, permute_10);  arg23_1 = view_19 = permute_10 = None
        view_20 = torch.ops.aten.view.default(addmm_6, [16, 128, 1024]);  addmm_6 = None
        view_21 = torch.ops.aten.view.default(add_10, [2048, 1024])
        permute_11 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_21, permute_11);  arg25_1 = view_21 = permute_11 = None
        view_22 = torch.ops.aten.view.default(addmm_7, [16, 128, 1024]);  addmm_7 = None
        view_23 = torch.ops.aten.view.default(view_22, [16, -1, 16, 64]);  view_22 = None
        permute_12 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        clone_7 = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_24 = torch.ops.aten.view.default(add_10, [2048, 1024]);  add_10 = None
        permute_13 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg27_1, view_24, permute_13);  arg27_1 = view_24 = permute_13 = None
        view_25 = torch.ops.aten.view.default(addmm_8, [16, 128, 1024]);  addmm_8 = None
        view_26 = torch.ops.aten.view.default(view_25, [16, -1, 16, 64]);  view_25 = None
        permute_14 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_8 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_27 = torch.ops.aten.view.default(view_20, [16, 128, 16, 64]);  view_20 = None
        permute_15 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        clone_9 = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_9, clone_7, clone_8, None, False);  clone_9 = clone_7 = clone_8 = None
        getitem_10 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_28 = torch.ops.aten.view.default(permute_16, [16, 128, 1024]);  permute_16 = None
        view_29 = torch.ops.aten.view.default(view_28, [2048, 1024]);  view_28 = None
        permute_17 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg29_1, view_29, permute_17);  arg29_1 = view_29 = permute_17 = None
        view_30 = torch.ops.aten.view.default(addmm_9, [16, 128, 1024]);  addmm_9 = None
        add_11 = torch.ops.aten.add.Tensor(add_8, view_30);  add_8 = view_30 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_11, getitem_15);  getitem_15 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg30_1);  mul_8 = arg30_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_9, arg31_1);  mul_9 = arg31_1 = None
        view_31 = torch.ops.aten.view.default(add_13, [2048, 1024]);  add_13 = None
        permute_18 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg33_1, view_31, permute_18);  arg33_1 = view_31 = permute_18 = None
        view_32 = torch.ops.aten.view.default(addmm_10, [16, 128, 4096]);  addmm_10 = None
        relu_1 = torch.ops.aten.relu.default(view_32);  view_32 = None
        view_33 = torch.ops.aten.view.default(relu_1, [2048, 4096]);  relu_1 = None
        permute_19 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg35_1, view_33, permute_19);  arg35_1 = view_33 = permute_19 = None
        view_34 = torch.ops.aten.view.default(addmm_11, [16, 128, 1024]);  addmm_11 = None
        add_14 = torch.ops.aten.add.Tensor(add_11, view_34);  add_11 = view_34 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_14, getitem_17);  getitem_17 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg36_1);  mul_10 = arg36_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_11, arg37_1);  mul_11 = arg37_1 = None
        view_35 = torch.ops.aten.view.default(add_16, [2048, 1024])
        permute_20 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg39_1, view_35, permute_20);  arg39_1 = view_35 = permute_20 = None
        view_36 = torch.ops.aten.view.default(addmm_12, [16, 128, 1024]);  addmm_12 = None
        view_37 = torch.ops.aten.view.default(add_16, [2048, 1024])
        permute_21 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg41_1, view_37, permute_21);  arg41_1 = view_37 = permute_21 = None
        view_38 = torch.ops.aten.view.default(addmm_13, [16, 128, 1024]);  addmm_13 = None
        view_39 = torch.ops.aten.view.default(view_38, [16, -1, 16, 64]);  view_38 = None
        permute_22 = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        clone_13 = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_40 = torch.ops.aten.view.default(add_16, [2048, 1024]);  add_16 = None
        permute_23 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg43_1, view_40, permute_23);  arg43_1 = view_40 = permute_23 = None
        view_41 = torch.ops.aten.view.default(addmm_14, [16, 128, 1024]);  addmm_14 = None
        view_42 = torch.ops.aten.view.default(view_41, [16, -1, 16, 64]);  view_41 = None
        permute_24 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_14 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_43 = torch.ops.aten.view.default(view_36, [16, 128, 16, 64]);  view_36 = None
        permute_25 = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        clone_15 = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_15, clone_13, clone_14, None, False);  clone_15 = clone_13 = clone_14 = None
        getitem_18 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_44 = torch.ops.aten.view.default(permute_26, [16, 128, 1024]);  permute_26 = None
        view_45 = torch.ops.aten.view.default(view_44, [2048, 1024]);  view_44 = None
        permute_27 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg45_1, view_45, permute_27);  arg45_1 = view_45 = permute_27 = None
        view_46 = torch.ops.aten.view.default(addmm_15, [16, 128, 1024]);  addmm_15 = None
        add_17 = torch.ops.aten.add.Tensor(add_14, view_46);  add_14 = view_46 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg46_1);  mul_12 = arg46_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_13, arg47_1);  mul_13 = arg47_1 = None
        view_47 = torch.ops.aten.view.default(add_19, [2048, 1024]);  add_19 = None
        permute_28 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg49_1, view_47, permute_28);  arg49_1 = view_47 = permute_28 = None
        view_48 = torch.ops.aten.view.default(addmm_16, [16, 128, 4096]);  addmm_16 = None
        relu_2 = torch.ops.aten.relu.default(view_48);  view_48 = None
        view_49 = torch.ops.aten.view.default(relu_2, [2048, 4096]);  relu_2 = None
        permute_29 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg51_1, view_49, permute_29);  arg51_1 = view_49 = permute_29 = None
        view_50 = torch.ops.aten.view.default(addmm_17, [16, 128, 1024]);  addmm_17 = None
        add_20 = torch.ops.aten.add.Tensor(add_17, view_50);  add_17 = view_50 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_20, getitem_25);  getitem_25 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg52_1);  mul_14 = arg52_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_15, arg53_1);  mul_15 = arg53_1 = None
        view_51 = torch.ops.aten.view.default(add_22, [2048, 1024])
        permute_30 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg55_1, view_51, permute_30);  arg55_1 = view_51 = permute_30 = None
        view_52 = torch.ops.aten.view.default(addmm_18, [16, 128, 1024]);  addmm_18 = None
        view_53 = torch.ops.aten.view.default(add_22, [2048, 1024])
        permute_31 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg57_1, view_53, permute_31);  arg57_1 = view_53 = permute_31 = None
        view_54 = torch.ops.aten.view.default(addmm_19, [16, 128, 1024]);  addmm_19 = None
        view_55 = torch.ops.aten.view.default(view_54, [16, -1, 16, 64]);  view_54 = None
        permute_32 = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        clone_19 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_56 = torch.ops.aten.view.default(add_22, [2048, 1024]);  add_22 = None
        permute_33 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg59_1, view_56, permute_33);  arg59_1 = view_56 = permute_33 = None
        view_57 = torch.ops.aten.view.default(addmm_20, [16, 128, 1024]);  addmm_20 = None
        view_58 = torch.ops.aten.view.default(view_57, [16, -1, 16, 64]);  view_57 = None
        permute_34 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_20 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_59 = torch.ops.aten.view.default(view_52, [16, 128, 16, 64]);  view_52 = None
        permute_35 = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        clone_21 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_21, clone_19, clone_20, None, False);  clone_21 = clone_19 = clone_20 = None
        getitem_26 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_60 = torch.ops.aten.view.default(permute_36, [16, 128, 1024]);  permute_36 = None
        view_61 = torch.ops.aten.view.default(view_60, [2048, 1024]);  view_60 = None
        permute_37 = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg61_1, view_61, permute_37);  arg61_1 = view_61 = permute_37 = None
        view_62 = torch.ops.aten.view.default(addmm_21, [16, 128, 1024]);  addmm_21 = None
        add_23 = torch.ops.aten.add.Tensor(add_20, view_62);  add_20 = view_62 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_23, getitem_31);  getitem_31 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg62_1);  mul_16 = arg62_1 = None
        add_25 = torch.ops.aten.add.Tensor(mul_17, arg63_1);  mul_17 = arg63_1 = None
        view_63 = torch.ops.aten.view.default(add_25, [2048, 1024]);  add_25 = None
        permute_38 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg65_1, view_63, permute_38);  arg65_1 = view_63 = permute_38 = None
        view_64 = torch.ops.aten.view.default(addmm_22, [16, 128, 4096]);  addmm_22 = None
        relu_3 = torch.ops.aten.relu.default(view_64);  view_64 = None
        view_65 = torch.ops.aten.view.default(relu_3, [2048, 4096]);  relu_3 = None
        permute_39 = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg67_1, view_65, permute_39);  arg67_1 = view_65 = permute_39 = None
        view_66 = torch.ops.aten.view.default(addmm_23, [16, 128, 1024]);  addmm_23 = None
        add_26 = torch.ops.aten.add.Tensor(add_23, view_66);  add_23 = view_66 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_26, getitem_33);  getitem_33 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg68_1);  mul_18 = arg68_1 = None
        add_28 = torch.ops.aten.add.Tensor(mul_19, arg69_1);  mul_19 = arg69_1 = None
        view_67 = torch.ops.aten.view.default(add_28, [2048, 1024])
        permute_40 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg71_1, view_67, permute_40);  arg71_1 = view_67 = permute_40 = None
        view_68 = torch.ops.aten.view.default(addmm_24, [16, 128, 1024]);  addmm_24 = None
        view_69 = torch.ops.aten.view.default(add_28, [2048, 1024])
        permute_41 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg73_1, view_69, permute_41);  arg73_1 = view_69 = permute_41 = None
        view_70 = torch.ops.aten.view.default(addmm_25, [16, 128, 1024]);  addmm_25 = None
        view_71 = torch.ops.aten.view.default(view_70, [16, -1, 16, 64]);  view_70 = None
        permute_42 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_25 = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_72 = torch.ops.aten.view.default(add_28, [2048, 1024]);  add_28 = None
        permute_43 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg75_1, view_72, permute_43);  arg75_1 = view_72 = permute_43 = None
        view_73 = torch.ops.aten.view.default(addmm_26, [16, 128, 1024]);  addmm_26 = None
        view_74 = torch.ops.aten.view.default(view_73, [16, -1, 16, 64]);  view_73 = None
        permute_44 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_26 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_75 = torch.ops.aten.view.default(view_68, [16, 128, 16, 64]);  view_68 = None
        permute_45 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_27 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_27, clone_25, clone_26, None, False);  clone_27 = clone_25 = clone_26 = None
        getitem_34 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_76 = torch.ops.aten.view.default(permute_46, [16, 128, 1024]);  permute_46 = None
        view_77 = torch.ops.aten.view.default(view_76, [2048, 1024]);  view_76 = None
        permute_47 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg77_1, view_77, permute_47);  arg77_1 = view_77 = permute_47 = None
        view_78 = torch.ops.aten.view.default(addmm_27, [16, 128, 1024]);  addmm_27 = None
        add_29 = torch.ops.aten.add.Tensor(add_26, view_78);  add_26 = view_78 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_29, getitem_39);  getitem_39 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg78_1);  mul_20 = arg78_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_21, arg79_1);  mul_21 = arg79_1 = None
        view_79 = torch.ops.aten.view.default(add_31, [2048, 1024]);  add_31 = None
        permute_48 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg81_1, view_79, permute_48);  arg81_1 = view_79 = permute_48 = None
        view_80 = torch.ops.aten.view.default(addmm_28, [16, 128, 4096]);  addmm_28 = None
        relu_4 = torch.ops.aten.relu.default(view_80);  view_80 = None
        view_81 = torch.ops.aten.view.default(relu_4, [2048, 4096]);  relu_4 = None
        permute_49 = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg83_1, view_81, permute_49);  arg83_1 = view_81 = permute_49 = None
        view_82 = torch.ops.aten.view.default(addmm_29, [16, 128, 1024]);  addmm_29 = None
        add_32 = torch.ops.aten.add.Tensor(add_29, view_82);  add_29 = view_82 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_32, getitem_41);  getitem_41 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg84_1);  mul_22 = arg84_1 = None
        add_34 = torch.ops.aten.add.Tensor(mul_23, arg85_1);  mul_23 = arg85_1 = None
        view_83 = torch.ops.aten.view.default(add_34, [2048, 1024])
        permute_50 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg87_1, view_83, permute_50);  arg87_1 = view_83 = permute_50 = None
        view_84 = torch.ops.aten.view.default(addmm_30, [16, 128, 1024]);  addmm_30 = None
        view_85 = torch.ops.aten.view.default(add_34, [2048, 1024])
        permute_51 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg89_1, view_85, permute_51);  arg89_1 = view_85 = permute_51 = None
        view_86 = torch.ops.aten.view.default(addmm_31, [16, 128, 1024]);  addmm_31 = None
        view_87 = torch.ops.aten.view.default(view_86, [16, -1, 16, 64]);  view_86 = None
        permute_52 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        clone_31 = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_88 = torch.ops.aten.view.default(add_34, [2048, 1024]);  add_34 = None
        permute_53 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg91_1, view_88, permute_53);  arg91_1 = view_88 = permute_53 = None
        view_89 = torch.ops.aten.view.default(addmm_32, [16, 128, 1024]);  addmm_32 = None
        view_90 = torch.ops.aten.view.default(view_89, [16, -1, 16, 64]);  view_89 = None
        permute_54 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_32 = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_91 = torch.ops.aten.view.default(view_84, [16, 128, 16, 64]);  view_84 = None
        permute_55 = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None
        clone_33 = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_33, clone_31, clone_32, None, False);  clone_33 = clone_31 = clone_32 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_92 = torch.ops.aten.view.default(permute_56, [16, 128, 1024]);  permute_56 = None
        view_93 = torch.ops.aten.view.default(view_92, [2048, 1024]);  view_92 = None
        permute_57 = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg93_1, view_93, permute_57);  arg93_1 = view_93 = permute_57 = None
        view_94 = torch.ops.aten.view.default(addmm_33, [16, 128, 1024]);  addmm_33 = None
        add_35 = torch.ops.aten.add.Tensor(add_32, view_94);  add_32 = view_94 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_35, getitem_47);  getitem_47 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg94_1);  mul_24 = arg94_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_25, arg95_1);  mul_25 = arg95_1 = None
        view_95 = torch.ops.aten.view.default(add_37, [2048, 1024]);  add_37 = None
        permute_58 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg97_1, view_95, permute_58);  arg97_1 = view_95 = permute_58 = None
        view_96 = torch.ops.aten.view.default(addmm_34, [16, 128, 4096]);  addmm_34 = None
        relu_5 = torch.ops.aten.relu.default(view_96);  view_96 = None
        view_97 = torch.ops.aten.view.default(relu_5, [2048, 4096]);  relu_5 = None
        permute_59 = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg99_1, view_97, permute_59);  arg99_1 = view_97 = permute_59 = None
        view_98 = torch.ops.aten.view.default(addmm_35, [16, 128, 1024]);  addmm_35 = None
        add_38 = torch.ops.aten.add.Tensor(add_35, view_98);  add_35 = view_98 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_38, getitem_49);  getitem_49 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg100_1);  mul_26 = arg100_1 = None
        add_40 = torch.ops.aten.add.Tensor(mul_27, arg101_1);  mul_27 = arg101_1 = None
        view_99 = torch.ops.aten.view.default(add_40, [2048, 1024])
        permute_60 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg103_1, view_99, permute_60);  arg103_1 = view_99 = permute_60 = None
        view_100 = torch.ops.aten.view.default(addmm_36, [16, 128, 1024]);  addmm_36 = None
        view_101 = torch.ops.aten.view.default(add_40, [2048, 1024])
        permute_61 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg105_1, view_101, permute_61);  arg105_1 = view_101 = permute_61 = None
        view_102 = torch.ops.aten.view.default(addmm_37, [16, 128, 1024]);  addmm_37 = None
        view_103 = torch.ops.aten.view.default(view_102, [16, -1, 16, 64]);  view_102 = None
        permute_62 = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        clone_37 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_104 = torch.ops.aten.view.default(add_40, [2048, 1024]);  add_40 = None
        permute_63 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg107_1, view_104, permute_63);  arg107_1 = view_104 = permute_63 = None
        view_105 = torch.ops.aten.view.default(addmm_38, [16, 128, 1024]);  addmm_38 = None
        view_106 = torch.ops.aten.view.default(view_105, [16, -1, 16, 64]);  view_105 = None
        permute_64 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_38 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_107 = torch.ops.aten.view.default(view_100, [16, 128, 16, 64]);  view_100 = None
        permute_65 = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        clone_39 = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_39, clone_37, clone_38, None, False);  clone_39 = clone_37 = clone_38 = None
        getitem_50 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_66 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        view_108 = torch.ops.aten.view.default(permute_66, [16, 128, 1024]);  permute_66 = None
        view_109 = torch.ops.aten.view.default(view_108, [2048, 1024]);  view_108 = None
        permute_67 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg109_1, view_109, permute_67);  arg109_1 = view_109 = permute_67 = None
        view_110 = torch.ops.aten.view.default(addmm_39, [16, 128, 1024]);  addmm_39 = None
        add_41 = torch.ops.aten.add.Tensor(add_38, view_110);  add_38 = view_110 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_13[0]
        getitem_55 = var_mean_13[1];  var_mean_13 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_41, getitem_55);  getitem_55 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg110_1);  mul_28 = arg110_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_29, arg111_1);  mul_29 = arg111_1 = None
        view_111 = torch.ops.aten.view.default(add_43, [2048, 1024]);  add_43 = None
        permute_68 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg113_1, view_111, permute_68);  arg113_1 = view_111 = permute_68 = None
        view_112 = torch.ops.aten.view.default(addmm_40, [16, 128, 4096]);  addmm_40 = None
        relu_6 = torch.ops.aten.relu.default(view_112);  view_112 = None
        view_113 = torch.ops.aten.view.default(relu_6, [2048, 4096]);  relu_6 = None
        permute_69 = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg115_1, view_113, permute_69);  arg115_1 = view_113 = permute_69 = None
        view_114 = torch.ops.aten.view.default(addmm_41, [16, 128, 1024]);  addmm_41 = None
        add_44 = torch.ops.aten.add.Tensor(add_41, view_114);  add_41 = view_114 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_44, getitem_57);  getitem_57 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg116_1);  mul_30 = arg116_1 = None
        add_46 = torch.ops.aten.add.Tensor(mul_31, arg117_1);  mul_31 = arg117_1 = None
        view_115 = torch.ops.aten.view.default(add_46, [2048, 1024])
        permute_70 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg119_1, view_115, permute_70);  arg119_1 = view_115 = permute_70 = None
        view_116 = torch.ops.aten.view.default(addmm_42, [16, 128, 1024]);  addmm_42 = None
        view_117 = torch.ops.aten.view.default(add_46, [2048, 1024])
        permute_71 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg121_1, view_117, permute_71);  arg121_1 = view_117 = permute_71 = None
        view_118 = torch.ops.aten.view.default(addmm_43, [16, 128, 1024]);  addmm_43 = None
        view_119 = torch.ops.aten.view.default(view_118, [16, -1, 16, 64]);  view_118 = None
        permute_72 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        clone_43 = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_120 = torch.ops.aten.view.default(add_46, [2048, 1024]);  add_46 = None
        permute_73 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg123_1, view_120, permute_73);  arg123_1 = view_120 = permute_73 = None
        view_121 = torch.ops.aten.view.default(addmm_44, [16, 128, 1024]);  addmm_44 = None
        view_122 = torch.ops.aten.view.default(view_121, [16, -1, 16, 64]);  view_121 = None
        permute_74 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_44 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_123 = torch.ops.aten.view.default(view_116, [16, 128, 16, 64]);  view_116 = None
        permute_75 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        clone_45 = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_45, clone_43, clone_44, None, False);  clone_45 = clone_43 = clone_44 = None
        getitem_58 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_76 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        view_124 = torch.ops.aten.view.default(permute_76, [16, 128, 1024]);  permute_76 = None
        view_125 = torch.ops.aten.view.default(view_124, [2048, 1024]);  view_124 = None
        permute_77 = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg125_1, view_125, permute_77);  arg125_1 = view_125 = permute_77 = None
        view_126 = torch.ops.aten.view.default(addmm_45, [16, 128, 1024]);  addmm_45 = None
        add_47 = torch.ops.aten.add.Tensor(add_44, view_126);  add_44 = view_126 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_15[0]
        getitem_63 = var_mean_15[1];  var_mean_15 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_47, getitem_63);  getitem_63 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg126_1);  mul_32 = arg126_1 = None
        add_49 = torch.ops.aten.add.Tensor(mul_33, arg127_1);  mul_33 = arg127_1 = None
        view_127 = torch.ops.aten.view.default(add_49, [2048, 1024]);  add_49 = None
        permute_78 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg129_1, view_127, permute_78);  arg129_1 = view_127 = permute_78 = None
        view_128 = torch.ops.aten.view.default(addmm_46, [16, 128, 4096]);  addmm_46 = None
        relu_7 = torch.ops.aten.relu.default(view_128);  view_128 = None
        view_129 = torch.ops.aten.view.default(relu_7, [2048, 4096]);  relu_7 = None
        permute_79 = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg131_1, view_129, permute_79);  arg131_1 = view_129 = permute_79 = None
        view_130 = torch.ops.aten.view.default(addmm_47, [16, 128, 1024]);  addmm_47 = None
        add_50 = torch.ops.aten.add.Tensor(add_47, view_130);  add_47 = view_130 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_50, getitem_65);  getitem_65 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg132_1);  mul_34 = arg132_1 = None
        add_52 = torch.ops.aten.add.Tensor(mul_35, arg133_1);  mul_35 = arg133_1 = None
        view_131 = torch.ops.aten.view.default(add_52, [2048, 1024])
        permute_80 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg135_1, view_131, permute_80);  arg135_1 = view_131 = permute_80 = None
        view_132 = torch.ops.aten.view.default(addmm_48, [16, 128, 1024]);  addmm_48 = None
        view_133 = torch.ops.aten.view.default(add_52, [2048, 1024])
        permute_81 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg137_1, view_133, permute_81);  arg137_1 = view_133 = permute_81 = None
        view_134 = torch.ops.aten.view.default(addmm_49, [16, 128, 1024]);  addmm_49 = None
        view_135 = torch.ops.aten.view.default(view_134, [16, -1, 16, 64]);  view_134 = None
        permute_82 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        clone_49 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_136 = torch.ops.aten.view.default(add_52, [2048, 1024]);  add_52 = None
        permute_83 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg139_1, view_136, permute_83);  arg139_1 = view_136 = permute_83 = None
        view_137 = torch.ops.aten.view.default(addmm_50, [16, 128, 1024]);  addmm_50 = None
        view_138 = torch.ops.aten.view.default(view_137, [16, -1, 16, 64]);  view_137 = None
        permute_84 = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_50 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_139 = torch.ops.aten.view.default(view_132, [16, 128, 16, 64]);  view_132 = None
        permute_85 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        clone_51 = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_51, clone_49, clone_50, None, False);  clone_51 = clone_49 = clone_50 = None
        getitem_66 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_86 = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        view_140 = torch.ops.aten.view.default(permute_86, [16, 128, 1024]);  permute_86 = None
        view_141 = torch.ops.aten.view.default(view_140, [2048, 1024]);  view_140 = None
        permute_87 = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg141_1, view_141, permute_87);  arg141_1 = view_141 = permute_87 = None
        view_142 = torch.ops.aten.view.default(addmm_51, [16, 128, 1024]);  addmm_51 = None
        add_53 = torch.ops.aten.add.Tensor(add_50, view_142);  add_50 = view_142 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_17[0]
        getitem_71 = var_mean_17[1];  var_mean_17 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_53, getitem_71);  getitem_71 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg142_1);  mul_36 = arg142_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_37, arg143_1);  mul_37 = arg143_1 = None
        view_143 = torch.ops.aten.view.default(add_55, [2048, 1024]);  add_55 = None
        permute_88 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg145_1, view_143, permute_88);  arg145_1 = view_143 = permute_88 = None
        view_144 = torch.ops.aten.view.default(addmm_52, [16, 128, 4096]);  addmm_52 = None
        relu_8 = torch.ops.aten.relu.default(view_144);  view_144 = None
        view_145 = torch.ops.aten.view.default(relu_8, [2048, 4096]);  relu_8 = None
        permute_89 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg147_1, view_145, permute_89);  arg147_1 = view_145 = permute_89 = None
        view_146 = torch.ops.aten.view.default(addmm_53, [16, 128, 1024]);  addmm_53 = None
        add_56 = torch.ops.aten.add.Tensor(add_53, view_146);  add_53 = view_146 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_56, getitem_73);  getitem_73 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg148_1);  mul_38 = arg148_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_39, arg149_1);  mul_39 = arg149_1 = None
        view_147 = torch.ops.aten.view.default(add_58, [2048, 1024])
        permute_90 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg151_1, view_147, permute_90);  arg151_1 = view_147 = permute_90 = None
        view_148 = torch.ops.aten.view.default(addmm_54, [16, 128, 1024]);  addmm_54 = None
        view_149 = torch.ops.aten.view.default(add_58, [2048, 1024])
        permute_91 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg153_1, view_149, permute_91);  arg153_1 = view_149 = permute_91 = None
        view_150 = torch.ops.aten.view.default(addmm_55, [16, 128, 1024]);  addmm_55 = None
        view_151 = torch.ops.aten.view.default(view_150, [16, -1, 16, 64]);  view_150 = None
        permute_92 = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        clone_55 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_152 = torch.ops.aten.view.default(add_58, [2048, 1024]);  add_58 = None
        permute_93 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg155_1, view_152, permute_93);  arg155_1 = view_152 = permute_93 = None
        view_153 = torch.ops.aten.view.default(addmm_56, [16, 128, 1024]);  addmm_56 = None
        view_154 = torch.ops.aten.view.default(view_153, [16, -1, 16, 64]);  view_153 = None
        permute_94 = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_56 = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_155 = torch.ops.aten.view.default(view_148, [16, 128, 16, 64]);  view_148 = None
        permute_95 = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        clone_57 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_57, clone_55, clone_56, None, False);  clone_57 = clone_55 = clone_56 = None
        getitem_74 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_96 = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        view_156 = torch.ops.aten.view.default(permute_96, [16, 128, 1024]);  permute_96 = None
        view_157 = torch.ops.aten.view.default(view_156, [2048, 1024]);  view_156 = None
        permute_97 = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg157_1, view_157, permute_97);  arg157_1 = view_157 = permute_97 = None
        view_158 = torch.ops.aten.view.default(addmm_57, [16, 128, 1024]);  addmm_57 = None
        add_59 = torch.ops.aten.add.Tensor(add_56, view_158);  add_56 = view_158 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_19[0]
        getitem_79 = var_mean_19[1];  var_mean_19 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_59, getitem_79);  getitem_79 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg158_1);  mul_40 = arg158_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_41, arg159_1);  mul_41 = arg159_1 = None
        view_159 = torch.ops.aten.view.default(add_61, [2048, 1024]);  add_61 = None
        permute_98 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg161_1, view_159, permute_98);  arg161_1 = view_159 = permute_98 = None
        view_160 = torch.ops.aten.view.default(addmm_58, [16, 128, 4096]);  addmm_58 = None
        relu_9 = torch.ops.aten.relu.default(view_160);  view_160 = None
        view_161 = torch.ops.aten.view.default(relu_9, [2048, 4096]);  relu_9 = None
        permute_99 = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg163_1, view_161, permute_99);  arg163_1 = view_161 = permute_99 = None
        view_162 = torch.ops.aten.view.default(addmm_59, [16, 128, 1024]);  addmm_59 = None
        add_62 = torch.ops.aten.add.Tensor(add_59, view_162);  add_59 = view_162 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_63 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_62, getitem_81);  getitem_81 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg164_1);  mul_42 = arg164_1 = None
        add_64 = torch.ops.aten.add.Tensor(mul_43, arg165_1);  mul_43 = arg165_1 = None
        view_163 = torch.ops.aten.view.default(add_64, [2048, 1024])
        permute_100 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg167_1, view_163, permute_100);  arg167_1 = view_163 = permute_100 = None
        view_164 = torch.ops.aten.view.default(addmm_60, [16, 128, 1024]);  addmm_60 = None
        view_165 = torch.ops.aten.view.default(add_64, [2048, 1024])
        permute_101 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg169_1, view_165, permute_101);  arg169_1 = view_165 = permute_101 = None
        view_166 = torch.ops.aten.view.default(addmm_61, [16, 128, 1024]);  addmm_61 = None
        view_167 = torch.ops.aten.view.default(view_166, [16, -1, 16, 64]);  view_166 = None
        permute_102 = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        clone_61 = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_168 = torch.ops.aten.view.default(add_64, [2048, 1024]);  add_64 = None
        permute_103 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg171_1, view_168, permute_103);  arg171_1 = view_168 = permute_103 = None
        view_169 = torch.ops.aten.view.default(addmm_62, [16, 128, 1024]);  addmm_62 = None
        view_170 = torch.ops.aten.view.default(view_169, [16, -1, 16, 64]);  view_169 = None
        permute_104 = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_62 = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_171 = torch.ops.aten.view.default(view_164, [16, 128, 16, 64]);  view_164 = None
        permute_105 = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        clone_63 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_63, clone_61, clone_62, None, False);  clone_63 = clone_61 = clone_62 = None
        getitem_82 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_106 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_172 = torch.ops.aten.view.default(permute_106, [16, 128, 1024]);  permute_106 = None
        view_173 = torch.ops.aten.view.default(view_172, [2048, 1024]);  view_172 = None
        permute_107 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg173_1, view_173, permute_107);  arg173_1 = view_173 = permute_107 = None
        view_174 = torch.ops.aten.view.default(addmm_63, [16, 128, 1024]);  addmm_63 = None
        add_65 = torch.ops.aten.add.Tensor(add_62, view_174);  add_62 = view_174 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_21[0]
        getitem_87 = var_mean_21[1];  var_mean_21 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_65, getitem_87);  getitem_87 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg174_1);  mul_44 = arg174_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_45, arg175_1);  mul_45 = arg175_1 = None
        view_175 = torch.ops.aten.view.default(add_67, [2048, 1024]);  add_67 = None
        permute_108 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg177_1, view_175, permute_108);  arg177_1 = view_175 = permute_108 = None
        view_176 = torch.ops.aten.view.default(addmm_64, [16, 128, 4096]);  addmm_64 = None
        relu_10 = torch.ops.aten.relu.default(view_176);  view_176 = None
        view_177 = torch.ops.aten.view.default(relu_10, [2048, 4096]);  relu_10 = None
        permute_109 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg179_1, view_177, permute_109);  arg179_1 = view_177 = permute_109 = None
        view_178 = torch.ops.aten.view.default(addmm_65, [16, 128, 1024]);  addmm_65 = None
        add_68 = torch.ops.aten.add.Tensor(add_65, view_178);  add_65 = view_178 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_68, getitem_89);  getitem_89 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, arg180_1);  mul_46 = arg180_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_47, arg181_1);  mul_47 = arg181_1 = None
        view_179 = torch.ops.aten.view.default(add_70, [2048, 1024])
        permute_110 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg183_1, view_179, permute_110);  arg183_1 = view_179 = permute_110 = None
        view_180 = torch.ops.aten.view.default(addmm_66, [16, 128, 1024]);  addmm_66 = None
        view_181 = torch.ops.aten.view.default(add_70, [2048, 1024])
        permute_111 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg185_1, view_181, permute_111);  arg185_1 = view_181 = permute_111 = None
        view_182 = torch.ops.aten.view.default(addmm_67, [16, 128, 1024]);  addmm_67 = None
        view_183 = torch.ops.aten.view.default(view_182, [16, -1, 16, 64]);  view_182 = None
        permute_112 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_67 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_184 = torch.ops.aten.view.default(add_70, [2048, 1024]);  add_70 = None
        permute_113 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg187_1, view_184, permute_113);  arg187_1 = view_184 = permute_113 = None
        view_185 = torch.ops.aten.view.default(addmm_68, [16, 128, 1024]);  addmm_68 = None
        view_186 = torch.ops.aten.view.default(view_185, [16, -1, 16, 64]);  view_185 = None
        permute_114 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_68 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_187 = torch.ops.aten.view.default(view_180, [16, 128, 16, 64]);  view_180 = None
        permute_115 = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        clone_69 = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_69, clone_67, clone_68, None, False);  clone_69 = clone_67 = clone_68 = None
        getitem_90 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_116 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        view_188 = torch.ops.aten.view.default(permute_116, [16, 128, 1024]);  permute_116 = None
        view_189 = torch.ops.aten.view.default(view_188, [2048, 1024]);  view_188 = None
        permute_117 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg189_1, view_189, permute_117);  arg189_1 = view_189 = permute_117 = None
        view_190 = torch.ops.aten.view.default(addmm_69, [16, 128, 1024]);  addmm_69 = None
        add_71 = torch.ops.aten.add.Tensor(add_68, view_190);  add_68 = view_190 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_23[0]
        getitem_95 = var_mean_23[1];  var_mean_23 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_71, getitem_95);  getitem_95 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg190_1);  mul_48 = arg190_1 = None
        add_73 = torch.ops.aten.add.Tensor(mul_49, arg191_1);  mul_49 = arg191_1 = None
        view_191 = torch.ops.aten.view.default(add_73, [2048, 1024]);  add_73 = None
        permute_118 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg193_1, view_191, permute_118);  arg193_1 = view_191 = permute_118 = None
        view_192 = torch.ops.aten.view.default(addmm_70, [16, 128, 4096]);  addmm_70 = None
        relu_11 = torch.ops.aten.relu.default(view_192);  view_192 = None
        view_193 = torch.ops.aten.view.default(relu_11, [2048, 4096]);  relu_11 = None
        permute_119 = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg195_1, view_193, permute_119);  arg195_1 = view_193 = permute_119 = None
        view_194 = torch.ops.aten.view.default(addmm_71, [16, 128, 1024]);  addmm_71 = None
        add_74 = torch.ops.aten.add.Tensor(add_71, view_194);  add_71 = view_194 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_74, getitem_97);  add_74 = getitem_97 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg196_1);  mul_50 = arg196_1 = None
        add_76 = torch.ops.aten.add.Tensor(mul_51, arg197_1);  mul_51 = arg197_1 = None
        view_195 = torch.ops.aten.view.default(arg1_1, [-1, 128]);  arg1_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, view_195, 1)
        mul_52 = torch.ops.aten.mul.Tensor(embedding_1, 32.0);  embedding_1 = None
        full_default = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_77 = torch.ops.aten.add.Tensor(iota, 1)
        view_196 = torch.ops.aten.view.default(add_77, [128, 1]);  add_77 = None
        lt = torch.ops.aten.lt.Tensor(iota, view_196);  iota = view_196 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        ne_1 = torch.ops.aten.ne.Scalar(view_195, 1);  view_195 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None
        cumsum_1 = torch.ops.aten.cumsum.default(convert_element_type_3, 1)
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
        add_78 = torch.ops.aten.add.Tensor(convert_element_type_4, 0);  convert_element_type_4 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_78, convert_element_type_3);  add_78 = convert_element_type_3 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_53, torch.int64);  mul_53 = None
        add_79 = torch.ops.aten.add.Tensor(convert_element_type_5, 1);  convert_element_type_5 = None
        view_197 = torch.ops.aten.view.default(add_79, [-1]);  add_79 = None
        index_1 = torch.ops.aten.index.Tensor(arg198_1, [view_197]);  arg198_1 = view_197 = None
        view_198 = torch.ops.aten.view.default(index_1, [16, 128, 1024]);  index_1 = None
        add_80 = torch.ops.aten.add.Tensor(mul_52, view_198);  mul_52 = view_198 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_98 = var_mean_25[0]
        getitem_99 = var_mean_25[1];  var_mean_25 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_80, getitem_99);  getitem_99 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, arg199_1);  mul_54 = arg199_1 = None
        add_82 = torch.ops.aten.add.Tensor(mul_55, arg200_1);  mul_55 = arg200_1 = None
        view_199 = torch.ops.aten.view.default(add_82, [2048, 1024])
        permute_120 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg202_1, view_199, permute_120);  arg202_1 = view_199 = permute_120 = None
        view_200 = torch.ops.aten.view.default(addmm_72, [16, 128, 1024]);  addmm_72 = None
        view_201 = torch.ops.aten.view.default(add_82, [2048, 1024])
        permute_121 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg204_1, view_201, permute_121);  arg204_1 = view_201 = permute_121 = None
        view_202 = torch.ops.aten.view.default(addmm_73, [16, 128, 1024]);  addmm_73 = None
        view_203 = torch.ops.aten.view.default(view_202, [16, -1, 16, 64]);  view_202 = None
        permute_122 = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        clone_74 = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_204 = torch.ops.aten.view.default(add_82, [2048, 1024]);  add_82 = None
        permute_123 = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg206_1, view_204, permute_123);  arg206_1 = view_204 = permute_123 = None
        view_205 = torch.ops.aten.view.default(addmm_74, [16, 128, 1024]);  addmm_74 = None
        view_206 = torch.ops.aten.view.default(view_205, [16, -1, 16, 64]);  view_205 = None
        permute_124 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_75 = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_207 = torch.ops.aten.view.default(view_200, [16, 128, 16, 64]);  view_200 = None
        permute_125 = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        clone_76 = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_3, [16, 1, 128, 128]);  unsqueeze_3 = None
        expand_3 = torch.ops.aten.expand.default(expand_2, [16, 16, 128, 128]);  expand_2 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_76, clone_74, clone_75, expand_3, False);  clone_76 = expand_3 = None
        getitem_100 = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        permute_126 = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        view_208 = torch.ops.aten.view.default(permute_126, [16, 128, 1024]);  permute_126 = None
        view_209 = torch.ops.aten.view.default(view_208, [2048, 1024]);  view_208 = None
        permute_127 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_75 = torch.ops.aten.addmm.default(arg208_1, view_209, permute_127);  arg208_1 = view_209 = permute_127 = None
        view_210 = torch.ops.aten.view.default(addmm_75, [16, 128, 1024]);  addmm_75 = None
        add_83 = torch.ops.aten.add.Tensor(add_80, view_210);  add_80 = view_210 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_104 = var_mean_26[0]
        getitem_105 = var_mean_26[1];  var_mean_26 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_83, getitem_105);  getitem_105 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg209_1);  mul_56 = arg209_1 = None
        add_85 = torch.ops.aten.add.Tensor(mul_57, arg210_1);  mul_57 = arg210_1 = None
        view_211 = torch.ops.aten.view.default(add_85, [2048, 1024]);  add_85 = None
        permute_128 = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_76 = torch.ops.aten.addmm.default(arg212_1, view_211, permute_128);  arg212_1 = view_211 = permute_128 = None
        view_212 = torch.ops.aten.view.default(addmm_76, [16, 128, 1024]);  addmm_76 = None
        view_213 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_129 = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_77 = torch.ops.aten.addmm.default(arg214_1, view_213, permute_129);  arg214_1 = view_213 = permute_129 = None
        view_214 = torch.ops.aten.view.default(addmm_77, [16, 128, 1024]);  addmm_77 = None
        view_215 = torch.ops.aten.view.default(view_214, [16, -1, 16, 64]);  view_214 = None
        permute_130 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        clone_78 = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_216 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_131 = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg216_1, view_216, permute_131);  arg216_1 = view_216 = permute_131 = None
        view_217 = torch.ops.aten.view.default(addmm_78, [16, 128, 1024]);  addmm_78 = None
        view_218 = torch.ops.aten.view.default(view_217, [16, -1, 16, 64]);  view_217 = None
        permute_132 = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_79 = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_219 = torch.ops.aten.view.default(view_212, [16, 128, 16, 64]);  view_212 = None
        permute_133 = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
        clone_80 = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_80, clone_78, clone_79, None, False);  clone_80 = None
        getitem_106 = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        permute_134 = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        view_220 = torch.ops.aten.view.default(permute_134, [16, 128, 1024]);  permute_134 = None
        view_221 = torch.ops.aten.view.default(view_220, [2048, 1024]);  view_220 = None
        permute_135 = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_79 = torch.ops.aten.addmm.default(arg218_1, view_221, permute_135);  arg218_1 = view_221 = permute_135 = None
        view_222 = torch.ops.aten.view.default(addmm_79, [16, 128, 1024]);  addmm_79 = None
        add_86 = torch.ops.aten.add.Tensor(add_83, view_222);  add_83 = view_222 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_27[0]
        getitem_111 = var_mean_27[1];  var_mean_27 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_86, getitem_111);  getitem_111 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg219_1);  mul_58 = arg219_1 = None
        add_88 = torch.ops.aten.add.Tensor(mul_59, arg220_1);  mul_59 = arg220_1 = None
        view_223 = torch.ops.aten.view.default(add_88, [2048, 1024]);  add_88 = None
        permute_136 = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_80 = torch.ops.aten.addmm.default(arg222_1, view_223, permute_136);  arg222_1 = view_223 = permute_136 = None
        view_224 = torch.ops.aten.view.default(addmm_80, [16, 128, 4096]);  addmm_80 = None
        relu_12 = torch.ops.aten.relu.default(view_224);  view_224 = None
        view_225 = torch.ops.aten.view.default(relu_12, [2048, 4096]);  relu_12 = None
        permute_137 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_81 = torch.ops.aten.addmm.default(arg224_1, view_225, permute_137);  arg224_1 = view_225 = permute_137 = None
        view_226 = torch.ops.aten.view.default(addmm_81, [16, 128, 1024]);  addmm_81 = None
        add_89 = torch.ops.aten.add.Tensor(add_86, view_226);  add_86 = view_226 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_112 = var_mean_28[0]
        getitem_113 = var_mean_28[1];  var_mean_28 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_89, getitem_113);  getitem_113 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg225_1);  mul_60 = arg225_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_61, arg226_1);  mul_61 = arg226_1 = None
        view_227 = torch.ops.aten.view.default(add_91, [2048, 1024])
        permute_138 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_82 = torch.ops.aten.addmm.default(arg228_1, view_227, permute_138);  arg228_1 = view_227 = permute_138 = None
        view_228 = torch.ops.aten.view.default(addmm_82, [16, 128, 1024]);  addmm_82 = None
        view_229 = torch.ops.aten.view.default(add_91, [2048, 1024])
        permute_139 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_83 = torch.ops.aten.addmm.default(arg230_1, view_229, permute_139);  arg230_1 = view_229 = permute_139 = None
        view_230 = torch.ops.aten.view.default(addmm_83, [16, 128, 1024]);  addmm_83 = None
        view_231 = torch.ops.aten.view.default(view_230, [16, -1, 16, 64]);  view_230 = None
        permute_140 = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        clone_84 = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_232 = torch.ops.aten.view.default(add_91, [2048, 1024]);  add_91 = None
        permute_141 = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_84 = torch.ops.aten.addmm.default(arg232_1, view_232, permute_141);  arg232_1 = view_232 = permute_141 = None
        view_233 = torch.ops.aten.view.default(addmm_84, [16, 128, 1024]);  addmm_84 = None
        view_234 = torch.ops.aten.view.default(view_233, [16, -1, 16, 64]);  view_233 = None
        permute_142 = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_85 = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_235 = torch.ops.aten.view.default(view_228, [16, 128, 16, 64]);  view_228 = None
        permute_143 = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
        clone_86 = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_5, [16, 1, 128, 128]);  unsqueeze_5 = None
        expand_6 = torch.ops.aten.expand.default(expand_5, [16, 16, 128, 128]);  expand_5 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_86, clone_84, clone_85, expand_6, False);  clone_86 = expand_6 = None
        getitem_114 = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        permute_144 = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3]);  getitem_114 = None
        view_236 = torch.ops.aten.view.default(permute_144, [16, 128, 1024]);  permute_144 = None
        view_237 = torch.ops.aten.view.default(view_236, [2048, 1024]);  view_236 = None
        permute_145 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_85 = torch.ops.aten.addmm.default(arg234_1, view_237, permute_145);  arg234_1 = view_237 = permute_145 = None
        view_238 = torch.ops.aten.view.default(addmm_85, [16, 128, 1024]);  addmm_85 = None
        add_92 = torch.ops.aten.add.Tensor(add_89, view_238);  add_89 = view_238 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_118 = var_mean_29[0]
        getitem_119 = var_mean_29[1];  var_mean_29 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_92, getitem_119);  getitem_119 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_62, arg235_1);  mul_62 = arg235_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_63, arg236_1);  mul_63 = arg236_1 = None
        view_239 = torch.ops.aten.view.default(add_94, [2048, 1024]);  add_94 = None
        permute_146 = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_86 = torch.ops.aten.addmm.default(arg238_1, view_239, permute_146);  arg238_1 = view_239 = permute_146 = None
        view_240 = torch.ops.aten.view.default(addmm_86, [16, 128, 1024]);  addmm_86 = None
        view_241 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_147 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_87 = torch.ops.aten.addmm.default(arg240_1, view_241, permute_147);  arg240_1 = view_241 = permute_147 = None
        view_242 = torch.ops.aten.view.default(addmm_87, [16, 128, 1024]);  addmm_87 = None
        view_243 = torch.ops.aten.view.default(view_242, [16, -1, 16, 64]);  view_242 = None
        permute_148 = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        clone_88 = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_244 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_149 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_88 = torch.ops.aten.addmm.default(arg242_1, view_244, permute_149);  arg242_1 = view_244 = permute_149 = None
        view_245 = torch.ops.aten.view.default(addmm_88, [16, 128, 1024]);  addmm_88 = None
        view_246 = torch.ops.aten.view.default(view_245, [16, -1, 16, 64]);  view_245 = None
        permute_150 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_89 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_247 = torch.ops.aten.view.default(view_240, [16, 128, 16, 64]);  view_240 = None
        permute_151 = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        clone_90 = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_90, clone_88, clone_89, None, False);  clone_90 = None
        getitem_120 = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        permute_152 = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        view_248 = torch.ops.aten.view.default(permute_152, [16, 128, 1024]);  permute_152 = None
        view_249 = torch.ops.aten.view.default(view_248, [2048, 1024]);  view_248 = None
        permute_153 = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_89 = torch.ops.aten.addmm.default(arg244_1, view_249, permute_153);  arg244_1 = view_249 = permute_153 = None
        view_250 = torch.ops.aten.view.default(addmm_89, [16, 128, 1024]);  addmm_89 = None
        add_95 = torch.ops.aten.add.Tensor(add_92, view_250);  add_92 = view_250 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_124 = var_mean_30[0]
        getitem_125 = var_mean_30[1];  var_mean_30 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_95, getitem_125);  getitem_125 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg245_1);  mul_64 = arg245_1 = None
        add_97 = torch.ops.aten.add.Tensor(mul_65, arg246_1);  mul_65 = arg246_1 = None
        view_251 = torch.ops.aten.view.default(add_97, [2048, 1024]);  add_97 = None
        permute_154 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_90 = torch.ops.aten.addmm.default(arg248_1, view_251, permute_154);  arg248_1 = view_251 = permute_154 = None
        view_252 = torch.ops.aten.view.default(addmm_90, [16, 128, 4096]);  addmm_90 = None
        relu_13 = torch.ops.aten.relu.default(view_252);  view_252 = None
        view_253 = torch.ops.aten.view.default(relu_13, [2048, 4096]);  relu_13 = None
        permute_155 = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_91 = torch.ops.aten.addmm.default(arg250_1, view_253, permute_155);  arg250_1 = view_253 = permute_155 = None
        view_254 = torch.ops.aten.view.default(addmm_91, [16, 128, 1024]);  addmm_91 = None
        add_98 = torch.ops.aten.add.Tensor(add_95, view_254);  add_95 = view_254 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_126 = var_mean_31[0]
        getitem_127 = var_mean_31[1];  var_mean_31 = None
        add_99 = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_98, getitem_127);  getitem_127 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg251_1);  mul_66 = arg251_1 = None
        add_100 = torch.ops.aten.add.Tensor(mul_67, arg252_1);  mul_67 = arg252_1 = None
        view_255 = torch.ops.aten.view.default(add_100, [2048, 1024])
        permute_156 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_92 = torch.ops.aten.addmm.default(arg254_1, view_255, permute_156);  arg254_1 = view_255 = permute_156 = None
        view_256 = torch.ops.aten.view.default(addmm_92, [16, 128, 1024]);  addmm_92 = None
        view_257 = torch.ops.aten.view.default(add_100, [2048, 1024])
        permute_157 = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_93 = torch.ops.aten.addmm.default(arg256_1, view_257, permute_157);  arg256_1 = view_257 = permute_157 = None
        view_258 = torch.ops.aten.view.default(addmm_93, [16, 128, 1024]);  addmm_93 = None
        view_259 = torch.ops.aten.view.default(view_258, [16, -1, 16, 64]);  view_258 = None
        permute_158 = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        clone_94 = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_260 = torch.ops.aten.view.default(add_100, [2048, 1024]);  add_100 = None
        permute_159 = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_94 = torch.ops.aten.addmm.default(arg258_1, view_260, permute_159);  arg258_1 = view_260 = permute_159 = None
        view_261 = torch.ops.aten.view.default(addmm_94, [16, 128, 1024]);  addmm_94 = None
        view_262 = torch.ops.aten.view.default(view_261, [16, -1, 16, 64]);  view_261 = None
        permute_160 = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        clone_95 = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_263 = torch.ops.aten.view.default(view_256, [16, 128, 16, 64]);  view_256 = None
        permute_161 = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
        clone_96 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_7, [16, 1, 128, 128]);  unsqueeze_7 = None
        expand_9 = torch.ops.aten.expand.default(expand_8, [16, 16, 128, 128]);  expand_8 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_96, clone_94, clone_95, expand_9, False);  clone_96 = expand_9 = None
        getitem_128 = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        permute_162 = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None
        view_264 = torch.ops.aten.view.default(permute_162, [16, 128, 1024]);  permute_162 = None
        view_265 = torch.ops.aten.view.default(view_264, [2048, 1024]);  view_264 = None
        permute_163 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_95 = torch.ops.aten.addmm.default(arg260_1, view_265, permute_163);  arg260_1 = view_265 = permute_163 = None
        view_266 = torch.ops.aten.view.default(addmm_95, [16, 128, 1024]);  addmm_95 = None
        add_101 = torch.ops.aten.add.Tensor(add_98, view_266);  add_98 = view_266 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_132 = var_mean_32[0]
        getitem_133 = var_mean_32[1];  var_mean_32 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_101, getitem_133);  getitem_133 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg261_1);  mul_68 = arg261_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_69, arg262_1);  mul_69 = arg262_1 = None
        view_267 = torch.ops.aten.view.default(add_103, [2048, 1024]);  add_103 = None
        permute_164 = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_96 = torch.ops.aten.addmm.default(arg264_1, view_267, permute_164);  arg264_1 = view_267 = permute_164 = None
        view_268 = torch.ops.aten.view.default(addmm_96, [16, 128, 1024]);  addmm_96 = None
        view_269 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_165 = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_97 = torch.ops.aten.addmm.default(arg266_1, view_269, permute_165);  arg266_1 = view_269 = permute_165 = None
        view_270 = torch.ops.aten.view.default(addmm_97, [16, 128, 1024]);  addmm_97 = None
        view_271 = torch.ops.aten.view.default(view_270, [16, -1, 16, 64]);  view_270 = None
        permute_166 = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        clone_98 = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_272 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_167 = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_98 = torch.ops.aten.addmm.default(arg268_1, view_272, permute_167);  arg268_1 = view_272 = permute_167 = None
        view_273 = torch.ops.aten.view.default(addmm_98, [16, 128, 1024]);  addmm_98 = None
        view_274 = torch.ops.aten.view.default(view_273, [16, -1, 16, 64]);  view_273 = None
        permute_168 = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_99 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_275 = torch.ops.aten.view.default(view_268, [16, 128, 16, 64]);  view_268 = None
        permute_169 = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
        clone_100 = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_100, clone_98, clone_99, None, False);  clone_100 = None
        getitem_134 = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        permute_170 = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_276 = torch.ops.aten.view.default(permute_170, [16, 128, 1024]);  permute_170 = None
        view_277 = torch.ops.aten.view.default(view_276, [2048, 1024]);  view_276 = None
        permute_171 = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_99 = torch.ops.aten.addmm.default(arg270_1, view_277, permute_171);  arg270_1 = view_277 = permute_171 = None
        view_278 = torch.ops.aten.view.default(addmm_99, [16, 128, 1024]);  addmm_99 = None
        add_104 = torch.ops.aten.add.Tensor(add_101, view_278);  add_101 = view_278 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
        getitem_138 = var_mean_33[0]
        getitem_139 = var_mean_33[1];  var_mean_33 = None
        add_105 = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_104, getitem_139);  getitem_139 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg271_1);  mul_70 = arg271_1 = None
        add_106 = torch.ops.aten.add.Tensor(mul_71, arg272_1);  mul_71 = arg272_1 = None
        view_279 = torch.ops.aten.view.default(add_106, [2048, 1024]);  add_106 = None
        permute_172 = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_100 = torch.ops.aten.addmm.default(arg274_1, view_279, permute_172);  arg274_1 = view_279 = permute_172 = None
        view_280 = torch.ops.aten.view.default(addmm_100, [16, 128, 4096]);  addmm_100 = None
        relu_14 = torch.ops.aten.relu.default(view_280);  view_280 = None
        view_281 = torch.ops.aten.view.default(relu_14, [2048, 4096]);  relu_14 = None
        permute_173 = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_101 = torch.ops.aten.addmm.default(arg276_1, view_281, permute_173);  arg276_1 = view_281 = permute_173 = None
        view_282 = torch.ops.aten.view.default(addmm_101, [16, 128, 1024]);  addmm_101 = None
        add_107 = torch.ops.aten.add.Tensor(add_104, view_282);  add_104 = view_282 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
        getitem_140 = var_mean_34[0]
        getitem_141 = var_mean_34[1];  var_mean_34 = None
        add_108 = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_107, getitem_141);  getitem_141 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg277_1);  mul_72 = arg277_1 = None
        add_109 = torch.ops.aten.add.Tensor(mul_73, arg278_1);  mul_73 = arg278_1 = None
        view_283 = torch.ops.aten.view.default(add_109, [2048, 1024])
        permute_174 = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_102 = torch.ops.aten.addmm.default(arg280_1, view_283, permute_174);  arg280_1 = view_283 = permute_174 = None
        view_284 = torch.ops.aten.view.default(addmm_102, [16, 128, 1024]);  addmm_102 = None
        view_285 = torch.ops.aten.view.default(add_109, [2048, 1024])
        permute_175 = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_103 = torch.ops.aten.addmm.default(arg282_1, view_285, permute_175);  arg282_1 = view_285 = permute_175 = None
        view_286 = torch.ops.aten.view.default(addmm_103, [16, 128, 1024]);  addmm_103 = None
        view_287 = torch.ops.aten.view.default(view_286, [16, -1, 16, 64]);  view_286 = None
        permute_176 = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        clone_104 = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_288 = torch.ops.aten.view.default(add_109, [2048, 1024]);  add_109 = None
        permute_177 = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_104 = torch.ops.aten.addmm.default(arg284_1, view_288, permute_177);  arg284_1 = view_288 = permute_177 = None
        view_289 = torch.ops.aten.view.default(addmm_104, [16, 128, 1024]);  addmm_104 = None
        view_290 = torch.ops.aten.view.default(view_289, [16, -1, 16, 64]);  view_289 = None
        permute_178 = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_105 = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_291 = torch.ops.aten.view.default(view_284, [16, 128, 16, 64]);  view_284 = None
        permute_179 = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_106 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_9, [16, 1, 128, 128]);  unsqueeze_9 = None
        expand_12 = torch.ops.aten.expand.default(expand_11, [16, 16, 128, 128]);  expand_11 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_106, clone_104, clone_105, expand_12, False);  clone_106 = expand_12 = None
        getitem_142 = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        permute_180 = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        view_292 = torch.ops.aten.view.default(permute_180, [16, 128, 1024]);  permute_180 = None
        view_293 = torch.ops.aten.view.default(view_292, [2048, 1024]);  view_292 = None
        permute_181 = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_105 = torch.ops.aten.addmm.default(arg286_1, view_293, permute_181);  arg286_1 = view_293 = permute_181 = None
        view_294 = torch.ops.aten.view.default(addmm_105, [16, 128, 1024]);  addmm_105 = None
        add_110 = torch.ops.aten.add.Tensor(add_107, view_294);  add_107 = view_294 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_146 = var_mean_35[0]
        getitem_147 = var_mean_35[1];  var_mean_35 = None
        add_111 = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_110, getitem_147);  getitem_147 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg287_1);  mul_74 = arg287_1 = None
        add_112 = torch.ops.aten.add.Tensor(mul_75, arg288_1);  mul_75 = arg288_1 = None
        view_295 = torch.ops.aten.view.default(add_112, [2048, 1024]);  add_112 = None
        permute_182 = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_106 = torch.ops.aten.addmm.default(arg290_1, view_295, permute_182);  arg290_1 = view_295 = permute_182 = None
        view_296 = torch.ops.aten.view.default(addmm_106, [16, 128, 1024]);  addmm_106 = None
        view_297 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_183 = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_107 = torch.ops.aten.addmm.default(arg292_1, view_297, permute_183);  arg292_1 = view_297 = permute_183 = None
        view_298 = torch.ops.aten.view.default(addmm_107, [16, 128, 1024]);  addmm_107 = None
        view_299 = torch.ops.aten.view.default(view_298, [16, -1, 16, 64]);  view_298 = None
        permute_184 = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
        clone_108 = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_300 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_185 = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108 = torch.ops.aten.addmm.default(arg294_1, view_300, permute_185);  arg294_1 = view_300 = permute_185 = None
        view_301 = torch.ops.aten.view.default(addmm_108, [16, 128, 1024]);  addmm_108 = None
        view_302 = torch.ops.aten.view.default(view_301, [16, -1, 16, 64]);  view_301 = None
        permute_186 = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_109 = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_303 = torch.ops.aten.view.default(view_296, [16, 128, 16, 64]);  view_296 = None
        permute_187 = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        clone_110 = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_110, clone_108, clone_109, None, False);  clone_110 = None
        getitem_148 = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        permute_188 = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        view_304 = torch.ops.aten.view.default(permute_188, [16, 128, 1024]);  permute_188 = None
        view_305 = torch.ops.aten.view.default(view_304, [2048, 1024]);  view_304 = None
        permute_189 = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109 = torch.ops.aten.addmm.default(arg296_1, view_305, permute_189);  arg296_1 = view_305 = permute_189 = None
        view_306 = torch.ops.aten.view.default(addmm_109, [16, 128, 1024]);  addmm_109 = None
        add_113 = torch.ops.aten.add.Tensor(add_110, view_306);  add_110 = view_306 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_152 = var_mean_36[0]
        getitem_153 = var_mean_36[1];  var_mean_36 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_113, getitem_153);  getitem_153 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, arg297_1);  mul_76 = arg297_1 = None
        add_115 = torch.ops.aten.add.Tensor(mul_77, arg298_1);  mul_77 = arg298_1 = None
        view_307 = torch.ops.aten.view.default(add_115, [2048, 1024]);  add_115 = None
        permute_190 = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_110 = torch.ops.aten.addmm.default(arg300_1, view_307, permute_190);  arg300_1 = view_307 = permute_190 = None
        view_308 = torch.ops.aten.view.default(addmm_110, [16, 128, 4096]);  addmm_110 = None
        relu_15 = torch.ops.aten.relu.default(view_308);  view_308 = None
        view_309 = torch.ops.aten.view.default(relu_15, [2048, 4096]);  relu_15 = None
        permute_191 = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_111 = torch.ops.aten.addmm.default(arg302_1, view_309, permute_191);  arg302_1 = view_309 = permute_191 = None
        view_310 = torch.ops.aten.view.default(addmm_111, [16, 128, 1024]);  addmm_111 = None
        add_116 = torch.ops.aten.add.Tensor(add_113, view_310);  add_113 = view_310 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
        getitem_154 = var_mean_37[0]
        getitem_155 = var_mean_37[1];  var_mean_37 = None
        add_117 = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_116, getitem_155);  getitem_155 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, arg303_1);  mul_78 = arg303_1 = None
        add_118 = torch.ops.aten.add.Tensor(mul_79, arg304_1);  mul_79 = arg304_1 = None
        view_311 = torch.ops.aten.view.default(add_118, [2048, 1024])
        permute_192 = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_112 = torch.ops.aten.addmm.default(arg306_1, view_311, permute_192);  arg306_1 = view_311 = permute_192 = None
        view_312 = torch.ops.aten.view.default(addmm_112, [16, 128, 1024]);  addmm_112 = None
        view_313 = torch.ops.aten.view.default(add_118, [2048, 1024])
        permute_193 = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_113 = torch.ops.aten.addmm.default(arg308_1, view_313, permute_193);  arg308_1 = view_313 = permute_193 = None
        view_314 = torch.ops.aten.view.default(addmm_113, [16, 128, 1024]);  addmm_113 = None
        view_315 = torch.ops.aten.view.default(view_314, [16, -1, 16, 64]);  view_314 = None
        permute_194 = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        clone_114 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_316 = torch.ops.aten.view.default(add_118, [2048, 1024]);  add_118 = None
        permute_195 = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114 = torch.ops.aten.addmm.default(arg310_1, view_316, permute_195);  arg310_1 = view_316 = permute_195 = None
        view_317 = torch.ops.aten.view.default(addmm_114, [16, 128, 1024]);  addmm_114 = None
        view_318 = torch.ops.aten.view.default(view_317, [16, -1, 16, 64]);  view_317 = None
        permute_196 = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_115 = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_319 = torch.ops.aten.view.default(view_312, [16, 128, 16, 64]);  view_312 = None
        permute_197 = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_116 = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_11, [16, 1, 128, 128]);  unsqueeze_11 = None
        expand_15 = torch.ops.aten.expand.default(expand_14, [16, 16, 128, 128]);  expand_14 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_116, clone_114, clone_115, expand_15, False);  clone_116 = expand_15 = None
        getitem_156 = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        permute_198 = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        view_320 = torch.ops.aten.view.default(permute_198, [16, 128, 1024]);  permute_198 = None
        view_321 = torch.ops.aten.view.default(view_320, [2048, 1024]);  view_320 = None
        permute_199 = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115 = torch.ops.aten.addmm.default(arg312_1, view_321, permute_199);  arg312_1 = view_321 = permute_199 = None
        view_322 = torch.ops.aten.view.default(addmm_115, [16, 128, 1024]);  addmm_115 = None
        add_119 = torch.ops.aten.add.Tensor(add_116, view_322);  add_116 = view_322 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_160 = var_mean_38[0]
        getitem_161 = var_mean_38[1];  var_mean_38 = None
        add_120 = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_119, getitem_161);  getitem_161 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg313_1);  mul_80 = arg313_1 = None
        add_121 = torch.ops.aten.add.Tensor(mul_81, arg314_1);  mul_81 = arg314_1 = None
        view_323 = torch.ops.aten.view.default(add_121, [2048, 1024]);  add_121 = None
        permute_200 = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_116 = torch.ops.aten.addmm.default(arg316_1, view_323, permute_200);  arg316_1 = view_323 = permute_200 = None
        view_324 = torch.ops.aten.view.default(addmm_116, [16, 128, 1024]);  addmm_116 = None
        view_325 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_201 = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_117 = torch.ops.aten.addmm.default(arg318_1, view_325, permute_201);  arg318_1 = view_325 = permute_201 = None
        view_326 = torch.ops.aten.view.default(addmm_117, [16, 128, 1024]);  addmm_117 = None
        view_327 = torch.ops.aten.view.default(view_326, [16, -1, 16, 64]);  view_326 = None
        permute_202 = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        clone_118 = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_328 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_203 = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118 = torch.ops.aten.addmm.default(arg320_1, view_328, permute_203);  arg320_1 = view_328 = permute_203 = None
        view_329 = torch.ops.aten.view.default(addmm_118, [16, 128, 1024]);  addmm_118 = None
        view_330 = torch.ops.aten.view.default(view_329, [16, -1, 16, 64]);  view_329 = None
        permute_204 = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        clone_119 = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_331 = torch.ops.aten.view.default(view_324, [16, 128, 16, 64]);  view_324 = None
        permute_205 = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
        clone_120 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_120, clone_118, clone_119, None, False);  clone_120 = None
        getitem_162 = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        permute_206 = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        view_332 = torch.ops.aten.view.default(permute_206, [16, 128, 1024]);  permute_206 = None
        view_333 = torch.ops.aten.view.default(view_332, [2048, 1024]);  view_332 = None
        permute_207 = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119 = torch.ops.aten.addmm.default(arg322_1, view_333, permute_207);  arg322_1 = view_333 = permute_207 = None
        view_334 = torch.ops.aten.view.default(addmm_119, [16, 128, 1024]);  addmm_119 = None
        add_122 = torch.ops.aten.add.Tensor(add_119, view_334);  add_119 = view_334 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_166 = var_mean_39[0]
        getitem_167 = var_mean_39[1];  var_mean_39 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_122, getitem_167);  getitem_167 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg323_1);  mul_82 = arg323_1 = None
        add_124 = torch.ops.aten.add.Tensor(mul_83, arg324_1);  mul_83 = arg324_1 = None
        view_335 = torch.ops.aten.view.default(add_124, [2048, 1024]);  add_124 = None
        permute_208 = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120 = torch.ops.aten.addmm.default(arg326_1, view_335, permute_208);  arg326_1 = view_335 = permute_208 = None
        view_336 = torch.ops.aten.view.default(addmm_120, [16, 128, 4096]);  addmm_120 = None
        relu_16 = torch.ops.aten.relu.default(view_336);  view_336 = None
        view_337 = torch.ops.aten.view.default(relu_16, [2048, 4096]);  relu_16 = None
        permute_209 = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121 = torch.ops.aten.addmm.default(arg328_1, view_337, permute_209);  arg328_1 = view_337 = permute_209 = None
        view_338 = torch.ops.aten.view.default(addmm_121, [16, 128, 1024]);  addmm_121 = None
        add_125 = torch.ops.aten.add.Tensor(add_122, view_338);  add_122 = view_338 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_168 = var_mean_40[0]
        getitem_169 = var_mean_40[1];  var_mean_40 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_125, getitem_169);  getitem_169 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg329_1);  mul_84 = arg329_1 = None
        add_127 = torch.ops.aten.add.Tensor(mul_85, arg330_1);  mul_85 = arg330_1 = None
        view_339 = torch.ops.aten.view.default(add_127, [2048, 1024])
        permute_210 = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_122 = torch.ops.aten.addmm.default(arg332_1, view_339, permute_210);  arg332_1 = view_339 = permute_210 = None
        view_340 = torch.ops.aten.view.default(addmm_122, [16, 128, 1024]);  addmm_122 = None
        view_341 = torch.ops.aten.view.default(add_127, [2048, 1024])
        permute_211 = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_123 = torch.ops.aten.addmm.default(arg334_1, view_341, permute_211);  arg334_1 = view_341 = permute_211 = None
        view_342 = torch.ops.aten.view.default(addmm_123, [16, 128, 1024]);  addmm_123 = None
        view_343 = torch.ops.aten.view.default(view_342, [16, -1, 16, 64]);  view_342 = None
        permute_212 = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
        clone_124 = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_344 = torch.ops.aten.view.default(add_127, [2048, 1024]);  add_127 = None
        permute_213 = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124 = torch.ops.aten.addmm.default(arg336_1, view_344, permute_213);  arg336_1 = view_344 = permute_213 = None
        view_345 = torch.ops.aten.view.default(addmm_124, [16, 128, 1024]);  addmm_124 = None
        view_346 = torch.ops.aten.view.default(view_345, [16, -1, 16, 64]);  view_345 = None
        permute_214 = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_125 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_347 = torch.ops.aten.view.default(view_340, [16, 128, 16, 64]);  view_340 = None
        permute_215 = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        clone_126 = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        expand_17 = torch.ops.aten.expand.default(unsqueeze_13, [16, 1, 128, 128]);  unsqueeze_13 = None
        expand_18 = torch.ops.aten.expand.default(expand_17, [16, 16, 128, 128]);  expand_17 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_126, clone_124, clone_125, expand_18, False);  clone_126 = expand_18 = None
        getitem_170 = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        permute_216 = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3]);  getitem_170 = None
        view_348 = torch.ops.aten.view.default(permute_216, [16, 128, 1024]);  permute_216 = None
        view_349 = torch.ops.aten.view.default(view_348, [2048, 1024]);  view_348 = None
        permute_217 = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125 = torch.ops.aten.addmm.default(arg338_1, view_349, permute_217);  arg338_1 = view_349 = permute_217 = None
        view_350 = torch.ops.aten.view.default(addmm_125, [16, 128, 1024]);  addmm_125 = None
        add_128 = torch.ops.aten.add.Tensor(add_125, view_350);  add_125 = view_350 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_174 = var_mean_41[0]
        getitem_175 = var_mean_41[1];  var_mean_41 = None
        add_129 = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_128, getitem_175);  getitem_175 = None
        mul_86 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_86, arg339_1);  mul_86 = arg339_1 = None
        add_130 = torch.ops.aten.add.Tensor(mul_87, arg340_1);  mul_87 = arg340_1 = None
        view_351 = torch.ops.aten.view.default(add_130, [2048, 1024]);  add_130 = None
        permute_218 = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126 = torch.ops.aten.addmm.default(arg342_1, view_351, permute_218);  arg342_1 = view_351 = permute_218 = None
        view_352 = torch.ops.aten.view.default(addmm_126, [16, 128, 1024]);  addmm_126 = None
        view_353 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_219 = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127 = torch.ops.aten.addmm.default(arg344_1, view_353, permute_219);  arg344_1 = view_353 = permute_219 = None
        view_354 = torch.ops.aten.view.default(addmm_127, [16, 128, 1024]);  addmm_127 = None
        view_355 = torch.ops.aten.view.default(view_354, [16, -1, 16, 64]);  view_354 = None
        permute_220 = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None
        clone_128 = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_356 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_221 = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128 = torch.ops.aten.addmm.default(arg346_1, view_356, permute_221);  arg346_1 = view_356 = permute_221 = None
        view_357 = torch.ops.aten.view.default(addmm_128, [16, 128, 1024]);  addmm_128 = None
        view_358 = torch.ops.aten.view.default(view_357, [16, -1, 16, 64]);  view_357 = None
        permute_222 = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_129 = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_359 = torch.ops.aten.view.default(view_352, [16, 128, 16, 64]);  view_352 = None
        permute_223 = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
        clone_130 = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_130, clone_128, clone_129, None, False);  clone_130 = None
        getitem_176 = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        permute_224 = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        view_360 = torch.ops.aten.view.default(permute_224, [16, 128, 1024]);  permute_224 = None
        view_361 = torch.ops.aten.view.default(view_360, [2048, 1024]);  view_360 = None
        permute_225 = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129 = torch.ops.aten.addmm.default(arg348_1, view_361, permute_225);  arg348_1 = view_361 = permute_225 = None
        view_362 = torch.ops.aten.view.default(addmm_129, [16, 128, 1024]);  addmm_129 = None
        add_131 = torch.ops.aten.add.Tensor(add_128, view_362);  add_128 = view_362 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
        getitem_180 = var_mean_42[0]
        getitem_181 = var_mean_42[1];  var_mean_42 = None
        add_132 = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_131, getitem_181);  getitem_181 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg349_1);  mul_88 = arg349_1 = None
        add_133 = torch.ops.aten.add.Tensor(mul_89, arg350_1);  mul_89 = arg350_1 = None
        view_363 = torch.ops.aten.view.default(add_133, [2048, 1024]);  add_133 = None
        permute_226 = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130 = torch.ops.aten.addmm.default(arg352_1, view_363, permute_226);  arg352_1 = view_363 = permute_226 = None
        view_364 = torch.ops.aten.view.default(addmm_130, [16, 128, 4096]);  addmm_130 = None
        relu_17 = torch.ops.aten.relu.default(view_364);  view_364 = None
        view_365 = torch.ops.aten.view.default(relu_17, [2048, 4096]);  relu_17 = None
        permute_227 = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131 = torch.ops.aten.addmm.default(arg354_1, view_365, permute_227);  arg354_1 = view_365 = permute_227 = None
        view_366 = torch.ops.aten.view.default(addmm_131, [16, 128, 1024]);  addmm_131 = None
        add_134 = torch.ops.aten.add.Tensor(add_131, view_366);  add_131 = view_366 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
        getitem_182 = var_mean_43[0]
        getitem_183 = var_mean_43[1];  var_mean_43 = None
        add_135 = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_134, getitem_183);  getitem_183 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg355_1);  mul_90 = arg355_1 = None
        add_136 = torch.ops.aten.add.Tensor(mul_91, arg356_1);  mul_91 = arg356_1 = None
        view_367 = torch.ops.aten.view.default(add_136, [2048, 1024])
        permute_228 = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132 = torch.ops.aten.addmm.default(arg358_1, view_367, permute_228);  arg358_1 = view_367 = permute_228 = None
        view_368 = torch.ops.aten.view.default(addmm_132, [16, 128, 1024]);  addmm_132 = None
        view_369 = torch.ops.aten.view.default(add_136, [2048, 1024])
        permute_229 = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133 = torch.ops.aten.addmm.default(arg360_1, view_369, permute_229);  arg360_1 = view_369 = permute_229 = None
        view_370 = torch.ops.aten.view.default(addmm_133, [16, 128, 1024]);  addmm_133 = None
        view_371 = torch.ops.aten.view.default(view_370, [16, -1, 16, 64]);  view_370 = None
        permute_230 = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        clone_134 = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_372 = torch.ops.aten.view.default(add_136, [2048, 1024]);  add_136 = None
        permute_231 = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134 = torch.ops.aten.addmm.default(arg362_1, view_372, permute_231);  arg362_1 = view_372 = permute_231 = None
        view_373 = torch.ops.aten.view.default(addmm_134, [16, 128, 1024]);  addmm_134 = None
        view_374 = torch.ops.aten.view.default(view_373, [16, -1, 16, 64]);  view_373 = None
        permute_232 = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_135 = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_375 = torch.ops.aten.view.default(view_368, [16, 128, 16, 64]);  view_368 = None
        permute_233 = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        clone_136 = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
        expand_20 = torch.ops.aten.expand.default(unsqueeze_15, [16, 1, 128, 128]);  unsqueeze_15 = None
        expand_21 = torch.ops.aten.expand.default(expand_20, [16, 16, 128, 128]);  expand_20 = None
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_136, clone_134, clone_135, expand_21, False);  clone_136 = expand_21 = None
        getitem_184 = _scaled_dot_product_efficient_attention_24[0];  _scaled_dot_product_efficient_attention_24 = None
        permute_234 = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3]);  getitem_184 = None
        view_376 = torch.ops.aten.view.default(permute_234, [16, 128, 1024]);  permute_234 = None
        view_377 = torch.ops.aten.view.default(view_376, [2048, 1024]);  view_376 = None
        permute_235 = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135 = torch.ops.aten.addmm.default(arg364_1, view_377, permute_235);  arg364_1 = view_377 = permute_235 = None
        view_378 = torch.ops.aten.view.default(addmm_135, [16, 128, 1024]);  addmm_135 = None
        add_137 = torch.ops.aten.add.Tensor(add_134, view_378);  add_134 = view_378 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_188 = var_mean_44[0]
        getitem_189 = var_mean_44[1];  var_mean_44 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_137, getitem_189);  getitem_189 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, arg365_1);  mul_92 = arg365_1 = None
        add_139 = torch.ops.aten.add.Tensor(mul_93, arg366_1);  mul_93 = arg366_1 = None
        view_379 = torch.ops.aten.view.default(add_139, [2048, 1024]);  add_139 = None
        permute_236 = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136 = torch.ops.aten.addmm.default(arg368_1, view_379, permute_236);  arg368_1 = view_379 = permute_236 = None
        view_380 = torch.ops.aten.view.default(addmm_136, [16, 128, 1024]);  addmm_136 = None
        view_381 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_237 = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137 = torch.ops.aten.addmm.default(arg370_1, view_381, permute_237);  arg370_1 = view_381 = permute_237 = None
        view_382 = torch.ops.aten.view.default(addmm_137, [16, 128, 1024]);  addmm_137 = None
        view_383 = torch.ops.aten.view.default(view_382, [16, -1, 16, 64]);  view_382 = None
        permute_238 = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_138 = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_384 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_239 = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
        addmm_138 = torch.ops.aten.addmm.default(arg372_1, view_384, permute_239);  arg372_1 = view_384 = permute_239 = None
        view_385 = torch.ops.aten.view.default(addmm_138, [16, 128, 1024]);  addmm_138 = None
        view_386 = torch.ops.aten.view.default(view_385, [16, -1, 16, 64]);  view_385 = None
        permute_240 = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        clone_139 = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_387 = torch.ops.aten.view.default(view_380, [16, 128, 16, 64]);  view_380 = None
        permute_241 = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        clone_140 = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_140, clone_138, clone_139, None, False);  clone_140 = None
        getitem_190 = _scaled_dot_product_efficient_attention_25[0];  _scaled_dot_product_efficient_attention_25 = None
        permute_242 = torch.ops.aten.permute.default(getitem_190, [0, 2, 1, 3]);  getitem_190 = None
        view_388 = torch.ops.aten.view.default(permute_242, [16, 128, 1024]);  permute_242 = None
        view_389 = torch.ops.aten.view.default(view_388, [2048, 1024]);  view_388 = None
        permute_243 = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_139 = torch.ops.aten.addmm.default(arg374_1, view_389, permute_243);  arg374_1 = view_389 = permute_243 = None
        view_390 = torch.ops.aten.view.default(addmm_139, [16, 128, 1024]);  addmm_139 = None
        add_140 = torch.ops.aten.add.Tensor(add_137, view_390);  add_137 = view_390 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_194 = var_mean_45[0]
        getitem_195 = var_mean_45[1];  var_mean_45 = None
        add_141 = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_140, getitem_195);  getitem_195 = None
        mul_94 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, arg375_1);  mul_94 = arg375_1 = None
        add_142 = torch.ops.aten.add.Tensor(mul_95, arg376_1);  mul_95 = arg376_1 = None
        view_391 = torch.ops.aten.view.default(add_142, [2048, 1024]);  add_142 = None
        permute_244 = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140 = torch.ops.aten.addmm.default(arg378_1, view_391, permute_244);  arg378_1 = view_391 = permute_244 = None
        view_392 = torch.ops.aten.view.default(addmm_140, [16, 128, 4096]);  addmm_140 = None
        relu_18 = torch.ops.aten.relu.default(view_392);  view_392 = None
        view_393 = torch.ops.aten.view.default(relu_18, [2048, 4096]);  relu_18 = None
        permute_245 = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141 = torch.ops.aten.addmm.default(arg380_1, view_393, permute_245);  arg380_1 = view_393 = permute_245 = None
        view_394 = torch.ops.aten.view.default(addmm_141, [16, 128, 1024]);  addmm_141 = None
        add_143 = torch.ops.aten.add.Tensor(add_140, view_394);  add_140 = view_394 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_196 = var_mean_46[0]
        getitem_197 = var_mean_46[1];  var_mean_46 = None
        add_144 = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_143, getitem_197);  getitem_197 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg381_1);  mul_96 = arg381_1 = None
        add_145 = torch.ops.aten.add.Tensor(mul_97, arg382_1);  mul_97 = arg382_1 = None
        view_395 = torch.ops.aten.view.default(add_145, [2048, 1024])
        permute_246 = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142 = torch.ops.aten.addmm.default(arg384_1, view_395, permute_246);  arg384_1 = view_395 = permute_246 = None
        view_396 = torch.ops.aten.view.default(addmm_142, [16, 128, 1024]);  addmm_142 = None
        view_397 = torch.ops.aten.view.default(add_145, [2048, 1024])
        permute_247 = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143 = torch.ops.aten.addmm.default(arg386_1, view_397, permute_247);  arg386_1 = view_397 = permute_247 = None
        view_398 = torch.ops.aten.view.default(addmm_143, [16, 128, 1024]);  addmm_143 = None
        view_399 = torch.ops.aten.view.default(view_398, [16, -1, 16, 64]);  view_398 = None
        permute_248 = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_144 = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_400 = torch.ops.aten.view.default(add_145, [2048, 1024]);  add_145 = None
        permute_249 = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_144 = torch.ops.aten.addmm.default(arg388_1, view_400, permute_249);  arg388_1 = view_400 = permute_249 = None
        view_401 = torch.ops.aten.view.default(addmm_144, [16, 128, 1024]);  addmm_144 = None
        view_402 = torch.ops.aten.view.default(view_401, [16, -1, 16, 64]);  view_401 = None
        permute_250 = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_145 = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_403 = torch.ops.aten.view.default(view_396, [16, 128, 16, 64]);  view_396 = None
        permute_251 = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        clone_146 = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        expand_23 = torch.ops.aten.expand.default(unsqueeze_17, [16, 1, 128, 128]);  unsqueeze_17 = None
        expand_24 = torch.ops.aten.expand.default(expand_23, [16, 16, 128, 128]);  expand_23 = None
        _scaled_dot_product_efficient_attention_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_146, clone_144, clone_145, expand_24, False);  clone_146 = expand_24 = None
        getitem_198 = _scaled_dot_product_efficient_attention_26[0];  _scaled_dot_product_efficient_attention_26 = None
        permute_252 = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None
        view_404 = torch.ops.aten.view.default(permute_252, [16, 128, 1024]);  permute_252 = None
        view_405 = torch.ops.aten.view.default(view_404, [2048, 1024]);  view_404 = None
        permute_253 = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_145 = torch.ops.aten.addmm.default(arg390_1, view_405, permute_253);  arg390_1 = view_405 = permute_253 = None
        view_406 = torch.ops.aten.view.default(addmm_145, [16, 128, 1024]);  addmm_145 = None
        add_146 = torch.ops.aten.add.Tensor(add_143, view_406);  add_143 = view_406 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
        getitem_202 = var_mean_47[0]
        getitem_203 = var_mean_47[1];  var_mean_47 = None
        add_147 = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_146, getitem_203);  getitem_203 = None
        mul_98 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, arg391_1);  mul_98 = arg391_1 = None
        add_148 = torch.ops.aten.add.Tensor(mul_99, arg392_1);  mul_99 = arg392_1 = None
        view_407 = torch.ops.aten.view.default(add_148, [2048, 1024]);  add_148 = None
        permute_254 = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_146 = torch.ops.aten.addmm.default(arg394_1, view_407, permute_254);  arg394_1 = view_407 = permute_254 = None
        view_408 = torch.ops.aten.view.default(addmm_146, [16, 128, 1024]);  addmm_146 = None
        view_409 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_255 = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_147 = torch.ops.aten.addmm.default(arg396_1, view_409, permute_255);  arg396_1 = view_409 = permute_255 = None
        view_410 = torch.ops.aten.view.default(addmm_147, [16, 128, 1024]);  addmm_147 = None
        view_411 = torch.ops.aten.view.default(view_410, [16, -1, 16, 64]);  view_410 = None
        permute_256 = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        clone_148 = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_412 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_257 = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
        addmm_148 = torch.ops.aten.addmm.default(arg398_1, view_412, permute_257);  arg398_1 = view_412 = permute_257 = None
        view_413 = torch.ops.aten.view.default(addmm_148, [16, 128, 1024]);  addmm_148 = None
        view_414 = torch.ops.aten.view.default(view_413, [16, -1, 16, 64]);  view_413 = None
        permute_258 = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        clone_149 = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_415 = torch.ops.aten.view.default(view_408, [16, 128, 16, 64]);  view_408 = None
        permute_259 = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
        clone_150 = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        _scaled_dot_product_efficient_attention_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_150, clone_148, clone_149, None, False);  clone_150 = None
        getitem_204 = _scaled_dot_product_efficient_attention_27[0];  _scaled_dot_product_efficient_attention_27 = None
        permute_260 = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
        view_416 = torch.ops.aten.view.default(permute_260, [16, 128, 1024]);  permute_260 = None
        view_417 = torch.ops.aten.view.default(view_416, [2048, 1024]);  view_416 = None
        permute_261 = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_149 = torch.ops.aten.addmm.default(arg400_1, view_417, permute_261);  arg400_1 = view_417 = permute_261 = None
        view_418 = torch.ops.aten.view.default(addmm_149, [16, 128, 1024]);  addmm_149 = None
        add_149 = torch.ops.aten.add.Tensor(add_146, view_418);  add_146 = view_418 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_208 = var_mean_48[0]
        getitem_209 = var_mean_48[1];  var_mean_48 = None
        add_150 = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_149, getitem_209);  getitem_209 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, arg401_1);  mul_100 = arg401_1 = None
        add_151 = torch.ops.aten.add.Tensor(mul_101, arg402_1);  mul_101 = arg402_1 = None
        view_419 = torch.ops.aten.view.default(add_151, [2048, 1024]);  add_151 = None
        permute_262 = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_150 = torch.ops.aten.addmm.default(arg404_1, view_419, permute_262);  arg404_1 = view_419 = permute_262 = None
        view_420 = torch.ops.aten.view.default(addmm_150, [16, 128, 4096]);  addmm_150 = None
        relu_19 = torch.ops.aten.relu.default(view_420);  view_420 = None
        view_421 = torch.ops.aten.view.default(relu_19, [2048, 4096]);  relu_19 = None
        permute_263 = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_151 = torch.ops.aten.addmm.default(arg406_1, view_421, permute_263);  arg406_1 = view_421 = permute_263 = None
        view_422 = torch.ops.aten.view.default(addmm_151, [16, 128, 1024]);  addmm_151 = None
        add_152 = torch.ops.aten.add.Tensor(add_149, view_422);  add_149 = view_422 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
        getitem_210 = var_mean_49[0]
        getitem_211 = var_mean_49[1];  var_mean_49 = None
        add_153 = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_152, getitem_211);  getitem_211 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg407_1);  mul_102 = arg407_1 = None
        add_154 = torch.ops.aten.add.Tensor(mul_103, arg408_1);  mul_103 = arg408_1 = None
        view_423 = torch.ops.aten.view.default(add_154, [2048, 1024])
        permute_264 = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_152 = torch.ops.aten.addmm.default(arg410_1, view_423, permute_264);  arg410_1 = view_423 = permute_264 = None
        view_424 = torch.ops.aten.view.default(addmm_152, [16, 128, 1024]);  addmm_152 = None
        view_425 = torch.ops.aten.view.default(add_154, [2048, 1024])
        permute_265 = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_153 = torch.ops.aten.addmm.default(arg412_1, view_425, permute_265);  arg412_1 = view_425 = permute_265 = None
        view_426 = torch.ops.aten.view.default(addmm_153, [16, 128, 1024]);  addmm_153 = None
        view_427 = torch.ops.aten.view.default(view_426, [16, -1, 16, 64]);  view_426 = None
        permute_266 = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_154 = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_428 = torch.ops.aten.view.default(add_154, [2048, 1024]);  add_154 = None
        permute_267 = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        addmm_154 = torch.ops.aten.addmm.default(arg414_1, view_428, permute_267);  arg414_1 = view_428 = permute_267 = None
        view_429 = torch.ops.aten.view.default(addmm_154, [16, 128, 1024]);  addmm_154 = None
        view_430 = torch.ops.aten.view.default(view_429, [16, -1, 16, 64]);  view_429 = None
        permute_268 = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_155 = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
        view_431 = torch.ops.aten.view.default(view_424, [16, 128, 16, 64]);  view_424 = None
        permute_269 = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
        clone_156 = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
        expand_26 = torch.ops.aten.expand.default(unsqueeze_19, [16, 1, 128, 128]);  unsqueeze_19 = None
        expand_27 = torch.ops.aten.expand.default(expand_26, [16, 16, 128, 128]);  expand_26 = None
        _scaled_dot_product_efficient_attention_28 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_156, clone_154, clone_155, expand_27, False);  clone_156 = expand_27 = None
        getitem_212 = _scaled_dot_product_efficient_attention_28[0];  _scaled_dot_product_efficient_attention_28 = None
        permute_270 = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3]);  getitem_212 = None
        view_432 = torch.ops.aten.view.default(permute_270, [16, 128, 1024]);  permute_270 = None
        view_433 = torch.ops.aten.view.default(view_432, [2048, 1024]);  view_432 = None
        permute_271 = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_155 = torch.ops.aten.addmm.default(arg416_1, view_433, permute_271);  arg416_1 = view_433 = permute_271 = None
        view_434 = torch.ops.aten.view.default(addmm_155, [16, 128, 1024]);  addmm_155 = None
        add_155 = torch.ops.aten.add.Tensor(add_152, view_434);  add_152 = view_434 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
        getitem_216 = var_mean_50[0]
        getitem_217 = var_mean_50[1];  var_mean_50 = None
        add_156 = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_50 = torch.ops.aten.sub.Tensor(add_155, getitem_217);  getitem_217 = None
        mul_104 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, arg417_1);  mul_104 = arg417_1 = None
        add_157 = torch.ops.aten.add.Tensor(mul_105, arg418_1);  mul_105 = arg418_1 = None
        view_435 = torch.ops.aten.view.default(add_157, [2048, 1024]);  add_157 = None
        permute_272 = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_156 = torch.ops.aten.addmm.default(arg420_1, view_435, permute_272);  arg420_1 = view_435 = permute_272 = None
        view_436 = torch.ops.aten.view.default(addmm_156, [16, 128, 1024]);  addmm_156 = None
        view_437 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_273 = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_157 = torch.ops.aten.addmm.default(arg422_1, view_437, permute_273);  arg422_1 = view_437 = permute_273 = None
        view_438 = torch.ops.aten.view.default(addmm_157, [16, 128, 1024]);  addmm_157 = None
        view_439 = torch.ops.aten.view.default(view_438, [16, -1, 16, 64]);  view_438 = None
        permute_274 = torch.ops.aten.permute.default(view_439, [0, 2, 1, 3]);  view_439 = None
        clone_158 = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
        view_440 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_275 = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_158 = torch.ops.aten.addmm.default(arg424_1, view_440, permute_275);  arg424_1 = view_440 = permute_275 = None
        view_441 = torch.ops.aten.view.default(addmm_158, [16, 128, 1024]);  addmm_158 = None
        view_442 = torch.ops.aten.view.default(view_441, [16, -1, 16, 64]);  view_441 = None
        permute_276 = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        clone_159 = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
        view_443 = torch.ops.aten.view.default(view_436, [16, 128, 16, 64]);  view_436 = None
        permute_277 = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        clone_160 = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
        _scaled_dot_product_efficient_attention_29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_160, clone_158, clone_159, None, False);  clone_160 = None
        getitem_218 = _scaled_dot_product_efficient_attention_29[0];  _scaled_dot_product_efficient_attention_29 = None
        permute_278 = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        view_444 = torch.ops.aten.view.default(permute_278, [16, 128, 1024]);  permute_278 = None
        view_445 = torch.ops.aten.view.default(view_444, [2048, 1024]);  view_444 = None
        permute_279 = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        addmm_159 = torch.ops.aten.addmm.default(arg426_1, view_445, permute_279);  arg426_1 = view_445 = permute_279 = None
        view_446 = torch.ops.aten.view.default(addmm_159, [16, 128, 1024]);  addmm_159 = None
        add_158 = torch.ops.aten.add.Tensor(add_155, view_446);  add_155 = view_446 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_222 = var_mean_51[0]
        getitem_223 = var_mean_51[1];  var_mean_51 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_158, getitem_223);  getitem_223 = None
        mul_106 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, arg427_1);  mul_106 = arg427_1 = None
        add_160 = torch.ops.aten.add.Tensor(mul_107, arg428_1);  mul_107 = arg428_1 = None
        view_447 = torch.ops.aten.view.default(add_160, [2048, 1024]);  add_160 = None
        permute_280 = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        addmm_160 = torch.ops.aten.addmm.default(arg430_1, view_447, permute_280);  arg430_1 = view_447 = permute_280 = None
        view_448 = torch.ops.aten.view.default(addmm_160, [16, 128, 4096]);  addmm_160 = None
        relu_20 = torch.ops.aten.relu.default(view_448);  view_448 = None
        view_449 = torch.ops.aten.view.default(relu_20, [2048, 4096]);  relu_20 = None
        permute_281 = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_161 = torch.ops.aten.addmm.default(arg432_1, view_449, permute_281);  arg432_1 = view_449 = permute_281 = None
        view_450 = torch.ops.aten.view.default(addmm_161, [16, 128, 1024]);  addmm_161 = None
        add_161 = torch.ops.aten.add.Tensor(add_158, view_450);  add_158 = view_450 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_224 = var_mean_52[0]
        getitem_225 = var_mean_52[1];  var_mean_52 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_161, getitem_225);  getitem_225 = None
        mul_108 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, arg433_1);  mul_108 = arg433_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_109, arg434_1);  mul_109 = arg434_1 = None
        view_451 = torch.ops.aten.view.default(add_163, [2048, 1024])
        permute_282 = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_162 = torch.ops.aten.addmm.default(arg436_1, view_451, permute_282);  arg436_1 = view_451 = permute_282 = None
        view_452 = torch.ops.aten.view.default(addmm_162, [16, 128, 1024]);  addmm_162 = None
        view_453 = torch.ops.aten.view.default(add_163, [2048, 1024])
        permute_283 = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_163 = torch.ops.aten.addmm.default(arg438_1, view_453, permute_283);  arg438_1 = view_453 = permute_283 = None
        view_454 = torch.ops.aten.view.default(addmm_163, [16, 128, 1024]);  addmm_163 = None
        view_455 = torch.ops.aten.view.default(view_454, [16, -1, 16, 64]);  view_454 = None
        permute_284 = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        clone_164 = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        view_456 = torch.ops.aten.view.default(add_163, [2048, 1024]);  add_163 = None
        permute_285 = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
        addmm_164 = torch.ops.aten.addmm.default(arg440_1, view_456, permute_285);  arg440_1 = view_456 = permute_285 = None
        view_457 = torch.ops.aten.view.default(addmm_164, [16, 128, 1024]);  addmm_164 = None
        view_458 = torch.ops.aten.view.default(view_457, [16, -1, 16, 64]);  view_457 = None
        permute_286 = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_165 = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_459 = torch.ops.aten.view.default(view_452, [16, 128, 16, 64]);  view_452 = None
        permute_287 = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        clone_166 = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
        expand_29 = torch.ops.aten.expand.default(unsqueeze_21, [16, 1, 128, 128]);  unsqueeze_21 = None
        expand_30 = torch.ops.aten.expand.default(expand_29, [16, 16, 128, 128]);  expand_29 = None
        _scaled_dot_product_efficient_attention_30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_166, clone_164, clone_165, expand_30, False);  clone_166 = expand_30 = None
        getitem_226 = _scaled_dot_product_efficient_attention_30[0];  _scaled_dot_product_efficient_attention_30 = None
        permute_288 = torch.ops.aten.permute.default(getitem_226, [0, 2, 1, 3]);  getitem_226 = None
        view_460 = torch.ops.aten.view.default(permute_288, [16, 128, 1024]);  permute_288 = None
        view_461 = torch.ops.aten.view.default(view_460, [2048, 1024]);  view_460 = None
        permute_289 = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_165 = torch.ops.aten.addmm.default(arg442_1, view_461, permute_289);  arg442_1 = view_461 = permute_289 = None
        view_462 = torch.ops.aten.view.default(addmm_165, [16, 128, 1024]);  addmm_165 = None
        add_164 = torch.ops.aten.add.Tensor(add_161, view_462);  add_161 = view_462 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_230 = var_mean_53[0]
        getitem_231 = var_mean_53[1];  var_mean_53 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_230, 1e-05);  getitem_230 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_53 = torch.ops.aten.sub.Tensor(add_164, getitem_231);  getitem_231 = None
        mul_110 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, arg443_1);  mul_110 = arg443_1 = None
        add_166 = torch.ops.aten.add.Tensor(mul_111, arg444_1);  mul_111 = arg444_1 = None
        view_463 = torch.ops.aten.view.default(add_166, [2048, 1024]);  add_166 = None
        permute_290 = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_166 = torch.ops.aten.addmm.default(arg446_1, view_463, permute_290);  arg446_1 = view_463 = permute_290 = None
        view_464 = torch.ops.aten.view.default(addmm_166, [16, 128, 1024]);  addmm_166 = None
        view_465 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_291 = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_167 = torch.ops.aten.addmm.default(arg448_1, view_465, permute_291);  arg448_1 = view_465 = permute_291 = None
        view_466 = torch.ops.aten.view.default(addmm_167, [16, 128, 1024]);  addmm_167 = None
        view_467 = torch.ops.aten.view.default(view_466, [16, -1, 16, 64]);  view_466 = None
        permute_292 = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
        clone_168 = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_468 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_293 = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        addmm_168 = torch.ops.aten.addmm.default(arg450_1, view_468, permute_293);  arg450_1 = view_468 = permute_293 = None
        view_469 = torch.ops.aten.view.default(addmm_168, [16, 128, 1024]);  addmm_168 = None
        view_470 = torch.ops.aten.view.default(view_469, [16, -1, 16, 64]);  view_469 = None
        permute_294 = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        clone_169 = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_471 = torch.ops.aten.view.default(view_464, [16, 128, 16, 64]);  view_464 = None
        permute_295 = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
        clone_170 = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
        _scaled_dot_product_efficient_attention_31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_170, clone_168, clone_169, None, False);  clone_170 = None
        getitem_232 = _scaled_dot_product_efficient_attention_31[0];  _scaled_dot_product_efficient_attention_31 = None
        permute_296 = torch.ops.aten.permute.default(getitem_232, [0, 2, 1, 3]);  getitem_232 = None
        view_472 = torch.ops.aten.view.default(permute_296, [16, 128, 1024]);  permute_296 = None
        view_473 = torch.ops.aten.view.default(view_472, [2048, 1024]);  view_472 = None
        permute_297 = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg452_1, view_473, permute_297);  arg452_1 = view_473 = permute_297 = None
        view_474 = torch.ops.aten.view.default(addmm_169, [16, 128, 1024]);  addmm_169 = None
        add_167 = torch.ops.aten.add.Tensor(add_164, view_474);  add_164 = view_474 = None
        var_mean_54 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
        getitem_236 = var_mean_54[0]
        getitem_237 = var_mean_54[1];  var_mean_54 = None
        add_168 = torch.ops.aten.add.Tensor(getitem_236, 1e-05);  getitem_236 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_167, getitem_237);  getitem_237 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, arg453_1);  mul_112 = arg453_1 = None
        add_169 = torch.ops.aten.add.Tensor(mul_113, arg454_1);  mul_113 = arg454_1 = None
        view_475 = torch.ops.aten.view.default(add_169, [2048, 1024]);  add_169 = None
        permute_298 = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
        addmm_170 = torch.ops.aten.addmm.default(arg456_1, view_475, permute_298);  arg456_1 = view_475 = permute_298 = None
        view_476 = torch.ops.aten.view.default(addmm_170, [16, 128, 4096]);  addmm_170 = None
        relu_21 = torch.ops.aten.relu.default(view_476);  view_476 = None
        view_477 = torch.ops.aten.view.default(relu_21, [2048, 4096]);  relu_21 = None
        permute_299 = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_171 = torch.ops.aten.addmm.default(arg458_1, view_477, permute_299);  arg458_1 = view_477 = permute_299 = None
        view_478 = torch.ops.aten.view.default(addmm_171, [16, 128, 1024]);  addmm_171 = None
        add_170 = torch.ops.aten.add.Tensor(add_167, view_478);  add_167 = view_478 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
        getitem_238 = var_mean_55[0]
        getitem_239 = var_mean_55[1];  var_mean_55 = None
        add_171 = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_55 = torch.ops.aten.sub.Tensor(add_170, getitem_239);  getitem_239 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, arg459_1);  mul_114 = arg459_1 = None
        add_172 = torch.ops.aten.add.Tensor(mul_115, arg460_1);  mul_115 = arg460_1 = None
        view_479 = torch.ops.aten.view.default(add_172, [2048, 1024])
        permute_300 = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_172 = torch.ops.aten.addmm.default(arg462_1, view_479, permute_300);  arg462_1 = view_479 = permute_300 = None
        view_480 = torch.ops.aten.view.default(addmm_172, [16, 128, 1024]);  addmm_172 = None
        view_481 = torch.ops.aten.view.default(add_172, [2048, 1024])
        permute_301 = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_173 = torch.ops.aten.addmm.default(arg464_1, view_481, permute_301);  arg464_1 = view_481 = permute_301 = None
        view_482 = torch.ops.aten.view.default(addmm_173, [16, 128, 1024]);  addmm_173 = None
        view_483 = torch.ops.aten.view.default(view_482, [16, -1, 16, 64]);  view_482 = None
        permute_302 = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
        clone_174 = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_484 = torch.ops.aten.view.default(add_172, [2048, 1024]);  add_172 = None
        permute_303 = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_174 = torch.ops.aten.addmm.default(arg466_1, view_484, permute_303);  arg466_1 = view_484 = permute_303 = None
        view_485 = torch.ops.aten.view.default(addmm_174, [16, 128, 1024]);  addmm_174 = None
        view_486 = torch.ops.aten.view.default(view_485, [16, -1, 16, 64]);  view_485 = None
        permute_304 = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_175 = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
        view_487 = torch.ops.aten.view.default(view_480, [16, 128, 16, 64]);  view_480 = None
        permute_305 = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
        clone_176 = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
        expand_32 = torch.ops.aten.expand.default(unsqueeze_23, [16, 1, 128, 128]);  unsqueeze_23 = None
        expand_33 = torch.ops.aten.expand.default(expand_32, [16, 16, 128, 128]);  expand_32 = None
        _scaled_dot_product_efficient_attention_32 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_176, clone_174, clone_175, expand_33, False);  clone_176 = expand_33 = None
        getitem_240 = _scaled_dot_product_efficient_attention_32[0];  _scaled_dot_product_efficient_attention_32 = None
        permute_306 = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3]);  getitem_240 = None
        view_488 = torch.ops.aten.view.default(permute_306, [16, 128, 1024]);  permute_306 = None
        view_489 = torch.ops.aten.view.default(view_488, [2048, 1024]);  view_488 = None
        permute_307 = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        addmm_175 = torch.ops.aten.addmm.default(arg468_1, view_489, permute_307);  arg468_1 = view_489 = permute_307 = None
        view_490 = torch.ops.aten.view.default(addmm_175, [16, 128, 1024]);  addmm_175 = None
        add_173 = torch.ops.aten.add.Tensor(add_170, view_490);  add_170 = view_490 = None
        var_mean_56 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_244 = var_mean_56[0]
        getitem_245 = var_mean_56[1];  var_mean_56 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_244, 1e-05);  getitem_244 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_56 = torch.ops.aten.sub.Tensor(add_173, getitem_245);  getitem_245 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, arg469_1);  mul_116 = arg469_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_117, arg470_1);  mul_117 = arg470_1 = None
        view_491 = torch.ops.aten.view.default(add_175, [2048, 1024]);  add_175 = None
        permute_308 = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        addmm_176 = torch.ops.aten.addmm.default(arg472_1, view_491, permute_308);  arg472_1 = view_491 = permute_308 = None
        view_492 = torch.ops.aten.view.default(addmm_176, [16, 128, 1024]);  addmm_176 = None
        view_493 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_309 = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_177 = torch.ops.aten.addmm.default(arg474_1, view_493, permute_309);  arg474_1 = view_493 = permute_309 = None
        view_494 = torch.ops.aten.view.default(addmm_177, [16, 128, 1024]);  addmm_177 = None
        view_495 = torch.ops.aten.view.default(view_494, [16, -1, 16, 64]);  view_494 = None
        permute_310 = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None
        clone_178 = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_496 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_311 = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_178 = torch.ops.aten.addmm.default(arg476_1, view_496, permute_311);  arg476_1 = view_496 = permute_311 = None
        view_497 = torch.ops.aten.view.default(addmm_178, [16, 128, 1024]);  addmm_178 = None
        view_498 = torch.ops.aten.view.default(view_497, [16, -1, 16, 64]);  view_497 = None
        permute_312 = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_179 = torch.ops.aten.clone.default(permute_312, memory_format = torch.contiguous_format);  permute_312 = None
        view_499 = torch.ops.aten.view.default(view_492, [16, 128, 16, 64]);  view_492 = None
        permute_313 = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
        clone_180 = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
        _scaled_dot_product_efficient_attention_33 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_180, clone_178, clone_179, None, False);  clone_180 = None
        getitem_246 = _scaled_dot_product_efficient_attention_33[0];  _scaled_dot_product_efficient_attention_33 = None
        permute_314 = torch.ops.aten.permute.default(getitem_246, [0, 2, 1, 3]);  getitem_246 = None
        view_500 = torch.ops.aten.view.default(permute_314, [16, 128, 1024]);  permute_314 = None
        view_501 = torch.ops.aten.view.default(view_500, [2048, 1024]);  view_500 = None
        permute_315 = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_179 = torch.ops.aten.addmm.default(arg478_1, view_501, permute_315);  arg478_1 = view_501 = permute_315 = None
        view_502 = torch.ops.aten.view.default(addmm_179, [16, 128, 1024]);  addmm_179 = None
        add_176 = torch.ops.aten.add.Tensor(add_173, view_502);  add_173 = view_502 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
        getitem_250 = var_mean_57[0]
        getitem_251 = var_mean_57[1];  var_mean_57 = None
        add_177 = torch.ops.aten.add.Tensor(getitem_250, 1e-05);  getitem_250 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_176, getitem_251);  getitem_251 = None
        mul_118 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, arg479_1);  mul_118 = arg479_1 = None
        add_178 = torch.ops.aten.add.Tensor(mul_119, arg480_1);  mul_119 = arg480_1 = None
        view_503 = torch.ops.aten.view.default(add_178, [2048, 1024]);  add_178 = None
        permute_316 = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_180 = torch.ops.aten.addmm.default(arg482_1, view_503, permute_316);  arg482_1 = view_503 = permute_316 = None
        view_504 = torch.ops.aten.view.default(addmm_180, [16, 128, 4096]);  addmm_180 = None
        relu_22 = torch.ops.aten.relu.default(view_504);  view_504 = None
        view_505 = torch.ops.aten.view.default(relu_22, [2048, 4096]);  relu_22 = None
        permute_317 = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg484_1, view_505, permute_317);  arg484_1 = view_505 = permute_317 = None
        view_506 = torch.ops.aten.view.default(addmm_181, [16, 128, 1024]);  addmm_181 = None
        add_179 = torch.ops.aten.add.Tensor(add_176, view_506);  add_176 = view_506 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
        getitem_252 = var_mean_58[0]
        getitem_253 = var_mean_58[1];  var_mean_58 = None
        add_180 = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        sub_58 = torch.ops.aten.sub.Tensor(add_179, getitem_253);  getitem_253 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, arg485_1);  mul_120 = arg485_1 = None
        add_181 = torch.ops.aten.add.Tensor(mul_121, arg486_1);  mul_121 = arg486_1 = None
        view_507 = torch.ops.aten.view.default(add_181, [2048, 1024])
        permute_318 = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg488_1, view_507, permute_318);  arg488_1 = view_507 = permute_318 = None
        view_508 = torch.ops.aten.view.default(addmm_182, [16, 128, 1024]);  addmm_182 = None
        view_509 = torch.ops.aten.view.default(add_181, [2048, 1024])
        permute_319 = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg490_1, view_509, permute_319);  arg490_1 = view_509 = permute_319 = None
        view_510 = torch.ops.aten.view.default(addmm_183, [16, 128, 1024]);  addmm_183 = None
        view_511 = torch.ops.aten.view.default(view_510, [16, -1, 16, 64]);  view_510 = None
        permute_320 = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        clone_184 = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_512 = torch.ops.aten.view.default(add_181, [2048, 1024]);  add_181 = None
        permute_321 = torch.ops.aten.permute.default(arg491_1, [1, 0]);  arg491_1 = None
        addmm_184 = torch.ops.aten.addmm.default(arg492_1, view_512, permute_321);  arg492_1 = view_512 = permute_321 = None
        view_513 = torch.ops.aten.view.default(addmm_184, [16, 128, 1024]);  addmm_184 = None
        view_514 = torch.ops.aten.view.default(view_513, [16, -1, 16, 64]);  view_513 = None
        permute_322 = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        clone_185 = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_515 = torch.ops.aten.view.default(view_508, [16, 128, 16, 64]);  view_508 = None
        permute_323 = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
        clone_186 = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
        expand_35 = torch.ops.aten.expand.default(unsqueeze_25, [16, 1, 128, 128]);  unsqueeze_25 = None
        expand_36 = torch.ops.aten.expand.default(expand_35, [16, 16, 128, 128]);  expand_35 = None
        _scaled_dot_product_efficient_attention_34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_186, clone_184, clone_185, expand_36, False);  clone_186 = expand_36 = None
        getitem_254 = _scaled_dot_product_efficient_attention_34[0];  _scaled_dot_product_efficient_attention_34 = None
        permute_324 = torch.ops.aten.permute.default(getitem_254, [0, 2, 1, 3]);  getitem_254 = None
        view_516 = torch.ops.aten.view.default(permute_324, [16, 128, 1024]);  permute_324 = None
        view_517 = torch.ops.aten.view.default(view_516, [2048, 1024]);  view_516 = None
        permute_325 = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg494_1, view_517, permute_325);  arg494_1 = view_517 = permute_325 = None
        view_518 = torch.ops.aten.view.default(addmm_185, [16, 128, 1024]);  addmm_185 = None
        add_182 = torch.ops.aten.add.Tensor(add_179, view_518);  add_179 = view_518 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
        getitem_258 = var_mean_59[0]
        getitem_259 = var_mean_59[1];  var_mean_59 = None
        add_183 = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_182, getitem_259);  getitem_259 = None
        mul_122 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, arg495_1);  mul_122 = arg495_1 = None
        add_184 = torch.ops.aten.add.Tensor(mul_123, arg496_1);  mul_123 = arg496_1 = None
        view_519 = torch.ops.aten.view.default(add_184, [2048, 1024]);  add_184 = None
        permute_326 = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        addmm_186 = torch.ops.aten.addmm.default(arg498_1, view_519, permute_326);  arg498_1 = view_519 = permute_326 = None
        view_520 = torch.ops.aten.view.default(addmm_186, [16, 128, 1024]);  addmm_186 = None
        view_521 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_327 = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg500_1, view_521, permute_327);  arg500_1 = view_521 = permute_327 = None
        view_522 = torch.ops.aten.view.default(addmm_187, [16, 128, 1024]);  addmm_187 = None
        view_523 = torch.ops.aten.view.default(view_522, [16, -1, 16, 64]);  view_522 = None
        permute_328 = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
        clone_188 = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_524 = torch.ops.aten.view.default(add_76, [2048, 1024])
        permute_329 = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg502_1, view_524, permute_329);  arg502_1 = view_524 = permute_329 = None
        view_525 = torch.ops.aten.view.default(addmm_188, [16, 128, 1024]);  addmm_188 = None
        view_526 = torch.ops.aten.view.default(view_525, [16, -1, 16, 64]);  view_525 = None
        permute_330 = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        clone_189 = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_527 = torch.ops.aten.view.default(view_520, [16, 128, 16, 64]);  view_520 = None
        permute_331 = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
        clone_190 = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
        _scaled_dot_product_efficient_attention_35 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_190, clone_188, clone_189, None, False);  clone_190 = None
        getitem_260 = _scaled_dot_product_efficient_attention_35[0];  _scaled_dot_product_efficient_attention_35 = None
        permute_332 = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        view_528 = torch.ops.aten.view.default(permute_332, [16, 128, 1024]);  permute_332 = None
        view_529 = torch.ops.aten.view.default(view_528, [2048, 1024]);  view_528 = None
        permute_333 = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg504_1, view_529, permute_333);  arg504_1 = view_529 = permute_333 = None
        view_530 = torch.ops.aten.view.default(addmm_189, [16, 128, 1024]);  addmm_189 = None
        add_185 = torch.ops.aten.add.Tensor(add_182, view_530);  add_182 = view_530 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_264 = var_mean_60[0]
        getitem_265 = var_mean_60[1];  var_mean_60 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_60 = torch.ops.aten.sub.Tensor(add_185, getitem_265);  getitem_265 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, arg505_1);  mul_124 = arg505_1 = None
        add_187 = torch.ops.aten.add.Tensor(mul_125, arg506_1);  mul_125 = arg506_1 = None
        view_531 = torch.ops.aten.view.default(add_187, [2048, 1024]);  add_187 = None
        permute_334 = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg508_1, view_531, permute_334);  arg508_1 = view_531 = permute_334 = None
        view_532 = torch.ops.aten.view.default(addmm_190, [16, 128, 4096]);  addmm_190 = None
        relu_23 = torch.ops.aten.relu.default(view_532);  view_532 = None
        view_533 = torch.ops.aten.view.default(relu_23, [2048, 4096]);  relu_23 = None
        permute_335 = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_191 = torch.ops.aten.addmm.default(arg510_1, view_533, permute_335);  arg510_1 = view_533 = permute_335 = None
        view_534 = torch.ops.aten.view.default(addmm_191, [16, 128, 1024]);  addmm_191 = None
        add_188 = torch.ops.aten.add.Tensor(add_185, view_534);  add_185 = view_534 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(add_188, [2], correction = 0, keepdim = True)
        getitem_266 = var_mean_61[0]
        getitem_267 = var_mean_61[1];  var_mean_61 = None
        add_189 = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        sub_61 = torch.ops.aten.sub.Tensor(add_188, getitem_267);  add_188 = getitem_267 = None
        mul_126 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_126, arg511_1);  mul_126 = arg511_1 = None
        add_190 = torch.ops.aten.add.Tensor(mul_127, arg512_1);  mul_127 = arg512_1 = None
        permute_336 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_535 = torch.ops.aten.view.default(add_190, [2048, 1024]);  add_190 = None
        mm = torch.ops.aten.mm.default(view_535, permute_336);  view_535 = permute_336 = None
        view_536 = torch.ops.aten.view.default(mm, [16, 128, 128112]);  mm = None
        view_537 = torch.ops.aten.view.default(view_536, [-1, 128112])
        view_538 = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        amax = torch.ops.aten.amax.default(view_537, [1], True)
        sub_62 = torch.ops.aten.sub.Tensor(view_537, amax);  view_537 = amax = None
        exp = torch.ops.aten.exp.default(sub_62)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, log);  sub_62 = log = None
        ne_2 = torch.ops.aten.ne.Scalar(view_538, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_2, view_538, full_default_2);  ne_2 = full_default_2 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather = torch.ops.aten.gather.default(sub_63, 1, unsqueeze_26);  sub_63 = unsqueeze_26 = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_3 = torch.ops.aten.ne.Scalar(view_538, -100)
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(ne_3, neg, full_default_3);  ne_3 = neg = full_default_3 = None
        ne_4 = torch.ops.aten.ne.Scalar(view_538, -100);  view_538 = None
        sum_2 = torch.ops.aten.sum.default(ne_4);  ne_4 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3 = torch.ops.aten.sum.default(where_2);  where_2 = None
        div = torch.ops.aten.div.Tensor(sum_3, convert_element_type_6);  sum_3 = convert_element_type_6 = None
        return (div, view_536, clone_74, clone_75, clone_78, clone_79, clone_84, clone_85, clone_88, clone_89, clone_94, clone_95, clone_98, clone_99, clone_104, clone_105, clone_108, clone_109, clone_114, clone_115, clone_118, clone_119, clone_124, clone_125, clone_128, clone_129, clone_134, clone_135, clone_138, clone_139, clone_144, clone_145, clone_148, clone_149, clone_154, clone_155, clone_158, clone_159, clone_164, clone_165, clone_168, clone_169, clone_174, clone_175, clone_178, clone_179, clone_184, clone_185, clone_188, clone_189, add_76)
        
def load_args(reader):
    buf0 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (16, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (16, 128), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 524746752, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128112, 1024), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4202496, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1026, 1024), is_leaf=True)  # arg3_1
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
    buf198 = reader.storage(None, 4202496, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1026, 1024), is_leaf=True)  # arg198_1
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
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)