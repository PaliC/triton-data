
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

torch._inductor.config.fallback_random = True
torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.6.0.dev20241021+cu118
# torch cuda version: 11.8
# torch git version: 5553778a0095e7234b2cd0874c2ff4dcc0216323


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA H100 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1):
        convolution_111 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_257 = torch.ops.aten.add.Tensor(arg3_1, 0.001);  arg3_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_333 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_258 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
        relu_111 = torch.ops.aten.relu.default(add_258);  add_258 = None
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_111, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_111 = None
        getitem_2 = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        add_259 = torch.ops.aten.add.Tensor(arg7_1, 0.001);  arg7_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_336 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(getitem_2, unsqueeze_897);  unsqueeze_897 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_260 = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
        relu_112 = torch.ops.aten.relu.default(add_260);  add_260 = None
        convolution_112 = torch.ops.aten.convolution.default(relu_112, arg10_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_112 = arg10_1 = None
        slice_314 = torch.ops.aten.slice.Tensor(convolution_112, 1, 0, 256)
        slice_318 = torch.ops.aten.slice.Tensor(convolution_112, 1, 256, 9223372036854775807);  convolution_112 = None
        add_261 = torch.ops.aten.add.Tensor(arg12_1, 0.001);  arg12_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_339 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(getitem_2, unsqueeze_905);  getitem_2 = unsqueeze_905 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_262 = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
        relu_113 = torch.ops.aten.relu.default(add_262);  add_262 = None
        convolution_113 = torch.ops.aten.convolution.default(relu_113, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_113 = arg15_1 = None
        add_263 = torch.ops.aten.add.Tensor(arg17_1, 0.001);  arg17_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_342 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_913);  convolution_113 = unsqueeze_913 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_264 = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
        relu_114 = torch.ops.aten.relu.default(add_264);  add_264 = None
        convolution_114 = torch.ops.aten.convolution.default(relu_114, arg20_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_114 = arg20_1 = None
        add_265 = torch.ops.aten.add.Tensor(arg22_1, 0.001);  arg22_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_345 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_921);  convolution_114 = unsqueeze_921 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_266 = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
        relu_115 = torch.ops.aten.relu.default(add_266);  add_266 = None
        convolution_115 = torch.ops.aten.convolution.default(relu_115, arg25_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_115 = arg25_1 = None
        slice_322 = torch.ops.aten.slice.Tensor(convolution_115, 1, 0, 256)
        slice_326 = torch.ops.aten.slice.Tensor(convolution_115, 1, 256, 9223372036854775807);  convolution_115 = None
        add_267 = torch.ops.aten.add.Tensor(slice_314, slice_322);  slice_314 = slice_322 = None
        cat_70 = torch.ops.aten.cat.default([slice_318, slice_326], 1);  slice_318 = slice_326 = None
        cat_71 = torch.ops.aten.cat.default([add_267, cat_70], 1)
        add_268 = torch.ops.aten.add.Tensor(arg27_1, 0.001);  arg27_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_268);  add_268 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_348 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928 = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_929 = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116 = torch.ops.aten.sub.Tensor(cat_71, unsqueeze_929);  cat_71 = unsqueeze_929 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_269 = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
        relu_116 = torch.ops.aten.relu.default(add_269);  add_269 = None
        convolution_116 = torch.ops.aten.convolution.default(relu_116, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_116 = arg30_1 = None
        add_270 = torch.ops.aten.add.Tensor(arg32_1, 0.001);  arg32_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_270);  add_270 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_351 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_939 = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117 = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_937);  convolution_116 = unsqueeze_937 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_941 = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_271 = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
        relu_117 = torch.ops.aten.relu.default(add_271);  add_271 = None
        convolution_117 = torch.ops.aten.convolution.default(relu_117, arg35_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_117 = arg35_1 = None
        add_272 = torch.ops.aten.add.Tensor(arg37_1, 0.001);  arg37_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_272);  add_272 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_354 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118 = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_945);  convolution_117 = unsqueeze_945 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_951 = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_273 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
        relu_118 = torch.ops.aten.relu.default(add_273);  add_273 = None
        convolution_118 = torch.ops.aten.convolution.default(relu_118, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_118 = arg40_1 = None
        slice_330 = torch.ops.aten.slice.Tensor(convolution_118, 1, 0, 256)
        slice_334 = torch.ops.aten.slice.Tensor(convolution_118, 1, 256, 9223372036854775807);  convolution_118 = None
        add_274 = torch.ops.aten.add.Tensor(add_267, slice_330);  add_267 = slice_330 = None
        cat_72 = torch.ops.aten.cat.default([cat_70, slice_334], 1);  cat_70 = slice_334 = None
        cat_73 = torch.ops.aten.cat.default([add_274, cat_72], 1)
        add_275 = torch.ops.aten.add.Tensor(arg42_1, 0.001);  arg42_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_275);  add_275 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_357 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952 = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_953 = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119 = torch.ops.aten.sub.Tensor(cat_73, unsqueeze_953);  cat_73 = unsqueeze_953 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_276 = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
        relu_119 = torch.ops.aten.relu.default(add_276);  add_276 = None
        convolution_119 = torch.ops.aten.convolution.default(relu_119, arg45_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_119 = arg45_1 = None
        add_277 = torch.ops.aten.add.Tensor(arg47_1, 0.001);  arg47_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_963 = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120 = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_961);  convolution_119 = unsqueeze_961 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_965 = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_278 = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
        relu_120 = torch.ops.aten.relu.default(add_278);  add_278 = None
        convolution_120 = torch.ops.aten.convolution.default(relu_120, arg50_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_120 = arg50_1 = None
        add_279 = torch.ops.aten.add.Tensor(arg52_1, 0.001);  arg52_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_363 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121 = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_969);  convolution_120 = unsqueeze_969 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_975 = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_280 = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
        relu_121 = torch.ops.aten.relu.default(add_280);  add_280 = None
        convolution_121 = torch.ops.aten.convolution.default(relu_121, arg55_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_121 = arg55_1 = None
        slice_338 = torch.ops.aten.slice.Tensor(convolution_121, 1, 0, 256)
        slice_342 = torch.ops.aten.slice.Tensor(convolution_121, 1, 256, 9223372036854775807);  convolution_121 = None
        add_281 = torch.ops.aten.add.Tensor(add_274, slice_338);  add_274 = slice_338 = None
        cat_74 = torch.ops.aten.cat.default([cat_72, slice_342], 1);  cat_72 = slice_342 = None
        cat_75 = torch.ops.aten.cat.default([add_281, cat_74], 1)
        add_282 = torch.ops.aten.add.Tensor(arg57_1, 0.001);  arg57_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_282);  add_282 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_366 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_977 = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122 = torch.ops.aten.sub.Tensor(cat_75, unsqueeze_977);  cat_75 = unsqueeze_977 = None
        mul_367 = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_283 = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
        relu_122 = torch.ops.aten.relu.default(add_283);  add_283 = None
        convolution_122 = torch.ops.aten.convolution.default(relu_122, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_122 = arg60_1 = None
        add_284 = torch.ops.aten.add.Tensor(arg62_1, 0.001);  arg62_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_284);  add_284 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_369 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_987 = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_985);  convolution_122 = unsqueeze_985 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_989 = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_285 = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
        relu_123 = torch.ops.aten.relu.default(add_285);  add_285 = None
        convolution_123 = torch.ops.aten.convolution.default(relu_123, arg65_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_123 = arg65_1 = None
        add_286 = torch.ops.aten.add.Tensor(arg67_1, 0.001);  arg67_1 = None
        sqrt_124 = torch.ops.aten.sqrt.default(add_286);  add_286 = None
        reciprocal_124 = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992 = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_993 = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994 = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_995 = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124 = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_993);  convolution_123 = unsqueeze_993 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_997 = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
        unsqueeze_998 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_999 = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_287 = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
        relu_124 = torch.ops.aten.relu.default(add_287);  add_287 = None
        convolution_124 = torch.ops.aten.convolution.default(relu_124, arg70_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_124 = arg70_1 = None
        slice_346 = torch.ops.aten.slice.Tensor(convolution_124, 1, 0, 256)
        slice_350 = torch.ops.aten.slice.Tensor(convolution_124, 1, 256, 9223372036854775807);  convolution_124 = None
        add_288 = torch.ops.aten.add.Tensor(add_281, slice_346);  add_281 = slice_346 = None
        cat_76 = torch.ops.aten.cat.default([cat_74, slice_350], 1);  cat_74 = slice_350 = None
        cat_77 = torch.ops.aten.cat.default([add_288, cat_76], 1);  add_288 = cat_76 = None
        add_289 = torch.ops.aten.add.Tensor(arg72_1, 0.001);  arg72_1 = None
        sqrt_125 = torch.ops.aten.sqrt.default(add_289);  add_289 = None
        reciprocal_125 = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_375 = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000 = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_1001 = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002 = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_1003 = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125 = torch.ops.aten.sub.Tensor(cat_77, unsqueeze_1001);  unsqueeze_1001 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_1005 = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
        unsqueeze_1006 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_1007 = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_290 = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
        relu_125 = torch.ops.aten.relu.default(add_290);  add_290 = None
        convolution_125 = torch.ops.aten.convolution.default(relu_125, arg75_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_125 = arg75_1 = None
        slice_354 = torch.ops.aten.slice.Tensor(convolution_125, 1, 0, 512)
        slice_358 = torch.ops.aten.slice.Tensor(convolution_125, 1, 512, 9223372036854775807);  convolution_125 = None
        add_291 = torch.ops.aten.add.Tensor(arg77_1, 0.001);  arg77_1 = None
        sqrt_126 = torch.ops.aten.sqrt.default(add_291);  add_291 = None
        reciprocal_126 = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_1009 = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010 = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_1011 = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126 = torch.ops.aten.sub.Tensor(cat_77, unsqueeze_1009);  cat_77 = unsqueeze_1009 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_1013 = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
        unsqueeze_1014 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_1015 = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_292 = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
        relu_126 = torch.ops.aten.relu.default(add_292);  add_292 = None
        convolution_126 = torch.ops.aten.convolution.default(relu_126, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_126 = arg80_1 = None
        add_293 = torch.ops.aten.add.Tensor(arg82_1, 0.001);  arg82_1 = None
        sqrt_127 = torch.ops.aten.sqrt.default(add_293);  add_293 = None
        reciprocal_127 = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_381 = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_1017 = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018 = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_1019 = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1017);  convolution_126 = unsqueeze_1017 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_1021 = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
        unsqueeze_1022 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_1023 = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_294 = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
        relu_127 = torch.ops.aten.relu.default(add_294);  add_294 = None
        convolution_127 = torch.ops.aten.convolution.default(relu_127, arg85_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_127 = arg85_1 = None
        add_295 = torch.ops.aten.add.Tensor(arg87_1, 0.001);  arg87_1 = None
        sqrt_128 = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_128 = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_384 = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1024 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_1025 = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        unsqueeze_1026 = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_1027 = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        sub_128 = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1025);  convolution_127 = unsqueeze_1025 = None
        mul_385 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_1029 = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_386 = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1029);  mul_385 = unsqueeze_1029 = None
        unsqueeze_1030 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_1031 = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_296 = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1031);  mul_386 = unsqueeze_1031 = None
        relu_128 = torch.ops.aten.relu.default(add_296);  add_296 = None
        convolution_128 = torch.ops.aten.convolution.default(relu_128, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_128 = arg90_1 = None
        slice_362 = torch.ops.aten.slice.Tensor(convolution_128, 1, 0, 512)
        slice_366 = torch.ops.aten.slice.Tensor(convolution_128, 1, 512, 9223372036854775807);  convolution_128 = None
        add_297 = torch.ops.aten.add.Tensor(slice_354, slice_362);  slice_354 = slice_362 = None
        cat_78 = torch.ops.aten.cat.default([slice_358, slice_366], 1);  slice_358 = slice_366 = None
        cat_79 = torch.ops.aten.cat.default([add_297, cat_78], 1)
        add_298 = torch.ops.aten.add.Tensor(arg92_1, 0.001);  arg92_1 = None
        sqrt_129 = torch.ops.aten.sqrt.default(add_298);  add_298 = None
        reciprocal_129 = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_387 = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1032 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_1033 = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        unsqueeze_1034 = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_1035 = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        sub_129 = torch.ops.aten.sub.Tensor(cat_79, unsqueeze_1033);  cat_79 = unsqueeze_1033 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_1037 = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1037);  mul_388 = unsqueeze_1037 = None
        unsqueeze_1038 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_1039 = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_299 = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1039);  mul_389 = unsqueeze_1039 = None
        relu_129 = torch.ops.aten.relu.default(add_299);  add_299 = None
        convolution_129 = torch.ops.aten.convolution.default(relu_129, arg95_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_129 = arg95_1 = None
        add_300 = torch.ops.aten.add.Tensor(arg97_1, 0.001);  arg97_1 = None
        sqrt_130 = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_130 = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_390 = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1040 = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_1041 = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        unsqueeze_1042 = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_1043 = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        sub_130 = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1041);  convolution_129 = unsqueeze_1041 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_1045 = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1045);  mul_391 = unsqueeze_1045 = None
        unsqueeze_1046 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_1047 = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_301 = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1047);  mul_392 = unsqueeze_1047 = None
        relu_130 = torch.ops.aten.relu.default(add_301);  add_301 = None
        convolution_130 = torch.ops.aten.convolution.default(relu_130, arg100_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_130 = arg100_1 = None
        add_302 = torch.ops.aten.add.Tensor(arg102_1, 0.001);  arg102_1 = None
        sqrt_131 = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_131 = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_393 = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1048 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_1049 = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        unsqueeze_1050 = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_1051 = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        sub_131 = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1049);  convolution_130 = unsqueeze_1049 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_1053 = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_1053);  mul_394 = unsqueeze_1053 = None
        unsqueeze_1054 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_1055 = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_303 = torch.ops.aten.add.Tensor(mul_395, unsqueeze_1055);  mul_395 = unsqueeze_1055 = None
        relu_131 = torch.ops.aten.relu.default(add_303);  add_303 = None
        convolution_131 = torch.ops.aten.convolution.default(relu_131, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_131 = arg105_1 = None
        slice_370 = torch.ops.aten.slice.Tensor(convolution_131, 1, 0, 512)
        slice_374 = torch.ops.aten.slice.Tensor(convolution_131, 1, 512, 9223372036854775807);  convolution_131 = None
        add_304 = torch.ops.aten.add.Tensor(add_297, slice_370);  add_297 = slice_370 = None
        cat_80 = torch.ops.aten.cat.default([cat_78, slice_374], 1);  cat_78 = slice_374 = None
        cat_81 = torch.ops.aten.cat.default([add_304, cat_80], 1)
        add_305 = torch.ops.aten.add.Tensor(arg107_1, 0.001);  arg107_1 = None
        sqrt_132 = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_132 = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_396 = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1056 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_1057 = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        unsqueeze_1058 = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_1059 = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        sub_132 = torch.ops.aten.sub.Tensor(cat_81, unsqueeze_1057);  cat_81 = unsqueeze_1057 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060 = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_1061 = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_1061);  mul_397 = unsqueeze_1061 = None
        unsqueeze_1062 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_1063 = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_306 = torch.ops.aten.add.Tensor(mul_398, unsqueeze_1063);  mul_398 = unsqueeze_1063 = None
        relu_132 = torch.ops.aten.relu.default(add_306);  add_306 = None
        convolution_132 = torch.ops.aten.convolution.default(relu_132, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_132 = arg110_1 = None
        add_307 = torch.ops.aten.add.Tensor(arg112_1, 0.001);  arg112_1 = None
        sqrt_133 = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_133 = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_399 = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1064 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_1065 = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        unsqueeze_1066 = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
        unsqueeze_1067 = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        sub_133 = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1065);  convolution_132 = unsqueeze_1065 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_1069 = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_1069);  mul_400 = unsqueeze_1069 = None
        unsqueeze_1070 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_1071 = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_308 = torch.ops.aten.add.Tensor(mul_401, unsqueeze_1071);  mul_401 = unsqueeze_1071 = None
        relu_133 = torch.ops.aten.relu.default(add_308);  add_308 = None
        convolution_133 = torch.ops.aten.convolution.default(relu_133, arg115_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_133 = arg115_1 = None
        add_309 = torch.ops.aten.add.Tensor(arg117_1, 0.001);  arg117_1 = None
        sqrt_134 = torch.ops.aten.sqrt.default(add_309);  add_309 = None
        reciprocal_134 = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_402 = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1072 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_1073 = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        unsqueeze_1074 = torch.ops.aten.unsqueeze.default(mul_402, -1);  mul_402 = None
        unsqueeze_1075 = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        sub_134 = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1073);  convolution_133 = unsqueeze_1073 = None
        mul_403 = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_1077 = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_1077);  mul_403 = unsqueeze_1077 = None
        unsqueeze_1078 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_1079 = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_310 = torch.ops.aten.add.Tensor(mul_404, unsqueeze_1079);  mul_404 = unsqueeze_1079 = None
        relu_134 = torch.ops.aten.relu.default(add_310);  add_310 = None
        convolution_134 = torch.ops.aten.convolution.default(relu_134, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_134 = arg120_1 = None
        slice_378 = torch.ops.aten.slice.Tensor(convolution_134, 1, 0, 512)
        slice_382 = torch.ops.aten.slice.Tensor(convolution_134, 1, 512, 9223372036854775807);  convolution_134 = None
        add_311 = torch.ops.aten.add.Tensor(add_304, slice_378);  add_304 = slice_378 = None
        cat_82 = torch.ops.aten.cat.default([cat_80, slice_382], 1);  cat_80 = slice_382 = None
        cat_83 = torch.ops.aten.cat.default([add_311, cat_82], 1)
        add_312 = torch.ops.aten.add.Tensor(arg122_1, 0.001);  arg122_1 = None
        sqrt_135 = torch.ops.aten.sqrt.default(add_312);  add_312 = None
        reciprocal_135 = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_405 = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1080 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_1081 = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        unsqueeze_1082 = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
        unsqueeze_1083 = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        sub_135 = torch.ops.aten.sub.Tensor(cat_83, unsqueeze_1081);  cat_83 = unsqueeze_1081 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_1085 = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_1085);  mul_406 = unsqueeze_1085 = None
        unsqueeze_1086 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_1087 = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_313 = torch.ops.aten.add.Tensor(mul_407, unsqueeze_1087);  mul_407 = unsqueeze_1087 = None
        relu_135 = torch.ops.aten.relu.default(add_313);  add_313 = None
        convolution_135 = torch.ops.aten.convolution.default(relu_135, arg125_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_135 = arg125_1 = None
        add_314 = torch.ops.aten.add.Tensor(arg127_1, 0.001);  arg127_1 = None
        sqrt_136 = torch.ops.aten.sqrt.default(add_314);  add_314 = None
        reciprocal_136 = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_408 = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1088 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_1089 = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        unsqueeze_1090 = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_1091 = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        sub_136 = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1089);  convolution_135 = unsqueeze_1089 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_1093 = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_1093);  mul_409 = unsqueeze_1093 = None
        unsqueeze_1094 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_1095 = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_315 = torch.ops.aten.add.Tensor(mul_410, unsqueeze_1095);  mul_410 = unsqueeze_1095 = None
        relu_136 = torch.ops.aten.relu.default(add_315);  add_315 = None
        convolution_136 = torch.ops.aten.convolution.default(relu_136, arg130_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_136 = arg130_1 = None
        add_316 = torch.ops.aten.add.Tensor(arg132_1, 0.001);  arg132_1 = None
        sqrt_137 = torch.ops.aten.sqrt.default(add_316);  add_316 = None
        reciprocal_137 = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_411 = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1096 = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_1097 = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        unsqueeze_1098 = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_1099 = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        sub_137 = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1097);  convolution_136 = unsqueeze_1097 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_1101 = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_1101);  mul_412 = unsqueeze_1101 = None
        unsqueeze_1102 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_1103 = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_317 = torch.ops.aten.add.Tensor(mul_413, unsqueeze_1103);  mul_413 = unsqueeze_1103 = None
        relu_137 = torch.ops.aten.relu.default(add_317);  add_317 = None
        convolution_137 = torch.ops.aten.convolution.default(relu_137, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_137 = arg135_1 = None
        slice_386 = torch.ops.aten.slice.Tensor(convolution_137, 1, 0, 512)
        slice_390 = torch.ops.aten.slice.Tensor(convolution_137, 1, 512, 9223372036854775807);  convolution_137 = None
        add_318 = torch.ops.aten.add.Tensor(add_311, slice_386);  add_311 = slice_386 = None
        cat_84 = torch.ops.aten.cat.default([cat_82, slice_390], 1);  cat_82 = slice_390 = None
        cat_85 = torch.ops.aten.cat.default([add_318, cat_84], 1)
        add_319 = torch.ops.aten.add.Tensor(arg137_1, 0.001);  arg137_1 = None
        sqrt_138 = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_138 = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_414 = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1104 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_1105 = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        unsqueeze_1106 = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
        unsqueeze_1107 = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        sub_138 = torch.ops.aten.sub.Tensor(cat_85, unsqueeze_1105);  cat_85 = unsqueeze_1105 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_1109 = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_1109);  mul_415 = unsqueeze_1109 = None
        unsqueeze_1110 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_1111 = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_320 = torch.ops.aten.add.Tensor(mul_416, unsqueeze_1111);  mul_416 = unsqueeze_1111 = None
        relu_138 = torch.ops.aten.relu.default(add_320);  add_320 = None
        convolution_138 = torch.ops.aten.convolution.default(relu_138, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_138 = arg140_1 = None
        add_321 = torch.ops.aten.add.Tensor(arg142_1, 0.001);  arg142_1 = None
        sqrt_139 = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_139 = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_417 = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_1113 = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114 = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_1115 = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_139 = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1113);  convolution_138 = unsqueeze_1113 = None
        mul_418 = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_1117 = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_1117);  mul_418 = unsqueeze_1117 = None
        unsqueeze_1118 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_1119 = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_322 = torch.ops.aten.add.Tensor(mul_419, unsqueeze_1119);  mul_419 = unsqueeze_1119 = None
        relu_139 = torch.ops.aten.relu.default(add_322);  add_322 = None
        convolution_139 = torch.ops.aten.convolution.default(relu_139, arg145_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_139 = arg145_1 = None
        add_323 = torch.ops.aten.add.Tensor(arg147_1, 0.001);  arg147_1 = None
        sqrt_140 = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_140 = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_420 = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_1121 = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122 = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_1123 = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_140 = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1121);  convolution_139 = unsqueeze_1121 = None
        mul_421 = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_1125 = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_422 = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_1125);  mul_421 = unsqueeze_1125 = None
        unsqueeze_1126 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_1127 = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_324 = torch.ops.aten.add.Tensor(mul_422, unsqueeze_1127);  mul_422 = unsqueeze_1127 = None
        relu_140 = torch.ops.aten.relu.default(add_324);  add_324 = None
        convolution_140 = torch.ops.aten.convolution.default(relu_140, arg150_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_140 = arg150_1 = None
        slice_394 = torch.ops.aten.slice.Tensor(convolution_140, 1, 0, 512)
        slice_398 = torch.ops.aten.slice.Tensor(convolution_140, 1, 512, 9223372036854775807);  convolution_140 = None
        add_325 = torch.ops.aten.add.Tensor(add_318, slice_394);  add_318 = slice_394 = None
        cat_86 = torch.ops.aten.cat.default([cat_84, slice_398], 1);  cat_84 = slice_398 = None
        cat_87 = torch.ops.aten.cat.default([add_325, cat_86], 1)
        add_326 = torch.ops.aten.add.Tensor(arg152_1, 0.001);  arg152_1 = None
        sqrt_141 = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_141 = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_423 = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_1129 = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130 = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
        unsqueeze_1131 = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_141 = torch.ops.aten.sub.Tensor(cat_87, unsqueeze_1129);  cat_87 = unsqueeze_1129 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132 = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_1133 = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_425 = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_1133);  mul_424 = unsqueeze_1133 = None
        unsqueeze_1134 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_1135 = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_327 = torch.ops.aten.add.Tensor(mul_425, unsqueeze_1135);  mul_425 = unsqueeze_1135 = None
        relu_141 = torch.ops.aten.relu.default(add_327);  add_327 = None
        convolution_141 = torch.ops.aten.convolution.default(relu_141, arg155_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_141 = arg155_1 = None
        add_328 = torch.ops.aten.add.Tensor(arg157_1, 0.001);  arg157_1 = None
        sqrt_142 = torch.ops.aten.sqrt.default(add_328);  add_328 = None
        reciprocal_142 = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_426 = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_1137 = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138 = torch.ops.aten.unsqueeze.default(mul_426, -1);  mul_426 = None
        unsqueeze_1139 = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_142 = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1137);  convolution_141 = unsqueeze_1137 = None
        mul_427 = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140 = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_1141 = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_1141);  mul_427 = unsqueeze_1141 = None
        unsqueeze_1142 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_1143 = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_329 = torch.ops.aten.add.Tensor(mul_428, unsqueeze_1143);  mul_428 = unsqueeze_1143 = None
        relu_142 = torch.ops.aten.relu.default(add_329);  add_329 = None
        convolution_142 = torch.ops.aten.convolution.default(relu_142, arg160_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_142 = arg160_1 = None
        add_330 = torch.ops.aten.add.Tensor(arg162_1, 0.001);  arg162_1 = None
        sqrt_143 = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_143 = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_429 = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_1145 = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146 = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
        unsqueeze_1147 = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_143 = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1145);  convolution_142 = unsqueeze_1145 = None
        mul_430 = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_1149 = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1149);  mul_430 = unsqueeze_1149 = None
        unsqueeze_1150 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_1151 = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_331 = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1151);  mul_431 = unsqueeze_1151 = None
        relu_143 = torch.ops.aten.relu.default(add_331);  add_331 = None
        convolution_143 = torch.ops.aten.convolution.default(relu_143, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_143 = arg165_1 = None
        slice_402 = torch.ops.aten.slice.Tensor(convolution_143, 1, 0, 512)
        slice_406 = torch.ops.aten.slice.Tensor(convolution_143, 1, 512, 9223372036854775807);  convolution_143 = None
        add_332 = torch.ops.aten.add.Tensor(add_325, slice_402);  add_325 = slice_402 = None
        cat_88 = torch.ops.aten.cat.default([cat_86, slice_406], 1);  cat_86 = slice_406 = None
        cat_89 = torch.ops.aten.cat.default([add_332, cat_88], 1)
        add_333 = torch.ops.aten.add.Tensor(arg167_1, 0.001);  arg167_1 = None
        sqrt_144 = torch.ops.aten.sqrt.default(add_333);  add_333 = None
        reciprocal_144 = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_432 = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_1153 = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154 = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_1155 = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_144 = torch.ops.aten.sub.Tensor(cat_89, unsqueeze_1153);  cat_89 = unsqueeze_1153 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_1157 = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_434 = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1157);  mul_433 = unsqueeze_1157 = None
        unsqueeze_1158 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_1159 = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_334 = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1159);  mul_434 = unsqueeze_1159 = None
        relu_144 = torch.ops.aten.relu.default(add_334);  add_334 = None
        convolution_144 = torch.ops.aten.convolution.default(relu_144, arg170_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_144 = arg170_1 = None
        add_335 = torch.ops.aten.add.Tensor(arg172_1, 0.001);  arg172_1 = None
        sqrt_145 = torch.ops.aten.sqrt.default(add_335);  add_335 = None
        reciprocal_145 = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_435 = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160 = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_1161 = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162 = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_1163 = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_145 = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1161);  convolution_144 = unsqueeze_1161 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_1165 = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1165);  mul_436 = unsqueeze_1165 = None
        unsqueeze_1166 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_1167 = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_336 = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1167);  mul_437 = unsqueeze_1167 = None
        relu_145 = torch.ops.aten.relu.default(add_336);  add_336 = None
        convolution_145 = torch.ops.aten.convolution.default(relu_145, arg175_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_145 = arg175_1 = None
        add_337 = torch.ops.aten.add.Tensor(arg177_1, 0.001);  arg177_1 = None
        sqrt_146 = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_146 = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168 = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_1169 = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170 = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_1171 = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_146 = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1169);  convolution_145 = unsqueeze_1169 = None
        mul_439 = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172 = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_1173 = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1173);  mul_439 = unsqueeze_1173 = None
        unsqueeze_1174 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_1175 = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_338 = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1175);  mul_440 = unsqueeze_1175 = None
        relu_146 = torch.ops.aten.relu.default(add_338);  add_338 = None
        convolution_146 = torch.ops.aten.convolution.default(relu_146, arg180_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_146 = arg180_1 = None
        slice_410 = torch.ops.aten.slice.Tensor(convolution_146, 1, 0, 512)
        slice_414 = torch.ops.aten.slice.Tensor(convolution_146, 1, 512, 9223372036854775807);  convolution_146 = None
        add_339 = torch.ops.aten.add.Tensor(add_332, slice_410);  add_332 = slice_410 = None
        cat_90 = torch.ops.aten.cat.default([cat_88, slice_414], 1);  cat_88 = slice_414 = None
        cat_91 = torch.ops.aten.cat.default([add_339, cat_90], 1)
        add_340 = torch.ops.aten.add.Tensor(arg182_1, 0.001);  arg182_1 = None
        sqrt_147 = torch.ops.aten.sqrt.default(add_340);  add_340 = None
        reciprocal_147 = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_441 = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176 = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_1177 = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178 = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_1179 = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_147 = torch.ops.aten.sub.Tensor(cat_91, unsqueeze_1177);  cat_91 = unsqueeze_1177 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_1181 = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1181);  mul_442 = unsqueeze_1181 = None
        unsqueeze_1182 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_1183 = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_341 = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1183);  mul_443 = unsqueeze_1183 = None
        relu_147 = torch.ops.aten.relu.default(add_341);  add_341 = None
        convolution_147 = torch.ops.aten.convolution.default(relu_147, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_147 = arg185_1 = None
        add_342 = torch.ops.aten.add.Tensor(arg187_1, 0.001);  arg187_1 = None
        sqrt_148 = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_148 = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_444 = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184 = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_1185 = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186 = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_1187 = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_148 = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_1185);  convolution_147 = unsqueeze_1185 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188 = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_1189 = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1189);  mul_445 = unsqueeze_1189 = None
        unsqueeze_1190 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_1191 = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_343 = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1191);  mul_446 = unsqueeze_1191 = None
        relu_148 = torch.ops.aten.relu.default(add_343);  add_343 = None
        convolution_148 = torch.ops.aten.convolution.default(relu_148, arg190_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_148 = arg190_1 = None
        add_344 = torch.ops.aten.add.Tensor(arg192_1, 0.001);  arg192_1 = None
        sqrt_149 = torch.ops.aten.sqrt.default(add_344);  add_344 = None
        reciprocal_149 = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_447 = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192 = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_1193 = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194 = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1195 = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_149 = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_1193);  convolution_148 = unsqueeze_1193 = None
        mul_448 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196 = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_1197 = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_449 = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1197);  mul_448 = unsqueeze_1197 = None
        unsqueeze_1198 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_1199 = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_345 = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1199);  mul_449 = unsqueeze_1199 = None
        relu_149 = torch.ops.aten.relu.default(add_345);  add_345 = None
        convolution_149 = torch.ops.aten.convolution.default(relu_149, arg195_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_149 = arg195_1 = None
        slice_418 = torch.ops.aten.slice.Tensor(convolution_149, 1, 0, 512)
        slice_422 = torch.ops.aten.slice.Tensor(convolution_149, 1, 512, 9223372036854775807);  convolution_149 = None
        add_346 = torch.ops.aten.add.Tensor(add_339, slice_418);  add_339 = slice_418 = None
        cat_92 = torch.ops.aten.cat.default([cat_90, slice_422], 1);  cat_90 = slice_422 = None
        cat_93 = torch.ops.aten.cat.default([add_346, cat_92], 1);  add_346 = cat_92 = None
        add_347 = torch.ops.aten.add.Tensor(arg197_1, 0.001);  arg197_1 = None
        sqrt_150 = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_150 = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_450 = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200 = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_1201 = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202 = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1203 = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_150 = torch.ops.aten.sub.Tensor(cat_93, unsqueeze_1201);  unsqueeze_1201 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204 = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_1205 = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1205);  mul_451 = unsqueeze_1205 = None
        unsqueeze_1206 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_1207 = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_348 = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1207);  mul_452 = unsqueeze_1207 = None
        relu_150 = torch.ops.aten.relu.default(add_348);  add_348 = None
        convolution_150 = torch.ops.aten.convolution.default(relu_150, arg200_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_150 = arg200_1 = None
        slice_426 = torch.ops.aten.slice.Tensor(convolution_150, 1, 0, 1024)
        slice_430 = torch.ops.aten.slice.Tensor(convolution_150, 1, 1024, 9223372036854775807);  convolution_150 = None
        add_349 = torch.ops.aten.add.Tensor(arg202_1, 0.001);  arg202_1 = None
        sqrt_151 = torch.ops.aten.sqrt.default(add_349);  add_349 = None
        reciprocal_151 = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_453 = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208 = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_1209 = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210 = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1211 = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_151 = torch.ops.aten.sub.Tensor(cat_93, unsqueeze_1209);  cat_93 = unsqueeze_1209 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212 = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_1213 = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1213);  mul_454 = unsqueeze_1213 = None
        unsqueeze_1214 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_1215 = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_350 = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1215);  mul_455 = unsqueeze_1215 = None
        relu_151 = torch.ops.aten.relu.default(add_350);  add_350 = None
        convolution_151 = torch.ops.aten.convolution.default(relu_151, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_151 = arg205_1 = None
        add_351 = torch.ops.aten.add.Tensor(arg207_1, 0.001);  arg207_1 = None
        sqrt_152 = torch.ops.aten.sqrt.default(add_351);  add_351 = None
        reciprocal_152 = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_456 = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216 = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_1217 = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218 = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1219 = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_152 = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1217);  convolution_151 = unsqueeze_1217 = None
        mul_457 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_1221 = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1221);  mul_457 = unsqueeze_1221 = None
        unsqueeze_1222 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_1223 = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_352 = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1223);  mul_458 = unsqueeze_1223 = None
        relu_152 = torch.ops.aten.relu.default(add_352);  add_352 = None
        convolution_152 = torch.ops.aten.convolution.default(relu_152, arg210_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_152 = arg210_1 = None
        add_353 = torch.ops.aten.add.Tensor(arg212_1, 0.001);  arg212_1 = None
        sqrt_153 = torch.ops.aten.sqrt.default(add_353);  add_353 = None
        reciprocal_153 = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_459 = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_1225 = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226 = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1227 = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_153 = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1225);  convolution_152 = unsqueeze_1225 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_1229 = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1229);  mul_460 = unsqueeze_1229 = None
        unsqueeze_1230 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1231 = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_354 = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1231);  mul_461 = unsqueeze_1231 = None
        relu_153 = torch.ops.aten.relu.default(add_354);  add_354 = None
        convolution_153 = torch.ops.aten.convolution.default(relu_153, arg215_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_153 = arg215_1 = None
        slice_434 = torch.ops.aten.slice.Tensor(convolution_153, 1, 0, 1024)
        slice_438 = torch.ops.aten.slice.Tensor(convolution_153, 1, 1024, 9223372036854775807);  convolution_153 = None
        add_355 = torch.ops.aten.add.Tensor(slice_426, slice_434);  slice_426 = slice_434 = None
        cat_94 = torch.ops.aten.cat.default([slice_430, slice_438], 1);  slice_430 = slice_438 = None
        cat_95 = torch.ops.aten.cat.default([add_355, cat_94], 1)
        add_356 = torch.ops.aten.add.Tensor(arg217_1, 0.001);  arg217_1 = None
        sqrt_154 = torch.ops.aten.sqrt.default(add_356);  add_356 = None
        reciprocal_154 = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_462 = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_1233 = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234 = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1235 = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_154 = torch.ops.aten.sub.Tensor(cat_95, unsqueeze_1233);  cat_95 = unsqueeze_1233 = None
        mul_463 = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_1237 = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_464 = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1237);  mul_463 = unsqueeze_1237 = None
        unsqueeze_1238 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1239 = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_357 = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1239);  mul_464 = unsqueeze_1239 = None
        relu_154 = torch.ops.aten.relu.default(add_357);  add_357 = None
        convolution_154 = torch.ops.aten.convolution.default(relu_154, arg220_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_154 = arg220_1 = None
        add_358 = torch.ops.aten.add.Tensor(arg222_1, 0.001);  arg222_1 = None
        sqrt_155 = torch.ops.aten.sqrt.default(add_358);  add_358 = None
        reciprocal_155 = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_465 = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_1241 = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242 = torch.ops.aten.unsqueeze.default(mul_465, -1);  mul_465 = None
        unsqueeze_1243 = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_155 = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1241);  convolution_154 = unsqueeze_1241 = None
        mul_466 = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244 = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_1245 = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_1245);  mul_466 = unsqueeze_1245 = None
        unsqueeze_1246 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1247 = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_359 = torch.ops.aten.add.Tensor(mul_467, unsqueeze_1247);  mul_467 = unsqueeze_1247 = None
        relu_155 = torch.ops.aten.relu.default(add_359);  add_359 = None
        convolution_155 = torch.ops.aten.convolution.default(relu_155, arg225_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_155 = arg225_1 = None
        add_360 = torch.ops.aten.add.Tensor(arg227_1, 0.001);  arg227_1 = None
        sqrt_156 = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_156 = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_468 = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248 = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_1249 = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250 = torch.ops.aten.unsqueeze.default(mul_468, -1);  mul_468 = None
        unsqueeze_1251 = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_156 = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_1249);  convolution_155 = unsqueeze_1249 = None
        mul_469 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252 = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_1253 = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_1253);  mul_469 = unsqueeze_1253 = None
        unsqueeze_1254 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1255 = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_361 = torch.ops.aten.add.Tensor(mul_470, unsqueeze_1255);  mul_470 = unsqueeze_1255 = None
        relu_156 = torch.ops.aten.relu.default(add_361);  add_361 = None
        convolution_156 = torch.ops.aten.convolution.default(relu_156, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_156 = arg230_1 = None
        slice_442 = torch.ops.aten.slice.Tensor(convolution_156, 1, 0, 1024)
        slice_446 = torch.ops.aten.slice.Tensor(convolution_156, 1, 1024, 9223372036854775807);  convolution_156 = None
        add_362 = torch.ops.aten.add.Tensor(add_355, slice_442);  add_355 = slice_442 = None
        cat_96 = torch.ops.aten.cat.default([cat_94, slice_446], 1);  cat_94 = slice_446 = None
        cat_97 = torch.ops.aten.cat.default([add_362, cat_96], 1)
        add_363 = torch.ops.aten.add.Tensor(arg232_1, 0.001);  arg232_1 = None
        sqrt_157 = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_157 = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_471 = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256 = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_1257 = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258 = torch.ops.aten.unsqueeze.default(mul_471, -1);  mul_471 = None
        unsqueeze_1259 = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_157 = torch.ops.aten.sub.Tensor(cat_97, unsqueeze_1257);  cat_97 = unsqueeze_1257 = None
        mul_472 = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260 = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_1261 = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_473 = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_1261);  mul_472 = unsqueeze_1261 = None
        unsqueeze_1262 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1263 = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_364 = torch.ops.aten.add.Tensor(mul_473, unsqueeze_1263);  mul_473 = unsqueeze_1263 = None
        relu_157 = torch.ops.aten.relu.default(add_364);  add_364 = None
        convolution_157 = torch.ops.aten.convolution.default(relu_157, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_157 = arg235_1 = None
        add_365 = torch.ops.aten.add.Tensor(arg237_1, 0.001);  arg237_1 = None
        sqrt_158 = torch.ops.aten.sqrt.default(add_365);  add_365 = None
        reciprocal_158 = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_474 = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264 = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_1265 = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266 = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
        unsqueeze_1267 = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_158 = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1265);  convolution_157 = unsqueeze_1265 = None
        mul_475 = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268 = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_1269 = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_476 = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_1269);  mul_475 = unsqueeze_1269 = None
        unsqueeze_1270 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1271 = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_366 = torch.ops.aten.add.Tensor(mul_476, unsqueeze_1271);  mul_476 = unsqueeze_1271 = None
        relu_158 = torch.ops.aten.relu.default(add_366);  add_366 = None
        convolution_158 = torch.ops.aten.convolution.default(relu_158, arg240_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_158 = arg240_1 = None
        add_367 = torch.ops.aten.add.Tensor(arg242_1, 0.001);  arg242_1 = None
        sqrt_159 = torch.ops.aten.sqrt.default(add_367);  add_367 = None
        reciprocal_159 = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_477 = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272 = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_1273 = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274 = torch.ops.aten.unsqueeze.default(mul_477, -1);  mul_477 = None
        unsqueeze_1275 = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_159 = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1273);  convolution_158 = unsqueeze_1273 = None
        mul_478 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276 = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_1277 = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_1277);  mul_478 = unsqueeze_1277 = None
        unsqueeze_1278 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1279 = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_368 = torch.ops.aten.add.Tensor(mul_479, unsqueeze_1279);  mul_479 = unsqueeze_1279 = None
        relu_159 = torch.ops.aten.relu.default(add_368);  add_368 = None
        convolution_159 = torch.ops.aten.convolution.default(relu_159, arg245_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_159 = arg245_1 = None
        slice_450 = torch.ops.aten.slice.Tensor(convolution_159, 1, 0, 1024)
        slice_454 = torch.ops.aten.slice.Tensor(convolution_159, 1, 1024, 9223372036854775807);  convolution_159 = None
        add_369 = torch.ops.aten.add.Tensor(add_362, slice_450);  add_362 = slice_450 = None
        cat_98 = torch.ops.aten.cat.default([cat_96, slice_454], 1);  cat_96 = slice_454 = None
        cat_99 = torch.ops.aten.cat.default([add_369, cat_98], 1)
        add_370 = torch.ops.aten.add.Tensor(arg247_1, 0.001);  arg247_1 = None
        sqrt_160 = torch.ops.aten.sqrt.default(add_370);  add_370 = None
        reciprocal_160 = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_480 = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1280 = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_1281 = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        unsqueeze_1282 = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_1283 = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        sub_160 = torch.ops.aten.sub.Tensor(cat_99, unsqueeze_1281);  cat_99 = unsqueeze_1281 = None
        mul_481 = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
        unsqueeze_1284 = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_1285 = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_482 = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_1285);  mul_481 = unsqueeze_1285 = None
        unsqueeze_1286 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1287 = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_371 = torch.ops.aten.add.Tensor(mul_482, unsqueeze_1287);  mul_482 = unsqueeze_1287 = None
        relu_160 = torch.ops.aten.relu.default(add_371);  add_371 = None
        convolution_160 = torch.ops.aten.convolution.default(relu_160, arg250_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_160 = arg250_1 = None
        add_372 = torch.ops.aten.add.Tensor(arg252_1, 0.001);  arg252_1 = None
        sqrt_161 = torch.ops.aten.sqrt.default(add_372);  add_372 = None
        reciprocal_161 = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_483 = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1288 = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_1289 = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        unsqueeze_1290 = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_1291 = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        sub_161 = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1289);  convolution_160 = unsqueeze_1289 = None
        mul_484 = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
        unsqueeze_1292 = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_1293 = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_485 = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_1293);  mul_484 = unsqueeze_1293 = None
        unsqueeze_1294 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1295 = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_373 = torch.ops.aten.add.Tensor(mul_485, unsqueeze_1295);  mul_485 = unsqueeze_1295 = None
        relu_161 = torch.ops.aten.relu.default(add_373);  add_373 = None
        convolution_161 = torch.ops.aten.convolution.default(relu_161, arg255_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_161 = arg255_1 = None
        add_374 = torch.ops.aten.add.Tensor(arg257_1, 0.001);  arg257_1 = None
        sqrt_162 = torch.ops.aten.sqrt.default(add_374);  add_374 = None
        reciprocal_162 = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_486 = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1296 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_1297 = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        unsqueeze_1298 = torch.ops.aten.unsqueeze.default(mul_486, -1);  mul_486 = None
        unsqueeze_1299 = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        sub_162 = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1297);  convolution_161 = unsqueeze_1297 = None
        mul_487 = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
        unsqueeze_1300 = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_1301 = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_488 = torch.ops.aten.mul.Tensor(mul_487, unsqueeze_1301);  mul_487 = unsqueeze_1301 = None
        unsqueeze_1302 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1303 = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_375 = torch.ops.aten.add.Tensor(mul_488, unsqueeze_1303);  mul_488 = unsqueeze_1303 = None
        relu_162 = torch.ops.aten.relu.default(add_375);  add_375 = None
        convolution_162 = torch.ops.aten.convolution.default(relu_162, arg260_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_162 = arg260_1 = None
        slice_458 = torch.ops.aten.slice.Tensor(convolution_162, 1, 0, 1024)
        slice_462 = torch.ops.aten.slice.Tensor(convolution_162, 1, 1024, 9223372036854775807);  convolution_162 = None
        add_376 = torch.ops.aten.add.Tensor(add_369, slice_458);  add_369 = slice_458 = None
        cat_100 = torch.ops.aten.cat.default([cat_98, slice_462], 1);  cat_98 = slice_462 = None
        cat_101 = torch.ops.aten.cat.default([add_376, cat_100], 1)
        add_377 = torch.ops.aten.add.Tensor(arg262_1, 0.001);  arg262_1 = None
        sqrt_163 = torch.ops.aten.sqrt.default(add_377);  add_377 = None
        reciprocal_163 = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_489 = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1304 = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_1305 = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        unsqueeze_1306 = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
        unsqueeze_1307 = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        sub_163 = torch.ops.aten.sub.Tensor(cat_101, unsqueeze_1305);  cat_101 = unsqueeze_1305 = None
        mul_490 = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
        unsqueeze_1308 = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_1309 = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_1309);  mul_490 = unsqueeze_1309 = None
        unsqueeze_1310 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1311 = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_378 = torch.ops.aten.add.Tensor(mul_491, unsqueeze_1311);  mul_491 = unsqueeze_1311 = None
        relu_163 = torch.ops.aten.relu.default(add_378);  add_378 = None
        convolution_163 = torch.ops.aten.convolution.default(relu_163, arg265_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_163 = arg265_1 = None
        add_379 = torch.ops.aten.add.Tensor(arg267_1, 0.001);  arg267_1 = None
        sqrt_164 = torch.ops.aten.sqrt.default(add_379);  add_379 = None
        reciprocal_164 = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_492 = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1312 = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_1313 = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        unsqueeze_1314 = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
        unsqueeze_1315 = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        sub_164 = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_1313);  convolution_163 = unsqueeze_1313 = None
        mul_493 = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
        unsqueeze_1316 = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_1317 = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_494 = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_1317);  mul_493 = unsqueeze_1317 = None
        unsqueeze_1318 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1319 = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_380 = torch.ops.aten.add.Tensor(mul_494, unsqueeze_1319);  mul_494 = unsqueeze_1319 = None
        relu_164 = torch.ops.aten.relu.default(add_380);  add_380 = None
        convolution_164 = torch.ops.aten.convolution.default(relu_164, arg270_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_164 = arg270_1 = None
        add_381 = torch.ops.aten.add.Tensor(arg272_1, 0.001);  arg272_1 = None
        sqrt_165 = torch.ops.aten.sqrt.default(add_381);  add_381 = None
        reciprocal_165 = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_495 = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1320 = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_1321 = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        unsqueeze_1322 = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_1323 = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        sub_165 = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1321);  convolution_164 = unsqueeze_1321 = None
        mul_496 = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
        unsqueeze_1324 = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1325 = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_497 = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_1325);  mul_496 = unsqueeze_1325 = None
        unsqueeze_1326 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1327 = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_382 = torch.ops.aten.add.Tensor(mul_497, unsqueeze_1327);  mul_497 = unsqueeze_1327 = None
        relu_165 = torch.ops.aten.relu.default(add_382);  add_382 = None
        convolution_165 = torch.ops.aten.convolution.default(relu_165, arg275_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_165 = arg275_1 = None
        slice_466 = torch.ops.aten.slice.Tensor(convolution_165, 1, 0, 1024)
        slice_470 = torch.ops.aten.slice.Tensor(convolution_165, 1, 1024, 9223372036854775807);  convolution_165 = None
        add_383 = torch.ops.aten.add.Tensor(add_376, slice_466);  add_376 = slice_466 = None
        cat_102 = torch.ops.aten.cat.default([cat_100, slice_470], 1);  cat_100 = slice_470 = None
        cat_103 = torch.ops.aten.cat.default([add_383, cat_102], 1)
        add_384 = torch.ops.aten.add.Tensor(arg277_1, 0.001);  arg277_1 = None
        sqrt_166 = torch.ops.aten.sqrt.default(add_384);  add_384 = None
        reciprocal_166 = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_498 = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1328 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1329 = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        unsqueeze_1330 = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_1331 = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        sub_166 = torch.ops.aten.sub.Tensor(cat_103, unsqueeze_1329);  cat_103 = unsqueeze_1329 = None
        mul_499 = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
        unsqueeze_1332 = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_1333 = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_1333);  mul_499 = unsqueeze_1333 = None
        unsqueeze_1334 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1335 = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_385 = torch.ops.aten.add.Tensor(mul_500, unsqueeze_1335);  mul_500 = unsqueeze_1335 = None
        relu_166 = torch.ops.aten.relu.default(add_385);  add_385 = None
        convolution_166 = torch.ops.aten.convolution.default(relu_166, arg280_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_166 = arg280_1 = None
        add_386 = torch.ops.aten.add.Tensor(arg282_1, 0.001);  arg282_1 = None
        sqrt_167 = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_167 = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_501 = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1336 = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_1337 = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        unsqueeze_1338 = torch.ops.aten.unsqueeze.default(mul_501, -1);  mul_501 = None
        unsqueeze_1339 = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        sub_167 = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1337);  convolution_166 = unsqueeze_1337 = None
        mul_502 = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
        unsqueeze_1340 = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_1341 = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_503 = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_1341);  mul_502 = unsqueeze_1341 = None
        unsqueeze_1342 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1343 = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_387 = torch.ops.aten.add.Tensor(mul_503, unsqueeze_1343);  mul_503 = unsqueeze_1343 = None
        relu_167 = torch.ops.aten.relu.default(add_387);  add_387 = None
        convolution_167 = torch.ops.aten.convolution.default(relu_167, arg285_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_167 = arg285_1 = None
        add_388 = torch.ops.aten.add.Tensor(arg287_1, 0.001);  arg287_1 = None
        sqrt_168 = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_168 = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_504 = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1344 = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
        unsqueeze_1345 = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        unsqueeze_1346 = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1347 = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        sub_168 = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1345);  convolution_167 = unsqueeze_1345 = None
        mul_505 = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
        unsqueeze_1348 = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_1349 = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_506 = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1349);  mul_505 = unsqueeze_1349 = None
        unsqueeze_1350 = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_1351 = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_389 = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1351);  mul_506 = unsqueeze_1351 = None
        relu_168 = torch.ops.aten.relu.default(add_389);  add_389 = None
        convolution_168 = torch.ops.aten.convolution.default(relu_168, arg290_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_168 = arg290_1 = None
        slice_474 = torch.ops.aten.slice.Tensor(convolution_168, 1, 0, 1024)
        slice_478 = torch.ops.aten.slice.Tensor(convolution_168, 1, 1024, 9223372036854775807);  convolution_168 = None
        add_390 = torch.ops.aten.add.Tensor(add_383, slice_474);  add_383 = slice_474 = None
        cat_104 = torch.ops.aten.cat.default([cat_102, slice_478], 1);  cat_102 = slice_478 = None
        cat_105 = torch.ops.aten.cat.default([add_390, cat_104], 1)
        add_391 = torch.ops.aten.add.Tensor(arg292_1, 0.001);  arg292_1 = None
        sqrt_169 = torch.ops.aten.sqrt.default(add_391);  add_391 = None
        reciprocal_169 = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_507 = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1352 = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_1353 = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        unsqueeze_1354 = torch.ops.aten.unsqueeze.default(mul_507, -1);  mul_507 = None
        unsqueeze_1355 = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        sub_169 = torch.ops.aten.sub.Tensor(cat_105, unsqueeze_1353);  cat_105 = unsqueeze_1353 = None
        mul_508 = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
        unsqueeze_1356 = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1357 = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_1357);  mul_508 = unsqueeze_1357 = None
        unsqueeze_1358 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_1359 = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_392 = torch.ops.aten.add.Tensor(mul_509, unsqueeze_1359);  mul_509 = unsqueeze_1359 = None
        relu_169 = torch.ops.aten.relu.default(add_392);  add_392 = None
        convolution_169 = torch.ops.aten.convolution.default(relu_169, arg295_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_169 = arg295_1 = None
        add_393 = torch.ops.aten.add.Tensor(arg297_1, 0.001);  arg297_1 = None
        sqrt_170 = torch.ops.aten.sqrt.default(add_393);  add_393 = None
        reciprocal_170 = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_510 = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1360 = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_1361 = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        unsqueeze_1362 = torch.ops.aten.unsqueeze.default(mul_510, -1);  mul_510 = None
        unsqueeze_1363 = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        sub_170 = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1361);  convolution_169 = unsqueeze_1361 = None
        mul_511 = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_1363);  sub_170 = unsqueeze_1363 = None
        unsqueeze_1364 = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1365 = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_512 = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_1365);  mul_511 = unsqueeze_1365 = None
        unsqueeze_1366 = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1367 = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_394 = torch.ops.aten.add.Tensor(mul_512, unsqueeze_1367);  mul_512 = unsqueeze_1367 = None
        relu_170 = torch.ops.aten.relu.default(add_394);  add_394 = None
        convolution_170 = torch.ops.aten.convolution.default(relu_170, arg300_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_170 = arg300_1 = None
        add_395 = torch.ops.aten.add.Tensor(arg302_1, 0.001);  arg302_1 = None
        sqrt_171 = torch.ops.aten.sqrt.default(add_395);  add_395 = None
        reciprocal_171 = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_513 = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1368 = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_1369 = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        unsqueeze_1370 = torch.ops.aten.unsqueeze.default(mul_513, -1);  mul_513 = None
        unsqueeze_1371 = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        sub_171 = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_1369);  convolution_170 = unsqueeze_1369 = None
        mul_514 = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_1371);  sub_171 = unsqueeze_1371 = None
        unsqueeze_1372 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1373 = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_515 = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_1373);  mul_514 = unsqueeze_1373 = None
        unsqueeze_1374 = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_1375 = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_396 = torch.ops.aten.add.Tensor(mul_515, unsqueeze_1375);  mul_515 = unsqueeze_1375 = None
        relu_171 = torch.ops.aten.relu.default(add_396);  add_396 = None
        convolution_171 = torch.ops.aten.convolution.default(relu_171, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_171 = arg305_1 = None
        slice_482 = torch.ops.aten.slice.Tensor(convolution_171, 1, 0, 1024)
        slice_486 = torch.ops.aten.slice.Tensor(convolution_171, 1, 1024, 9223372036854775807);  convolution_171 = None
        add_397 = torch.ops.aten.add.Tensor(add_390, slice_482);  add_390 = slice_482 = None
        cat_106 = torch.ops.aten.cat.default([cat_104, slice_486], 1);  cat_104 = slice_486 = None
        cat_107 = torch.ops.aten.cat.default([add_397, cat_106], 1)
        add_398 = torch.ops.aten.add.Tensor(arg307_1, 0.001);  arg307_1 = None
        sqrt_172 = torch.ops.aten.sqrt.default(add_398);  add_398 = None
        reciprocal_172 = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_516 = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1376 = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_1377 = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        unsqueeze_1378 = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1379 = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        sub_172 = torch.ops.aten.sub.Tensor(cat_107, unsqueeze_1377);  cat_107 = unsqueeze_1377 = None
        mul_517 = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1379);  sub_172 = unsqueeze_1379 = None
        unsqueeze_1380 = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_1381 = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_518 = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1381);  mul_517 = unsqueeze_1381 = None
        unsqueeze_1382 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1383 = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_399 = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1383);  mul_518 = unsqueeze_1383 = None
        relu_172 = torch.ops.aten.relu.default(add_399);  add_399 = None
        convolution_172 = torch.ops.aten.convolution.default(relu_172, arg310_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_172 = arg310_1 = None
        add_400 = torch.ops.aten.add.Tensor(arg312_1, 0.001);  arg312_1 = None
        sqrt_173 = torch.ops.aten.sqrt.default(add_400);  add_400 = None
        reciprocal_173 = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_519 = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1384 = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1385 = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        unsqueeze_1386 = torch.ops.aten.unsqueeze.default(mul_519, -1);  mul_519 = None
        unsqueeze_1387 = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        sub_173 = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1385);  convolution_172 = unsqueeze_1385 = None
        mul_520 = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1387);  sub_173 = unsqueeze_1387 = None
        unsqueeze_1388 = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_1389 = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_521 = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_1389);  mul_520 = unsqueeze_1389 = None
        unsqueeze_1390 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1391 = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_401 = torch.ops.aten.add.Tensor(mul_521, unsqueeze_1391);  mul_521 = unsqueeze_1391 = None
        relu_173 = torch.ops.aten.relu.default(add_401);  add_401 = None
        convolution_173 = torch.ops.aten.convolution.default(relu_173, arg315_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_173 = arg315_1 = None
        add_402 = torch.ops.aten.add.Tensor(arg317_1, 0.001);  arg317_1 = None
        sqrt_174 = torch.ops.aten.sqrt.default(add_402);  add_402 = None
        reciprocal_174 = torch.ops.aten.reciprocal.default(sqrt_174);  sqrt_174 = None
        mul_522 = torch.ops.aten.mul.Tensor(reciprocal_174, 1);  reciprocal_174 = None
        unsqueeze_1392 = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
        unsqueeze_1393 = torch.ops.aten.unsqueeze.default(unsqueeze_1392, -1);  unsqueeze_1392 = None
        unsqueeze_1394 = torch.ops.aten.unsqueeze.default(mul_522, -1);  mul_522 = None
        unsqueeze_1395 = torch.ops.aten.unsqueeze.default(unsqueeze_1394, -1);  unsqueeze_1394 = None
        sub_174 = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1393);  convolution_173 = unsqueeze_1393 = None
        mul_523 = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_1395);  sub_174 = unsqueeze_1395 = None
        unsqueeze_1396 = torch.ops.aten.unsqueeze.default(arg318_1, -1);  arg318_1 = None
        unsqueeze_1397 = torch.ops.aten.unsqueeze.default(unsqueeze_1396, -1);  unsqueeze_1396 = None
        mul_524 = torch.ops.aten.mul.Tensor(mul_523, unsqueeze_1397);  mul_523 = unsqueeze_1397 = None
        unsqueeze_1398 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1399 = torch.ops.aten.unsqueeze.default(unsqueeze_1398, -1);  unsqueeze_1398 = None
        add_403 = torch.ops.aten.add.Tensor(mul_524, unsqueeze_1399);  mul_524 = unsqueeze_1399 = None
        relu_174 = torch.ops.aten.relu.default(add_403);  add_403 = None
        convolution_174 = torch.ops.aten.convolution.default(relu_174, arg320_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_174 = arg320_1 = None
        slice_490 = torch.ops.aten.slice.Tensor(convolution_174, 1, 0, 1024)
        slice_494 = torch.ops.aten.slice.Tensor(convolution_174, 1, 1024, 9223372036854775807);  convolution_174 = None
        add_404 = torch.ops.aten.add.Tensor(add_397, slice_490);  add_397 = slice_490 = None
        cat_108 = torch.ops.aten.cat.default([cat_106, slice_494], 1);  cat_106 = slice_494 = None
        cat_109 = torch.ops.aten.cat.default([add_404, cat_108], 1)
        add_405 = torch.ops.aten.add.Tensor(arg322_1, 0.001);  arg322_1 = None
        sqrt_175 = torch.ops.aten.sqrt.default(add_405);  add_405 = None
        reciprocal_175 = torch.ops.aten.reciprocal.default(sqrt_175);  sqrt_175 = None
        mul_525 = torch.ops.aten.mul.Tensor(reciprocal_175, 1);  reciprocal_175 = None
        unsqueeze_1400 = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_1401 = torch.ops.aten.unsqueeze.default(unsqueeze_1400, -1);  unsqueeze_1400 = None
        unsqueeze_1402 = torch.ops.aten.unsqueeze.default(mul_525, -1);  mul_525 = None
        unsqueeze_1403 = torch.ops.aten.unsqueeze.default(unsqueeze_1402, -1);  unsqueeze_1402 = None
        sub_175 = torch.ops.aten.sub.Tensor(cat_109, unsqueeze_1401);  cat_109 = unsqueeze_1401 = None
        mul_526 = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_1403);  sub_175 = unsqueeze_1403 = None
        unsqueeze_1404 = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_1405 = torch.ops.aten.unsqueeze.default(unsqueeze_1404, -1);  unsqueeze_1404 = None
        mul_527 = torch.ops.aten.mul.Tensor(mul_526, unsqueeze_1405);  mul_526 = unsqueeze_1405 = None
        unsqueeze_1406 = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1407 = torch.ops.aten.unsqueeze.default(unsqueeze_1406, -1);  unsqueeze_1406 = None
        add_406 = torch.ops.aten.add.Tensor(mul_527, unsqueeze_1407);  mul_527 = unsqueeze_1407 = None
        relu_175 = torch.ops.aten.relu.default(add_406);  add_406 = None
        convolution_175 = torch.ops.aten.convolution.default(relu_175, arg325_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_175 = arg325_1 = None
        add_407 = torch.ops.aten.add.Tensor(arg327_1, 0.001);  arg327_1 = None
        sqrt_176 = torch.ops.aten.sqrt.default(add_407);  add_407 = None
        reciprocal_176 = torch.ops.aten.reciprocal.default(sqrt_176);  sqrt_176 = None
        mul_528 = torch.ops.aten.mul.Tensor(reciprocal_176, 1);  reciprocal_176 = None
        unsqueeze_1408 = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_1409 = torch.ops.aten.unsqueeze.default(unsqueeze_1408, -1);  unsqueeze_1408 = None
        unsqueeze_1410 = torch.ops.aten.unsqueeze.default(mul_528, -1);  mul_528 = None
        unsqueeze_1411 = torch.ops.aten.unsqueeze.default(unsqueeze_1410, -1);  unsqueeze_1410 = None
        sub_176 = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1409);  convolution_175 = unsqueeze_1409 = None
        mul_529 = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_1411);  sub_176 = unsqueeze_1411 = None
        unsqueeze_1412 = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_1413 = torch.ops.aten.unsqueeze.default(unsqueeze_1412, -1);  unsqueeze_1412 = None
        mul_530 = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_1413);  mul_529 = unsqueeze_1413 = None
        unsqueeze_1414 = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1415 = torch.ops.aten.unsqueeze.default(unsqueeze_1414, -1);  unsqueeze_1414 = None
        add_408 = torch.ops.aten.add.Tensor(mul_530, unsqueeze_1415);  mul_530 = unsqueeze_1415 = None
        relu_176 = torch.ops.aten.relu.default(add_408);  add_408 = None
        convolution_176 = torch.ops.aten.convolution.default(relu_176, arg330_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_176 = arg330_1 = None
        add_409 = torch.ops.aten.add.Tensor(arg332_1, 0.001);  arg332_1 = None
        sqrt_177 = torch.ops.aten.sqrt.default(add_409);  add_409 = None
        reciprocal_177 = torch.ops.aten.reciprocal.default(sqrt_177);  sqrt_177 = None
        mul_531 = torch.ops.aten.mul.Tensor(reciprocal_177, 1);  reciprocal_177 = None
        unsqueeze_1416 = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_1417 = torch.ops.aten.unsqueeze.default(unsqueeze_1416, -1);  unsqueeze_1416 = None
        unsqueeze_1418 = torch.ops.aten.unsqueeze.default(mul_531, -1);  mul_531 = None
        unsqueeze_1419 = torch.ops.aten.unsqueeze.default(unsqueeze_1418, -1);  unsqueeze_1418 = None
        sub_177 = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1417);  convolution_176 = unsqueeze_1417 = None
        mul_532 = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_1419);  sub_177 = unsqueeze_1419 = None
        unsqueeze_1420 = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_1421 = torch.ops.aten.unsqueeze.default(unsqueeze_1420, -1);  unsqueeze_1420 = None
        mul_533 = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_1421);  mul_532 = unsqueeze_1421 = None
        unsqueeze_1422 = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1423 = torch.ops.aten.unsqueeze.default(unsqueeze_1422, -1);  unsqueeze_1422 = None
        add_410 = torch.ops.aten.add.Tensor(mul_533, unsqueeze_1423);  mul_533 = unsqueeze_1423 = None
        relu_177 = torch.ops.aten.relu.default(add_410);  add_410 = None
        convolution_177 = torch.ops.aten.convolution.default(relu_177, arg335_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_177 = arg335_1 = None
        slice_498 = torch.ops.aten.slice.Tensor(convolution_177, 1, 0, 1024)
        slice_502 = torch.ops.aten.slice.Tensor(convolution_177, 1, 1024, 9223372036854775807);  convolution_177 = None
        add_411 = torch.ops.aten.add.Tensor(add_404, slice_498);  add_404 = slice_498 = None
        cat_110 = torch.ops.aten.cat.default([cat_108, slice_502], 1);  cat_108 = slice_502 = None
        cat_111 = torch.ops.aten.cat.default([add_411, cat_110], 1)
        add_412 = torch.ops.aten.add.Tensor(arg337_1, 0.001);  arg337_1 = None
        sqrt_178 = torch.ops.aten.sqrt.default(add_412);  add_412 = None
        reciprocal_178 = torch.ops.aten.reciprocal.default(sqrt_178);  sqrt_178 = None
        mul_534 = torch.ops.aten.mul.Tensor(reciprocal_178, 1);  reciprocal_178 = None
        unsqueeze_1424 = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_1425 = torch.ops.aten.unsqueeze.default(unsqueeze_1424, -1);  unsqueeze_1424 = None
        unsqueeze_1426 = torch.ops.aten.unsqueeze.default(mul_534, -1);  mul_534 = None
        unsqueeze_1427 = torch.ops.aten.unsqueeze.default(unsqueeze_1426, -1);  unsqueeze_1426 = None
        sub_178 = torch.ops.aten.sub.Tensor(cat_111, unsqueeze_1425);  cat_111 = unsqueeze_1425 = None
        mul_535 = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_1427);  sub_178 = unsqueeze_1427 = None
        unsqueeze_1428 = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_1429 = torch.ops.aten.unsqueeze.default(unsqueeze_1428, -1);  unsqueeze_1428 = None
        mul_536 = torch.ops.aten.mul.Tensor(mul_535, unsqueeze_1429);  mul_535 = unsqueeze_1429 = None
        unsqueeze_1430 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1431 = torch.ops.aten.unsqueeze.default(unsqueeze_1430, -1);  unsqueeze_1430 = None
        add_413 = torch.ops.aten.add.Tensor(mul_536, unsqueeze_1431);  mul_536 = unsqueeze_1431 = None
        relu_178 = torch.ops.aten.relu.default(add_413);  add_413 = None
        convolution_178 = torch.ops.aten.convolution.default(relu_178, arg340_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_178 = arg340_1 = None
        add_414 = torch.ops.aten.add.Tensor(arg342_1, 0.001);  arg342_1 = None
        sqrt_179 = torch.ops.aten.sqrt.default(add_414);  add_414 = None
        reciprocal_179 = torch.ops.aten.reciprocal.default(sqrt_179);  sqrt_179 = None
        mul_537 = torch.ops.aten.mul.Tensor(reciprocal_179, 1);  reciprocal_179 = None
        unsqueeze_1432 = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_1433 = torch.ops.aten.unsqueeze.default(unsqueeze_1432, -1);  unsqueeze_1432 = None
        unsqueeze_1434 = torch.ops.aten.unsqueeze.default(mul_537, -1);  mul_537 = None
        unsqueeze_1435 = torch.ops.aten.unsqueeze.default(unsqueeze_1434, -1);  unsqueeze_1434 = None
        sub_179 = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1433);  convolution_178 = unsqueeze_1433 = None
        mul_538 = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_1435);  sub_179 = unsqueeze_1435 = None
        unsqueeze_1436 = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_1437 = torch.ops.aten.unsqueeze.default(unsqueeze_1436, -1);  unsqueeze_1436 = None
        mul_539 = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_1437);  mul_538 = unsqueeze_1437 = None
        unsqueeze_1438 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1439 = torch.ops.aten.unsqueeze.default(unsqueeze_1438, -1);  unsqueeze_1438 = None
        add_415 = torch.ops.aten.add.Tensor(mul_539, unsqueeze_1439);  mul_539 = unsqueeze_1439 = None
        relu_179 = torch.ops.aten.relu.default(add_415);  add_415 = None
        convolution_179 = torch.ops.aten.convolution.default(relu_179, arg345_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_179 = arg345_1 = None
        add_416 = torch.ops.aten.add.Tensor(arg347_1, 0.001);  arg347_1 = None
        sqrt_180 = torch.ops.aten.sqrt.default(add_416);  add_416 = None
        reciprocal_180 = torch.ops.aten.reciprocal.default(sqrt_180);  sqrt_180 = None
        mul_540 = torch.ops.aten.mul.Tensor(reciprocal_180, 1);  reciprocal_180 = None
        unsqueeze_1440 = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_1441 = torch.ops.aten.unsqueeze.default(unsqueeze_1440, -1);  unsqueeze_1440 = None
        unsqueeze_1442 = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
        unsqueeze_1443 = torch.ops.aten.unsqueeze.default(unsqueeze_1442, -1);  unsqueeze_1442 = None
        sub_180 = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1441);  convolution_179 = unsqueeze_1441 = None
        mul_541 = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_1443);  sub_180 = unsqueeze_1443 = None
        unsqueeze_1444 = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
        unsqueeze_1445 = torch.ops.aten.unsqueeze.default(unsqueeze_1444, -1);  unsqueeze_1444 = None
        mul_542 = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_1445);  mul_541 = unsqueeze_1445 = None
        unsqueeze_1446 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1447 = torch.ops.aten.unsqueeze.default(unsqueeze_1446, -1);  unsqueeze_1446 = None
        add_417 = torch.ops.aten.add.Tensor(mul_542, unsqueeze_1447);  mul_542 = unsqueeze_1447 = None
        relu_180 = torch.ops.aten.relu.default(add_417);  add_417 = None
        convolution_180 = torch.ops.aten.convolution.default(relu_180, arg350_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_180 = arg350_1 = None
        slice_506 = torch.ops.aten.slice.Tensor(convolution_180, 1, 0, 1024)
        slice_510 = torch.ops.aten.slice.Tensor(convolution_180, 1, 1024, 9223372036854775807);  convolution_180 = None
        add_418 = torch.ops.aten.add.Tensor(add_411, slice_506);  add_411 = slice_506 = None
        cat_112 = torch.ops.aten.cat.default([cat_110, slice_510], 1);  cat_110 = slice_510 = None
        cat_113 = torch.ops.aten.cat.default([add_418, cat_112], 1)
        add_419 = torch.ops.aten.add.Tensor(arg352_1, 0.001);  arg352_1 = None
        sqrt_181 = torch.ops.aten.sqrt.default(add_419);  add_419 = None
        reciprocal_181 = torch.ops.aten.reciprocal.default(sqrt_181);  sqrt_181 = None
        mul_543 = torch.ops.aten.mul.Tensor(reciprocal_181, 1);  reciprocal_181 = None
        unsqueeze_1448 = torch.ops.aten.unsqueeze.default(arg351_1, -1);  arg351_1 = None
        unsqueeze_1449 = torch.ops.aten.unsqueeze.default(unsqueeze_1448, -1);  unsqueeze_1448 = None
        unsqueeze_1450 = torch.ops.aten.unsqueeze.default(mul_543, -1);  mul_543 = None
        unsqueeze_1451 = torch.ops.aten.unsqueeze.default(unsqueeze_1450, -1);  unsqueeze_1450 = None
        sub_181 = torch.ops.aten.sub.Tensor(cat_113, unsqueeze_1449);  cat_113 = unsqueeze_1449 = None
        mul_544 = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_1451);  sub_181 = unsqueeze_1451 = None
        unsqueeze_1452 = torch.ops.aten.unsqueeze.default(arg353_1, -1);  arg353_1 = None
        unsqueeze_1453 = torch.ops.aten.unsqueeze.default(unsqueeze_1452, -1);  unsqueeze_1452 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_544, unsqueeze_1453);  mul_544 = unsqueeze_1453 = None
        unsqueeze_1454 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1455 = torch.ops.aten.unsqueeze.default(unsqueeze_1454, -1);  unsqueeze_1454 = None
        add_420 = torch.ops.aten.add.Tensor(mul_545, unsqueeze_1455);  mul_545 = unsqueeze_1455 = None
        relu_181 = torch.ops.aten.relu.default(add_420);  add_420 = None
        convolution_181 = torch.ops.aten.convolution.default(relu_181, arg355_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_181 = arg355_1 = None
        add_421 = torch.ops.aten.add.Tensor(arg357_1, 0.001);  arg357_1 = None
        sqrt_182 = torch.ops.aten.sqrt.default(add_421);  add_421 = None
        reciprocal_182 = torch.ops.aten.reciprocal.default(sqrt_182);  sqrt_182 = None
        mul_546 = torch.ops.aten.mul.Tensor(reciprocal_182, 1);  reciprocal_182 = None
        unsqueeze_1456 = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
        unsqueeze_1457 = torch.ops.aten.unsqueeze.default(unsqueeze_1456, -1);  unsqueeze_1456 = None
        unsqueeze_1458 = torch.ops.aten.unsqueeze.default(mul_546, -1);  mul_546 = None
        unsqueeze_1459 = torch.ops.aten.unsqueeze.default(unsqueeze_1458, -1);  unsqueeze_1458 = None
        sub_182 = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1457);  convolution_181 = unsqueeze_1457 = None
        mul_547 = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_1459);  sub_182 = unsqueeze_1459 = None
        unsqueeze_1460 = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_1461 = torch.ops.aten.unsqueeze.default(unsqueeze_1460, -1);  unsqueeze_1460 = None
        mul_548 = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_1461);  mul_547 = unsqueeze_1461 = None
        unsqueeze_1462 = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1463 = torch.ops.aten.unsqueeze.default(unsqueeze_1462, -1);  unsqueeze_1462 = None
        add_422 = torch.ops.aten.add.Tensor(mul_548, unsqueeze_1463);  mul_548 = unsqueeze_1463 = None
        relu_182 = torch.ops.aten.relu.default(add_422);  add_422 = None
        convolution_182 = torch.ops.aten.convolution.default(relu_182, arg360_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_182 = arg360_1 = None
        add_423 = torch.ops.aten.add.Tensor(arg362_1, 0.001);  arg362_1 = None
        sqrt_183 = torch.ops.aten.sqrt.default(add_423);  add_423 = None
        reciprocal_183 = torch.ops.aten.reciprocal.default(sqrt_183);  sqrt_183 = None
        mul_549 = torch.ops.aten.mul.Tensor(reciprocal_183, 1);  reciprocal_183 = None
        unsqueeze_1464 = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_1465 = torch.ops.aten.unsqueeze.default(unsqueeze_1464, -1);  unsqueeze_1464 = None
        unsqueeze_1466 = torch.ops.aten.unsqueeze.default(mul_549, -1);  mul_549 = None
        unsqueeze_1467 = torch.ops.aten.unsqueeze.default(unsqueeze_1466, -1);  unsqueeze_1466 = None
        sub_183 = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1465);  convolution_182 = unsqueeze_1465 = None
        mul_550 = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_1467);  sub_183 = unsqueeze_1467 = None
        unsqueeze_1468 = torch.ops.aten.unsqueeze.default(arg363_1, -1);  arg363_1 = None
        unsqueeze_1469 = torch.ops.aten.unsqueeze.default(unsqueeze_1468, -1);  unsqueeze_1468 = None
        mul_551 = torch.ops.aten.mul.Tensor(mul_550, unsqueeze_1469);  mul_550 = unsqueeze_1469 = None
        unsqueeze_1470 = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1471 = torch.ops.aten.unsqueeze.default(unsqueeze_1470, -1);  unsqueeze_1470 = None
        add_424 = torch.ops.aten.add.Tensor(mul_551, unsqueeze_1471);  mul_551 = unsqueeze_1471 = None
        relu_183 = torch.ops.aten.relu.default(add_424);  add_424 = None
        convolution_183 = torch.ops.aten.convolution.default(relu_183, arg365_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_183 = arg365_1 = None
        slice_514 = torch.ops.aten.slice.Tensor(convolution_183, 1, 0, 1024)
        slice_518 = torch.ops.aten.slice.Tensor(convolution_183, 1, 1024, 9223372036854775807);  convolution_183 = None
        add_425 = torch.ops.aten.add.Tensor(add_418, slice_514);  add_418 = slice_514 = None
        cat_114 = torch.ops.aten.cat.default([cat_112, slice_518], 1);  cat_112 = slice_518 = None
        cat_115 = torch.ops.aten.cat.default([add_425, cat_114], 1)
        add_426 = torch.ops.aten.add.Tensor(arg367_1, 0.001);  arg367_1 = None
        sqrt_184 = torch.ops.aten.sqrt.default(add_426);  add_426 = None
        reciprocal_184 = torch.ops.aten.reciprocal.default(sqrt_184);  sqrt_184 = None
        mul_552 = torch.ops.aten.mul.Tensor(reciprocal_184, 1);  reciprocal_184 = None
        unsqueeze_1472 = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_1473 = torch.ops.aten.unsqueeze.default(unsqueeze_1472, -1);  unsqueeze_1472 = None
        unsqueeze_1474 = torch.ops.aten.unsqueeze.default(mul_552, -1);  mul_552 = None
        unsqueeze_1475 = torch.ops.aten.unsqueeze.default(unsqueeze_1474, -1);  unsqueeze_1474 = None
        sub_184 = torch.ops.aten.sub.Tensor(cat_115, unsqueeze_1473);  cat_115 = unsqueeze_1473 = None
        mul_553 = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_1475);  sub_184 = unsqueeze_1475 = None
        unsqueeze_1476 = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_1477 = torch.ops.aten.unsqueeze.default(unsqueeze_1476, -1);  unsqueeze_1476 = None
        mul_554 = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_1477);  mul_553 = unsqueeze_1477 = None
        unsqueeze_1478 = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1479 = torch.ops.aten.unsqueeze.default(unsqueeze_1478, -1);  unsqueeze_1478 = None
        add_427 = torch.ops.aten.add.Tensor(mul_554, unsqueeze_1479);  mul_554 = unsqueeze_1479 = None
        relu_184 = torch.ops.aten.relu.default(add_427);  add_427 = None
        convolution_184 = torch.ops.aten.convolution.default(relu_184, arg370_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_184 = arg370_1 = None
        add_428 = torch.ops.aten.add.Tensor(arg372_1, 0.001);  arg372_1 = None
        sqrt_185 = torch.ops.aten.sqrt.default(add_428);  add_428 = None
        reciprocal_185 = torch.ops.aten.reciprocal.default(sqrt_185);  sqrt_185 = None
        mul_555 = torch.ops.aten.mul.Tensor(reciprocal_185, 1);  reciprocal_185 = None
        unsqueeze_1480 = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_1481 = torch.ops.aten.unsqueeze.default(unsqueeze_1480, -1);  unsqueeze_1480 = None
        unsqueeze_1482 = torch.ops.aten.unsqueeze.default(mul_555, -1);  mul_555 = None
        unsqueeze_1483 = torch.ops.aten.unsqueeze.default(unsqueeze_1482, -1);  unsqueeze_1482 = None
        sub_185 = torch.ops.aten.sub.Tensor(convolution_184, unsqueeze_1481);  convolution_184 = unsqueeze_1481 = None
        mul_556 = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_1483);  sub_185 = unsqueeze_1483 = None
        unsqueeze_1484 = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_1485 = torch.ops.aten.unsqueeze.default(unsqueeze_1484, -1);  unsqueeze_1484 = None
        mul_557 = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_1485);  mul_556 = unsqueeze_1485 = None
        unsqueeze_1486 = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1487 = torch.ops.aten.unsqueeze.default(unsqueeze_1486, -1);  unsqueeze_1486 = None
        add_429 = torch.ops.aten.add.Tensor(mul_557, unsqueeze_1487);  mul_557 = unsqueeze_1487 = None
        relu_185 = torch.ops.aten.relu.default(add_429);  add_429 = None
        convolution_185 = torch.ops.aten.convolution.default(relu_185, arg375_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_185 = arg375_1 = None
        add_430 = torch.ops.aten.add.Tensor(arg377_1, 0.001);  arg377_1 = None
        sqrt_186 = torch.ops.aten.sqrt.default(add_430);  add_430 = None
        reciprocal_186 = torch.ops.aten.reciprocal.default(sqrt_186);  sqrt_186 = None
        mul_558 = torch.ops.aten.mul.Tensor(reciprocal_186, 1);  reciprocal_186 = None
        unsqueeze_1488 = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_1489 = torch.ops.aten.unsqueeze.default(unsqueeze_1488, -1);  unsqueeze_1488 = None
        unsqueeze_1490 = torch.ops.aten.unsqueeze.default(mul_558, -1);  mul_558 = None
        unsqueeze_1491 = torch.ops.aten.unsqueeze.default(unsqueeze_1490, -1);  unsqueeze_1490 = None
        sub_186 = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1489);  convolution_185 = unsqueeze_1489 = None
        mul_559 = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_1491);  sub_186 = unsqueeze_1491 = None
        unsqueeze_1492 = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_1493 = torch.ops.aten.unsqueeze.default(unsqueeze_1492, -1);  unsqueeze_1492 = None
        mul_560 = torch.ops.aten.mul.Tensor(mul_559, unsqueeze_1493);  mul_559 = unsqueeze_1493 = None
        unsqueeze_1494 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1495 = torch.ops.aten.unsqueeze.default(unsqueeze_1494, -1);  unsqueeze_1494 = None
        add_431 = torch.ops.aten.add.Tensor(mul_560, unsqueeze_1495);  mul_560 = unsqueeze_1495 = None
        relu_186 = torch.ops.aten.relu.default(add_431);  add_431 = None
        convolution_186 = torch.ops.aten.convolution.default(relu_186, arg380_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_186 = arg380_1 = None
        slice_522 = torch.ops.aten.slice.Tensor(convolution_186, 1, 0, 1024)
        slice_526 = torch.ops.aten.slice.Tensor(convolution_186, 1, 1024, 9223372036854775807);  convolution_186 = None
        add_432 = torch.ops.aten.add.Tensor(add_425, slice_522);  add_425 = slice_522 = None
        cat_116 = torch.ops.aten.cat.default([cat_114, slice_526], 1);  cat_114 = slice_526 = None
        cat_117 = torch.ops.aten.cat.default([add_432, cat_116], 1)
        add_433 = torch.ops.aten.add.Tensor(arg382_1, 0.001);  arg382_1 = None
        sqrt_187 = torch.ops.aten.sqrt.default(add_433);  add_433 = None
        reciprocal_187 = torch.ops.aten.reciprocal.default(sqrt_187);  sqrt_187 = None
        mul_561 = torch.ops.aten.mul.Tensor(reciprocal_187, 1);  reciprocal_187 = None
        unsqueeze_1496 = torch.ops.aten.unsqueeze.default(arg381_1, -1);  arg381_1 = None
        unsqueeze_1497 = torch.ops.aten.unsqueeze.default(unsqueeze_1496, -1);  unsqueeze_1496 = None
        unsqueeze_1498 = torch.ops.aten.unsqueeze.default(mul_561, -1);  mul_561 = None
        unsqueeze_1499 = torch.ops.aten.unsqueeze.default(unsqueeze_1498, -1);  unsqueeze_1498 = None
        sub_187 = torch.ops.aten.sub.Tensor(cat_117, unsqueeze_1497);  cat_117 = unsqueeze_1497 = None
        mul_562 = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_1499);  sub_187 = unsqueeze_1499 = None
        unsqueeze_1500 = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
        unsqueeze_1501 = torch.ops.aten.unsqueeze.default(unsqueeze_1500, -1);  unsqueeze_1500 = None
        mul_563 = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_1501);  mul_562 = unsqueeze_1501 = None
        unsqueeze_1502 = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1503 = torch.ops.aten.unsqueeze.default(unsqueeze_1502, -1);  unsqueeze_1502 = None
        add_434 = torch.ops.aten.add.Tensor(mul_563, unsqueeze_1503);  mul_563 = unsqueeze_1503 = None
        relu_187 = torch.ops.aten.relu.default(add_434);  add_434 = None
        convolution_187 = torch.ops.aten.convolution.default(relu_187, arg385_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_187 = arg385_1 = None
        add_435 = torch.ops.aten.add.Tensor(arg387_1, 0.001);  arg387_1 = None
        sqrt_188 = torch.ops.aten.sqrt.default(add_435);  add_435 = None
        reciprocal_188 = torch.ops.aten.reciprocal.default(sqrt_188);  sqrt_188 = None
        mul_564 = torch.ops.aten.mul.Tensor(reciprocal_188, 1);  reciprocal_188 = None
        unsqueeze_1504 = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
        unsqueeze_1505 = torch.ops.aten.unsqueeze.default(unsqueeze_1504, -1);  unsqueeze_1504 = None
        unsqueeze_1506 = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_1507 = torch.ops.aten.unsqueeze.default(unsqueeze_1506, -1);  unsqueeze_1506 = None
        sub_188 = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1505);  convolution_187 = unsqueeze_1505 = None
        mul_565 = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_1507);  sub_188 = unsqueeze_1507 = None
        unsqueeze_1508 = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1509 = torch.ops.aten.unsqueeze.default(unsqueeze_1508, -1);  unsqueeze_1508 = None
        mul_566 = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_1509);  mul_565 = unsqueeze_1509 = None
        unsqueeze_1510 = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1511 = torch.ops.aten.unsqueeze.default(unsqueeze_1510, -1);  unsqueeze_1510 = None
        add_436 = torch.ops.aten.add.Tensor(mul_566, unsqueeze_1511);  mul_566 = unsqueeze_1511 = None
        relu_188 = torch.ops.aten.relu.default(add_436);  add_436 = None
        convolution_188 = torch.ops.aten.convolution.default(relu_188, arg390_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_188 = arg390_1 = None
        add_437 = torch.ops.aten.add.Tensor(arg392_1, 0.001);  arg392_1 = None
        sqrt_189 = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_189 = torch.ops.aten.reciprocal.default(sqrt_189);  sqrt_189 = None
        mul_567 = torch.ops.aten.mul.Tensor(reciprocal_189, 1);  reciprocal_189 = None
        unsqueeze_1512 = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_1513 = torch.ops.aten.unsqueeze.default(unsqueeze_1512, -1);  unsqueeze_1512 = None
        unsqueeze_1514 = torch.ops.aten.unsqueeze.default(mul_567, -1);  mul_567 = None
        unsqueeze_1515 = torch.ops.aten.unsqueeze.default(unsqueeze_1514, -1);  unsqueeze_1514 = None
        sub_189 = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1513);  convolution_188 = unsqueeze_1513 = None
        mul_568 = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_1515);  sub_189 = unsqueeze_1515 = None
        unsqueeze_1516 = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_1517 = torch.ops.aten.unsqueeze.default(unsqueeze_1516, -1);  unsqueeze_1516 = None
        mul_569 = torch.ops.aten.mul.Tensor(mul_568, unsqueeze_1517);  mul_568 = unsqueeze_1517 = None
        unsqueeze_1518 = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1519 = torch.ops.aten.unsqueeze.default(unsqueeze_1518, -1);  unsqueeze_1518 = None
        add_438 = torch.ops.aten.add.Tensor(mul_569, unsqueeze_1519);  mul_569 = unsqueeze_1519 = None
        relu_189 = torch.ops.aten.relu.default(add_438);  add_438 = None
        convolution_189 = torch.ops.aten.convolution.default(relu_189, arg395_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_189 = arg395_1 = None
        slice_530 = torch.ops.aten.slice.Tensor(convolution_189, 1, 0, 1024)
        slice_534 = torch.ops.aten.slice.Tensor(convolution_189, 1, 1024, 9223372036854775807);  convolution_189 = None
        add_439 = torch.ops.aten.add.Tensor(add_432, slice_530);  add_432 = slice_530 = None
        cat_118 = torch.ops.aten.cat.default([cat_116, slice_534], 1);  cat_116 = slice_534 = None
        cat_119 = torch.ops.aten.cat.default([add_439, cat_118], 1)
        add_440 = torch.ops.aten.add.Tensor(arg397_1, 0.001);  arg397_1 = None
        sqrt_190 = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_190 = torch.ops.aten.reciprocal.default(sqrt_190);  sqrt_190 = None
        mul_570 = torch.ops.aten.mul.Tensor(reciprocal_190, 1);  reciprocal_190 = None
        unsqueeze_1520 = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_1521 = torch.ops.aten.unsqueeze.default(unsqueeze_1520, -1);  unsqueeze_1520 = None
        unsqueeze_1522 = torch.ops.aten.unsqueeze.default(mul_570, -1);  mul_570 = None
        unsqueeze_1523 = torch.ops.aten.unsqueeze.default(unsqueeze_1522, -1);  unsqueeze_1522 = None
        sub_190 = torch.ops.aten.sub.Tensor(cat_119, unsqueeze_1521);  cat_119 = unsqueeze_1521 = None
        mul_571 = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_1523);  sub_190 = unsqueeze_1523 = None
        unsqueeze_1524 = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1525 = torch.ops.aten.unsqueeze.default(unsqueeze_1524, -1);  unsqueeze_1524 = None
        mul_572 = torch.ops.aten.mul.Tensor(mul_571, unsqueeze_1525);  mul_571 = unsqueeze_1525 = None
        unsqueeze_1526 = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1527 = torch.ops.aten.unsqueeze.default(unsqueeze_1526, -1);  unsqueeze_1526 = None
        add_441 = torch.ops.aten.add.Tensor(mul_572, unsqueeze_1527);  mul_572 = unsqueeze_1527 = None
        relu_190 = torch.ops.aten.relu.default(add_441);  add_441 = None
        convolution_190 = torch.ops.aten.convolution.default(relu_190, arg400_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_190 = arg400_1 = None
        add_442 = torch.ops.aten.add.Tensor(arg402_1, 0.001);  arg402_1 = None
        sqrt_191 = torch.ops.aten.sqrt.default(add_442);  add_442 = None
        reciprocal_191 = torch.ops.aten.reciprocal.default(sqrt_191);  sqrt_191 = None
        mul_573 = torch.ops.aten.mul.Tensor(reciprocal_191, 1);  reciprocal_191 = None
        unsqueeze_1528 = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
        unsqueeze_1529 = torch.ops.aten.unsqueeze.default(unsqueeze_1528, -1);  unsqueeze_1528 = None
        unsqueeze_1530 = torch.ops.aten.unsqueeze.default(mul_573, -1);  mul_573 = None
        unsqueeze_1531 = torch.ops.aten.unsqueeze.default(unsqueeze_1530, -1);  unsqueeze_1530 = None
        sub_191 = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1529);  convolution_190 = unsqueeze_1529 = None
        mul_574 = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_1531);  sub_191 = unsqueeze_1531 = None
        unsqueeze_1532 = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_1533 = torch.ops.aten.unsqueeze.default(unsqueeze_1532, -1);  unsqueeze_1532 = None
        mul_575 = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_1533);  mul_574 = unsqueeze_1533 = None
        unsqueeze_1534 = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1535 = torch.ops.aten.unsqueeze.default(unsqueeze_1534, -1);  unsqueeze_1534 = None
        add_443 = torch.ops.aten.add.Tensor(mul_575, unsqueeze_1535);  mul_575 = unsqueeze_1535 = None
        relu_191 = torch.ops.aten.relu.default(add_443);  add_443 = None
        convolution_191 = torch.ops.aten.convolution.default(relu_191, arg405_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_191 = arg405_1 = None
        add_444 = torch.ops.aten.add.Tensor(arg407_1, 0.001);  arg407_1 = None
        sqrt_192 = torch.ops.aten.sqrt.default(add_444);  add_444 = None
        reciprocal_192 = torch.ops.aten.reciprocal.default(sqrt_192);  sqrt_192 = None
        mul_576 = torch.ops.aten.mul.Tensor(reciprocal_192, 1);  reciprocal_192 = None
        unsqueeze_1536 = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_1537 = torch.ops.aten.unsqueeze.default(unsqueeze_1536, -1);  unsqueeze_1536 = None
        unsqueeze_1538 = torch.ops.aten.unsqueeze.default(mul_576, -1);  mul_576 = None
        unsqueeze_1539 = torch.ops.aten.unsqueeze.default(unsqueeze_1538, -1);  unsqueeze_1538 = None
        sub_192 = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1537);  convolution_191 = unsqueeze_1537 = None
        mul_577 = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_1539);  sub_192 = unsqueeze_1539 = None
        unsqueeze_1540 = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_1541 = torch.ops.aten.unsqueeze.default(unsqueeze_1540, -1);  unsqueeze_1540 = None
        mul_578 = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_1541);  mul_577 = unsqueeze_1541 = None
        unsqueeze_1542 = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1543 = torch.ops.aten.unsqueeze.default(unsqueeze_1542, -1);  unsqueeze_1542 = None
        add_445 = torch.ops.aten.add.Tensor(mul_578, unsqueeze_1543);  mul_578 = unsqueeze_1543 = None
        relu_192 = torch.ops.aten.relu.default(add_445);  add_445 = None
        convolution_192 = torch.ops.aten.convolution.default(relu_192, arg410_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_192 = arg410_1 = None
        slice_538 = torch.ops.aten.slice.Tensor(convolution_192, 1, 0, 1024)
        slice_542 = torch.ops.aten.slice.Tensor(convolution_192, 1, 1024, 9223372036854775807);  convolution_192 = None
        add_446 = torch.ops.aten.add.Tensor(add_439, slice_538);  add_439 = slice_538 = None
        cat_120 = torch.ops.aten.cat.default([cat_118, slice_542], 1);  cat_118 = slice_542 = None
        cat_121 = torch.ops.aten.cat.default([add_446, cat_120], 1)
        add_447 = torch.ops.aten.add.Tensor(arg412_1, 0.001);  arg412_1 = None
        sqrt_193 = torch.ops.aten.sqrt.default(add_447);  add_447 = None
        reciprocal_193 = torch.ops.aten.reciprocal.default(sqrt_193);  sqrt_193 = None
        mul_579 = torch.ops.aten.mul.Tensor(reciprocal_193, 1);  reciprocal_193 = None
        unsqueeze_1544 = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_1545 = torch.ops.aten.unsqueeze.default(unsqueeze_1544, -1);  unsqueeze_1544 = None
        unsqueeze_1546 = torch.ops.aten.unsqueeze.default(mul_579, -1);  mul_579 = None
        unsqueeze_1547 = torch.ops.aten.unsqueeze.default(unsqueeze_1546, -1);  unsqueeze_1546 = None
        sub_193 = torch.ops.aten.sub.Tensor(cat_121, unsqueeze_1545);  cat_121 = unsqueeze_1545 = None
        mul_580 = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_1547);  sub_193 = unsqueeze_1547 = None
        unsqueeze_1548 = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
        unsqueeze_1549 = torch.ops.aten.unsqueeze.default(unsqueeze_1548, -1);  unsqueeze_1548 = None
        mul_581 = torch.ops.aten.mul.Tensor(mul_580, unsqueeze_1549);  mul_580 = unsqueeze_1549 = None
        unsqueeze_1550 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1551 = torch.ops.aten.unsqueeze.default(unsqueeze_1550, -1);  unsqueeze_1550 = None
        add_448 = torch.ops.aten.add.Tensor(mul_581, unsqueeze_1551);  mul_581 = unsqueeze_1551 = None
        relu_193 = torch.ops.aten.relu.default(add_448);  add_448 = None
        convolution_193 = torch.ops.aten.convolution.default(relu_193, arg415_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_193 = arg415_1 = None
        add_449 = torch.ops.aten.add.Tensor(arg417_1, 0.001);  arg417_1 = None
        sqrt_194 = torch.ops.aten.sqrt.default(add_449);  add_449 = None
        reciprocal_194 = torch.ops.aten.reciprocal.default(sqrt_194);  sqrt_194 = None
        mul_582 = torch.ops.aten.mul.Tensor(reciprocal_194, 1);  reciprocal_194 = None
        unsqueeze_1552 = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
        unsqueeze_1553 = torch.ops.aten.unsqueeze.default(unsqueeze_1552, -1);  unsqueeze_1552 = None
        unsqueeze_1554 = torch.ops.aten.unsqueeze.default(mul_582, -1);  mul_582 = None
        unsqueeze_1555 = torch.ops.aten.unsqueeze.default(unsqueeze_1554, -1);  unsqueeze_1554 = None
        sub_194 = torch.ops.aten.sub.Tensor(convolution_193, unsqueeze_1553);  convolution_193 = unsqueeze_1553 = None
        mul_583 = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_1555);  sub_194 = unsqueeze_1555 = None
        unsqueeze_1556 = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_1557 = torch.ops.aten.unsqueeze.default(unsqueeze_1556, -1);  unsqueeze_1556 = None
        mul_584 = torch.ops.aten.mul.Tensor(mul_583, unsqueeze_1557);  mul_583 = unsqueeze_1557 = None
        unsqueeze_1558 = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_1559 = torch.ops.aten.unsqueeze.default(unsqueeze_1558, -1);  unsqueeze_1558 = None
        add_450 = torch.ops.aten.add.Tensor(mul_584, unsqueeze_1559);  mul_584 = unsqueeze_1559 = None
        relu_194 = torch.ops.aten.relu.default(add_450);  add_450 = None
        convolution_194 = torch.ops.aten.convolution.default(relu_194, arg420_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_194 = arg420_1 = None
        add_451 = torch.ops.aten.add.Tensor(arg422_1, 0.001);  arg422_1 = None
        sqrt_195 = torch.ops.aten.sqrt.default(add_451);  add_451 = None
        reciprocal_195 = torch.ops.aten.reciprocal.default(sqrt_195);  sqrt_195 = None
        mul_585 = torch.ops.aten.mul.Tensor(reciprocal_195, 1);  reciprocal_195 = None
        unsqueeze_1560 = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
        unsqueeze_1561 = torch.ops.aten.unsqueeze.default(unsqueeze_1560, -1);  unsqueeze_1560 = None
        unsqueeze_1562 = torch.ops.aten.unsqueeze.default(mul_585, -1);  mul_585 = None
        unsqueeze_1563 = torch.ops.aten.unsqueeze.default(unsqueeze_1562, -1);  unsqueeze_1562 = None
        sub_195 = torch.ops.aten.sub.Tensor(convolution_194, unsqueeze_1561);  convolution_194 = unsqueeze_1561 = None
        mul_586 = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_1563);  sub_195 = unsqueeze_1563 = None
        unsqueeze_1564 = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1565 = torch.ops.aten.unsqueeze.default(unsqueeze_1564, -1);  unsqueeze_1564 = None
        mul_587 = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_1565);  mul_586 = unsqueeze_1565 = None
        unsqueeze_1566 = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_1567 = torch.ops.aten.unsqueeze.default(unsqueeze_1566, -1);  unsqueeze_1566 = None
        add_452 = torch.ops.aten.add.Tensor(mul_587, unsqueeze_1567);  mul_587 = unsqueeze_1567 = None
        relu_195 = torch.ops.aten.relu.default(add_452);  add_452 = None
        convolution_195 = torch.ops.aten.convolution.default(relu_195, arg425_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_195 = arg425_1 = None
        slice_546 = torch.ops.aten.slice.Tensor(convolution_195, 1, 0, 1024)
        slice_550 = torch.ops.aten.slice.Tensor(convolution_195, 1, 1024, 9223372036854775807);  convolution_195 = None
        add_453 = torch.ops.aten.add.Tensor(add_446, slice_546);  add_446 = slice_546 = None
        cat_122 = torch.ops.aten.cat.default([cat_120, slice_550], 1);  cat_120 = slice_550 = None
        cat_123 = torch.ops.aten.cat.default([add_453, cat_122], 1)
        add_454 = torch.ops.aten.add.Tensor(arg427_1, 0.001);  arg427_1 = None
        sqrt_196 = torch.ops.aten.sqrt.default(add_454);  add_454 = None
        reciprocal_196 = torch.ops.aten.reciprocal.default(sqrt_196);  sqrt_196 = None
        mul_588 = torch.ops.aten.mul.Tensor(reciprocal_196, 1);  reciprocal_196 = None
        unsqueeze_1568 = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_1569 = torch.ops.aten.unsqueeze.default(unsqueeze_1568, -1);  unsqueeze_1568 = None
        unsqueeze_1570 = torch.ops.aten.unsqueeze.default(mul_588, -1);  mul_588 = None
        unsqueeze_1571 = torch.ops.aten.unsqueeze.default(unsqueeze_1570, -1);  unsqueeze_1570 = None
        sub_196 = torch.ops.aten.sub.Tensor(cat_123, unsqueeze_1569);  cat_123 = unsqueeze_1569 = None
        mul_589 = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_1571);  sub_196 = unsqueeze_1571 = None
        unsqueeze_1572 = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1573 = torch.ops.aten.unsqueeze.default(unsqueeze_1572, -1);  unsqueeze_1572 = None
        mul_590 = torch.ops.aten.mul.Tensor(mul_589, unsqueeze_1573);  mul_589 = unsqueeze_1573 = None
        unsqueeze_1574 = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_1575 = torch.ops.aten.unsqueeze.default(unsqueeze_1574, -1);  unsqueeze_1574 = None
        add_455 = torch.ops.aten.add.Tensor(mul_590, unsqueeze_1575);  mul_590 = unsqueeze_1575 = None
        relu_196 = torch.ops.aten.relu.default(add_455);  add_455 = None
        convolution_196 = torch.ops.aten.convolution.default(relu_196, arg430_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_196 = arg430_1 = None
        add_456 = torch.ops.aten.add.Tensor(arg432_1, 0.001);  arg432_1 = None
        sqrt_197 = torch.ops.aten.sqrt.default(add_456);  add_456 = None
        reciprocal_197 = torch.ops.aten.reciprocal.default(sqrt_197);  sqrt_197 = None
        mul_591 = torch.ops.aten.mul.Tensor(reciprocal_197, 1);  reciprocal_197 = None
        unsqueeze_1576 = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_1577 = torch.ops.aten.unsqueeze.default(unsqueeze_1576, -1);  unsqueeze_1576 = None
        unsqueeze_1578 = torch.ops.aten.unsqueeze.default(mul_591, -1);  mul_591 = None
        unsqueeze_1579 = torch.ops.aten.unsqueeze.default(unsqueeze_1578, -1);  unsqueeze_1578 = None
        sub_197 = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1577);  convolution_196 = unsqueeze_1577 = None
        mul_592 = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_1579);  sub_197 = unsqueeze_1579 = None
        unsqueeze_1580 = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
        unsqueeze_1581 = torch.ops.aten.unsqueeze.default(unsqueeze_1580, -1);  unsqueeze_1580 = None
        mul_593 = torch.ops.aten.mul.Tensor(mul_592, unsqueeze_1581);  mul_592 = unsqueeze_1581 = None
        unsqueeze_1582 = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_1583 = torch.ops.aten.unsqueeze.default(unsqueeze_1582, -1);  unsqueeze_1582 = None
        add_457 = torch.ops.aten.add.Tensor(mul_593, unsqueeze_1583);  mul_593 = unsqueeze_1583 = None
        relu_197 = torch.ops.aten.relu.default(add_457);  add_457 = None
        convolution_197 = torch.ops.aten.convolution.default(relu_197, arg435_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_197 = arg435_1 = None
        add_458 = torch.ops.aten.add.Tensor(arg437_1, 0.001);  arg437_1 = None
        sqrt_198 = torch.ops.aten.sqrt.default(add_458);  add_458 = None
        reciprocal_198 = torch.ops.aten.reciprocal.default(sqrt_198);  sqrt_198 = None
        mul_594 = torch.ops.aten.mul.Tensor(reciprocal_198, 1);  reciprocal_198 = None
        unsqueeze_1584 = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_1585 = torch.ops.aten.unsqueeze.default(unsqueeze_1584, -1);  unsqueeze_1584 = None
        unsqueeze_1586 = torch.ops.aten.unsqueeze.default(mul_594, -1);  mul_594 = None
        unsqueeze_1587 = torch.ops.aten.unsqueeze.default(unsqueeze_1586, -1);  unsqueeze_1586 = None
        sub_198 = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1585);  convolution_197 = unsqueeze_1585 = None
        mul_595 = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_1587);  sub_198 = unsqueeze_1587 = None
        unsqueeze_1588 = torch.ops.aten.unsqueeze.default(arg438_1, -1);  arg438_1 = None
        unsqueeze_1589 = torch.ops.aten.unsqueeze.default(unsqueeze_1588, -1);  unsqueeze_1588 = None
        mul_596 = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_1589);  mul_595 = unsqueeze_1589 = None
        unsqueeze_1590 = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_1591 = torch.ops.aten.unsqueeze.default(unsqueeze_1590, -1);  unsqueeze_1590 = None
        add_459 = torch.ops.aten.add.Tensor(mul_596, unsqueeze_1591);  mul_596 = unsqueeze_1591 = None
        relu_198 = torch.ops.aten.relu.default(add_459);  add_459 = None
        convolution_198 = torch.ops.aten.convolution.default(relu_198, arg440_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_198 = arg440_1 = None
        slice_554 = torch.ops.aten.slice.Tensor(convolution_198, 1, 0, 1024)
        slice_558 = torch.ops.aten.slice.Tensor(convolution_198, 1, 1024, 9223372036854775807);  convolution_198 = None
        add_460 = torch.ops.aten.add.Tensor(add_453, slice_554);  add_453 = slice_554 = None
        cat_124 = torch.ops.aten.cat.default([cat_122, slice_558], 1);  cat_122 = slice_558 = None
        cat_125 = torch.ops.aten.cat.default([add_460, cat_124], 1)
        add_461 = torch.ops.aten.add.Tensor(arg442_1, 0.001);  arg442_1 = None
        sqrt_199 = torch.ops.aten.sqrt.default(add_461);  add_461 = None
        reciprocal_199 = torch.ops.aten.reciprocal.default(sqrt_199);  sqrt_199 = None
        mul_597 = torch.ops.aten.mul.Tensor(reciprocal_199, 1);  reciprocal_199 = None
        unsqueeze_1592 = torch.ops.aten.unsqueeze.default(arg441_1, -1);  arg441_1 = None
        unsqueeze_1593 = torch.ops.aten.unsqueeze.default(unsqueeze_1592, -1);  unsqueeze_1592 = None
        unsqueeze_1594 = torch.ops.aten.unsqueeze.default(mul_597, -1);  mul_597 = None
        unsqueeze_1595 = torch.ops.aten.unsqueeze.default(unsqueeze_1594, -1);  unsqueeze_1594 = None
        sub_199 = torch.ops.aten.sub.Tensor(cat_125, unsqueeze_1593);  cat_125 = unsqueeze_1593 = None
        mul_598 = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_1595);  sub_199 = unsqueeze_1595 = None
        unsqueeze_1596 = torch.ops.aten.unsqueeze.default(arg443_1, -1);  arg443_1 = None
        unsqueeze_1597 = torch.ops.aten.unsqueeze.default(unsqueeze_1596, -1);  unsqueeze_1596 = None
        mul_599 = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_1597);  mul_598 = unsqueeze_1597 = None
        unsqueeze_1598 = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1599 = torch.ops.aten.unsqueeze.default(unsqueeze_1598, -1);  unsqueeze_1598 = None
        add_462 = torch.ops.aten.add.Tensor(mul_599, unsqueeze_1599);  mul_599 = unsqueeze_1599 = None
        relu_199 = torch.ops.aten.relu.default(add_462);  add_462 = None
        convolution_199 = torch.ops.aten.convolution.default(relu_199, arg445_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_199 = arg445_1 = None
        add_463 = torch.ops.aten.add.Tensor(arg447_1, 0.001);  arg447_1 = None
        sqrt_200 = torch.ops.aten.sqrt.default(add_463);  add_463 = None
        reciprocal_200 = torch.ops.aten.reciprocal.default(sqrt_200);  sqrt_200 = None
        mul_600 = torch.ops.aten.mul.Tensor(reciprocal_200, 1);  reciprocal_200 = None
        unsqueeze_1600 = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
        unsqueeze_1601 = torch.ops.aten.unsqueeze.default(unsqueeze_1600, -1);  unsqueeze_1600 = None
        unsqueeze_1602 = torch.ops.aten.unsqueeze.default(mul_600, -1);  mul_600 = None
        unsqueeze_1603 = torch.ops.aten.unsqueeze.default(unsqueeze_1602, -1);  unsqueeze_1602 = None
        sub_200 = torch.ops.aten.sub.Tensor(convolution_199, unsqueeze_1601);  convolution_199 = unsqueeze_1601 = None
        mul_601 = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_1603);  sub_200 = unsqueeze_1603 = None
        unsqueeze_1604 = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
        unsqueeze_1605 = torch.ops.aten.unsqueeze.default(unsqueeze_1604, -1);  unsqueeze_1604 = None
        mul_602 = torch.ops.aten.mul.Tensor(mul_601, unsqueeze_1605);  mul_601 = unsqueeze_1605 = None
        unsqueeze_1606 = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1607 = torch.ops.aten.unsqueeze.default(unsqueeze_1606, -1);  unsqueeze_1606 = None
        add_464 = torch.ops.aten.add.Tensor(mul_602, unsqueeze_1607);  mul_602 = unsqueeze_1607 = None
        relu_200 = torch.ops.aten.relu.default(add_464);  add_464 = None
        convolution_200 = torch.ops.aten.convolution.default(relu_200, arg450_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_200 = arg450_1 = None
        add_465 = torch.ops.aten.add.Tensor(arg452_1, 0.001);  arg452_1 = None
        sqrt_201 = torch.ops.aten.sqrt.default(add_465);  add_465 = None
        reciprocal_201 = torch.ops.aten.reciprocal.default(sqrt_201);  sqrt_201 = None
        mul_603 = torch.ops.aten.mul.Tensor(reciprocal_201, 1);  reciprocal_201 = None
        unsqueeze_1608 = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
        unsqueeze_1609 = torch.ops.aten.unsqueeze.default(unsqueeze_1608, -1);  unsqueeze_1608 = None
        unsqueeze_1610 = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1611 = torch.ops.aten.unsqueeze.default(unsqueeze_1610, -1);  unsqueeze_1610 = None
        sub_201 = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1609);  convolution_200 = unsqueeze_1609 = None
        mul_604 = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_1611);  sub_201 = unsqueeze_1611 = None
        unsqueeze_1612 = torch.ops.aten.unsqueeze.default(arg453_1, -1);  arg453_1 = None
        unsqueeze_1613 = torch.ops.aten.unsqueeze.default(unsqueeze_1612, -1);  unsqueeze_1612 = None
        mul_605 = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1613);  mul_604 = unsqueeze_1613 = None
        unsqueeze_1614 = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1615 = torch.ops.aten.unsqueeze.default(unsqueeze_1614, -1);  unsqueeze_1614 = None
        add_466 = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1615);  mul_605 = unsqueeze_1615 = None
        relu_201 = torch.ops.aten.relu.default(add_466);  add_466 = None
        convolution_201 = torch.ops.aten.convolution.default(relu_201, arg455_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_201 = arg455_1 = None
        slice_562 = torch.ops.aten.slice.Tensor(convolution_201, 1, 0, 1024)
        slice_566 = torch.ops.aten.slice.Tensor(convolution_201, 1, 1024, 9223372036854775807);  convolution_201 = None
        add_467 = torch.ops.aten.add.Tensor(add_460, slice_562);  add_460 = slice_562 = None
        cat_126 = torch.ops.aten.cat.default([cat_124, slice_566], 1);  cat_124 = slice_566 = None
        cat_127 = torch.ops.aten.cat.default([add_467, cat_126], 1)
        add_468 = torch.ops.aten.add.Tensor(arg457_1, 0.001);  arg457_1 = None
        sqrt_202 = torch.ops.aten.sqrt.default(add_468);  add_468 = None
        reciprocal_202 = torch.ops.aten.reciprocal.default(sqrt_202);  sqrt_202 = None
        mul_606 = torch.ops.aten.mul.Tensor(reciprocal_202, 1);  reciprocal_202 = None
        unsqueeze_1616 = torch.ops.aten.unsqueeze.default(arg456_1, -1);  arg456_1 = None
        unsqueeze_1617 = torch.ops.aten.unsqueeze.default(unsqueeze_1616, -1);  unsqueeze_1616 = None
        unsqueeze_1618 = torch.ops.aten.unsqueeze.default(mul_606, -1);  mul_606 = None
        unsqueeze_1619 = torch.ops.aten.unsqueeze.default(unsqueeze_1618, -1);  unsqueeze_1618 = None
        sub_202 = torch.ops.aten.sub.Tensor(cat_127, unsqueeze_1617);  cat_127 = unsqueeze_1617 = None
        mul_607 = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_1619);  sub_202 = unsqueeze_1619 = None
        unsqueeze_1620 = torch.ops.aten.unsqueeze.default(arg458_1, -1);  arg458_1 = None
        unsqueeze_1621 = torch.ops.aten.unsqueeze.default(unsqueeze_1620, -1);  unsqueeze_1620 = None
        mul_608 = torch.ops.aten.mul.Tensor(mul_607, unsqueeze_1621);  mul_607 = unsqueeze_1621 = None
        unsqueeze_1622 = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_1623 = torch.ops.aten.unsqueeze.default(unsqueeze_1622, -1);  unsqueeze_1622 = None
        add_469 = torch.ops.aten.add.Tensor(mul_608, unsqueeze_1623);  mul_608 = unsqueeze_1623 = None
        relu_202 = torch.ops.aten.relu.default(add_469);  add_469 = None
        convolution_202 = torch.ops.aten.convolution.default(relu_202, arg460_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_202 = arg460_1 = None
        add_470 = torch.ops.aten.add.Tensor(arg462_1, 0.001);  arg462_1 = None
        sqrt_203 = torch.ops.aten.sqrt.default(add_470);  add_470 = None
        reciprocal_203 = torch.ops.aten.reciprocal.default(sqrt_203);  sqrt_203 = None
        mul_609 = torch.ops.aten.mul.Tensor(reciprocal_203, 1);  reciprocal_203 = None
        unsqueeze_1624 = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
        unsqueeze_1625 = torch.ops.aten.unsqueeze.default(unsqueeze_1624, -1);  unsqueeze_1624 = None
        unsqueeze_1626 = torch.ops.aten.unsqueeze.default(mul_609, -1);  mul_609 = None
        unsqueeze_1627 = torch.ops.aten.unsqueeze.default(unsqueeze_1626, -1);  unsqueeze_1626 = None
        sub_203 = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1625);  convolution_202 = unsqueeze_1625 = None
        mul_610 = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_1627);  sub_203 = unsqueeze_1627 = None
        unsqueeze_1628 = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
        unsqueeze_1629 = torch.ops.aten.unsqueeze.default(unsqueeze_1628, -1);  unsqueeze_1628 = None
        mul_611 = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_1629);  mul_610 = unsqueeze_1629 = None
        unsqueeze_1630 = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1631 = torch.ops.aten.unsqueeze.default(unsqueeze_1630, -1);  unsqueeze_1630 = None
        add_471 = torch.ops.aten.add.Tensor(mul_611, unsqueeze_1631);  mul_611 = unsqueeze_1631 = None
        relu_203 = torch.ops.aten.relu.default(add_471);  add_471 = None
        convolution_203 = torch.ops.aten.convolution.default(relu_203, arg465_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_203 = arg465_1 = None
        add_472 = torch.ops.aten.add.Tensor(arg467_1, 0.001);  arg467_1 = None
        sqrt_204 = torch.ops.aten.sqrt.default(add_472);  add_472 = None
        reciprocal_204 = torch.ops.aten.reciprocal.default(sqrt_204);  sqrt_204 = None
        mul_612 = torch.ops.aten.mul.Tensor(reciprocal_204, 1);  reciprocal_204 = None
        unsqueeze_1632 = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_1633 = torch.ops.aten.unsqueeze.default(unsqueeze_1632, -1);  unsqueeze_1632 = None
        unsqueeze_1634 = torch.ops.aten.unsqueeze.default(mul_612, -1);  mul_612 = None
        unsqueeze_1635 = torch.ops.aten.unsqueeze.default(unsqueeze_1634, -1);  unsqueeze_1634 = None
        sub_204 = torch.ops.aten.sub.Tensor(convolution_203, unsqueeze_1633);  convolution_203 = unsqueeze_1633 = None
        mul_613 = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_1635);  sub_204 = unsqueeze_1635 = None
        unsqueeze_1636 = torch.ops.aten.unsqueeze.default(arg468_1, -1);  arg468_1 = None
        unsqueeze_1637 = torch.ops.aten.unsqueeze.default(unsqueeze_1636, -1);  unsqueeze_1636 = None
        mul_614 = torch.ops.aten.mul.Tensor(mul_613, unsqueeze_1637);  mul_613 = unsqueeze_1637 = None
        unsqueeze_1638 = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1639 = torch.ops.aten.unsqueeze.default(unsqueeze_1638, -1);  unsqueeze_1638 = None
        add_473 = torch.ops.aten.add.Tensor(mul_614, unsqueeze_1639);  mul_614 = unsqueeze_1639 = None
        relu_204 = torch.ops.aten.relu.default(add_473);  add_473 = None
        convolution_204 = torch.ops.aten.convolution.default(relu_204, arg470_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_204 = arg470_1 = None
        slice_570 = torch.ops.aten.slice.Tensor(convolution_204, 1, 0, 1024)
        slice_574 = torch.ops.aten.slice.Tensor(convolution_204, 1, 1024, 9223372036854775807);  convolution_204 = None
        add_474 = torch.ops.aten.add.Tensor(add_467, slice_570);  add_467 = slice_570 = None
        cat_128 = torch.ops.aten.cat.default([cat_126, slice_574], 1);  cat_126 = slice_574 = None
        cat_129 = torch.ops.aten.cat.default([add_474, cat_128], 1)
        add_475 = torch.ops.aten.add.Tensor(arg472_1, 0.001);  arg472_1 = None
        sqrt_205 = torch.ops.aten.sqrt.default(add_475);  add_475 = None
        reciprocal_205 = torch.ops.aten.reciprocal.default(sqrt_205);  sqrt_205 = None
        mul_615 = torch.ops.aten.mul.Tensor(reciprocal_205, 1);  reciprocal_205 = None
        unsqueeze_1640 = torch.ops.aten.unsqueeze.default(arg471_1, -1);  arg471_1 = None
        unsqueeze_1641 = torch.ops.aten.unsqueeze.default(unsqueeze_1640, -1);  unsqueeze_1640 = None
        unsqueeze_1642 = torch.ops.aten.unsqueeze.default(mul_615, -1);  mul_615 = None
        unsqueeze_1643 = torch.ops.aten.unsqueeze.default(unsqueeze_1642, -1);  unsqueeze_1642 = None
        sub_205 = torch.ops.aten.sub.Tensor(cat_129, unsqueeze_1641);  cat_129 = unsqueeze_1641 = None
        mul_616 = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_1643);  sub_205 = unsqueeze_1643 = None
        unsqueeze_1644 = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_1645 = torch.ops.aten.unsqueeze.default(unsqueeze_1644, -1);  unsqueeze_1644 = None
        mul_617 = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_1645);  mul_616 = unsqueeze_1645 = None
        unsqueeze_1646 = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1647 = torch.ops.aten.unsqueeze.default(unsqueeze_1646, -1);  unsqueeze_1646 = None
        add_476 = torch.ops.aten.add.Tensor(mul_617, unsqueeze_1647);  mul_617 = unsqueeze_1647 = None
        relu_205 = torch.ops.aten.relu.default(add_476);  add_476 = None
        convolution_205 = torch.ops.aten.convolution.default(relu_205, arg475_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_205 = arg475_1 = None
        add_477 = torch.ops.aten.add.Tensor(arg477_1, 0.001);  arg477_1 = None
        sqrt_206 = torch.ops.aten.sqrt.default(add_477);  add_477 = None
        reciprocal_206 = torch.ops.aten.reciprocal.default(sqrt_206);  sqrt_206 = None
        mul_618 = torch.ops.aten.mul.Tensor(reciprocal_206, 1);  reciprocal_206 = None
        unsqueeze_1648 = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
        unsqueeze_1649 = torch.ops.aten.unsqueeze.default(unsqueeze_1648, -1);  unsqueeze_1648 = None
        unsqueeze_1650 = torch.ops.aten.unsqueeze.default(mul_618, -1);  mul_618 = None
        unsqueeze_1651 = torch.ops.aten.unsqueeze.default(unsqueeze_1650, -1);  unsqueeze_1650 = None
        sub_206 = torch.ops.aten.sub.Tensor(convolution_205, unsqueeze_1649);  convolution_205 = unsqueeze_1649 = None
        mul_619 = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_1651);  sub_206 = unsqueeze_1651 = None
        unsqueeze_1652 = torch.ops.aten.unsqueeze.default(arg478_1, -1);  arg478_1 = None
        unsqueeze_1653 = torch.ops.aten.unsqueeze.default(unsqueeze_1652, -1);  unsqueeze_1652 = None
        mul_620 = torch.ops.aten.mul.Tensor(mul_619, unsqueeze_1653);  mul_619 = unsqueeze_1653 = None
        unsqueeze_1654 = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_1655 = torch.ops.aten.unsqueeze.default(unsqueeze_1654, -1);  unsqueeze_1654 = None
        add_478 = torch.ops.aten.add.Tensor(mul_620, unsqueeze_1655);  mul_620 = unsqueeze_1655 = None
        relu_206 = torch.ops.aten.relu.default(add_478);  add_478 = None
        convolution_206 = torch.ops.aten.convolution.default(relu_206, arg480_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_206 = arg480_1 = None
        add_479 = torch.ops.aten.add.Tensor(arg482_1, 0.001);  arg482_1 = None
        sqrt_207 = torch.ops.aten.sqrt.default(add_479);  add_479 = None
        reciprocal_207 = torch.ops.aten.reciprocal.default(sqrt_207);  sqrt_207 = None
        mul_621 = torch.ops.aten.mul.Tensor(reciprocal_207, 1);  reciprocal_207 = None
        unsqueeze_1656 = torch.ops.aten.unsqueeze.default(arg481_1, -1);  arg481_1 = None
        unsqueeze_1657 = torch.ops.aten.unsqueeze.default(unsqueeze_1656, -1);  unsqueeze_1656 = None
        unsqueeze_1658 = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
        unsqueeze_1659 = torch.ops.aten.unsqueeze.default(unsqueeze_1658, -1);  unsqueeze_1658 = None
        sub_207 = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1657);  convolution_206 = unsqueeze_1657 = None
        mul_622 = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_1659);  sub_207 = unsqueeze_1659 = None
        unsqueeze_1660 = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
        unsqueeze_1661 = torch.ops.aten.unsqueeze.default(unsqueeze_1660, -1);  unsqueeze_1660 = None
        mul_623 = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_1661);  mul_622 = unsqueeze_1661 = None
        unsqueeze_1662 = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_1663 = torch.ops.aten.unsqueeze.default(unsqueeze_1662, -1);  unsqueeze_1662 = None
        add_480 = torch.ops.aten.add.Tensor(mul_623, unsqueeze_1663);  mul_623 = unsqueeze_1663 = None
        relu_207 = torch.ops.aten.relu.default(add_480);  add_480 = None
        convolution_207 = torch.ops.aten.convolution.default(relu_207, arg485_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_207 = arg485_1 = None
        slice_578 = torch.ops.aten.slice.Tensor(convolution_207, 1, 0, 1024)
        slice_582 = torch.ops.aten.slice.Tensor(convolution_207, 1, 1024, 9223372036854775807);  convolution_207 = None
        add_481 = torch.ops.aten.add.Tensor(add_474, slice_578);  add_474 = slice_578 = None
        cat_130 = torch.ops.aten.cat.default([cat_128, slice_582], 1);  cat_128 = slice_582 = None
        cat_131 = torch.ops.aten.cat.default([add_481, cat_130], 1)
        add_482 = torch.ops.aten.add.Tensor(arg487_1, 0.001);  arg487_1 = None
        sqrt_208 = torch.ops.aten.sqrt.default(add_482);  add_482 = None
        reciprocal_208 = torch.ops.aten.reciprocal.default(sqrt_208);  sqrt_208 = None
        mul_624 = torch.ops.aten.mul.Tensor(reciprocal_208, 1);  reciprocal_208 = None
        unsqueeze_1664 = torch.ops.aten.unsqueeze.default(arg486_1, -1);  arg486_1 = None
        unsqueeze_1665 = torch.ops.aten.unsqueeze.default(unsqueeze_1664, -1);  unsqueeze_1664 = None
        unsqueeze_1666 = torch.ops.aten.unsqueeze.default(mul_624, -1);  mul_624 = None
        unsqueeze_1667 = torch.ops.aten.unsqueeze.default(unsqueeze_1666, -1);  unsqueeze_1666 = None
        sub_208 = torch.ops.aten.sub.Tensor(cat_131, unsqueeze_1665);  cat_131 = unsqueeze_1665 = None
        mul_625 = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_1667);  sub_208 = unsqueeze_1667 = None
        unsqueeze_1668 = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
        unsqueeze_1669 = torch.ops.aten.unsqueeze.default(unsqueeze_1668, -1);  unsqueeze_1668 = None
        mul_626 = torch.ops.aten.mul.Tensor(mul_625, unsqueeze_1669);  mul_625 = unsqueeze_1669 = None
        unsqueeze_1670 = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_1671 = torch.ops.aten.unsqueeze.default(unsqueeze_1670, -1);  unsqueeze_1670 = None
        add_483 = torch.ops.aten.add.Tensor(mul_626, unsqueeze_1671);  mul_626 = unsqueeze_1671 = None
        relu_208 = torch.ops.aten.relu.default(add_483);  add_483 = None
        convolution_208 = torch.ops.aten.convolution.default(relu_208, arg490_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_208 = arg490_1 = None
        add_484 = torch.ops.aten.add.Tensor(arg492_1, 0.001);  arg492_1 = None
        sqrt_209 = torch.ops.aten.sqrt.default(add_484);  add_484 = None
        reciprocal_209 = torch.ops.aten.reciprocal.default(sqrt_209);  sqrt_209 = None
        mul_627 = torch.ops.aten.mul.Tensor(reciprocal_209, 1);  reciprocal_209 = None
        unsqueeze_1672 = torch.ops.aten.unsqueeze.default(arg491_1, -1);  arg491_1 = None
        unsqueeze_1673 = torch.ops.aten.unsqueeze.default(unsqueeze_1672, -1);  unsqueeze_1672 = None
        unsqueeze_1674 = torch.ops.aten.unsqueeze.default(mul_627, -1);  mul_627 = None
        unsqueeze_1675 = torch.ops.aten.unsqueeze.default(unsqueeze_1674, -1);  unsqueeze_1674 = None
        sub_209 = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1673);  convolution_208 = unsqueeze_1673 = None
        mul_628 = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_1675);  sub_209 = unsqueeze_1675 = None
        unsqueeze_1676 = torch.ops.aten.unsqueeze.default(arg493_1, -1);  arg493_1 = None
        unsqueeze_1677 = torch.ops.aten.unsqueeze.default(unsqueeze_1676, -1);  unsqueeze_1676 = None
        mul_629 = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_1677);  mul_628 = unsqueeze_1677 = None
        unsqueeze_1678 = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_1679 = torch.ops.aten.unsqueeze.default(unsqueeze_1678, -1);  unsqueeze_1678 = None
        add_485 = torch.ops.aten.add.Tensor(mul_629, unsqueeze_1679);  mul_629 = unsqueeze_1679 = None
        relu_209 = torch.ops.aten.relu.default(add_485);  add_485 = None
        convolution_209 = torch.ops.aten.convolution.default(relu_209, arg495_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_209 = arg495_1 = None
        add_486 = torch.ops.aten.add.Tensor(arg497_1, 0.001);  arg497_1 = None
        sqrt_210 = torch.ops.aten.sqrt.default(add_486);  add_486 = None
        reciprocal_210 = torch.ops.aten.reciprocal.default(sqrt_210);  sqrt_210 = None
        mul_630 = torch.ops.aten.mul.Tensor(reciprocal_210, 1);  reciprocal_210 = None
        unsqueeze_1680 = torch.ops.aten.unsqueeze.default(arg496_1, -1);  arg496_1 = None
        unsqueeze_1681 = torch.ops.aten.unsqueeze.default(unsqueeze_1680, -1);  unsqueeze_1680 = None
        unsqueeze_1682 = torch.ops.aten.unsqueeze.default(mul_630, -1);  mul_630 = None
        unsqueeze_1683 = torch.ops.aten.unsqueeze.default(unsqueeze_1682, -1);  unsqueeze_1682 = None
        sub_210 = torch.ops.aten.sub.Tensor(convolution_209, unsqueeze_1681);  convolution_209 = unsqueeze_1681 = None
        mul_631 = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_1683);  sub_210 = unsqueeze_1683 = None
        unsqueeze_1684 = torch.ops.aten.unsqueeze.default(arg498_1, -1);  arg498_1 = None
        unsqueeze_1685 = torch.ops.aten.unsqueeze.default(unsqueeze_1684, -1);  unsqueeze_1684 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_1685);  mul_631 = unsqueeze_1685 = None
        unsqueeze_1686 = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1687 = torch.ops.aten.unsqueeze.default(unsqueeze_1686, -1);  unsqueeze_1686 = None
        add_487 = torch.ops.aten.add.Tensor(mul_632, unsqueeze_1687);  mul_632 = unsqueeze_1687 = None
        relu_210 = torch.ops.aten.relu.default(add_487);  add_487 = None
        convolution_210 = torch.ops.aten.convolution.default(relu_210, arg500_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_210 = arg500_1 = None
        slice_586 = torch.ops.aten.slice.Tensor(convolution_210, 1, 0, 1024)
        slice_590 = torch.ops.aten.slice.Tensor(convolution_210, 1, 1024, 9223372036854775807);  convolution_210 = None
        add_488 = torch.ops.aten.add.Tensor(add_481, slice_586);  add_481 = slice_586 = None
        cat_132 = torch.ops.aten.cat.default([cat_130, slice_590], 1);  cat_130 = slice_590 = None
        cat_133 = torch.ops.aten.cat.default([add_488, cat_132], 1);  add_488 = cat_132 = None
        add_489 = torch.ops.aten.add.Tensor(arg502_1, 0.001);  arg502_1 = None
        sqrt_211 = torch.ops.aten.sqrt.default(add_489);  add_489 = None
        reciprocal_211 = torch.ops.aten.reciprocal.default(sqrt_211);  sqrt_211 = None
        mul_633 = torch.ops.aten.mul.Tensor(reciprocal_211, 1);  reciprocal_211 = None
        unsqueeze_1688 = torch.ops.aten.unsqueeze.default(arg501_1, -1);  arg501_1 = None
        unsqueeze_1689 = torch.ops.aten.unsqueeze.default(unsqueeze_1688, -1);  unsqueeze_1688 = None
        unsqueeze_1690 = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1691 = torch.ops.aten.unsqueeze.default(unsqueeze_1690, -1);  unsqueeze_1690 = None
        sub_211 = torch.ops.aten.sub.Tensor(cat_133, unsqueeze_1689);  unsqueeze_1689 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_1691);  sub_211 = unsqueeze_1691 = None
        unsqueeze_1692 = torch.ops.aten.unsqueeze.default(arg503_1, -1);  arg503_1 = None
        unsqueeze_1693 = torch.ops.aten.unsqueeze.default(unsqueeze_1692, -1);  unsqueeze_1692 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1693);  mul_634 = unsqueeze_1693 = None
        unsqueeze_1694 = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_1695 = torch.ops.aten.unsqueeze.default(unsqueeze_1694, -1);  unsqueeze_1694 = None
        add_490 = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1695);  mul_635 = unsqueeze_1695 = None
        relu_211 = torch.ops.aten.relu.default(add_490);  add_490 = None
        convolution_211 = torch.ops.aten.convolution.default(relu_211, arg505_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_211 = arg505_1 = None
        slice_594 = torch.ops.aten.slice.Tensor(convolution_211, 1, 0, 2048)
        slice_598 = torch.ops.aten.slice.Tensor(convolution_211, 1, 2048, 9223372036854775807);  convolution_211 = None
        add_491 = torch.ops.aten.add.Tensor(arg507_1, 0.001);  arg507_1 = None
        sqrt_212 = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_212 = torch.ops.aten.reciprocal.default(sqrt_212);  sqrt_212 = None
        mul_636 = torch.ops.aten.mul.Tensor(reciprocal_212, 1);  reciprocal_212 = None
        unsqueeze_1696 = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_1697 = torch.ops.aten.unsqueeze.default(unsqueeze_1696, -1);  unsqueeze_1696 = None
        unsqueeze_1698 = torch.ops.aten.unsqueeze.default(mul_636, -1);  mul_636 = None
        unsqueeze_1699 = torch.ops.aten.unsqueeze.default(unsqueeze_1698, -1);  unsqueeze_1698 = None
        sub_212 = torch.ops.aten.sub.Tensor(cat_133, unsqueeze_1697);  cat_133 = unsqueeze_1697 = None
        mul_637 = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_1699);  sub_212 = unsqueeze_1699 = None
        unsqueeze_1700 = torch.ops.aten.unsqueeze.default(arg508_1, -1);  arg508_1 = None
        unsqueeze_1701 = torch.ops.aten.unsqueeze.default(unsqueeze_1700, -1);  unsqueeze_1700 = None
        mul_638 = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_1701);  mul_637 = unsqueeze_1701 = None
        unsqueeze_1702 = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_1703 = torch.ops.aten.unsqueeze.default(unsqueeze_1702, -1);  unsqueeze_1702 = None
        add_492 = torch.ops.aten.add.Tensor(mul_638, unsqueeze_1703);  mul_638 = unsqueeze_1703 = None
        relu_212 = torch.ops.aten.relu.default(add_492);  add_492 = None
        convolution_212 = torch.ops.aten.convolution.default(relu_212, arg510_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_212 = arg510_1 = None
        add_493 = torch.ops.aten.add.Tensor(arg512_1, 0.001);  arg512_1 = None
        sqrt_213 = torch.ops.aten.sqrt.default(add_493);  add_493 = None
        reciprocal_213 = torch.ops.aten.reciprocal.default(sqrt_213);  sqrt_213 = None
        mul_639 = torch.ops.aten.mul.Tensor(reciprocal_213, 1);  reciprocal_213 = None
        unsqueeze_1704 = torch.ops.aten.unsqueeze.default(arg511_1, -1);  arg511_1 = None
        unsqueeze_1705 = torch.ops.aten.unsqueeze.default(unsqueeze_1704, -1);  unsqueeze_1704 = None
        unsqueeze_1706 = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1707 = torch.ops.aten.unsqueeze.default(unsqueeze_1706, -1);  unsqueeze_1706 = None
        sub_213 = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1705);  convolution_212 = unsqueeze_1705 = None
        mul_640 = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_1707);  sub_213 = unsqueeze_1707 = None
        unsqueeze_1708 = torch.ops.aten.unsqueeze.default(arg513_1, -1);  arg513_1 = None
        unsqueeze_1709 = torch.ops.aten.unsqueeze.default(unsqueeze_1708, -1);  unsqueeze_1708 = None
        mul_641 = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1709);  mul_640 = unsqueeze_1709 = None
        unsqueeze_1710 = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_1711 = torch.ops.aten.unsqueeze.default(unsqueeze_1710, -1);  unsqueeze_1710 = None
        add_494 = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1711);  mul_641 = unsqueeze_1711 = None
        relu_213 = torch.ops.aten.relu.default(add_494);  add_494 = None
        convolution_213 = torch.ops.aten.convolution.default(relu_213, arg515_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_213 = arg515_1 = None
        add_495 = torch.ops.aten.add.Tensor(arg517_1, 0.001);  arg517_1 = None
        sqrt_214 = torch.ops.aten.sqrt.default(add_495);  add_495 = None
        reciprocal_214 = torch.ops.aten.reciprocal.default(sqrt_214);  sqrt_214 = None
        mul_642 = torch.ops.aten.mul.Tensor(reciprocal_214, 1);  reciprocal_214 = None
        unsqueeze_1712 = torch.ops.aten.unsqueeze.default(arg516_1, -1);  arg516_1 = None
        unsqueeze_1713 = torch.ops.aten.unsqueeze.default(unsqueeze_1712, -1);  unsqueeze_1712 = None
        unsqueeze_1714 = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1715 = torch.ops.aten.unsqueeze.default(unsqueeze_1714, -1);  unsqueeze_1714 = None
        sub_214 = torch.ops.aten.sub.Tensor(convolution_213, unsqueeze_1713);  convolution_213 = unsqueeze_1713 = None
        mul_643 = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_1715);  sub_214 = unsqueeze_1715 = None
        unsqueeze_1716 = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
        unsqueeze_1717 = torch.ops.aten.unsqueeze.default(unsqueeze_1716, -1);  unsqueeze_1716 = None
        mul_644 = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1717);  mul_643 = unsqueeze_1717 = None
        unsqueeze_1718 = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_1719 = torch.ops.aten.unsqueeze.default(unsqueeze_1718, -1);  unsqueeze_1718 = None
        add_496 = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1719);  mul_644 = unsqueeze_1719 = None
        relu_214 = torch.ops.aten.relu.default(add_496);  add_496 = None
        convolution_214 = torch.ops.aten.convolution.default(relu_214, arg520_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_214 = arg520_1 = None
        slice_602 = torch.ops.aten.slice.Tensor(convolution_214, 1, 0, 2048)
        slice_606 = torch.ops.aten.slice.Tensor(convolution_214, 1, 2048, 9223372036854775807);  convolution_214 = None
        add_497 = torch.ops.aten.add.Tensor(slice_594, slice_602);  slice_594 = slice_602 = None
        cat_134 = torch.ops.aten.cat.default([slice_598, slice_606], 1);  slice_598 = slice_606 = None
        cat_135 = torch.ops.aten.cat.default([add_497, cat_134], 1)
        add_498 = torch.ops.aten.add.Tensor(arg522_1, 0.001);  arg522_1 = None
        sqrt_215 = torch.ops.aten.sqrt.default(add_498);  add_498 = None
        reciprocal_215 = torch.ops.aten.reciprocal.default(sqrt_215);  sqrt_215 = None
        mul_645 = torch.ops.aten.mul.Tensor(reciprocal_215, 1);  reciprocal_215 = None
        unsqueeze_1720 = torch.ops.aten.unsqueeze.default(arg521_1, -1);  arg521_1 = None
        unsqueeze_1721 = torch.ops.aten.unsqueeze.default(unsqueeze_1720, -1);  unsqueeze_1720 = None
        unsqueeze_1722 = torch.ops.aten.unsqueeze.default(mul_645, -1);  mul_645 = None
        unsqueeze_1723 = torch.ops.aten.unsqueeze.default(unsqueeze_1722, -1);  unsqueeze_1722 = None
        sub_215 = torch.ops.aten.sub.Tensor(cat_135, unsqueeze_1721);  cat_135 = unsqueeze_1721 = None
        mul_646 = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_1723);  sub_215 = unsqueeze_1723 = None
        unsqueeze_1724 = torch.ops.aten.unsqueeze.default(arg523_1, -1);  arg523_1 = None
        unsqueeze_1725 = torch.ops.aten.unsqueeze.default(unsqueeze_1724, -1);  unsqueeze_1724 = None
        mul_647 = torch.ops.aten.mul.Tensor(mul_646, unsqueeze_1725);  mul_646 = unsqueeze_1725 = None
        unsqueeze_1726 = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_1727 = torch.ops.aten.unsqueeze.default(unsqueeze_1726, -1);  unsqueeze_1726 = None
        add_499 = torch.ops.aten.add.Tensor(mul_647, unsqueeze_1727);  mul_647 = unsqueeze_1727 = None
        relu_215 = torch.ops.aten.relu.default(add_499);  add_499 = None
        convolution_215 = torch.ops.aten.convolution.default(relu_215, arg525_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_215 = arg525_1 = None
        add_500 = torch.ops.aten.add.Tensor(arg527_1, 0.001);  arg527_1 = None
        sqrt_216 = torch.ops.aten.sqrt.default(add_500);  add_500 = None
        reciprocal_216 = torch.ops.aten.reciprocal.default(sqrt_216);  sqrt_216 = None
        mul_648 = torch.ops.aten.mul.Tensor(reciprocal_216, 1);  reciprocal_216 = None
        unsqueeze_1728 = torch.ops.aten.unsqueeze.default(arg526_1, -1);  arg526_1 = None
        unsqueeze_1729 = torch.ops.aten.unsqueeze.default(unsqueeze_1728, -1);  unsqueeze_1728 = None
        unsqueeze_1730 = torch.ops.aten.unsqueeze.default(mul_648, -1);  mul_648 = None
        unsqueeze_1731 = torch.ops.aten.unsqueeze.default(unsqueeze_1730, -1);  unsqueeze_1730 = None
        sub_216 = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1729);  convolution_215 = unsqueeze_1729 = None
        mul_649 = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_1731);  sub_216 = unsqueeze_1731 = None
        unsqueeze_1732 = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
        unsqueeze_1733 = torch.ops.aten.unsqueeze.default(unsqueeze_1732, -1);  unsqueeze_1732 = None
        mul_650 = torch.ops.aten.mul.Tensor(mul_649, unsqueeze_1733);  mul_649 = unsqueeze_1733 = None
        unsqueeze_1734 = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_1735 = torch.ops.aten.unsqueeze.default(unsqueeze_1734, -1);  unsqueeze_1734 = None
        add_501 = torch.ops.aten.add.Tensor(mul_650, unsqueeze_1735);  mul_650 = unsqueeze_1735 = None
        relu_216 = torch.ops.aten.relu.default(add_501);  add_501 = None
        convolution_216 = torch.ops.aten.convolution.default(relu_216, arg530_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_216 = arg530_1 = None
        add_502 = torch.ops.aten.add.Tensor(arg532_1, 0.001);  arg532_1 = None
        sqrt_217 = torch.ops.aten.sqrt.default(add_502);  add_502 = None
        reciprocal_217 = torch.ops.aten.reciprocal.default(sqrt_217);  sqrt_217 = None
        mul_651 = torch.ops.aten.mul.Tensor(reciprocal_217, 1);  reciprocal_217 = None
        unsqueeze_1736 = torch.ops.aten.unsqueeze.default(arg531_1, -1);  arg531_1 = None
        unsqueeze_1737 = torch.ops.aten.unsqueeze.default(unsqueeze_1736, -1);  unsqueeze_1736 = None
        unsqueeze_1738 = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
        unsqueeze_1739 = torch.ops.aten.unsqueeze.default(unsqueeze_1738, -1);  unsqueeze_1738 = None
        sub_217 = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1737);  convolution_216 = unsqueeze_1737 = None
        mul_652 = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_1739);  sub_217 = unsqueeze_1739 = None
        unsqueeze_1740 = torch.ops.aten.unsqueeze.default(arg533_1, -1);  arg533_1 = None
        unsqueeze_1741 = torch.ops.aten.unsqueeze.default(unsqueeze_1740, -1);  unsqueeze_1740 = None
        mul_653 = torch.ops.aten.mul.Tensor(mul_652, unsqueeze_1741);  mul_652 = unsqueeze_1741 = None
        unsqueeze_1742 = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_1743 = torch.ops.aten.unsqueeze.default(unsqueeze_1742, -1);  unsqueeze_1742 = None
        add_503 = torch.ops.aten.add.Tensor(mul_653, unsqueeze_1743);  mul_653 = unsqueeze_1743 = None
        relu_217 = torch.ops.aten.relu.default(add_503);  add_503 = None
        convolution_217 = torch.ops.aten.convolution.default(relu_217, arg535_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_217 = arg535_1 = None
        slice_610 = torch.ops.aten.slice.Tensor(convolution_217, 1, 0, 2048)
        slice_614 = torch.ops.aten.slice.Tensor(convolution_217, 1, 2048, 9223372036854775807);  convolution_217 = None
        add_504 = torch.ops.aten.add.Tensor(add_497, slice_610);  add_497 = slice_610 = None
        cat_136 = torch.ops.aten.cat.default([cat_134, slice_614], 1);  cat_134 = slice_614 = None
        cat_137 = torch.ops.aten.cat.default([add_504, cat_136], 1)
        add_505 = torch.ops.aten.add.Tensor(arg537_1, 0.001);  arg537_1 = None
        sqrt_218 = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_218 = torch.ops.aten.reciprocal.default(sqrt_218);  sqrt_218 = None
        mul_654 = torch.ops.aten.mul.Tensor(reciprocal_218, 1);  reciprocal_218 = None
        unsqueeze_1744 = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
        unsqueeze_1745 = torch.ops.aten.unsqueeze.default(unsqueeze_1744, -1);  unsqueeze_1744 = None
        unsqueeze_1746 = torch.ops.aten.unsqueeze.default(mul_654, -1);  mul_654 = None
        unsqueeze_1747 = torch.ops.aten.unsqueeze.default(unsqueeze_1746, -1);  unsqueeze_1746 = None
        sub_218 = torch.ops.aten.sub.Tensor(cat_137, unsqueeze_1745);  cat_137 = unsqueeze_1745 = None
        mul_655 = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_1747);  sub_218 = unsqueeze_1747 = None
        unsqueeze_1748 = torch.ops.aten.unsqueeze.default(arg538_1, -1);  arg538_1 = None
        unsqueeze_1749 = torch.ops.aten.unsqueeze.default(unsqueeze_1748, -1);  unsqueeze_1748 = None
        mul_656 = torch.ops.aten.mul.Tensor(mul_655, unsqueeze_1749);  mul_655 = unsqueeze_1749 = None
        unsqueeze_1750 = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
        unsqueeze_1751 = torch.ops.aten.unsqueeze.default(unsqueeze_1750, -1);  unsqueeze_1750 = None
        add_506 = torch.ops.aten.add.Tensor(mul_656, unsqueeze_1751);  mul_656 = unsqueeze_1751 = None
        relu_218 = torch.ops.aten.relu.default(add_506);  add_506 = None
        convolution_218 = torch.ops.aten.convolution.default(relu_218, arg540_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_218 = arg540_1 = None
        add_507 = torch.ops.aten.add.Tensor(arg542_1, 0.001);  arg542_1 = None
        sqrt_219 = torch.ops.aten.sqrt.default(add_507);  add_507 = None
        reciprocal_219 = torch.ops.aten.reciprocal.default(sqrt_219);  sqrt_219 = None
        mul_657 = torch.ops.aten.mul.Tensor(reciprocal_219, 1);  reciprocal_219 = None
        unsqueeze_1752 = torch.ops.aten.unsqueeze.default(arg541_1, -1);  arg541_1 = None
        unsqueeze_1753 = torch.ops.aten.unsqueeze.default(unsqueeze_1752, -1);  unsqueeze_1752 = None
        unsqueeze_1754 = torch.ops.aten.unsqueeze.default(mul_657, -1);  mul_657 = None
        unsqueeze_1755 = torch.ops.aten.unsqueeze.default(unsqueeze_1754, -1);  unsqueeze_1754 = None
        sub_219 = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1753);  convolution_218 = unsqueeze_1753 = None
        mul_658 = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_1755);  sub_219 = unsqueeze_1755 = None
        unsqueeze_1756 = torch.ops.aten.unsqueeze.default(arg543_1, -1);  arg543_1 = None
        unsqueeze_1757 = torch.ops.aten.unsqueeze.default(unsqueeze_1756, -1);  unsqueeze_1756 = None
        mul_659 = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_1757);  mul_658 = unsqueeze_1757 = None
        unsqueeze_1758 = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
        unsqueeze_1759 = torch.ops.aten.unsqueeze.default(unsqueeze_1758, -1);  unsqueeze_1758 = None
        add_508 = torch.ops.aten.add.Tensor(mul_659, unsqueeze_1759);  mul_659 = unsqueeze_1759 = None
        relu_219 = torch.ops.aten.relu.default(add_508);  add_508 = None
        convolution_219 = torch.ops.aten.convolution.default(relu_219, arg545_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_219 = arg545_1 = None
        add_509 = torch.ops.aten.add.Tensor(arg547_1, 0.001);  arg547_1 = None
        sqrt_220 = torch.ops.aten.sqrt.default(add_509);  add_509 = None
        reciprocal_220 = torch.ops.aten.reciprocal.default(sqrt_220);  sqrt_220 = None
        mul_660 = torch.ops.aten.mul.Tensor(reciprocal_220, 1);  reciprocal_220 = None
        unsqueeze_1760 = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
        unsqueeze_1761 = torch.ops.aten.unsqueeze.default(unsqueeze_1760, -1);  unsqueeze_1760 = None
        unsqueeze_1762 = torch.ops.aten.unsqueeze.default(mul_660, -1);  mul_660 = None
        unsqueeze_1763 = torch.ops.aten.unsqueeze.default(unsqueeze_1762, -1);  unsqueeze_1762 = None
        sub_220 = torch.ops.aten.sub.Tensor(convolution_219, unsqueeze_1761);  convolution_219 = unsqueeze_1761 = None
        mul_661 = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_1763);  sub_220 = unsqueeze_1763 = None
        unsqueeze_1764 = torch.ops.aten.unsqueeze.default(arg548_1, -1);  arg548_1 = None
        unsqueeze_1765 = torch.ops.aten.unsqueeze.default(unsqueeze_1764, -1);  unsqueeze_1764 = None
        mul_662 = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_1765);  mul_661 = unsqueeze_1765 = None
        unsqueeze_1766 = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_1767 = torch.ops.aten.unsqueeze.default(unsqueeze_1766, -1);  unsqueeze_1766 = None
        add_510 = torch.ops.aten.add.Tensor(mul_662, unsqueeze_1767);  mul_662 = unsqueeze_1767 = None
        relu_220 = torch.ops.aten.relu.default(add_510);  add_510 = None
        convolution_220 = torch.ops.aten.convolution.default(relu_220, arg550_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_220 = arg550_1 = None
        slice_618 = torch.ops.aten.slice.Tensor(convolution_220, 1, 0, 2048)
        slice_622 = torch.ops.aten.slice.Tensor(convolution_220, 1, 2048, 9223372036854775807);  convolution_220 = None
        add_511 = torch.ops.aten.add.Tensor(add_504, slice_618);  add_504 = slice_618 = None
        cat_138 = torch.ops.aten.cat.default([cat_136, slice_622], 1);  cat_136 = slice_622 = None
        cat_139 = torch.ops.aten.cat.default([add_511, cat_138], 1);  add_511 = cat_138 = None
        add_512 = torch.ops.aten.add.Tensor(arg552_1, 0.001);  arg552_1 = None
        sqrt_221 = torch.ops.aten.sqrt.default(add_512);  add_512 = None
        reciprocal_221 = torch.ops.aten.reciprocal.default(sqrt_221);  sqrt_221 = None
        mul_663 = torch.ops.aten.mul.Tensor(reciprocal_221, 1);  reciprocal_221 = None
        unsqueeze_1768 = torch.ops.aten.unsqueeze.default(arg551_1, -1);  arg551_1 = None
        unsqueeze_1769 = torch.ops.aten.unsqueeze.default(unsqueeze_1768, -1);  unsqueeze_1768 = None
        unsqueeze_1770 = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_1771 = torch.ops.aten.unsqueeze.default(unsqueeze_1770, -1);  unsqueeze_1770 = None
        sub_221 = torch.ops.aten.sub.Tensor(cat_139, unsqueeze_1769);  cat_139 = unsqueeze_1769 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_1771);  sub_221 = unsqueeze_1771 = None
        unsqueeze_1772 = torch.ops.aten.unsqueeze.default(arg553_1, -1);  arg553_1 = None
        unsqueeze_1773 = torch.ops.aten.unsqueeze.default(unsqueeze_1772, -1);  unsqueeze_1772 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_1773);  mul_664 = unsqueeze_1773 = None
        unsqueeze_1774 = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_1775 = torch.ops.aten.unsqueeze.default(unsqueeze_1774, -1);  unsqueeze_1774 = None
        add_513 = torch.ops.aten.add.Tensor(mul_665, unsqueeze_1775);  mul_665 = unsqueeze_1775 = None
        relu_221 = torch.ops.aten.relu.default(add_513);  add_513 = None
        mean_1 = torch.ops.aten.mean.dim(relu_221, [-1, -2], True);  relu_221 = None
        convolution_221 = torch.ops.aten.convolution.default(mean_1, arg555_1, arg556_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg555_1 = arg556_1 = None
        view_1 = torch.ops.aten.view.default(convolution_221, [8, 1000]);  convolution_221 = None
        return (view_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 3, 7, 7), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 151552, device=device(type='cuda', index=0))
    reader.tensor(buf10, (296, 128, 1, 1), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 102400, device=device(type='cuda', index=0))
    reader.tensor(buf15, (200, 128, 1, 1), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf16, (200,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf17, (200,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf18, (200,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf19, (200,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf20, (200, 4, 3, 3), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf21, (200,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf22, (200,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf23, (200,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf24, (200,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 220800, device=device(type='cuda', index=0))
    reader.tensor(buf25, (276, 200, 1, 1), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1264, device=device(type='cuda', index=0))
    reader.tensor(buf26, (316,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1264, device=device(type='cuda', index=0))
    reader.tensor(buf27, (316,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1264, device=device(type='cuda', index=0))
    reader.tensor(buf28, (316,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1264, device=device(type='cuda', index=0))
    reader.tensor(buf29, (316,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 252800, device=device(type='cuda', index=0))
    reader.tensor(buf30, (200, 316, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf31, (200,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf32, (200,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf33, (200,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf34, (200,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf35, (200, 4, 3, 3), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf36, (200,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf37, (200,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf38, (200,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf39, (200,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 220800, device=device(type='cuda', index=0))
    reader.tensor(buf40, (276, 200, 1, 1), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf41, (336,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf42, (336,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf43, (336,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf44, (336,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 268800, device=device(type='cuda', index=0))
    reader.tensor(buf45, (200, 336, 1, 1), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf46, (200,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf47, (200,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf48, (200,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf49, (200,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf50, (200, 4, 3, 3), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf51, (200,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf52, (200,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf53, (200,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf54, (200,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 220800, device=device(type='cuda', index=0))
    reader.tensor(buf55, (276, 200, 1, 1), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1424, device=device(type='cuda', index=0))
    reader.tensor(buf56, (356,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1424, device=device(type='cuda', index=0))
    reader.tensor(buf57, (356,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1424, device=device(type='cuda', index=0))
    reader.tensor(buf58, (356,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1424, device=device(type='cuda', index=0))
    reader.tensor(buf59, (356,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 284800, device=device(type='cuda', index=0))
    reader.tensor(buf60, (200, 356, 1, 1), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf61, (200,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf62, (200,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf63, (200,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf64, (200,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf65, (200, 4, 3, 3), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf66, (200,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf67, (200,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf68, (200,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf69, (200,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 220800, device=device(type='cuda', index=0))
    reader.tensor(buf70, (276, 200, 1, 1), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf71, (376,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf72, (376,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf73, (376,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf74, (376,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 962560, device=device(type='cuda', index=0))
    reader.tensor(buf75, (640, 376, 1, 1), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf76, (376,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf77, (376,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf78, (376,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 1504, device=device(type='cuda', index=0))
    reader.tensor(buf79, (376,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 601600, device=device(type='cuda', index=0))
    reader.tensor(buf80, (400, 376, 1, 1), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf81, (400,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf82, (400,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf83, (400,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf84, (400,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf85, (400, 8, 3, 3), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf86, (400,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf87, (400,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf88, (400,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf89, (400,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf90, (576, 400, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2816, device=device(type='cuda', index=0))
    reader.tensor(buf91, (704,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2816, device=device(type='cuda', index=0))
    reader.tensor(buf92, (704,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2816, device=device(type='cuda', index=0))
    reader.tensor(buf93, (704,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 2816, device=device(type='cuda', index=0))
    reader.tensor(buf94, (704,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1126400, device=device(type='cuda', index=0))
    reader.tensor(buf95, (400, 704, 1, 1), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf96, (400,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf97, (400,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf98, (400,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf99, (400,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf100, (400, 8, 3, 3), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf101, (400,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf102, (400,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf103, (400,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf104, (400,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf105, (576, 400, 1, 1), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf110, (400, 768, 1, 1), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf111, (400,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf112, (400,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf113, (400,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf114, (400,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf115, (400, 8, 3, 3), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf116, (400,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf117, (400,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf118, (400,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf119, (400,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf120, (576, 400, 1, 1), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3328, device=device(type='cuda', index=0))
    reader.tensor(buf121, (832,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3328, device=device(type='cuda', index=0))
    reader.tensor(buf122, (832,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3328, device=device(type='cuda', index=0))
    reader.tensor(buf123, (832,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3328, device=device(type='cuda', index=0))
    reader.tensor(buf124, (832,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1331200, device=device(type='cuda', index=0))
    reader.tensor(buf125, (400, 832, 1, 1), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf126, (400,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf127, (400,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf128, (400,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf129, (400,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf130, (400, 8, 3, 3), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf131, (400,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf132, (400,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf133, (400,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf134, (400,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf135, (576, 400, 1, 1), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3584, device=device(type='cuda', index=0))
    reader.tensor(buf136, (896,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3584, device=device(type='cuda', index=0))
    reader.tensor(buf137, (896,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3584, device=device(type='cuda', index=0))
    reader.tensor(buf138, (896,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3584, device=device(type='cuda', index=0))
    reader.tensor(buf139, (896,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1433600, device=device(type='cuda', index=0))
    reader.tensor(buf140, (400, 896, 1, 1), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf141, (400,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf142, (400,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf143, (400,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf144, (400,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf145, (400, 8, 3, 3), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf146, (400,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf147, (400,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf148, (400,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf149, (400,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf150, (576, 400, 1, 1), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf151, (960,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf152, (960,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf153, (960,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf154, (960,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1536000, device=device(type='cuda', index=0))
    reader.tensor(buf155, (400, 960, 1, 1), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf156, (400,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf157, (400,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf158, (400,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf159, (400,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf160, (400, 8, 3, 3), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf161, (400,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf162, (400,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf163, (400,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf164, (400,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf165, (576, 400, 1, 1), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1024,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1024,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1024,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1024,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf170, (400, 1024, 1, 1), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf171, (400,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf172, (400,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf173, (400,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf174, (400,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf175, (400, 8, 3, 3), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf176, (400,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf177, (400,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf178, (400,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf179, (400,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf180, (576, 400, 1, 1), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 4352, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1088,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 4352, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1088,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 4352, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1088,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 4352, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1088,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1740800, device=device(type='cuda', index=0))
    reader.tensor(buf185, (400, 1088, 1, 1), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf186, (400,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf187, (400,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf188, (400,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf189, (400,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf190, (400, 8, 3, 3), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf191, (400,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf192, (400,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf193, (400,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf194, (400,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf195, (576, 400, 1, 1), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1152,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1152,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1152,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1152,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 5308416, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1152, 1152, 1, 1), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1152,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1152,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1152,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1152,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 3686400, device=device(type='cuda', index=0))
    reader.tensor(buf205, (800, 1152, 1, 1), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf206, (800,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf207, (800,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf208, (800,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf209, (800,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf210, (800, 16, 3, 3), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf211, (800,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf212, (800,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf213, (800,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf214, (800,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1088, 800, 1, 1), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 4864, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1216,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 4864, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1216,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 4864, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1216,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 4864, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1216,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 3891200, device=device(type='cuda', index=0))
    reader.tensor(buf220, (800, 1216, 1, 1), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf221, (800,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf222, (800,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf223, (800,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf224, (800,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf225, (800, 16, 3, 3), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf226, (800,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf227, (800,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf228, (800,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf229, (800,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf230, (1088, 800, 1, 1), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1280,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1280,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1280,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf234, (1280,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf235, (800, 1280, 1, 1), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf236, (800,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf237, (800,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf238, (800,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf239, (800,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf240, (800, 16, 3, 3), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf241, (800,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf242, (800,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf243, (800,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf244, (800,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf245, (1088, 800, 1, 1), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf246, (1344,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf247, (1344,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf248, (1344,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 5376, device=device(type='cuda', index=0))
    reader.tensor(buf249, (1344,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 4300800, device=device(type='cuda', index=0))
    reader.tensor(buf250, (800, 1344, 1, 1), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf251, (800,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf252, (800,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf253, (800,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf254, (800,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf255, (800, 16, 3, 3), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf256, (800,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf257, (800,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf258, (800,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf259, (800,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1088, 800, 1, 1), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 5632, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1408,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 5632, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1408,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 5632, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1408,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 5632, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1408,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 4505600, device=device(type='cuda', index=0))
    reader.tensor(buf265, (800, 1408, 1, 1), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf266, (800,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf267, (800,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf268, (800,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf269, (800,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf270, (800, 16, 3, 3), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf271, (800,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf272, (800,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf273, (800,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf274, (800,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1088, 800, 1, 1), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 5888, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1472,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 5888, device=device(type='cuda', index=0))
    reader.tensor(buf277, (1472,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 5888, device=device(type='cuda', index=0))
    reader.tensor(buf278, (1472,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 5888, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1472,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4710400, device=device(type='cuda', index=0))
    reader.tensor(buf280, (800, 1472, 1, 1), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf281, (800,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf282, (800,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf283, (800,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf284, (800,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf285, (800, 16, 3, 3), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf286, (800,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf287, (800,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf288, (800,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf289, (800,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1088, 800, 1, 1), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1536,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1536,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1536,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1536,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4915200, device=device(type='cuda', index=0))
    reader.tensor(buf295, (800, 1536, 1, 1), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf296, (800,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf297, (800,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf298, (800,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf299, (800,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf300, (800, 16, 3, 3), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf301, (800,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf302, (800,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf303, (800,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf304, (800,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1088, 800, 1, 1), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1600,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf307, (1600,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf308, (1600,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1600,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf310, (800, 1600, 1, 1), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf311, (800,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf312, (800,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf313, (800,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf314, (800,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf315, (800, 16, 3, 3), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf316, (800,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf317, (800,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf318, (800,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf319, (800,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf320, (1088, 800, 1, 1), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 6656, device=device(type='cuda', index=0))
    reader.tensor(buf321, (1664,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 6656, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1664,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 6656, device=device(type='cuda', index=0))
    reader.tensor(buf323, (1664,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 6656, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1664,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 5324800, device=device(type='cuda', index=0))
    reader.tensor(buf325, (800, 1664, 1, 1), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf326, (800,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf327, (800,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf328, (800,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf329, (800,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf330, (800, 16, 3, 3), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf331, (800,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf332, (800,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf333, (800,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf334, (800,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf335, (1088, 800, 1, 1), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf336, (1728,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf337, (1728,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf338, (1728,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1728,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 5529600, device=device(type='cuda', index=0))
    reader.tensor(buf340, (800, 1728, 1, 1), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf341, (800,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf342, (800,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf343, (800,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf344, (800,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf345, (800, 16, 3, 3), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf346, (800,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf347, (800,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf348, (800,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf349, (800,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1088, 800, 1, 1), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 7168, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1792,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 7168, device=device(type='cuda', index=0))
    reader.tensor(buf352, (1792,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 7168, device=device(type='cuda', index=0))
    reader.tensor(buf353, (1792,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 7168, device=device(type='cuda', index=0))
    reader.tensor(buf354, (1792,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 5734400, device=device(type='cuda', index=0))
    reader.tensor(buf355, (800, 1792, 1, 1), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf356, (800,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf357, (800,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf358, (800,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf359, (800,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf360, (800, 16, 3, 3), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf361, (800,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf362, (800,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf363, (800,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf364, (800,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf365, (1088, 800, 1, 1), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 7424, device=device(type='cuda', index=0))
    reader.tensor(buf366, (1856,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 7424, device=device(type='cuda', index=0))
    reader.tensor(buf367, (1856,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 7424, device=device(type='cuda', index=0))
    reader.tensor(buf368, (1856,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 7424, device=device(type='cuda', index=0))
    reader.tensor(buf369, (1856,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 5939200, device=device(type='cuda', index=0))
    reader.tensor(buf370, (800, 1856, 1, 1), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf371, (800,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf372, (800,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf373, (800,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf374, (800,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf375, (800, 16, 3, 3), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf376, (800,), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf377, (800,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf378, (800,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf379, (800,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf380, (1088, 800, 1, 1), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf381, (1920,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf382, (1920,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf383, (1920,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf384, (1920,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 6144000, device=device(type='cuda', index=0))
    reader.tensor(buf385, (800, 1920, 1, 1), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf386, (800,), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf387, (800,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf388, (800,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf389, (800,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf390, (800, 16, 3, 3), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf391, (800,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf392, (800,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf393, (800,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf394, (800,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf395, (1088, 800, 1, 1), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 7936, device=device(type='cuda', index=0))
    reader.tensor(buf396, (1984,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 7936, device=device(type='cuda', index=0))
    reader.tensor(buf397, (1984,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 7936, device=device(type='cuda', index=0))
    reader.tensor(buf398, (1984,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 7936, device=device(type='cuda', index=0))
    reader.tensor(buf399, (1984,), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 6348800, device=device(type='cuda', index=0))
    reader.tensor(buf400, (800, 1984, 1, 1), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf401, (800,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf402, (800,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf403, (800,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf404, (800,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf405, (800, 16, 3, 3), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf406, (800,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf407, (800,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf408, (800,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf409, (800,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf410, (1088, 800, 1, 1), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf411, (2048,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf412, (2048,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf413, (2048,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf414, (2048,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 6553600, device=device(type='cuda', index=0))
    reader.tensor(buf415, (800, 2048, 1, 1), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf416, (800,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf417, (800,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf418, (800,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf419, (800,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf420, (800, 16, 3, 3), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf421, (800,), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf422, (800,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf423, (800,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf424, (800,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf425, (1088, 800, 1, 1), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 8448, device=device(type='cuda', index=0))
    reader.tensor(buf426, (2112,), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 8448, device=device(type='cuda', index=0))
    reader.tensor(buf427, (2112,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 8448, device=device(type='cuda', index=0))
    reader.tensor(buf428, (2112,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 8448, device=device(type='cuda', index=0))
    reader.tensor(buf429, (2112,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 6758400, device=device(type='cuda', index=0))
    reader.tensor(buf430, (800, 2112, 1, 1), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf431, (800,), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf432, (800,), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf433, (800,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf434, (800,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf435, (800, 16, 3, 3), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf436, (800,), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf437, (800,), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf438, (800,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf439, (800,), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf440, (1088, 800, 1, 1), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 8704, device=device(type='cuda', index=0))
    reader.tensor(buf441, (2176,), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 8704, device=device(type='cuda', index=0))
    reader.tensor(buf442, (2176,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 8704, device=device(type='cuda', index=0))
    reader.tensor(buf443, (2176,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 8704, device=device(type='cuda', index=0))
    reader.tensor(buf444, (2176,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 6963200, device=device(type='cuda', index=0))
    reader.tensor(buf445, (800, 2176, 1, 1), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf446, (800,), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf447, (800,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf448, (800,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf449, (800,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf450, (800, 16, 3, 3), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf451, (800,), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf452, (800,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf453, (800,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf454, (800,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf455, (1088, 800, 1, 1), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 8960, device=device(type='cuda', index=0))
    reader.tensor(buf456, (2240,), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 8960, device=device(type='cuda', index=0))
    reader.tensor(buf457, (2240,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 8960, device=device(type='cuda', index=0))
    reader.tensor(buf458, (2240,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 8960, device=device(type='cuda', index=0))
    reader.tensor(buf459, (2240,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 7168000, device=device(type='cuda', index=0))
    reader.tensor(buf460, (800, 2240, 1, 1), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf461, (800,), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf462, (800,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf463, (800,), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf464, (800,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf465, (800, 16, 3, 3), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf466, (800,), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf467, (800,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf468, (800,), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf469, (800,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf470, (1088, 800, 1, 1), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf471, (2304,), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf472, (2304,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf473, (2304,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf474, (2304,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 7372800, device=device(type='cuda', index=0))
    reader.tensor(buf475, (800, 2304, 1, 1), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf476, (800,), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf477, (800,), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf478, (800,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf479, (800,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf480, (800, 16, 3, 3), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf481, (800,), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf482, (800,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf483, (800,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf484, (800,), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf485, (1088, 800, 1, 1), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 9472, device=device(type='cuda', index=0))
    reader.tensor(buf486, (2368,), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 9472, device=device(type='cuda', index=0))
    reader.tensor(buf487, (2368,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 9472, device=device(type='cuda', index=0))
    reader.tensor(buf488, (2368,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 9472, device=device(type='cuda', index=0))
    reader.tensor(buf489, (2368,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 7577600, device=device(type='cuda', index=0))
    reader.tensor(buf490, (800, 2368, 1, 1), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf491, (800,), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf492, (800,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf493, (800,), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf494, (800,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf495, (800, 16, 3, 3), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf496, (800,), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf497, (800,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf498, (800,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf499, (800,), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 3481600, device=device(type='cuda', index=0))
    reader.tensor(buf500, (1088, 800, 1, 1), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf501, (2432,), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf502, (2432,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf503, (2432,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf504, (2432,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 22413312, device=device(type='cuda', index=0))
    reader.tensor(buf505, (2304, 2432, 1, 1), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf506, (2432,), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf507, (2432,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf508, (2432,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf509, (2432,), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 15564800, device=device(type='cuda', index=0))
    reader.tensor(buf510, (1600, 2432, 1, 1), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf511, (1600,), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf512, (1600,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf513, (1600,), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf514, (1600,), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 1843200, device=device(type='cuda', index=0))
    reader.tensor(buf515, (1600, 32, 3, 3), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf516, (1600,), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf517, (1600,), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf518, (1600,), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf519, (1600,), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 13926400, device=device(type='cuda', index=0))
    reader.tensor(buf520, (2176, 1600, 1, 1), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf521, (2432,), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf522, (2432,), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf523, (2432,), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 9728, device=device(type='cuda', index=0))
    reader.tensor(buf524, (2432,), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 15564800, device=device(type='cuda', index=0))
    reader.tensor(buf525, (1600, 2432, 1, 1), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf526, (1600,), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf527, (1600,), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf528, (1600,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf529, (1600,), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 1843200, device=device(type='cuda', index=0))
    reader.tensor(buf530, (1600, 32, 3, 3), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf531, (1600,), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf532, (1600,), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf533, (1600,), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf534, (1600,), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 13926400, device=device(type='cuda', index=0))
    reader.tensor(buf535, (2176, 1600, 1, 1), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf536, (2560,), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf537, (2560,), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf538, (2560,), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 10240, device=device(type='cuda', index=0))
    reader.tensor(buf539, (2560,), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 16384000, device=device(type='cuda', index=0))
    reader.tensor(buf540, (1600, 2560, 1, 1), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf541, (1600,), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf542, (1600,), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf543, (1600,), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf544, (1600,), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 1843200, device=device(type='cuda', index=0))
    reader.tensor(buf545, (1600, 32, 3, 3), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf546, (1600,), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf547, (1600,), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf548, (1600,), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf549, (1600,), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 13926400, device=device(type='cuda', index=0))
    reader.tensor(buf550, (2176, 1600, 1, 1), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 10752, device=device(type='cuda', index=0))
    reader.tensor(buf551, (2688,), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 10752, device=device(type='cuda', index=0))
    reader.tensor(buf552, (2688,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 10752, device=device(type='cuda', index=0))
    reader.tensor(buf553, (2688,), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 10752, device=device(type='cuda', index=0))
    reader.tensor(buf554, (2688,), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 10752000, device=device(type='cuda', index=0))
    reader.tensor(buf555, (1000, 2688, 1, 1), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf556, (1000,), is_leaf=True)  # arg556_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)