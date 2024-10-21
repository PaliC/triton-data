
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1):
        convolution_155 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_130 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_238 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_238, -1);  mul_238 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_465);  convolution_155 = unsqueeze_465 = None
        mul_239 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_469);  mul_239 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_131 = torch.ops.aten.add.Tensor(mul_240, unsqueeze_471);  mul_240 = unsqueeze_471 = None
        relu_7 = torch.ops.aten.relu.default(add_131);  add_131 = None
        convolution_156 = torch.ops.aten.convolution.default(relu_7, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  arg6_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_241 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_473);  convolution_156 = unsqueeze_473 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_477);  mul_242 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_133 = torch.ops.aten.add.Tensor(mul_243, unsqueeze_479);  mul_243 = unsqueeze_479 = None
        relu_8 = torch.ops.aten.relu.default(add_133);  add_133 = None
        convolution_157 = torch.ops.aten.convolution.default(relu_8, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg11_1 = None
        add_134 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_244 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60 = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_481);  convolution_157 = unsqueeze_481 = None
        mul_245 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_485);  mul_245 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_135 = torch.ops.aten.add.Tensor(mul_246, unsqueeze_487);  mul_246 = unsqueeze_487 = None
        add_136 = torch.ops.aten.add.Tensor(add_135, relu_7);  add_135 = relu_7 = None
        split_with_sizes_101 = torch.ops.aten.split_with_sizes.default(add_136, [16, 16], 1);  add_136 = None
        getitem_320 = split_with_sizes_101[0]
        getitem_321 = split_with_sizes_101[1];  split_with_sizes_101 = None
        convolution_158 = torch.ops.aten.convolution.default(getitem_320, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_320 = arg16_1 = None
        convolution_159 = torch.ops.aten.convolution.default(getitem_321, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_321 = arg17_1 = None
        cat_41 = torch.ops.aten.cat.default([convolution_158, convolution_159], 1);  convolution_158 = convolution_159 = None
        add_137 = torch.ops.aten.add.Tensor(arg19_1, 1e-05);  arg19_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_247 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61 = torch.ops.aten.sub.Tensor(cat_41, unsqueeze_489);  cat_41 = unsqueeze_489 = None
        mul_248 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_493);  mul_248 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_138 = torch.ops.aten.add.Tensor(mul_249, unsqueeze_495);  mul_249 = unsqueeze_495 = None
        relu_9 = torch.ops.aten.relu.default(add_138);  add_138 = None
        split_with_sizes_103 = torch.ops.aten.split_with_sizes.default(relu_9, [64, 64, 64], 1)
        getitem_325 = split_with_sizes_103[0];  split_with_sizes_103 = None
        convolution_160 = torch.ops.aten.convolution.default(getitem_325, arg22_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  getitem_325 = arg22_1 = None
        split_with_sizes_104 = torch.ops.aten.split_with_sizes.default(relu_9, [64, 64, 64], 1)
        getitem_329 = split_with_sizes_104[1];  split_with_sizes_104 = None
        convolution_161 = torch.ops.aten.convolution.default(getitem_329, arg23_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 64);  getitem_329 = arg23_1 = None
        split_with_sizes_105 = torch.ops.aten.split_with_sizes.default(relu_9, [64, 64, 64], 1);  relu_9 = None
        getitem_333 = split_with_sizes_105[2];  split_with_sizes_105 = None
        convolution_162 = torch.ops.aten.convolution.default(getitem_333, arg24_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 64);  getitem_333 = arg24_1 = None
        cat_42 = torch.ops.aten.cat.default([convolution_160, convolution_161, convolution_162], 1);  convolution_160 = convolution_161 = convolution_162 = None
        add_139 = torch.ops.aten.add.Tensor(arg26_1, 1e-05);  arg26_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_250 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62 = torch.ops.aten.sub.Tensor(cat_42, unsqueeze_497);  cat_42 = unsqueeze_497 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_501);  mul_251 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_140 = torch.ops.aten.add.Tensor(mul_252, unsqueeze_503);  mul_252 = unsqueeze_503 = None
        relu_10 = torch.ops.aten.relu.default(add_140);  add_140 = None
        split_with_sizes_107 = torch.ops.aten.split_with_sizes.default(relu_10, [96, 96], 1)
        getitem_336 = split_with_sizes_107[0];  split_with_sizes_107 = None
        convolution_163 = torch.ops.aten.convolution.default(getitem_336, arg29_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_336 = arg29_1 = None
        split_with_sizes_108 = torch.ops.aten.split_with_sizes.default(relu_10, [96, 96], 1);  relu_10 = None
        getitem_339 = split_with_sizes_108[1];  split_with_sizes_108 = None
        convolution_164 = torch.ops.aten.convolution.default(getitem_339, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_339 = arg30_1 = None
        cat_43 = torch.ops.aten.cat.default([convolution_163, convolution_164], 1);  convolution_163 = convolution_164 = None
        add_141 = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_253 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63 = torch.ops.aten.sub.Tensor(cat_43, unsqueeze_505);  cat_43 = unsqueeze_505 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_509);  mul_254 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_142 = torch.ops.aten.add.Tensor(mul_255, unsqueeze_511);  mul_255 = unsqueeze_511 = None
        split_with_sizes_109 = torch.ops.aten.split_with_sizes.default(add_142, [20, 20], 1)
        getitem_340 = split_with_sizes_109[0]
        getitem_341 = split_with_sizes_109[1];  split_with_sizes_109 = None
        convolution_165 = torch.ops.aten.convolution.default(getitem_340, arg35_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_340 = arg35_1 = None
        convolution_166 = torch.ops.aten.convolution.default(getitem_341, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_341 = arg36_1 = None
        cat_44 = torch.ops.aten.cat.default([convolution_165, convolution_166], 1);  convolution_165 = convolution_166 = None
        add_143 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_256 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_256, -1);  mul_256 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64 = torch.ops.aten.sub.Tensor(cat_44, unsqueeze_513);  cat_44 = unsqueeze_513 = None
        mul_257 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_517);  mul_257 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_144 = torch.ops.aten.add.Tensor(mul_258, unsqueeze_519);  mul_258 = unsqueeze_519 = None
        relu_11 = torch.ops.aten.relu.default(add_144);  add_144 = None
        convolution_167 = torch.ops.aten.convolution.default(relu_11, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  relu_11 = arg41_1 = None
        add_145 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_259 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_259, -1);  mul_259 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_521);  convolution_167 = unsqueeze_521 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_525);  mul_260 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_146 = torch.ops.aten.add.Tensor(mul_261, unsqueeze_527);  mul_261 = unsqueeze_527 = None
        relu_12 = torch.ops.aten.relu.default(add_146);  add_146 = None
        split_with_sizes_111 = torch.ops.aten.split_with_sizes.default(relu_12, [60, 60], 1)
        getitem_344 = split_with_sizes_111[0];  split_with_sizes_111 = None
        convolution_168 = torch.ops.aten.convolution.default(getitem_344, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_344 = arg46_1 = None
        split_with_sizes_112 = torch.ops.aten.split_with_sizes.default(relu_12, [60, 60], 1);  relu_12 = None
        getitem_347 = split_with_sizes_112[1];  split_with_sizes_112 = None
        convolution_169 = torch.ops.aten.convolution.default(getitem_347, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_347 = arg47_1 = None
        cat_45 = torch.ops.aten.cat.default([convolution_168, convolution_169], 1);  convolution_168 = convolution_169 = None
        add_147 = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_262 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_262, -1);  mul_262 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66 = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_529);  cat_45 = unsqueeze_529 = None
        mul_263 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_533);  mul_263 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_148 = torch.ops.aten.add.Tensor(mul_264, unsqueeze_535);  mul_264 = unsqueeze_535 = None
        add_149 = torch.ops.aten.add.Tensor(add_148, add_142);  add_148 = add_142 = None
        convolution_170 = torch.ops.aten.convolution.default(add_149, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_149 = arg52_1 = None
        add_150 = torch.ops.aten.add.Tensor(arg54_1, 1e-05);  arg54_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_265 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_265, -1);  mul_265 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_537);  convolution_170 = unsqueeze_537 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_541);  mul_266 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_151 = torch.ops.aten.add.Tensor(mul_267, unsqueeze_543);  mul_267 = unsqueeze_543 = None
        sigmoid_64 = torch.ops.aten.sigmoid.default(add_151)
        mul_268 = torch.ops.aten.mul.Tensor(add_151, sigmoid_64);  add_151 = sigmoid_64 = None
        split_with_sizes_114 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1)
        getitem_352 = split_with_sizes_114[0];  split_with_sizes_114 = None
        convolution_171 = torch.ops.aten.convolution.default(getitem_352, arg57_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 60);  getitem_352 = arg57_1 = None
        split_with_sizes_115 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1)
        getitem_357 = split_with_sizes_115[1];  split_with_sizes_115 = None
        convolution_172 = torch.ops.aten.convolution.default(getitem_357, arg58_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 60);  getitem_357 = arg58_1 = None
        split_with_sizes_116 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1)
        getitem_362 = split_with_sizes_116[2];  split_with_sizes_116 = None
        convolution_173 = torch.ops.aten.convolution.default(getitem_362, arg59_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 60);  getitem_362 = arg59_1 = None
        split_with_sizes_117 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1);  mul_268 = None
        getitem_367 = split_with_sizes_117[3];  split_with_sizes_117 = None
        convolution_174 = torch.ops.aten.convolution.default(getitem_367, arg60_1, None, [2, 2], [4, 4], [1, 1], False, [0, 0], 60);  getitem_367 = arg60_1 = None
        cat_46 = torch.ops.aten.cat.default([convolution_171, convolution_172, convolution_173, convolution_174], 1);  convolution_171 = convolution_172 = convolution_173 = convolution_174 = None
        add_152 = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_269 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_269, -1);  mul_269 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68 = torch.ops.aten.sub.Tensor(cat_46, unsqueeze_545);  cat_46 = unsqueeze_545 = None
        mul_270 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_549);  mul_270 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_153 = torch.ops.aten.add.Tensor(mul_271, unsqueeze_551);  mul_271 = unsqueeze_551 = None
        sigmoid_65 = torch.ops.aten.sigmoid.default(add_153)
        mul_272 = torch.ops.aten.mul.Tensor(add_153, sigmoid_65);  add_153 = sigmoid_65 = None
        mean_17 = torch.ops.aten.mean.dim(mul_272, [2, 3], True)
        convolution_175 = torch.ops.aten.convolution.default(mean_17, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg65_1 = arg66_1 = None
        sigmoid_66 = torch.ops.aten.sigmoid.default(convolution_175)
        mul_273 = torch.ops.aten.mul.Tensor(convolution_175, sigmoid_66);  convolution_175 = sigmoid_66 = None
        convolution_176 = torch.ops.aten.convolution.default(mul_273, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg67_1 = arg68_1 = None
        sigmoid_67 = torch.ops.aten.sigmoid.default(convolution_176);  convolution_176 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_272, sigmoid_67);  mul_272 = sigmoid_67 = None
        convolution_177 = torch.ops.aten.convolution.default(mul_274, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_274 = arg69_1 = None
        add_154 = torch.ops.aten.add.Tensor(arg71_1, 1e-05);  arg71_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_275 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_275, -1);  mul_275 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_553);  convolution_177 = unsqueeze_553 = None
        mul_276 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_557);  mul_276 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_155 = torch.ops.aten.add.Tensor(mul_277, unsqueeze_559);  mul_277 = unsqueeze_559 = None
        split_with_sizes_118 = torch.ops.aten.split_with_sizes.default(add_155, [28, 28], 1)
        getitem_368 = split_with_sizes_118[0]
        getitem_369 = split_with_sizes_118[1];  split_with_sizes_118 = None
        convolution_178 = torch.ops.aten.convolution.default(getitem_368, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_368 = arg74_1 = None
        convolution_179 = torch.ops.aten.convolution.default(getitem_369, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_369 = arg75_1 = None
        cat_47 = torch.ops.aten.cat.default([convolution_178, convolution_179], 1);  convolution_178 = convolution_179 = None
        add_156 = torch.ops.aten.add.Tensor(arg77_1, 1e-05);  arg77_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_278 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_278, -1);  mul_278 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70 = torch.ops.aten.sub.Tensor(cat_47, unsqueeze_561);  cat_47 = unsqueeze_561 = None
        mul_279 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_280 = torch.ops.aten.mul.Tensor(mul_279, unsqueeze_565);  mul_279 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_157 = torch.ops.aten.add.Tensor(mul_280, unsqueeze_567);  mul_280 = unsqueeze_567 = None
        sigmoid_68 = torch.ops.aten.sigmoid.default(add_157)
        mul_281 = torch.ops.aten.mul.Tensor(add_157, sigmoid_68);  add_157 = sigmoid_68 = None
        split_with_sizes_120 = torch.ops.aten.split_with_sizes.default(mul_281, [168, 168], 1)
        getitem_372 = split_with_sizes_120[0];  split_with_sizes_120 = None
        convolution_180 = torch.ops.aten.convolution.default(getitem_372, arg80_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_372 = arg80_1 = None
        split_with_sizes_121 = torch.ops.aten.split_with_sizes.default(mul_281, [168, 168], 1);  mul_281 = None
        getitem_375 = split_with_sizes_121[1];  split_with_sizes_121 = None
        convolution_181 = torch.ops.aten.convolution.default(getitem_375, arg81_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_375 = arg81_1 = None
        cat_48 = torch.ops.aten.cat.default([convolution_180, convolution_181], 1);  convolution_180 = convolution_181 = None
        add_158 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_282 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71 = torch.ops.aten.sub.Tensor(cat_48, unsqueeze_569);  cat_48 = unsqueeze_569 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_573);  mul_283 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_159 = torch.ops.aten.add.Tensor(mul_284, unsqueeze_575);  mul_284 = unsqueeze_575 = None
        sigmoid_69 = torch.ops.aten.sigmoid.default(add_159)
        mul_285 = torch.ops.aten.mul.Tensor(add_159, sigmoid_69);  add_159 = sigmoid_69 = None
        mean_18 = torch.ops.aten.mean.dim(mul_285, [2, 3], True)
        convolution_182 = torch.ops.aten.convolution.default(mean_18, arg86_1, arg87_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg86_1 = arg87_1 = None
        sigmoid_70 = torch.ops.aten.sigmoid.default(convolution_182)
        mul_286 = torch.ops.aten.mul.Tensor(convolution_182, sigmoid_70);  convolution_182 = sigmoid_70 = None
        convolution_183 = torch.ops.aten.convolution.default(mul_286, arg88_1, arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg88_1 = arg89_1 = None
        sigmoid_71 = torch.ops.aten.sigmoid.default(convolution_183);  convolution_183 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_285, sigmoid_71);  mul_285 = sigmoid_71 = None
        split_with_sizes_122 = torch.ops.aten.split_with_sizes.default(mul_287, [168, 168], 1);  mul_287 = None
        getitem_376 = split_with_sizes_122[0]
        getitem_377 = split_with_sizes_122[1];  split_with_sizes_122 = None
        convolution_184 = torch.ops.aten.convolution.default(getitem_376, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_376 = arg90_1 = None
        convolution_185 = torch.ops.aten.convolution.default(getitem_377, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_377 = arg91_1 = None
        cat_49 = torch.ops.aten.cat.default([convolution_184, convolution_185], 1);  convolution_184 = convolution_185 = None
        add_160 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_288 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72 = torch.ops.aten.sub.Tensor(cat_49, unsqueeze_577);  cat_49 = unsqueeze_577 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_581);  mul_289 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_161 = torch.ops.aten.add.Tensor(mul_290, unsqueeze_583);  mul_290 = unsqueeze_583 = None
        add_162 = torch.ops.aten.add.Tensor(add_161, add_155);  add_161 = add_155 = None
        split_with_sizes_123 = torch.ops.aten.split_with_sizes.default(add_162, [28, 28], 1)
        getitem_378 = split_with_sizes_123[0]
        getitem_379 = split_with_sizes_123[1];  split_with_sizes_123 = None
        convolution_186 = torch.ops.aten.convolution.default(getitem_378, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_378 = arg96_1 = None
        convolution_187 = torch.ops.aten.convolution.default(getitem_379, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_379 = arg97_1 = None
        cat_50 = torch.ops.aten.cat.default([convolution_186, convolution_187], 1);  convolution_186 = convolution_187 = None
        add_163 = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_291 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73 = torch.ops.aten.sub.Tensor(cat_50, unsqueeze_585);  cat_50 = unsqueeze_585 = None
        mul_292 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_589);  mul_292 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_164 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_591);  mul_293 = unsqueeze_591 = None
        sigmoid_72 = torch.ops.aten.sigmoid.default(add_164)
        mul_294 = torch.ops.aten.mul.Tensor(add_164, sigmoid_72);  add_164 = sigmoid_72 = None
        split_with_sizes_125 = torch.ops.aten.split_with_sizes.default(mul_294, [168, 168], 1)
        getitem_382 = split_with_sizes_125[0];  split_with_sizes_125 = None
        convolution_188 = torch.ops.aten.convolution.default(getitem_382, arg102_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_382 = arg102_1 = None
        split_with_sizes_126 = torch.ops.aten.split_with_sizes.default(mul_294, [168, 168], 1);  mul_294 = None
        getitem_385 = split_with_sizes_126[1];  split_with_sizes_126 = None
        convolution_189 = torch.ops.aten.convolution.default(getitem_385, arg103_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_385 = arg103_1 = None
        cat_51 = torch.ops.aten.cat.default([convolution_188, convolution_189], 1);  convolution_188 = convolution_189 = None
        add_165 = torch.ops.aten.add.Tensor(arg105_1, 1e-05);  arg105_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_165);  add_165 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_295 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74 = torch.ops.aten.sub.Tensor(cat_51, unsqueeze_593);  cat_51 = unsqueeze_593 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_597);  mul_296 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_166 = torch.ops.aten.add.Tensor(mul_297, unsqueeze_599);  mul_297 = unsqueeze_599 = None
        sigmoid_73 = torch.ops.aten.sigmoid.default(add_166)
        mul_298 = torch.ops.aten.mul.Tensor(add_166, sigmoid_73);  add_166 = sigmoid_73 = None
        mean_19 = torch.ops.aten.mean.dim(mul_298, [2, 3], True)
        convolution_190 = torch.ops.aten.convolution.default(mean_19, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg108_1 = arg109_1 = None
        sigmoid_74 = torch.ops.aten.sigmoid.default(convolution_190)
        mul_299 = torch.ops.aten.mul.Tensor(convolution_190, sigmoid_74);  convolution_190 = sigmoid_74 = None
        convolution_191 = torch.ops.aten.convolution.default(mul_299, arg110_1, arg111_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg110_1 = arg111_1 = None
        sigmoid_75 = torch.ops.aten.sigmoid.default(convolution_191);  convolution_191 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_298, sigmoid_75);  mul_298 = sigmoid_75 = None
        split_with_sizes_127 = torch.ops.aten.split_with_sizes.default(mul_300, [168, 168], 1);  mul_300 = None
        getitem_386 = split_with_sizes_127[0]
        getitem_387 = split_with_sizes_127[1];  split_with_sizes_127 = None
        convolution_192 = torch.ops.aten.convolution.default(getitem_386, arg112_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_386 = arg112_1 = None
        convolution_193 = torch.ops.aten.convolution.default(getitem_387, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_387 = arg113_1 = None
        cat_52 = torch.ops.aten.cat.default([convolution_192, convolution_193], 1);  convolution_192 = convolution_193 = None
        add_167 = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_167);  add_167 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_301 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_301, -1);  mul_301 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75 = torch.ops.aten.sub.Tensor(cat_52, unsqueeze_601);  cat_52 = unsqueeze_601 = None
        mul_302 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_605);  mul_302 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_168 = torch.ops.aten.add.Tensor(mul_303, unsqueeze_607);  mul_303 = unsqueeze_607 = None
        add_169 = torch.ops.aten.add.Tensor(add_168, add_162);  add_168 = add_162 = None
        split_with_sizes_128 = torch.ops.aten.split_with_sizes.default(add_169, [28, 28], 1)
        getitem_388 = split_with_sizes_128[0]
        getitem_389 = split_with_sizes_128[1];  split_with_sizes_128 = None
        convolution_194 = torch.ops.aten.convolution.default(getitem_388, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_388 = arg118_1 = None
        convolution_195 = torch.ops.aten.convolution.default(getitem_389, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_389 = arg119_1 = None
        cat_53 = torch.ops.aten.cat.default([convolution_194, convolution_195], 1);  convolution_194 = convolution_195 = None
        add_170 = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_304 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76 = torch.ops.aten.sub.Tensor(cat_53, unsqueeze_609);  cat_53 = unsqueeze_609 = None
        mul_305 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_613);  mul_305 = unsqueeze_613 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_171 = torch.ops.aten.add.Tensor(mul_306, unsqueeze_615);  mul_306 = unsqueeze_615 = None
        sigmoid_76 = torch.ops.aten.sigmoid.default(add_171)
        mul_307 = torch.ops.aten.mul.Tensor(add_171, sigmoid_76);  add_171 = sigmoid_76 = None
        split_with_sizes_130 = torch.ops.aten.split_with_sizes.default(mul_307, [168, 168], 1)
        getitem_392 = split_with_sizes_130[0];  split_with_sizes_130 = None
        convolution_196 = torch.ops.aten.convolution.default(getitem_392, arg124_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_392 = arg124_1 = None
        split_with_sizes_131 = torch.ops.aten.split_with_sizes.default(mul_307, [168, 168], 1);  mul_307 = None
        getitem_395 = split_with_sizes_131[1];  split_with_sizes_131 = None
        convolution_197 = torch.ops.aten.convolution.default(getitem_395, arg125_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_395 = arg125_1 = None
        cat_54 = torch.ops.aten.cat.default([convolution_196, convolution_197], 1);  convolution_196 = convolution_197 = None
        add_172 = torch.ops.aten.add.Tensor(arg127_1, 1e-05);  arg127_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_308 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77 = torch.ops.aten.sub.Tensor(cat_54, unsqueeze_617);  cat_54 = unsqueeze_617 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_621);  mul_309 = unsqueeze_621 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_173 = torch.ops.aten.add.Tensor(mul_310, unsqueeze_623);  mul_310 = unsqueeze_623 = None
        sigmoid_77 = torch.ops.aten.sigmoid.default(add_173)
        mul_311 = torch.ops.aten.mul.Tensor(add_173, sigmoid_77);  add_173 = sigmoid_77 = None
        mean_20 = torch.ops.aten.mean.dim(mul_311, [2, 3], True)
        convolution_198 = torch.ops.aten.convolution.default(mean_20, arg130_1, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg130_1 = arg131_1 = None
        sigmoid_78 = torch.ops.aten.sigmoid.default(convolution_198)
        mul_312 = torch.ops.aten.mul.Tensor(convolution_198, sigmoid_78);  convolution_198 = sigmoid_78 = None
        convolution_199 = torch.ops.aten.convolution.default(mul_312, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg132_1 = arg133_1 = None
        sigmoid_79 = torch.ops.aten.sigmoid.default(convolution_199);  convolution_199 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_311, sigmoid_79);  mul_311 = sigmoid_79 = None
        split_with_sizes_132 = torch.ops.aten.split_with_sizes.default(mul_313, [168, 168], 1);  mul_313 = None
        getitem_396 = split_with_sizes_132[0]
        getitem_397 = split_with_sizes_132[1];  split_with_sizes_132 = None
        convolution_200 = torch.ops.aten.convolution.default(getitem_396, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_396 = arg134_1 = None
        convolution_201 = torch.ops.aten.convolution.default(getitem_397, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_397 = arg135_1 = None
        cat_55 = torch.ops.aten.cat.default([convolution_200, convolution_201], 1);  convolution_200 = convolution_201 = None
        add_174 = torch.ops.aten.add.Tensor(arg137_1, 1e-05);  arg137_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_314 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(mul_314, -1);  mul_314 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78 = torch.ops.aten.sub.Tensor(cat_55, unsqueeze_625);  cat_55 = unsqueeze_625 = None
        mul_315 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_629);  mul_315 = unsqueeze_629 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_175 = torch.ops.aten.add.Tensor(mul_316, unsqueeze_631);  mul_316 = unsqueeze_631 = None
        add_176 = torch.ops.aten.add.Tensor(add_175, add_169);  add_175 = add_169 = None
        convolution_202 = torch.ops.aten.convolution.default(add_176, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_176 = arg140_1 = None
        add_177 = torch.ops.aten.add.Tensor(arg142_1, 1e-05);  arg142_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_317 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(mul_317, -1);  mul_317 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_633);  convolution_202 = unsqueeze_633 = None
        mul_318 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_637);  mul_318 = unsqueeze_637 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_178 = torch.ops.aten.add.Tensor(mul_319, unsqueeze_639);  mul_319 = unsqueeze_639 = None
        sigmoid_80 = torch.ops.aten.sigmoid.default(add_178)
        mul_320 = torch.ops.aten.mul.Tensor(add_178, sigmoid_80);  add_178 = sigmoid_80 = None
        split_with_sizes_134 = torch.ops.aten.split_with_sizes.default(mul_320, [112, 112, 112], 1)
        getitem_401 = split_with_sizes_134[0];  split_with_sizes_134 = None
        convolution_203 = torch.ops.aten.convolution.default(getitem_401, arg145_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 112);  getitem_401 = arg145_1 = None
        split_with_sizes_135 = torch.ops.aten.split_with_sizes.default(mul_320, [112, 112, 112], 1)
        getitem_405 = split_with_sizes_135[1];  split_with_sizes_135 = None
        convolution_204 = torch.ops.aten.convolution.default(getitem_405, arg146_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 112);  getitem_405 = arg146_1 = None
        split_with_sizes_136 = torch.ops.aten.split_with_sizes.default(mul_320, [112, 112, 112], 1);  mul_320 = None
        getitem_409 = split_with_sizes_136[2];  split_with_sizes_136 = None
        convolution_205 = torch.ops.aten.convolution.default(getitem_409, arg147_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 112);  getitem_409 = arg147_1 = None
        cat_56 = torch.ops.aten.cat.default([convolution_203, convolution_204, convolution_205], 1);  convolution_203 = convolution_204 = convolution_205 = None
        add_179 = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_179);  add_179 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_321 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(cat_56, unsqueeze_641);  cat_56 = unsqueeze_641 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_645);  mul_322 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_180 = torch.ops.aten.add.Tensor(mul_323, unsqueeze_647);  mul_323 = unsqueeze_647 = None
        sigmoid_81 = torch.ops.aten.sigmoid.default(add_180)
        mul_324 = torch.ops.aten.mul.Tensor(add_180, sigmoid_81);  add_180 = sigmoid_81 = None
        mean_21 = torch.ops.aten.mean.dim(mul_324, [2, 3], True)
        convolution_206 = torch.ops.aten.convolution.default(mean_21, arg152_1, arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg152_1 = arg153_1 = None
        sigmoid_82 = torch.ops.aten.sigmoid.default(convolution_206)
        mul_325 = torch.ops.aten.mul.Tensor(convolution_206, sigmoid_82);  convolution_206 = sigmoid_82 = None
        convolution_207 = torch.ops.aten.convolution.default(mul_325, arg154_1, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = arg154_1 = arg155_1 = None
        sigmoid_83 = torch.ops.aten.sigmoid.default(convolution_207);  convolution_207 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_324, sigmoid_83);  mul_324 = sigmoid_83 = None
        convolution_208 = torch.ops.aten.convolution.default(mul_326, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_326 = arg156_1 = None
        add_181 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_327 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_649);  convolution_208 = unsqueeze_649 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_653);  mul_328 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_182 = torch.ops.aten.add.Tensor(mul_329, unsqueeze_655);  mul_329 = unsqueeze_655 = None
        split_with_sizes_137 = torch.ops.aten.split_with_sizes.default(add_182, [52, 52], 1)
        getitem_410 = split_with_sizes_137[0]
        getitem_411 = split_with_sizes_137[1];  split_with_sizes_137 = None
        convolution_209 = torch.ops.aten.convolution.default(getitem_410, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_410 = arg161_1 = None
        convolution_210 = torch.ops.aten.convolution.default(getitem_411, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_411 = arg162_1 = None
        cat_57 = torch.ops.aten.cat.default([convolution_209, convolution_210], 1);  convolution_209 = convolution_210 = None
        add_183 = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_330 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82 = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_657);  cat_57 = unsqueeze_657 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_661);  mul_331 = unsqueeze_661 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_184 = torch.ops.aten.add.Tensor(mul_332, unsqueeze_663);  mul_332 = unsqueeze_663 = None
        sigmoid_84 = torch.ops.aten.sigmoid.default(add_184)
        mul_333 = torch.ops.aten.mul.Tensor(add_184, sigmoid_84);  add_184 = sigmoid_84 = None
        split_with_sizes_139 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1)
        getitem_416 = split_with_sizes_139[0];  split_with_sizes_139 = None
        convolution_211 = torch.ops.aten.convolution.default(getitem_416, arg167_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_416 = arg167_1 = None
        split_with_sizes_140 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1)
        getitem_421 = split_with_sizes_140[1];  split_with_sizes_140 = None
        convolution_212 = torch.ops.aten.convolution.default(getitem_421, arg168_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_421 = arg168_1 = None
        split_with_sizes_141 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1)
        getitem_426 = split_with_sizes_141[2];  split_with_sizes_141 = None
        convolution_213 = torch.ops.aten.convolution.default(getitem_426, arg169_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_426 = arg169_1 = None
        split_with_sizes_142 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1);  mul_333 = None
        getitem_431 = split_with_sizes_142[3];  split_with_sizes_142 = None
        convolution_214 = torch.ops.aten.convolution.default(getitem_431, arg170_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_431 = arg170_1 = None
        cat_58 = torch.ops.aten.cat.default([convolution_211, convolution_212, convolution_213, convolution_214], 1);  convolution_211 = convolution_212 = convolution_213 = convolution_214 = None
        add_185 = torch.ops.aten.add.Tensor(arg172_1, 1e-05);  arg172_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_334 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_334, -1);  mul_334 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83 = torch.ops.aten.sub.Tensor(cat_58, unsqueeze_665);  cat_58 = unsqueeze_665 = None
        mul_335 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_669);  mul_335 = unsqueeze_669 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_186 = torch.ops.aten.add.Tensor(mul_336, unsqueeze_671);  mul_336 = unsqueeze_671 = None
        sigmoid_85 = torch.ops.aten.sigmoid.default(add_186)
        mul_337 = torch.ops.aten.mul.Tensor(add_186, sigmoid_85);  add_186 = sigmoid_85 = None
        mean_22 = torch.ops.aten.mean.dim(mul_337, [2, 3], True)
        convolution_215 = torch.ops.aten.convolution.default(mean_22, arg175_1, arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg175_1 = arg176_1 = None
        sigmoid_86 = torch.ops.aten.sigmoid.default(convolution_215)
        mul_338 = torch.ops.aten.mul.Tensor(convolution_215, sigmoid_86);  convolution_215 = sigmoid_86 = None
        convolution_216 = torch.ops.aten.convolution.default(mul_338, arg177_1, arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_338 = arg177_1 = arg178_1 = None
        sigmoid_87 = torch.ops.aten.sigmoid.default(convolution_216);  convolution_216 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_337, sigmoid_87);  mul_337 = sigmoid_87 = None
        split_with_sizes_143 = torch.ops.aten.split_with_sizes.default(mul_339, [312, 312], 1);  mul_339 = None
        getitem_432 = split_with_sizes_143[0]
        getitem_433 = split_with_sizes_143[1];  split_with_sizes_143 = None
        convolution_217 = torch.ops.aten.convolution.default(getitem_432, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_432 = arg179_1 = None
        convolution_218 = torch.ops.aten.convolution.default(getitem_433, arg180_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_433 = arg180_1 = None
        cat_59 = torch.ops.aten.cat.default([convolution_217, convolution_218], 1);  convolution_217 = convolution_218 = None
        add_187 = torch.ops.aten.add.Tensor(arg182_1, 1e-05);  arg182_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_340 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(mul_340, -1);  mul_340 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84 = torch.ops.aten.sub.Tensor(cat_59, unsqueeze_673);  cat_59 = unsqueeze_673 = None
        mul_341 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_341, unsqueeze_677);  mul_341 = unsqueeze_677 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_188 = torch.ops.aten.add.Tensor(mul_342, unsqueeze_679);  mul_342 = unsqueeze_679 = None
        add_189 = torch.ops.aten.add.Tensor(add_188, add_182);  add_188 = add_182 = None
        split_with_sizes_144 = torch.ops.aten.split_with_sizes.default(add_189, [52, 52], 1)
        getitem_434 = split_with_sizes_144[0]
        getitem_435 = split_with_sizes_144[1];  split_with_sizes_144 = None
        convolution_219 = torch.ops.aten.convolution.default(getitem_434, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_434 = arg185_1 = None
        convolution_220 = torch.ops.aten.convolution.default(getitem_435, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_435 = arg186_1 = None
        cat_60 = torch.ops.aten.cat.default([convolution_219, convolution_220], 1);  convolution_219 = convolution_220 = None
        add_190 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_343 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_343, -1);  mul_343 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(cat_60, unsqueeze_681);  cat_60 = unsqueeze_681 = None
        mul_344 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_345 = torch.ops.aten.mul.Tensor(mul_344, unsqueeze_685);  mul_344 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_191 = torch.ops.aten.add.Tensor(mul_345, unsqueeze_687);  mul_345 = unsqueeze_687 = None
        sigmoid_88 = torch.ops.aten.sigmoid.default(add_191)
        mul_346 = torch.ops.aten.mul.Tensor(add_191, sigmoid_88);  add_191 = sigmoid_88 = None
        split_with_sizes_146 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1)
        getitem_440 = split_with_sizes_146[0];  split_with_sizes_146 = None
        convolution_221 = torch.ops.aten.convolution.default(getitem_440, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_440 = arg191_1 = None
        split_with_sizes_147 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1)
        getitem_445 = split_with_sizes_147[1];  split_with_sizes_147 = None
        convolution_222 = torch.ops.aten.convolution.default(getitem_445, arg192_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_445 = arg192_1 = None
        split_with_sizes_148 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1)
        getitem_450 = split_with_sizes_148[2];  split_with_sizes_148 = None
        convolution_223 = torch.ops.aten.convolution.default(getitem_450, arg193_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_450 = arg193_1 = None
        split_with_sizes_149 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1);  mul_346 = None
        getitem_455 = split_with_sizes_149[3];  split_with_sizes_149 = None
        convolution_224 = torch.ops.aten.convolution.default(getitem_455, arg194_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_455 = arg194_1 = None
        cat_61 = torch.ops.aten.cat.default([convolution_221, convolution_222, convolution_223, convolution_224], 1);  convolution_221 = convolution_222 = convolution_223 = convolution_224 = None
        add_192 = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_347 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_347, -1);  mul_347 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(cat_61, unsqueeze_689);  cat_61 = unsqueeze_689 = None
        mul_348 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_348, unsqueeze_693);  mul_348 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_193 = torch.ops.aten.add.Tensor(mul_349, unsqueeze_695);  mul_349 = unsqueeze_695 = None
        sigmoid_89 = torch.ops.aten.sigmoid.default(add_193)
        mul_350 = torch.ops.aten.mul.Tensor(add_193, sigmoid_89);  add_193 = sigmoid_89 = None
        mean_23 = torch.ops.aten.mean.dim(mul_350, [2, 3], True)
        convolution_225 = torch.ops.aten.convolution.default(mean_23, arg199_1, arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg199_1 = arg200_1 = None
        sigmoid_90 = torch.ops.aten.sigmoid.default(convolution_225)
        mul_351 = torch.ops.aten.mul.Tensor(convolution_225, sigmoid_90);  convolution_225 = sigmoid_90 = None
        convolution_226 = torch.ops.aten.convolution.default(mul_351, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_351 = arg201_1 = arg202_1 = None
        sigmoid_91 = torch.ops.aten.sigmoid.default(convolution_226);  convolution_226 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_350, sigmoid_91);  mul_350 = sigmoid_91 = None
        split_with_sizes_150 = torch.ops.aten.split_with_sizes.default(mul_352, [312, 312], 1);  mul_352 = None
        getitem_456 = split_with_sizes_150[0]
        getitem_457 = split_with_sizes_150[1];  split_with_sizes_150 = None
        convolution_227 = torch.ops.aten.convolution.default(getitem_456, arg203_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_456 = arg203_1 = None
        convolution_228 = torch.ops.aten.convolution.default(getitem_457, arg204_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_457 = arg204_1 = None
        cat_62 = torch.ops.aten.cat.default([convolution_227, convolution_228], 1);  convolution_227 = convolution_228 = None
        add_194 = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_353 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_353, -1);  mul_353 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(cat_62, unsqueeze_697);  cat_62 = unsqueeze_697 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, unsqueeze_701);  mul_354 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_195 = torch.ops.aten.add.Tensor(mul_355, unsqueeze_703);  mul_355 = unsqueeze_703 = None
        add_196 = torch.ops.aten.add.Tensor(add_195, add_189);  add_195 = add_189 = None
        split_with_sizes_151 = torch.ops.aten.split_with_sizes.default(add_196, [52, 52], 1)
        getitem_458 = split_with_sizes_151[0]
        getitem_459 = split_with_sizes_151[1];  split_with_sizes_151 = None
        convolution_229 = torch.ops.aten.convolution.default(getitem_458, arg209_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_458 = arg209_1 = None
        convolution_230 = torch.ops.aten.convolution.default(getitem_459, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_459 = arg210_1 = None
        cat_63 = torch.ops.aten.cat.default([convolution_229, convolution_230], 1);  convolution_229 = convolution_230 = None
        add_197 = torch.ops.aten.add.Tensor(arg212_1, 1e-05);  arg212_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_356 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_356, -1);  mul_356 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(cat_63, unsqueeze_705);  cat_63 = unsqueeze_705 = None
        mul_357 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_358 = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_709);  mul_357 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_198 = torch.ops.aten.add.Tensor(mul_358, unsqueeze_711);  mul_358 = unsqueeze_711 = None
        sigmoid_92 = torch.ops.aten.sigmoid.default(add_198)
        mul_359 = torch.ops.aten.mul.Tensor(add_198, sigmoid_92);  add_198 = sigmoid_92 = None
        split_with_sizes_153 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1)
        getitem_464 = split_with_sizes_153[0];  split_with_sizes_153 = None
        convolution_231 = torch.ops.aten.convolution.default(getitem_464, arg215_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_464 = arg215_1 = None
        split_with_sizes_154 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1)
        getitem_469 = split_with_sizes_154[1];  split_with_sizes_154 = None
        convolution_232 = torch.ops.aten.convolution.default(getitem_469, arg216_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_469 = arg216_1 = None
        split_with_sizes_155 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1)
        getitem_474 = split_with_sizes_155[2];  split_with_sizes_155 = None
        convolution_233 = torch.ops.aten.convolution.default(getitem_474, arg217_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_474 = arg217_1 = None
        split_with_sizes_156 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1);  mul_359 = None
        getitem_479 = split_with_sizes_156[3];  split_with_sizes_156 = None
        convolution_234 = torch.ops.aten.convolution.default(getitem_479, arg218_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_479 = arg218_1 = None
        cat_64 = torch.ops.aten.cat.default([convolution_231, convolution_232, convolution_233, convolution_234], 1);  convolution_231 = convolution_232 = convolution_233 = convolution_234 = None
        add_199 = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(cat_64, unsqueeze_713);  cat_64 = unsqueeze_713 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_717);  mul_361 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_200 = torch.ops.aten.add.Tensor(mul_362, unsqueeze_719);  mul_362 = unsqueeze_719 = None
        sigmoid_93 = torch.ops.aten.sigmoid.default(add_200)
        mul_363 = torch.ops.aten.mul.Tensor(add_200, sigmoid_93);  add_200 = sigmoid_93 = None
        mean_24 = torch.ops.aten.mean.dim(mul_363, [2, 3], True)
        convolution_235 = torch.ops.aten.convolution.default(mean_24, arg223_1, arg224_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg223_1 = arg224_1 = None
        sigmoid_94 = torch.ops.aten.sigmoid.default(convolution_235)
        mul_364 = torch.ops.aten.mul.Tensor(convolution_235, sigmoid_94);  convolution_235 = sigmoid_94 = None
        convolution_236 = torch.ops.aten.convolution.default(mul_364, arg225_1, arg226_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_364 = arg225_1 = arg226_1 = None
        sigmoid_95 = torch.ops.aten.sigmoid.default(convolution_236);  convolution_236 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_363, sigmoid_95);  mul_363 = sigmoid_95 = None
        split_with_sizes_157 = torch.ops.aten.split_with_sizes.default(mul_365, [312, 312], 1);  mul_365 = None
        getitem_480 = split_with_sizes_157[0]
        getitem_481 = split_with_sizes_157[1];  split_with_sizes_157 = None
        convolution_237 = torch.ops.aten.convolution.default(getitem_480, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_480 = arg227_1 = None
        convolution_238 = torch.ops.aten.convolution.default(getitem_481, arg228_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_481 = arg228_1 = None
        cat_65 = torch.ops.aten.cat.default([convolution_237, convolution_238], 1);  convolution_237 = convolution_238 = None
        add_201 = torch.ops.aten.add.Tensor(arg230_1, 1e-05);  arg230_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_366 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(cat_65, unsqueeze_721);  cat_65 = unsqueeze_721 = None
        mul_367 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_725);  mul_367 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_202 = torch.ops.aten.add.Tensor(mul_368, unsqueeze_727);  mul_368 = unsqueeze_727 = None
        add_203 = torch.ops.aten.add.Tensor(add_202, add_196);  add_202 = add_196 = None
        convolution_239 = torch.ops.aten.convolution.default(add_203, arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_203 = arg233_1 = None
        add_204 = torch.ops.aten.add.Tensor(arg235_1, 1e-05);  arg235_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_369 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_239, unsqueeze_729);  convolution_239 = unsqueeze_729 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_733);  mul_370 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_205 = torch.ops.aten.add.Tensor(mul_371, unsqueeze_735);  mul_371 = unsqueeze_735 = None
        sigmoid_96 = torch.ops.aten.sigmoid.default(add_205)
        mul_372 = torch.ops.aten.mul.Tensor(add_205, sigmoid_96);  add_205 = sigmoid_96 = None
        convolution_240 = torch.ops.aten.convolution.default(mul_372, arg238_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 624);  mul_372 = arg238_1 = None
        add_206 = torch.ops.aten.add.Tensor(arg240_1, 1e-05);  arg240_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_373 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_373, -1);  mul_373 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_737);  convolution_240 = unsqueeze_737 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, unsqueeze_741);  mul_374 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_207 = torch.ops.aten.add.Tensor(mul_375, unsqueeze_743);  mul_375 = unsqueeze_743 = None
        sigmoid_97 = torch.ops.aten.sigmoid.default(add_207)
        mul_376 = torch.ops.aten.mul.Tensor(add_207, sigmoid_97);  add_207 = sigmoid_97 = None
        mean_25 = torch.ops.aten.mean.dim(mul_376, [2, 3], True)
        convolution_241 = torch.ops.aten.convolution.default(mean_25, arg243_1, arg244_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg243_1 = arg244_1 = None
        sigmoid_98 = torch.ops.aten.sigmoid.default(convolution_241)
        mul_377 = torch.ops.aten.mul.Tensor(convolution_241, sigmoid_98);  convolution_241 = sigmoid_98 = None
        convolution_242 = torch.ops.aten.convolution.default(mul_377, arg245_1, arg246_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_377 = arg245_1 = arg246_1 = None
        sigmoid_99 = torch.ops.aten.sigmoid.default(convolution_242);  convolution_242 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_376, sigmoid_99);  mul_376 = sigmoid_99 = None
        convolution_243 = torch.ops.aten.convolution.default(mul_378, arg247_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_378 = arg247_1 = None
        add_208 = torch.ops.aten.add.Tensor(arg249_1, 1e-05);  arg249_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_379 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_379, -1);  mul_379 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_243, unsqueeze_745);  convolution_243 = unsqueeze_745 = None
        mul_380 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, unsqueeze_749);  mul_380 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_209 = torch.ops.aten.add.Tensor(mul_381, unsqueeze_751);  mul_381 = unsqueeze_751 = None
        split_with_sizes_158 = torch.ops.aten.split_with_sizes.default(add_209, [80, 80], 1)
        getitem_482 = split_with_sizes_158[0]
        getitem_483 = split_with_sizes_158[1];  split_with_sizes_158 = None
        convolution_244 = torch.ops.aten.convolution.default(getitem_482, arg252_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_482 = arg252_1 = None
        convolution_245 = torch.ops.aten.convolution.default(getitem_483, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_483 = arg253_1 = None
        cat_66 = torch.ops.aten.cat.default([convolution_244, convolution_245], 1);  convolution_244 = convolution_245 = None
        add_210 = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_382 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_382, -1);  mul_382 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(cat_66, unsqueeze_753);  cat_66 = unsqueeze_753 = None
        mul_383 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_757);  mul_383 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_211 = torch.ops.aten.add.Tensor(mul_384, unsqueeze_759);  mul_384 = unsqueeze_759 = None
        sigmoid_100 = torch.ops.aten.sigmoid.default(add_211)
        mul_385 = torch.ops.aten.mul.Tensor(add_211, sigmoid_100);  add_211 = sigmoid_100 = None
        split_with_sizes_160 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1)
        getitem_488 = split_with_sizes_160[0];  split_with_sizes_160 = None
        convolution_246 = torch.ops.aten.convolution.default(getitem_488, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_488 = arg258_1 = None
        split_with_sizes_161 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1)
        getitem_493 = split_with_sizes_161[1];  split_with_sizes_161 = None
        convolution_247 = torch.ops.aten.convolution.default(getitem_493, arg259_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_493 = arg259_1 = None
        split_with_sizes_162 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1)
        getitem_498 = split_with_sizes_162[2];  split_with_sizes_162 = None
        convolution_248 = torch.ops.aten.convolution.default(getitem_498, arg260_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_498 = arg260_1 = None
        split_with_sizes_163 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1);  mul_385 = None
        getitem_503 = split_with_sizes_163[3];  split_with_sizes_163 = None
        convolution_249 = torch.ops.aten.convolution.default(getitem_503, arg261_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_503 = arg261_1 = None
        cat_67 = torch.ops.aten.cat.default([convolution_246, convolution_247, convolution_248, convolution_249], 1);  convolution_246 = convolution_247 = convolution_248 = convolution_249 = None
        add_212 = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_386 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_386, -1);  mul_386 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(cat_67, unsqueeze_761);  cat_67 = unsqueeze_761 = None
        mul_387 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_388 = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_765);  mul_387 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_213 = torch.ops.aten.add.Tensor(mul_388, unsqueeze_767);  mul_388 = unsqueeze_767 = None
        sigmoid_101 = torch.ops.aten.sigmoid.default(add_213)
        mul_389 = torch.ops.aten.mul.Tensor(add_213, sigmoid_101);  add_213 = sigmoid_101 = None
        mean_26 = torch.ops.aten.mean.dim(mul_389, [2, 3], True)
        convolution_250 = torch.ops.aten.convolution.default(mean_26, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg266_1 = arg267_1 = None
        sigmoid_102 = torch.ops.aten.sigmoid.default(convolution_250)
        mul_390 = torch.ops.aten.mul.Tensor(convolution_250, sigmoid_102);  convolution_250 = sigmoid_102 = None
        convolution_251 = torch.ops.aten.convolution.default(mul_390, arg268_1, arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_390 = arg268_1 = arg269_1 = None
        sigmoid_103 = torch.ops.aten.sigmoid.default(convolution_251);  convolution_251 = None
        mul_391 = torch.ops.aten.mul.Tensor(mul_389, sigmoid_103);  mul_389 = sigmoid_103 = None
        split_with_sizes_164 = torch.ops.aten.split_with_sizes.default(mul_391, [240, 240], 1);  mul_391 = None
        getitem_504 = split_with_sizes_164[0]
        getitem_505 = split_with_sizes_164[1];  split_with_sizes_164 = None
        convolution_252 = torch.ops.aten.convolution.default(getitem_504, arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_504 = arg270_1 = None
        convolution_253 = torch.ops.aten.convolution.default(getitem_505, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_505 = arg271_1 = None
        cat_68 = torch.ops.aten.cat.default([convolution_252, convolution_253], 1);  convolution_252 = convolution_253 = None
        add_214 = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_392 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_392, -1);  mul_392 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(cat_68, unsqueeze_769);  cat_68 = unsqueeze_769 = None
        mul_393 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_394 = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_773);  mul_393 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_215 = torch.ops.aten.add.Tensor(mul_394, unsqueeze_775);  mul_394 = unsqueeze_775 = None
        add_216 = torch.ops.aten.add.Tensor(add_215, add_209);  add_215 = add_209 = None
        split_with_sizes_165 = torch.ops.aten.split_with_sizes.default(add_216, [80, 80], 1)
        getitem_506 = split_with_sizes_165[0]
        getitem_507 = split_with_sizes_165[1];  split_with_sizes_165 = None
        convolution_254 = torch.ops.aten.convolution.default(getitem_506, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_506 = arg276_1 = None
        convolution_255 = torch.ops.aten.convolution.default(getitem_507, arg277_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_507 = arg277_1 = None
        cat_69 = torch.ops.aten.cat.default([convolution_254, convolution_255], 1);  convolution_254 = convolution_255 = None
        add_217 = torch.ops.aten.add.Tensor(arg279_1, 1e-05);  arg279_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_395 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_395, -1);  mul_395 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(cat_69, unsqueeze_777);  cat_69 = unsqueeze_777 = None
        mul_396 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_397 = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_781);  mul_396 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_218 = torch.ops.aten.add.Tensor(mul_397, unsqueeze_783);  mul_397 = unsqueeze_783 = None
        sigmoid_104 = torch.ops.aten.sigmoid.default(add_218)
        mul_398 = torch.ops.aten.mul.Tensor(add_218, sigmoid_104);  add_218 = sigmoid_104 = None
        split_with_sizes_167 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1)
        getitem_512 = split_with_sizes_167[0];  split_with_sizes_167 = None
        convolution_256 = torch.ops.aten.convolution.default(getitem_512, arg282_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_512 = arg282_1 = None
        split_with_sizes_168 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1)
        getitem_517 = split_with_sizes_168[1];  split_with_sizes_168 = None
        convolution_257 = torch.ops.aten.convolution.default(getitem_517, arg283_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_517 = arg283_1 = None
        split_with_sizes_169 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1)
        getitem_522 = split_with_sizes_169[2];  split_with_sizes_169 = None
        convolution_258 = torch.ops.aten.convolution.default(getitem_522, arg284_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_522 = arg284_1 = None
        split_with_sizes_170 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1);  mul_398 = None
        getitem_527 = split_with_sizes_170[3];  split_with_sizes_170 = None
        convolution_259 = torch.ops.aten.convolution.default(getitem_527, arg285_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_527 = arg285_1 = None
        cat_70 = torch.ops.aten.cat.default([convolution_256, convolution_257, convolution_258, convolution_259], 1);  convolution_256 = convolution_257 = convolution_258 = convolution_259 = None
        add_219 = torch.ops.aten.add.Tensor(arg287_1, 1e-05);  arg287_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_399 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(cat_70, unsqueeze_785);  cat_70 = unsqueeze_785 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_789);  mul_400 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_220 = torch.ops.aten.add.Tensor(mul_401, unsqueeze_791);  mul_401 = unsqueeze_791 = None
        sigmoid_105 = torch.ops.aten.sigmoid.default(add_220)
        mul_402 = torch.ops.aten.mul.Tensor(add_220, sigmoid_105);  add_220 = sigmoid_105 = None
        mean_27 = torch.ops.aten.mean.dim(mul_402, [2, 3], True)
        convolution_260 = torch.ops.aten.convolution.default(mean_27, arg290_1, arg291_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg290_1 = arg291_1 = None
        sigmoid_106 = torch.ops.aten.sigmoid.default(convolution_260)
        mul_403 = torch.ops.aten.mul.Tensor(convolution_260, sigmoid_106);  convolution_260 = sigmoid_106 = None
        convolution_261 = torch.ops.aten.convolution.default(mul_403, arg292_1, arg293_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_403 = arg292_1 = arg293_1 = None
        sigmoid_107 = torch.ops.aten.sigmoid.default(convolution_261);  convolution_261 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_402, sigmoid_107);  mul_402 = sigmoid_107 = None
        split_with_sizes_171 = torch.ops.aten.split_with_sizes.default(mul_404, [240, 240], 1);  mul_404 = None
        getitem_528 = split_with_sizes_171[0]
        getitem_529 = split_with_sizes_171[1];  split_with_sizes_171 = None
        convolution_262 = torch.ops.aten.convolution.default(getitem_528, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_528 = arg294_1 = None
        convolution_263 = torch.ops.aten.convolution.default(getitem_529, arg295_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_529 = arg295_1 = None
        cat_71 = torch.ops.aten.cat.default([convolution_262, convolution_263], 1);  convolution_262 = convolution_263 = None
        add_221 = torch.ops.aten.add.Tensor(arg297_1, 1e-05);  arg297_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_405 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(cat_71, unsqueeze_793);  cat_71 = unsqueeze_793 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_797);  mul_406 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_222 = torch.ops.aten.add.Tensor(mul_407, unsqueeze_799);  mul_407 = unsqueeze_799 = None
        add_223 = torch.ops.aten.add.Tensor(add_222, add_216);  add_222 = add_216 = None
        split_with_sizes_172 = torch.ops.aten.split_with_sizes.default(add_223, [80, 80], 1)
        getitem_530 = split_with_sizes_172[0]
        getitem_531 = split_with_sizes_172[1];  split_with_sizes_172 = None
        convolution_264 = torch.ops.aten.convolution.default(getitem_530, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_530 = arg300_1 = None
        convolution_265 = torch.ops.aten.convolution.default(getitem_531, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_531 = arg301_1 = None
        cat_72 = torch.ops.aten.cat.default([convolution_264, convolution_265], 1);  convolution_264 = convolution_265 = None
        add_224 = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_408 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(cat_72, unsqueeze_801);  cat_72 = unsqueeze_801 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_805);  mul_409 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_225 = torch.ops.aten.add.Tensor(mul_410, unsqueeze_807);  mul_410 = unsqueeze_807 = None
        sigmoid_108 = torch.ops.aten.sigmoid.default(add_225)
        mul_411 = torch.ops.aten.mul.Tensor(add_225, sigmoid_108);  add_225 = sigmoid_108 = None
        split_with_sizes_174 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1)
        getitem_536 = split_with_sizes_174[0];  split_with_sizes_174 = None
        convolution_266 = torch.ops.aten.convolution.default(getitem_536, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_536 = arg306_1 = None
        split_with_sizes_175 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1)
        getitem_541 = split_with_sizes_175[1];  split_with_sizes_175 = None
        convolution_267 = torch.ops.aten.convolution.default(getitem_541, arg307_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_541 = arg307_1 = None
        split_with_sizes_176 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1)
        getitem_546 = split_with_sizes_176[2];  split_with_sizes_176 = None
        convolution_268 = torch.ops.aten.convolution.default(getitem_546, arg308_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_546 = arg308_1 = None
        split_with_sizes_177 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1);  mul_411 = None
        getitem_551 = split_with_sizes_177[3];  split_with_sizes_177 = None
        convolution_269 = torch.ops.aten.convolution.default(getitem_551, arg309_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_551 = arg309_1 = None
        cat_73 = torch.ops.aten.cat.default([convolution_266, convolution_267, convolution_268, convolution_269], 1);  convolution_266 = convolution_267 = convolution_268 = convolution_269 = None
        add_226 = torch.ops.aten.add.Tensor(arg311_1, 1e-05);  arg311_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_412 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_412, -1);  mul_412 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(cat_73, unsqueeze_809);  cat_73 = unsqueeze_809 = None
        mul_413 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_414 = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_813);  mul_413 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_227 = torch.ops.aten.add.Tensor(mul_414, unsqueeze_815);  mul_414 = unsqueeze_815 = None
        sigmoid_109 = torch.ops.aten.sigmoid.default(add_227)
        mul_415 = torch.ops.aten.mul.Tensor(add_227, sigmoid_109);  add_227 = sigmoid_109 = None
        mean_28 = torch.ops.aten.mean.dim(mul_415, [2, 3], True)
        convolution_270 = torch.ops.aten.convolution.default(mean_28, arg314_1, arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg314_1 = arg315_1 = None
        sigmoid_110 = torch.ops.aten.sigmoid.default(convolution_270)
        mul_416 = torch.ops.aten.mul.Tensor(convolution_270, sigmoid_110);  convolution_270 = sigmoid_110 = None
        convolution_271 = torch.ops.aten.convolution.default(mul_416, arg316_1, arg317_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_416 = arg316_1 = arg317_1 = None
        sigmoid_111 = torch.ops.aten.sigmoid.default(convolution_271);  convolution_271 = None
        mul_417 = torch.ops.aten.mul.Tensor(mul_415, sigmoid_111);  mul_415 = sigmoid_111 = None
        split_with_sizes_178 = torch.ops.aten.split_with_sizes.default(mul_417, [240, 240], 1);  mul_417 = None
        getitem_552 = split_with_sizes_178[0]
        getitem_553 = split_with_sizes_178[1];  split_with_sizes_178 = None
        convolution_272 = torch.ops.aten.convolution.default(getitem_552, arg318_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_552 = arg318_1 = None
        convolution_273 = torch.ops.aten.convolution.default(getitem_553, arg319_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_553 = arg319_1 = None
        cat_74 = torch.ops.aten.cat.default([convolution_272, convolution_273], 1);  convolution_272 = convolution_273 = None
        add_228 = torch.ops.aten.add.Tensor(arg321_1, 1e-05);  arg321_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_418 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_418, -1);  mul_418 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(cat_74, unsqueeze_817);  cat_74 = unsqueeze_817 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_821);  mul_419 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_229 = torch.ops.aten.add.Tensor(mul_420, unsqueeze_823);  mul_420 = unsqueeze_823 = None
        add_230 = torch.ops.aten.add.Tensor(add_229, add_223);  add_229 = add_223 = None
        convolution_274 = torch.ops.aten.convolution.default(add_230, arg324_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_230 = arg324_1 = None
        add_231 = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_421 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_421, -1);  mul_421 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_825);  convolution_274 = unsqueeze_825 = None
        mul_422 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_422, unsqueeze_829);  mul_422 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_232 = torch.ops.aten.add.Tensor(mul_423, unsqueeze_831);  mul_423 = unsqueeze_831 = None
        sigmoid_112 = torch.ops.aten.sigmoid.default(add_232)
        mul_424 = torch.ops.aten.mul.Tensor(add_232, sigmoid_112);  add_232 = sigmoid_112 = None
        split_with_sizes_180 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1)
        getitem_558 = split_with_sizes_180[0];  split_with_sizes_180 = None
        convolution_275 = torch.ops.aten.convolution.default(getitem_558, arg329_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  getitem_558 = arg329_1 = None
        split_with_sizes_181 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1)
        getitem_563 = split_with_sizes_181[1];  split_with_sizes_181 = None
        convolution_276 = torch.ops.aten.convolution.default(getitem_563, arg330_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 240);  getitem_563 = arg330_1 = None
        split_with_sizes_182 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1)
        getitem_568 = split_with_sizes_182[2];  split_with_sizes_182 = None
        convolution_277 = torch.ops.aten.convolution.default(getitem_568, arg331_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 240);  getitem_568 = arg331_1 = None
        split_with_sizes_183 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1);  mul_424 = None
        getitem_573 = split_with_sizes_183[3];  split_with_sizes_183 = None
        convolution_278 = torch.ops.aten.convolution.default(getitem_573, arg332_1, None, [2, 2], [4, 4], [1, 1], False, [0, 0], 240);  getitem_573 = arg332_1 = None
        cat_75 = torch.ops.aten.cat.default([convolution_275, convolution_276, convolution_277, convolution_278], 1);  convolution_275 = convolution_276 = convolution_277 = convolution_278 = None
        add_233 = torch.ops.aten.add.Tensor(arg334_1, 1e-05);  arg334_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_233);  add_233 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_425 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_425, -1);  mul_425 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(cat_75, unsqueeze_833);  cat_75 = unsqueeze_833 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_427 = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_837);  mul_426 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_234 = torch.ops.aten.add.Tensor(mul_427, unsqueeze_839);  mul_427 = unsqueeze_839 = None
        sigmoid_113 = torch.ops.aten.sigmoid.default(add_234)
        mul_428 = torch.ops.aten.mul.Tensor(add_234, sigmoid_113);  add_234 = sigmoid_113 = None
        mean_29 = torch.ops.aten.mean.dim(mul_428, [2, 3], True)
        convolution_279 = torch.ops.aten.convolution.default(mean_29, arg337_1, arg338_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg337_1 = arg338_1 = None
        sigmoid_114 = torch.ops.aten.sigmoid.default(convolution_279)
        mul_429 = torch.ops.aten.mul.Tensor(convolution_279, sigmoid_114);  convolution_279 = sigmoid_114 = None
        convolution_280 = torch.ops.aten.convolution.default(mul_429, arg339_1, arg340_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_429 = arg339_1 = arg340_1 = None
        sigmoid_115 = torch.ops.aten.sigmoid.default(convolution_280);  convolution_280 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_428, sigmoid_115);  mul_428 = sigmoid_115 = None
        convolution_281 = torch.ops.aten.convolution.default(mul_430, arg341_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_430 = arg341_1 = None
        add_235 = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_235);  add_235 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_431 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_431, -1);  mul_431 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_281, unsqueeze_841);  convolution_281 = unsqueeze_841 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_433 = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_845);  mul_432 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_236 = torch.ops.aten.add.Tensor(mul_433, unsqueeze_847);  mul_433 = unsqueeze_847 = None
        convolution_282 = torch.ops.aten.convolution.default(add_236, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg346_1 = None
        add_237 = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_434 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_434, -1);  mul_434 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_849);  convolution_282 = unsqueeze_849 = None
        mul_435 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_436 = torch.ops.aten.mul.Tensor(mul_435, unsqueeze_853);  mul_435 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_238 = torch.ops.aten.add.Tensor(mul_436, unsqueeze_855);  mul_436 = unsqueeze_855 = None
        sigmoid_116 = torch.ops.aten.sigmoid.default(add_238)
        mul_437 = torch.ops.aten.mul.Tensor(add_238, sigmoid_116);  add_238 = sigmoid_116 = None
        split_with_sizes_185 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1)
        getitem_578 = split_with_sizes_185[0];  split_with_sizes_185 = None
        convolution_283 = torch.ops.aten.convolution.default(getitem_578, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_578 = arg351_1 = None
        split_with_sizes_186 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1)
        getitem_583 = split_with_sizes_186[1];  split_with_sizes_186 = None
        convolution_284 = torch.ops.aten.convolution.default(getitem_583, arg352_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_583 = arg352_1 = None
        split_with_sizes_187 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1)
        getitem_588 = split_with_sizes_187[2];  split_with_sizes_187 = None
        convolution_285 = torch.ops.aten.convolution.default(getitem_588, arg353_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_588 = arg353_1 = None
        split_with_sizes_188 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1);  mul_437 = None
        getitem_593 = split_with_sizes_188[3];  split_with_sizes_188 = None
        convolution_286 = torch.ops.aten.convolution.default(getitem_593, arg354_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_593 = arg354_1 = None
        cat_76 = torch.ops.aten.cat.default([convolution_283, convolution_284, convolution_285, convolution_286], 1);  convolution_283 = convolution_284 = convolution_285 = convolution_286 = None
        add_239 = torch.ops.aten.add.Tensor(arg356_1, 1e-05);  arg356_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(cat_76, unsqueeze_857);  cat_76 = unsqueeze_857 = None
        mul_439 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_861);  mul_439 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_240 = torch.ops.aten.add.Tensor(mul_440, unsqueeze_863);  mul_440 = unsqueeze_863 = None
        sigmoid_117 = torch.ops.aten.sigmoid.default(add_240)
        mul_441 = torch.ops.aten.mul.Tensor(add_240, sigmoid_117);  add_240 = sigmoid_117 = None
        mean_30 = torch.ops.aten.mean.dim(mul_441, [2, 3], True)
        convolution_287 = torch.ops.aten.convolution.default(mean_30, arg359_1, arg360_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg359_1 = arg360_1 = None
        sigmoid_118 = torch.ops.aten.sigmoid.default(convolution_287)
        mul_442 = torch.ops.aten.mul.Tensor(convolution_287, sigmoid_118);  convolution_287 = sigmoid_118 = None
        convolution_288 = torch.ops.aten.convolution.default(mul_442, arg361_1, arg362_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_442 = arg361_1 = arg362_1 = None
        sigmoid_119 = torch.ops.aten.sigmoid.default(convolution_288);  convolution_288 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_441, sigmoid_119);  mul_441 = sigmoid_119 = None
        split_with_sizes_189 = torch.ops.aten.split_with_sizes.default(mul_443, [792, 792], 1);  mul_443 = None
        getitem_594 = split_with_sizes_189[0]
        getitem_595 = split_with_sizes_189[1];  split_with_sizes_189 = None
        convolution_289 = torch.ops.aten.convolution.default(getitem_594, arg363_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_594 = arg363_1 = None
        convolution_290 = torch.ops.aten.convolution.default(getitem_595, arg364_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_595 = arg364_1 = None
        cat_77 = torch.ops.aten.cat.default([convolution_289, convolution_290], 1);  convolution_289 = convolution_290 = None
        add_241 = torch.ops.aten.add.Tensor(arg366_1, 1e-05);  arg366_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_444 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(cat_77, unsqueeze_865);  cat_77 = unsqueeze_865 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_869);  mul_445 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_242 = torch.ops.aten.add.Tensor(mul_446, unsqueeze_871);  mul_446 = unsqueeze_871 = None
        add_243 = torch.ops.aten.add.Tensor(add_242, add_236);  add_242 = add_236 = None
        convolution_291 = torch.ops.aten.convolution.default(add_243, arg369_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg369_1 = None
        add_244 = torch.ops.aten.add.Tensor(arg371_1, 1e-05);  arg371_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_447 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_291, unsqueeze_873);  convolution_291 = unsqueeze_873 = None
        mul_448 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_449 = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_877);  mul_448 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_245 = torch.ops.aten.add.Tensor(mul_449, unsqueeze_879);  mul_449 = unsqueeze_879 = None
        sigmoid_120 = torch.ops.aten.sigmoid.default(add_245)
        mul_450 = torch.ops.aten.mul.Tensor(add_245, sigmoid_120);  add_245 = sigmoid_120 = None
        split_with_sizes_191 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1)
        getitem_600 = split_with_sizes_191[0];  split_with_sizes_191 = None
        convolution_292 = torch.ops.aten.convolution.default(getitem_600, arg374_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_600 = arg374_1 = None
        split_with_sizes_192 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1)
        getitem_605 = split_with_sizes_192[1];  split_with_sizes_192 = None
        convolution_293 = torch.ops.aten.convolution.default(getitem_605, arg375_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_605 = arg375_1 = None
        split_with_sizes_193 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1)
        getitem_610 = split_with_sizes_193[2];  split_with_sizes_193 = None
        convolution_294 = torch.ops.aten.convolution.default(getitem_610, arg376_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_610 = arg376_1 = None
        split_with_sizes_194 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1);  mul_450 = None
        getitem_615 = split_with_sizes_194[3];  split_with_sizes_194 = None
        convolution_295 = torch.ops.aten.convolution.default(getitem_615, arg377_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_615 = arg377_1 = None
        cat_78 = torch.ops.aten.cat.default([convolution_292, convolution_293, convolution_294, convolution_295], 1);  convolution_292 = convolution_293 = convolution_294 = convolution_295 = None
        add_246 = torch.ops.aten.add.Tensor(arg379_1, 1e-05);  arg379_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_451 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_451, -1);  mul_451 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(cat_78, unsqueeze_881);  cat_78 = unsqueeze_881 = None
        mul_452 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_453 = torch.ops.aten.mul.Tensor(mul_452, unsqueeze_885);  mul_452 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg381_1, -1);  arg381_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_247 = torch.ops.aten.add.Tensor(mul_453, unsqueeze_887);  mul_453 = unsqueeze_887 = None
        sigmoid_121 = torch.ops.aten.sigmoid.default(add_247)
        mul_454 = torch.ops.aten.mul.Tensor(add_247, sigmoid_121);  add_247 = sigmoid_121 = None
        mean_31 = torch.ops.aten.mean.dim(mul_454, [2, 3], True)
        convolution_296 = torch.ops.aten.convolution.default(mean_31, arg382_1, arg383_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg382_1 = arg383_1 = None
        sigmoid_122 = torch.ops.aten.sigmoid.default(convolution_296)
        mul_455 = torch.ops.aten.mul.Tensor(convolution_296, sigmoid_122);  convolution_296 = sigmoid_122 = None
        convolution_297 = torch.ops.aten.convolution.default(mul_455, arg384_1, arg385_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_455 = arg384_1 = arg385_1 = None
        sigmoid_123 = torch.ops.aten.sigmoid.default(convolution_297);  convolution_297 = None
        mul_456 = torch.ops.aten.mul.Tensor(mul_454, sigmoid_123);  mul_454 = sigmoid_123 = None
        split_with_sizes_195 = torch.ops.aten.split_with_sizes.default(mul_456, [792, 792], 1);  mul_456 = None
        getitem_616 = split_with_sizes_195[0]
        getitem_617 = split_with_sizes_195[1];  split_with_sizes_195 = None
        convolution_298 = torch.ops.aten.convolution.default(getitem_616, arg386_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_616 = arg386_1 = None
        convolution_299 = torch.ops.aten.convolution.default(getitem_617, arg387_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_617 = arg387_1 = None
        cat_79 = torch.ops.aten.cat.default([convolution_298, convolution_299], 1);  convolution_298 = convolution_299 = None
        add_248 = torch.ops.aten.add.Tensor(arg389_1, 1e-05);  arg389_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_457 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_457, -1);  mul_457 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(cat_79, unsqueeze_889);  cat_79 = unsqueeze_889 = None
        mul_458 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_459 = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_893);  mul_458 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_249 = torch.ops.aten.add.Tensor(mul_459, unsqueeze_895);  mul_459 = unsqueeze_895 = None
        add_250 = torch.ops.aten.add.Tensor(add_249, add_243);  add_249 = add_243 = None
        convolution_300 = torch.ops.aten.convolution.default(add_250, arg392_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg392_1 = None
        add_251 = torch.ops.aten.add.Tensor(arg394_1, 1e-05);  arg394_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_251);  add_251 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_460 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_460, -1);  mul_460 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_300, unsqueeze_897);  convolution_300 = unsqueeze_897 = None
        mul_461 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_462 = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_901);  mul_461 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_252 = torch.ops.aten.add.Tensor(mul_462, unsqueeze_903);  mul_462 = unsqueeze_903 = None
        sigmoid_124 = torch.ops.aten.sigmoid.default(add_252)
        mul_463 = torch.ops.aten.mul.Tensor(add_252, sigmoid_124);  add_252 = sigmoid_124 = None
        split_with_sizes_197 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1)
        getitem_622 = split_with_sizes_197[0];  split_with_sizes_197 = None
        convolution_301 = torch.ops.aten.convolution.default(getitem_622, arg397_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_622 = arg397_1 = None
        split_with_sizes_198 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1)
        getitem_627 = split_with_sizes_198[1];  split_with_sizes_198 = None
        convolution_302 = torch.ops.aten.convolution.default(getitem_627, arg398_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_627 = arg398_1 = None
        split_with_sizes_199 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1)
        getitem_632 = split_with_sizes_199[2];  split_with_sizes_199 = None
        convolution_303 = torch.ops.aten.convolution.default(getitem_632, arg399_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_632 = arg399_1 = None
        split_with_sizes_200 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1);  mul_463 = None
        getitem_637 = split_with_sizes_200[3];  split_with_sizes_200 = None
        convolution_304 = torch.ops.aten.convolution.default(getitem_637, arg400_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_637 = arg400_1 = None
        cat_80 = torch.ops.aten.cat.default([convolution_301, convolution_302, convolution_303, convolution_304], 1);  convolution_301 = convolution_302 = convolution_303 = convolution_304 = None
        add_253 = torch.ops.aten.add.Tensor(arg402_1, 1e-05);  arg402_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_253);  add_253 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_464 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_464, -1);  mul_464 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(cat_80, unsqueeze_905);  cat_80 = unsqueeze_905 = None
        mul_465 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_466 = torch.ops.aten.mul.Tensor(mul_465, unsqueeze_909);  mul_465 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_254 = torch.ops.aten.add.Tensor(mul_466, unsqueeze_911);  mul_466 = unsqueeze_911 = None
        sigmoid_125 = torch.ops.aten.sigmoid.default(add_254)
        mul_467 = torch.ops.aten.mul.Tensor(add_254, sigmoid_125);  add_254 = sigmoid_125 = None
        mean_32 = torch.ops.aten.mean.dim(mul_467, [2, 3], True)
        convolution_305 = torch.ops.aten.convolution.default(mean_32, arg405_1, arg406_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg405_1 = arg406_1 = None
        sigmoid_126 = torch.ops.aten.sigmoid.default(convolution_305)
        mul_468 = torch.ops.aten.mul.Tensor(convolution_305, sigmoid_126);  convolution_305 = sigmoid_126 = None
        convolution_306 = torch.ops.aten.convolution.default(mul_468, arg407_1, arg408_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_468 = arg407_1 = arg408_1 = None
        sigmoid_127 = torch.ops.aten.sigmoid.default(convolution_306);  convolution_306 = None
        mul_469 = torch.ops.aten.mul.Tensor(mul_467, sigmoid_127);  mul_467 = sigmoid_127 = None
        split_with_sizes_201 = torch.ops.aten.split_with_sizes.default(mul_469, [792, 792], 1);  mul_469 = None
        getitem_638 = split_with_sizes_201[0]
        getitem_639 = split_with_sizes_201[1];  split_with_sizes_201 = None
        convolution_307 = torch.ops.aten.convolution.default(getitem_638, arg409_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_638 = arg409_1 = None
        convolution_308 = torch.ops.aten.convolution.default(getitem_639, arg410_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_639 = arg410_1 = None
        cat_81 = torch.ops.aten.cat.default([convolution_307, convolution_308], 1);  convolution_307 = convolution_308 = None
        add_255 = torch.ops.aten.add.Tensor(arg412_1, 1e-05);  arg412_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_470 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_470, -1);  mul_470 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(cat_81, unsqueeze_913);  cat_81 = unsqueeze_913 = None
        mul_471 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_472 = torch.ops.aten.mul.Tensor(mul_471, unsqueeze_917);  mul_471 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_256 = torch.ops.aten.add.Tensor(mul_472, unsqueeze_919);  mul_472 = unsqueeze_919 = None
        add_257 = torch.ops.aten.add.Tensor(add_256, add_250);  add_256 = add_250 = None
        convolution_309 = torch.ops.aten.convolution.default(add_257, arg415_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_257 = arg415_1 = None
        add_258 = torch.ops.aten.add.Tensor(arg417_1, 1e-05);  arg417_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_258);  add_258 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_473 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_473, -1);  mul_473 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_309, unsqueeze_921);  convolution_309 = unsqueeze_921 = None
        mul_474 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_475 = torch.ops.aten.mul.Tensor(mul_474, unsqueeze_925);  mul_474 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_259 = torch.ops.aten.add.Tensor(mul_475, unsqueeze_927);  mul_475 = unsqueeze_927 = None
        relu_13 = torch.ops.aten.relu.default(add_259);  add_259 = None
        mean_33 = torch.ops.aten.mean.dim(relu_13, [-1, -2], True);  relu_13 = None
        view_1 = torch.ops.aten.view.default(mean_33, [8, 1536]);  mean_33 = None
        permute_1 = torch.ops.aten.permute.default(arg420_1, [1, 0]);  arg420_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg421_1, view_1, permute_1);  arg421_1 = view_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf3, (32,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf4, (32,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf5, (32,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf6, (32, 1, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf7, (32,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf8, (32,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf9, (32,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf10, (32,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf11, (32, 32, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf15, (32,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (96, 16, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf17, (96, 16, 1, 1), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf18, (192,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf19, (192,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf20, (192,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf21, (192,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64, 1, 3, 3), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 6400, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64, 1, 5, 5), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 12544, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64, 1, 7, 7), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf25, (192,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf26, (192,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf27, (192,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf28, (192,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf29, (20, 96, 1, 1), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf30, (20, 96, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf31, (40,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf32, (40,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf33, (40,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf34, (40,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf35, (60, 20, 1, 1), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf36, (60, 20, 1, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf37, (120,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf38, (120,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf39, (120,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf40, (120,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf41, (120, 1, 3, 3), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf42, (120,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf43, (120,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf44, (120,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf45, (120,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf46, (20, 60, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 4800, device=device(type='cuda', index=0))
    reader.tensor(buf47, (20, 60, 1, 1), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf48, (40,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf49, (40,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf50, (40,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf51, (40,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf52, (240, 40, 1, 1), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf53, (240,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf54, (240,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf55, (240,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf56, (240,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2160, device=device(type='cuda', index=0))
    reader.tensor(buf57, (60, 1, 3, 3), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 6000, device=device(type='cuda', index=0))
    reader.tensor(buf58, (60, 1, 5, 5), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 11760, device=device(type='cuda', index=0))
    reader.tensor(buf59, (60, 1, 7, 7), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 19440, device=device(type='cuda', index=0))
    reader.tensor(buf60, (60, 1, 9, 9), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf61, (240,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf62, (240,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf63, (240,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf64, (240,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf65, (20, 240, 1, 1), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf66, (20,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf67, (240, 20, 1, 1), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf68, (240,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 53760, device=device(type='cuda', index=0))
    reader.tensor(buf69, (56, 240, 1, 1), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf70, (56,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf71, (56,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf72, (56,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf73, (56,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf74, (168, 28, 1, 1), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf75, (168, 28, 1, 1), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf76, (336,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf77, (336,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf78, (336,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf79, (336,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 6048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (168, 1, 3, 3), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 16800, device=device(type='cuda', index=0))
    reader.tensor(buf81, (168, 1, 5, 5), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf82, (336,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf83, (336,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf84, (336,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf85, (336,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf86, (28, 336, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf87, (28,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf88, (336, 28, 1, 1), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf89, (336,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf90, (28, 168, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf91, (28, 168, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf92, (56,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf93, (56,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf94, (56,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf95, (56,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf96, (168, 28, 1, 1), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf97, (168, 28, 1, 1), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf98, (336,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf99, (336,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf100, (336,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf101, (336,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 6048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (168, 1, 3, 3), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 16800, device=device(type='cuda', index=0))
    reader.tensor(buf103, (168, 1, 5, 5), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf104, (336,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf105, (336,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf106, (336,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf107, (336,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf108, (28, 336, 1, 1), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf109, (28,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf110, (336, 28, 1, 1), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf111, (336,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf112, (28, 168, 1, 1), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf113, (28, 168, 1, 1), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf114, (56,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf115, (56,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf116, (56,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf117, (56,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf118, (168, 28, 1, 1), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf119, (168, 28, 1, 1), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf120, (336,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf121, (336,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf122, (336,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf123, (336,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 6048, device=device(type='cuda', index=0))
    reader.tensor(buf124, (168, 1, 3, 3), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 16800, device=device(type='cuda', index=0))
    reader.tensor(buf125, (168, 1, 5, 5), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf126, (336,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf127, (336,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf128, (336,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf129, (336,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf130, (28, 336, 1, 1), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf131, (28,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf132, (336, 28, 1, 1), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf133, (336,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf134, (28, 168, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf135, (28, 168, 1, 1), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf136, (56,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf137, (56,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf138, (56,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 224, device=device(type='cuda', index=0))
    reader.tensor(buf139, (56,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf140, (336, 56, 1, 1), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf141, (336,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf142, (336,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf143, (336,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf144, (336,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 4032, device=device(type='cuda', index=0))
    reader.tensor(buf145, (112, 1, 3, 3), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 11200, device=device(type='cuda', index=0))
    reader.tensor(buf146, (112, 1, 5, 5), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 21952, device=device(type='cuda', index=0))
    reader.tensor(buf147, (112, 1, 7, 7), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf148, (336,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf149, (336,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf150, (336,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf151, (336,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf152, (14, 336, 1, 1), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 56, device=device(type='cuda', index=0))
    reader.tensor(buf153, (14,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf154, (336, 14, 1, 1), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1344, device=device(type='cuda', index=0))
    reader.tensor(buf155, (336,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 139776, device=device(type='cuda', index=0))
    reader.tensor(buf156, (104, 336, 1, 1), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf157, (104,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf158, (104,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf159, (104,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf160, (104,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf161, (312, 52, 1, 1), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf162, (312, 52, 1, 1), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf163, (624,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf164, (624,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf165, (624,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf166, (624,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 5616, device=device(type='cuda', index=0))
    reader.tensor(buf167, (156, 1, 3, 3), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 15600, device=device(type='cuda', index=0))
    reader.tensor(buf168, (156, 1, 5, 5), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 30576, device=device(type='cuda', index=0))
    reader.tensor(buf169, (156, 1, 7, 7), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 50544, device=device(type='cuda', index=0))
    reader.tensor(buf170, (156, 1, 9, 9), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf171, (624,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf172, (624,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf173, (624,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf174, (624,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf175, (26, 624, 1, 1), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 104, device=device(type='cuda', index=0))
    reader.tensor(buf176, (26,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf177, (624, 26, 1, 1), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf178, (624,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf179, (52, 312, 1, 1), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf180, (52, 312, 1, 1), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf181, (104,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf182, (104,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf183, (104,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf184, (104,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf185, (312, 52, 1, 1), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf186, (312, 52, 1, 1), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf187, (624,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf188, (624,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf189, (624,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf190, (624,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 5616, device=device(type='cuda', index=0))
    reader.tensor(buf191, (156, 1, 3, 3), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 15600, device=device(type='cuda', index=0))
    reader.tensor(buf192, (156, 1, 5, 5), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 30576, device=device(type='cuda', index=0))
    reader.tensor(buf193, (156, 1, 7, 7), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 50544, device=device(type='cuda', index=0))
    reader.tensor(buf194, (156, 1, 9, 9), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf195, (624,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf196, (624,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf197, (624,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf198, (624,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf199, (26, 624, 1, 1), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 104, device=device(type='cuda', index=0))
    reader.tensor(buf200, (26,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf201, (624, 26, 1, 1), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf202, (624,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf203, (52, 312, 1, 1), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf204, (52, 312, 1, 1), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf205, (104,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf206, (104,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf207, (104,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf208, (104,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf209, (312, 52, 1, 1), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf210, (312, 52, 1, 1), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf211, (624,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf212, (624,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf213, (624,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf214, (624,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 5616, device=device(type='cuda', index=0))
    reader.tensor(buf215, (156, 1, 3, 3), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 15600, device=device(type='cuda', index=0))
    reader.tensor(buf216, (156, 1, 5, 5), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 30576, device=device(type='cuda', index=0))
    reader.tensor(buf217, (156, 1, 7, 7), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 50544, device=device(type='cuda', index=0))
    reader.tensor(buf218, (156, 1, 9, 9), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf219, (624,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf220, (624,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf221, (624,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf222, (624,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf223, (26, 624, 1, 1), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 104, device=device(type='cuda', index=0))
    reader.tensor(buf224, (26,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf225, (624, 26, 1, 1), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf226, (624,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf227, (52, 312, 1, 1), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 64896, device=device(type='cuda', index=0))
    reader.tensor(buf228, (52, 312, 1, 1), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf229, (104,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf230, (104,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf231, (104,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 416, device=device(type='cuda', index=0))
    reader.tensor(buf232, (104,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 259584, device=device(type='cuda', index=0))
    reader.tensor(buf233, (624, 104, 1, 1), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf234, (624,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf235, (624,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf236, (624,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf237, (624,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 22464, device=device(type='cuda', index=0))
    reader.tensor(buf238, (624, 1, 3, 3), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf239, (624,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf240, (624,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf241, (624,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf242, (624,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 129792, device=device(type='cuda', index=0))
    reader.tensor(buf243, (52, 624, 1, 1), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 208, device=device(type='cuda', index=0))
    reader.tensor(buf244, (52,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 129792, device=device(type='cuda', index=0))
    reader.tensor(buf245, (624, 52, 1, 1), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 2496, device=device(type='cuda', index=0))
    reader.tensor(buf246, (624,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 399360, device=device(type='cuda', index=0))
    reader.tensor(buf247, (160, 624, 1, 1), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf248, (160,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf249, (160,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf250, (160,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf251, (160,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf252, (240, 80, 1, 1), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf253, (240, 80, 1, 1), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf254, (480,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf255, (480,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf256, (480,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf257, (480,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf258, (120, 1, 3, 3), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf259, (120, 1, 5, 5), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 23520, device=device(type='cuda', index=0))
    reader.tensor(buf260, (120, 1, 7, 7), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 38880, device=device(type='cuda', index=0))
    reader.tensor(buf261, (120, 1, 9, 9), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf262, (480,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf263, (480,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf264, (480,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf265, (480,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf266, (80, 480, 1, 1), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf267, (80,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf268, (480, 80, 1, 1), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf269, (480,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf270, (80, 240, 1, 1), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf271, (80, 240, 1, 1), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf272, (160,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf273, (160,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf274, (160,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf275, (160,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf276, (240, 80, 1, 1), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf277, (240, 80, 1, 1), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf278, (480,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf279, (480,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf280, (480,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf281, (480,), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf282, (120, 1, 3, 3), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf283, (120, 1, 5, 5), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 23520, device=device(type='cuda', index=0))
    reader.tensor(buf284, (120, 1, 7, 7), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 38880, device=device(type='cuda', index=0))
    reader.tensor(buf285, (120, 1, 9, 9), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf286, (480,), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf287, (480,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf288, (480,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf289, (480,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf290, (80, 480, 1, 1), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf291, (80,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf292, (480, 80, 1, 1), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf293, (480,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf294, (80, 240, 1, 1), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf295, (80, 240, 1, 1), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf296, (160,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf297, (160,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf298, (160,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf299, (160,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf300, (240, 80, 1, 1), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf301, (240, 80, 1, 1), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf302, (480,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf303, (480,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf304, (480,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf305, (480,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf306, (120, 1, 3, 3), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 12000, device=device(type='cuda', index=0))
    reader.tensor(buf307, (120, 1, 5, 5), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 23520, device=device(type='cuda', index=0))
    reader.tensor(buf308, (120, 1, 7, 7), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 38880, device=device(type='cuda', index=0))
    reader.tensor(buf309, (120, 1, 9, 9), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf310, (480,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf311, (480,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf312, (480,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf313, (480,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf314, (80, 480, 1, 1), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf315, (80,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf316, (480, 80, 1, 1), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf317, (480,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf318, (80, 240, 1, 1), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf319, (80, 240, 1, 1), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf320, (160,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf321, (160,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf322, (160,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf323, (160,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf324, (960, 160, 1, 1), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf325, (960,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf326, (960,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf327, (960,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf328, (960,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf329, (240, 1, 3, 3), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 24000, device=device(type='cuda', index=0))
    reader.tensor(buf330, (240, 1, 5, 5), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 47040, device=device(type='cuda', index=0))
    reader.tensor(buf331, (240, 1, 7, 7), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 77760, device=device(type='cuda', index=0))
    reader.tensor(buf332, (240, 1, 9, 9), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf333, (960,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf334, (960,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf335, (960,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf336, (960,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf337, (80, 960, 1, 1), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf338, (80,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf339, (960, 80, 1, 1), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf340, (960,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 1013760, device=device(type='cuda', index=0))
    reader.tensor(buf341, (264, 960, 1, 1), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf342, (264,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf343, (264,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf344, (264,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf345, (264,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 1672704, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1584, 264, 1, 1), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1584,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1584,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1584,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1584,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 14256, device=device(type='cuda', index=0))
    reader.tensor(buf351, (396, 1, 3, 3), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 39600, device=device(type='cuda', index=0))
    reader.tensor(buf352, (396, 1, 5, 5), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 77616, device=device(type='cuda', index=0))
    reader.tensor(buf353, (396, 1, 7, 7), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 128304, device=device(type='cuda', index=0))
    reader.tensor(buf354, (396, 1, 9, 9), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1584,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1584,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf357, (1584,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf358, (1584,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 836352, device=device(type='cuda', index=0))
    reader.tensor(buf359, (132, 1584, 1, 1), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 528, device=device(type='cuda', index=0))
    reader.tensor(buf360, (132,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 836352, device=device(type='cuda', index=0))
    reader.tensor(buf361, (1584, 132, 1, 1), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf362, (1584,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 418176, device=device(type='cuda', index=0))
    reader.tensor(buf363, (132, 792, 1, 1), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 418176, device=device(type='cuda', index=0))
    reader.tensor(buf364, (132, 792, 1, 1), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf365, (264,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf366, (264,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf367, (264,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf368, (264,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 1672704, device=device(type='cuda', index=0))
    reader.tensor(buf369, (1584, 264, 1, 1), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf370, (1584,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf371, (1584,), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf372, (1584,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf373, (1584,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 14256, device=device(type='cuda', index=0))
    reader.tensor(buf374, (396, 1, 3, 3), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 39600, device=device(type='cuda', index=0))
    reader.tensor(buf375, (396, 1, 5, 5), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 77616, device=device(type='cuda', index=0))
    reader.tensor(buf376, (396, 1, 7, 7), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 128304, device=device(type='cuda', index=0))
    reader.tensor(buf377, (396, 1, 9, 9), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf378, (1584,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf379, (1584,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf380, (1584,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf381, (1584,), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 836352, device=device(type='cuda', index=0))
    reader.tensor(buf382, (132, 1584, 1, 1), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 528, device=device(type='cuda', index=0))
    reader.tensor(buf383, (132,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 836352, device=device(type='cuda', index=0))
    reader.tensor(buf384, (1584, 132, 1, 1), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf385, (1584,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 418176, device=device(type='cuda', index=0))
    reader.tensor(buf386, (132, 792, 1, 1), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 418176, device=device(type='cuda', index=0))
    reader.tensor(buf387, (132, 792, 1, 1), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf388, (264,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf389, (264,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf390, (264,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf391, (264,), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 1672704, device=device(type='cuda', index=0))
    reader.tensor(buf392, (1584, 264, 1, 1), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf393, (1584,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf394, (1584,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf395, (1584,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf396, (1584,), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 14256, device=device(type='cuda', index=0))
    reader.tensor(buf397, (396, 1, 3, 3), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 39600, device=device(type='cuda', index=0))
    reader.tensor(buf398, (396, 1, 5, 5), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 77616, device=device(type='cuda', index=0))
    reader.tensor(buf399, (396, 1, 7, 7), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 128304, device=device(type='cuda', index=0))
    reader.tensor(buf400, (396, 1, 9, 9), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf401, (1584,), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf402, (1584,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf403, (1584,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf404, (1584,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 836352, device=device(type='cuda', index=0))
    reader.tensor(buf405, (132, 1584, 1, 1), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 528, device=device(type='cuda', index=0))
    reader.tensor(buf406, (132,), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 836352, device=device(type='cuda', index=0))
    reader.tensor(buf407, (1584, 132, 1, 1), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 6336, device=device(type='cuda', index=0))
    reader.tensor(buf408, (1584,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 418176, device=device(type='cuda', index=0))
    reader.tensor(buf409, (132, 792, 1, 1), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 418176, device=device(type='cuda', index=0))
    reader.tensor(buf410, (132, 792, 1, 1), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf411, (264,), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf412, (264,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf413, (264,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 1056, device=device(type='cuda', index=0))
    reader.tensor(buf414, (264,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 1622016, device=device(type='cuda', index=0))
    reader.tensor(buf415, (1536, 264, 1, 1), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf416, (1536,), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf417, (1536,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf418, (1536,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf419, (1536,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 6144000, device=device(type='cuda', index=0))
    reader.tensor(buf420, (1000, 1536), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf421, (1000,), is_leaf=True)  # arg421_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)