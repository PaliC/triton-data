
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1):
        convolution_96 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_128 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_251 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_465);  convolution_96 = unsqueeze_465 = None
        mul_252 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_469);  mul_252 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_129 = torch.ops.aten.add.Tensor(mul_253, unsqueeze_471);  mul_253 = unsqueeze_471 = None
        sigmoid_77 = torch.ops.aten.sigmoid.default(add_129)
        mul_254 = torch.ops.aten.mul.Tensor(add_129, sigmoid_77);  add_129 = sigmoid_77 = None
        convolution_97 = torch.ops.aten.convolution.default(mul_254, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  mul_254 = arg6_1 = None
        add_130 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_255 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_473);  convolution_97 = unsqueeze_473 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_477);  mul_256 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_131 = torch.ops.aten.add.Tensor(mul_257, unsqueeze_479);  mul_257 = unsqueeze_479 = None
        sigmoid_78 = torch.ops.aten.sigmoid.default(add_131)
        mul_258 = torch.ops.aten.mul.Tensor(add_131, sigmoid_78);  add_131 = sigmoid_78 = None
        mean_20 = torch.ops.aten.mean.dim(mul_258, [2, 3], True)
        convolution_98 = torch.ops.aten.convolution.default(mean_20, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg11_1 = arg12_1 = None
        sigmoid_79 = torch.ops.aten.sigmoid.default(convolution_98)
        mul_259 = torch.ops.aten.mul.Tensor(convolution_98, sigmoid_79);  convolution_98 = sigmoid_79 = None
        convolution_99 = torch.ops.aten.convolution.default(mul_259, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_259 = arg13_1 = arg14_1 = None
        sigmoid_80 = torch.ops.aten.sigmoid.default(convolution_99);  convolution_99 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_258, sigmoid_80);  mul_258 = sigmoid_80 = None
        convolution_100 = torch.ops.aten.convolution.default(mul_260, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_260 = arg15_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg17_1, 1e-05);  arg17_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_261 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60 = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_481);  convolution_100 = unsqueeze_481 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_485);  mul_262 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_133 = torch.ops.aten.add.Tensor(mul_263, unsqueeze_487);  mul_263 = unsqueeze_487 = None
        convolution_101 = torch.ops.aten.convolution.default(add_133, arg20_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_133 = arg20_1 = None
        add_134 = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61 = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_489);  convolution_101 = unsqueeze_489 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_493);  mul_265 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_135 = torch.ops.aten.add.Tensor(mul_266, unsqueeze_495);  mul_266 = unsqueeze_495 = None
        sigmoid_81 = torch.ops.aten.sigmoid.default(add_135)
        mul_267 = torch.ops.aten.mul.Tensor(add_135, sigmoid_81);  add_135 = sigmoid_81 = None
        convolution_102 = torch.ops.aten.convolution.default(mul_267, arg25_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96);  mul_267 = arg25_1 = None
        add_136 = torch.ops.aten.add.Tensor(arg27_1, 1e-05);  arg27_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_268 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_268, -1);  mul_268 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_497);  convolution_102 = unsqueeze_497 = None
        mul_269 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_501);  mul_269 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_137 = torch.ops.aten.add.Tensor(mul_270, unsqueeze_503);  mul_270 = unsqueeze_503 = None
        sigmoid_82 = torch.ops.aten.sigmoid.default(add_137)
        mul_271 = torch.ops.aten.mul.Tensor(add_137, sigmoid_82);  add_137 = sigmoid_82 = None
        mean_21 = torch.ops.aten.mean.dim(mul_271, [2, 3], True)
        convolution_103 = torch.ops.aten.convolution.default(mean_21, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg30_1 = arg31_1 = None
        sigmoid_83 = torch.ops.aten.sigmoid.default(convolution_103)
        mul_272 = torch.ops.aten.mul.Tensor(convolution_103, sigmoid_83);  convolution_103 = sigmoid_83 = None
        convolution_104 = torch.ops.aten.convolution.default(mul_272, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_272 = arg32_1 = arg33_1 = None
        sigmoid_84 = torch.ops.aten.sigmoid.default(convolution_104);  convolution_104 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_271, sigmoid_84);  mul_271 = sigmoid_84 = None
        convolution_105 = torch.ops.aten.convolution.default(mul_273, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg34_1 = None
        add_138 = torch.ops.aten.add.Tensor(arg36_1, 1e-05);  arg36_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_274 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_505);  convolution_105 = unsqueeze_505 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_509);  mul_275 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_139 = torch.ops.aten.add.Tensor(mul_276, unsqueeze_511);  mul_276 = unsqueeze_511 = None
        convolution_106 = torch.ops.aten.convolution.default(add_139, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg39_1 = None
        add_140 = torch.ops.aten.add.Tensor(arg41_1, 1e-05);  arg41_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_277 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_513);  convolution_106 = unsqueeze_513 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_517);  mul_278 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_141 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_519);  mul_279 = unsqueeze_519 = None
        sigmoid_85 = torch.ops.aten.sigmoid.default(add_141)
        mul_280 = torch.ops.aten.mul.Tensor(add_141, sigmoid_85);  add_141 = sigmoid_85 = None
        convolution_107 = torch.ops.aten.convolution.default(mul_280, arg44_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144);  mul_280 = arg44_1 = None
        add_142 = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_281 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_281, -1);  mul_281 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_521);  convolution_107 = unsqueeze_521 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_283 = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_525);  mul_282 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_143 = torch.ops.aten.add.Tensor(mul_283, unsqueeze_527);  mul_283 = unsqueeze_527 = None
        sigmoid_86 = torch.ops.aten.sigmoid.default(add_143)
        mul_284 = torch.ops.aten.mul.Tensor(add_143, sigmoid_86);  add_143 = sigmoid_86 = None
        mean_22 = torch.ops.aten.mean.dim(mul_284, [2, 3], True)
        convolution_108 = torch.ops.aten.convolution.default(mean_22, arg49_1, arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg49_1 = arg50_1 = None
        sigmoid_87 = torch.ops.aten.sigmoid.default(convolution_108)
        mul_285 = torch.ops.aten.mul.Tensor(convolution_108, sigmoid_87);  convolution_108 = sigmoid_87 = None
        convolution_109 = torch.ops.aten.convolution.default(mul_285, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_285 = arg51_1 = arg52_1 = None
        sigmoid_88 = torch.ops.aten.sigmoid.default(convolution_109);  convolution_109 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_284, sigmoid_88);  mul_284 = sigmoid_88 = None
        convolution_110 = torch.ops.aten.convolution.default(mul_286, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg53_1 = None
        add_144 = torch.ops.aten.add.Tensor(arg55_1, 1e-05);  arg55_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_287 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_287, -1);  mul_287 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_529);  convolution_110 = unsqueeze_529 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_533);  mul_288 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_145 = torch.ops.aten.add.Tensor(mul_289, unsqueeze_535);  mul_289 = unsqueeze_535 = None
        add_146 = torch.ops.aten.add.Tensor(add_145, add_139);  add_145 = add_139 = None
        convolution_111 = torch.ops.aten.convolution.default(add_146, arg58_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_146 = arg58_1 = None
        add_147 = torch.ops.aten.add.Tensor(arg60_1, 1e-05);  arg60_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_290 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_290, -1);  mul_290 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_537);  convolution_111 = unsqueeze_537 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_541);  mul_291 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_148 = torch.ops.aten.add.Tensor(mul_292, unsqueeze_543);  mul_292 = unsqueeze_543 = None
        sigmoid_89 = torch.ops.aten.sigmoid.default(add_148)
        mul_293 = torch.ops.aten.mul.Tensor(add_148, sigmoid_89);  add_148 = sigmoid_89 = None
        convolution_112 = torch.ops.aten.convolution.default(mul_293, arg63_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144);  mul_293 = arg63_1 = None
        add_149 = torch.ops.aten.add.Tensor(arg65_1, 1e-05);  arg65_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_294 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_545);  convolution_112 = unsqueeze_545 = None
        mul_295 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_549);  mul_295 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_150 = torch.ops.aten.add.Tensor(mul_296, unsqueeze_551);  mul_296 = unsqueeze_551 = None
        sigmoid_90 = torch.ops.aten.sigmoid.default(add_150)
        mul_297 = torch.ops.aten.mul.Tensor(add_150, sigmoid_90);  add_150 = sigmoid_90 = None
        mean_23 = torch.ops.aten.mean.dim(mul_297, [2, 3], True)
        convolution_113 = torch.ops.aten.convolution.default(mean_23, arg68_1, arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg68_1 = arg69_1 = None
        sigmoid_91 = torch.ops.aten.sigmoid.default(convolution_113)
        mul_298 = torch.ops.aten.mul.Tensor(convolution_113, sigmoid_91);  convolution_113 = sigmoid_91 = None
        convolution_114 = torch.ops.aten.convolution.default(mul_298, arg70_1, arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_298 = arg70_1 = arg71_1 = None
        sigmoid_92 = torch.ops.aten.sigmoid.default(convolution_114);  convolution_114 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_297, sigmoid_92);  mul_297 = sigmoid_92 = None
        convolution_115 = torch.ops.aten.convolution.default(mul_299, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg72_1 = None
        add_151 = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_300 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_553);  convolution_115 = unsqueeze_553 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_557);  mul_301 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_152 = torch.ops.aten.add.Tensor(mul_302, unsqueeze_559);  mul_302 = unsqueeze_559 = None
        convolution_116 = torch.ops.aten.convolution.default(add_152, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None
        add_153 = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_303 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70 = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_561);  convolution_116 = unsqueeze_561 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_565);  mul_304 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_154 = torch.ops.aten.add.Tensor(mul_305, unsqueeze_567);  mul_305 = unsqueeze_567 = None
        sigmoid_93 = torch.ops.aten.sigmoid.default(add_154)
        mul_306 = torch.ops.aten.mul.Tensor(add_154, sigmoid_93);  add_154 = sigmoid_93 = None
        convolution_117 = torch.ops.aten.convolution.default(mul_306, arg82_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240);  mul_306 = arg82_1 = None
        add_155 = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_155);  add_155 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_307 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_307, -1);  mul_307 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71 = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_569);  convolution_117 = unsqueeze_569 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_573);  mul_308 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_156 = torch.ops.aten.add.Tensor(mul_309, unsqueeze_575);  mul_309 = unsqueeze_575 = None
        sigmoid_94 = torch.ops.aten.sigmoid.default(add_156)
        mul_310 = torch.ops.aten.mul.Tensor(add_156, sigmoid_94);  add_156 = sigmoid_94 = None
        mean_24 = torch.ops.aten.mean.dim(mul_310, [2, 3], True)
        convolution_118 = torch.ops.aten.convolution.default(mean_24, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg87_1 = arg88_1 = None
        sigmoid_95 = torch.ops.aten.sigmoid.default(convolution_118)
        mul_311 = torch.ops.aten.mul.Tensor(convolution_118, sigmoid_95);  convolution_118 = sigmoid_95 = None
        convolution_119 = torch.ops.aten.convolution.default(mul_311, arg89_1, arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_311 = arg89_1 = arg90_1 = None
        sigmoid_96 = torch.ops.aten.sigmoid.default(convolution_119);  convolution_119 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_310, sigmoid_96);  mul_310 = sigmoid_96 = None
        convolution_120 = torch.ops.aten.convolution.default(mul_312, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg91_1 = None
        add_157 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_313 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72 = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_577);  convolution_120 = unsqueeze_577 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_581);  mul_314 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_158 = torch.ops.aten.add.Tensor(mul_315, unsqueeze_583);  mul_315 = unsqueeze_583 = None
        add_159 = torch.ops.aten.add.Tensor(add_158, add_152);  add_158 = add_152 = None
        convolution_121 = torch.ops.aten.convolution.default(add_159, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_159 = arg96_1 = None
        add_160 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_316 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73 = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_585);  convolution_121 = unsqueeze_585 = None
        mul_317 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_589);  mul_317 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_161 = torch.ops.aten.add.Tensor(mul_318, unsqueeze_591);  mul_318 = unsqueeze_591 = None
        sigmoid_97 = torch.ops.aten.sigmoid.default(add_161)
        mul_319 = torch.ops.aten.mul.Tensor(add_161, sigmoid_97);  add_161 = sigmoid_97 = None
        convolution_122 = torch.ops.aten.convolution.default(mul_319, arg101_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  mul_319 = arg101_1 = None
        add_162 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_320 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_593);  convolution_122 = unsqueeze_593 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_597);  mul_321 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_163 = torch.ops.aten.add.Tensor(mul_322, unsqueeze_599);  mul_322 = unsqueeze_599 = None
        sigmoid_98 = torch.ops.aten.sigmoid.default(add_163)
        mul_323 = torch.ops.aten.mul.Tensor(add_163, sigmoid_98);  add_163 = sigmoid_98 = None
        mean_25 = torch.ops.aten.mean.dim(mul_323, [2, 3], True)
        convolution_123 = torch.ops.aten.convolution.default(mean_25, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg106_1 = arg107_1 = None
        sigmoid_99 = torch.ops.aten.sigmoid.default(convolution_123)
        mul_324 = torch.ops.aten.mul.Tensor(convolution_123, sigmoid_99);  convolution_123 = sigmoid_99 = None
        convolution_124 = torch.ops.aten.convolution.default(mul_324, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_324 = arg108_1 = arg109_1 = None
        sigmoid_100 = torch.ops.aten.sigmoid.default(convolution_124);  convolution_124 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_323, sigmoid_100);  mul_323 = sigmoid_100 = None
        convolution_125 = torch.ops.aten.convolution.default(mul_325, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = arg110_1 = None
        add_164 = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_326 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_326, -1);  mul_326 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75 = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_601);  convolution_125 = unsqueeze_601 = None
        mul_327 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, unsqueeze_605);  mul_327 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_165 = torch.ops.aten.add.Tensor(mul_328, unsqueeze_607);  mul_328 = unsqueeze_607 = None
        convolution_126 = torch.ops.aten.convolution.default(add_165, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None
        add_166 = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_329 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(mul_329, -1);  mul_329 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_609);  convolution_126 = unsqueeze_609 = None
        mul_330 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_331 = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_613);  mul_330 = unsqueeze_613 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_167 = torch.ops.aten.add.Tensor(mul_331, unsqueeze_615);  mul_331 = unsqueeze_615 = None
        sigmoid_101 = torch.ops.aten.sigmoid.default(add_167)
        mul_332 = torch.ops.aten.mul.Tensor(add_167, sigmoid_101);  add_167 = sigmoid_101 = None
        convolution_127 = torch.ops.aten.convolution.default(mul_332, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_332 = arg120_1 = None
        add_168 = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_333 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_617);  convolution_127 = unsqueeze_617 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_621);  mul_334 = unsqueeze_621 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_169 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_623);  mul_335 = unsqueeze_623 = None
        sigmoid_102 = torch.ops.aten.sigmoid.default(add_169)
        mul_336 = torch.ops.aten.mul.Tensor(add_169, sigmoid_102);  add_169 = sigmoid_102 = None
        mean_26 = torch.ops.aten.mean.dim(mul_336, [2, 3], True)
        convolution_128 = torch.ops.aten.convolution.default(mean_26, arg125_1, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg125_1 = arg126_1 = None
        sigmoid_103 = torch.ops.aten.sigmoid.default(convolution_128)
        mul_337 = torch.ops.aten.mul.Tensor(convolution_128, sigmoid_103);  convolution_128 = sigmoid_103 = None
        convolution_129 = torch.ops.aten.convolution.default(mul_337, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_337 = arg127_1 = arg128_1 = None
        sigmoid_104 = torch.ops.aten.sigmoid.default(convolution_129);  convolution_129 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_336, sigmoid_104);  mul_336 = sigmoid_104 = None
        convolution_130 = torch.ops.aten.convolution.default(mul_338, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_338 = arg129_1 = None
        add_170 = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_339 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_625);  convolution_130 = unsqueeze_625 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_629);  mul_340 = unsqueeze_629 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_171 = torch.ops.aten.add.Tensor(mul_341, unsqueeze_631);  mul_341 = unsqueeze_631 = None
        add_172 = torch.ops.aten.add.Tensor(add_171, add_165);  add_171 = add_165 = None
        convolution_131 = torch.ops.aten.convolution.default(add_172, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg134_1 = None
        add_173 = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_342 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_633);  convolution_131 = unsqueeze_633 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_637);  mul_343 = unsqueeze_637 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_174 = torch.ops.aten.add.Tensor(mul_344, unsqueeze_639);  mul_344 = unsqueeze_639 = None
        sigmoid_105 = torch.ops.aten.sigmoid.default(add_174)
        mul_345 = torch.ops.aten.mul.Tensor(add_174, sigmoid_105);  add_174 = sigmoid_105 = None
        convolution_132 = torch.ops.aten.convolution.default(mul_345, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_345 = arg139_1 = None
        add_175 = torch.ops.aten.add.Tensor(arg141_1, 1e-05);  arg141_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_346 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_346, -1);  mul_346 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_641);  convolution_132 = unsqueeze_641 = None
        mul_347 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, unsqueeze_645);  mul_347 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_176 = torch.ops.aten.add.Tensor(mul_348, unsqueeze_647);  mul_348 = unsqueeze_647 = None
        sigmoid_106 = torch.ops.aten.sigmoid.default(add_176)
        mul_349 = torch.ops.aten.mul.Tensor(add_176, sigmoid_106);  add_176 = sigmoid_106 = None
        mean_27 = torch.ops.aten.mean.dim(mul_349, [2, 3], True)
        convolution_133 = torch.ops.aten.convolution.default(mean_27, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg144_1 = arg145_1 = None
        sigmoid_107 = torch.ops.aten.sigmoid.default(convolution_133)
        mul_350 = torch.ops.aten.mul.Tensor(convolution_133, sigmoid_107);  convolution_133 = sigmoid_107 = None
        convolution_134 = torch.ops.aten.convolution.default(mul_350, arg146_1, arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_350 = arg146_1 = arg147_1 = None
        sigmoid_108 = torch.ops.aten.sigmoid.default(convolution_134);  convolution_134 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_349, sigmoid_108);  mul_349 = sigmoid_108 = None
        convolution_135 = torch.ops.aten.convolution.default(mul_351, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_351 = arg148_1 = None
        add_177 = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_352 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_352, -1);  mul_352 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_649);  convolution_135 = unsqueeze_649 = None
        mul_353 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_354 = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_653);  mul_353 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_178 = torch.ops.aten.add.Tensor(mul_354, unsqueeze_655);  mul_354 = unsqueeze_655 = None
        add_179 = torch.ops.aten.add.Tensor(add_178, add_172);  add_178 = add_172 = None
        convolution_136 = torch.ops.aten.convolution.default(add_179, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg153_1 = None
        add_180 = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_355 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(mul_355, -1);  mul_355 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_657);  convolution_136 = unsqueeze_657 = None
        mul_356 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_661);  mul_356 = unsqueeze_661 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_181 = torch.ops.aten.add.Tensor(mul_357, unsqueeze_663);  mul_357 = unsqueeze_663 = None
        sigmoid_109 = torch.ops.aten.sigmoid.default(add_181)
        mul_358 = torch.ops.aten.mul.Tensor(add_181, sigmoid_109);  add_181 = sigmoid_109 = None
        convolution_137 = torch.ops.aten.convolution.default(mul_358, arg158_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_358 = arg158_1 = None
        add_182 = torch.ops.aten.add.Tensor(arg160_1, 1e-05);  arg160_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_182);  add_182 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_359 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_359, -1);  mul_359 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83 = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_665);  convolution_137 = unsqueeze_665 = None
        mul_360 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_669);  mul_360 = unsqueeze_669 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_183 = torch.ops.aten.add.Tensor(mul_361, unsqueeze_671);  mul_361 = unsqueeze_671 = None
        sigmoid_110 = torch.ops.aten.sigmoid.default(add_183)
        mul_362 = torch.ops.aten.mul.Tensor(add_183, sigmoid_110);  add_183 = sigmoid_110 = None
        mean_28 = torch.ops.aten.mean.dim(mul_362, [2, 3], True)
        convolution_138 = torch.ops.aten.convolution.default(mean_28, arg163_1, arg164_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg163_1 = arg164_1 = None
        sigmoid_111 = torch.ops.aten.sigmoid.default(convolution_138)
        mul_363 = torch.ops.aten.mul.Tensor(convolution_138, sigmoid_111);  convolution_138 = sigmoid_111 = None
        convolution_139 = torch.ops.aten.convolution.default(mul_363, arg165_1, arg166_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_363 = arg165_1 = arg166_1 = None
        sigmoid_112 = torch.ops.aten.sigmoid.default(convolution_139);  convolution_139 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_362, sigmoid_112);  mul_362 = sigmoid_112 = None
        convolution_140 = torch.ops.aten.convolution.default(mul_364, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_364 = arg167_1 = None
        add_184 = torch.ops.aten.add.Tensor(arg169_1, 1e-05);  arg169_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_184);  add_184 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_365 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(mul_365, -1);  mul_365 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84 = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_673);  convolution_140 = unsqueeze_673 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_677);  mul_366 = unsqueeze_677 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_185 = torch.ops.aten.add.Tensor(mul_367, unsqueeze_679);  mul_367 = unsqueeze_679 = None
        add_186 = torch.ops.aten.add.Tensor(add_185, add_179);  add_185 = add_179 = None
        convolution_141 = torch.ops.aten.convolution.default(add_186, arg172_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_186 = arg172_1 = None
        add_187 = torch.ops.aten.add.Tensor(arg174_1, 1e-05);  arg174_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_368 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_368, -1);  mul_368 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_681);  convolution_141 = unsqueeze_681 = None
        mul_369 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_369, unsqueeze_685);  mul_369 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_188 = torch.ops.aten.add.Tensor(mul_370, unsqueeze_687);  mul_370 = unsqueeze_687 = None
        sigmoid_113 = torch.ops.aten.sigmoid.default(add_188)
        mul_371 = torch.ops.aten.mul.Tensor(add_188, sigmoid_113);  add_188 = sigmoid_113 = None
        convolution_142 = torch.ops.aten.convolution.default(mul_371, arg177_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480);  mul_371 = arg177_1 = None
        add_189 = torch.ops.aten.add.Tensor(arg179_1, 1e-05);  arg179_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_189);  add_189 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_689);  convolution_142 = unsqueeze_689 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_693);  mul_373 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_190 = torch.ops.aten.add.Tensor(mul_374, unsqueeze_695);  mul_374 = unsqueeze_695 = None
        sigmoid_114 = torch.ops.aten.sigmoid.default(add_190)
        mul_375 = torch.ops.aten.mul.Tensor(add_190, sigmoid_114);  add_190 = sigmoid_114 = None
        mean_29 = torch.ops.aten.mean.dim(mul_375, [2, 3], True)
        convolution_143 = torch.ops.aten.convolution.default(mean_29, arg182_1, arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg182_1 = arg183_1 = None
        sigmoid_115 = torch.ops.aten.sigmoid.default(convolution_143)
        mul_376 = torch.ops.aten.mul.Tensor(convolution_143, sigmoid_115);  convolution_143 = sigmoid_115 = None
        convolution_144 = torch.ops.aten.convolution.default(mul_376, arg184_1, arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_376 = arg184_1 = arg185_1 = None
        sigmoid_116 = torch.ops.aten.sigmoid.default(convolution_144);  convolution_144 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_375, sigmoid_116);  mul_375 = sigmoid_116 = None
        convolution_145 = torch.ops.aten.convolution.default(mul_377, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_377 = arg186_1 = None
        add_191 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_191);  add_191 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_697);  convolution_145 = unsqueeze_697 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_701);  mul_379 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_192 = torch.ops.aten.add.Tensor(mul_380, unsqueeze_703);  mul_380 = unsqueeze_703 = None
        convolution_146 = torch.ops.aten.convolution.default(add_192, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None
        add_193 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_193);  add_193 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_381 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_705);  convolution_146 = unsqueeze_705 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_709);  mul_382 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_194 = torch.ops.aten.add.Tensor(mul_383, unsqueeze_711);  mul_383 = unsqueeze_711 = None
        sigmoid_117 = torch.ops.aten.sigmoid.default(add_194)
        mul_384 = torch.ops.aten.mul.Tensor(add_194, sigmoid_117);  add_194 = sigmoid_117 = None
        convolution_147 = torch.ops.aten.convolution.default(mul_384, arg196_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_384 = arg196_1 = None
        add_195 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_385 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_385, -1);  mul_385 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_713);  convolution_147 = unsqueeze_713 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_717);  mul_386 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_196 = torch.ops.aten.add.Tensor(mul_387, unsqueeze_719);  mul_387 = unsqueeze_719 = None
        sigmoid_118 = torch.ops.aten.sigmoid.default(add_196)
        mul_388 = torch.ops.aten.mul.Tensor(add_196, sigmoid_118);  add_196 = sigmoid_118 = None
        mean_30 = torch.ops.aten.mean.dim(mul_388, [2, 3], True)
        convolution_148 = torch.ops.aten.convolution.default(mean_30, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg201_1 = arg202_1 = None
        sigmoid_119 = torch.ops.aten.sigmoid.default(convolution_148)
        mul_389 = torch.ops.aten.mul.Tensor(convolution_148, sigmoid_119);  convolution_148 = sigmoid_119 = None
        convolution_149 = torch.ops.aten.convolution.default(mul_389, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_389 = arg203_1 = arg204_1 = None
        sigmoid_120 = torch.ops.aten.sigmoid.default(convolution_149);  convolution_149 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_388, sigmoid_120);  mul_388 = sigmoid_120 = None
        convolution_150 = torch.ops.aten.convolution.default(mul_390, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_390 = arg205_1 = None
        add_197 = torch.ops.aten.add.Tensor(arg207_1, 1e-05);  arg207_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_391 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_391, -1);  mul_391 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_721);  convolution_150 = unsqueeze_721 = None
        mul_392 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_725);  mul_392 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_198 = torch.ops.aten.add.Tensor(mul_393, unsqueeze_727);  mul_393 = unsqueeze_727 = None
        add_199 = torch.ops.aten.add.Tensor(add_198, add_192);  add_198 = add_192 = None
        convolution_151 = torch.ops.aten.convolution.default(add_199, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg210_1 = None
        add_200 = torch.ops.aten.add.Tensor(arg212_1, 1e-05);  arg212_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_394 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_394, -1);  mul_394 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_729);  convolution_151 = unsqueeze_729 = None
        mul_395 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_733);  mul_395 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_201 = torch.ops.aten.add.Tensor(mul_396, unsqueeze_735);  mul_396 = unsqueeze_735 = None
        sigmoid_121 = torch.ops.aten.sigmoid.default(add_201)
        mul_397 = torch.ops.aten.mul.Tensor(add_201, sigmoid_121);  add_201 = sigmoid_121 = None
        convolution_152 = torch.ops.aten.convolution.default(mul_397, arg215_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_397 = arg215_1 = None
        add_202 = torch.ops.aten.add.Tensor(arg217_1, 1e-05);  arg217_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_398 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_398, -1);  mul_398 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_737);  convolution_152 = unsqueeze_737 = None
        mul_399 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_741);  mul_399 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_203 = torch.ops.aten.add.Tensor(mul_400, unsqueeze_743);  mul_400 = unsqueeze_743 = None
        sigmoid_122 = torch.ops.aten.sigmoid.default(add_203)
        mul_401 = torch.ops.aten.mul.Tensor(add_203, sigmoid_122);  add_203 = sigmoid_122 = None
        mean_31 = torch.ops.aten.mean.dim(mul_401, [2, 3], True)
        convolution_153 = torch.ops.aten.convolution.default(mean_31, arg220_1, arg221_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg220_1 = arg221_1 = None
        sigmoid_123 = torch.ops.aten.sigmoid.default(convolution_153)
        mul_402 = torch.ops.aten.mul.Tensor(convolution_153, sigmoid_123);  convolution_153 = sigmoid_123 = None
        convolution_154 = torch.ops.aten.convolution.default(mul_402, arg222_1, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_402 = arg222_1 = arg223_1 = None
        sigmoid_124 = torch.ops.aten.sigmoid.default(convolution_154);  convolution_154 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_401, sigmoid_124);  mul_401 = sigmoid_124 = None
        convolution_155 = torch.ops.aten.convolution.default(mul_403, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_403 = arg224_1 = None
        add_204 = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_404 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_745);  convolution_155 = unsqueeze_745 = None
        mul_405 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_406 = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_749);  mul_405 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_205 = torch.ops.aten.add.Tensor(mul_406, unsqueeze_751);  mul_406 = unsqueeze_751 = None
        add_206 = torch.ops.aten.add.Tensor(add_205, add_199);  add_205 = add_199 = None
        convolution_156 = torch.ops.aten.convolution.default(add_206, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg229_1 = None
        add_207 = torch.ops.aten.add.Tensor(arg231_1, 1e-05);  arg231_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_207);  add_207 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_407 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_753);  convolution_156 = unsqueeze_753 = None
        mul_408 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_409 = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_757);  mul_408 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_208 = torch.ops.aten.add.Tensor(mul_409, unsqueeze_759);  mul_409 = unsqueeze_759 = None
        sigmoid_125 = torch.ops.aten.sigmoid.default(add_208)
        mul_410 = torch.ops.aten.mul.Tensor(add_208, sigmoid_125);  add_208 = sigmoid_125 = None
        convolution_157 = torch.ops.aten.convolution.default(mul_410, arg234_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_410 = arg234_1 = None
        add_209 = torch.ops.aten.add.Tensor(arg236_1, 1e-05);  arg236_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_411 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_761);  convolution_157 = unsqueeze_761 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_765);  mul_412 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_210 = torch.ops.aten.add.Tensor(mul_413, unsqueeze_767);  mul_413 = unsqueeze_767 = None
        sigmoid_126 = torch.ops.aten.sigmoid.default(add_210)
        mul_414 = torch.ops.aten.mul.Tensor(add_210, sigmoid_126);  add_210 = sigmoid_126 = None
        mean_32 = torch.ops.aten.mean.dim(mul_414, [2, 3], True)
        convolution_158 = torch.ops.aten.convolution.default(mean_32, arg239_1, arg240_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg239_1 = arg240_1 = None
        sigmoid_127 = torch.ops.aten.sigmoid.default(convolution_158)
        mul_415 = torch.ops.aten.mul.Tensor(convolution_158, sigmoid_127);  convolution_158 = sigmoid_127 = None
        convolution_159 = torch.ops.aten.convolution.default(mul_415, arg241_1, arg242_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_415 = arg241_1 = arg242_1 = None
        sigmoid_128 = torch.ops.aten.sigmoid.default(convolution_159);  convolution_159 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_414, sigmoid_128);  mul_414 = sigmoid_128 = None
        convolution_160 = torch.ops.aten.convolution.default(mul_416, arg243_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_416 = arg243_1 = None
        add_211 = torch.ops.aten.add.Tensor(arg245_1, 1e-05);  arg245_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_417 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_769);  convolution_160 = unsqueeze_769 = None
        mul_418 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_773);  mul_418 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_212 = torch.ops.aten.add.Tensor(mul_419, unsqueeze_775);  mul_419 = unsqueeze_775 = None
        add_213 = torch.ops.aten.add.Tensor(add_212, add_206);  add_212 = add_206 = None
        convolution_161 = torch.ops.aten.convolution.default(add_213, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_213 = arg248_1 = None
        add_214 = torch.ops.aten.add.Tensor(arg250_1, 1e-05);  arg250_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_420 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_777);  convolution_161 = unsqueeze_777 = None
        mul_421 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_422 = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_781);  mul_421 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_215 = torch.ops.aten.add.Tensor(mul_422, unsqueeze_783);  mul_422 = unsqueeze_783 = None
        sigmoid_129 = torch.ops.aten.sigmoid.default(add_215)
        mul_423 = torch.ops.aten.mul.Tensor(add_215, sigmoid_129);  add_215 = sigmoid_129 = None
        convolution_162 = torch.ops.aten.convolution.default(mul_423, arg253_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  mul_423 = arg253_1 = None
        add_216 = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_216);  add_216 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_424 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_424, -1);  mul_424 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_785);  convolution_162 = unsqueeze_785 = None
        mul_425 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_426 = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_789);  mul_425 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_217 = torch.ops.aten.add.Tensor(mul_426, unsqueeze_791);  mul_426 = unsqueeze_791 = None
        sigmoid_130 = torch.ops.aten.sigmoid.default(add_217)
        mul_427 = torch.ops.aten.mul.Tensor(add_217, sigmoid_130);  add_217 = sigmoid_130 = None
        mean_33 = torch.ops.aten.mean.dim(mul_427, [2, 3], True)
        convolution_163 = torch.ops.aten.convolution.default(mean_33, arg258_1, arg259_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_33 = arg258_1 = arg259_1 = None
        sigmoid_131 = torch.ops.aten.sigmoid.default(convolution_163)
        mul_428 = torch.ops.aten.mul.Tensor(convolution_163, sigmoid_131);  convolution_163 = sigmoid_131 = None
        convolution_164 = torch.ops.aten.convolution.default(mul_428, arg260_1, arg261_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_428 = arg260_1 = arg261_1 = None
        sigmoid_132 = torch.ops.aten.sigmoid.default(convolution_164);  convolution_164 = None
        mul_429 = torch.ops.aten.mul.Tensor(mul_427, sigmoid_132);  mul_427 = sigmoid_132 = None
        convolution_165 = torch.ops.aten.convolution.default(mul_429, arg262_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_429 = arg262_1 = None
        add_218 = torch.ops.aten.add.Tensor(arg264_1, 1e-05);  arg264_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_430 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_430, -1);  mul_430 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_793);  convolution_165 = unsqueeze_793 = None
        mul_431 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_431, unsqueeze_797);  mul_431 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_219 = torch.ops.aten.add.Tensor(mul_432, unsqueeze_799);  mul_432 = unsqueeze_799 = None
        convolution_166 = torch.ops.aten.convolution.default(add_219, arg267_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg267_1 = None
        add_220 = torch.ops.aten.add.Tensor(arg269_1, 1e-05);  arg269_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_433 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_433, -1);  mul_433 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_801);  convolution_166 = unsqueeze_801 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_435 = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_805);  mul_434 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_221 = torch.ops.aten.add.Tensor(mul_435, unsqueeze_807);  mul_435 = unsqueeze_807 = None
        sigmoid_133 = torch.ops.aten.sigmoid.default(add_221)
        mul_436 = torch.ops.aten.mul.Tensor(add_221, sigmoid_133);  add_221 = sigmoid_133 = None
        convolution_167 = torch.ops.aten.convolution.default(mul_436, arg272_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_436 = arg272_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg274_1, 1e-05);  arg274_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_437 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_437, -1);  mul_437 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_809);  convolution_167 = unsqueeze_809 = None
        mul_438 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_439 = torch.ops.aten.mul.Tensor(mul_438, unsqueeze_813);  mul_438 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_223 = torch.ops.aten.add.Tensor(mul_439, unsqueeze_815);  mul_439 = unsqueeze_815 = None
        sigmoid_134 = torch.ops.aten.sigmoid.default(add_223)
        mul_440 = torch.ops.aten.mul.Tensor(add_223, sigmoid_134);  add_223 = sigmoid_134 = None
        mean_34 = torch.ops.aten.mean.dim(mul_440, [2, 3], True)
        convolution_168 = torch.ops.aten.convolution.default(mean_34, arg277_1, arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_34 = arg277_1 = arg278_1 = None
        sigmoid_135 = torch.ops.aten.sigmoid.default(convolution_168)
        mul_441 = torch.ops.aten.mul.Tensor(convolution_168, sigmoid_135);  convolution_168 = sigmoid_135 = None
        convolution_169 = torch.ops.aten.convolution.default(mul_441, arg279_1, arg280_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_441 = arg279_1 = arg280_1 = None
        sigmoid_136 = torch.ops.aten.sigmoid.default(convolution_169);  convolution_169 = None
        mul_442 = torch.ops.aten.mul.Tensor(mul_440, sigmoid_136);  mul_440 = sigmoid_136 = None
        convolution_170 = torch.ops.aten.convolution.default(mul_442, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_442 = arg281_1 = None
        add_224 = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_443 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_443, -1);  mul_443 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_817);  convolution_170 = unsqueeze_817 = None
        mul_444 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, unsqueeze_821);  mul_444 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_225 = torch.ops.aten.add.Tensor(mul_445, unsqueeze_823);  mul_445 = unsqueeze_823 = None
        add_226 = torch.ops.aten.add.Tensor(add_225, add_219);  add_225 = add_219 = None
        convolution_171 = torch.ops.aten.convolution.default(add_226, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg286_1 = None
        add_227 = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_446 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_446, -1);  mul_446 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_825);  convolution_171 = unsqueeze_825 = None
        mul_447 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_447, unsqueeze_829);  mul_447 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_228 = torch.ops.aten.add.Tensor(mul_448, unsqueeze_831);  mul_448 = unsqueeze_831 = None
        sigmoid_137 = torch.ops.aten.sigmoid.default(add_228)
        mul_449 = torch.ops.aten.mul.Tensor(add_228, sigmoid_137);  add_228 = sigmoid_137 = None
        convolution_172 = torch.ops.aten.convolution.default(mul_449, arg291_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_449 = arg291_1 = None
        add_229 = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_450 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_833);  convolution_172 = unsqueeze_833 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_837);  mul_451 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_230 = torch.ops.aten.add.Tensor(mul_452, unsqueeze_839);  mul_452 = unsqueeze_839 = None
        sigmoid_138 = torch.ops.aten.sigmoid.default(add_230)
        mul_453 = torch.ops.aten.mul.Tensor(add_230, sigmoid_138);  add_230 = sigmoid_138 = None
        mean_35 = torch.ops.aten.mean.dim(mul_453, [2, 3], True)
        convolution_173 = torch.ops.aten.convolution.default(mean_35, arg296_1, arg297_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_35 = arg296_1 = arg297_1 = None
        sigmoid_139 = torch.ops.aten.sigmoid.default(convolution_173)
        mul_454 = torch.ops.aten.mul.Tensor(convolution_173, sigmoid_139);  convolution_173 = sigmoid_139 = None
        convolution_174 = torch.ops.aten.convolution.default(mul_454, arg298_1, arg299_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_454 = arg298_1 = arg299_1 = None
        sigmoid_140 = torch.ops.aten.sigmoid.default(convolution_174);  convolution_174 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_453, sigmoid_140);  mul_453 = sigmoid_140 = None
        convolution_175 = torch.ops.aten.convolution.default(mul_455, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_455 = arg300_1 = None
        add_231 = torch.ops.aten.add.Tensor(arg302_1, 1e-05);  arg302_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_456 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_841);  convolution_175 = unsqueeze_841 = None
        mul_457 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_845);  mul_457 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_232 = torch.ops.aten.add.Tensor(mul_458, unsqueeze_847);  mul_458 = unsqueeze_847 = None
        add_233 = torch.ops.aten.add.Tensor(add_232, add_226);  add_232 = add_226 = None
        convolution_176 = torch.ops.aten.convolution.default(add_233, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg305_1 = None
        add_234 = torch.ops.aten.add.Tensor(arg307_1, 1e-05);  arg307_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_459 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_849);  convolution_176 = unsqueeze_849 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_853);  mul_460 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_235 = torch.ops.aten.add.Tensor(mul_461, unsqueeze_855);  mul_461 = unsqueeze_855 = None
        sigmoid_141 = torch.ops.aten.sigmoid.default(add_235)
        mul_462 = torch.ops.aten.mul.Tensor(add_235, sigmoid_141);  add_235 = sigmoid_141 = None
        convolution_177 = torch.ops.aten.convolution.default(mul_462, arg310_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_462 = arg310_1 = None
        add_236 = torch.ops.aten.add.Tensor(arg312_1, 1e-05);  arg312_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_463 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_463, -1);  mul_463 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_857);  convolution_177 = unsqueeze_857 = None
        mul_464 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_464, unsqueeze_861);  mul_464 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_237 = torch.ops.aten.add.Tensor(mul_465, unsqueeze_863);  mul_465 = unsqueeze_863 = None
        sigmoid_142 = torch.ops.aten.sigmoid.default(add_237)
        mul_466 = torch.ops.aten.mul.Tensor(add_237, sigmoid_142);  add_237 = sigmoid_142 = None
        mean_36 = torch.ops.aten.mean.dim(mul_466, [2, 3], True)
        convolution_178 = torch.ops.aten.convolution.default(mean_36, arg315_1, arg316_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_36 = arg315_1 = arg316_1 = None
        sigmoid_143 = torch.ops.aten.sigmoid.default(convolution_178)
        mul_467 = torch.ops.aten.mul.Tensor(convolution_178, sigmoid_143);  convolution_178 = sigmoid_143 = None
        convolution_179 = torch.ops.aten.convolution.default(mul_467, arg317_1, arg318_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_467 = arg317_1 = arg318_1 = None
        sigmoid_144 = torch.ops.aten.sigmoid.default(convolution_179);  convolution_179 = None
        mul_468 = torch.ops.aten.mul.Tensor(mul_466, sigmoid_144);  mul_466 = sigmoid_144 = None
        convolution_180 = torch.ops.aten.convolution.default(mul_468, arg319_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_468 = arg319_1 = None
        add_238 = torch.ops.aten.add.Tensor(arg321_1, 1e-05);  arg321_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_469 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_469, -1);  mul_469 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_865);  convolution_180 = unsqueeze_865 = None
        mul_470 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_471 = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_869);  mul_470 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_239 = torch.ops.aten.add.Tensor(mul_471, unsqueeze_871);  mul_471 = unsqueeze_871 = None
        add_240 = torch.ops.aten.add.Tensor(add_239, add_233);  add_239 = add_233 = None
        convolution_181 = torch.ops.aten.convolution.default(add_240, arg324_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg324_1 = None
        add_241 = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_472 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_873);  convolution_181 = unsqueeze_873 = None
        mul_473 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_474 = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_877);  mul_473 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_242 = torch.ops.aten.add.Tensor(mul_474, unsqueeze_879);  mul_474 = unsqueeze_879 = None
        sigmoid_145 = torch.ops.aten.sigmoid.default(add_242)
        mul_475 = torch.ops.aten.mul.Tensor(add_242, sigmoid_145);  add_242 = sigmoid_145 = None
        convolution_182 = torch.ops.aten.convolution.default(mul_475, arg329_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_475 = arg329_1 = None
        add_243 = torch.ops.aten.add.Tensor(arg331_1, 1e-05);  arg331_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_243);  add_243 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_476 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_476, -1);  mul_476 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_881);  convolution_182 = unsqueeze_881 = None
        mul_477 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_478 = torch.ops.aten.mul.Tensor(mul_477, unsqueeze_885);  mul_477 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_244 = torch.ops.aten.add.Tensor(mul_478, unsqueeze_887);  mul_478 = unsqueeze_887 = None
        sigmoid_146 = torch.ops.aten.sigmoid.default(add_244)
        mul_479 = torch.ops.aten.mul.Tensor(add_244, sigmoid_146);  add_244 = sigmoid_146 = None
        mean_37 = torch.ops.aten.mean.dim(mul_479, [2, 3], True)
        convolution_183 = torch.ops.aten.convolution.default(mean_37, arg334_1, arg335_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_37 = arg334_1 = arg335_1 = None
        sigmoid_147 = torch.ops.aten.sigmoid.default(convolution_183)
        mul_480 = torch.ops.aten.mul.Tensor(convolution_183, sigmoid_147);  convolution_183 = sigmoid_147 = None
        convolution_184 = torch.ops.aten.convolution.default(mul_480, arg336_1, arg337_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_480 = arg336_1 = arg337_1 = None
        sigmoid_148 = torch.ops.aten.sigmoid.default(convolution_184);  convolution_184 = None
        mul_481 = torch.ops.aten.mul.Tensor(mul_479, sigmoid_148);  mul_479 = sigmoid_148 = None
        convolution_185 = torch.ops.aten.convolution.default(mul_481, arg338_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_481 = arg338_1 = None
        add_245 = torch.ops.aten.add.Tensor(arg340_1, 1e-05);  arg340_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_245);  add_245 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_482 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_482, -1);  mul_482 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_889);  convolution_185 = unsqueeze_889 = None
        mul_483 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_484 = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_893);  mul_483 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_246 = torch.ops.aten.add.Tensor(mul_484, unsqueeze_895);  mul_484 = unsqueeze_895 = None
        add_247 = torch.ops.aten.add.Tensor(add_246, add_240);  add_246 = add_240 = None
        convolution_186 = torch.ops.aten.convolution.default(add_247, arg343_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_247 = arg343_1 = None
        add_248 = torch.ops.aten.add.Tensor(arg345_1, 1e-05);  arg345_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_485 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_897);  convolution_186 = unsqueeze_897 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_487 = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_901);  mul_486 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_249 = torch.ops.aten.add.Tensor(mul_487, unsqueeze_903);  mul_487 = unsqueeze_903 = None
        sigmoid_149 = torch.ops.aten.sigmoid.default(add_249)
        mul_488 = torch.ops.aten.mul.Tensor(add_249, sigmoid_149);  add_249 = sigmoid_149 = None
        convolution_187 = torch.ops.aten.convolution.default(mul_488, arg348_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152);  mul_488 = arg348_1 = None
        add_250 = torch.ops.aten.add.Tensor(arg350_1, 1e-05);  arg350_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_489 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_905);  convolution_187 = unsqueeze_905 = None
        mul_490 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg351_1, -1);  arg351_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_909);  mul_490 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_251 = torch.ops.aten.add.Tensor(mul_491, unsqueeze_911);  mul_491 = unsqueeze_911 = None
        sigmoid_150 = torch.ops.aten.sigmoid.default(add_251)
        mul_492 = torch.ops.aten.mul.Tensor(add_251, sigmoid_150);  add_251 = sigmoid_150 = None
        mean_38 = torch.ops.aten.mean.dim(mul_492, [2, 3], True)
        convolution_188 = torch.ops.aten.convolution.default(mean_38, arg353_1, arg354_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_38 = arg353_1 = arg354_1 = None
        sigmoid_151 = torch.ops.aten.sigmoid.default(convolution_188)
        mul_493 = torch.ops.aten.mul.Tensor(convolution_188, sigmoid_151);  convolution_188 = sigmoid_151 = None
        convolution_189 = torch.ops.aten.convolution.default(mul_493, arg355_1, arg356_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_493 = arg355_1 = arg356_1 = None
        sigmoid_152 = torch.ops.aten.sigmoid.default(convolution_189);  convolution_189 = None
        mul_494 = torch.ops.aten.mul.Tensor(mul_492, sigmoid_152);  mul_492 = sigmoid_152 = None
        convolution_190 = torch.ops.aten.convolution.default(mul_494, arg357_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_494 = arg357_1 = None
        add_252 = torch.ops.aten.add.Tensor(arg359_1, 1e-05);  arg359_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_495 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_913);  convolution_190 = unsqueeze_913 = None
        mul_496 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_497 = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_917);  mul_496 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_253 = torch.ops.aten.add.Tensor(mul_497, unsqueeze_919);  mul_497 = unsqueeze_919 = None
        convolution_191 = torch.ops.aten.convolution.default(add_253, arg362_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_253 = arg362_1 = None
        add_254 = torch.ops.aten.add.Tensor(arg364_1, 1e-05);  arg364_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_498 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg363_1, -1);  arg363_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_921);  convolution_191 = unsqueeze_921 = None
        mul_499 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_925);  mul_499 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_255 = torch.ops.aten.add.Tensor(mul_500, unsqueeze_927);  mul_500 = unsqueeze_927 = None
        sigmoid_153 = torch.ops.aten.sigmoid.default(add_255)
        mul_501 = torch.ops.aten.mul.Tensor(add_255, sigmoid_153);  add_255 = sigmoid_153 = None
        mean_39 = torch.ops.aten.mean.dim(mul_501, [-1, -2], True);  mul_501 = None
        view_1 = torch.ops.aten.view.default(mean_39, [8, 1280]);  mean_39 = None
        permute_1 = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg368_1, view_1, permute_1);  arg368_1 = view_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 192, 192), is_leaf=True)  # arg1_1
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
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf11, (8, 32, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf12, (8,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32, 8, 1, 1), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16, 32, 1, 1), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf16, (16,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf17, (16,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf18, (16,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf19, (16,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf20, (96, 16, 1, 1), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf21, (96,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf22, (96,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf23, (96,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf24, (96,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf25, (96, 1, 3, 3), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf26, (96,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf27, (96,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf28, (96,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf29, (96,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf30, (4, 96, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 16, device=device(type='cuda', index=0))
    reader.tensor(buf31, (4,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf32, (96, 4, 1, 1), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf33, (96,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf34, (24, 96, 1, 1), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf35, (24,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf36, (24,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf37, (24,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf38, (24,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf39, (144, 24, 1, 1), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (144,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf41, (144,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf42, (144,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf43, (144,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf44, (144, 1, 3, 3), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf45, (144,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf46, (144,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf47, (144,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf48, (144,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf49, (6, 144, 1, 1), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 24, device=device(type='cuda', index=0))
    reader.tensor(buf50, (6,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf51, (144, 6, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf52, (144,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf53, (24, 144, 1, 1), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf54, (24,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf55, (24,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf56, (24,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf57, (24,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf58, (144, 24, 1, 1), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf59, (144,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf60, (144,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf61, (144,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf62, (144,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 14400, device=device(type='cuda', index=0))
    reader.tensor(buf63, (144, 1, 5, 5), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf64, (144,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf65, (144,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf66, (144,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (144,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf68, (6, 144, 1, 1), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 24, device=device(type='cuda', index=0))
    reader.tensor(buf69, (6,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf70, (144, 6, 1, 1), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf71, (144,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 23040, device=device(type='cuda', index=0))
    reader.tensor(buf72, (40, 144, 1, 1), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf73, (40,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf74, (40,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf75, (40,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf76, (40,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf77, (240, 40, 1, 1), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf78, (240,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf79, (240,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf80, (240,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf81, (240,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 24000, device=device(type='cuda', index=0))
    reader.tensor(buf82, (240, 1, 5, 5), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf83, (240,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf84, (240,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf85, (240,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf86, (240,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf87, (10, 240, 1, 1), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf88, (10,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf89, (240, 10, 1, 1), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf90, (240,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf91, (40, 240, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf92, (40,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf93, (40,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf94, (40,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf95, (40,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf96, (240, 40, 1, 1), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf97, (240,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf98, (240,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf99, (240,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf100, (240,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf101, (240, 1, 3, 3), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf102, (240,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf103, (240,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf104, (240,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf105, (240,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf106, (10, 240, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf107, (10,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 9600, device=device(type='cuda', index=0))
    reader.tensor(buf108, (240, 10, 1, 1), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf109, (240,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf110, (80, 240, 1, 1), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf111, (80,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf112, (80,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf113, (80,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf114, (80,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf115, (480, 80, 1, 1), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf116, (480,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf117, (480,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf118, (480,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf119, (480,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf120, (480, 1, 3, 3), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf121, (480,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf122, (480,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf123, (480,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf124, (480,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf125, (20, 480, 1, 1), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf126, (20,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf127, (480, 20, 1, 1), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf128, (480,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf129, (80, 480, 1, 1), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf130, (80,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf131, (80,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf132, (80,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf133, (80,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf134, (480, 80, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf135, (480,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf136, (480,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf137, (480,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf138, (480,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf139, (480, 1, 3, 3), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf140, (480,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf141, (480,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf142, (480,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf143, (480,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf144, (20, 480, 1, 1), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf145, (20,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf146, (480, 20, 1, 1), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf147, (480,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf148, (80, 480, 1, 1), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf149, (80,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf150, (80,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf151, (80,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf152, (80,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf153, (480, 80, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf154, (480,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf155, (480,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf156, (480,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf157, (480,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 17280, device=device(type='cuda', index=0))
    reader.tensor(buf158, (480, 1, 3, 3), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf159, (480,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf160, (480,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf161, (480,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf162, (480,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf163, (20, 480, 1, 1), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf164, (20,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf165, (480, 20, 1, 1), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf166, (480,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf167, (80, 480, 1, 1), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf168, (80,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf169, (80,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf170, (80,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf171, (80,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf172, (480, 80, 1, 1), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf173, (480,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf174, (480,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf175, (480,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf176, (480,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 48000, device=device(type='cuda', index=0))
    reader.tensor(buf177, (480, 1, 5, 5), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf178, (480,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf179, (480,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf180, (480,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf181, (480,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf182, (20, 480, 1, 1), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 80, device=device(type='cuda', index=0))
    reader.tensor(buf183, (20,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf184, (480, 20, 1, 1), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf185, (480,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 215040, device=device(type='cuda', index=0))
    reader.tensor(buf186, (112, 480, 1, 1), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf187, (112,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf188, (112,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf189, (112,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf190, (112,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf191, (672, 112, 1, 1), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf192, (672,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf193, (672,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf194, (672,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf195, (672,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 67200, device=device(type='cuda', index=0))
    reader.tensor(buf196, (672, 1, 5, 5), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf197, (672,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf198, (672,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf199, (672,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf200, (672,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf201, (28, 672, 1, 1), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf202, (28,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf203, (672, 28, 1, 1), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf204, (672,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf205, (112, 672, 1, 1), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf206, (112,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf207, (112,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf208, (112,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf209, (112,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf210, (672, 112, 1, 1), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf211, (672,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf212, (672,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf213, (672,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf214, (672,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 67200, device=device(type='cuda', index=0))
    reader.tensor(buf215, (672, 1, 5, 5), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf216, (672,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf217, (672,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf218, (672,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf219, (672,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf220, (28, 672, 1, 1), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf221, (28,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf222, (672, 28, 1, 1), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf223, (672,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf224, (112, 672, 1, 1), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf225, (112,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf226, (112,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf227, (112,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf228, (112,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf229, (672, 112, 1, 1), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf230, (672,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf231, (672,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf232, (672,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf233, (672,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 67200, device=device(type='cuda', index=0))
    reader.tensor(buf234, (672, 1, 5, 5), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf235, (672,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf236, (672,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf237, (672,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf238, (672,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf239, (28, 672, 1, 1), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf240, (28,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf241, (672, 28, 1, 1), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf242, (672,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf243, (112, 672, 1, 1), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf244, (112,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf245, (112,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf246, (112,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 448, device=device(type='cuda', index=0))
    reader.tensor(buf247, (112,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf248, (672, 112, 1, 1), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf249, (672,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf250, (672,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf251, (672,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf252, (672,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 67200, device=device(type='cuda', index=0))
    reader.tensor(buf253, (672, 1, 5, 5), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf254, (672,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf255, (672,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf256, (672,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf257, (672,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf258, (28, 672, 1, 1), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 112, device=device(type='cuda', index=0))
    reader.tensor(buf259, (28,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 75264, device=device(type='cuda', index=0))
    reader.tensor(buf260, (672, 28, 1, 1), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 2688, device=device(type='cuda', index=0))
    reader.tensor(buf261, (672,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 516096, device=device(type='cuda', index=0))
    reader.tensor(buf262, (192, 672, 1, 1), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf263, (192,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf264, (192,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf265, (192,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf266, (192,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf267, (1152, 192, 1, 1), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf268, (1152,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf269, (1152,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf270, (1152,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1152,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf272, (1152, 1, 5, 5), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf273, (1152,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf274, (1152,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1152,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1152,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf277, (48, 1152, 1, 1), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf278, (48,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1152, 48, 1, 1), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf280, (1152,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf281, (192, 1152, 1, 1), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf282, (192,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf283, (192,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf284, (192,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf285, (192,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf286, (1152, 192, 1, 1), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf287, (1152,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf288, (1152,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf289, (1152,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1152,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf291, (1152, 1, 5, 5), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf292, (1152,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf293, (1152,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf294, (1152,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf295, (1152,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf296, (48, 1152, 1, 1), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf297, (48,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf298, (1152, 48, 1, 1), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf299, (1152,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf300, (192, 1152, 1, 1), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf301, (192,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf302, (192,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf303, (192,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf304, (192,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1152, 192, 1, 1), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1152,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf307, (1152,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf308, (1152,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1152,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf310, (1152, 1, 5, 5), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf311, (1152,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf312, (1152,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf313, (1152,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf314, (1152,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf315, (48, 1152, 1, 1), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf316, (48,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1152, 48, 1, 1), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1152,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf319, (192, 1152, 1, 1), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf320, (192,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf321, (192,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf322, (192,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf323, (192,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1152, 192, 1, 1), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1152,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1152,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1152,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf328, (1152,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1152, 1, 5, 5), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1152,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1152,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf332, (1152,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf333, (1152,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf334, (48, 1152, 1, 1), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf335, (48,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf336, (1152, 48, 1, 1), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf337, (1152,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf338, (192, 1152, 1, 1), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf339, (192,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf340, (192,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf341, (192,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf342, (192,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf343, (1152, 192, 1, 1), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf344, (1152,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1152,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1152,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1152,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1152, 1, 3, 3), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1152,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1152,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1152,), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf352, (1152,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf353, (48, 1152, 1, 1), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf354, (48,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf355, (1152, 48, 1, 1), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf356, (1152,), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 1474560, device=device(type='cuda', index=0))
    reader.tensor(buf357, (320, 1152, 1, 1), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf358, (320,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf359, (320,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf360, (320,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf361, (320,), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf362, (1280, 320, 1, 1), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf363, (1280,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf364, (1280,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf365, (1280,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf366, (1280,), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf367, (1000, 1280), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf368, (1000,), is_leaf=True)  # arg368_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)